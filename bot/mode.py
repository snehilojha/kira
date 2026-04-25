"""Kira mode state machine for V1.5.

Tracks Kira's operating mode based on user presence (keyboard/mouse activity).
Transitions are logged to DB so they survive restarts and feed the away summary.

States
------
idle               — initial state before any presence signal arrives
active_session     — user is present (recent keyboard/mouse activity)
autonomous         — user absent for longer than KIRA_IDLE_THRESHOLD_SECONDS
awaiting_confirmation — Kira is blocked waiting for a Telegram approve/deny
recovering         — bot just restarted while last known mode was autonomous

Presence detection
------------------
Uses Windows ``GetLastInputInfo`` via ctypes — zero dependencies beyond stdlib,
zero API cost, accurate for both keyboard and mouse activity.
Fallback on non-Windows platforms: always reports present (returns 0.0 seconds).

Public API
----------
get_mode()          — return current mode string
is_autonomous()     — True when mode is 'autonomous'
get_last_input_seconds() — seconds since last keyboard/mouse input
start()             — background asyncio loop (called from main.py)
set_mode(new, reason) — async; changes state and logs to DB
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import os
import platform
from datetime import datetime, timezone
from typing import Literal

logger = logging.getLogger(__name__)

# ── Types ─────────────────────────────────────────────────────────

ModeState = Literal[
    "idle",
    "active_session",
    "autonomous",
    "awaiting_confirmation",
    "recovering",
]

# ── Configuration (read at runtime so .env is loaded first) ───────

def _idle_threshold() -> float:
    return float(os.environ.get("KIRA_IDLE_THRESHOLD_SECONDS", "180"))

def _poll_interval() -> float:
    return float(os.environ.get("KIRA_MODE_POLL_SECONDS", "30"))

# ── Module state ──────────────────────────────────────────────────

_current_mode: ModeState = "idle"
_mode_entered_at: datetime = datetime.now(timezone.utc)

# Timestamp when autonomous mode started — used for return summary scoping.
_autonomous_since: datetime | None = None

# Monotonic timestamp of last voice activity — prevents autonomous flip while talking.
_last_voice_activity: float = 0.0


# ── Public API ────────────────────────────────────────────────────

def get_mode() -> ModeState:
    """Return the current Kira operating mode."""
    return _current_mode


def is_autonomous() -> bool:
    """Return True when Kira is in autonomous mode."""
    return _current_mode == "autonomous"


def mark_user_active() -> None:
    """Signal that the user is active via voice. Resets the voice idle clock."""
    global _last_voice_activity
    import time
    _last_voice_activity = time.monotonic()


def get_last_input_seconds() -> float:
    """Return seconds elapsed since the last keyboard or mouse event.

    Uses Windows ``GetLastInputInfo``. Returns 0.0 on non-Windows platforms
    (treated as 'present').
    """
    return _get_last_input_seconds()


async def set_mode(new_mode: ModeState, reason: str = "") -> None:
    """Transition to a new mode, logging the change to DB.

    No-op if the mode is already *new_mode*.
    """
    global _current_mode, _mode_entered_at, _autonomous_since

    if new_mode == _current_mode:
        return

    from_mode = _current_mode
    _current_mode = new_mode
    _mode_entered_at = datetime.now(timezone.utc)

    if new_mode == "autonomous":
        _autonomous_since = _mode_entered_at
    elif from_mode == "autonomous":
        # Returning — capture the window before clearing it.
        away_since = _autonomous_since
        _autonomous_since = None
        asyncio.create_task(
            _send_return_summary(away_since),
            name="kira-return-summary",
        )

    logger.info(
        "Mode transition: %s → %s (%s)",
        from_mode, new_mode, reason or "no reason given",
    )

    try:
        from bot import db
        await db.log_mode_transition(from_mode, new_mode, reason)
    except Exception as exc:
        logger.warning("Failed to log mode transition to DB: %s", exc)


async def start() -> None:
    """Background presence-detection loop.

    Polls every KIRA_MODE_POLL_SECONDS. Transitions between active_session
    and autonomous based on last keyboard/mouse input time.
    """
    import time
    global _last_voice_activity

    idle_threshold = _idle_threshold()
    poll_interval = _poll_interval()
    logger.info(
        "Mode monitor started (threshold=%.0fs, poll=%.0fs)",
        idle_threshold, poll_interval,
    )

    # Treat startup as user-present — GetLastInputInfo is unreliable when
    # launched via Task Scheduler (non-interactive session). Give a full
    # idle_threshold grace period before the first autonomous check.
    _last_voice_activity = time.monotonic()
    await set_mode("active_session", "startup grace period")

    while True:
        try:
            await asyncio.sleep(poll_interval)
            await _tick(idle_threshold)
        except asyncio.CancelledError:
            logger.info("Mode monitor cancelled")
            raise
        except Exception as exc:
            logger.warning("Mode monitor tick failed: %s", exc)


# ── Internal helpers ──────────────────────────────────────────────

async def _tick(idle_threshold: float = 180.0) -> None:
    """Single presence check — called on startup and every poll interval."""
    import time
    idle_seconds = _get_last_input_seconds()

    # Voice activity counts as presence — if wake word or transcript fired recently,
    # treat user as active regardless of keyboard/mouse idle time.
    voice_idle_seconds = time.monotonic() - _last_voice_activity
    effective_idle = min(idle_seconds, voice_idle_seconds)

    if effective_idle < idle_threshold:
        if _current_mode not in ("active_session", "awaiting_confirmation"):
            await set_mode(
                "active_session",
                f"input detected ({effective_idle:.0f}s idle, threshold {idle_threshold:.0f}s)",
            )
    else:
        if _current_mode not in ("autonomous", "awaiting_confirmation", "recovering"):
            await set_mode(
                "autonomous",
                f"no input for {effective_idle:.0f}s (threshold {idle_threshold:.0f}s)",
            )


def _get_last_input_seconds() -> float:
    """Return seconds since last keyboard/mouse input using Windows API.

    Falls back to 0.0 on non-Windows (treat as present).
    """
    if platform.system() != "Windows":
        return 0.0

    try:
        class _LASTINPUTINFO(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

        info = _LASTINPUTINFO()
        info.cbSize = ctypes.sizeof(_LASTINPUTINFO)
        ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info))  # type: ignore[attr-defined]

        tick_count = ctypes.windll.kernel32.GetTickCount()  # type: ignore[attr-defined]
        elapsed_ms = tick_count - info.dwTime
        return max(0.0, elapsed_ms / 1000.0)
    except Exception as exc:
        logger.debug("GetLastInputInfo failed: %s", exc)
        return 0.0


async def _send_return_summary(away_since: datetime | None) -> None:
    """Build and send a natural-language summary of what happened while away."""
    try:
        from bot import db
        from bot import notifier
        from bot import observer
        from bot import provider

        now = datetime.now(timezone.utc)
        elapsed_minutes = 0
        if away_since is not None:
            elapsed_minutes = int((now - away_since).total_seconds() // 60)

        # Collect raw facts
        facts: list[str] = []

        if elapsed_minutes > 0:
            facts.append(f"User was away for {elapsed_minutes} minute(s).")

        # Escalation pings that fired during absence
        absence_events = observer.get_absence_log()
        observer.clear_absence_log()
        if absence_events:
            for ev in absence_events:
                facts.append(f"Escalation fired: {ev}")
        else:
            facts.append("No escalations fired while away.")

        # Conversations sent during absence
        try:
            recent_convs = await db.get_recent_conversations(20)
            if away_since is not None:
                away_msgs = [
                    c for c in recent_convs
                    if c.get("role") == "assistant"
                    and _parse_ts(c.get("timestamp", "")) >= away_since
                ]
                if away_msgs:
                    facts.append(f"{len(away_msgs)} message(s) sent to user while away.")
        except Exception as exc:
            logger.debug("Return summary: failed to fetch conversations: %s", exc)

        # Vision triggers during absence
        try:
            triggers = await db.get_recent_vision_triggers(5)
            if away_since is not None:
                away_triggers = [
                    t for t in triggers
                    if _parse_ts(t.get("occurred_at", "")) >= away_since
                ]
                if away_triggers:
                    types = [t["trigger_type"] for t in away_triggers]
                    facts.append(f"Screen vision fired {len(away_triggers)} time(s): {', '.join(types)}.")
        except Exception as exc:
            logger.debug("Return summary: failed to fetch vision triggers: %s", exc)

        raw_facts = "\n".join(facts)

        # Ask GPT to write a natural welcome-back message
        try:
            response = await provider.create_chat_completion(
                role="fast",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are Kira, a personal AI assistant. The user has just returned to their computer. "
                            "Write a short, natural welcome-back message (2-4 sentences) based on the facts below. "
                            "Be direct and informative, not overly chatty. Address the user as 'sir'. "
                            "If nothing notable happened, keep it brief."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Facts about what happened while the user was away:\n{raw_facts}",
                    },
                ],
                max_tokens=120,
                temperature=0.4,
            )
            summary = response.choices[0].message.content.strip()
        except Exception as exc:
            logger.warning("Return summary GPT call failed, using raw facts: %s", exc)
            summary = "Welcome back, sir.\n" + raw_facts

        await notifier.send(summary)
        logger.info("Return summary sent (%d chars)", len(summary))

        # Also speak it on the local system
        try:
            from bot import local_voice
            await local_voice.speak(summary)
        except Exception as exc:
            logger.debug("Return summary TTS failed: %s", exc)

    except Exception as exc:
        logger.warning("Failed to send return summary: %s", exc)


def _parse_ts(ts_str: str) -> datetime:
    """Parse a DB timestamp string to an aware UTC datetime, or epoch on failure."""
    try:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
