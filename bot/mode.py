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

# ── Configuration ─────────────────────────────────────────────────

_IDLE_THRESHOLD_SECONDS = float(
    os.environ.get("KIRA_IDLE_THRESHOLD_SECONDS", "180")
)
_POLL_INTERVAL_SECONDS = float(
    os.environ.get("KIRA_MODE_POLL_SECONDS", "30")
)

# ── Module state ──────────────────────────────────────────────────

_current_mode: ModeState = "idle"
_mode_entered_at: datetime = datetime.now(timezone.utc)

# Timestamp when autonomous mode started — used for return summary scoping.
_autonomous_since: datetime | None = None


# ── Public API ────────────────────────────────────────────────────

def get_mode() -> ModeState:
    """Return the current Kira operating mode."""
    return _current_mode


def is_autonomous() -> bool:
    """Return True when Kira is in autonomous mode."""
    return _current_mode == "autonomous"


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
    logger.info(
        "Mode monitor started (threshold=%.0fs, poll=%.0fs)",
        _IDLE_THRESHOLD_SECONDS,
        _POLL_INTERVAL_SECONDS,
    )

    # On startup, transition out of 'idle' based on current presence signal.
    await _tick()

    while True:
        try:
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)
            await _tick()
        except asyncio.CancelledError:
            logger.info("Mode monitor cancelled")
            raise
        except Exception as exc:
            logger.warning("Mode monitor tick failed: %s", exc)


# ── Internal helpers ──────────────────────────────────────────────

async def _tick() -> None:
    """Single presence check — called on startup and every poll interval."""
    idle_seconds = _get_last_input_seconds()

    if idle_seconds < _IDLE_THRESHOLD_SECONDS:
        if _current_mode not in ("active_session", "awaiting_confirmation"):
            await set_mode(
                "active_session",
                f"input detected ({idle_seconds:.0f}s idle, threshold {_IDLE_THRESHOLD_SECONDS:.0f}s)",
            )
    else:
        if _current_mode not in ("autonomous", "awaiting_confirmation", "recovering"):
            await set_mode(
                "autonomous",
                f"no input for {idle_seconds:.0f}s (threshold {_IDLE_THRESHOLD_SECONDS:.0f}s)",
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
    """Build and send a summary of what happened during an autonomous period."""
    try:
        from bot import db
        from bot import notifier

        lines: list[str] = ["Welcome back, sir."]

        if away_since is not None:
            elapsed = datetime.now(timezone.utc) - away_since
            minutes = int(elapsed.total_seconds() // 60)
            lines.append(f"Kira was in autonomous mode for {minutes} minute(s).")

        # Monitor jobs that fired during the away window.
        try:
            all_jobs = await db.get_all_monitor_jobs()
            fired = []
            for job in all_jobs:
                if job.get("last_fired_at") and away_since is not None:
                    try:
                        fired_dt = datetime.fromisoformat(job["last_fired_at"])
                        if fired_dt.tzinfo is None:
                            fired_dt = fired_dt.replace(tzinfo=timezone.utc)
                        if fired_dt >= away_since:
                            fired.append(job["name"])
                    except ValueError:
                        pass
            if fired:
                lines.append(f"Monitor jobs that triggered: {', '.join(fired)}.")
            else:
                lines.append("No monitor jobs triggered while you were away.")
        except Exception as exc:
            logger.debug("Return summary: failed to fetch monitor jobs: %s", exc)

        # Notifications sent during the away window (assistant messages logged to DB).
        try:
            recent_convs = await db.get_recent_conversations(20)
            if away_since is not None:
                away_messages = [
                    c for c in recent_convs
                    if c.get("role") == "assistant"
                    and _parse_ts(c.get("timestamp", "")) >= away_since
                ]
                if away_messages:
                    lines.append(
                        f"{len(away_messages)} message(s) sent while you were away."
                    )
        except Exception as exc:
            logger.debug("Return summary: failed to fetch conversations: %s", exc)

        # Vision triggers during the away window.
        try:
            triggers = await db.get_recent_vision_triggers(5)
            if away_since is not None:
                away_triggers = [
                    t for t in triggers
                    if _parse_ts(t.get("occurred_at", "")) >= away_since
                ]
                if away_triggers:
                    types = [t["trigger_type"] for t in away_triggers]
                    lines.append(
                        f"Screen vision fired {len(away_triggers)} time(s): {', '.join(types)}."
                    )
        except Exception as exc:
            logger.debug("Return summary: failed to fetch vision triggers: %s", exc)

        await notifier.send("\n".join(lines))
        logger.info("Return summary sent")

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
