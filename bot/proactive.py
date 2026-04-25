"""Feature 4 — Proactive check-ins for Kira.

Runs a background async loop. Every 45 minutes (within configured time slots)
Kira decides whether she has something worth saying — if yes, she speaks it.

Configuration (all in .env):
    KIRA_PROACTIVE_ENABLED=true          # set to false to disable entirely
    KIRA_PROACTIVE_SLOTS=19:00-04:00,... # comma-separated HH:MM-HH:MM ranges
    KIRA_PROACTIVE_INTERVAL_MIN=45       # minutes between checks (default 45)
    KIRA_PROACTIVE_COOLDOWN_MIN=30       # minimum minutes between spoken messages

Public API
----------
start(speak_fn)   — launch the background task
stop()            — cancel the task
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, time
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

SpeakFn = Callable[[str], Awaitable[None]]

_task: asyncio.Task | None = None
_last_spoken: datetime | None = None

_DEFAULT_SLOTS = "19:00-04:00"
_DEFAULT_INTERVAL_MIN = 45
_DEFAULT_COOLDOWN_MIN = 30


def _parse_slots(raw: str) -> list[tuple[time, time]]:
    """Parse 'HH:MM-HH:MM,...' into list of (start, end) time pairs."""
    slots = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            start_str, end_str = part.split("-", 1)
            sh, sm = map(int, start_str.strip().split(":"))
            eh, em = map(int, end_str.strip().split(":"))
            slots.append((time(sh, sm), time(eh, em)))
        except Exception:
            logger.warning("Could not parse proactive slot %r — skipping", part)
    return slots


def _in_slots(now: datetime, slots: list[tuple[time, time]]) -> bool:
    """Return True if current time falls within any configured slot.

    Slots that wrap midnight (e.g. 19:00-04:00) are handled correctly.
    """
    t = now.time()
    for start, end in slots:
        if start <= end:
            if start <= t <= end:
                return True
        else:
            # wraps midnight
            if t >= start or t <= end:
                return True
    return False


async def _decide(ambient_context: str) -> str | None:
    """Ask the fast model if there's anything worth saying right now.

    Returns a spoken sentence or None if nothing is worth saying.
    """
    from bot import identity as _identity
    from bot import provider

    user_name = _identity.get_user_name() or "the user"
    facts = _identity.get_all_facts()
    facts_block = (" ".join(f[:80] for f in facts[:4])) if facts else ""

    now = datetime.now()
    time_str = now.strftime("%A %d %B, %H:%M")

    system = (
        "You are Kira, a personal AI assistant. You proactively check in with your user.\n"
        "Given the current time and context, decide if there is ONE genuinely useful or caring thing to say.\n\n"
        "Rules:\n"
        "- Be natural, warm, and brief (one sentence max).\n"
        "- Only speak if it adds real value: health reminder, timely nudge, quick observation.\n"
        "- Examples: reminding them it's late, they've been working long, upcoming time of day things.\n"
        "- Do NOT ask questions. Do NOT give generic affirmations. Do NOT be chatty.\n"
        "- If nothing is genuinely worth saying right now, reply with exactly: SKIP\n"
        "- Never start with 'Certainly', 'Sure', 'Of course', 'I', or 'As your assistant'.\n"
        "- No markdown. Response is spoken aloud via TTS.\n"
    )

    user_msg = (
        f"Current time: {time_str}\n"
        f"User: {user_name}\n"
        + (f"Known facts: {facts_block}\n" if facts_block else "")
        + (f"Screen context: {ambient_context}\n" if ambient_context else "")
        + "Should Kira say something right now? If yes, say it. If no, reply SKIP."
    )

    try:
        response = await provider.create_chat_completion(
            role="fast",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.7,
            max_tokens=80,
        )
        reply = (response.choices[0].message.content or "").strip()
        if not reply or reply.upper() == "SKIP" or "SKIP" in reply[:10]:
            return None
        return reply
    except Exception as exc:
        logger.warning("Proactive decide call failed: %s", exc)
        return None


async def _loop(speak_fn: SpeakFn) -> None:
    global _last_spoken

    enabled = os.environ.get("KIRA_PROACTIVE_ENABLED", "true").strip().lower()
    if enabled in ("false", "0", "no"):
        logger.info("Proactive check-ins disabled via KIRA_PROACTIVE_ENABLED")
        return

    slots_raw = os.environ.get("KIRA_PROACTIVE_SLOTS", _DEFAULT_SLOTS)
    interval_min = int(os.environ.get("KIRA_PROACTIVE_INTERVAL_MIN", _DEFAULT_INTERVAL_MIN))
    cooldown_min = int(os.environ.get("KIRA_PROACTIVE_COOLDOWN_MIN", _DEFAULT_COOLDOWN_MIN))
    slots = _parse_slots(slots_raw)

    logger.info(
        "Proactive loop started — interval=%dm, cooldown=%dm, slots=%s",
        interval_min, cooldown_min, slots_raw,
    )

    while True:
        await asyncio.sleep(interval_min * 60)

        now = datetime.now()

        if slots and not _in_slots(now, slots):
            logger.debug("Proactive: outside time slots, skipping")
            continue

        if _last_spoken is not None:
            elapsed_min = (now - _last_spoken).total_seconds() / 60
            if elapsed_min < cooldown_min:
                logger.debug("Proactive: cooldown active (%.0fm < %dm), skipping", elapsed_min, cooldown_min)
                continue

        # Pull latest ambient context (Feature 5 module, graceful fallback)
        ambient_context = ""
        try:
            from bot import ambient as _ambient
            ambient_context = _ambient.get_description()
        except Exception:
            pass

        message = await _decide(ambient_context)
        if message:
            logger.info("Proactive speaking: %r", message)
            _last_spoken = datetime.now()
            try:
                from bot import overlay as _overlay
                _overlay.set_state("speaking")
            except Exception:
                pass
            await speak_fn(message)
            try:
                from bot import overlay as _overlay
                _overlay.set_state("idle")
            except Exception:
                pass


def start(speak_fn: SpeakFn) -> None:
    """Launch the proactive loop as a background asyncio task."""
    global _task
    if _task is not None and not _task.done():
        return
    _task = asyncio.ensure_future(_loop(speak_fn))
    logger.info("Proactive check-in task started")


def stop() -> None:
    """Cancel the proactive loop task."""
    global _task
    if _task is not None and not _task.done():
        _task.cancel()
        _task = None
    logger.info("Proactive check-in task stopped")
