"""Feature 5 — Ambient screen context for Kira.

Runs a background async loop that takes a screenshot every 15 minutes,
asks the vision model to describe what the user is doing, and stores
that description. The description is injected into every brain call
so Kira has passive awareness of what's on screen without being asked.

Configuration:
    KIRA_AMBIENT_ENABLED=true        # set false to disable
    KIRA_AMBIENT_INTERVAL_MIN=15     # minutes between snapshots (default 15)

Public API
----------
start()              — launch the background task
stop()               — cancel the task
get_description()    — return the latest description string (thread-safe)
"""

from __future__ import annotations

import asyncio
import logging
import os

logger = logging.getLogger(__name__)

_task: asyncio.Task | None = None
_description: str = ""

_DEFAULT_INTERVAL_MIN = 15
_VISION_PROMPT = (
    "Describe what the user is currently doing on their computer in one concise sentence. "
    "Focus on the active application and task (e.g. 'writing code in VS Code', "
    "'watching a YouTube video', 'reading a document'). "
    "If the screen is locked or blank, say so. Be factual, no commentary."
)


def get_description() -> str:
    """Return the latest ambient screen description. Empty string if none yet."""
    return _description


async def _capture_once() -> None:
    global _description
    try:
        from bot import screen_vision as _sv
        import asyncio as _asyncio

        png_bytes = await _asyncio.to_thread(_sv.take_screenshot_png)
        if not png_bytes:
            return

        import base64
        b64 = base64.b64encode(png_bytes).decode()

        from bot import provider
        response = await provider.create_vision_completion(
            prompt=_VISION_PROMPT,
            image_b64=b64,
            max_tokens=80,
            image_format="png",
        )
        desc = (response.choices[0].message.content or "").strip()
        if desc:
            _description = desc
            logger.debug("Ambient screen: %r", desc)
    except Exception as exc:
        logger.debug("Ambient capture failed: %s", exc)


async def _loop() -> None:
    enabled = os.environ.get("KIRA_AMBIENT_ENABLED", "true").strip().lower()
    if enabled in ("false", "0", "no"):
        logger.info("Ambient screen context disabled via KIRA_AMBIENT_ENABLED")
        return

    interval_min = int(os.environ.get("KIRA_AMBIENT_INTERVAL_MIN", _DEFAULT_INTERVAL_MIN))
    logger.info("Ambient screen loop started — interval=%dm", interval_min)

    # First capture after a short delay so startup isn't noisy
    await asyncio.sleep(60)
    await _capture_once()

    while True:
        await asyncio.sleep(interval_min * 60)
        await _capture_once()


def start() -> None:
    """Launch the ambient screen loop as a background asyncio task."""
    global _task
    if _task is not None and not _task.done():
        return
    _task = asyncio.ensure_future(_loop())
    logger.info("Ambient screen task started")


def stop() -> None:
    """Cancel the ambient screen loop task."""
    global _task
    if _task is not None and not _task.done():
        _task.cancel()
        _task = None
    logger.info("Ambient screen task stopped")
