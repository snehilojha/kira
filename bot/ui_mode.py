"""Kira UI mode — tracks whether the full HUD is active.

Compact mode: small orb in corner, shown only on wake word.
Full mode:    full-screen orb + side panels, activated by voice ('take over')
              or hotkey, stays until deactivated ('stand down' / hotkey).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_full_mode: bool = False


def is_full_mode() -> bool:
    return _full_mode


def activate(reason: str = "") -> None:
    global _full_mode
    if _full_mode:
        return
    _full_mode = True
    logger.info("UI full mode activated (%s)", reason or "no reason")
    _notify()


def deactivate(reason: str = "") -> None:
    global _full_mode
    if not _full_mode:
        return
    _full_mode = False
    logger.info("UI full mode deactivated (%s)", reason or "no reason")
    _notify()


def toggle(reason: str = "") -> None:
    if _full_mode:
        deactivate(reason)
    else:
        activate(reason)


def _notify() -> None:
    """Tell the overlay about the mode change if it's running."""
    try:
        from bot import overlay
        overlay.set_full_mode(_full_mode)
    except Exception:
        pass
