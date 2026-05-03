"""Kira UI mode — tracks whether the full HUD is active.

Compact mode: small orb in corner, shown only on wake word.
Full mode:    full-screen orb + side panels, activated by voice ('take over')
              or hotkey, stays until deactivated ('stand down' / hotkey).
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

_full_mode: bool = False
# Maps hwnd → SW_SHOWNORMAL or SW_SHOWMAXIMIZED so restore puts each window back exactly
_minimized_hwnds: dict[int, int] = {}


def is_full_mode() -> bool:
    return _full_mode


def activate(reason: str = "") -> None:
    global _full_mode
    if _full_mode:
        return
    _full_mode = True
    logger.info("UI full mode activated (%s)", reason or "no reason")
    _notify()
    threading.Thread(target=_minimize_all, daemon=True).start()


def deactivate(reason: str = "") -> None:
    global _full_mode
    if not _full_mode:
        return
    _full_mode = False
    logger.info("UI full mode deactivated (%s)", reason or "no reason")
    _notify()
    threading.Thread(target=_restore_all, daemon=True).start()


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


def _minimize_all() -> None:
    global _minimized_hwnds
    try:
        import win32gui
        import win32con
        import win32process

        kira_pid = __import__("os").getpid()
        snapshot: dict[int, int] = {}

        def _visit(hwnd, _):
            if not win32gui.IsWindowVisible(hwnd):
                return
            if not win32gui.GetWindowText(hwnd):
                return
            try:
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                if pid == kira_pid:
                    return
            except Exception:
                pass
            placement = win32gui.GetWindowPlacement(hwnd)
            show_cmd = placement[1]
            if show_cmd in (win32con.SW_SHOWNORMAL, win32con.SW_SHOWMAXIMIZED,
                            win32con.SW_SHOWNOACTIVATE, win32con.SW_RESTORE):
                snapshot[hwnd] = show_cmd

        win32gui.EnumWindows(_visit, None)
        _minimized_hwnds = snapshot
        for hwnd in snapshot:
            try:
                win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
            except Exception:
                pass
        logger.info("Minimized %d windows for full mode", len(snapshot))
    except Exception as exc:
        logger.warning("Could not minimize windows: %s", exc)


def _restore_all() -> None:
    global _minimized_hwnds
    try:
        import win32gui
        import win32con

        for hwnd, original_cmd in _minimized_hwnds.items():
            try:
                if not win32gui.IsWindow(hwnd):
                    continue
                # Only restore windows that are still minimized — don't touch ones the user
                # un-minimized manually during full mode
                placement = win32gui.GetWindowPlacement(hwnd)
                if placement[1] != win32con.SW_SHOWMINIMIZED:
                    continue
                restore_cmd = (win32con.SW_SHOWMAXIMIZED
                               if original_cmd == win32con.SW_SHOWMAXIMIZED
                               else win32con.SW_RESTORE)
                win32gui.ShowWindow(hwnd, restore_cmd)
            except Exception:
                pass
        logger.info("Restored %d windows after full mode", len(_minimized_hwnds))
    except Exception as exc:
        logger.warning("Could not restore windows: %s", exc)
    finally:
        _minimized_hwnds = {}
