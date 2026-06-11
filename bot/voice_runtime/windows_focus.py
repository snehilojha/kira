"""Windows foreground-window capture, lookup, and restore (ctypes / Win32)."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _capture_foreground_hwnd() -> int:
    """Return the topmost visible non-Kira window handle (Windows only).

    The Kira terminal is always in the foreground when the wake word fires,
    so we walk the z-order to find the first window owned by a different process.
    """
    try:
        import ctypes
        import os as _os

        GW_HWNDNEXT = 2
        own_pid = _os.getpid()

        hwnd = ctypes.windll.user32.GetTopWindow(0)
        while hwnd:
            if ctypes.windll.user32.IsWindowVisible(hwnd):
                pid = ctypes.c_ulong()
                ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                if pid.value != own_pid:
                    # Verify it has a title (skip taskbar/tray shells)
                    buf = ctypes.create_unicode_buffer(256)
                    ctypes.windll.user32.GetWindowTextW(hwnd, buf, 256)
                    if buf.value.strip():
                        logger.debug("Captured user window: %r (hwnd=%d)", buf.value, hwnd)
                        return hwnd
            hwnd = ctypes.windll.user32.GetWindow(hwnd, GW_HWNDNEXT)
        return 0
    except Exception:
        logger.debug("Foreground HWND capture failed", exc_info=True)
        return 0


def _find_hwnd_from_transcript(transcript: str) -> int:
    """Find a window handle by matching app keywords in the transcript."""
    _APP_KEYWORDS = {
        "chrome": "chrome",
        "browser": "chrome",
        "youtube": "youtube",
        "firefox": "firefox",
        "edge": "edge",
        "spotify": "spotify",
        "vscode": "visual studio code",
        "visual studio": "visual studio code",
        "notepad": "notepad",
        "explorer": "explorer",
        "terminal": "windows terminal",
        "cmd": "cmd",
    }
    normalized = transcript.lower()
    hint = None
    for keyword, window_hint in _APP_KEYWORDS.items():
        if keyword in normalized:
            hint = window_hint
            break
    if not hint:
        return 0
    return _find_foreground_after_open(hint)


def _find_foreground_after_open(app_hint: str) -> int:
    """Find the hwnd of a recently opened app by matching its window title."""
    try:
        import ctypes
        hint = app_hint.lower().strip()
        GW_HWNDNEXT = 2
        own_pid = __import__("os").getpid()
        hwnd = ctypes.windll.user32.GetTopWindow(0)
        while hwnd:
            if ctypes.windll.user32.IsWindowVisible(hwnd):
                pid = ctypes.c_ulong()
                ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                if pid.value != own_pid:
                    buf = ctypes.create_unicode_buffer(256)
                    ctypes.windll.user32.GetWindowTextW(hwnd, buf, 256)
                    title = buf.value.strip().lower()
                    if title and (not hint or hint in title or "chrome" in title or "firefox" in title or "edge" in title):
                        logger.debug("Found app window after open: %r (hwnd=%d)", buf.value, hwnd)
                        return hwnd
            hwnd = ctypes.windll.user32.GetWindow(hwnd, GW_HWNDNEXT)
    except Exception:
        logger.debug("Window lookup after open failed for hint %r", app_hint, exc_info=True)
    return 0


def _restore_foreground(hwnd: int) -> None:
    """Bring a window to the foreground using AttachThreadInput trick (Windows).

    SetForegroundWindow alone silently fails when the calling process isn't
    already in the foreground. Attaching to the foreground thread first
    bypasses the restriction.
    """
    if not hwnd:
        return
    try:
        import ctypes
        import time

        user32 = ctypes.windll.user32

        # Get thread IDs
        current_thread = ctypes.windll.kernel32.GetCurrentThreadId()
        fg_thread = user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), None)

        # Attach our thread to the foreground thread so we're allowed to steal focus
        attached = False
        if fg_thread and fg_thread != current_thread:
            attached = user32.AttachThreadInput(current_thread, fg_thread, True)

        user32.ShowWindow(hwnd, 9)  # SW_RESTORE — unminimize if needed
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)

        if attached:
            user32.AttachThreadInput(current_thread, fg_thread, False)

        time.sleep(0.3)
    except Exception as exc:
        logger.debug("_restore_foreground failed: %s", exc)
