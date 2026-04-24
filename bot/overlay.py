"""Minimal always-on-top recording indicator overlay.

Shows a small red dot in the top-right corner while Kira is listening,
then hides it. Runs on a dedicated daemon thread so it never blocks the
asyncio event loop.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk


_CMD_SHOW = "show"
_CMD_HIDE = "hide"
_CMD_STOP = "stop"

_q: queue.Queue[str] = queue.Queue()
_thread: threading.Thread | None = None


def _run(q: queue.Queue[str]) -> None:
    root = tk.Tk()
    root.overrideredirect(True)          # no title bar / borders
    root.attributes("-topmost", True)    # always on top
    root.attributes("-alpha", 0.85)
    root.configure(bg="red")
    root.withdraw()                       # start hidden

    size = 18
    pad = 12
    screen_w = root.winfo_screenwidth()
    x = screen_w - size - pad
    y = pad
    root.geometry(f"{size}x{size}+{x}+{y}")

    canvas = tk.Canvas(root, width=size, height=size, bg="red", highlightthickness=0)
    canvas.pack()
    canvas.create_oval(2, 2, size - 2, size - 2, fill="#ff2222", outline="")

    def _poll() -> None:
        try:
            while True:
                cmd = q.get_nowait()
                if cmd == _CMD_SHOW:
                    root.deiconify()
                elif cmd == _CMD_HIDE:
                    root.withdraw()
                elif cmd == _CMD_STOP:
                    root.destroy()
                    return
        except queue.Empty:
            pass
        root.after(50, _poll)

    root.after(50, _poll)
    root.mainloop()


def start() -> None:
    """Start the overlay thread (call once at program startup)."""
    global _thread
    if _thread is not None and _thread.is_alive():
        return
    _thread = threading.Thread(target=_run, args=(_q,), daemon=True, name="kira-overlay")
    _thread.start()


def show() -> None:
    """Show the recording indicator."""
    _q.put(_CMD_SHOW)


def hide() -> None:
    """Hide the recording indicator."""
    _q.put(_CMD_HIDE)


def stop() -> None:
    """Shut down the overlay thread."""
    _q.put(_CMD_STOP)
