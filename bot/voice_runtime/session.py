"""Shared mutable state for the voice runtime, behind accessor functions.

Every cluster that was a peer inside the old monolith now lives in its own
module, so the globals they used to share via ``global`` statements live here
and are read/written through functions. Keeping them in one place makes the
data flow explicit and avoids each module trying to reach into another's
namespace.
"""

from __future__ import annotations

import collections
import time
from datetime import datetime

# Rolling history of the last 5 voice commands (transcript, result message).
_command_history: "collections.deque[tuple[str, str]]" = collections.deque(maxlen=5)

# Persistent in-session conversation history for LLM context (max 16 = 8 turns).
_session_history: list[dict] = []
_SESSION_HISTORY_MAX = 16

# Last spoken response — replayed on "say that again".
_last_spoken: str = ""

# Timestamp of the most recent voice command — read by proactive.py via the
# facade to detect silence.
_last_voice_activity: "datetime | None" = None

# HWND of the window in focus just before Kira started listening. Restored
# before desktop control actions so keystrokes/scroll hit the right app.
_last_user_hwnd: int = 0

# When True, the voice loop re-arms immediately after each response (compact
# mode conversation). Cleared by "stop listening" / "go to sleep", or when full
# mode is deactivated.
_stay_hot: bool = False


# ── stay-hot (public surface) ─────────────────────────────────────
def is_stay_hot() -> bool:
    return _stay_hot


def set_stay_hot(value: bool) -> None:
    global _stay_hot
    _stay_hot = value


# ── voice activity ────────────────────────────────────────────────
def get_last_voice_activity() -> "datetime | None":
    """Return the timestamp of the most recent voice command (for proactive tracking)."""
    return _last_voice_activity


def set_last_voice_activity(ts: "datetime | None") -> None:
    global _last_voice_activity
    _last_voice_activity = ts


# ── session (conversation) history ────────────────────────────────
def get_session_history() -> list[dict]:
    """Return the live in-session history list (callers may append in place)."""
    return _session_history


def clear_session_history() -> None:
    """Clear the in-session conversation history. Call on session end / stand-down."""
    global _session_history
    _session_history = []


def append_session_history(user_text: str, assistant_text: str) -> None:
    global _session_history
    _session_history.append({"role": "user", "content": user_text})
    _session_history.append({"role": "assistant", "content": assistant_text})
    if len(_session_history) > _SESSION_HISTORY_MAX:
        _session_history = _session_history[-_SESSION_HISTORY_MAX:]


# ── command history ───────────────────────────────────────────────
def get_command_history() -> "collections.deque[tuple[str, str]]":
    return _command_history


def append_command(transcript: str, result_message: str) -> None:
    _command_history.append((transcript, result_message))


# ── last spoken ───────────────────────────────────────────────────
def get_last_spoken() -> str:
    return _last_spoken


def set_last_spoken(value: str) -> None:
    global _last_spoken
    _last_spoken = value


# ── last user window handle ───────────────────────────────────────
def get_last_user_hwnd() -> int:
    return _last_user_hwnd


def set_last_user_hwnd(value: int) -> None:
    global _last_user_hwnd
    _last_user_hwnd = value


# ── desktop-control safety gate ───────────────────────────────────
class DesktopArmState:
    """Time-boxed safety gate for raw desktop control (mouse/keyboard).

    Desktop control commands are rejected unless the gate was armed within the
    last ``duration`` seconds. A misheard transcript therefore can't click or
    type on screen unless the user explicitly said "arm desktop control" just
    before.
    """

    def __init__(self) -> None:
        self._armed_until: float = 0.0

    def arm(self, duration_seconds: float = 5.0) -> None:
        self._armed_until = time.monotonic() + duration_seconds

    def disarm(self) -> None:
        self._armed_until = 0.0

    def is_armed(self) -> bool:
        return time.monotonic() < self._armed_until


# Module-level singleton, re-exported by the facade as ``_DESKTOP_ARM_STATE``.
DESKTOP_ARM_STATE = DesktopArmState()
