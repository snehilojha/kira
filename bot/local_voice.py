"""Compatibility facade for Kira's local voice runtime.

The implementation was split out of this once-2000-line module into the
``bot.voice_runtime`` package. This file preserves the original public import
surface so existing callers — ``main.py``, ``mode.py``, ``proactive.py`` and
the test suite — keep working unchanged. Import from here, not from the
submodules.

``app_control`` and ``desktop_control`` are intentionally imported into this
namespace because the tests patch attributes on them via
``local_voice.app_control.open_app`` / ``local_voice.desktop_control.execute_command``.
Those patch the real shared module objects, so the runtime (which calls the
same module objects) sees the patches.
"""

from __future__ import annotations

from bot import app_control
from bot import desktop_control

from bot.voice_runtime.models import ConfirmCallback, LocalVoiceResult, ParsedCommand
from bot.voice_runtime.parsing import is_risky_command, parse_deterministic, parse_with_llm
from bot.voice_runtime.executor import execute_command
from bot.voice_runtime.dispatcher import handle_transcript
from bot.voice_runtime.session import (
    DESKTOP_ARM_STATE as _DESKTOP_ARM_STATE,
    clear_session_history,
    get_last_voice_activity,
    is_stay_hot,
    set_stay_hot,
)
from bot.voice_runtime.tts import speak
from bot.voice_runtime.capture import record_wav_bytes, run_capture_once
from bot.voice_runtime.triggers import (
    _DEFAULT_HOTKEY,
    _DEFAULT_TRIGGER,
    _queue_hold_trigger,
    _queue_hotkey_trigger,
    _resolve_hotkey_behavior,
)
from bot.voice_runtime.runtime import main, run_loop, start_as_task

__all__ = [
    "app_control",
    "desktop_control",
    "ConfirmCallback",
    "LocalVoiceResult",
    "ParsedCommand",
    "is_risky_command",
    "parse_deterministic",
    "parse_with_llm",
    "execute_command",
    "handle_transcript",
    "clear_session_history",
    "get_last_voice_activity",
    "is_stay_hot",
    "set_stay_hot",
    "_DESKTOP_ARM_STATE",
    "speak",
    "record_wav_bytes",
    "run_capture_once",
    "_DEFAULT_HOTKEY",
    "_DEFAULT_TRIGGER",
    "_queue_hotkey_trigger",
    "_queue_hold_trigger",
    "_resolve_hotkey_behavior",
    "main",
    "run_loop",
    "start_as_task",
]


if __name__ == "__main__":
    main()
