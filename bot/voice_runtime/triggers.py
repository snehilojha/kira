"""Trigger loops (enter / global hotkey / wake word) that drive captures."""

from __future__ import annotations

import asyncio
import threading

from bot import app_control
from bot import overlay
from bot import wake_word as wake_word_mod
from bot.voice_runtime import session
from bot.voice_runtime.capture import run_capture_once
from bot.voice_runtime.models import ConfirmCallback
from bot.voice_runtime.tts import _activation_cue
from bot.voice_runtime.util import _DEFAULT_HOTKEY, _DEFAULT_TRIGGER  # re-exported via facade

__all__ = [
    "_run_enter_loop",
    "_run_hotkey_loop",
    "_run_wake_word_loop",
    "_queue_hotkey_trigger",
    "_queue_hold_trigger",
    "_resolve_hotkey_behavior",
    "_confirm_in_terminal",
    "_DEFAULT_TRIGGER",
    "_DEFAULT_HOTKEY",
]


async def _confirm_in_terminal(command_preview: str) -> bool:
    answer = await asyncio.to_thread(
        input,
        f"Confirm risky command `{command_preview}`? Type y to run: ",
    )
    return answer.strip().lower() in {"y", "yes"}


async def _run_enter_loop(
    *,
    record_seconds: float,
    sample_rate: int,
    config: app_control.AppsConfig,
    confirm: ConfirmCallback | None = None,
) -> None:
    """Run the original terminal Enter push-to-talk loop."""
    print(f"Press Enter to record {record_seconds:g}s. Press Ctrl+C to stop.")
    while True:
        await asyncio.to_thread(input, "\nPress Enter and speak...")
        await run_capture_once(
            record_seconds=record_seconds,
            sample_rate=sample_rate,
            confirm=confirm or _confirm_in_terminal,
            config=config,
        )


async def _run_hotkey_loop(
    *,
    hotkey: str,
    record_seconds: float,
    sample_rate: int,
    config: app_control.AppsConfig,
    confirm: ConfirmCallback | None = None,
) -> None:
    """Run the global hotkey push-to-talk loop."""
    try:
        import keyboard
    except ImportError:
        print("Global hotkey support needs the `keyboard` package. Falling back to Enter mode.")
        await _run_enter_loop(record_seconds=record_seconds, sample_rate=sample_rate, config=config, confirm=confirm)
        return

    loop = asyncio.get_running_loop()
    trigger_queue: asyncio.Queue[None] = asyncio.Queue(maxsize=1)

    def _on_hotkey() -> None:
        loop.call_soon_threadsafe(_queue_hotkey_trigger, trigger_queue)

    try:
        keyboard.add_hotkey(hotkey, _on_hotkey)
    except Exception as exc:
        print(f"Could not register global hotkey `{hotkey}`: {exc}")
        print("Falling back to Enter mode.")
        await _run_enter_loop(record_seconds=record_seconds, sample_rate=sample_rate, config=config, confirm=confirm)
        return

    print(f"Press {hotkey} to record. Press Ctrl+C to stop.")
    try:
        while True:
            if not session.is_stay_hot():
                await trigger_queue.get()
                while not trigger_queue.empty():
                    trigger_queue.get_nowait()
                await _activation_cue()
            overlay.show()
            try:
                await run_capture_once(
                    record_seconds=record_seconds,
                    sample_rate=sample_rate,
                    confirm=confirm or _confirm_in_terminal,
                    config=config,
                    kira_filter=session.is_stay_hot(),
                )
            finally:
                if not session.is_stay_hot():
                    overlay.hide()
    finally:
        try:
            keyboard.remove_hotkey(hotkey)
        except Exception:
            pass


async def _run_wake_word_loop(
    *,
    wake_word: str,
    threshold: float,
    record_seconds: float,
    sample_rate: int,
    config: app_control.AppsConfig,
    confirm: ConfirmCallback | None = None,
) -> None:
    """Run the wake-word triggered voice loop."""
    loop = asyncio.get_running_loop()
    trigger_queue: asyncio.Queue[None] = asyncio.Queue(maxsize=1)

    detector = wake_word_mod.start(
        trigger_queue,
        loop,
        model_name=wake_word,
        threshold=threshold,
    )
    print(f"Say '{wake_word.replace('_', ' ')}' to activate Kira. Press Ctrl+C to stop.")
    try:
        while True:
            from bot import ui_mode as _ui_mode
            in_full_mode = _ui_mode.is_full_mode()
            in_stay_hot = session.is_stay_hot()

            if not in_full_mode and not in_stay_hot:
                await trigger_queue.get()
                while not trigger_queue.empty():
                    trigger_queue.get_nowait()
                await _activation_cue()

            overlay.show()
            try:
                await run_capture_once(
                    record_seconds=record_seconds,
                    sample_rate=sample_rate,
                    confirm=confirm or _confirm_in_terminal,
                    config=config,
                    kira_filter=in_stay_hot and not in_full_mode,
                )
            finally:
                if not _ui_mode.is_full_mode() and not session.is_stay_hot():
                    overlay.hide()
    finally:
        detector.stop()


def _queue_hotkey_trigger(queue: asyncio.Queue[None]) -> None:
    """Queue one hotkey trigger, dropping extras while a capture is pending."""
    try:
        queue.put_nowait(None)
    except asyncio.QueueFull:
        print("Hotkey pressed while Kira is already busy; ignoring.")


def _queue_hold_trigger(queue: "asyncio.Queue[threading.Event]", stop_event: threading.Event) -> None:
    """Queue one hold-to-talk trigger carrying its stop event.

    If a capture is already pending the queue is full; release the stranded
    press immediately by setting its stop event so nothing waits on it.
    """
    try:
        queue.put_nowait(stop_event)
    except asyncio.QueueFull:
        stop_event.set()
        print("Hold trigger while Kira is already busy; ignoring.")


def _resolve_hotkey_behavior(hotkey: str, behavior: str) -> str:
    """Resolve the effective trigger behavior for a hotkey.

    "hold" (push-to-talk) is only physically detectable for a single key —
    a key combination like ctrl+alt+k has no reliable single release event,
    so it falls back to "tap". Anything other than "hold" is "tap".
    """
    if (behavior or "").strip().lower() == "hold":
        return "tap" if "+" in hotkey else "hold"
    return "tap"
