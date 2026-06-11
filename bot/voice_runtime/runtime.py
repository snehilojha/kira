"""Entry points: the standalone run_loop, the main.py task, and the project hook."""

from __future__ import annotations

import asyncio
import logging
import os

from dotenv import load_dotenv

from bot import app_control
from bot import db
from bot import overlay
from bot import provider
from bot.voice_runtime.tts import _prerender_cues, speak
from bot.voice_runtime.triggers import (
    _DEFAULT_HOTKEY,
    _DEFAULT_TRIGGER,
    _run_enter_loop,
    _run_hotkey_loop,
    _run_wake_word_loop,
    _confirm_in_terminal,
)
from bot.voice_runtime.util import (
    _DEFAULT_RECORD_SECONDS,
    _DEFAULT_SAMPLE_RATE,
    _ENV_PATH,
    _setup_logging,
)

logger = logging.getLogger(__name__)


def _register_project_switch_hook(speak_fn) -> None:
    """Register a callback with observer that speaks a nudge on project switch."""
    from bot import observer as _observer

    async def _on_switch(project_summary: str) -> None:
        try:
            response = await provider.create_chat_completion(
                role="fast",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are Kira, a personal AI assistant. "
                            "The user just switched to a project in their editor. "
                            "Given the project summary, say ONE short sentence surfacing the most useful context — "
                            "last commit, active branch, recently changed files, or an open note. "
                            "Be specific, not generic. No greeting. No markdown. Spoken aloud via TTS."
                        ),
                    },
                    {"role": "user", "content": project_summary},
                ],
                max_tokens=60,
                temperature=0.3,
            )
            line = (response.choices[0].message.content or "").strip()
            if line:
                await speak_fn(line)
        except Exception as exc:
            logger.debug("Project switch nudge failed: %s", exc)

    _observer.register_project_switch_callback(_on_switch)


async def run_loop() -> None:
    """Run the local push-to-talk loop until interrupted."""
    load_dotenv(_ENV_PATH, override=True)
    _setup_logging()

    record_seconds = float(os.environ.get("KIRA_LOCAL_VOICE_RECORD_SECONDS", _DEFAULT_RECORD_SECONDS))
    sample_rate = int(os.environ.get("KIRA_LOCAL_VOICE_SAMPLE_RATE", _DEFAULT_SAMPLE_RATE))
    trigger = os.environ.get("KIRA_LOCAL_VOICE_TRIGGER", _DEFAULT_TRIGGER).strip().lower()
    hotkey = os.environ.get("KIRA_LOCAL_VOICE_HOTKEY", _DEFAULT_HOTKEY).strip() or _DEFAULT_HOTKEY
    config = app_control.load_apps_config()

    wake_word_name = os.environ.get("KIRA_WAKE_WORD", "hey_jarvis").strip()
    wake_word_threshold = float(os.environ.get("KIRA_WAKE_WORD_THRESHOLD", "0.5"))

    await db.init_db()
    overlay.start()
    await _prerender_cues()
    print("Kira local voice is ready.")
    logger.info("Kira local voice is ready")

    # Feature 4 + 5: start proactive and ambient loops
    from bot import proactive as _proactive
    from bot import ambient as _ambient
    _ambient.start()
    _proactive.start(speak)
    _register_project_switch_hook(speak)

    if trigger == "enter":
        await _run_enter_loop(record_seconds=record_seconds, sample_rate=sample_rate, config=config)
        return

    if trigger == "wake_word":
        await _run_wake_word_loop(
            wake_word=wake_word_name,
            threshold=wake_word_threshold,
            record_seconds=record_seconds,
            sample_rate=sample_rate,
            config=config,
        )
        return

    await _run_hotkey_loop(
        hotkey=hotkey,
        record_seconds=record_seconds,
        sample_rate=sample_rate,
        config=config,
    )


async def start_as_task() -> None:
    """Start the voice loop as a background task from main.py.

    Skips env/logging/db setup (main.py already did those).
    Reads trigger config from environment and runs the appropriate loop.
    Risky command confirmations are routed to Telegram (no terminal available).
    """
    from bot import mode as mode_mod
    from bot import notifier

    record_seconds = float(os.environ.get("KIRA_LOCAL_VOICE_RECORD_SECONDS", _DEFAULT_RECORD_SECONDS))
    sample_rate = int(os.environ.get("KIRA_LOCAL_VOICE_SAMPLE_RATE", _DEFAULT_SAMPLE_RATE))
    trigger = os.environ.get("KIRA_LOCAL_VOICE_TRIGGER", _DEFAULT_TRIGGER).strip().lower()
    hotkey = os.environ.get("KIRA_LOCAL_VOICE_HOTKEY", _DEFAULT_HOTKEY).strip() or _DEFAULT_HOTKEY
    wake_word_name = os.environ.get("KIRA_WAKE_WORD", "hey_jarvis").strip()
    wake_word_threshold = float(os.environ.get("KIRA_WAKE_WORD_THRESHOLD", "0.5"))
    config = app_control.load_apps_config()
    await _prerender_cues()

    async def _smart_confirm(command_preview: str) -> bool:
        if mode_mod.is_autonomous():
            await speak("Check Telegram to confirm.")
            return await notifier.confirm_via_telegram(command_preview)
        return await _confirm_in_terminal(command_preview)

    confirm = _smart_confirm

    logger.info("Kira voice loop starting (trigger=%s)", trigger)

    # Feature 4 + 5: start proactive and ambient loops
    from bot import proactive as _proactive
    from bot import ambient as _ambient
    _ambient.start()
    _proactive.start(speak)
    _register_project_switch_hook(speak)

    # No catch-all here: cancellation must propagate for clean shutdown, and
    # crashes must reach the supervisor (bot.supervision) so the voice loop is
    # logged, alerted, and restarted rather than silently dying.
    if trigger == "enter":
        await _run_enter_loop(record_seconds=record_seconds, sample_rate=sample_rate, config=config, confirm=confirm)
    elif trigger == "wake_word":
        await _run_wake_word_loop(
            wake_word=wake_word_name,
            threshold=wake_word_threshold,
            record_seconds=record_seconds,
            sample_rate=sample_rate,
            config=config,
            confirm=confirm,
        )
    else:
        await _run_hotkey_loop(
            hotkey=hotkey,
            record_seconds=record_seconds,
            sample_rate=sample_rate,
            config=config,
            confirm=confirm,
        )


def main() -> None:
    """Console entry point."""
    try:
        asyncio.run(run_loop())
    except KeyboardInterrupt:
        print("\nKira local voice stopped.")
        logger.info("Kira local voice stopped")
    except Exception:
        logger.exception("Kira local voice crashed")
        raise
