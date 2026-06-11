"""The transcript router: classify one transcript and dispatch to a handler."""

from __future__ import annotations

import logging
import os
from datetime import datetime

from bot import app_control
from bot import db
from bot import identity
from bot import screen_vision
from bot.voice_runtime import routing
from bot.voice_runtime import session
from bot.voice_runtime.brain_fallback import _handle_with_brain
from bot.voice_runtime.desktop_agent import _handle_desktop_action
from bot.voice_runtime.executor import execute_command
from bot.voice_runtime.models import ConfirmCallback, LocalVoiceResult, ParsedCommand
from bot.voice_runtime.multistep import _handle_multistep
from bot.voice_runtime.parsing import parse_deterministic
from bot.voice_runtime.webcam_intent import _handle_webcam_intent, _mentions_webcam

logger = logging.getLogger(__name__)


def _llm_fallback_enabled() -> bool:
    """Whether to fall back to the LLM command parser for unmatched transcripts."""
    def _flag(name: str) -> bool:
        return os.environ.get(name, "true").strip().lower() not in {"false", "0", "no", "off"}
    return _flag("KIRA_LLM_FALLBACK_ENABLED") and _flag("KIRA_AI_FALLBACK_ENABLED")


async def handle_transcript(
    transcript: str,
    *,
    confirm: ConfirmCallback | None = None,
    config: app_control.AppsConfig | None = None,
) -> tuple[ParsedCommand | None, LocalVoiceResult]:
    """Parse and execute one transcript."""
    session.set_last_voice_activity(datetime.now())

    # Correction detection — log before processing so the reflector sees the signal
    if routing._is_correction(transcript) and session.get_command_history():
        await _log(transcript, "correction detected", "correction")

    # Memory / identity commands are intercepted before anything else.
    memory_reply = identity.extract_memory_from_transcript(transcript)
    if memory_reply:
        session.append_command(transcript, memory_reply[:80])
        await _log(transcript, memory_reply, "memory")
        return None, LocalVoiceResult(ok=True, message=memory_reply, spoken=memory_reply)

    if routing._is_identity_query(transcript):
        spoken = routing._build_identity_reply(transcript)
        session.append_command(transcript, spoken[:80])
        await _log(transcript, spoken, "identity")
        return None, LocalVoiceResult(ok=True, message=spoken, spoken=spoken)

    if routing._is_screen_query(transcript):
        description = await screen_vision.capture_screen()
        session.append_command(transcript, description[:80])
        await _log(transcript, description, "screen")
        return None, LocalVoiceResult(ok=True, message=description, spoken=description)

    # Only invoke the (network) webcam intent classifier when the request
    # plausibly concerns the camera — otherwise every command would pay for it.
    if _mentions_webcam(transcript):
        webcam_result = await _handle_webcam_intent(transcript)
        if webcam_result is not None:
            session.append_command(transcript, webcam_result.message[:80])
            await _log(transcript, webcam_result.message, "webcam")
            return None, webcam_result

    if routing._is_multistep(transcript):
        result = await _handle_multistep(transcript)
        session.append_command(transcript, result.message[:80])
        await _log(transcript, result.message, "multistep")
        return None, result

    apps_config = config or app_control.load_apps_config()
    parsed = parse_deterministic(transcript, apps_config)
    if parsed is not None:
        result = await execute_command(parsed, confirm=confirm, config=apps_config)
        session.append_command(transcript, result.message[:80])
        await _log(transcript, result.message, "desktop")
        return parsed, result

    # No deterministic match. Optionally fall back to the LLM command parser
    # before the heavier brain/desktop routing.
    if not _llm_fallback_enabled():
        result = LocalVoiceResult(
            ok=False,
            message=(
                "I couldn't match that to a local command, and AI fallback is disabled. "
                "Try a deterministic command like 'open chrome' or 'press enter'."
            ),
            spoken="I couldn't match that to a command, and AI fallback is off.",
        )
        session.append_command(transcript, result.message[:80])
        return None, result

    # Looked up via the facade so tests patching local_voice.parse_with_llm
    # (and any future runtime override of it) are honoured.
    from bot import local_voice as _public
    llm_parsed = await _public.parse_with_llm(transcript)
    if llm_parsed is not None:
        result = await execute_command(llm_parsed, confirm=confirm, config=apps_config)
        session.append_command(transcript, result.message[:80])
        await _log(transcript, result.message, "llm")
        return llm_parsed, result

    # Informational questions go straight to brain — the desktop action LLM
    # routinely misclassifies "how's my PNL", "what's the weather" etc. as
    # UI actions when the relevant app happens to be visible on screen.
    if not routing._is_desktop_action_candidate(transcript):
        result = await _handle_with_brain(transcript)
        session.append_command(transcript, result.message[:80])
        session.append_session_history(transcript, result.spoken)
        await _log(transcript, result.spoken, "brain")
        return None, result

    # For everything else: vision-based desktop action first (LLM sees screen,
    # uses real coordinates). Falls through to brain for non-UI requests.
    desktop_result = await _handle_desktop_action(transcript)
    if desktop_result is not None:
        session.append_command(transcript, desktop_result.message[:80])
        await _log(transcript, desktop_result.message, "desktop")
        return None, desktop_result

    result = await _handle_with_brain(transcript)
    session.append_command(transcript, result.message[:80])
    session.append_session_history(transcript, result.spoken)
    await _log(transcript, result.spoken, "brain")
    return None, result


async def _log(transcript: str, result: str, intent: str) -> None:
    """Fire-and-forget voice command persistence to both logs."""
    try:
        await db.log_voice_command(transcript, result, intent)
    except Exception:
        logger.warning("Failed to persist voice command to DB", exc_info=True)
    try:
        await db.log_conversation("user", transcript, channel="voice")
        if result:
            await db.log_conversation("assistant", result, channel="voice")
    except Exception:
        logger.warning("Failed to persist voice conversation to DB", exc_info=True)
