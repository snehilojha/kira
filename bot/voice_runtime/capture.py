"""Microphone recording and one full record→transcribe→execute→speak cycle."""

from __future__ import annotations

import asyncio
import io
import wave

from bot import app_control
from bot import overlay
from bot import voice
from bot.voice_runtime import routing
from bot.voice_runtime import session
from bot.voice_runtime.dispatcher import handle_transcript
from bot.voice_runtime.models import ConfirmCallback, LocalVoiceResult
from bot.voice_runtime.tts import speak
from bot.voice_runtime.util import (
    _DEFAULT_MAX_RECORD_SECONDS,
    _DEFAULT_RECORD_SECONDS,
    _DEFAULT_SAMPLE_RATE,
    _DEFAULT_SILENCE_RMS,
    _DEFAULT_SILENCE_SECONDS,
    _format_command,
    _format_provider_error,
)
from bot.voice_runtime.windows_focus import _capture_foreground_hwnd


async def run_capture_once(
    *,
    record_seconds: float = _DEFAULT_RECORD_SECONDS,
    sample_rate: int = _DEFAULT_SAMPLE_RATE,
    confirm: ConfirmCallback | None = None,
    config: app_control.AppsConfig | None = None,
    kira_filter: bool = False,
) -> LocalVoiceResult:
    """Record, transcribe, execute, and speak one local voice command.

    kira_filter: when True (stay-hot compact mode), silently discard transcripts
    that don't contain 'kira' — prevents ambient noise from triggering responses.
    """
    apps_config = config or app_control.load_apps_config()
    from bot import mode as _mode
    from bot import ui_mode as _ui_mode
    _mode.mark_user_active()
    session.set_last_user_hwnd(_capture_foreground_hwnd())
    print("Recording...")
    overlay.set_state("listening")
    try:
        wav_bytes = await asyncio.to_thread(
            record_wav_bytes,
            seconds=record_seconds,
            sample_rate=sample_rate,
        )
    except Exception as exc:
        message = f"Recording failed: {exc}"
        print(message)
        result = LocalVoiceResult(
            ok=False,
            message=message,
            spoken="Recording failed. Please check the microphone.",
        )
        overlay.set_state("idle")
        await speak(result.spoken)
        return result

    print("Transcribing...")
    overlay.set_state("thinking")
    try:
        transcript = await voice.transcribe(wav_bytes, suffix=".wav")
    except Exception as exc:
        message = _format_provider_error(exc, "Transcription failed")
        print(message)
        result = LocalVoiceResult(
            ok=False,
            message=message,
            spoken="Transcription failed. Please check the API key.",
        )
        overlay.set_state("idle")
        await speak(result.spoken)
        return result
    print(f"Heard: {transcript}")

    # ── Presence: any voice = activity; check for wake phrase ─
    try:
        from bot import presence as _presence
        _presence.on_activity()
        if "kira wake up" in transcript.strip().lower():
            _presence.on_wake_phrase()
            overlay.set_state("idle")
            return LocalVoiceResult(ok=True, message="Presence wake phrase detected.", spoken="")
        if _presence.is_locked():
            overlay.set_state("idle")
            return LocalVoiceResult(ok=True, message="System locked — ignoring command.", spoken="")
    except ImportError:
        pass

    # ── Kira filter (stay-hot compact mode) ───────────────────
    if kira_filter and "kira" not in transcript.strip().lower():
        overlay.set_state("idle")
        return LocalVoiceResult(ok=True, message="Filtered (no 'kira')", spoken="")

    # ── Repeat last response ──────────────────────────────────
    _lower = transcript.strip().lower()
    if any(p in _lower for p in ("say that again", "repeat that", "what did you say", "say again")):
        last_spoken = session.get_last_spoken()
        if last_spoken:
            overlay.set_state("speaking")
            await speak(last_spoken)
            overlay.set_state("idle")
            return LocalVoiceResult(ok=True, message=last_spoken, spoken=last_spoken)

    # ── Full mode voice triggers ───────────────────────────────
    if any(p in _lower for p in ("take over", "takeover", "activate full", "full mode")):
        _ui_mode.activate("voice command")
        spoken = "Full mode activated."
        overlay.set_transcript(transcript, spoken)
        overlay.set_state("speaking")
        await speak(spoken)
        overlay.set_state("idle")
        return LocalVoiceResult(ok=True, message="Full mode activated.", spoken=spoken)
    elif any(p in _lower for p in ("stand down", "deactivate", "exit full", "compact mode")):
        _ui_mode.deactivate("voice command")
        session.clear_session_history()
        session.set_stay_hot(False)
        spoken = "Standing down."
        overlay.set_transcript(transcript, spoken)
        overlay.set_state("speaking")
        await speak(spoken)
        overlay.set_state("idle")
        return LocalVoiceResult(ok=True, message="Compact mode restored.", spoken=spoken)

    # ── Stay-hot mode triggers ─────────────────────────────────
    if any(p in _lower for p in ("stay with me", "keep listening", "stay hot")):
        session.set_stay_hot(True)
        spoken = "I'm with you. Say 'Kira' before your command."
        overlay.set_transcript(transcript, spoken)
        overlay.set_state("speaking")
        await speak(spoken)
        overlay.set_state("hot")
        return LocalVoiceResult(ok=True, message="Stay-hot mode enabled.", spoken=spoken)
    if any(p in _lower for p in ("stop listening", "go to sleep", "kira sleep")):
        session.set_stay_hot(False)
        spoken = "Going quiet."
        overlay.set_transcript(transcript, spoken)
        overlay.set_state("speaking")
        await speak(spoken)
        overlay.set_state("idle")
        return LocalVoiceResult(ok=True, message="Stay-hot mode disabled.", spoken=spoken)

    overlay.set_transcript(transcript, "")

    filler = routing._pick_filler(transcript)
    if filler:
        await speak(filler)

    overlay.set_state("thinking")
    try:
        parsed, result = await handle_transcript(
            transcript,
            confirm=confirm,
            config=apps_config,
        )
    except Exception as exc:
        message = _format_provider_error(exc, "Command handling failed")
        print(message)
        result = LocalVoiceResult(
            ok=False,
            message=message,
            spoken="I ran into an error while handling that command.",
        )
        overlay.set_state("idle")
        await speak(result.spoken)
        return result

    if parsed is not None:
        print(f"Command: {_format_command(parsed)} [{parsed.source}]")
    print(result.message)
    overlay.set_state("speaking")
    overlay.set_transcript(transcript, result.spoken or "")
    if result.spoken:
        session.set_last_spoken(result.spoken)
    await speak(result.spoken)
    overlay.set_state("satisfied" if result.ok else "idle")
    return result


def record_wav_bytes(
    *,
    seconds: float = _DEFAULT_RECORD_SECONDS,
    sample_rate: int = _DEFAULT_SAMPLE_RATE,
    max_seconds: float = _DEFAULT_MAX_RECORD_SECONDS,
    silence_seconds: float = _DEFAULT_SILENCE_SECONDS,
    silence_rms: int = _DEFAULT_SILENCE_RMS,
) -> bytes:
    """Record from the default microphone and return WAV bytes.

    Stops early when the mic has been silent for ``silence_seconds``.
    Never records longer than ``max_seconds`` regardless of input.
    The legacy ``seconds`` parameter is kept for callers that pass it
    explicitly, but is no longer used as the fixed clip length.
    """
    import numpy as np
    import sounddevice as sd

    chunk = int(sample_rate * 0.1)   # 100 ms per chunk
    max_chunks = int(max_seconds / 0.1)
    silent_chunks_needed = int(silence_seconds / 0.1)

    recorded: list[np.ndarray] = []
    silent_count = 0
    speech_started = False

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16") as stream:
        for _ in range(max_chunks):
            data, _ = stream.read(chunk)
            recorded.append(data.copy())
            rms = int(np.sqrt(np.mean(data.astype(np.float32) ** 2)))
            if rms >= silence_rms:
                speech_started = True
                silent_count = 0
            elif speech_started:
                silent_count += 1
                if silent_count >= silent_chunks_needed:
                    break

    audio = np.concatenate(recorded, axis=0)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(np.dtype("int16").itemsize)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())
    return buffer.getvalue()
