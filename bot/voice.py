"""Voice I/O for Kira.

Two responsibilities:
1. ``transcribe(ogg_bytes) -> str``
2. ``synthesise(text) -> bytes``

Primary provider: ElevenLabs (if ELEVENLABS_API_KEY is set).
Fallback: OpenAI gpt-4o-mini-tts / gpt-4o-transcribe.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path

from bot import provider

logger = logging.getLogger(__name__)

_EMOTION_TAG_RE = re.compile(r"<(?:laugh|chuckle|sigh|gasp|yawn|cough|sob)>", re.IGNORECASE)


def _supports_emotion_tags() -> bool:
    model = (os.getenv("KIRA_TTS_MODEL") or os.getenv("KIRA_VOICE_SYNTHESISE_MODEL", "")).lower()
    return "orpheus" in model


def _tts_format() -> str:
    model = (os.getenv("KIRA_TTS_MODEL") or os.getenv("KIRA_VOICE_SYNTHESISE_MODEL", "")).lower()
    if "orpheus" in model or os.getenv("KIRA_TTS_BASE_URL", ""):
        return "mp3"
    return "wav"


def _prepare_text(text: str) -> str:
    if _supports_emotion_tags():
        return text
    return _EMOTION_TAG_RE.sub("", text).strip()


def _get_voice_name() -> str:
    return os.getenv("KIRA_VOICE", "nova")


def _elevenlabs_api_key() -> str | None:
    return os.getenv("ELEVENLABS_API_KEY") or None


# ── ElevenLabs TTS ────────────────────────────────────────────────

async def _synthesise_elevenlabs(text: str) -> tuple[bytes, str]:
    import asyncio
    from elevenlabs.client import ElevenLabs

    api_key = _elevenlabs_api_key()
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")  # default: George
    model_id = os.getenv("ELEVENLABS_TTS_MODEL", "eleven_flash_v2_5")

    client = ElevenLabs(api_key=api_key)

    # ElevenLabs SDK is sync — run in executor to avoid blocking event loop
    def _call() -> bytes:
        audio_iter = client.text_to_speech.convert(
            voice_id=voice_id,
            text=_prepare_text(text),
            model_id=model_id,
            output_format="mp3_44100_128",
        )
        return b"".join(audio_iter)

    audio_bytes = await asyncio.get_event_loop().run_in_executor(None, _call)
    logger.info("ElevenLabs TTS synthesised %d bytes for %d chars", len(audio_bytes), len(text))
    return audio_bytes, "mp3"


# ── ElevenLabs STT ────────────────────────────────────────────────

async def _transcribe_elevenlabs(audio_bytes: bytes, suffix: str) -> str:
    import asyncio
    from elevenlabs.client import ElevenLabs

    api_key = _elevenlabs_api_key()
    model_id = os.getenv("ELEVENLABS_STT_MODEL", "scribe_v1")

    client = ElevenLabs(api_key=api_key)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = Path(tmp.name)

    try:
        def _call() -> str:
            with open(tmp_path, "rb") as f:
                result = client.speech_to_text.convert(
                    file=f,
                    model_id=model_id,
                    language_code="en",
                )
            return result.text.strip()

        transcript = await asyncio.get_event_loop().run_in_executor(None, _call)
        logger.info("ElevenLabs STT transcript: %r", transcript)
        return transcript
    finally:
        tmp_path.unlink(missing_ok=True)


# ── OpenAI fallback TTS ───────────────────────────────────────────

async def _synthesise_openai_fallback(text: str) -> tuple[bytes, str]:
    from openai import AsyncOpenAI
    fallback_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    fallback_model = os.environ.get("KIRA_TTS_FALLBACK_MODEL", "gpt-4o-mini-tts")
    fallback_voice = os.environ.get("KIRA_TTS_FALLBACK_VOICE", "nova")
    fallback_instructions = os.environ.get(
        "KIRA_TTS_FALLBACK_INSTRUCTIONS",
        "Speak as Kira, a calm, intelligent, and warm personal AI assistant. "
        "Natural pace, slightly warm tone. Never robotic or overly chipper.",
    )
    kwargs: dict = dict(
        model=fallback_model,
        voice=fallback_voice,
        input=_prepare_text(text),
        response_format="mp3",
    )
    if "mini-tts" in fallback_model or "4o" in fallback_model:
        kwargs["instructions"] = fallback_instructions
    response = await fallback_client.audio.speech.create(**kwargs)
    audio_bytes = response.read()
    logger.info("OpenAI fallback TTS synthesised %d bytes", len(audio_bytes))
    return audio_bytes, "mp3"


# ── Public API ────────────────────────────────────────────────────

async def transcribe(audio_bytes: bytes, suffix: str = ".ogg") -> str:
    """Transcribe audio. Uses ElevenLabs Scribe if configured, falls back to OpenAI."""
    if _elevenlabs_api_key():
        try:
            return await _transcribe_elevenlabs(audio_bytes, suffix)
        except Exception as exc:
            logger.warning("ElevenLabs STT failed (%s) — falling back to OpenAI", exc)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = Path(tmp.name)
    try:
        with open(tmp_path, "rb") as audio_file:
            response = await provider.transcribe_audio(file=audio_file, language="en")
        transcript = response.text.strip()
        logger.info("OpenAI STT transcript: %r", transcript)
        return transcript
    finally:
        tmp_path.unlink(missing_ok=True)


async def synthesise(text: str, response_format: str | None = None) -> tuple[bytes, str]:
    """Convert text to speech. Uses ElevenLabs if configured, falls back to OpenAI."""
    if _elevenlabs_api_key():
        try:
            return await _synthesise_elevenlabs(text)
        except Exception as exc:
            logger.warning("ElevenLabs TTS failed (%s) — falling back to OpenAI", exc)

    fmt = response_format or _tts_format()
    try:
        response = await provider.synthesise_speech(
            text=_prepare_text(text),
            voice=_get_voice_name(),
            response_format=fmt,
        )
        audio_bytes = response.read()
        logger.info("OpenAI TTS synthesised %d bytes for %d chars (fmt=%s)", len(audio_bytes), len(text), fmt)
        return audio_bytes, fmt
    except Exception as primary_exc:
        logger.warning("OpenAI TTS failed (%s) — trying fallback model", primary_exc)
        try:
            audio_bytes, fmt = await _synthesise_openai_fallback(text)
            return audio_bytes, fmt
        except Exception as fallback_exc:
            logger.error("All TTS providers failed: %s", fallback_exc)
            raise primary_exc


def mp3_to_wav_bytes(mp3_bytes: bytes) -> bytes:
    """Decode mp3 bytes to WAV bytes using ffmpeg subprocess."""
    import shutil
    import subprocess
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH — cannot convert MP3 to WAV")
    result = subprocess.run(
        [ffmpeg, "-y", "-f", "mp3", "-i", "pipe:0", "-f", "wav", "pipe:1"],
        input=mp3_bytes,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg mp3→wav failed: {result.stderr.decode()[-300:]}")
    return result.stdout
