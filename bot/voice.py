"""Voice I/O for Kira.

Two responsibilities:
1. ``transcribe(ogg_bytes) -> str``
2. ``synthesise(text) -> bytes``

The actual client and model selection are delegated to ``bot.provider`` so
voice follows the same provider configuration as the rest of the bot.
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
    """Return True if the active TTS model understands Orpheus emotion tags."""
    model = (os.getenv("KIRA_TTS_MODEL") or os.getenv("KIRA_VOICE_SYNTHESISE_MODEL", "")).lower()
    return "orpheus" in model

def _tts_format() -> str:
    """Return the best response_format for the active TTS model.

    Orpheus (and some OpenRouter models) only accept mp3 or pcm.
    OpenAI tts-1/tts-1-hd accept wav directly.
    """
    model = (os.getenv("KIRA_TTS_MODEL") or os.getenv("KIRA_VOICE_SYNTHESISE_MODEL", "")).lower()
    if "orpheus" in model or os.getenv("KIRA_TTS_BASE_URL", ""):
        return "mp3"
    return "wav"

def _prepare_text(text: str) -> str:
    """Strip emotion tags if the current TTS model doesn't support them."""
    if _supports_emotion_tags():
        return text
    return _EMOTION_TAG_RE.sub("", text).strip()


def _get_voice_name() -> str:
    """Return the configured TTS voice, defaulting to a low-latency option."""
    return os.getenv("KIRA_VOICE", "nova")


async def transcribe(audio_bytes: bytes, suffix: str = ".ogg") -> str:
    """Send raw audio bytes to the configured speech-to-text model."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = Path(tmp.name)

    try:
        with open(tmp_path, "rb") as audio_file:
            response = await provider.transcribe_audio(file=audio_file, language="en")
        transcript = response.text.strip()
        logger.info("Whisper transcript: %r", transcript)
        return transcript
    finally:
        tmp_path.unlink(missing_ok=True)


async def synthesise(text: str, response_format: str | None = None) -> tuple[bytes, str]:
    """Convert text to speech and return (audio_bytes, format).

    Format is chosen automatically based on the active TTS model.
    Falls back to OpenAI gpt-4o-mini-tts if the primary provider fails.
    """
    fmt = response_format or _tts_format()
    try:
        response = await provider.synthesise_speech(
            text=_prepare_text(text),
            voice=_get_voice_name(),
            response_format=fmt,
        )
        audio_bytes = response.read()
        logger.info("TTS synthesised %d bytes for %d chars (fmt=%s)", len(audio_bytes), len(text), fmt)
        return audio_bytes, fmt
    except Exception as primary_exc:
        logger.warning("Primary TTS failed (%s) — falling back to fallback TTS", primary_exc)
        try:
            from openai import AsyncOpenAI
            fallback_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            fallback_model = os.environ.get("KIRA_TTS_FALLBACK_MODEL", "gpt-4o-mini-tts")
            fallback_voice = os.environ.get("KIRA_TTS_FALLBACK_VOICE", "nova")
            fallback_instructions = os.environ.get(
                "KIRA_TTS_FALLBACK_INSTRUCTIONS",
                "Speak as Kira, a calm, intelligent, and warm personal AI assistant. "
                "Natural pace, slightly warm tone. Never robotic or overly chipper.",
            )
            fallback_kwargs: dict = dict(
                model=fallback_model,
                voice=fallback_voice,
                input=_prepare_text(text),
                response_format="mp3",
            )
            if "mini-tts" in fallback_model or "4o" in fallback_model:
                fallback_kwargs["instructions"] = fallback_instructions
            fallback_response = await fallback_client.audio.speech.create(**fallback_kwargs)
            audio_bytes = fallback_response.read()
            logger.info("Fallback TTS synthesised %d bytes", len(audio_bytes))
            return audio_bytes, "mp3"
        except Exception as fallback_exc:
            logger.error("Fallback TTS also failed: %s", fallback_exc)
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
