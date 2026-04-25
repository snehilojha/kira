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
    Callers should use play_audio() which handles both wav and mp3.
    """
    fmt = response_format or _tts_format()
    response = await provider.synthesise_speech(
        text=_prepare_text(text),
        voice=_get_voice_name(),
        response_format=fmt,
    )
    audio_bytes = response.read()
    logger.info("TTS synthesised %d bytes for %d chars (fmt=%s)", len(audio_bytes), len(text), fmt)
    return audio_bytes, fmt


def mp3_to_wav_bytes(mp3_bytes: bytes) -> bytes:
    """Decode mp3 bytes to WAV bytes using ffmpeg subprocess."""
    import subprocess
    import io as _io
    result = subprocess.run(
        ["ffmpeg", "-y", "-f", "mp3", "-i", "pipe:0", "-f", "wav", "pipe:1"],
        input=mp3_bytes,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg mp3→wav failed: {result.stderr.decode()[-300:]}")
    return result.stdout
