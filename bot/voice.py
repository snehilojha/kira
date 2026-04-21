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
import tempfile
from pathlib import Path

from bot import provider

logger = logging.getLogger(__name__)


def _get_voice_name() -> str:
    """Return the configured TTS voice, defaulting to a low-latency option."""
    return os.getenv("KIRA_VOICE", "onyx")


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


async def synthesise(text: str, response_format: str = "mp3") -> bytes:
    """Convert text to speech bytes using the configured text-to-speech model."""
    response = await provider.synthesise_speech(
        text=text,
        voice=_get_voice_name(),
        response_format=response_format,
    )
    audio_bytes = response.read()
    logger.info("TTS synthesised %d bytes for %d chars", len(audio_bytes), len(text))
    return audio_bytes
