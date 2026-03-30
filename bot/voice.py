"""bot/voice.py — Voice I/O for Kira.

Two responsibilities:
  1. transcribe(ogg_bytes) → str
     Sends raw Telegram voice bytes to OpenAI Whisper and returns transcript.

  2. synthesise(text) → bytes
     Converts a text string to MP3 audio via OpenAI TTS (tts-1).

Design decisions:
  - Uses the same AsyncOpenAI client and API key as the rest of the bot.
    No new dependencies — openai>=1.0.0 already covers Whisper and TTS.
  - TTS model is "tts-1" not "tts-1-hd". tts-1-hd sounds slightly better
    but has ~2x latency. For a real-time assistant feel, latency wins.
  - Voice is "alloy" by default. Override with KIRA_VOICE in .env.
    Options: alloy | echo | fable | onyx | nova | shimmer
  - All temp files are written to the system temp dir and deleted immediately
    after use regardless of success or failure. Audio never sits on disk.
  - Whisper accepts .ogg (Opus) natively — no conversion needed. Telegram
    voice messages are always Opus-encoded .ogg files.
  - We don't cache the AsyncOpenAI client at module level because .env may
    not be loaded yet at import time. Client is created lazily per call.
"""

import logging
import os
import tempfile
from pathlib import Path

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# TTS voice character. Override via KIRA_VOICE env var.
# Options: alloy | echo | fable | onyx | nova | shimmer
_VOICE = os.getenv("KIRA_VOICE", "onyx")


def _get_client() -> AsyncOpenAI:
    """Return an AsyncOpenAI client from the environment key."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in .env")
    return AsyncOpenAI(api_key=api_key)


async def transcribe(ogg_bytes: bytes) -> str:
    """Send raw .ogg bytes to Whisper and return the transcript.

    Args:
        ogg_bytes: Raw bytes of a Telegram voice message (.ogg / Opus).

    Returns:
        Transcribed text string. Empty string if Whisper returns nothing.

    Raises:
        RuntimeError: If OPENAI_API_KEY is not set.
        openai.OpenAIError: On API errors.
    """
    client = _get_client()

    # Write to a named temp file — the OpenAI SDK needs a file-like object
    # with a .name that includes the extension to determine the audio format.
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp.write(ogg_bytes)
        tmp_path = Path(tmp.name)

    try:
        with open(tmp_path, "rb") as audio_file:
            response = await client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en",  # Explicit language = faster + more accurate.
                                # Remove this line for auto-detection.
            )
        transcript = response.text.strip()
        logger.info("Whisper transcript: %r", transcript)
        return transcript
    finally:
        # Always clean up — even if the API call throws.
        tmp_path.unlink(missing_ok=True)


async def synthesise(text: str) -> bytes:
    """Convert text to MP3 bytes via OpenAI TTS.

    Args:
        text: The text Kira should speak.

    Returns:
        Raw MP3 bytes ready to send as a Telegram voice message.

    Raises:
        RuntimeError: If OPENAI_API_KEY is not set.
        openai.OpenAIError: On API errors.
    """
    client = _get_client()

    response = await client.audio.speech.create(
        model="tts-1",
        voice=_VOICE,
        input=text,
        response_format="mp3",
    )

    audio_bytes = response.read()
    logger.info(
        "TTS synthesised %d bytes for %d chars", len(audio_bytes), len(text)
    )
    return audio_bytes
