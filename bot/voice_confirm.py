"""Voice confirmation registry and TTS send helpers.

Separated from handlers so notifier can import these without pulling in the
full handlers dependency graph.
"""

from __future__ import annotations

import asyncio
import io
import logging

logger = logging.getLogger(__name__)

# Pending voice confirmations keyed by callback token.
_PENDING_VOICE_CONFIRMS: dict[str, asyncio.Future[bool]] = {}


def register_voice_confirm(token: str, future: "asyncio.Future[bool]") -> None:
    """Register a voice confirmation future so the callback handler can resolve it."""
    _PENDING_VOICE_CONFIRMS[token] = future


async def _safe_synthesise(text: str) -> bytes | None:
    """Call voice.synthesise and return None on failure instead of raising.

    TTS failure should never crash the handler — Kira falls back to text only.
    """
    from bot import voice
    try:
        return await voice.synthesise(text)
    except Exception as exc:
        logger.warning("TTS synthesis failed (falling back to text): %s", exc)
        return None


async def _send_voice(msg, audio_bytes: bytes) -> None:
    """Send MP3 bytes as a Telegram voice message."""
    bio = io.BytesIO(audio_bytes)
    bio.name = "response.mp3"
    try:
        await msg.reply_voice(voice=bio)
    except Exception as exc:
        logger.warning("Failed to send voice reply: %s", exc)
