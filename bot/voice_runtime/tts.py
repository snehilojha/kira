"""Text-to-speech playback and the pre-rendered activation cues."""

from __future__ import annotations

import asyncio
import logging
import random

from bot import overlay
from bot import voice
from bot import voice_playback

logger = logging.getLogger(__name__)

# Pre-rendered WAV bytes for the acknowledgement cues — synthesized once at startup.
_CUE_PHRASES = ["Sir.", "Yes, sir.", "Mm?"]
_cue_wavs: list[bytes] = []


async def _prerender_cues() -> None:
    global _cue_wavs
    try:
        rendered: list[bytes] = []
        for phrase in _CUE_PHRASES:
            audio_bytes, fmt = await voice.synthesise(phrase)
            if fmt != "wav":
                audio_bytes = await asyncio.to_thread(voice.mp3_to_wav_bytes, audio_bytes)
            rendered.append(audio_bytes)
        _cue_wavs = rendered
        logger.info("Pre-rendered %d activation cues", len(_cue_wavs))
    except Exception as exc:
        logger.warning("Could not pre-render acknowledgement cues: %s", exc)


async def _activation_cue() -> None:
    """Play a random acknowledgement cue + flash orb on wake word or hotkey."""
    overlay.set_state("listening")
    if _cue_wavs:
        await asyncio.to_thread(voice_playback.play_wav_bytes, random.choice(_cue_wavs))


async def speak(text: str) -> None:
    """Speak a short local response, falling back to console only on failure."""
    try:
        if voice._elevenlabs_api_key():
            audio_bytes, fmt = await voice.synthesise(text)
            if fmt != "wav":
                audio_bytes = await asyncio.to_thread(voice.mp3_to_wav_bytes, audio_bytes)
            await asyncio.to_thread(voice_playback.play_wav_bytes, audio_bytes)
        else:
            import queue as _queue
            pcm_queue: _queue.Queue = _queue.Queue()

            async def _feed() -> None:
                # finally guarantees the sentinel even when synthesis fails —
                # otherwise play_pcm_stream's feeder blocks on the queue forever
                try:
                    async for chunk in voice.synthesise_stream(text):
                        pcm_queue.put(chunk)
                finally:
                    pcm_queue.put(None)

            await asyncio.gather(
                _feed(),
                asyncio.to_thread(voice_playback.play_pcm_stream, pcm_queue),
            )
    except Exception as exc:
        logger.warning("Local TTS playback failed: %s", exc)
        overlay.set_transcript("", f"[TTS failed] {text}")
        overlay.show()
    finally:
        from bot import mode as _mode
        _mode.mark_user_active()
