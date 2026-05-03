"""Wake-word detection for Kira using openwakeword.

Runs a background daemon thread that continuously streams mic audio through
the openwakeword model. When the configured wake word is detected above the
confidence threshold it puts a token into an asyncio.Queue so the voice loop
can pick it up — same interface as the hotkey trigger.

Usage
-----
    queue: asyncio.Queue[None] = asyncio.Queue(maxsize=1)
    thread = start(queue, loop)
    # wait on queue.get() in the voice loop
    thread.stop()

Config (.env)
-------------
    KIRA_WAKE_WORD=hey_jarvis          # openwakeword model name (default)
    KIRA_WAKE_WORD_THRESHOLD=0.5       # detection confidence threshold (0-1)
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncio

logger = logging.getLogger(__name__)

_CHUNK_SAMPLES = 1280   # openwakeword expects 80ms chunks at 16kHz (1280 samples)
_SAMPLE_RATE = 16000
_DEFAULT_MODEL = "hey_jarvis"
_DEFAULT_THRESHOLD = 0.5


class WakeWordDetector:
    """Background thread that fires an asyncio queue entry on wake word."""

    def __init__(
        self,
        trigger_queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
        *,
        model_name: str = _DEFAULT_MODEL,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        self._queue = trigger_queue
        self._loop = loop
        self._model_name = model_name
        self._threshold = threshold
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="wake-word")

    def start(self) -> None:
        self._thread.start()
        logger.info("Wake word detector started (model=%s, threshold=%.2f)", self._model_name, self._threshold)

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=3)
        logger.info("Wake word detector stopped")

    def _run(self) -> None:
        try:
            import numpy as np
            import sounddevice as sd
            from openwakeword.model import Model

            model = Model(
                wakeword_models=[self._model_name],
                inference_framework="onnx",
            )
        except Exception as exc:
            logger.error("Wake word detector failed to initialise: %s", exc)
            return

        logger.info("Listening for wake word '%s'...", self._model_name)
        cooldown_until = 0.0

        try:
            with sd.InputStream(
                samplerate=_SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=_CHUNK_SAMPLES,
            ) as stream:
                while not self._stop_event.is_set():
                    data, _ = stream.read(_CHUNK_SAMPLES)
                    audio = data[:, 0] if data.ndim > 1 else data.flatten()

                    prediction = model.predict(audio)
                    score = prediction.get(self._model_name, 0.0)

                    now = time.monotonic()
                    if score >= self._threshold and now >= cooldown_until:
                        logger.info("Wake word detected (score=%.3f)", score)
                        cooldown_until = now + 3.0  # 3s cooldown to avoid double-firing
                        self._loop.call_soon_threadsafe(self._fire)
                        model.reset()  # clear model state after detection

        except Exception as exc:
            if not self._stop_event.is_set():
                logger.error("Wake word stream error: %s", exc)

    def _fire(self) -> None:
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass  # queue full = already pending, drop


def start(
    trigger_queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    *,
    model_name: str = _DEFAULT_MODEL,
    threshold: float = _DEFAULT_THRESHOLD,
) -> WakeWordDetector:
    """Create and start a wake word detector, returning it for later cleanup."""
    detector = WakeWordDetector(
        trigger_queue,
        loop,
        model_name=model_name,
        threshold=threshold,
    )
    detector.start()
    return detector
