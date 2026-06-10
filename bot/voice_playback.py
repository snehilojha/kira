"""PCM and WAV playback for Kira's local voice runtime.

Two public functions:
- play_wav_bytes(audio_bytes)  — blocking, plays a full WAV buffer
- play_pcm_stream(pcm_queue)   — blocking, plays a Queue of raw PCM chunks

Both are intended to be called via asyncio.to_thread so the event loop stays free.
"""

from __future__ import annotations

import queue
import threading
import time

from bot import overlay
from bot import voice as voice_mod


# ── WAV playback ──────────────────────────────────────────────────────────────

def play_wav_bytes(audio_bytes: bytes) -> None:
    """Play WAV bytes through the default output device.

    Uses a sounddevice callback stream so audio runs on its own OS audio
    thread — the calling thread never blocks on audio I/O. A lightweight
    daemon thread pushes RMS amplitude to the orb every 50 ms in parallel.
    """
    import io
    import wave
    import numpy as np
    import sounddevice as sd

    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        sample_rate  = wav_file.getframerate()
        channels     = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frames       = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    raw      = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    audio_f32 = raw.reshape(-1, channels)
    mono      = raw.reshape(-1, channels).mean(axis=1)

    _MAX_RMS      = 0.18
    _AMP_INTERVAL = 0.05
    _amp_chunk    = int(sample_rate * _AMP_INTERVAL)

    n_chunks     = max(1, len(mono) // _amp_chunk)
    amp_timeline = []
    for i in range(n_chunks):
        sl  = mono[i * _amp_chunk : (i + 1) * _amp_chunk]
        rms = float(np.sqrt(np.mean(sl ** 2))) if len(sl) else 0.0
        amp_timeline.append(min(rms / _MAX_RMS, 1.0))

    cursor     = [0]
    done_event = threading.Event()

    def _callback(outdata, frames, time_info, status):
        start = cursor[0]
        end   = start + frames
        chunk = audio_f32[start:end]
        n     = len(chunk)
        outdata[:n]  = chunk
        if n < frames:
            outdata[n:] = 0
        cursor[0] = end
        if end >= len(audio_f32):
            done_event.set()
            raise sd.CallbackStop()

    def _push_amplitudes():
        for amp in amp_timeline:
            overlay.push_amplitude(amp)
            threading.Event().wait(_AMP_INTERVAL)
        overlay.push_amplitude(0.0)

    amp_thread = threading.Thread(target=_push_amplitudes, daemon=True)

    with sd.OutputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        callback=_callback,
        finished_callback=done_event.set,
    ):
        amp_thread.start()
        done_event.wait()


# ── PCM stream playback ───────────────────────────────────────────────────────

def play_pcm_stream(pcm_queue: queue.Queue) -> None:
    """Play a queue of raw PCM chunks (24kHz, mono, int16).

    Blocks until a None sentinel is received from the queue and all audio
    has been played. Intended to run via asyncio.to_thread alongside a
    coroutine that feeds the queue.

    Pre-buffers ~200 ms before starting playback to avoid underruns at the
    start of a response.
    """
    import numpy as np
    import sounddevice as sd

    SAMPLE_RATE   = voice_mod.PCM_SAMPLE_RATE   # 24000
    CHANNELS      = voice_mod.PCM_CHANNELS       # 1
    PRE_BUFFER_MS = 200
    _pre_buffer_bytes = int(SAMPLE_RATE * CHANNELS * 2 * PRE_BUFFER_MS / 1000)

    _MAX_RMS      = 0.18
    _AMP_INTERVAL = 0.05
    _amp_chunk    = int(SAMPLE_RATE * _AMP_INTERVAL)

    buf      = bytearray()
    buf_lock = threading.Lock()
    exhausted   = [False]
    done_event  = threading.Event()

    # ── feed thread: drain queue into buf ────────────────────────────────────
    def _feed_thread():
        while True:
            chunk = pcm_queue.get()
            if chunk is None:
                with buf_lock:
                    exhausted[0] = True
                return
            with buf_lock:
                buf.extend(chunk)

    feeder = threading.Thread(target=_feed_thread, daemon=True)
    feeder.start()

    # Pre-buffer: wait until we have enough data or the stream is exhausted.
    # Deadline guards against a dead feeder (TTS failure upstream) — without
    # it this loop spins forever inside an executor thread and blocks
    # interpreter shutdown.
    _deadline = time.monotonic() + 30.0
    while True:
        with buf_lock:
            ready = len(buf) >= _pre_buffer_bytes or exhausted[0]
        if ready:
            break
        if time.monotonic() > _deadline:
            return
        threading.Event().wait(0.01)

    # ── amplitude pusher ─────────────────────────────────────────────────────
    amp_buf  = bytearray()
    amp_lock = threading.Lock()

    def _push_amplitudes():
        pos = 0
        while not done_event.is_set():
            with amp_lock:
                chunk_bytes = bytes(amp_buf[pos : pos + _amp_chunk * 2])
            if len(chunk_bytes) < _amp_chunk * 2:
                threading.Event().wait(_AMP_INTERVAL)
                continue
            samples = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            rms     = float(np.sqrt(np.mean(samples ** 2)))
            overlay.push_amplitude(min(rms / _MAX_RMS, 1.0))
            pos += _amp_chunk * 2
        overlay.push_amplitude(0.0)

    amp_thread = threading.Thread(target=_push_amplitudes, daemon=True)
    amp_thread.start()

    # ── sounddevice callback ──────────────────────────────────────────────────
    cursor = [0]

    def _callback(outdata, frames, time_info, status):
        with buf_lock:
            start         = cursor[0]
            end           = start + frames * 2          # bytes (int16 = 2 bytes)
            available     = bytes(buf[start:end])
            buf_exhausted = exhausted[0]
            buf_len       = len(buf)

        n_bytes  = len(available)
        n_frames = n_bytes // 2
        samples  = np.frombuffer(available, dtype=np.int16).astype(np.float32) / 32768.0

        outdata[:n_frames, 0] = samples
        if n_frames < frames:
            outdata[n_frames:, 0] = 0.0

        with buf_lock:
            cursor[0] = start + n_bytes

        with amp_lock:
            amp_buf.extend(available)

        if n_frames < frames and buf_exhausted:
            done_event.set()
            raise sd.CallbackStop()
        if n_frames == frames and buf_exhausted and (start + n_bytes) >= buf_len:
            done_event.set()
            raise sd.CallbackStop()

    with sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=_callback,
        finished_callback=done_event.set,
    ):
        done_event.wait(timeout=60)

    feeder.join(timeout=5)
