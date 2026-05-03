"""Presence detection and system lock guard for Kira.

State machine
-------------
ACTIVE      Normal operation. Inactivity timer ticking.
CHECKING    Idle threshold reached — webcam check in progress.
EMPTY       No one detected. Full mode active, input locked.
LOCKED      Someone unrecognised detected. Telegram alert sent, awaiting response.

Transitions
-----------
ACTIVE  → CHECKING   after IDLE_MINUTES of no keyboard/mouse
CHECKING → ACTIVE    face recognised as owner
CHECKING → EMPTY     no face detected
CHECKING → LOCKED    unknown face detected
EMPTY   → CHECKING   "kira wake up" spoken (presence.on_wake_phrase())
EMPTY   → ACTIVE     Telegram Allow pressed
LOCKED  → ACTIVE     Telegram Allow pressed
LOCKED  → (Windows lock screen)  Telegram Lock pressed
EMPTY/LOCKED → ACTIVE  any activity detected while in EMPTY (re-check)

Configuration (.env)
--------------------
KIRA_PRESENCE_IDLE_MINUTES=5       minutes of inactivity before first check
KIRA_PRESENCE_ENABLED=true         set false to disable entirely
KIRA_PRESENCE_FACE_DATA=data/face_embedding.npy   path to enrolled embedding

Public API
----------
start(speak_fn)         launch background task
stop()                  cancel task
is_locked() -> bool     True when input should be blocked
on_wake_phrase()        call when "kira wake up" is heard — triggers recheck
on_activity()           call on any keyboard/mouse event to reset idle timer
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from enum import Enum, auto
from pathlib import Path
from typing import Awaitable, Callable

import numpy as np

logger = logging.getLogger(__name__)

SpeakFn = Callable[[str], Awaitable[None]]

# ── Configuration defaults ────────────────────────────────────────
_DEFAULT_IDLE_MINUTES = 5
_DEFAULT_FACE_DATA    = "data/face_embedding.npy"
_RECHECK_INTERVAL_S   = 30   # seconds between rechecks while EMPTY
_CONFIRM_TIMEOUT_S    = 120  # seconds to wait for Telegram response

# ── Module state ─────────────────────────────────────────────────
class _State(Enum):
    ACTIVE   = auto()
    CHECKING = auto()
    EMPTY    = auto()
    LOCKED   = auto()

_state: _State = _State.ACTIVE
_last_activity: float = time.monotonic()
_task: asyncio.Task | None = None
_wake_phrase_event: asyncio.Event | None = None
_speak_fn: SpeakFn | None = None

# Pending Telegram futures keyed by token
_PENDING: dict[str, "asyncio.Future[str]"] = {}


def is_locked() -> bool:
    """True when Kira should block command execution."""
    return _state in (_State.EMPTY, _State.LOCKED)


def on_activity() -> None:
    """Reset idle timer. Call on any keyboard/mouse event."""
    global _last_activity
    _last_activity = time.monotonic()


def on_wake_phrase() -> None:
    """Trigger an immediate webcam recheck. Call when 'kira wake up' is heard."""
    if _wake_phrase_event is not None:
        _wake_phrase_event.set()


def register_presence_future(token: str, future: "asyncio.Future[str]") -> None:
    _PENDING[token] = future


# ── Webcam + face recognition ─────────────────────────────────────

def _load_enrolled() -> np.ndarray | None:
    path = Path(os.environ.get("KIRA_PRESENCE_FACE_DATA", _DEFAULT_FACE_DATA))
    if not path.exists():
        logger.warning("Presence: no enrolled face at %s — run bot/enroll_face.py", path)
        return None
    try:
        return np.load(str(path))
    except Exception as exc:
        logger.error("Presence: failed to load face embedding: %s", exc)
        return None


def _grab_frame() -> "np.ndarray | None":
    """Grab a single frame from the default webcam. Returns BGR ndarray or None."""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.warning("Presence: webcam not accessible")
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Discard first few frames — some cams need warm-up
        for _ in range(3):
            cap.read()
        ok, frame = cap.read()
        cap.release()
        return frame if ok else None
    except Exception as exc:
        logger.warning("Presence: webcam grab failed: %s", exc)
        return None


def _frame_to_png(frame: "np.ndarray") -> bytes:
    import cv2
    ok, buf = cv2.imencode(".png", frame)
    return bytes(buf) if ok else b""


class _CheckResult(Enum):
    OWNER    = auto()   # face recognised as the enrolled owner
    STRANGER = auto()   # face detected but not the owner
    EMPTY    = auto()   # no face detected at all
    ERROR    = auto()   # webcam/model failure


_face_app = None

def _get_face_app():
    global _face_app
    if _face_app is None:
        from insightface.app import FaceAnalysis
        _face_app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
        _face_app.prepare(ctx_id=0, det_size=(320, 320))
        logger.info("Presence: insightface model loaded")
    return _face_app


def _check_frame(frame: "np.ndarray", enrolled: np.ndarray) -> _CheckResult:
    """Run insightface on frame, compare against enrolled embedding."""
    try:
        app = _get_face_app()
        faces = app.get(frame)
        if not faces:
            return _CheckResult.EMPTY

        # Take the largest detected face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        embedding = face.normed_embedding

        similarity = float(np.dot(embedding, enrolled))
        logger.debug("Presence: face similarity=%.3f", similarity)

        # Threshold: 0.35 is conservative for cosine similarity on normed embeddings
        return _CheckResult.OWNER if similarity >= 0.35 else _CheckResult.STRANGER

    except Exception as exc:
        logger.warning("Presence: face check failed: %s", exc)
        return _CheckResult.ERROR


# ── Telegram alert ────────────────────────────────────────────────

async def _send_alert(frame: "np.ndarray") -> "asyncio.Future[str] | None":
    """Send photo + Allow/Lock buttons to Telegram. Returns a future resolved by callback."""
    try:
        import time as _time
        from bot import notifier

        token = f"presence_{int(_time.monotonic() * 1000)}"
        loop  = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        register_presence_future(token, future)

        keyboard = [[
            {"text": "Allow",      "callback_data": f"presence_allow_{token}"},
            {"text": "Lock Screen", "callback_data": f"presence_lock_{token}"},
        ]]

        png = await asyncio.to_thread(_frame_to_png, frame)
        msg_id = await notifier.send_photo_with_buttons(
            caption="Unrecognised person detected at your system. Allow or lock?",
            png_bytes=png,
            keyboard=keyboard,
        )
        if msg_id is None:
            future.cancel()
            return None
        return future

    except Exception as exc:
        logger.error("Presence: failed to send Telegram alert: %s", exc)
        return None


def _lock_windows_screen() -> None:
    try:
        import ctypes
        ctypes.windll.user32.LockWorkStation()
        logger.info("Presence: Windows lock screen triggered")
    except Exception as exc:
        logger.warning("Presence: could not lock screen: %s", exc)


# ── State machine ─────────────────────────────────────────────────

async def _do_check(enrolled: np.ndarray) -> tuple[_CheckResult, "np.ndarray | None"]:
    """Grab a frame and check it on a thread (blocking CV ops off event loop)."""
    frame = await asyncio.to_thread(_grab_frame)
    if frame is None:
        return _CheckResult.ERROR, None
    result = await asyncio.to_thread(_check_frame, frame, enrolled)
    return result, frame


async def _loop(speak_fn: SpeakFn) -> None:
    global _state, _wake_phrase_event

    enabled = os.environ.get("KIRA_PRESENCE_ENABLED", "true").strip().lower()
    if enabled in ("false", "0", "no"):
        logger.info("Presence detection disabled via KIRA_PRESENCE_ENABLED")
        return

    idle_minutes = int(os.environ.get("KIRA_PRESENCE_IDLE_MINUTES", _DEFAULT_IDLE_MINUTES))
    idle_seconds = idle_minutes * 60

    enrolled = await asyncio.to_thread(_load_enrolled)
    if enrolled is None:
        logger.warning("Presence: no enrolled face — detection disabled until enrollment")
        return

    _wake_phrase_event = asyncio.Event()

    logger.info("Presence loop started — idle threshold=%dm", idle_minutes)

    while True:
        # ── ACTIVE: wait for idle ─────────────────────────────────
        _state = _State.ACTIVE
        while True:
            await asyncio.sleep(10)
            from bot.mode import get_last_input_seconds
            idle = get_last_input_seconds()
            if idle >= idle_seconds:
                break

        # ── CHECKING ──────────────────────────────────────────────
        _state = _State.CHECKING
        logger.debug("Presence: idle for %.0fs — running webcam check", idle)

        result, frame = await _do_check(enrolled)

        if result == _CheckResult.OWNER:
            logger.debug("Presence: owner detected — staying active")
            on_activity()   # reset timer so we don't immediately re-check
            continue

        if result == _CheckResult.ERROR:
            logger.warning("Presence: check error — staying active")
            on_activity()
            continue

        if result == _CheckResult.EMPTY:
            # ── EMPTY: activate full mode, lock input ─────────────
            _state = _State.EMPTY
            logger.info("Presence: no one detected — activating full mode lock")
            try:
                from bot import ui_mode
                ui_mode.activate("presence: empty")
            except Exception:
                pass
            try:
                from bot import notifier
                await notifier.send("🔒 Autonomous mode activated — no one detected at the system.")
            except Exception:
                pass

            _wake_phrase_event.clear()

            while _state == _State.EMPTY:
                # Wait for wake phrase OR recheck interval
                try:
                    await asyncio.wait_for(
                        asyncio.shield(_wake_phrase_event.wait()),
                        timeout=_RECHECK_INTERVAL_S,
                    )
                    _wake_phrase_event.clear()
                except asyncio.TimeoutError:
                    pass

                recheck_result, recheck_frame = await _do_check(enrolled)

                if recheck_result == _CheckResult.OWNER:
                    logger.info("Presence: owner returned — unlocking")
                    _state = _State.ACTIVE
                    try:
                        from bot import ui_mode
                        ui_mode.deactivate("presence: owner returned")
                    except Exception:
                        pass
                    try:
                        from bot import notifier
                        await notifier.send("Autonomous mode deactivated — owner recognised.")
                    except Exception:
                        pass
                    await speak_fn("Welcome back.")
                    on_activity()
                    break

                elif recheck_result == _CheckResult.STRANGER:
                    # Someone unknown — escalate to LOCKED and wait for Telegram reply.
                    # No more webcam rechecks until a decision arrives.
                    _state = _State.LOCKED
                    logger.info("Presence: stranger detected — sending Telegram alert")

                    while _state == _State.LOCKED:
                        future = await _send_alert(recheck_frame)
                        if future is None:
                            logger.warning("Presence: alert send failed — retrying in 30s")
                            await asyncio.sleep(30)
                            continue

                        try:
                            decision = await asyncio.wait_for(future, timeout=_CONFIRM_TIMEOUT_S)
                        except asyncio.TimeoutError:
                            logger.info("Presence: Telegram alert timed out — resending")
                            _PENDING.pop(next((k for k, v in _PENDING.items() if v is future), ""), None)
                            continue  # re-send alert, still LOCKED, no webcam check

                        if decision == "allow":
                            logger.info("Presence: Telegram allow — unlocking")
                            _state = _State.ACTIVE
                            try:
                                from bot import ui_mode
                                ui_mode.deactivate("presence: telegram allow")
                            except Exception:
                                pass
                            on_activity()
                        else:
                            logger.info("Presence: Telegram lock — locking Windows screen")
                            await asyncio.to_thread(_lock_windows_screen)
                            _state = _State.ACTIVE
                            try:
                                from bot import ui_mode
                                ui_mode.deactivate("presence: screen locked")
                            except Exception:
                                pass
                            on_activity()

                    break  # exit the EMPTY while loop — state is now ACTIVE
                # EMPTY or ERROR — keep waiting
            continue

        if result == _CheckResult.STRANGER:
            # Stranger seen on initial check (while system was idle).
            # Stay LOCKED and keep re-sending alert until a decision arrives.
            _state = _State.LOCKED
            logger.info("Presence: stranger on initial check — sending Telegram alert")
            try:
                from bot import ui_mode
                ui_mode.activate("presence: stranger")
            except Exception:
                pass

            while _state == _State.LOCKED:
                future = await _send_alert(frame)
                if future is None:
                    logger.warning("Presence: alert send failed — retrying in 30s")
                    await asyncio.sleep(30)
                    continue

                try:
                    decision = await asyncio.wait_for(future, timeout=_CONFIRM_TIMEOUT_S)
                except asyncio.TimeoutError:
                    logger.info("Presence: alert timed out — resending")
                    _PENDING.pop(next((k for k, v in _PENDING.items() if v is future), ""), None)
                    continue  # re-send, no webcam check

                if decision == "allow":
                    logger.info("Presence: Telegram allow — unlocking")
                    try:
                        from bot import ui_mode
                        ui_mode.deactivate("presence: telegram allow")
                    except Exception:
                        pass
                else:
                    logger.info("Presence: Telegram lock — locking Windows screen")
                    await asyncio.to_thread(_lock_windows_screen)
                    try:
                        from bot import ui_mode
                        ui_mode.deactivate("presence: screen locked")
                    except Exception:
                        pass

                _state = _State.ACTIVE
                on_activity()


def start(speak_fn: SpeakFn) -> None:
    global _task, _speak_fn
    if _task is not None and not _task.done():
        return
    _speak_fn = speak_fn
    _task = asyncio.ensure_future(_loop(speak_fn))
    logger.info("Presence detection task started")


def stop() -> None:
    global _task
    if _task is not None and not _task.done():
        _task.cancel()
        _task = None
    logger.info("Presence detection task stopped")
