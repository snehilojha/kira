"""Webcam session manager for Kira.

Keeps the camera open across multiple queries in a session.
Camera only opens on user intent and closes on explicit close command.

Public API
----------
open_session()          — open webcam + launch preview window
close_session()         — close webcam + destroy preview
is_open() -> bool       — check if session is active
query(question) -> str  — grab frame, ask vision model, return answer
"""

from __future__ import annotations

import asyncio
import base64
import logging
import threading
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

# ── Module-level session state ────────────────────────────────────

_lock = threading.Lock()
_cap = None          # cv2.VideoCapture instance
_preview = None      # WebcamPreview widget (created on Qt thread)
_session_open = False


def is_open() -> bool:
    return _session_open


def open_session(camera_index: int = 0) -> bool:
    """Open the webcam and launch the preview window.

    Returns True on success, False if camera unavailable.
    """
    global _cap, _session_open

    with _lock:
        if _session_open:
            return True

        try:
            import cv2
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                logger.warning("Could not open webcam (index=%d)", camera_index)
                return False

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)

            _cap = cap
            _session_open = True
            logger.info("Webcam session opened")
        except Exception as exc:
            logger.error("Failed to open webcam: %s", exc)
            return False

    _launch_preview()
    return True


def close_session() -> None:
    """Close the webcam and destroy the preview window."""
    global _cap, _session_open

    _destroy_preview()

    with _lock:
        if _cap is not None:
            try:
                _cap.release()
            except Exception:
                pass
            _cap = None
        _session_open = False
        logger.info("Webcam session closed")


def capture_frame_b64() -> str | None:
    """Grab the current webcam frame and return it as base64 JPEG.

    Returns None if the session is not open or capture fails.
    """
    with _lock:
        if not _session_open or _cap is None:
            return None
        try:
            import cv2
            ret, frame = _cap.read()
            if not ret or frame is None:
                return None
            # Encode as JPEG at 85% quality — good balance of size vs detail
            ret2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret2:
                return None
            return base64.b64encode(buf.tobytes()).decode("ascii")
        except Exception as exc:
            logger.warning("Frame capture failed: %s", exc)
            return None


async def query(question: str) -> str:
    """Grab the current frame and ask the vision model the question.

    Args:
        question: Natural language question about what the camera sees.

    Returns:
        Vision model's plain-text answer.
    """
    image_b64 = await asyncio.to_thread(capture_frame_b64)

    if not image_b64:
        return "I couldn't grab a frame from the camera. Is the webcam connected?"

    from bot import identity as _identity
    user_name = _identity.get_user_name()
    user_facts = _identity.get_all_facts()

    identity_context = ""
    if user_name or user_facts:
        parts = []
        if user_name:
            parts.append(f"The person in front of the camera is {user_name}, the user.")
        if user_facts:
            parts.append("What you know about them: " + " ".join(
                f"{f}." if not f.endswith(".") else f for f in user_facts[:4]
            ))
        identity_context = " ".join(parts) + " "

    base_question = question.strip() or (
        "Describe what you see in this image in 2-3 sentences. "
        "Focus on the main subject and any notable details."
    )
    prompt = identity_context + base_question

    try:
        from bot import provider
        response = await provider.create_vision_completion(
            prompt=prompt,
            image_b64=image_b64,
            max_tokens=300,
            image_format="jpeg",
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning("Webcam vision query failed: %s", exc)
        return f"Vision query failed: {exc}"


# ── Preview window management (Qt thread) ────────────────────────

def _launch_preview() -> None:
    """Post preview creation to the Qt thread."""
    try:
        from PyQt6.QtCore import QMetaObject, Qt
        from bot import overlay as _overlay_mod

        if _overlay_mod._window is None:
            return

        QMetaObject.invokeMethod(
            _overlay_mod._window,
            "_launch_webcam_preview",
            Qt.ConnectionType.QueuedConnection,
        )
    except Exception as exc:
        logger.debug("Could not launch preview on Qt thread: %s", exc)


def _destroy_preview() -> None:
    """Post preview destruction to the Qt thread."""
    try:
        from PyQt6.QtCore import QMetaObject, Qt
        from bot import overlay as _overlay_mod

        if _overlay_mod._window is None:
            return

        QMetaObject.invokeMethod(
            _overlay_mod._window,
            "_destroy_webcam_preview",
            Qt.ConnectionType.QueuedConnection,
        )
    except Exception as exc:
        logger.debug("Could not destroy preview on Qt thread: %s", exc)
