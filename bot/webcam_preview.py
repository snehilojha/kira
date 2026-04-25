"""Live webcam preview window for Kira.

Small always-on-top PyQt6 widget (320×240) that pulls cv2 frames at 15 fps
via a QTimer on the Qt thread. Draggable. Close button calls webcam.close_session().

Not imported at module level — created on demand by overlay._launch_webcam_preview.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class WebcamPreview:
    """Factory — call WebcamPreview.create() on the Qt thread."""

    @staticmethod
    def create() -> "WebcamPreviewWidget | None":
        try:
            from PyQt6.QtWidgets import QWidget
            widget = WebcamPreviewWidget()
            widget.show()
            return widget
        except Exception as exc:
            logger.warning("Could not create WebcamPreview: %s", exc)
            return None


class WebcamPreviewWidget:  # not a QWidget at class level so it's safe to import anywhere
    """Actual Qt widget — only instantiated on the Qt thread."""

    def __init__(self):
        from PyQt6.QtCore import Qt, QTimer
        from PyQt6.QtGui import QImage, QPixmap, QColor, QPainter, QFont
        from PyQt6.QtWidgets import (
            QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        )

        self._w = QWidget()
        self._w.setWindowTitle("Kira — Camera")
        self._w.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self._w.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self._w.setFixedSize(328, 268)
        self._w.setStyleSheet("background:#0a0612; border-radius:6px;")

        v = QVBoxLayout(self._w)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(0)

        # Title bar
        title_bar = QWidget()
        title_bar.setFixedHeight(24)
        title_bar.setStyleSheet("background:transparent;")
        h = QHBoxLayout(title_bar)
        h.setContentsMargins(6, 0, 4, 0)

        title_lbl = QLabel("KIRA — CAMERA")
        title_lbl.setStyleSheet(
            "color:rgba(160,120,255,140);font-size:7px;letter-spacing:3px;"
        )
        h.addWidget(title_lbl)
        h.addStretch()

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(16, 16)
        close_btn.setStyleSheet(
            "QPushButton{background:rgba(200,80,80,140);color:white;"
            "border-radius:8px;font-size:9px;padding:0;}"
            "QPushButton:hover{background:rgba(220,80,80,200);}"
        )
        close_btn.clicked.connect(self._on_close)
        h.addWidget(close_btn)
        v.addWidget(title_bar)

        # Video label
        self._video = QLabel()
        self._video.setFixedSize(320, 240)
        self._video.setStyleSheet("background:#000;border-radius:3px;")
        self._video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self._video)

        # Drag support
        self._drag_pos = None

        def _press(event):
            from PyQt6.QtCore import Qt as _Qt
            if event.button() == _Qt.MouseButton.LeftButton:
                self._drag_pos = event.globalPosition().toPoint() - self._w.frameGeometry().topLeft()

        def _move(event):
            from PyQt6.QtCore import Qt as _Qt
            if self._drag_pos is not None and event.buttons() & _Qt.MouseButton.LeftButton:
                self._w.move(event.globalPosition().toPoint() - self._drag_pos)

        def _release(event):
            self._drag_pos = None

        title_bar.mousePressEvent   = _press
        title_bar.mouseMoveEvent    = _move
        title_bar.mouseReleaseEvent = _release

        # Place bottom-left corner above taskbar
        from PyQt6.QtWidgets import QApplication
        sg = QApplication.primaryScreen().availableGeometry()
        self._w.move(sg.left() + 16, sg.bottom() - 268 - 16)

        # Frame timer — 15 fps
        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(1000 // 15)

    def show(self):
        self._w.show()

    def close(self):
        self._timer.stop()
        self._w.close()

    def _tick(self):
        try:
            import cv2
            from PyQt6.QtGui import QImage, QPixmap

            frame = _get_raw_frame()
            if frame is None:
                return
            frame = cv2.resize(frame, (320, 240))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            img = QImage(rgb.data, w, h, w * ch, QImage.Format.Format_RGB888)
            self._video.setPixmap(QPixmap.fromImage(img))
        except Exception as exc:
            logger.debug("Preview tick error: %s", exc)

    def _on_close(self):
        try:
            from bot import webcam as _webcam
            _webcam.close_session()
        except Exception:
            pass
        self.close()


def _get_raw_frame():
    """Grab a raw cv2 frame from the open session without encoding."""
    try:
        from bot import webcam as _webcam
        import cv2

        with _webcam._lock:
            if not _webcam._session_open or _webcam._cap is None:
                return None
            ret, frame = _webcam._cap.read()
            return frame if ret and frame is not None else None
    except Exception:
        return None
