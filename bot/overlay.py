"""Kira HUD overlay — PyQt6 particle-sphere orb.

Two modes
---------
Compact  Small orb (200 px) pinned to bottom-right corner.
         Hidden by default, fades in on wake word, fades out after response.

Full     Full-screen centered orb (440 px) + collapsible side panels.
         Activated by voice ('take over') or ui_mode.activate().
         Stays until 'stand down' / ui_mode.deactivate().

Thread safety
-------------
All Qt calls happen on the dedicated Qt thread.
External callers use the module-level helpers (show, hide, set_state, …)
which post work via QMetaObject.invokeMethod so they're safe from any thread.

Public API
----------
start()             — launch Qt thread (call once from main.py)
show()              — fade in compact orb
hide()              — fade out compact orb
set_state(state)    — 'idle'|'listening'|'thinking'|'speaking'|'autonomous'
set_transcript(you, kira)  — update right-panel transcript text
set_full_mode(bool) — switch between compact and full mode
stop()              — shut down Qt app
"""

from __future__ import annotations

import logging
import math
import os
import sys
import threading
import time
from typing import Literal

logger = logging.getLogger(__name__)

# ── Qt imports — lazy so the module loads even if PyQt6 missing ───
_qt_available: bool = False
try:
    from PyQt6.QtCore import (
        QMetaObject, QPropertyAnimation, Qt, QTimer, pyqtSlot,
        Q_ARG, QEasingCurve, QThread,
    )
    from PyQt6.QtGui import (
        QColor, QFont, QFontMetrics, QPainter, QPen,
        QLinearGradient, QRadialGradient,
    )
    from PyQt6.QtWidgets import (
        QApplication, QFrame, QGraphicsOpacityEffect,
        QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget,
    )
    from bot.overlay_renderer import OrbRenderer
    _qt_available = True
except Exception as exc:
    logger.warning("PyQt6 not available — overlay disabled: %s", exc)

OrbState = Literal["idle", "listening", "thinking", "speaking", "autonomous"]

# ── Module-level state ────────────────────────────────────────────
_app: "QApplication | None"    = None
_window: "_KiraOverlay | None" = None
_thread: "threading.Thread | None" = None


# ═══════════════════════════════════════════════════════════════════
# Widget: side-panel card
# ═══════════════════════════════════════════════════════════════════

class _Card(QFrame):
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setStyleSheet("""
            QFrame#card {
                background: rgba(5,3,12,230);
                border: 1px solid rgba(140,100,255,18);
                border-radius: 4px;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)

        lbl = QLabel(label.upper())
        lbl.setStyleSheet("color: rgba(160,120,255,70); font-size:7px; letter-spacing:4px;")
        layout.addWidget(lbl)

        self._body_layout = layout

    def body(self) -> QVBoxLayout:
        return self._body_layout


class _StatRow(QWidget):
    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        self._name  = QLabel(name)
        self._value = QLabel("—")
        self._name.setStyleSheet("color:rgba(150,120,200,115);font-size:9px;")
        self._value.setStyleSheet("color:rgba(200,170,255,230);font-size:11px;font-weight:600;")
        h.addWidget(self._name)
        h.addStretch()
        h.addWidget(self._value)

    def set_value(self, v: str) -> None:
        self._value.setText(v)


# ═══════════════════════════════════════════════════════════════════
# Left panel
# ═══════════════════════════════════════════════════════════════════

class _LeftPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(230)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)

        # ── Clock card
        clock_card = _Card("", self)
        self._time_lbl = QLabel("00:00:00")
        self._time_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._time_lbl.setStyleSheet(
            "color:rgba(220,200,255,230);font-size:20px;font-weight:200;letter-spacing:3px;"
        )
        self._date_lbl = QLabel("")
        self._date_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._date_lbl.setStyleSheet("color:rgba(160,130,210,120);font-size:8px;letter-spacing:2px;")
        clock_card.body().addWidget(self._time_lbl)
        clock_card.body().addWidget(self._date_lbl)
        v.addWidget(clock_card)

        # ── System card
        sys_card = _Card("System", self)
        self._cpu  = _StatRow("CPU")
        self._ram  = _StatRow("RAM")
        self._disk = _StatRow("Disk C:")
        self._gpu  = _StatRow("GPU")
        self._bat  = _StatRow("Battery")
        self._mode = _StatRow("Mode")
        self._up   = _StatRow("Uptime")
        for row in (self._cpu, self._ram, self._disk, self._gpu, self._bat, self._mode, self._up):
            sys_card.body().addWidget(row)
        v.addWidget(sys_card)

        # ── Top CPU hogs card
        hogs_card = _Card("Top Processes", self)
        self._hog_labels = []
        for _ in range(4):
            lbl = QLabel("")
            lbl.setStyleSheet("color:rgba(180,150,230,160);font-size:9px;font-family:monospace;")
            lbl.setWordWrap(False)
            hogs_card.body().addWidget(lbl)
            self._hog_labels.append(lbl)
        v.addWidget(hogs_card)
        v.addStretch()

        self._start = time.monotonic()

        # Clock ticks every second
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._tick_clock)
        self._clock_timer.start(1000)
        self._tick_clock()

        # System stats refresh every 3 s
        self._sys_timer = QTimer(self)
        self._sys_timer.timeout.connect(self._refresh_sys)
        self._sys_timer.start(3000)
        self._refresh_sys()

    def _tick_clock(self) -> None:
        import datetime
        now = datetime.datetime.now()
        self._time_lbl.setText(now.strftime("%H:%M:%S"))
        self._date_lbl.setText(now.strftime("%A, %d %B %Y").upper())

    def _refresh_sys(self) -> None:
        try:
            import psutil

            cpu = psutil.cpu_percent()
            vm  = psutil.virtual_memory()
            ram_used  = vm.used  / (1024 ** 3)
            ram_total = vm.total / (1024 ** 3)
            self._cpu.set_value(f"{cpu:.0f}%")
            self._ram.set_value(f"{ram_used:.1f} / {ram_total:.0f} GB")

            try:
                disk = psutil.disk_usage("C:\\")
                self._disk.set_value(f"{disk.percent:.0f}%  {disk.used/(1024**3):.0f}/{disk.total/(1024**3):.0f} GB")
            except Exception:
                self._disk.set_value("N/A")

            # GPU via nvidia-smi (non-blocking)
            try:
                import subprocess
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    timeout=1, stderr=subprocess.DEVNULL,
                ).decode().strip().split(",")
                util  = out[0].strip()
                mem_u = int(out[1].strip())
                mem_t = int(out[2].strip())
                self._gpu.set_value(f"{util}%  {mem_u}/{mem_t} MB")
            except Exception:
                self._gpu.set_value("N/A")

            bat = psutil.sensors_battery()
            if bat:
                status = "charging" if bat.power_plugged else "·"
                self._bat.set_value(f"{bat.percent:.0f}% {status}")
            else:
                self._bat.set_value("N/A")

            # Top 4 CPU hogs — normalize by cpu_count so max is 100%
            cpu_cores = psutil.cpu_count(logical=True) or 1
            procs = sorted(
                psutil.process_iter(["name", "cpu_percent"]),
                key=lambda p: p.info.get("cpu_percent") or 0,
                reverse=True,
            )
            for i, lbl in enumerate(self._hog_labels):
                if i < len(procs):
                    name = (procs[i].info.get("name") or "?")[:18]
                    raw  = procs[i].info.get("cpu_percent") or 0
                    pct  = min(raw / cpu_cores, 100.0)
                    lbl.setText(f"{name:<18} {pct:>5.1f}%")
                else:
                    lbl.setText("")

        except Exception:
            pass

        elapsed = int(time.monotonic() - self._start)
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        self._up.set_value(f"{h:02d}:{m:02d}:{s:02d}")

    def set_mode(self, mode: str) -> None:
        self._mode.set_value(mode.capitalize())


# ═══════════════════════════════════════════════════════════════════
# Right panel
# ═══════════════════════════════════════════════════════════════════

def _fetch_weather() -> str:
    """Return a one-line weather string using wttr.in (no API key needed)."""
    try:
        import urllib.request
        import json as _json
        url = "https://wttr.in/?format=j1"
        with urllib.request.urlopen(url, timeout=5) as r:
            data = _json.loads(r.read())
        cur  = data["current_condition"][0]
        area = data["nearest_area"][0]
        city = area["areaName"][0]["value"]
        country = area["country"][0]["value"]
        temp_c   = cur["temp_C"]
        desc     = cur["weatherDesc"][0]["value"]
        feels    = cur["FeelsLikeC"]
        humidity = cur["humidity"]
        return f"{city}, {country}\n{desc}  {temp_c}°C  (feels {feels}°C)  💧{humidity}%"
    except Exception:
        return "Weather unavailable"


class _RightPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(230)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)

        # ── Weather card
        wx_card = _Card("Weather", self)
        self._wx = QLabel("Fetching…")
        self._wx.setWordWrap(True)
        self._wx.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._wx.setStyleSheet("color:rgba(180,200,255,180);font-size:10px;")
        wx_card.body().addWidget(self._wx)
        v.addWidget(wx_card)

        # ── You card
        you_card = _Card("You", self)
        self._you = QLabel("—")
        self._you.setWordWrap(True)
        self._you.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._you.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)
        self._you.setStyleSheet("color:rgba(200,190,230,165);font-size:11px;")
        you_card.body().addWidget(self._you)
        v.addWidget(you_card)

        # ── Kira card
        kira_card = _Card("Kira", self)
        self._kira = QLabel("Ready.")
        self._kira.setWordWrap(True)
        self._kira.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._kira.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)
        self._kira.setStyleSheet("color:rgba(140,100,200,130);font-size:11px;")
        self._action = QLabel("")
        self._action.setStyleSheet("color:rgba(160,100,255,70);font-size:7px;letter-spacing:3px;")
        kira_card.body().addWidget(self._kira)
        kira_card.body().addWidget(self._action)
        v.addWidget(kira_card)
        v.addStretch()

        # Fetch weather once immediately on a thread, then refresh every 10 min
        self._refresh_weather()
        self._wx_timer = QTimer(self)
        self._wx_timer.timeout.connect(self._refresh_weather)
        self._wx_timer.start(10 * 60 * 1000)

    def _refresh_weather(self) -> None:
        threading.Thread(target=self._fetch_and_set_weather, daemon=True).start()

    def _fetch_and_set_weather(self) -> None:
        text = _fetch_weather()
        if _window is not None:
            QMetaObject.invokeMethod(
                _window, "_set_weather_text",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(str, text),
            )

    def set_weather(self, text: str) -> None:
        self._wx.setText(text)

    def set_transcript(self, you: str, kira: str) -> None:
        self._you.setText(you or "—")
        self._kira.setText(kira or "")

    def set_state(self, state: OrbState) -> None:
        actions = {
            "idle":       ("", "rgba(140,100,200,130)"),
            "listening":  ("recording audio", "rgba(140,100,200,130)"),
            "thinking":   ("calling brain...", "rgba(140,100,200,130)"),
            "speaking":   ("synthesising speech", "rgba(200,160,255,242)"),
            "autonomous": ("watching for changes", "rgba(100,70,160,128)"),
        }
        action, color = actions.get(state, ("", "rgba(140,100,200,130)"))
        self._action.setText(action.upper() if action else "")
        self._kira.setStyleSheet(f"color:{color};font-size:11px;")


# ═══════════════════════════════════════════════════════════════════
# Global hotkey (fires even when Qt window has no focus)
# ═══════════════════════════════════════════════════════════════════

def _register_global_hotkey(window: "QWidget") -> None:
    """Register KIRA_FULL_MODE_HOTKEY via the keyboard package on a daemon thread."""
    hk = os.environ.get("KIRA_FULL_MODE_HOTKEY", "ctrl+alt+f")
    try:
        import keyboard as _kb

        def _on_hotkey():
            if _window is None:
                return
            QMetaObject.invokeMethod(
                _window, "_toggle_full_mode",
                Qt.ConnectionType.QueuedConnection,
            )

        t = threading.Thread(target=lambda: _kb.add_hotkey(hk, _on_hotkey) or _kb.wait(),
                             daemon=True, name="kira-fullmode-hotkey")
        t.start()
        logger.info("Full-mode hotkey registered: %s", hk)
    except Exception as exc:
        logger.warning("Could not register full-mode hotkey %r: %s", hk, exc)


# ═══════════════════════════════════════════════════════════════════
# Main overlay window
# ═══════════════════════════════════════════════════════════════════

class _CompactDot(QWidget):
    """Small glowing dot for compact mode — replaces the old red dot."""
    SIZE = 40

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self._hue   = 270   # purple-ish default
        self._pulse = 0.0
        self._t     = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(50)  # 20fps is plenty for a dot

    def set_state(self, state: str) -> None:
        self._hue = {
            "idle":       270,
            "listening":  200,
            "thinking":   260,
            "speaking":   300,
            "autonomous": 220,
        }.get(state, 270)
        self._pulse = 1.0 if state in ("listening", "speaking") else 0.0

    def _tick(self) -> None:
        self._t += 1
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        S = self.SIZE
        CX = CY = S / 2

        pulse = 0.3 + 0.3 * math.sin(self._t * 0.18) if self._pulse else 0.0

        # Outer glow
        g = QRadialGradient(CX, CY, S * 0.5)
        c = QColor.fromHsv(self._hue, 200, 255)
        c.setAlpha(int((0.15 + pulse * 0.12) * 255))
        g.setColorAt(0, c)
        g.setColorAt(1, QColor(0, 0, 0, 0))
        painter.setBrush(g)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(0, 0, S, S)

        # Core dot
        r = 7 + pulse * 3
        core = QColor.fromHsv(self._hue, 160, 255, 220)
        painter.setBrush(core)
        painter.drawEllipse(int(CX - r), int(CY - r), int(r * 2), int(r * 2))

        # Bright center
        painter.setBrush(QColor(255, 255, 255, 180))
        painter.drawEllipse(int(CX - 2), int(CY - 2), 4, 4)
        painter.end()


class _KiraOverlay(QWidget):
    _FULL_ORB = 440
    _FADE_MS  = 300

    def __init__(self):
        super().__init__()
        self._full_mode  = False
        self._visible    = False
        self._drag_pos   = None

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        self._opacity = QGraphicsOpacityEffect(self)
        self._opacity.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity)

        self._fade_anim = QPropertyAnimation(self._opacity, b"opacity", self)
        self._fade_anim.setDuration(self._FADE_MS)
        self._fade_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)

        self._mode_label = QLabel("STANDBY", self)
        self._mode_label.setStyleSheet(
            "color:rgba(180,140,255,120);font-size:7px;letter-spacing:5px;"
        )
        self._mode_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Compact: small glowing dot
        self._dot = _CompactDot(self)

        # Full mode: large orb + panels
        self._orb_full = OrbRenderer(self._FULL_ORB, self)
        self._orb_full.hide()
        self._left  = _LeftPanel(self)
        self._right = _RightPanel(self)
        self._left.hide()
        self._right.hide()
        self._mode_label.hide()

        # Global hotkey registered via keyboard package (works even when Qt has no focus)
        _register_global_hotkey(self)

        self._apply_compact_geometry()
        self.hide()

    # ── Geometry ──────────────────────────────────────────────────

    def _screen(self):
        return QApplication.primaryScreen().availableGeometry()

    def _apply_compact_geometry(self) -> None:
        sg  = self._screen()
        pad = 16
        S   = _CompactDot.SIZE
        self.setMinimumSize(S, S)
        self.setMaximumSize(S, S)
        self.resize(S, S)
        self.move(sg.right() - S - pad, sg.bottom() - S - pad)
        self._dot.move(0, 0)

    def _apply_full_geometry(self) -> None:
        sg      = self._screen()
        sw, sh  = sg.width(), sg.height()
        orb     = self._FULL_ORB
        panel_w = 230
        gap     = 32
        label_h = 28
        total_w = panel_w + gap + orb + gap + panel_w
        total_h = max(orb + label_h, 500)
        self.setMinimumSize(400, 300)
        self.setMaximumSize(16777215, 16777215)  # remove any prior fixed constraint
        self.resize(total_w, total_h)
        self.move(sg.left() + (sw - total_w) // 2,
                  sg.top()  + (sh - total_h) // 2)
        orb_x = panel_w + gap
        orb_y = (total_h - orb - label_h) // 2
        self._orb_full.move(orb_x, orb_y)
        self._mode_label.setGeometry(orb_x, orb_y + orb, orb, label_h)
        panel_y = (total_h - 480) // 2
        self._left.move(0, max(0, panel_y))
        self._right.move(orb_x + orb + gap, max(0, panel_y))

    # ── Slots ─────────────────────────────────────────────────────

    @pyqtSlot()
    def _show_compact(self) -> None:
        if self._full_mode:
            return
        if not self._visible:
            self._visible = True
            self.show()
        self._fade_to(1.0)

    @pyqtSlot()
    def _hide_compact(self) -> None:
        if self._full_mode:
            return
        self._visible = False
        self._fade_to(0.0, on_done=self._after_hide_compact)

    def _after_hide_compact(self) -> None:
        if not self._full_mode:
            self.hide()

    @pyqtSlot()
    def _toggle_full_mode(self) -> None:
        self._set_full_mode(not self._full_mode)

    @pyqtSlot(bool)
    def _set_full_mode(self, on: bool) -> None:
        if on == self._full_mode:
            return
        self._full_mode = on
        if on:
            self._dot.hide()
            self._apply_full_geometry()
            self._orb_full.show()
            self._left.show()
            self._right.show()
            self._mode_label.show()
            self._visible = True
            self.show()
            self._fade_to(1.0)
        else:
            self._orb_full.hide()
            self._left.hide()
            self._right.hide()
            self._mode_label.hide()
            self._apply_compact_geometry()
            self._dot.show()
            self._visible = False
            self._fade_to(0.0, on_done=self.hide)

    @pyqtSlot(str)
    def _set_state(self, state: str) -> None:
        labels = {
            "idle": "STANDBY", "listening": "LISTENING",
            "thinking": "PROCESSING", "speaking": "SPEAKING",
            "autonomous": "AUTONOMOUS",
        }
        self._mode_label.setText(labels.get(state, state.upper()))
        self._dot.set_state(state)
        self._orb_full.set_state(state)
        self._right.set_state(state)
        self._left.set_mode(state)

    @pyqtSlot(str, str)
    def _set_transcript(self, you: str, kira: str) -> None:
        self._right.set_transcript(you, kira)

    @pyqtSlot(str)
    def _set_weather_text(self, text: str) -> None:
        self._right.set_weather(text)

    # ── Fade ──────────────────────────────────────────────────────

    def _fade_to(self, target: float, on_done=None) -> None:
        self._fade_anim.stop()
        try:
            self._fade_anim.finished.disconnect()
        except Exception:
            pass
        self._fade_anim.setStartValue(self._opacity.opacity())
        self._fade_anim.setEndValue(target)
        if on_done:
            self._fade_anim.finished.connect(on_done)
        self._fade_anim.start()

    def mousePressEvent(self, event):
        if self._full_mode and event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._full_mode and self._drag_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        super().mouseReleaseEvent(event)

    def closeEvent(self, event):
        event.ignore()

    def paintEvent(self, event):
        # Full mode: draw semi-dark background so panels are readable
        if self._full_mode:
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor(0, 0, 0, 200))
            painter.end()


# ═══════════════════════════════════════════════════════════════════
# Thread entrypoint
# ═══════════════════════════════════════════════════════════════════

def _qt_main() -> None:
    global _app, _window
    _app = QApplication.instance() or QApplication(sys.argv)
    _app.setQuitOnLastWindowClosed(False)
    _window = _KiraOverlay()
    _app.exec()


# ═══════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════

def start_on_main_thread() -> None:
    """Run the Qt event loop on the calling (main) thread. Blocks until Qt quits.

    Call this last in main() after starting the bot thread.
    """
    global _app, _window
    if not _qt_available:
        logger.warning("Overlay disabled (PyQt6 not available)")
        # Block forever so main thread doesn't exit and kill daemon threads
        import time as _t
        while True:
            _t.sleep(3600)
        return
    _qt_main()


def start() -> None:
    """Spawn Qt overlay on a background thread (standalone / local_voice use only).

    Not used when running under main.py — use start_on_main_thread() there.
    """
    global _thread
    if not _qt_available:
        logger.warning("Overlay disabled (PyQt6 not available)")
        return
    if _window is not None:
        return  # already running on main thread
    if _thread is not None and _thread.is_alive():
        return
    _thread = threading.Thread(target=_qt_main, daemon=True, name="kira-overlay-qt")
    _thread.start()
    import time as _t; _t.sleep(0.5)


def _invoke(slot_name: str, *args) -> None:
    """Post a call to the overlay window on the Qt thread."""
    if _window is None:
        return
    try:
        if args:
            types = [Q_ARG(type(a), a) for a in args]
            QMetaObject.invokeMethod(_window, slot_name,
                                     Qt.ConnectionType.QueuedConnection, *types)
        else:
            QMetaObject.invokeMethod(_window, slot_name,
                                     Qt.ConnectionType.QueuedConnection)
    except Exception as exc:
        logger.debug("overlay invoke failed: %s", exc)


def show() -> None:
    """Fade in the compact orb (no-op in full mode)."""
    _invoke("_show_compact")


def hide() -> None:
    """Fade out the compact orb (no-op in full mode)."""
    _invoke("_hide_compact")


def set_state(state: OrbState) -> None:
    """Update orb animation state."""
    _invoke("_set_state", state)


def set_transcript(you: str = "", kira: str = "") -> None:
    """Update right-panel transcript (full mode only)."""
    _invoke("_set_transcript", you, kira)


def set_full_mode(on: bool) -> None:
    """Switch between compact and full mode."""
    _invoke("_set_full_mode", on)


def stop() -> None:
    """Quit the Qt event loop."""
    if _app:
        QMetaObject.invokeMethod(_app, "quit", Qt.ConnectionType.QueuedConnection)
