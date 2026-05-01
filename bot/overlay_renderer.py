"""Kira particle-sphere orb renderer — numpy-accelerated.

Uses numpy for all projection/noise math so the full sphere
runs smoothly at 60 fps without per-dot Python loops.

Public API
----------
set_state(state)  — 'idle'|'listening'|'thinking'|'speaking'|'autonomous'
"""

from __future__ import annotations

import math
import random
from datetime import datetime
from typing import Literal

import numpy as np
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QColor, QPainter, QRadialGradient
from PyQt6.QtWidgets import QWidget

OrbState = Literal["idle", "listening", "thinking", "speaking", "autonomous", "satisfied", "alert", "hot"]

_STATE_ENERGY: dict[OrbState, float] = {
    "idle":       0.32,
    "listening":  0.80,
    "thinking":   0.60,
    "speaking":   0.72,   # base — amplitude pushes it higher in real time
    "autonomous": 0.20,
    "satisfied":  0.55,   # brief warm pulse after success
    "alert":      0.70,   # proactive message incoming
    "hot":        0.38,   # stay-hot mode — slightly above idle, slow sway
}

# Breathing rhythm: (period_seconds, amplitude) — slows down late at night
_BREATHE_PERIOD = 5.0   # seconds per breath cycle
_BREATHE_AMP    = 0.14  # ± energy variation around idle


def _time_of_day_tint() -> tuple[int, int, int]:
    """Return (r, g, b) tint offset based on hour. Cool purple by day, warm amber by night."""
    hour = datetime.now().hour
    # 6-18: daytime — cool purple/blue. 18-22: evening — warm amber. 22-6: deep night — dim blue.
    if 6 <= hour < 18:
        return (160, 130, 255)   # cool purple
    elif 18 <= hour < 22:
        t = (hour - 18) / 4.0   # 0→1 over the evening
        r = int(160 + t * 80)
        g = int(130 - t * 50)
        b = int(255 - t * 130)
        return (r, g, b)         # fades from purple to amber
    else:
        return (120, 100, 200)   # dim blue-purple at night


def _value_noise_batch(pts: np.ndarray) -> np.ndarray:
    """Fast value noise for Nx3 array. Returns N floats in [0,1]."""
    ix = np.floor(pts[:, 0]).astype(np.int32)
    iy = np.floor(pts[:, 1]).astype(np.int32)
    iz = np.floor(pts[:, 2]).astype(np.int32)
    fx = pts[:, 0] - ix
    fy = pts[:, 1] - iy
    fz = pts[:, 2] - iz
    ux = fx * fx * (3 - 2 * fx)
    uy = fy * fy * (3 - 2 * fy)
    uz = fz * fz * (3 - 2 * fz)

    def h(n):
        s = np.sin(n.astype(np.float32)) * 43758.5453
        return s - np.floor(s)

    def hh(x, y, z):
        return h(x + h(y + h(z)))

    n000 = hh(ix,   iy,   iz  )
    n100 = hh(ix+1, iy,   iz  )
    n010 = hh(ix,   iy+1, iz  )
    n110 = hh(ix+1, iy+1, iz  )
    n001 = hh(ix,   iy,   iz+1)
    n101 = hh(ix+1, iy,   iz+1)
    n011 = hh(ix,   iy+1, iz+1)
    n111 = hh(ix+1, iy+1, iz+1)

    return (
        n000*(1-ux)*(1-uy)*(1-uz) + n100*ux*(1-uy)*(1-uz) +
        n010*(1-ux)*uy*(1-uz)     + n110*ux*uy*(1-uz) +
        n001*(1-ux)*(1-uy)*uz     + n101*ux*(1-uy)*uz +
        n011*(1-ux)*uy*uz         + n111*ux*uy*uz
    )


class OrbRenderer(QWidget):
    NUM_DOTS = 2400
    FPS      = 60

    def __init__(self, size: int = 200, parent=None):
        super().__init__(parent)
        self.orb_size = size
        self.setFixedSize(size, size)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)

        self._energy        = 0.32
        self._target_energy = 0.32
        self._speaking      = False
        self._amplitude     = 0.0   # live audio amplitude [0,1], decays each tick
        self._t             = 0
        self._rot_y         = 0.0
        self._rot_x         = 0.35
        self._state         = "idle"
        self._rot_drift     = 0.0   # slow random walk on rot_x
        self._tint          = _time_of_day_tint()
        self._tint_tick     = 0     # refresh tint every ~30 s
        self._micro_ticks   = 0     # countdown for transient micro-states

        # Fibonacci sphere — fixed unit vectors (Nx3)
        golden = math.pi * (3 - math.sqrt(5))
        idx = np.arange(self.NUM_DOTS, dtype=np.float64)
        y   = 1.0 - (idx / (self.NUM_DOTS - 1)) * 2.0
        r   = np.sqrt(np.clip(1.0 - y * y, 0, 1))
        th  = golden * idx
        self._pts0 = np.stack([r * np.cos(th), y, r * np.sin(th)], axis=1).astype(np.float32)
        self._seeds = (idx * 7.3391 % 100).astype(np.float32)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(1000 // self.FPS)

    # ── Public API ────────────────────────────────────────────────

    def set_state(self, state: OrbState) -> None:
        self._state = state
        self._target_energy = _STATE_ENERGY.get(state, 0.18)
        self._speaking = (state == "speaking")
        if not self._speaking:
            self._amplitude = 0.0
        # Transient micro-states auto-return to idle
        if state == "satisfied":
            self._micro_ticks = int(self.FPS * 1.5)   # 1.5 s
        elif state == "alert":
            self._micro_ticks = int(self.FPS * 2.0)   # 2 s
        else:
            self._micro_ticks = 0

    def set_amplitude(self, v: float) -> None:
        """Push a live audio amplitude value [0,1]. Called ~20x/sec during TTS."""
        self._amplitude = max(0.0, min(1.0, v))

    # ── Internal ──────────────────────────────────────────────────

    def _tick(self) -> None:
        # Micro-state countdown — auto-return to idle when timer expires
        if self._micro_ticks > 0:
            self._micro_ticks -= 1
            if self._micro_ticks == 0:
                self.set_state("idle")

        # During speaking: energy rides on base + live amplitude boost
        if self._speaking:
            target = self._target_energy + self._amplitude * 0.28
        elif self._state in ("idle", "hot"):
            # Breathing: slow sine wave so idle/hot never looks frozen
            breathe = math.sin(self._t * (2 * math.pi / (self.FPS * _BREATHE_PERIOD)))
            target = self._target_energy + breathe * _BREATHE_AMP
        elif self._state == "satisfied":
            # Warm double-pulse: quick rise then settle
            phase = 1.0 - (self._micro_ticks / (self.FPS * 1.5))
            target = self._target_energy + math.sin(phase * math.pi * 2) * 0.12
        elif self._state == "alert":
            # Rapid flicker to draw attention
            target = self._target_energy + math.sin(self._t * 0.4) * 0.10
        else:
            target = self._target_energy
        self._energy += (target - self._energy) * 0.10
        self._amplitude *= 0.82
        self._t += 1
        self._rot_y += 0.004 + self._energy * 0.008

        # Slow random drift on rot_x in idle/hot — orb never loops identically
        if self._state in ("idle", "hot") and self._t % 40 == 0:
            self._rot_drift += random.uniform(-0.003, 0.003)
            self._rot_drift = max(-0.12, min(0.12, self._rot_drift))
        self._rot_x = 0.35 + self._rot_drift

        # Refresh time-of-day tint every ~30 s
        self._tint_tick += 1
        if self._tint_tick >= self.FPS * 30:
            self._tint = _time_of_day_tint()
            self._tint_tick = 0

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        W = H = self.orb_size
        CX = CY = W / 2.0
        BASE_R = W * 0.5 * 0.67
        energy = self._energy
        t      = self._t
        nt     = t * 0.004

        # ── Rotate unit sphere points (vectorized)
        cy, sy = math.cos(self._rot_y), math.sin(self._rot_y)
        cx, sx = math.cos(self._rot_x), math.sin(self._rot_x)

        pts = self._pts0.copy()
        x_ =  pts[:, 0] * cy - pts[:, 2] * sy
        z_ =  pts[:, 0] * sy + pts[:, 2] * cy
        pts[:, 0], pts[:, 2] = x_, z_
        y_ =  pts[:, 1] * cx - pts[:, 2] * sx
        z_ =  pts[:, 1] * sx + pts[:, 2] * cx
        pts[:, 1], pts[:, 2] = y_, z_

        # ── Noise displacement
        turb_scale = 0.55 + energy * 1.8
        turb_amp   = 0.12 + energy * 0.38
        noise_pts  = pts * turb_scale
        noise_pts[:, 0] += self._seeds
        noise_pts[:, 2] += nt
        n = _value_noise_batch(noise_pts)
        # Churn driven by live amplitude — surface erupts when Kira is loud
        speak_churn = (0.10 + self._amplitude * 0.22) if self._speaking else 0.0
        disp = (n * 2 - 1) * turb_amp + speak_churn * (n - 0.5)
        radius = BASE_R * (1.0 + disp)

        sx_arr = CX + pts[:, 0] * radius
        sy_arr = CY - pts[:, 1] * radius
        rz     = pts[:, 2]

        # ── Shell falloff: bright at silhouette, dark at centre
        abs_z    = np.abs(rz)
        edgeness = 1.0 - abs_z
        shell    = np.power(np.clip(edgeness, 0, 1), 1.8)   # softer falloff = more visible shell
        interior = np.where(abs_z > 0.4, (abs_z - 0.4) * 0.08 * energy, 0.0)
        alpha_arr = np.clip(shell * (0.65 + energy * 0.35) + interior, 0, 1)
        # Slightly larger dots for visibility
        size_arr  = 0.35 + shell * (1.1 + energy * 0.6)

        # ── Time-of-day tint: edge dots bright white, interior has tinted hue
        tr, tg, tb = self._tint
        bright_w = np.clip(210 + (shell * 45).astype(np.int32), 0, 255)
        tint_r   = np.clip(tr  + (shell * 40).astype(np.int32), 0, 255)
        tint_b   = np.clip(tb  + (shell * 25).astype(np.int32), 0, 255)

        # ── Sort back-to-front by rz
        order = np.argsort((rz + 1.0) * 0.5)

        # ── Outer aura (tinted glow)
        tr, tg, tb = self._tint
        aura = QRadialGradient(CX, CY, BASE_R * 1.7)
        aura.setColorAt(0,   QColor(tr, tg, tb, int(energy * 40)))
        aura.setColorAt(0.5, QColor(max(0, tr-60), max(0, tg-50), max(0, tb-35), int(energy * 18)))
        aura.setColorAt(1,   QColor(0, 0, 0, 0))
        painter.setBrush(aura)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(
            int(CX - BASE_R * 1.7), int(CY - BASE_R * 1.7),
            int(BASE_R * 3.4),      int(BASE_R * 3.4),
        )

        # ── Draw dots
        painter.setPen(Qt.PenStyle.NoPen)
        for i in order:
            a = float(alpha_arr[i])
            if a < 0.03:
                continue
            sx_i = float(sx_arr[i])
            sy_i = float(sy_arr[i])
            s    = float(size_arr[i])
            rv   = int(tint_r[i])
            gv   = int(bright_w[i])
            bv   = int(tint_b[i])

            # Glow halo on bright edge dots
            if a > 0.40 and s > 0.6:
                gr = s * 3.2
                painter.setBrush(QColor(rv, gv, bv, max(0, int(a * 22))))
                painter.drawEllipse(int(sx_i - gr), int(sy_i - gr),
                                    max(1, int(gr * 2)), max(1, int(gr * 2)))

            painter.setBrush(QColor(rv, gv, bv, max(0, int(a * 255))))
            r2 = max(0.5, s)
            painter.drawEllipse(int(sx_i - r2), int(sy_i - r2),
                                max(1, int(r2 * 2)), max(1, int(r2 * 2)))

        painter.end()
