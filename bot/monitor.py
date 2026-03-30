"""Background system resource monitoring for proactive Telegram alerts.

The monitor samples CPU, RAM, and GPU temperature on a fixed interval and
sends alerts through ``bot.notifier`` when configurable thresholds are crossed.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass

import psutil

logger = logging.getLogger(__name__)

_DEFAULT_CPU_ALERT_THRESHOLD = 95.0
_DEFAULT_RAM_ALERT_THRESHOLD = 95.0
_DEFAULT_GPU_TEMP_ALERT = 85.0
_DEFAULT_POLL_INTERVAL_SECONDS = 30.0
_DEFAULT_ALERT_COOLDOWN_SECONDS = 300.0
_DEFAULT_CPU_SAMPLE_INTERVAL_SECONDS = 0.1


@dataclass(frozen=True)
class MonitorConfig:
    """Thresholds and timing values used by the resource monitor."""

    cpu_alert_threshold: float
    ram_alert_threshold: float
    gpu_temp_alert: float
    poll_interval_seconds: float
    alert_cooldown_seconds: float


@dataclass(frozen=True)
class SystemMetrics:
    """Current system resource usage sampled by the monitor."""

    cpu_percent: float
    ram_percent: float
    gpu_temp_celsius: float | None


def load_config_from_env() -> MonitorConfig:
    """Load monitor thresholds and polling cadence from environment variables."""
    return MonitorConfig(
        cpu_alert_threshold=_read_env_float("CPU_ALERT_THRESHOLD", _DEFAULT_CPU_ALERT_THRESHOLD),
        ram_alert_threshold=_read_env_float("RAM_ALERT_THRESHOLD", _DEFAULT_RAM_ALERT_THRESHOLD),
        gpu_temp_alert=_read_env_float("GPU_TEMP_ALERT", _DEFAULT_GPU_TEMP_ALERT),
        poll_interval_seconds=_read_env_float("MONITOR_INTERVAL_SECONDS", _DEFAULT_POLL_INTERVAL_SECONDS),
        alert_cooldown_seconds=_read_env_float("MONITOR_ALERT_COOLDOWN_SECONDS", _DEFAULT_ALERT_COOLDOWN_SECONDS),
    )


def collect_system_metrics() -> SystemMetrics:
    """Collect the latest CPU, RAM, and GPU temperature readings."""
    cpu_percent = float(psutil.cpu_percent(interval=_DEFAULT_CPU_SAMPLE_INTERVAL_SECONDS))
    memory = psutil.virtual_memory()
    gpu_temp_celsius = _get_gpu_temperature_celsius()
    return SystemMetrics(
        cpu_percent=cpu_percent,
        ram_percent=float(memory.percent),
        gpu_temp_celsius=gpu_temp_celsius,
    )


def evaluate_alerts(
    metrics: SystemMetrics,
    config: MonitorConfig,
    last_alert_at: dict[str, float],
    now: float | None = None,
) -> tuple[list[str], dict[str, float]]:
    """Build alert messages for metrics that exceed configured thresholds.

    Args:
        metrics: The current sampled system metrics.
        config: Thresholds and timing values.
        last_alert_at: Per-metric monotonic timestamps of the last alert.
        now: Current monotonic time, injected for deterministic tests.

    Returns:
        A tuple of ``(messages, updated_last_alert_at)``.
    """
    current_time = time.monotonic() if now is None else now
    updated_last_alert_at = dict(last_alert_at)
    alerts: list[str] = []

    if metrics.cpu_percent >= config.cpu_alert_threshold and _cooldown_elapsed(
        updated_last_alert_at.get("cpu"), current_time, config.alert_cooldown_seconds
    ):
        alerts.append(
            _format_alert_message(
                label="CPU",
                current=f"{metrics.cpu_percent:.1f}%",
                threshold=f"{config.cpu_alert_threshold:.1f}%",
            )
        )
        updated_last_alert_at["cpu"] = current_time

    if metrics.ram_percent >= config.ram_alert_threshold and _cooldown_elapsed(
        updated_last_alert_at.get("ram"), current_time, config.alert_cooldown_seconds
    ):
        alerts.append(
            _format_alert_message(
                label="RAM",
                current=f"{metrics.ram_percent:.1f}%",
                threshold=f"{config.ram_alert_threshold:.1f}%",
            )
        )
        updated_last_alert_at["ram"] = current_time

    if metrics.gpu_temp_celsius is not None and metrics.gpu_temp_celsius >= config.gpu_temp_alert and _cooldown_elapsed(
        updated_last_alert_at.get("gpu"), current_time, config.alert_cooldown_seconds
    ):
        alerts.append(
            _format_alert_message(
                label="GPU temp",
                current=f"{metrics.gpu_temp_celsius:.1f}°C",
                threshold=f"{config.gpu_temp_alert:.1f}°C",
            )
        )
        updated_last_alert_at["gpu"] = current_time

    return alerts, updated_last_alert_at


async def start_monitor() -> None:
    """Run the background monitoring loop until the bot shuts down."""
    from bot import notifier

    config = load_config_from_env()
    last_alert_at: dict[str, float] = {}
    logger.info(
        "Resource monitor started (cpu>=%.1f%% ram>=%.1f%% gpu>=%.1f°C interval=%.1fs cooldown=%.1fs)",
        config.cpu_alert_threshold,
        config.ram_alert_threshold,
        config.gpu_temp_alert,
        config.poll_interval_seconds,
        config.alert_cooldown_seconds,
    )

    while True:
        try:
            metrics = await asyncio.to_thread(collect_system_metrics)
            alerts, last_alert_at = evaluate_alerts(metrics, config, last_alert_at)
            for message in alerts:
                await notifier.send(message)
        except asyncio.CancelledError:
            logger.info("Resource monitor cancelled")
            raise
        except Exception as exc:
            logger.exception("Resource monitor cycle failed: %s", exc)

        await asyncio.sleep(config.poll_interval_seconds)


def _cooldown_elapsed(last_alert_time: float | None, current_time: float, cooldown_seconds: float) -> bool:
    """Return ``True`` when a metric is allowed to alert again."""
    if last_alert_time is None:
        return True
    return current_time - last_alert_time >= cooldown_seconds


def _format_alert_message(*, label: str, current: str, threshold: str) -> str:
    """Format a human-readable alert message for Telegram."""
    return f"⚠️ Resource alert: {label} is at {current} (threshold: {threshold})."


def _get_gpu_temperature_celsius() -> float | None:
    """Return the highest detected GPU temperature, or ``None`` if unavailable."""
    try:
        import GPUtil
    except Exception as exc:
        logger.debug("GPU temperature monitoring unavailable: %s", exc)
        return None

    temperatures: list[float] = []
    try:
        for gpu in GPUtil.getGPUs():
            temp = getattr(gpu, "temperature", None)
            if temp is not None:
                temperatures.append(float(temp))
    except Exception as exc:
        logger.warning("Failed to query GPU temperature: %s", exc)
        return None

    if not temperatures:
        return None
    return max(temperatures)


def _read_env_float(name: str, default: float) -> float:
    """Read a float environment variable with a validated fallback."""
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default

    try:
        value = float(raw_value)
    except ValueError:
        logger.warning("Invalid %s value %r; using default %.1f", name, raw_value, default)
        return default

    if value <= 0:
        logger.warning("Non-positive %s value %r; using default %.1f", name, raw_value, default)
        return default

    return value
