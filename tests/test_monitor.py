"""Tests for the proactive resource monitoring helpers."""

from __future__ import annotations

import os
import unittest
from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import patch

from bot import monitor


class MonitorTests(unittest.TestCase):
    """Verify monitoring threshold handling and system metric collection."""

    def test_evaluate_alerts_emits_alerts_for_exceeded_thresholds(self) -> None:
        """All metrics above threshold should produce alerts and update cooldown state."""
        config = monitor.MonitorConfig(
            cpu_alert_threshold=90.0,
            ram_alert_threshold=80.0,
            gpu_temp_alert=70.0,
            poll_interval_seconds=30.0,
            alert_cooldown_seconds=300.0,
        )
        metrics = monitor.SystemMetrics(cpu_percent=95.1, ram_percent=81.2, gpu_temp_celsius=72.5)

        alerts, updated = monitor.evaluate_alerts(metrics, config, last_alert_at={}, now=100.0)

        self.assertEqual(len(alerts), 3)
        self.assertIn("CPU is at 95.1%", alerts[0])
        self.assertIn("RAM is at 81.2%", alerts[1])
        self.assertIn("GPU temp is at 72.5°C", alerts[2])
        self.assertEqual(updated, {"cpu": 100.0, "ram": 100.0, "gpu": 100.0})

    def test_evaluate_alerts_honours_cooldown_window(self) -> None:
        """Recently alerted metrics should not re-alert until cooldown expires."""
        config = monitor.MonitorConfig(
            cpu_alert_threshold=90.0,
            ram_alert_threshold=80.0,
            gpu_temp_alert=70.0,
            poll_interval_seconds=30.0,
            alert_cooldown_seconds=300.0,
        )
        metrics = monitor.SystemMetrics(cpu_percent=95.1, ram_percent=81.2, gpu_temp_celsius=72.5)

        alerts, updated = monitor.evaluate_alerts(
            metrics,
            config,
            last_alert_at={"cpu": 100.0, "ram": 100.0, "gpu": 100.0},
            now=200.0,
        )

        self.assertEqual(alerts, [])
        self.assertEqual(updated, {"cpu": 100.0, "ram": 100.0, "gpu": 100.0})

    def test_evaluate_alerts_ignores_missing_gpu_temperature(self) -> None:
        """The monitor should still alert on CPU and RAM when GPU data is unavailable."""
        config = monitor.MonitorConfig(
            cpu_alert_threshold=90.0,
            ram_alert_threshold=80.0,
            gpu_temp_alert=70.0,
            poll_interval_seconds=30.0,
            alert_cooldown_seconds=300.0,
        )
        metrics = monitor.SystemMetrics(cpu_percent=95.1, ram_percent=81.2, gpu_temp_celsius=None)

        alerts, updated = monitor.evaluate_alerts(metrics, config, last_alert_at={}, now=100.0)

        self.assertEqual(len(alerts), 2)
        self.assertEqual(updated, {"cpu": 100.0, "ram": 100.0})

    def test_collect_system_metrics_uses_psutil_and_gpu_helper(self) -> None:
        """collect_system_metrics should read CPU, RAM, and GPU values from helpers."""
        virtual_memory = namedtuple("virtual_memory", ["percent"])

        with patch.object(monitor.psutil, "cpu_percent", return_value=12.5) as cpu_mock, patch.object(
            monitor.psutil,
            "virtual_memory",
            return_value=virtual_memory(percent=34.5),
        ) as mem_mock, patch.object(monitor, "_get_gpu_temperature_celsius", return_value=77.0) as gpu_mock:
            metrics = monitor.collect_system_metrics()

        self.assertEqual(metrics.cpu_percent, 12.5)
        self.assertEqual(metrics.ram_percent, 34.5)
        self.assertEqual(metrics.gpu_temp_celsius, 77.0)
        cpu_mock.assert_called_once_with(interval=0.1)
        mem_mock.assert_called_once_with()
        gpu_mock.assert_called_once_with()

    def test_read_env_float_falls_back_for_invalid_values(self) -> None:
        """Invalid or non-positive env values should fall back to the default."""
        with patch.dict(os.environ, {"MONITOR_INTERVAL_SECONDS": "not-a-number"}, clear=False):
            self.assertEqual(monitor._read_env_float("MONITOR_INTERVAL_SECONDS", 30.0), 30.0)

        with patch.dict(os.environ, {"MONITOR_INTERVAL_SECONDS": "0"}, clear=False):
            self.assertEqual(monitor._read_env_float("MONITOR_INTERVAL_SECONDS", 30.0), 30.0)

    def test_format_alert_message(self) -> None:
        """Alert messages should use a consistent Telegram-friendly format."""
        self.assertEqual(
            monitor._format_alert_message(label="CPU", current="95.0%", threshold="90.0%"),
            "⚠️ Resource alert: CPU is at 95.0% (threshold: 90.0%).",
        )


if __name__ == "__main__":
    unittest.main()
