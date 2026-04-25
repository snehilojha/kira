"""Tests for the registered script run tool."""

from __future__ import annotations

import types
import unittest
from unittest.mock import patch

from bot import registered_script_tool


class RegisteredScriptRunToolTests(unittest.TestCase):
    """Verify bounded script execution behavior."""

    def test_unknown_alias_returns_error(self) -> None:
        tool = registered_script_tool.RegisteredScriptRunTool()
        payload = registered_script_tool.RegisteredScriptRunInput(alias="missing")

        with patch.object(registered_script_tool, "load_script_config", return_value={}):
            result = tool.execute(payload, types.SimpleNamespace())

        self.assertTrue(result.is_error)
        self.assertIn("Unknown script alias", result.output)

    def test_known_alias_starts_process_and_registers_pid(self) -> None:
        tool = registered_script_tool.RegisteredScriptRunTool()
        payload = registered_script_tool.RegisteredScriptRunInput(
            alias="demo",
            args=["--steps", "10"],
        )
        fake_proc = types.SimpleNamespace(pid=1234, wait=lambda: 0)

        with patch.object(
            registered_script_tool,
            "load_script_config",
            return_value={
                "demo": {
                    "interpreter": "python",
                    "path": "demo.py",
                    "args": ["--base"],
                }
            },
        ), patch.object(
            registered_script_tool.subprocess,
            "Popen",
            return_value=fake_proc,
        ) as popen_mock, patch.object(
            registered_script_tool.process_registry,
            "register",
        ) as register_mock, patch.object(
            registered_script_tool,
            "_schedule_process_cleanup",
        ) as cleanup_mock:
            result = tool.execute(payload, types.SimpleNamespace())

        self.assertFalse(result.is_error)
        self.assertIn("PID 1234", result.output)
        popen_mock.assert_called_once_with(
            ["python", "demo.py", "--base", "--steps", "10"],
            stdout=registered_script_tool.subprocess.DEVNULL,
            stderr=registered_script_tool.subprocess.DEVNULL,
        )
        register_mock.assert_called_once_with(1234, fake_proc, "demo")
        cleanup_mock.assert_called_once_with(1234, fake_proc)


if __name__ == "__main__":
    unittest.main()
