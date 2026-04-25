"""Tests for the approved check command tool."""

from __future__ import annotations

import types
import unittest
from unittest.mock import patch

from bot import check_tool


class TestCommandRunToolTests(unittest.TestCase):
    """Verify approved verification command execution."""

    def test_unknown_command_id_returns_error(self) -> None:
        tool = check_tool.TestCommandRunTool()
        payload = check_tool.TestCommandRunInput(command_id="missing")

        result = tool.execute(payload, types.SimpleNamespace())

        self.assertTrue(result.is_error)
        self.assertIn("Unknown check command id", result.output)

    def test_known_command_id_runs_mapped_command(self) -> None:
        tool = check_tool.TestCommandRunTool()
        payload = check_tool.TestCommandRunInput(command_id="unit_brain")
        completed = types.SimpleNamespace(
            returncode=0,
            stdout="Ran 7 tests\nOK\n",
            stderr="",
        )

        with patch.object(
            check_tool.subprocess,
            "run",
            return_value=completed,
        ) as run_mock:
            result = tool.execute(payload, types.SimpleNamespace())

        self.assertFalse(result.is_error)
        self.assertIn("Check unit_brain passed", result.output)
        self.assertIn("Ran 7 tests", result.output)
        args, kwargs = run_mock.call_args
        command = args[0]
        self.assertIn("-m", command)
        self.assertIn("unittest", command)
        self.assertEqual(kwargs["cwd"], str(check_tool._PROJECT_ROOT))
        self.assertTrue(kwargs["capture_output"])

    def test_failed_command_returns_error_with_output_tail(self) -> None:
        tool = check_tool.TestCommandRunTool()
        payload = check_tool.TestCommandRunInput(command_id="unit_policy")
        completed = types.SimpleNamespace(
            returncode=1,
            stdout="FAIL\n",
            stderr="traceback\n",
        )

        with patch.object(check_tool.subprocess, "run", return_value=completed):
            result = tool.execute(payload, types.SimpleNamespace())

        self.assertTrue(result.is_error)
        self.assertIn("Check unit_policy failed", result.output)
        self.assertIn("[stdout]", result.output)
        self.assertIn("[stderr]", result.output)


if __name__ == "__main__":
    unittest.main()
