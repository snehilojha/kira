"""Astra tool for running approved verification commands.

This tool is intentionally narrower than arbitrary shell. The model selects a
known command id, Kira asks for approval, and only then runs the mapped command.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from astra_node.core.tool import BaseTool, PermissionLevel, ToolContext, ToolResult

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_TIMEOUT_SECONDS = 300
_MAX_OUTPUT_CHARS = 6000


@dataclass(frozen=True)
class CheckCommand:
    """One approved verification command."""

    command_id: str
    description: str
    command: list[str]
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS


def get_allowed_check_commands(project_root: Path | None = None) -> dict[str, CheckCommand]:
    """Return the approved check command map."""
    root = project_root or _PROJECT_ROOT
    python = str(root / ".venv" / "Scripts" / "python.exe")
    return {
        "unit_all": CheckCommand(
            command_id="unit_all",
            description="Run the full unittest suite.",
            command=[python, "-m", "unittest", "discover", "-s", "tests"],
        ),
        "unit_brain": CheckCommand(
            command_id="unit_brain",
            description="Run brain runtime tests.",
            command=[python, "-m", "unittest", "discover", "-s", "tests", "-p", "test_brain.py"],
        ),
        "unit_router": CheckCommand(
            command_id="unit_router",
            description="Run router tests.",
            command=[python, "-m", "unittest", "discover", "-s", "tests", "-p", "test_router.py"],
        ),
        "unit_policy": CheckCommand(
            command_id="unit_policy",
            description="Run capability policy tests.",
            command=[
                python,
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-p",
                "test_capability_policy.py",
            ],
        ),
    }


class TestCommandRunInput(BaseModel):
    """Input schema for running one approved check command."""

    command_id: str = Field(..., description="Approved check id, e.g. unit_all.")

    @field_validator("command_id")
    @classmethod
    def validate_command_id(cls, value: str) -> str:
        command_id = value.strip()
        if not command_id:
            raise ValueError("command_id must not be empty")
        return command_id


class TestCommandRunTool(BaseTool):
    """Run an approved project verification command and return its result."""

    name = "test_command_run"
    description = (
        "Run an approved verification command by command_id. Available ids: "
        "unit_all, unit_brain, unit_router, unit_policy."
    )
    input_schema = TestCommandRunInput
    permission_level = PermissionLevel.ASK_USER

    def execute(self, input: TestCommandRunInput, ctx: ToolContext) -> ToolResult:
        allowed = get_allowed_check_commands(_PROJECT_ROOT)
        check = allowed.get(input.command_id)
        if check is None:
            available = ", ".join(sorted(allowed.keys()))
            return ToolResult.err(
                f"Unknown check command id: {input.command_id}. Available: {available}"
            )

        start = time.monotonic()
        try:
            completed = subprocess.run(
                check.command,
                cwd=str(_PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=check.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return ToolResult.err(
                f"Check {check.command_id} timed out after {check.timeout_seconds}s."
            )
        except OSError as exc:
            return ToolResult.err(f"Failed to run check {check.command_id}: {exc}")

        elapsed = time.monotonic() - start
        output = _combine_output(completed.stdout, completed.stderr)
        status = "passed" if completed.returncode == 0 else "failed"
        header = (
            f"Check {check.command_id} {status} "
            f"(exit_code={completed.returncode}, duration={elapsed:.1f}s).\n"
            f"Command: {' '.join(check.command)}"
        )
        body = _tail(output, _MAX_OUTPUT_CHARS)
        message = f"{header}\n\nOutput tail:\n{body}" if body else header
        if completed.returncode == 0:
            return ToolResult.ok(message)
        return ToolResult.err(message)


def _combine_output(stdout: str, stderr: str) -> str:
    """Combine stdout/stderr with labels when both are present."""
    if stdout and stderr:
        return f"[stdout]\n{stdout}\n[stderr]\n{stderr}"
    return stdout or stderr or ""


def _tail(text: str, max_chars: int) -> str:
    """Return a bounded tail of command output."""
    if len(text) <= max_chars:
        return text.strip()
    return "[...output truncated...]\n" + text[-max_chars:].strip()
