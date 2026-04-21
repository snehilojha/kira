"""Astra tool for launching configured Kira script aliases.

The tool intentionally exposes only aliases from ``config/scripts.toml`` and
requires confirmation through Kira's capability policy before execution.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Any

import toml
from pydantic import BaseModel, Field, field_validator

from astra_node.core.tool import BaseTool, PermissionLevel, ToolContext, ToolResult

from bot import process_registry

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_CONFIG_PATH = _PROJECT_ROOT / "config" / "scripts.toml"
_MAX_EXTRA_ARGS = 20


class RegisteredScriptRunInput(BaseModel):
    """Input schema for launching one configured script alias."""

    alias: str = Field(..., description="Script alias from config/scripts.toml.")
    args: list[str] = Field(default_factory=list, description="Optional extra CLI arguments.")

    @field_validator("alias")
    @classmethod
    def validate_alias(cls, value: str) -> str:
        alias = value.strip()
        if not alias:
            raise ValueError("alias must not be empty")
        return alias

    @field_validator("args")
    @classmethod
    def validate_args(cls, value: list[str]) -> list[str]:
        if len(value) > _MAX_EXTRA_ARGS:
            raise ValueError(f"args must contain at most {_MAX_EXTRA_ARGS} items")
        return [str(item) for item in value]


class RegisteredScriptRunTool(BaseTool):
    """Launch a configured Kira script alias as a tracked background process."""

    name = "registered_script_run"
    description = (
        "Run a script alias from Kira's scripts.toml. Use this instead of bash "
        "when the task asks to run a known configured script."
    )
    input_schema = RegisteredScriptRunInput
    permission_level = PermissionLevel.ASK_USER

    def execute(self, input: RegisteredScriptRunInput, ctx: ToolContext) -> ToolResult:
        scripts = load_script_config()
        script = scripts.get(input.alias)
        if script is None:
            available = ", ".join(sorted(scripts.keys())) or "(none configured)"
            return ToolResult.err(f"Unknown script alias: {input.alias}. Available: {available}")

        interpreter = str(script.get("interpreter", "")).strip()
        script_path = str(script.get("path", "")).strip()
        if not interpreter or not script_path:
            return ToolResult.err(f"Script alias {input.alias} is missing interpreter or path.")

        configured_args = [str(item) for item in script.get("args", [])]
        command = [interpreter, script_path, *configured_args, *input.args]

        try:
            proc = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            return ToolResult.err(f"Failed to start script {input.alias}: {exc}")

        process_registry.register(proc.pid, proc, input.alias)  # type: ignore[arg-type]
        _schedule_process_cleanup(proc.pid, proc)

        return ToolResult.ok(
            f"Started registered script '{input.alias}' as PID {proc.pid}."
        )


def load_script_config(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load script aliases from TOML."""
    config_path = path or _SCRIPTS_CONFIG_PATH
    if not config_path.exists():
        return {}
    loaded = toml.load(config_path)
    if not isinstance(loaded, dict):
        return {}
    return {
        str(alias): config
        for alias, config in loaded.items()
        if isinstance(config, dict)
    }


def _schedule_process_cleanup(pid: int, proc: subprocess.Popen) -> None:
    """Deregister a Popen-backed process after it exits."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    async def _cleanup() -> None:
        await asyncio.to_thread(proc.wait)
        process_registry.deregister(pid)

    loop.create_task(_cleanup(), name=f"registered-script-cleanup-{pid}")
