"""FastAPI MCP server bound to localhost only.

Exposes the same executor capabilities as the Telegram bot, but via
HTTP endpoints for Windsurf or other local tools. No auth required
since it is unreachable from outside the machine.

Start separately:  uvicorn mcp.server:app --host 127.0.0.1 --port 8000
"""

import os
import sys
from pathlib import Path

import toml
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

# Ensure project root is on sys.path so bot.* imports work
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
load_dotenv(_PROJECT_ROOT / ".env")

from bot import executor
from bot import process_registry
from bot import scheduler

app = FastAPI(title="telegram-runner MCP", docs_url="/docs")


# ── Request models ────────────────────────────────────────────────

class RunRequest(BaseModel):
    """Request body for POST /run."""

    alias: str
    args: list[str] = []


class ShellRequest(BaseModel):
    """Request body for POST /shell."""

    command: str


# ── Config loading ────────────────────────────────────────────────

_SCRIPTS_CONFIG: dict = {}


def _load_config() -> None:
    """Load scripts.toml once at startup."""
    global _SCRIPTS_CONFIG
    config_path = _PROJECT_ROOT / "config" / "scripts.toml"
    if config_path.exists():
        _SCRIPTS_CONFIG = toml.load(config_path)


_load_config()
_DEFAULT_TIMEOUT = int(os.environ.get("DEFAULT_TIMEOUT", "30"))


# ── Endpoints ─────────────────────────────────────────────────────

@app.post("/run")
async def run_script(req: RunRequest) -> dict:
    """Execute a script by alias and return collected output."""
    script = _SCRIPTS_CONFIG.get(req.alias)
    if script is None:
        return {"error": f"Unknown alias: {req.alias}", "exit_code": -1}

    timeout = script.get("timeout", _DEFAULT_TIMEOUT)
    checkpoint = script.get("checkpoint_interval")
    script_args = list(script.get("args", [])) + req.args

    output_lines: list[str] = []
    gen = executor.run_command(
        interpreter=script["interpreter"],
        script_path=script["path"],
        args=script_args,
        timeout=timeout,
        alias=req.alias,
        checkpoint_interval=checkpoint,
    )
    async for chunk in gen:
        output_lines.append(chunk)

    stdout = "\n".join(output_lines)
    # Infer exit code from the last chunk
    exit_code = 0 if "✅" in (output_lines[-1] if output_lines else "") else 1
    return {"stdout": stdout, "stderr": "", "exit_code": exit_code}


@app.post("/shell")
async def run_shell(req: ShellRequest) -> dict:
    """Execute an arbitrary shell command."""
    output_lines: list[str] = []
    gen = executor.run_shell(req.command, timeout=_DEFAULT_TIMEOUT)
    async for chunk in gen:
        output_lines.append(chunk)

    stdout = "\n".join(output_lines)
    exit_code = 0 if "✅" in (output_lines[-1] if output_lines else "") else 1
    return {"stdout": stdout, "stderr": "", "exit_code": exit_code}


@app.get("/status")
async def get_status() -> list[dict]:
    """Return all tracked running processes."""
    return process_registry.list_processes()


@app.get("/schedules")
async def get_schedules() -> list[dict]:
    """Return all pending scheduled runs."""
    return scheduler.list_schedules()
