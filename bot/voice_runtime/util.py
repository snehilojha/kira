"""Config constants, formatting helpers, and logging setup for the voice runtime."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import psutil

from bot import app_control
from bot.voice_runtime.models import LocalVoiceResult, ParsedCommand

logger = logging.getLogger(__name__)

# NOTE: this file lives at bot/voice_runtime/util.py, so the project root is
# three parents up (voice_runtime → bot → project), not two as in the old
# bot/local_voice.py.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"
_DEFAULT_LOG_PATH = _PROJECT_ROOT / "logs" / "local_voice.log"
_DEFAULT_SAMPLE_RATE = 16000
_DEFAULT_RECORD_SECONDS = 5.0
_DEFAULT_MAX_RECORD_SECONDS = 15.0
_DEFAULT_SILENCE_SECONDS = 0.8
_DEFAULT_SILENCE_RMS = 200
_DEFAULT_TRIGGER = "hotkey"
_DEFAULT_HOTKEY = "ctrl+alt+k"


def _format_command(parsed: ParsedCommand) -> str:
    return " ".join([parsed.command, *parsed.args]).strip()


def _format_process_status() -> str:
    processes = []
    for proc in psutil.process_iter(["pid", "name"]):
        name = proc.info.get("name") or ""
        if name:
            processes.append(f"{name} ({proc.info['pid']})")
        if len(processes) >= 12:
            break
    if not processes:
        return "No processes found."
    return "Running processes:\n" + "\n".join(f"- {item}" for item in processes)


def _format_sysinfo() -> str:
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    return (
        "System Info\n"
        f"CPU: {cpu}%\n"
        f"RAM: {mem.used / (1024**3):.1f} / {mem.total / (1024**3):.1f} GB ({mem.percent}%)"
    )


def _from_action_result(result: app_control.ActionResult) -> LocalVoiceResult:
    return LocalVoiceResult(ok=result.ok, message=result.message, spoken=result.spoken)


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().rstrip(".!?").split())


def _format_provider_error(exc: Exception, prefix: str) -> str:
    """Return a concise local-console error without exposing secret values."""
    status_code = getattr(exc, "status_code", None)
    class_name = type(exc).__name__
    if status_code == 401 or class_name == "AuthenticationError":
        return (
            f"{prefix}: OpenAI rejected the configured API key. "
            "Update OPENAI_API_KEY in .env, or set KIRA_API_KEY to a valid OpenAI-compatible key."
        )
    return f"{prefix}: {class_name}: {exc}"


def _setup_logging() -> None:
    """Configure console/file logging for local voice."""
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_path = Path(os.environ.get("KIRA_LOCAL_VOICE_LOG_FILE", str(_DEFAULT_LOG_PATH)))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handlers: list[logging.Handler] = [logging.FileHandler(log_path, encoding="utf-8")]
    try:
        handlers.append(logging.StreamHandler())
    except Exception:
        pass

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=handlers,
        force=True,
    )
