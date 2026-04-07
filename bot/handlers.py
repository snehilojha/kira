"""All Telegram command handlers.

Every handler is decorated with ``@require_auth``.  One function per command.
Handlers delegate heavy lifting to executor, process_registry, scheduler,
watchdog, and notifier — they never contain business logic directly.
"""

import asyncio
from collections import deque
import io
import json
import logging
import os
import re
import subprocess
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import mss
import mss.tools
import psutil
import pyperclip
import toml

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from bot.auth import require_auth
from bot import db
from bot import executor
from bot import notifier
from bot import process_registry
from bot import scheduler
from bot import voice
from bot import watchdog

logger = logging.getLogger(__name__)

# ── Config loading ────────────────────────────────────────────────

_SCRIPTS_CONFIG: dict = {}
_DEFAULT_TIMEOUT: int = 30


def load_config() -> None:
    """Load scripts.toml and default timeout. Called once from main.py."""
    global _SCRIPTS_CONFIG, _DEFAULT_TIMEOUT
    config_path = Path(__file__).resolve().parent.parent / "config" / "scripts.toml"
    if config_path.exists():
        _SCRIPTS_CONFIG = toml.load(config_path)
        logger.info("Loaded %d script aliases from %s", len(_SCRIPTS_CONFIG), config_path)
    else:
        logger.warning("scripts.toml not found at %s", config_path)
    _DEFAULT_TIMEOUT = int(os.environ.get("DEFAULT_TIMEOUT", "30"))


def _get_script(alias: str) -> dict | None:
    """Look up a script definition by alias."""
    return _SCRIPTS_CONFIG.get(alias)


# ── Persistent working directory ──────────────────────────────────

# Shared bot-session CWD. Updated by /cd, read by /ls and /shell.
# Starts at the directory the bot process was launched from.
_CWD: Path = Path.cwd()


def get_cwd() -> Path:
    """Return the current bot working directory."""
    return _CWD


def set_cwd(path: Path) -> None:
    """Update the bot working directory. Caller must validate first."""
    global _CWD
    _CWD = path


# ── Destructive-command keywords for /shell confirmation ──────────

_DESTRUCTIVE_PATTERNS = re.compile(
    r"\b(rm|del|format|rmdir|rd\s*/s|DROP|DELETE\s+FROM)\b", re.IGNORECASE
)

# Pending confirmations keyed by callback data token
_PENDING_CONFIRMS: dict[str, dict] = {}
_CONFIRM_TIMEOUT = 30  # seconds

# Optional context file injected into the /ask system prompt.
_PROJECT_CONTEXT_PATH = Path(os.environ.get("PROJECT_CONTEXT_PATH", str(Path(__file__).resolve().parent.parent / "context.md")))
_PROJECT_CONTEXT_MAX_CHARS = 12000
_RECENT_OUTPUT_MAX_CHARS = 2000
_RECENT_OUTPUT_LINES: deque[str] = deque(maxlen=20)


# ── Helper: stream executor output back to Telegram ───────────────

async def _stream_to_chat(update: Update, gen) -> None:
    """Consume an async generator from executor and send chunks as messages."""
    collected: list[str] = []
    async for chunk in gen:
        if chunk.strip():
            _record_recent_output(chunk)
            collected.append(chunk.strip())
            # Telegram limit is 4096; executor already caps at 4000
            await update.message.reply_text(chunk[:4000])
    # Persist a summary of the output for conversation history.
    if collected:
        tail = "\n".join(collected)[-2000:]
        try:
            await db.log_conversation("assistant", tail)
        except Exception:
            logger.debug("Failed to log streamed output to DB", exc_info=True)


def _record_recent_output(text: str) -> None:
    """Store recent command output for prompt injection."""
    cleaned = text.strip()
    if cleaned:
        _RECENT_OUTPUT_LINES.append(cleaned)


def _get_recent_output_tail() -> str:
    """Return the most recent command output tail, truncated for prompt safety."""
    if not _RECENT_OUTPUT_LINES:
        return "Recent command output: none"

    tail = "\n".join(_RECENT_OUTPUT_LINES)
    if len(tail) > _RECENT_OUTPUT_MAX_CHARS:
        tail = tail[-_RECENT_OUTPUT_MAX_CHARS:]
        tail = "[...recent output truncated...]\n" + tail
    return f"Recent command output:\n{tail}"


def _load_project_context() -> str:
    """Return the project context file content, capped for prompt safety."""
    try:
        text = _PROJECT_CONTEXT_PATH.read_text(encoding="utf-8", errors="replace").strip()
    except FileNotFoundError:
        logger.warning("Project context file not found at %s", _PROJECT_CONTEXT_PATH)
        return ""
    except OSError as exc:
        logger.warning("Could not read project context file %s: %s", _PROJECT_CONTEXT_PATH, exc)
        return ""

    if len(text) > _PROJECT_CONTEXT_MAX_CHARS:
        return text[:_PROJECT_CONTEXT_MAX_CHARS] + "\n\n[...project context truncated...]"
    return text


def _format_live_context() -> str:
    """Build a compact snapshot of the current bot/session state for /ask."""
    processes = process_registry.list_processes()
    schedules = scheduler.list_schedules()
    watches = watchdog.list_watches()

    lines = [
        "Live session context:",
        "",
        _format_process_snapshot(processes),
        "",
        _format_schedule_snapshot(schedules),
        "",
        _format_watch_snapshot(watches),
        "",
        _get_recent_output_tail(),
        "",
        _format_system_snapshot(),
    ]

    return "\n".join(line for line in lines if line).strip()


def _format_process_snapshot(processes: list[dict]) -> str:
    """Format active subprocesses for prompt injection."""
    if not processes:
        return "Running processes: none"

    lines = ["Running processes:"]
    for proc in processes:
        runtime = _format_runtime(proc["runtime_seconds"])
        status = "running" if proc["returncode"] is None else f"exited({proc['returncode']})"
        lines.append(f"- PID {proc['pid']}: {proc['alias']} ({runtime}, {status})")
    return "\n".join(lines)


def _format_schedule_snapshot(schedules: list[dict]) -> str:
    """Format pending schedules for prompt injection."""
    if not schedules:
        return "Pending schedules: none"

    lines = ["Pending schedules:"]
    for schedule_entry in schedules:
        lines.append(f"- {schedule_entry['id']}: {schedule_entry['alias']} at {schedule_entry['run_at']}")
    return "\n".join(lines)


def _format_watch_snapshot(watches: list[dict]) -> str:
    """Format active watchers for prompt injection."""
    if not watches:
        return "Active watchers: none"

    lines = ["Active watchers:"]
    for watch_entry in watches:
        lines.append(f"- {watch_entry['id']}: {watch_entry['type']} -> {watch_entry['target']}")
    return "\n".join(lines)


def _format_system_snapshot() -> str:
    """Format a compact live CPU/RAM/GPU snapshot for prompt injection."""
    cpu = psutil.cpu_percent(interval=0.0)
    memory = psutil.virtual_memory()

    lines = [
        f"System snapshot: CPU {cpu:.1f}% | RAM {memory.percent:.1f}%",
    ]

    try:
        import GPUtil

        gpus = GPUtil.getGPUs()
        temperatures = [float(gpu.temperature) for gpu in gpus if getattr(gpu, "temperature", None) is not None]
        if temperatures:
            lines.append(f"GPU temp: {max(temperatures):.1f}°C")
    except Exception:
        # GPU data is optional; failure here should not block /ask.
        pass

    return " | ".join(lines)


async def _format_conversation_history() -> str:
    """Fetch recent conversation log from DB and format for prompt injection."""
    try:
        rows = await db.get_recent_conversations(10)
    except Exception:
        logger.debug("Failed to fetch conversation history from DB", exc_info=True)
        return ""
    if not rows:
        return ""
    lines = ["Recent conversation history:"]
    for row in rows:
        role = row["role"].capitalize()
        content = row["content"][:300]
        lines.append(f"  [{role}] {content}")
    return "\n".join(lines)


async def _build_ask_system_prompt() -> str:
    """Build the system prompt used by /ask and voice translation."""
    scripts_info = "\n".join(
        f"- {alias}: {cfg.get('path', 'N/A')}"
        for alias, cfg in _SCRIPTS_CONFIG.items()
    ) or "- (none configured)"

    project_context = _load_project_context()
    live_context = _format_live_context()
    conversation_context = await _format_conversation_history()

    # Observer machine context (optional — degrades gracefully if not ready)
    observer_context = ""
    try:
        from bot import observer
        observer_context = observer.get_current_context()
    except Exception:
        pass

    # Session memory (optional — degrades gracefully)
    session_context = ""
    try:
        from bot import memory
        session_context = await memory.get_recent_sessions(3)
    except Exception:
        pass

    sections = [
        "You are a command translator for a Telegram bot called Kira.",
        "Your job is to convert natural language requests into exact Kira commands.",
        "",
        "Available commands:",
        "- /run <alias> - Execute a script from scripts.toml",
        "- /run <alias> <args> - Execute with extra arguments",
        "- /shell <command> - Run a shell command",
        "- /status - List running processes",
        "- /kill <pid> - Kill a process",
        "- /schedule <alias> <time> - Schedule a script (e.g., 23:00, 30m, 2h)",
        "- /schedules - List pending schedules",
        "- /sysinfo - Show system info (CPU, RAM, GPU)",
        "- /screenshot - Take a screenshot",
        "- /ls [path] - List directory",
        "- /find <pattern> [path] - Find files",
        "- /tail <path> [n] - Show last lines of file",
        "- /copy <text> - Copy to clipboard",
        "- /paste - Get clipboard content",
        "- /open <app> - Open an application by name",
        "- /sleep - Put PC to sleep",
        "- /shutdown <minutes> - Schedule shutdown",
        "- /reboot <minutes> - Schedule reboot",
        "- /watch pid <pid> - Watch a process",
        "- /watch file <path> - Watch a file",
        "- /watches - List active watches",
        "- /remind <time> <message> - Set a reminder",
        "",
        "Available script aliases:",
        scripts_info,
        "",
        live_context,
    ]

    if observer_context:
        sections.extend([
            "",
            "Machine awareness (auto-updated every 15 min):",
            observer_context,
        ])

    if session_context:
        sections.extend([
            "",
            session_context,
        ])

    if conversation_context:
        sections.extend([
            "",
            conversation_context,
        ])

    if project_context:
        sections.extend([
            "",
            "Project context file:",
            project_context,
        ])

    sections.extend([
        "",
        "Instructions:",
        "1. Analyse the user's request.",
        "2. Map it to the single most appropriate Kira command.",
        "3. Return ONLY a JSON object — no prose, no markdown fences.",
        "",
        "Response format (examples):",
        '{"command": "/run", "args": ["crypto_train_explore", "--fee_mult", "10"]}',
        '{"command": "/shell", "args": ["dir C:\\\\Users"]}',
        '{"command": "/status", "args": []}',
    ])

    return _truncate_for_prompt("\n".join(sections), 24000)


def _truncate_for_prompt(text: str, max_chars: int) -> str:
    """Truncate prompt content while preserving a clear overflow marker."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[...context truncated...]"


def _build_ask_confirmation_text(command: str, args: list[str]) -> str:
    """Format the confirmation message for a proposed Kira command."""
    args_display = " ".join(args)
    return f"Proposed command:\n`{command} {args_display}`\n\nExecute?"


# ══════════════════════════════════════════════════════════════════
#  SCRIPT EXECUTION COMMANDS
# ══════════════════════════════════════════════════════════════════

@require_auth
async def handle_run(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/run <alias> [args...]`` — execute a script from scripts.toml."""
    if not context.args:
        await update.message.reply_text("Usage: /run <alias> [args...]")
        return

    alias = context.args[0]
    extra_args = context.args[1:]
    script = _get_script(alias)

    if script is None:
        available = ", ".join(_SCRIPTS_CONFIG.keys()) or "(none)"
        await update.message.reply_text(f"Unknown alias: {alias}\nAvailable: {available}")
        return

    timeout = script.get("timeout", _DEFAULT_TIMEOUT)
    checkpoint = script.get("checkpoint_interval")
    script_args = list(script.get("args", [])) + extra_args

    await update.message.reply_text(f"▶️ Running {alias}...")
    gen = executor.run_command(
        interpreter=script["interpreter"],
        script_path=script["path"],
        args=script_args,
        timeout=timeout,
        alias=alias,
        checkpoint_interval=checkpoint,
    )
    await _stream_to_chat(update, gen)


@require_auth
async def handle_shell(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/shell <command>`` — run arbitrary shell command via cmd.exe /c.

    Prompts for confirmation if the command contains destructive keywords.
    """
    if not context.args:
        await update.message.reply_text("Usage: /shell <command>")
        return

    command = " ".join(context.args)

    if _DESTRUCTIVE_PATTERNS.search(command):
        token = f"shell_confirm_{id(update)}"
        _PENDING_CONFIRMS[token] = {"command": command, "update": update, "context": context}

        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ Yes", callback_data=f"confirm_{token}"),
                InlineKeyboardButton("❌ Cancel", callback_data=f"cancel_{token}"),
            ]
        ])
        await update.message.reply_text(
            f"⚠️ Destructive command detected:\n`{command}`\n\nProceed?",
            reply_markup=keyboard,
            parse_mode="Markdown",
        )
        # Auto-cancel after timeout
        asyncio.get_event_loop().call_later(
            _CONFIRM_TIMEOUT,
            lambda t=token: _PENDING_CONFIRMS.pop(t, None),
        )
        return

    await _run_shell(update, command)


async def _run_shell(update: Update, command: str) -> None:
    """Actually execute a shell command and stream output."""
    timeout = int(os.environ.get("DEFAULT_TIMEOUT", "30"))
    await update.message.reply_text(f"▶️ Shell: `{command}`", parse_mode="Markdown")
    gen = executor.run_shell(command, timeout=timeout)
    await _stream_to_chat(update, gen)


@require_auth
async def handle_chain(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/chain <alias>`` — run a script and all chained scripts sequentially."""
    if not context.args:
        await update.message.reply_text("Usage: /chain <alias>")
        return

    alias = context.args[0]
    chain_list = _resolve_chain(alias)

    if chain_list is None:
        await update.message.reply_text(f"Unknown alias: {alias}")
        return

    await update.message.reply_text(f"⛓️ Chain: {' → '.join(chain_list)}")

    for step_alias in chain_list:
        script = _get_script(step_alias)
        if script is None:
            await update.message.reply_text(f"❌ Chain broken — unknown alias: {step_alias}")
            return

        await update.message.reply_text(f"▶️ Running {step_alias}...")
        timeout = script.get("timeout", _DEFAULT_TIMEOUT)
        checkpoint = script.get("checkpoint_interval")
        script_args = list(script.get("args", []))

        last_chunk = ""
        gen = executor.run_command(
            interpreter=script["interpreter"],
            script_path=script["path"],
            args=script_args,
            timeout=timeout,
            alias=step_alias,
            checkpoint_interval=checkpoint,
        )
        async for chunk in gen:
            last_chunk = chunk
            if chunk.strip():
                await update.message.reply_text(chunk[:4000])

        # If the last chunk indicates failure, stop the chain
        if "❌" in last_chunk or "⏰" in last_chunk:
            await update.message.reply_text(f"⛓️ Chain stopped at {step_alias} due to failure.")
            return

    await update.message.reply_text("⛓️ Chain completed successfully.")


def _resolve_chain(alias: str) -> list[str] | None:
    """Build the full chain list starting from the given alias."""
    script = _get_script(alias)
    if script is None:
        return None

    chain = [alias]
    current = script
    while current.get("chain"):
        next_alias = current["chain"][0] if isinstance(current["chain"], list) else current["chain"]
        if next_alias in chain:
            break  # prevent infinite loops
        chain.append(next_alias)
        current = _get_script(next_alias) or {}
    return chain


@require_auth
async def handle_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/status`` — list all running processes."""
    processes = process_registry.list_processes()
    if not processes:
        await update.message.reply_text("No running processes.")
        return

    lines = ["**Running processes:**\n"]
    for p in processes:
        runtime = _format_runtime(p["runtime_seconds"])
        status = "running" if p["returncode"] is None else f"exited({p['returncode']})"
        lines.append(f"• PID {p['pid']} — {p['alias']} — {runtime} — {status}")
    await update.message.reply_text("\n".join(lines))


@require_auth
async def handle_kill(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/kill <pid>`` — terminate a running process."""
    if not context.args:
        await update.message.reply_text("Usage: /kill <pid>")
        return

    try:
        pid = int(context.args[0])
    except ValueError:
        await update.message.reply_text("PID must be an integer.")
        return

    result = await process_registry.kill(pid)
    await update.message.reply_text(result)


# ── Schedule commands ─────────────────────────────────────────────

async def _scheduled_run_callback(alias: str) -> None:
    """Module-level run callback used by scheduler.reload_from_db() on restart."""
    script = _get_script(alias)
    if script is None:
        await notifier.send(f"❌ Scheduled run failed — alias {alias} not found in scripts.toml.")
        return
    timeout = script.get("timeout", _DEFAULT_TIMEOUT)
    checkpoint = script.get("checkpoint_interval")
    args = list(script.get("args", []))
    gen = executor.run_command(
        interpreter=script["interpreter"],
        script_path=script["path"],
        args=args,
        timeout=timeout,
        alias=alias,
        checkpoint_interval=checkpoint,
    )
    output_lines = []
    async for chunk in gen:
        output_lines.append(chunk)
    full_output = "\n".join(output_lines)
    if full_output.strip():
        await notifier.send(f"📋 Output from scheduled {alias}:\n{full_output[-3000:]}")


@require_auth
async def handle_schedule(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/schedule <alias> <HH:MM|Xm|Xh>`` — queue a script for later."""
    if not context.args or len(context.args) < 2:
        await update.message.reply_text("Usage: /schedule <alias> <HH:MM|Xm|Xh>")
        return

    alias = context.args[0]
    time_spec = context.args[1]

    if _get_script(alias) is None:
        await update.message.reply_text(f"Unknown alias: {alias}")
        return

    run_at = _parse_time_spec(time_spec)
    if run_at is None:
        await update.message.reply_text("Invalid time. Use HH:MM, Xm, or Xh.")
        return

    sid = await scheduler.schedule(alias, run_at, _scheduled_run_callback)
    await update.message.reply_text(f"✅ Scheduled {alias} at {run_at.strftime('%H:%M:%S')} (ID: {sid})")


@require_auth
async def handle_schedules(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/schedules`` — list pending scheduled runs."""
    items = scheduler.list_schedules()
    if not items:
        await update.message.reply_text("No pending scheduled runs.")
        return
    lines = ["**Pending schedules:**\n"]
    for s in items:
        lines.append(f"• {s['id']} — {s['alias']} at {s['run_at']}")
    await update.message.reply_text("\n".join(lines))


@require_auth
async def handle_unschedule(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/unschedule <id>`` — cancel a pending scheduled run."""
    if not context.args:
        await update.message.reply_text("Usage: /unschedule <id>")
        return
    result = scheduler.cancel(context.args[0])
    await update.message.reply_text(result)


# ── System info ───────────────────────────────────────────────────

@require_auth
async def handle_sysinfo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/sysinfo`` — CPU %, RAM, GPU, disk."""
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("C:\\")

    lines = [
        f"🖥️ **System Info**\n",
        f"CPU:  {cpu}%",
        f"RAM:  {mem.used / (1024**3):.1f} / {mem.total / (1024**3):.1f} GB ({mem.percent}%)",
        f"Disk: {disk.free / (1024**3):.1f} GB free / {disk.total / (1024**3):.1f} GB ({disk.percent}%)",
    ]

    # GPU info — optional, NVIDIA only
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            lines.append(
                f"GPU:  {gpu.name} — {gpu.memoryUsed:.0f}/{gpu.memoryTotal:.0f} MB VRAM — {gpu.temperature}°C"
            )
    except Exception:
        lines.append("GPU:  No GPU detected or GPUtil unavailable.")

    await update.message.reply_text("\n".join(lines))


# ── File transfer ─────────────────────────────────────────────────

# Telegram bot API hard limit for file downloads is 20 MB for regular bots.
# Files larger than this cannot be downloaded via the bot API.
_TG_MAX_DOWNLOAD_MB = 20

def _format_size(n_bytes: int) -> str:
    """Return a human-readable file size string."""
    if n_bytes < 1024:
        return f"{n_bytes} B"
    if n_bytes < 1024 ** 2:
        return f"{n_bytes / 1024:.1f} KB"
    if n_bytes < 1024 ** 3:
        return f"{n_bytes / (1024 ** 2):.1f} MB"
    return f"{n_bytes / (1024 ** 3):.1f} GB"


@require_auth
async def handle_getfile(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/getfile <path>`` — send any file from the PC to your phone.

    Shows file size before sending. Telegram bots can send files up to 50 MB.
    For larger files the bot will warn and abort rather than hanging.
    """
    if not context.args:
        await update.message.reply_text(
            "Usage: /getfile <path>\n"
            "Example: /getfile C:/Users/Me/report.pdf"
        )
        return

    filepath = Path(" ".join(context.args))

    if not filepath.exists():
        await update.message.reply_text(f"❌ File not found: {filepath}")
        return
    if not filepath.is_file():
        await update.message.reply_text(
            f"❌ That path is a directory, not a file: {filepath}\n"
            "Use /ls to browse, then /getfile with a full file path."
        )
        return

    size_bytes = filepath.stat().st_size
    size_str = _format_size(size_bytes)

    # Telegram bot API upload limit is 50 MB
    if size_bytes > 50 * 1024 * 1024:
        await update.message.reply_text(
            f"❌ File too large to send via Telegram bot API.\n"
            f"  Size: {size_str} (limit: 50 MB)\n"
            f"  File: {filepath.name}"
        )
        return

    await update.message.reply_text(
        f"📤 Sending {filepath.name} ({size_str})..."
    )

    try:
        with open(filepath, "rb") as fh:
            await update.message.reply_document(
                document=fh,
                filename=filepath.name,
                caption=f"📄 {filepath.name}\n📦 {size_str}\n📁 {filepath.parent}",
            )
        logger.info(
            "getfile: sent %s (%s) to user %s",
            filepath, size_str, update.effective_user.id,
        )
    except Exception as exc:
        await update.message.reply_text(f"❌ Failed to send file: {exc}")


@require_auth
async def handle_putfile(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/putfile [path]`` — save a file sent from your phone to the PC.

    Two ways to use:
      1. Send a file to the bot with the caption ``/putfile C:/save/here/name.ext``
         (the path is the full destination including filename).
      2. Send ``/putfile [path]`` as a reply to a previously sent file message.

    If no path is given, the file is saved to the Downloads folder inside the
    project directory (``downloads/<original_filename>``).
    """
    msg = update.message

    # Resolve the attachment — accept document, photo, video, or audio
    attachment = (
        msg.document
        or msg.video
        or msg.audio
        or msg.voice
        or msg.animation
        or (msg.photo[-1] if msg.photo else None)
    )

    # If the message itself has no attachment, look in the replied-to message
    source_msg = msg
    if attachment is None and msg.reply_to_message:
        source_msg = msg.reply_to_message
        attachment = (
            source_msg.document
            or source_msg.video
            or source_msg.audio
            or source_msg.voice
            or source_msg.animation
            or (source_msg.photo[-1] if source_msg.photo else None)
        )

    if attachment is None:
        await msg.reply_text(
            "No file found. Two ways to use /putfile:\n\n"
            "1. Send a file to the bot with caption:\n"
            "   /putfile C:/path/to/save/filename.ext\n\n"
            "2. Reply to a file message with:\n"
            "   /putfile C:/path/to/save/filename.ext"
        )
        return

    # Determine original filename (photos don't have one)
    original_name: str
    file_size: int | None = None
    if hasattr(attachment, "file_name") and attachment.file_name:
        original_name = attachment.file_name
    elif hasattr(attachment, "file_unique_id"):
        # Fallback: derive extension from mime_type if available
        ext = ""
        if hasattr(attachment, "mime_type") and attachment.mime_type:
            ext = "." + attachment.mime_type.split("/")[-1]
        original_name = f"received_{attachment.file_unique_id}{ext}"
    else:
        original_name = "received_file"

    if hasattr(attachment, "file_size"):
        file_size = attachment.file_size

    # Check Telegram bot API download limit (20 MB)
    if file_size and file_size > _TG_MAX_DOWNLOAD_MB * 1024 * 1024:
        await msg.reply_text(
            f"❌ File too large to download via bot API.\n"
            f"  Size: {_format_size(file_size)} (limit: {_TG_MAX_DOWNLOAD_MB} MB)"
        )
        return

    # Resolve save path
    # Priority: args from /putfile command > caption of the source message
    path_str = ""
    if context.args:
        path_str = " ".join(context.args)
    elif source_msg.caption:
        # Strip the /putfile command from the caption if present
        cap = source_msg.caption.strip()
        if cap.lower().startswith("/putfile"):
            path_str = cap[len("/putfile"):].strip()

    if path_str:
        save_path = Path(path_str)
        # If the path looks like a directory (no suffix), append the filename
        if not save_path.suffix and not save_path.exists():
            save_path = save_path / original_name
        elif save_path.is_dir():
            save_path = save_path / original_name
    else:
        # Default: project-root/downloads/<original_filename>
        downloads_dir = Path(__file__).resolve().parent.parent / "downloads"
        save_path = downloads_dir / original_name

    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Warn if a file already exists at that path
    if save_path.exists():
        await msg.reply_text(
            f"⚠️ File already exists at {save_path} — overwriting."
        )

    size_str = _format_size(file_size) if file_size else "unknown size"
    await msg.reply_text(f"📥 Receiving {original_name} ({size_str})...")

    try:
        tg_file = await attachment.get_file()
        await tg_file.download_to_drive(str(save_path))
        final_size = _format_size(save_path.stat().st_size)
        await msg.reply_text(
            f"✅ Saved to:\n{save_path}\n\n"
            f"📦 Size: {final_size}"
        )
        logger.info(
            "putfile: saved %s (%s) from user %s",
            save_path, final_size, update.effective_user.id,
        )
    except Exception as exc:
        await msg.reply_text(f"❌ Failed to save file: {exc}")


# ── Filesystem commands ───────────────────────────────────────────

@require_auth
async def handle_cd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/cd <path>`` — change the bot's working directory.

    Supports:
      /cd ..               — go up one level
      /cd ~                — go to user home directory
      /cd logs             — relative path from current directory
      /cd C:/Users/Me      — absolute path
      /cd                  — print current directory (no argument)
    """
    if not context.args:
        await update.message.reply_text(f"📍 Current directory:\n{_CWD}")
        return

    raw = " ".join(context.args)

    # Resolve ~ to the user's home directory
    if raw == "~" or raw.startswith("~/") or raw.startswith("~\\"):
        target = Path.home() / raw[2:].lstrip("/\\")
    else:
        candidate = Path(raw)
        if candidate.is_absolute():
            target = candidate
        else:
            # Treat as relative to the current bot CWD
            target = _CWD / candidate

    # Normalise (resolves .., symlinks, etc.)
    try:
        target = target.resolve()
    except OSError as exc:
        await update.message.reply_text(f"❌ Cannot resolve path: {exc}")
        return

    if not target.exists():
        await update.message.reply_text(f"❌ Directory not found: {target}")
        return
    if not target.is_dir():
        await update.message.reply_text(f"❌ Not a directory: {target}")
        return

    set_cwd(target)
    logger.info("cd: CWD changed to %s by user %s", target, update.effective_user.id)
    await update.message.reply_text(f"📁 {target}")


@require_auth
async def handle_ls(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/ls [path]`` — list directory contents."""
    if context.args:
        raw = " ".join(context.args)
        candidate = Path(raw)
        target = candidate if candidate.is_absolute() else _CWD / candidate
    else:
        target = _CWD
    if not target.exists():
        await update.message.reply_text(f"Path not found: {target}")
        return

    try:
        entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        lines = [f"📁 {target}\n"]
        for entry in entries[:100]:
            prefix = "📁" if entry.is_dir() else "📄"
            size = ""
            if entry.is_file():
                size_bytes = entry.stat().st_size
                if size_bytes < 1024:
                    size = f" ({size_bytes} B)"
                elif size_bytes < 1024 ** 2:
                    size = f" ({size_bytes / 1024:.1f} KB)"
                else:
                    size = f" ({size_bytes / (1024**2):.1f} MB)"
            lines.append(f"{prefix} {entry.name}{size}")

        if len(list(target.iterdir())) > 100:
            lines.append(f"\n... and {len(list(target.iterdir())) - 100} more")

        await update.message.reply_text("\n".join(lines)[:4000])
    except PermissionError:
        await update.message.reply_text(f"❌ Permission denied: {target}")


@require_auth
async def handle_find(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/find <pattern> [path]`` — find files matching a glob pattern."""
    if not context.args:
        await update.message.reply_text("Usage: /find <pattern> [path]")
        return

    pattern = context.args[0]
    search_root = Path(context.args[1]) if len(context.args) > 1 else Path("C:\\")

    if not search_root.exists():
        await update.message.reply_text(f"Path not found: {search_root}")
        return

    matches = []
    try:
        for match in search_root.rglob(pattern):
            matches.append(str(match))
            if len(matches) >= 50:
                break
    except PermissionError:
        pass

    if not matches:
        await update.message.reply_text(f"No files matching `{pattern}` in {search_root}")
        return

    result = "\n".join(matches)
    header = f"🔍 Found {len(matches)} match(es) for `{pattern}`:\n\n"
    await update.message.reply_text((header + result)[:4000])


@require_auth
async def handle_tail(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/tail <path> [n]`` — last N lines of a file (default 20)."""
    if not context.args:
        await update.message.reply_text("Usage: /tail <path> [n]")
        return

    n = 20
    # Check if last arg is a number
    args = list(context.args)
    if len(args) >= 2:
        try:
            n = int(args[-1])
            args = args[:-1]
        except ValueError:
            pass

    filepath = Path(" ".join(args))
    if not filepath.exists():
        await update.message.reply_text(f"File not found: {filepath}")
        return

    try:
        lines = filepath.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = lines[-n:]
        result = "\n".join(tail)
        await update.message.reply_text(f"📄 Last {len(tail)} lines of {filepath.name}:\n\n{result}"[:4000])
    except Exception as exc:
        await update.message.reply_text(f"❌ Error reading file: {exc}")


@require_auth
async def handle_mkdir(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/mkdir <path>`` — create a directory (parents included)."""
    if not context.args:
        await update.message.reply_text("Usage: /mkdir <path>")
        return
    target = Path(" ".join(context.args))
    target.mkdir(parents=True, exist_ok=True)
    await update.message.reply_text(f"✅ Created: {target}")


@require_auth
async def handle_move(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/move <src> <dst>`` — move a file or directory."""
    if not context.args or len(context.args) < 2:
        await update.message.reply_text("Usage: /move <src> <dst>")
        return
    src = Path(context.args[0])
    dst = Path(context.args[1])
    if not src.exists():
        await update.message.reply_text(f"Source not found: {src}")
        return
    try:
        shutil.move(str(src), str(dst))
        await update.message.reply_text(f"✅ Moved {src} → {dst}")
    except Exception as exc:
        await update.message.reply_text(f"❌ Move failed: {exc}")


# ── Clipboard ─────────────────────────────────────────────────────

@require_auth
async def handle_copy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/copy <text>`` — set PC clipboard."""
    if not context.args:
        await update.message.reply_text("Usage: /copy <text>")
        return
    text = " ".join(context.args)
    pyperclip.copy(text)
    await update.message.reply_text(f"✅ Copied to clipboard ({len(text)} chars)")


@require_auth
async def handle_paste(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/paste`` — send back clipboard contents."""
    content = pyperclip.paste()
    if not content:
        await update.message.reply_text("Clipboard is empty.")
        return
    await update.message.reply_text(f"📋 Clipboard:\n{content[:4000]}")


# ── Screenshot ────────────────────────────────────────────────────

@require_auth
async def handle_screenshot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/screenshot [n]`` — capture screen and send as image."""
    monitor_index = None
    if context.args:
        try:
            monitor_index = int(context.args[0])
        except ValueError:
            await update.message.reply_text("Usage: /screenshot [monitor_number]")
            return

    try:
        with mss.mss() as sct:
            if monitor_index is not None:
                if monitor_index >= len(sct.monitors) - 1:
                    await update.message.reply_text(
                        f"Monitor {monitor_index} not found. Available: 0-{len(sct.monitors) - 2}"
                    )
                    return
                # monitors[0] is the virtual screen; real monitors start at 1
                monitor = sct.monitors[monitor_index + 1]
            else:
                monitor = sct.monitors[0]  # full virtual screen

            screenshot = sct.grab(monitor)
            png_bytes = mss.tools.to_png(screenshot.rgb, screenshot.size)

        bio = io.BytesIO(png_bytes)
        bio.name = "screenshot.png"
        await update.message.reply_photo(photo=bio)
        logger.info("Screenshot sent to user %s", update.effective_user.id)
    except Exception as exc:
        await update.message.reply_text(f"❌ Screenshot failed: {exc}")


# ── Power commands ────────────────────────────────────────────────

@require_auth
async def handle_sleep(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/sleep`` — put PC to sleep (with confirmation)."""
    await _power_confirm(update, "sleep", "Put PC to sleep?")


@require_auth
async def handle_shutdown(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/shutdown <minutes>`` — schedule shutdown."""
    if not context.args:
        await update.message.reply_text("Usage: /shutdown <minutes>")
        return
    try:
        minutes = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Minutes must be an integer.")
        return
    await _power_confirm(update, f"shutdown_{minutes}", f"Shutdown in {minutes} minutes?")


@require_auth
async def handle_reboot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/reboot <minutes>`` — schedule reboot."""
    if not context.args:
        await update.message.reply_text("Usage: /reboot <minutes>")
        return
    try:
        minutes = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Minutes must be an integer.")
        return
    await _power_confirm(update, f"reboot_{minutes}", f"Reboot in {minutes} minutes?")


@require_auth
async def handle_abort_shutdown(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/abort_shutdown`` — cancel pending shutdown/reboot. No confirmation."""
    os.system("shutdown /a")
    await update.message.reply_text("✅ Shutdown/reboot cancelled.")


async def _power_confirm(update: Update, action: str, prompt: str) -> None:
    """Show an inline confirmation for power commands."""
    token = f"power_{action}_{id(update)}"
    _PENDING_CONFIRMS[token] = {"action": action, "update": update}

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Yes", callback_data=f"confirm_{token}"),
            InlineKeyboardButton("❌ Cancel", callback_data=f"cancel_{token}"),
        ]
    ])
    await update.message.reply_text(f"⚠️ {prompt}", reply_markup=keyboard)
    asyncio.get_event_loop().call_later(
        _CONFIRM_TIMEOUT,
        lambda t=token: _PENDING_CONFIRMS.pop(t, None),
    )


# ── Watchdog commands ─────────────────────────────────────────────

@require_auth
async def handle_watch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/watch pid <pid>`` or ``/watch file <path>``."""
    if not context.args or len(context.args) < 2:
        await update.message.reply_text("Usage: /watch pid <pid> | /watch file <path>")
        return

    watch_type = context.args[0].lower()
    target = " ".join(context.args[1:])

    if watch_type == "pid":
        try:
            pid = int(target)
        except ValueError:
            await update.message.reply_text("PID must be an integer.")
            return
        if not psutil.pid_exists(pid):
            await update.message.reply_text(f"No process with PID {pid}.")
            return
        wid = await watchdog.watch_pid(pid)
        await update.message.reply_text(f"👁️ Watching PID {pid} (ID: {wid})")

    elif watch_type == "file":
        result = await watchdog.watch_file(target)
        if result.startswith("File not found"):
            await update.message.reply_text(result)
        else:
            await update.message.reply_text(f"👁️ Watching file {target} (ID: {result})")
    else:
        await update.message.reply_text("Unknown watch type. Use: pid or file")


@require_auth
async def handle_watches(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/watches`` — list active watchdog monitors."""
    items = watchdog.list_watches()
    if not items:
        await update.message.reply_text("No active watchers.")
        return
    lines = ["**Active watchers:**\n"]
    for w in items:
        lines.append(f"• {w['id']} — {w['type']}: {w['target']}")
    await update.message.reply_text("\n".join(lines))


@require_auth
async def handle_unwatch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/unwatch <id>`` — remove a watchdog monitor."""
    if not context.args:
        await update.message.reply_text("Usage: /unwatch <id>")
        return
    result = watchdog.cancel(context.args[0])
    await update.message.reply_text(result)


# ── Reminder ──────────────────────────────────────────────────────

@require_auth
async def handle_remind(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/remind <Xm|Xh> <message>`` — send a reminder after delay."""
    if not context.args or len(context.args) < 2:
        await update.message.reply_text("Usage: /remind <Xm|Xh> <message>")
        return

    time_spec = context.args[0]
    message = " ".join(context.args[1:])

    delay = _parse_delay(time_spec)
    if delay is None:
        await update.message.reply_text("Invalid time. Use Xm or Xh (e.g. 30m, 2h).")
        return

    fire_at = datetime.now() + timedelta(seconds=delay)

    # Persist so the reminder survives a restart.
    reminder_id: int | None = None
    try:
        reminder_id = await db.save_reminder(fire_at.isoformat(), message)
    except Exception:
        logger.debug("Failed to persist reminder to DB", exc_info=True)

    await update.message.reply_text(f"⏰ Reminder set for {time_spec} from now.")

    async def _fire_reminder() -> None:
        await asyncio.sleep(delay)
        await notifier.send(f"🔔 Reminder: {message}")
        if reminder_id is not None:
            try:
                await db.mark_reminder_fired(reminder_id)
            except Exception:
                logger.debug("Failed to mark reminder %d as fired", reminder_id, exc_info=True)

    asyncio.create_task(_fire_reminder())


async def reload_reminders() -> None:
    """Restore pending reminders from the database after a restart.

    Called once from ``main.py`` after ``db.init_db()``.
    """
    try:
        pending = await db.get_pending_reminders()
    except Exception:
        logger.warning("Failed to reload reminders from DB", exc_info=True)
        return

    now = datetime.now()
    restored = 0
    for row in pending:
        try:
            fire_at = datetime.fromisoformat(row["fire_at"])
        except (ValueError, TypeError):
            logger.warning("Skipping reminder %d with invalid fire_at: %r", row["id"], row["fire_at"])
            await db.mark_reminder_fired(row["id"])
            continue

        delay = (fire_at - now).total_seconds()
        if delay <= 0:
            # Already past due — fire immediately.
            await notifier.send(f"🔔 Reminder (delayed): {row['message']}")
            await db.mark_reminder_fired(row["id"])
            restored += 1
            continue

        rid = row["id"]
        msg = row["message"]

        async def _fire(r_id: int = rid, r_msg: str = msg) -> None:
            await asyncio.sleep(delay)
            await notifier.send(f"🔔 Reminder: {r_msg}")
            try:
                await db.mark_reminder_fired(r_id)
            except Exception:
                logger.debug("Failed to mark reminder %d as fired", r_id, exc_info=True)

        asyncio.create_task(_fire())
        restored += 1

    if restored:
        logger.info("Restored %d pending reminder(s) from DB", restored)


# ── History & Runs ────────────────────────────────────────────────

@require_auth
async def handle_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/history [n]`` — show recent conversation history."""
    n = 20
    if context.args:
        try:
            n = int(context.args[0])
        except ValueError:
            await update.message.reply_text("Usage: /history [n]  (n = number of entries)")
            return

    try:
        rows = await db.get_recent_conversations(n)
    except Exception as exc:
        await update.message.reply_text(f"Failed to read history: {exc}")
        return

    if not rows:
        await update.message.reply_text("No conversation history yet.")
        return

    lines = [f"📜 Last {len(rows)} conversation entries:\n"]
    for row in rows:
        ts = row["timestamp"][:16] if row.get("timestamp") else "?"
        role = row["role"].upper()
        content = row["content"][:200]
        lines.append(f"[{ts}] {role}: {content}")

    await update.message.reply_text("\n".join(lines)[:4000])


@require_auth
async def handle_runs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/runs [alias] [n]`` — show recent run history with metrics."""
    alias = None
    limit = 10

    if context.args:
        # If the last arg is a number, treat it as the limit.
        args = list(context.args)
        if len(args) >= 1:
            try:
                limit = int(args[-1])
                args = args[:-1]
            except ValueError:
                pass
        if args:
            alias = args[0]

    try:
        rows = await db.get_run_history(alias=alias, limit=limit)
    except Exception as exc:
        await update.message.reply_text(f"Failed to read run history: {exc}")
        return

    if not rows:
        msg = f"No runs recorded for {alias}." if alias else "No runs recorded yet."
        await update.message.reply_text(msg)
        return

    header = f"📊 Last {len(rows)} run(s)"
    if alias:
        header += f" for {alias}"
    lines = [f"{header}:\n"]

    for row in rows:
        date = (row.get("finished_at") or row.get("started_at") or "?")[:16]
        code = row.get("exit_code")
        icon = "✅" if code == 0 else "❌" if code is not None else "?"
        runtime = row.get("runtime_seconds")
        runtime_str = _format_runtime(runtime) if runtime else "?"

        parts = [f"{icon} {row['alias']} ({date}) — {runtime_str}"]

        reward = row.get("reward")
        if reward is not None:
            parts.append(f"  reward={reward:.4f}")
        loss = row.get("loss")
        if loss is not None:
            parts.append(f"  loss={loss}")
        steps = row.get("total_timesteps")
        if steps is not None:
            parts.append(f"  steps={steps:,}")

        lines.append("  ".join(parts))

    await update.message.reply_text("\n".join(lines)[:4000])


# ── Memory: session summary & recall ─────────────────────────────

@require_auth
async def handle_summarise(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/summarise`` — generate and save a GPT summary of today's activity."""
    from bot import memory
    await update.message.reply_text("⏳ Summarising today's activity...")
    try:
        summary = await memory.summarise_today()
        await update.message.reply_text(f"📋 Today's summary:\n\n{summary}")
    except Exception as exc:
        logger.error("handle_summarise failed: %s", exc)
        await update.message.reply_text(f"❌ Summarisation failed: {exc}")


@require_auth
async def handle_recall(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/recall <query>`` — ask a natural-language question over recent session history."""
    from bot import memory, db

    query = " ".join(context.args) if context.args else ""
    if not query:
        await update.message.reply_text("Usage: /recall <your question about past sessions>")
        return

    await update.message.reply_text("🔍 Searching session history...")

    try:
        rows = await db.get_recent_sessions(7)
    except Exception as exc:
        await update.message.reply_text(f"❌ Failed to fetch session history: {exc}")
        return

    if not rows:
        await update.message.reply_text("No session history found yet. Run /summarise after some activity.")
        return

    # Build context block from stored summaries
    session_block = "\n".join(
        f"[{r.get('date', '?')}] {r.get('summary', '')}"
        for r in rows
    )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        await update.message.reply_text("❌ OPENAI_API_KEY not set.")
        return

    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are answering questions about a developer's past work sessions. "
                        "Answer concisely based only on the provided session summaries."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Session summaries:\n\n{session_block}\n\nQuestion: {query}",
                },
            ],
            max_tokens=300,
            temperature=0.3,
        )
        answer = response.choices[0].message.content.strip()
        await update.message.reply_text(answer)
    except Exception as exc:
        logger.error("handle_recall GPT call failed: %s", exc)
        await update.message.reply_text(f"❌ Recall failed: {exc}")


# ── Help ──────────────────────────────────────────────────────────

@require_auth
async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/help`` — list all commands and script aliases."""
    aliases = ", ".join(_SCRIPTS_CONFIG.keys()) or "(none configured)"
    alias_lines = "\n".join(f"- `{name}`" for name in _SCRIPTS_CONFIG.keys()) or "- `(none configured)`"
    help_text = (
        "**kira commands:**\n\n"
        "**Script Execution**\n"
        "/run <alias> [args] — Run a script\n"
        "/shell <command> — Run shell command\n"
        "/status — List running processes\n"
        "/kill <pid> — Kill a process\n"
        "/chain <alias> — Run with chained scripts\n"
        "/schedule <alias> <time> — Schedule a run\n"
        "/schedules — List pending schedules\n"
        "/unschedule <id> — Cancel a schedule\n\n"
        "**Natural Language**\n"
        "/ask <request> — Translate plain English to commands\n\n"
        "**System**\n"
        "/sysinfo — CPU, RAM, GPU, disk\n"
        "/getfile <path> — Send file from PC to phone (up to 50 MB)\n"
        "/putfile [path] — Save file from phone to PC\n"
        "  · Send file with caption: /putfile C:/dest/name.ext\n"
        "  · Or reply to a file msg: /putfile C:/dest/name.ext\n"
        "  · No path = saved to downloads/ folder\n\n"
        "**Filesystem**\n"
        "/cd [path] — Change directory (.. and ~ supported)\n"
        "/ls [path] — List current or given directory\n"
        "/find <pattern> [path] — Find files\n"
        "/tail <path> [n] — Tail a file\n"
        "/mkdir <path> — Create directory\n"
        "/move <src> <dst> — Move file/dir\n\n"
        "**Clipboard**\n"
        "/copy <text> — Set clipboard\n"
        "/paste — Get clipboard\n\n"
        "**Application Management**\n"
        "/list_apps — List running applications\n"
        "/open <app> — Open an application by name\n"
        "/close_apps <app1> [app2] — Close applications\n\n"
        "**Power**\n"
        "/sleep — Sleep PC\n"
        "/shutdown <min> — Schedule shutdown\n"
        "/reboot <min> — Schedule reboot\n"
        "/abort_shutdown — Cancel shutdown\n\n"
        "**Watchdog**\n"
        "/watch pid <pid> — Watch process\n"
        "/watch file <path> — Watch file\n"
        "/watches — List watchers\n"
        "/unwatch <id> — Remove watcher\n\n"
        "**History & Runs**\n"
        "/history [n] — Show conversation history\n"
        "/runs [alias] [n] — Show run history with metrics\n\n"
        "**Misc**\n"
        "/remind <Xm|Xh> <msg> — Set reminder (persists across restarts)\n"
        "/help — This message\n\n"
        f"**Script aliases:**\n{alias_lines}"
    )
    await update.message.reply_text(help_text[:4000])


# ── Callback query handler for inline confirmations ──────────────

async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process Yes/Cancel inline button presses for confirmations."""
    query = update.callback_query
    await query.answer()

    data = query.data or ""

    if data.startswith("confirm_"):
        token = data[len("confirm_"):]
        entry = _PENDING_CONFIRMS.pop(token, None)
        if entry is None:
            await query.edit_message_text("⏰ Confirmation expired.")
            return

        # Shell confirmation
        if "command" in entry:
            await query.edit_message_text(f"✅ Executing: `{entry['command']}`", parse_mode="Markdown")
            await _run_shell(entry["update"], entry["command"])
            return

        # Power confirmation
        if "action" in entry:
            action = entry["action"]
            if action == "sleep":
                await query.edit_message_text("💤 Putting PC to sleep...")
                os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
            elif action.startswith("shutdown_"):
                minutes = int(action.split("_")[1])
                seconds = minutes * 60
                await query.edit_message_text(f"🔴 Shutdown in {minutes} minutes.")
                os.system(f"shutdown /s /t {seconds}")
            elif action.startswith("reboot_"):
                minutes = int(action.split("_")[1])
                seconds = minutes * 60
                await query.edit_message_text(f"🔄 Reboot in {minutes} minutes.")
                os.system(f"shutdown /r /t {seconds}")
            return

    elif data.startswith("cancel_"):
        token = data[len("cancel_"):]
        _PENDING_CONFIRMS.pop(token, None)
        await query.edit_message_text("❌ Cancelled.")


# ── Application Management Commands ────────────────────────────────

def _is_system_process(process_name: str, executable_path: str) -> bool:
    """Check if a process is a Windows system process."""
    # System process names to exclude
    system_names = {
        'explorer.exe', 'winlogon.exe', 'csrss.exe', 'lsass.exe',
        'services.exe', 'svchost.exe', 'system', 'idle', 'smss.exe',
        'dwm.exe', 'conhost.exe', 'spoolsv.exe', 'taskmgr.exe'
    }
    
    # System paths to exclude
    system_paths = {
        'c:\\windows\\system32\\',
        'c:\\windows\\syswow64\\',
        'c:\\windows\\',
        'c:\\program files\\windows defender\\',
        'c:\\program files (x86)\\windows defender\\'
    }
    
    # Check by name
    if process_name in system_names:
        return True
    
    # Check by path
    if executable_path:
        for sys_path in system_paths:
            if executable_path.startswith(sys_path):
                return True
    
    return False


def _normalize_app_name(raw_name: str) -> str | None:
    """Return a sanitized app name, or ``None`` if the value looks like a path."""
    cleaned = raw_name.strip().strip('"').strip("'").strip()
    if not cleaned:
        return None
    if any(sep in cleaned for sep in ("\\", "/", ":")):
        return None
    return cleaned


def _open_app_by_name(app_name: str) -> str:
    """Launch an application by its name only.

    The command intentionally rejects path-like input so /open stays name-only.
    """
    normalized = _normalize_app_name(app_name)
    if normalized is None:
        return "Usage: /open <app name>\nExample: /open notepad"

    try:
        subprocess.Popen(
            ["cmd", "/c", "start", "", normalized],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("open: launched app by name %r", normalized)
        return f"✅ Opening {normalized}..."
    except Exception as exc:
        logger.error("open: failed to launch %r: %s", normalized, exc, exc_info=True)
        return f"❌ Failed to open {normalized}: {exc}"


@require_auth
async def handle_list_apps(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/list_apps`` — list running installed applications."""
    apps = {}
    
    for proc in psutil.process_iter(['pid', 'name', 'exe']):
        proc_name = proc.info['name'].lower()
        proc_exe = proc.info.get('exe', '') if proc.info.get('exe') else ''
        
        # Skip system processes
        if _is_system_process(proc_name, proc_exe.lower()):
            continue
        
        # Group by application name (use executable name if available)
        if proc_exe:
            app_name = Path(proc_exe).stem.lower()
        else:
            app_name = proc_name
        
        apps.setdefault(app_name, []).append({
            'pid': proc.info['pid'],
            'name': proc.info['name']
        })
    
    lines = ["📋 Running applications:\n"]
    for app_name, processes in sorted(apps.items()):
        if len(processes) == 1:
            lines.append(f"• {app_name} (PID {processes[0]['pid']})")
        else:
            pids = [p['pid'] for p in processes]
            lines.append(f"• {app_name} ({len(processes)} instances: {', '.join(map(str, pids))})")
    
    if not apps:
        lines.append("No user applications running.")
    
    await update.message.reply_text("\n".join(lines)[:4000])


@require_auth
async def handle_open(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/open <app>`` — open an application by name only."""
    if not context.args:
        await update.message.reply_text("Usage: /open <app name>\nExample: /open notepad")
        return

    app_name = " ".join(context.args)
    result = _open_app_by_name(app_name)
    await update.message.reply_text(result)


@require_auth
async def handle_close_apps(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/close_apps`` — close specified installed applications."""
    if not context.args:
        await update.message.reply_text("Usage: /close_apps <app1> [app2]...")
        return
    
    apps_to_close = context.args
    closed_count = 0
    failed_count = 0
    
    lines = [f"🔄 Closing applications:\n"]
    
    for app_name in apps_to_close:
        try:
            # Find processes matching the app name
            matching_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'exe']):
                proc_name = proc.info['name'].lower()
                proc_exe = proc.info.get('exe', '').lower() if proc.info.get('exe') else ''
                
                # Match by process name or executable path
                if (app_name.lower() in proc_name or 
                    app_name.lower() in proc_exe):
                    
                    # Skip system processes
                    if _is_system_process(proc_name, proc_exe):
                        continue
                        
                    matching_processes.append(proc)
            
            if not matching_processes:
                lines.append(f"❌ {app_name} - no running instances found")
                failed_count += 1
                continue
            
            # Close all matching processes
            for proc in matching_processes:
                proc.terminate()
                proc_name = proc.info['name']
                lines.append(f"✅ {app_name} - closed {proc_name} (PID {proc.pid})")
                closed_count += 1
                
        except psutil.NoSuchProcess:
            lines.append(f"⚠️ {app_name} - process already terminated")
        except psutil.AccessDenied:
            lines.append(f"❌ {app_name} - access denied")
            failed_count += 1
        except Exception as exc:
            lines.append(f"❌ {app_name} - error: {exc}")
            failed_count += 1
    
    lines.append(f"\n📊 Summary: {closed_count} closed, {failed_count} failed")
    await update.message.reply_text("\n".join(lines))


# ── Helpers ───────────────────────────────────────────────────────

def _format_runtime(seconds: float) -> str:
    """Format seconds into human-readable runtime string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m"


def _parse_time_spec(spec: str) -> datetime | None:
    """Parse a time spec like ``14:30``, ``30m``, ``2h`` into a datetime."""
    # Relative: Xm or Xh
    delay = _parse_delay(spec)
    if delay is not None:
        return datetime.now() + timedelta(seconds=delay)

    # Absolute: HH:MM
    try:
        hour, minute = spec.split(":")
        target = datetime.now().replace(hour=int(hour), minute=int(minute), second=0, microsecond=0)
        if target < datetime.now():
            target += timedelta(days=1)
        return target
    except (ValueError, IndexError):
        return None


def _parse_delay(spec: str) -> float | None:
    """Parse ``30m`` or ``2h`` into seconds. Returns None if invalid."""
    spec = spec.strip().lower()
    if spec.endswith("m"):
        try:
            return float(spec[:-1]) * 60
        except ValueError:
            return None
    if spec.endswith("h"):
        try:
            return float(spec[:-1]) * 3600
        except ValueError:
            return None
    return None


# ══════════════════════════════════════════════════════════════════
#  NATURAL LANGUAGE COMMAND TRANSLATION (/ask)
# ══════════════════════════════════════════════════════════════════

# Telegram callback_data has a 64-byte limit, so args are serialised as
# a JSON list embedded in the callback token rather than space-joined strings.
# Format: "ask_yes|<command>|<json-args>"  e.g. ask_yes|/shell|["dir C:\\"]
_CB_MAX = 64


def _encode_ask_cb(command: str, args: list[str]) -> str:
    """Encode command + args into a callback_data string <= 64 bytes.

    Falls back to truncating the args JSON if it would exceed the limit.
    """
    args_json = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
    token = f"ask_yes|{command}|{args_json}"
    # Telegram limit is 64 bytes; truncate args gracefully if needed
    while len(token.encode()) > _CB_MAX and args:
        args = args[:-1]
        args_json = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
        token = f"ask_yes|{command}|{args_json}"
    return token


def _decode_ask_cb(data: str) -> tuple[str, list[str]]:
    """Decode callback_data back into (command, args).

    Returns ("", []) on any parse error.
    """
    try:
        _, command, args_json = data.split("|", 2)
        args = json.loads(args_json)
        if not isinstance(args, list):
            args = []
        return command, [str(a) for a in args]
    except Exception:
        return "", []


async def _ask_core(user_message: str) -> tuple[str, list[str]] | None:
    """Call GPT-4o Mini to translate natural language into a (command, args) pair.

    This is the shared brain behind both /ask and handle_voice. Extracting it
    here means zero duplication — both entry points call this one function and
    then present the result in their own way (text confirmation vs voice reply).

    Returns:
        (command, args) tuple on success, e.g. ("/run", ["crypto_train_full"])
        None if the model response could not be parsed or was empty.

    Raises:
        RuntimeError: If OPENAI_API_KEY is not set.
        openai.OpenAIError: On API errors.
    """
    from openai import AsyncOpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in .env")

    client = AsyncOpenAI(api_key=api_key)
    system_prompt = await _build_ask_system_prompt()

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        max_tokens=200,
    )

    result_text = response.choices[0].message.content.strip()

    # Strip accidental markdown fences
    result_text = re.sub(r"^```[a-z]*\n?", "", result_text)
    result_text = re.sub(r"\n?```$", "", result_text)
    result_text = result_text.strip()

    try:
        parsed = json.loads(result_text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                logger.warning("_ask_core: unparseable response: %r", result_text)
                return None
        else:
            logger.warning("_ask_core: no JSON found in response: %r", result_text)
            return None

    command = parsed.get("command", "").strip()
    args = [str(a) for a in parsed.get("args", [])]

    if not command:
        return None

    return command, args


@require_auth
async def handle_ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/ask <text>`` — translate natural language to a Kira command using GPT-4o Mini."""
    try:
        from openai import AsyncOpenAI  # noqa: F401 — ensure package present
    except ImportError:
        await update.message.reply_text("Error: openai package not installed. Run: pip install openai")
        return

    raw_text = update.message.text or ""
    user_message = raw_text.split(" ", 1)[1].strip() if " " in raw_text else ""

    if not user_message:
        await update.message.reply_text("Usage: /ask <your request in plain English>")
        return

    # Log the user's natural-language request for conversation history.
    try:
        await db.log_conversation("user", user_message)
    except Exception:
        logger.debug("Failed to log /ask user message to DB", exc_info=True)

    await update.message.reply_text("Thinking...")

    try:
        result = await _ask_core(user_message)
        if result is None:
            await update.message.reply_text(
                "Could not understand the request. Try being more specific."
            )
            return

        command, args = result
        confirmation_text = _build_ask_confirmation_text(command, args)

        # Log Kira's proposed command for conversation history.
        try:
            await db.log_conversation("assistant", f"{command} {' '.join(args)}")
        except Exception:
            logger.debug("Failed to log /ask response to DB", exc_info=True)

        callback_data = _encode_ask_cb(command, args)
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("Yes", callback_data=callback_data),
                InlineKeyboardButton("Cancel", callback_data="ask_cancel"),
            ]
        ])

        await update.message.reply_text(
            confirmation_text,
            reply_markup=keyboard,
            parse_mode="Markdown",
        )

    except Exception as exc:
        logger.error("handle_ask error: %s", exc, exc_info=True)
        await update.message.reply_text(f"Error: {exc}")


@require_auth
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming Telegram voice messages.

    Flow:
      1. Download the .ogg voice file from Telegram.
      2. Transcribe via OpenAI Whisper → plain text.
      3. Pass transcript through _ask_core → (command, args).
      4. Present an inline Yes/Cancel confirmation (same as /ask).
      5. When the user confirms, the existing handle_ask_callback executes it
         and Kira speaks the result back via OpenAI TTS.

    Why confirmation even for voice?
      Silent auto-execution from a voice command on a real machine is dangerous.
      The confirm step takes one tap and prevents misheard commands from doing
      something destructive. The confirmation message itself is also spoken back
      so the user knows what Kira understood.
    """
    msg = update.message

    # --- Step 1: Download the voice file ---
    try:
        tg_file = await msg.voice.get_file()
        ogg_bytes = await tg_file.download_as_bytearray()
    except Exception as exc:
        logger.error("handle_voice: download failed: %s", exc)
        await msg.reply_text("Failed to download voice message.")
        return

    # --- Step 2: Transcribe via Whisper ---
    try:
        transcript = await voice.transcribe(bytes(ogg_bytes))
    except Exception as exc:
        logger.error("handle_voice: transcription failed: %s", exc)
        await msg.reply_text("Voice transcription failed. Try again.")
        return

    if not transcript:
        await msg.reply_text("Couldn't make out what you said. Try again.")
        return

    logger.info(
        "handle_voice: user %s said: %r", update.effective_user.id, transcript
    )

    # Echo the transcript so the user can verify what Kira heard.
    await msg.reply_text(f"🎙️ Heard: _{transcript}_", parse_mode="Markdown")

    # Log the voice transcript for conversation history.
    try:
        await db.log_conversation("user", f"[voice] {transcript}")
    except Exception:
        logger.debug("Failed to log voice transcript to DB", exc_info=True)

    # --- Step 3: Translate transcript → command via GPT-4o Mini ---
    try:
        result = await _ask_core(transcript)
    except Exception as exc:
        logger.error("handle_voice: _ask_core failed: %s", exc)
        error_audio = await _safe_synthesise(f"Sorry, I ran into an error: {exc}")
        if error_audio:
            await _send_voice(msg, error_audio)
        return

    if result is None:
        response_text = "I couldn't map that to a command. Could you rephrase?"
        await msg.reply_text(response_text)
        audio = await _safe_synthesise(response_text)
        if audio:
            await _send_voice(msg, audio)
        return

    command, args = result
    args_display = " ".join(args)
    confirmation_text = f"Proposed command:\n`{command} {args_display}`\n\nExecute?"

    # --- Step 4: Speak the confirmation back ---
    spoken_confirm = f"I'll run {command} {args_display}. Shall I proceed?"
    audio = await _safe_synthesise(spoken_confirm)
    if audio:
        await _send_voice(msg, audio)

    # --- Step 5: Show the standard inline keyboard confirmation ---
    # From here the flow is identical to /ask — handle_ask_callback takes over.
    callback_data = _encode_ask_cb(command, args)
    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Yes", callback_data=callback_data),
            InlineKeyboardButton("Cancel", callback_data="ask_cancel"),
        ]
    ])
    await msg.reply_text(
        confirmation_text,
        reply_markup=keyboard,
        parse_mode="Markdown",
    )


# ── Voice helpers ─────────────────────────────────────────────────

async def _safe_synthesise(text: str) -> bytes | None:
    """Call voice.synthesise and return None on failure instead of raising.

    TTS failure should never crash the handler — Kira falls back to text only.
    """
    try:
        return await voice.synthesise(text)
    except Exception as exc:
        logger.warning("TTS synthesis failed (falling back to text): %s", exc)
        return None


async def _send_voice(msg, audio_bytes: bytes) -> None:
    """Send MP3 bytes as a Telegram voice message.

    Uses reply_voice so it appears inline in the conversation as a playable
    audio bubble, not as a file download.
    """
    import io
    bio = io.BytesIO(audio_bytes)
    bio.name = "response.mp3"
    try:
        await msg.reply_voice(voice=bio)
    except Exception as exc:
        logger.warning("Failed to send voice reply: %s", exc)


async def handle_ask_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle Yes/Cancel callbacks for /ask confirmations."""
    query = update.callback_query
    await query.answer()

    data = query.data or ""
    # Replies go to the chat via query.message, not update.message (which is None
    # in a callback context).
    reply = query.message.reply_text

    if data == "ask_cancel":
        await query.edit_message_text("Cancelled.")
        return

    if not data.startswith("ask_yes|"):
        return

    command, args = _decode_ask_cb(data)
    if not command:
        await query.edit_message_text("Error: could not decode command.")
        return

    args_display = " ".join(args)
    await query.edit_message_text(f"Executing: `{command} {args_display}`", parse_mode="Markdown")

    if command == "/run":
        if not args:
            await reply("Error: /run requires an alias.")
            return
        await _ask_exec_run(reply, args)

    elif command == "/shell":
        if not args:
            await reply("Error: /shell requires a command.")
            return
        await _ask_exec_shell(reply, args)

    elif command == "/schedule":
        if len(args) < 2:
            await reply("Error: /schedule requires <alias> <time>.")
            return
        await _ask_exec_schedule(reply, args)

    elif command == "/open":
        if not args:
            await reply("Error: /open requires an app name.")
            return
        await _ask_exec_open(reply, args)

    elif command == "/status":
        processes = process_registry.list_processes()
        if not processes:
            await reply("No running processes.")
        else:
            lines = ["Running processes:\n"]
            for p in processes:
                runtime = _format_runtime(p["runtime_seconds"])
                lines.append(f"PID {p['pid']} — {p['alias']} — {runtime}")
            await reply("\n".join(lines))

    elif command == "/kill":
        if not args:
            await reply("Error: /kill requires a PID.")
            return
        try:
            pid = int(args[0])
        except ValueError:
            await reply("Error: PID must be an integer.")
            return
        result = await process_registry.kill(pid)
        await reply(result)

    elif command == "/sysinfo":
        cpu = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("C:\\")
        lines = [
            "System Info\n",
            f"CPU:  {cpu}%",
            f"RAM:  {mem.used / (1024**3):.1f} / {mem.total / (1024**3):.1f} GB ({mem.percent}%)",
            f"Disk: {disk.free / (1024**3):.1f} GB free / {disk.total / (1024**3):.1f} GB ({disk.percent}%)",
        ]
        await reply("\n".join(lines))

    elif command == "/screenshot":
        try:
            with mss.mss() as sct:
                screenshot = sct.grab(sct.monitors[0])
                png_bytes = mss.tools.to_png(screenshot.rgb, screenshot.size)
            bio = io.BytesIO(png_bytes)
            bio.name = "screenshot.png"
            await query.message.reply_photo(photo=bio)
        except Exception as exc:
            await reply(f"Screenshot failed: {exc}")

    elif command == "/ls":
        target = Path(args[0]) if args else get_cwd()
        try:
            entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            lines = [f"{target}\n"]
            for entry in entries[:100]:
                prefix = "[dir]" if entry.is_dir() else "[file]"
                lines.append(f"{prefix} {entry.name}")
            await reply("\n".join(lines)[:4000])
        except Exception as exc:
            await reply(f"Error: {exc}")

    elif command == "/find":
        if not args:
            await reply("Error: /find requires a pattern.")
            return
        pattern = args[0]
        search_root = Path(args[1]) if len(args) > 1 else get_cwd()
        try:
            matches = []
            for m in search_root.rglob(pattern):
                matches.append(str(m))
                if len(matches) >= 50:
                    break
            if matches:
                await reply("\n".join(matches)[:4000])
            else:
                await reply("No matches found.")
        except Exception as exc:
            await reply(f"Error: {exc}")

    elif command == "/tail":
        if not args:
            await reply("Error: /tail requires a path.")
            return
        try:
            n = int(args[1]) if len(args) > 1 else 20
        except ValueError:
            n = 20
        try:
            lines = Path(args[0]).read_text(encoding="utf-8", errors="replace").splitlines()
            await reply("\n".join(lines[-n:])[:4000])
        except Exception as exc:
            await reply(f"Error: {exc}")

    elif command == "/copy":
        if not args:
            await reply("Error: /copy requires text.")
            return
        text = " ".join(args)
        pyperclip.copy(text)
        await reply(f"Copied to clipboard ({len(text)} chars).")

    elif command == "/paste":
        content = pyperclip.paste()
        await reply(content[:4000] if content else "Clipboard is empty.")

    elif command == "/sleep":
        await reply("Putting PC to sleep...")
        os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")

    elif command == "/shutdown":
        try:
            minutes = int(args[0]) if args else 0
        except ValueError:
            await reply("Error: minutes must be an integer.")
            return
        os.system(f"shutdown /s /t {minutes * 60}")
        await reply(f"Shutdown scheduled in {minutes} minutes.")

    elif command == "/reboot":
        try:
            minutes = int(args[0]) if args else 0
        except ValueError:
            await reply("Error: minutes must be an integer.")
            return
        os.system(f"shutdown /r /t {minutes * 60}")
        await reply(f"Reboot scheduled in {minutes} minutes.")

    elif command == "/remind":
        if len(args) < 2:
            await reply("Error: /remind requires <time> <message>.")
            return
        delay = _parse_delay(args[0])
        if delay is None:
            await reply("Invalid time. Use Xm or Xh.")
            return
        msg = " ".join(args[1:])
        await reply(f"Reminder set for {args[0]} from now.")

        async def _fire() -> None:
            await asyncio.sleep(delay)
            await notifier.send(f"Reminder: {msg}")

        asyncio.create_task(_fire())

    elif command == "/watches":
        items = watchdog.list_watches()
        if not items:
            await reply("No active watchers.")
        else:
            lines = ["Active watchers:\n"] + [
                f"{w['id']} — {w['type']}: {w['target']}" for w in items
            ]
            await reply("\n".join(lines))

    elif command == "/schedules":
        items = scheduler.list_schedules()
        if not items:
            await reply("No pending scheduled runs.")
        else:
            lines = ["Pending schedules:\n"] + [
                f"{s['id']} — {s['alias']} at {s['run_at']}" for s in items
            ]
            await reply("\n".join(lines))

    elif command == "/history":
        try:
            n = int(args[0]) if args else 20
        except ValueError:
            n = 20
        try:
            rows = await db.get_recent_conversations(n)
        except Exception as exc:
            await reply(f"Error: {exc}")
            return
        if not rows:
            await reply("No conversation history yet.")
        else:
            lines = [f"Last {len(rows)} entries:\n"]
            for r in rows:
                ts = r["timestamp"][:16] if r.get("timestamp") else "?"
                lines.append(f"[{ts}] {r['role'].upper()}: {r['content'][:200]}")
            await reply("\n".join(lines)[:4000])

    elif command == "/runs":
        alias_arg = args[0] if args else None
        try:
            rows = await db.get_run_history(alias=alias_arg, limit=10)
        except Exception as exc:
            await reply(f"Error: {exc}")
            return
        if not rows:
            await reply("No runs recorded yet.")
        else:
            lines = [f"Last {len(rows)} run(s):\n"]
            for r in rows:
                code = r.get("exit_code")
                icon = "✅" if code == 0 else "❌" if code is not None else "?"
                rt = _format_runtime(r["runtime_seconds"]) if r.get("runtime_seconds") else "?"
                lines.append(f"{icon} {r['alias']} — {rt}")
            await reply("\n".join(lines)[:4000])

    else:
        await reply(f"Command '{command}' is not yet supported via /ask.")


# ── /ask execution helpers ─────────────────────────────────────────

async def _ask_exec_run(reply, args: list[str]) -> None:
    """Run a script alias from an /ask callback."""
    alias = args[0]
    script_args = args[1:]
    script = _get_script(alias)
    if not script:
        await reply(f"Unknown alias: {alias}")
        return
    interpreter = script.get("interpreter")
    path = script.get("path")
    timeout = script.get("timeout", _DEFAULT_TIMEOUT)
    checkpoint = script.get("checkpoint_interval")
    full_args = list(script.get("args", [])) + script_args
    await reply(f"Running {alias}...")
    try:
        gen = executor.run_command(interpreter, path, full_args, timeout, alias=alias, checkpoint_interval=checkpoint)
        async for chunk in gen:
            if chunk.strip():
                await reply(chunk[:4000])
    except Exception as exc:
        await reply(f"Error: {exc}")


async def _ask_exec_shell(reply, args: list[str]) -> None:
    """Run a shell command from an /ask callback using executor (non-blocking)."""
    command = " ".join(args)
    if _DESTRUCTIVE_PATTERNS.search(command):
        await reply("Destructive command detected. Use /shell directly to confirm.")
        return
    timeout = int(os.environ.get("DEFAULT_TIMEOUT", "30"))
    await reply(f"Running: `{command}`")
    try:
        gen = executor.run_shell(command, timeout=timeout)
        async for chunk in gen:
            if chunk.strip():
                await reply(chunk[:4000])
    except Exception as exc:
        await reply(f"Error: {exc}")


async def _ask_exec_open(reply, args: list[str]) -> None:
    """Open an application from an /ask callback using the name-only launcher."""
    result = _open_app_by_name(" ".join(args))
    await reply(result)


async def _ask_exec_schedule(reply, args: list[str]) -> None:
    """Schedule a script from an /ask callback."""
    alias = args[0]
    time_spec = args[1]
    script = _get_script(alias)
    if not script:
        await reply(f"Unknown alias: {alias}")
        return
    target_time = _parse_time_spec(time_spec)
    if not target_time:
        await reply("Invalid time format. Use HH:MM, Xm, or Xh.")
        return

    sid = await scheduler.schedule(alias, target_time, _scheduled_run_callback)
    await reply(f"Scheduled {alias} for {target_time.strftime('%H:%M')} (ID: {sid}).")
