"""All Telegram command handlers.

Every handler is decorated with ``@require_auth``.  One function per command.
Handlers delegate heavy lifting to executor, process_registry, scheduler,
watchdog, and notifier — they never contain business logic directly.
"""

import asyncio
import io
import logging
import os
import re
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
from bot import executor
from bot import notifier
from bot import process_registry
from bot import scheduler
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


# ── Helper: stream executor output back to Telegram ───────────────

async def _stream_to_chat(update: Update, gen) -> None:
    """Consume an async generator from executor and send chunks as messages."""
    async for chunk in gen:
        if chunk.strip():
            # Telegram limit is 4096; executor already caps at 4000
            await update.message.reply_text(chunk[:4000])


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

    async def _run_scheduled(a: str) -> None:
        """Callback invoked by the scheduler when the time arrives."""
        script = _get_script(a)
        if script is None:
            await notifier.send(f"❌ Scheduled run failed — alias {a} not found.")
            return
        timeout = script.get("timeout", _DEFAULT_TIMEOUT)
        checkpoint = script.get("checkpoint_interval")
        args = list(script.get("args", []))
        gen = executor.run_command(
            interpreter=script["interpreter"],
            script_path=script["path"],
            args=args,
            timeout=timeout,
            alias=a,
            checkpoint_interval=checkpoint,
        )
        # Consume the generator (output goes via notifier for scheduled runs)
        output_lines = []
        async for chunk in gen:
            output_lines.append(chunk)
        # Send final output summary
        full_output = "\n".join(output_lines)
        if full_output.strip():
            await notifier.send(f"📋 Output from scheduled {a}:\n{full_output[-3000:]}")

    sid = await scheduler.schedule(alias, run_at, _run_scheduled)
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

    await update.message.reply_text(f"⏰ Reminder set for {time_spec} from now.")

    async def _fire_reminder() -> None:
        await asyncio.sleep(delay)
        await notifier.send(f"🔔 Reminder: {message}")

    asyncio.create_task(_fire_reminder())


# ── Help ──────────────────────────────────────────────────────────

@require_auth
async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """``/help`` — list all commands and script aliases."""
    aliases = ", ".join(_SCRIPTS_CONFIG.keys()) or "(none configured)"
    help_text = (
        "**telegram-runner commands:**\n\n"
        "**Script Execution**\n"
        "/run <alias> [args] — Run a script\n"
        "/shell <command> — Run shell command\n"
        "/status — List running processes\n"
        "/kill <pid> — Kill a process\n"
        "/chain <alias> — Run with chained scripts\n"
        "/schedule <alias> <time> — Schedule a run\n"
        "/schedules — List pending schedules\n"
        "/unschedule <id> — Cancel a schedule\n\n"
        "**ML / System**\n"
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
        "**Screenshot**\n"
        "/screenshot [n] — Capture screen\n\n"
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
        "**Misc**\n"
        "/remind <Xm|Xh> <msg> — Set reminder\n"
        "/help — This message\n\n"
        f"**Script aliases:** {aliases}"
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
