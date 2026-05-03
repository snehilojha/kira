"""Process execution and power command handlers: run, shell, chain, status, kill, sysinfo, sleep/shutdown/reboot."""

from __future__ import annotations

import asyncio
import logging
import os
import re

import psutil

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from bot.auth import require_auth
from bot import executor
from bot import notifier
from bot import process_registry

logger = logging.getLogger(__name__)

# Shared config — populated by handlers.load_config() at startup.
# These are module references rather than copies so handlers can update them.
_SCRIPTS_CONFIG: dict = {}
_DEFAULT_TIMEOUT: int = 30

_DESTRUCTIVE_PATTERNS = re.compile(
    r"\b(rm|del|format|rmdir|rd\s*/s|DROP|DELETE\s+FROM)\b", re.IGNORECASE
)

_PENDING_CONFIRMS: dict[str, dict] = {}
_CONFIRM_TIMEOUT = 30  # seconds
_OUTPUT_RECORDER = None


def _get_script(alias: str) -> dict | None:
    return _SCRIPTS_CONFIG.get(alias)


# ── Formatters ────────────────────────────────────────────────────

def _format_runtime(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m"


# ── Stream helper ─────────────────────────────────────────────────

async def _stream_to_chat(update: Update, gen) -> None:
    """Consume executor async generator and stream chunks to Telegram."""
    async for chunk in gen:
        if chunk.strip():
            if callable(_OUTPUT_RECORDER):
                try:
                    _OUTPUT_RECORDER(chunk)
                except Exception:
                    logger.debug("output recorder failed", exc_info=True)
            await update.message.reply_text(chunk[:4000])


async def _run_shell(update: Update, command: str) -> None:
    timeout = int(os.environ.get("DEFAULT_TIMEOUT", "30"))
    await update.message.reply_text(f"▶️ Shell: `{command}`", parse_mode="Markdown")
    gen = executor.run_shell(command, timeout=timeout)
    await _stream_to_chat(update, gen)


# ── Script execution ──────────────────────────────────────────────

@require_auth
async def handle_run(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/run <alias> [args...] — execute a script from scripts.toml."""
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
    """/shell <command> — run arbitrary shell command via cmd.exe /c."""
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
        asyncio.get_event_loop().call_later(
            _CONFIRM_TIMEOUT,
            lambda t=token: _PENDING_CONFIRMS.pop(t, None),
        )
        return

    await _run_shell(update, command)


@require_auth
async def handle_chain(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/chain <alias> — run a script and all chained scripts sequentially."""
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

        if "❌" in last_chunk or "⏰" in last_chunk:
            await update.message.reply_text(f"⛓️ Chain stopped at {step_alias} due to failure.")
            return

    await update.message.reply_text("⛓️ Chain completed successfully.")


def _resolve_chain(alias: str) -> list[str] | None:
    script = _get_script(alias)
    if script is None:
        return None
    chain = [alias]
    current = script
    while current.get("chain"):
        next_alias = current["chain"][0] if isinstance(current["chain"], list) else current["chain"]
        if next_alias in chain:
            break
        chain.append(next_alias)
        current = _get_script(next_alias) or {}
    return chain


# ── Process status ────────────────────────────────────────────────

@require_auth
async def handle_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/status — list all running processes."""
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
    """/kill <pid> — terminate a running process."""
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


# ── System info ───────────────────────────────────────────────────

@require_auth
async def handle_sysinfo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/sysinfo — CPU %, RAM, GPU, disk."""
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("C:\\")

    lines = [
        f"🖥️ **System Info**\n",
        f"CPU:  {cpu}%",
        f"RAM:  {mem.used / (1024**3):.1f} / {mem.total / (1024**3):.1f} GB ({mem.percent}%)",
        f"Disk: {disk.free / (1024**3):.1f} GB free / {disk.total / (1024**3):.1f} GB ({disk.percent}%)",
    ]

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


# ── Power commands ────────────────────────────────────────────────

@require_auth
async def handle_sleep(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/sleep — put PC to sleep (with confirmation)."""
    await _power_confirm(update, "sleep", "Put PC to sleep?")


@require_auth
async def handle_shutdown(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/shutdown <minutes> — schedule shutdown."""
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
    """/reboot <minutes> — schedule reboot."""
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
    """/abort_shutdown — cancel pending shutdown/reboot."""
    os.system("shutdown /a")
    await update.message.reply_text("✅ Shutdown/reboot cancelled.")


async def _power_confirm(update: Update, action: str, prompt: str) -> None:
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


# ── Scheduled run callback (used by scheduler on restart) ─────────

async def scheduled_run_callback(alias: str) -> None:
    """Run a script alias and notify via Telegram. Called by scheduler on restart."""
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
