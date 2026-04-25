"""Top-level Telegram handler coordinator for Kira.

This module now keeps the natural-language `/ask` and callback flows while
re-exporting command-family handlers from narrower modules.
"""

from __future__ import annotations

import asyncio
from collections import deque
import json
import logging
import os
import re
import subprocess
from pathlib import Path

import psutil
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from bot.auth import require_auth
from bot import brain
from bot import cmd_app
from bot import cmd_fs
from bot import cmd_jobs
from bot import cmd_process
from bot import cmd_schedule
from bot import db
from bot import process_registry
from bot import provider
from bot import router
from bot import scheduler
from bot import utils as _bot_utils
from bot import voice
from bot import voice_confirm
from bot import watchdog

logger = logging.getLogger(__name__)

_SCRIPTS_CONFIG: dict = {}
_DEFAULT_TIMEOUT: int = 30

# Exposed so tests can monkey-patch the path without touching bot.utils internals.
_PROJECT_CONTEXT_PATH = _bot_utils._PROJECT_CONTEXT_PATH
_RECENT_OUTPUT_MAX_CHARS = 2000
_RECENT_OUTPUT_LINES: deque[str] = deque(maxlen=20)

# Compatibility re-exports used by tests and other modules.
_PENDING_BRAIN_APPROVALS = cmd_jobs._PENDING_BRAIN_APPROVALS
_scheduled_run_callback = cmd_process.scheduled_run_callback
reload_reminders = cmd_schedule.reload_reminders
register_voice_confirm = voice_confirm.register_voice_confirm

# Command-family handler re-exports.
handle_run = cmd_process.handle_run
handle_shell = cmd_process.handle_shell
handle_chain = cmd_process.handle_chain
handle_status = cmd_process.handle_status
handle_kill = cmd_process.handle_kill
handle_sysinfo = cmd_process.handle_sysinfo
handle_sleep = cmd_process.handle_sleep
handle_shutdown = cmd_process.handle_shutdown
handle_reboot = cmd_process.handle_reboot
handle_abort_shutdown = cmd_process.handle_abort_shutdown

handle_schedule = cmd_schedule.handle_schedule
handle_schedules = cmd_schedule.handle_schedules
handle_unschedule = cmd_schedule.handle_unschedule
handle_watch = cmd_schedule.handle_watch
handle_watches = cmd_schedule.handle_watches
handle_unwatch = cmd_schedule.handle_unwatch
handle_remind = cmd_schedule.handle_remind

handle_getfile = cmd_fs.handle_getfile
handle_putfile = cmd_fs.handle_putfile
handle_cd = cmd_fs.handle_cd
handle_ls = cmd_fs.handle_ls
handle_find = cmd_fs.handle_find
handle_tail = cmd_fs.handle_tail
handle_mkdir = cmd_fs.handle_mkdir
handle_move = cmd_fs.handle_move
handle_copy = cmd_fs.handle_copy
handle_paste = cmd_fs.handle_paste
handle_screenshot = cmd_fs.handle_screenshot
get_cwd = cmd_fs.get_cwd
set_cwd = cmd_fs.set_cwd

handle_list_apps = cmd_app.handle_list_apps
handle_open = cmd_app.handle_open
handle_close_apps = cmd_app.handle_close_apps
_normalize_app_name = cmd_app._normalize_app_name
_open_app_by_name = cmd_app._open_app_by_name

handle_history = cmd_jobs.handle_history
handle_runs = cmd_jobs.handle_runs
handle_summarise = cmd_jobs.handle_summarise
handle_reflect = cmd_jobs.handle_reflect
handle_recall = cmd_jobs.handle_recall
handle_jobs = cmd_jobs.handle_jobs
handle_cancel_job = cmd_jobs.handle_cancel_job
handle_pause_job = cmd_jobs.handle_pause_job
handle_resume_job = cmd_jobs.handle_resume_job
handle_mode = cmd_jobs.handle_mode
handle_tasks = cmd_jobs.handle_tasks
handle_task = cmd_jobs.handle_task
_format_task_state_summary = cmd_jobs._format_task_state_summary
_format_task_state_detail = cmd_jobs._format_task_state_detail
_build_route_stub_text = cmd_jobs._build_route_stub_text
_run_complex_task_with_progress = cmd_jobs.run_complex_task_with_progress
_create_monitor_job_from_message = cmd_jobs.create_monitor_job_from_message


def load_config() -> None:
    """Load scripts.toml and propagate shared config to command modules."""
    global _SCRIPTS_CONFIG, _DEFAULT_TIMEOUT
    config_path = Path(__file__).resolve().parent.parent / "config" / "scripts.toml"
    if config_path.exists():
        import toml

        _SCRIPTS_CONFIG = toml.load(config_path)
        logger.info("Loaded %d script aliases from %s", len(_SCRIPTS_CONFIG), config_path)
    else:
        logger.warning("scripts.toml not found at %s", config_path)
        _SCRIPTS_CONFIG = {}

    _DEFAULT_TIMEOUT = int(os.environ.get("DEFAULT_TIMEOUT", "30"))

    cmd_process._SCRIPTS_CONFIG = _SCRIPTS_CONFIG
    cmd_process._DEFAULT_TIMEOUT = _DEFAULT_TIMEOUT
    cmd_process._OUTPUT_RECORDER = _record_recent_output


def _load_project_context() -> str:
    return _bot_utils.load_project_context(_PROJECT_CONTEXT_PATH)


def _record_recent_output(text: str) -> None:
    cleaned = text.strip()
    if cleaned:
        _RECENT_OUTPUT_LINES.append(cleaned)


def _get_recent_output_tail() -> str:
    if not _RECENT_OUTPUT_LINES:
        return "Recent command output: none"

    tail = "\n".join(_RECENT_OUTPUT_LINES)
    if len(tail) > _RECENT_OUTPUT_MAX_CHARS:
        tail = tail[-_RECENT_OUTPUT_MAX_CHARS :]
        tail = "[...recent output truncated...]\n" + tail
    return f"Recent command output:\n{tail}"


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


def _format_process_snapshot(processes: list[dict]) -> str:
    if not processes:
        return "Running processes: none"

    lines = ["Running processes:"]
    for proc in processes:
        runtime = _format_runtime(proc["runtime_seconds"])
        status = "running" if proc["returncode"] is None else f"exited({proc['returncode']})"
        lines.append(f"- PID {proc['pid']}: {proc['alias']} ({runtime}, {status})")
    return "\n".join(lines)


def _format_schedule_snapshot(schedules: list[dict]) -> str:
    if not schedules:
        return "Pending schedules: none"
    lines = ["Pending schedules:"]
    for item in schedules:
        lines.append(f"- {item['id']}: {item['alias']} at {item['run_at']}")
    return "\n".join(lines)


def _format_watch_snapshot(watches: list[dict]) -> str:
    if not watches:
        return "Active watchers: none"
    lines = ["Active watchers:"]
    for item in watches:
        lines.append(f"- {item['id']}: {item['type']} -> {item['target']}")
    return "\n".join(lines)


def _format_system_snapshot() -> str:
    cpu = psutil.cpu_percent(interval=0.0)
    memory = psutil.virtual_memory()
    lines = [f"System snapshot: CPU {cpu:.1f}% | RAM {memory.percent:.1f}%"]

    try:
        import GPUtil

        gpus = GPUtil.getGPUs()
        temperatures = [float(g.temperature) for g in gpus if getattr(g, "temperature", None) is not None]
        if temperatures:
            lines.append(f"GPU temp: {max(temperatures):.1f}°C")
    except Exception:
        pass

    return " | ".join(lines)


def _format_live_context() -> str:
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


async def _format_conversation_history() -> str:
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


def _truncate_for_prompt(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[...context truncated...]"


async def _build_ask_system_prompt() -> str:
    scripts_info = "\n".join(
        f"- {alias}: {cfg.get('path', 'N/A')}" for alias, cfg in _SCRIPTS_CONFIG.items()
    ) or "- (none configured)"

    project_context = _load_project_context()
    live_context = _format_live_context()
    conversation_context = await _format_conversation_history()

    observer_context = ""
    try:
        from bot import observer

        observer_context = observer.get_current_context()
    except Exception:
        pass

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
        "- /close_apps <app1> [app2] - Close running applications by name",
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
        sections.extend(["", "Machine awareness (auto-updated every 15 min):", observer_context])
    if session_context:
        sections.extend(["", session_context])
    if conversation_context:
        sections.extend(["", conversation_context])
    if project_context:
        sections.extend(["", "Project context file:", project_context])

    sections.extend(
        [
            "",
            "Instructions:",
            "1. Analyse the user's request.",
            "2. Map it to the single most appropriate Kira command.",
            "3. Return ONLY a JSON object — no prose, no markdown fences.",
            "",
            "Response format (examples):",
            '{"command": "/run", "args": ["crypto_train_explore", "--fee_mult", "10"]}',
            '{"command": "/shell", "args": ["dir C:\\\\Users"]}',
            '{"command": "/open", "args": ["chrome"]}',
            '{"command": "/close_apps", "args": ["chrome"]}',
            '{"command": "/status", "args": []}',
        ]
    )
    return _truncate_for_prompt("\n".join(sections), 24000)


def _build_ask_confirmation_text(command: str, args: list[str]) -> str:
    args_display = " ".join(args)
    return f"Proposed command:\n`{command} {args_display}`\n\nExecute?"


_CB_MAX = 64


def _encode_ask_cb(command: str, args: list[str]) -> str:
    args_json = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
    token = f"ask_yes|{command}|{args_json}"
    while len(token.encode()) > _CB_MAX and args:
        args = args[:-1]
        args_json = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
        token = f"ask_yes|{command}|{args_json}"
    return token


def _decode_ask_cb(data: str) -> tuple[str, list[str]]:
    try:
        _, command, args_json = data.split("|", 2)
        args = json.loads(args_json)
        if not isinstance(args, list):
            args = []
        return command, [str(a) for a in args]
    except Exception:
        return "", []


async def _ask_core(user_message: str) -> tuple[str, list[str]] | None:
    system_prompt = await _build_ask_system_prompt()
    response = await provider.create_chat_completion(
        role="fast",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        max_tokens=200,
    )

    result_text = (response.choices[0].message.content or "").strip()
    result_text = re.sub(r"^```[a-z]*\n?", "", result_text)
    result_text = re.sub(r"\n?```$", "", result_text).strip()

    try:
        parsed = json.loads(result_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", result_text, re.DOTALL)
        if not match:
            logger.warning("_ask_core: no JSON found in response: %r", result_text)
            return None
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            logger.warning("_ask_core: unparseable response: %r", result_text)
            return None

    command = parsed.get("command", "").strip()
    args = [str(a) for a in parsed.get("args", [])]
    if not command:
        return None
    return command, args


@require_auth
async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
        "/ask <request> — Translate plain English to commands\n"
        "/tasks [n] — Show recent complex task states\n"
        "/task <task_id> — Show one complex task state\n\n"
        "**System**\n"
        "/sysinfo — CPU, RAM, GPU, disk\n"
        "/getfile <path> — Send file from PC to phone (up to 50 MB)\n"
        "/putfile [path] — Save file from phone to PC\n\n"
        "**Filesystem**\n"
        "/cd [path] — Change directory\n"
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
        "/remind <Xm|Xh> <msg> — Set reminder\n"
        "/help — This message\n\n"
        f"**Script aliases:**\n{alias_lines}"
    )
    await update.message.reply_text(help_text[:4000])


async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    data = query.data or ""

    if data.startswith("confirm_"):
        token = data[len("confirm_") :]
        entry = cmd_process._PENDING_CONFIRMS.pop(token, None)
        if entry is None:
            await query.edit_message_text("⏰ Confirmation expired.")
            return

        if "command" in entry:
            await query.edit_message_text(f"✅ Executing: `{entry['command']}`", parse_mode="Markdown")
            await cmd_process._run_shell(entry["update"], entry["command"])
            return

        if "action" in entry:
            action = entry["action"]
            if action == "sleep":
                await query.edit_message_text("💤 Putting PC to sleep...")
                os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
            elif action.startswith("shutdown_"):
                minutes = int(action.split("_")[1])
                await query.edit_message_text(f"🔴 Shutdown in {minutes} minutes.")
                os.system(f"shutdown /s /t {minutes * 60}")
            elif action.startswith("reboot_"):
                minutes = int(action.split("_")[1])
                await query.edit_message_text(f"🔄 Reboot in {minutes} minutes.")
                os.system(f"shutdown /r /t {minutes * 60}")
            return

    if data.startswith("cancel_"):
        token = data[len("cancel_") :]
        cmd_process._PENDING_CONFIRMS.pop(token, None)
        await query.edit_message_text("❌ Cancelled.")
        return

    if data.startswith("voice_confirm_"):
        token = data[len("voice_confirm_") :]
        future = voice_confirm._PENDING_VOICE_CONFIRMS.pop(token, None)
        if future and not future.done():
            future.set_result(True)
        await query.edit_message_text("✅ Confirmed.")
        return

    if data.startswith("voice_cancel_"):
        token = data[len("voice_cancel_") :]
        future = voice_confirm._PENDING_VOICE_CONFIRMS.pop(token, None)
        if future and not future.done():
            future.set_result(False)
        await query.edit_message_text("❌ Cancelled.")
        return


@require_auth
async def handle_ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    raw_text = update.message.text or ""
    user_message = raw_text.split(" ", 1)[1].strip() if " " in raw_text else ""
    if not user_message:
        await update.message.reply_text("Usage: /ask <your request in plain English>")
        return

    try:
        await db.log_conversation("user", user_message)
    except Exception:
        logger.debug("Failed to log /ask user message to DB", exc_info=True)

    await update.message.reply_text("Thinking...")

    try:
        decision = await router.classify_request(user_message)
        logger.info(
            "/ask routed as %s via %s (confidence=%.2f): %s",
            decision.route,
            decision.source,
            decision.confidence,
            decision.reason,
        )

        if decision.needs_clarification:
            stub_text = _build_route_stub_text(decision)
            try:
                await db.log_conversation("assistant", stub_text)
            except Exception:
                logger.debug("Failed to log /ask route stub to DB", exc_info=True)
            await update.message.reply_text(stub_text)
            return

        if decision.route == "complex":
            task_request = brain.build_task_request(
                user_input=user_message,
                source="telegram",
                route="complex",
                conversation_id="telegram",
            )
            result = await _run_complex_task_with_progress(update.message.reply_text, task_request)
            try:
                await db.log_conversation("assistant", result.summary)
            except Exception:
                logger.debug("Failed to log complex /ask response to DB", exc_info=True)
            await update.message.reply_text(result.summary[:4000])
            return

        if decision.route == "monitor":
            created = await _create_monitor_job_from_message(user_message, update.message.reply_text)
            if not created:
                try:
                    await db.log_conversation("assistant", "Monitor job parse failed.")
                except Exception:
                    logger.debug("Failed to log /ask monitor failure to DB", exc_info=True)
            return

        result = await _ask_core(user_message)
        if result is None:
            await update.message.reply_text("Could not understand the request. Try being more specific.")
            return

        command, args = result
        try:
            await db.log_conversation("assistant", f"{command} {' '.join(args)}")
        except Exception:
            logger.debug("Failed to log /ask response to DB", exc_info=True)

        keyboard = InlineKeyboardMarkup(
            [[InlineKeyboardButton("Yes", callback_data=_encode_ask_cb(command, args)), InlineKeyboardButton("Cancel", callback_data="ask_cancel")]]
        )
        await update.message.reply_text(
            _build_ask_confirmation_text(command, args),
            reply_markup=keyboard,
            parse_mode="Markdown",
        )
    except Exception as exc:
        logger.error("handle_ask error: %s", exc, exc_info=True)
        await update.message.reply_text(f"Error: {exc}")


@require_auth
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_message(update, context)


@require_auth
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = (update.message.text or "").strip()
    if not user_message:
        return

    try:
        await db.log_conversation("user", user_message)
    except Exception:
        logger.debug("Failed to log message to DB", exc_info=True)

    await update.message.reply_text("...")

    try:
        task_request = brain.build_task_request(
            user_input=user_message,
            source="telegram",
            route="complex",
            conversation_id="telegram",
        )
        result = await _run_complex_task_with_progress(update.message.reply_text, task_request)
        try:
            await db.log_conversation("assistant", result.summary)
        except Exception:
            pass
        await update.message.reply_text(result.summary[:4000])
    except Exception as exc:
        logger.error("handle_message error: %s", exc, exc_info=True)
        await update.message.reply_text(f"Error: {exc}")


@require_auth
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    try:
        tg_file = await msg.voice.get_file()
        ogg_bytes = await tg_file.download_as_bytearray()
    except Exception as exc:
        logger.error("handle_voice: download failed: %s", exc)
        await msg.reply_text("Failed to download voice message.")
        return

    try:
        transcript = await voice.transcribe(bytes(ogg_bytes))
    except Exception as exc:
        logger.error("handle_voice: transcription failed: %s", exc)
        await msg.reply_text("Voice transcription failed. Try again.")
        return

    if not transcript:
        await msg.reply_text("Couldn't make out what you said. Try again.")
        return

    logger.info("handle_voice: user %s said: %r", update.effective_user.id, transcript)
    await msg.reply_text(f"🎙️ Heard: _{transcript}_", parse_mode="Markdown")

    try:
        await db.log_conversation("user", f"[voice] {transcript}")
    except Exception:
        logger.debug("Failed to log voice transcript to DB", exc_info=True)

    try:
        decision = await router.classify_request(transcript)
        logger.info(
            "voice routed as %s via %s (confidence=%.2f): %s",
            decision.route,
            decision.source,
            decision.confidence,
            decision.reason,
        )
        if decision.needs_clarification:
            response_text = _build_route_stub_text(decision)
            await msg.reply_text(response_text)
            audio = await voice_confirm._safe_synthesise(response_text)
            if audio:
                await voice_confirm._send_voice(msg, audio)
            return

        if decision.route == "complex":
            task_request = brain.build_task_request(
                user_input=transcript,
                source="voice",
                route="complex",
                conversation_id="voice",
            )
            await msg.reply_text("Working on a deeper read-only analysis...")
            result = await _run_complex_task_with_progress(msg.reply_text, task_request)
            await msg.reply_text(result.summary[:4000])
            audio = await voice_confirm._safe_synthesise(result.summary[:2000])
            if audio:
                await voice_confirm._send_voice(msg, audio)
            return

        if decision.route == "monitor":
            created = await _create_monitor_job_from_message(transcript, msg.reply_text)
            response_text = (
                "Monitor job created. I'll notify you when the condition is met."
                if created
                else _build_route_stub_text(decision)
            )
            audio = await voice_confirm._safe_synthesise(response_text)
            if audio:
                await voice_confirm._send_voice(msg, audio)
            return

        result = await _ask_core(transcript)
    except Exception as exc:
        logger.error("handle_voice routing failed: %s", exc)
        error_audio = await voice_confirm._safe_synthesise(f"Sorry, I ran into an error: {exc}")
        if error_audio:
            await voice_confirm._send_voice(msg, error_audio)
        return

    if result is None:
        response_text = "I couldn't map that to a command. Could you rephrase?"
        await msg.reply_text(response_text)
        audio = await voice_confirm._safe_synthesise(response_text)
        if audio:
            await voice_confirm._send_voice(msg, audio)
        return

    command, args = result
    args_display = " ".join(args)
    spoken_confirm = f"I'll run {command} {args_display}. Shall I proceed?"
    audio = await voice_confirm._safe_synthesise(spoken_confirm)
    if audio:
        await voice_confirm._send_voice(msg, audio)

    keyboard = InlineKeyboardMarkup(
        [[InlineKeyboardButton("Yes", callback_data=_encode_ask_cb(command, args)), InlineKeyboardButton("Cancel", callback_data="ask_cancel")]]
    )
    await msg.reply_text(
        f"Proposed command:\n`{command} {args_display}`\n\nExecute?",
        reply_markup=keyboard,
        parse_mode="Markdown",
    )


async def handle_ask_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    data = query.data or ""
    reply = query.message.reply_text

    if data.startswith("brain_yes|") or data.startswith("brain_no|"):
        approved = data.startswith("brain_yes|")
        request_id = data.split("|", 1)[1] if "|" in data else ""
        future = _PENDING_BRAIN_APPROVALS.pop(request_id, None)
        if future is None:
            await query.edit_message_text("That approval request is no longer pending.")
            return
        if not future.done():
            future.set_result(approved)
        await query.edit_message_text("Approved complex-task action." if approved else "Denied complex-task action.")
        return

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
        alias = args[0]
        script_args = args[1:]
        script = cmd_process._get_script(alias)
        if not script:
            await reply(f"Unknown alias: {alias}")
            return
        await reply(f"Running {alias}...")
        gen = cmd_process.executor.run_command(
            script["interpreter"],
            script["path"],
            list(script.get("args", [])) + script_args,
            script.get("timeout", _DEFAULT_TIMEOUT),
            alias=alias,
            checkpoint_interval=script.get("checkpoint_interval"),
        )
        async for chunk in gen:
            if chunk.strip():
                _record_recent_output(chunk)
                await reply(chunk[:4000])
        return

    if command == "/shell":
        if not args:
            await reply("Error: /shell requires a command.")
            return
        shell_command = " ".join(args)
        if cmd_process._DESTRUCTIVE_PATTERNS.search(shell_command):
            await reply("Destructive command detected. Use /shell directly to confirm.")
            return
        await reply(f"Running: `{shell_command}`")
        gen = cmd_process.executor.run_shell(shell_command, timeout=int(os.environ.get("DEFAULT_TIMEOUT", "30")))
        async for chunk in gen:
            if chunk.strip():
                _record_recent_output(chunk)
                await reply(chunk[:4000])
        return

    if command == "/schedule":
        if len(args) < 2:
            await reply("Error: /schedule requires <alias> <time>.")
            return
        alias, time_spec = args[0], args[1]
        if cmd_process._get_script(alias) is None:
            await reply(f"Unknown alias: {alias}")
            return
        target_time = cmd_schedule._parse_time_spec(time_spec)
        if not target_time:
            await reply("Invalid time format. Use HH:MM, Xm, or Xh.")
            return
        sid = await scheduler.schedule(alias, target_time, _scheduled_run_callback)
        await reply(f"Scheduled {alias} for {target_time.strftime('%H:%M')} (ID: {sid}).")
        return

    if command == "/open":
        if not args:
            await reply("Error: /open requires an app name.")
            return
        await reply(_open_app_by_name(" ".join(args)))
        return

    if command == "/close_apps":
        if not args:
            await reply("Error: /close_apps requires at least one app name.")
            return
        # Reuse the app_control-backed helper for consistent behavior.
        from bot import app_control

        close_result = app_control.close_apps(args)
        await reply(close_result.message)
        return

    if command == "/status":
        processes = process_registry.list_processes()
        if not processes:
            await reply("No running processes.")
        else:
            lines = ["Running processes:\n"]
            for proc in processes:
                lines.append(f"PID {proc['pid']} — {proc['alias']} — {_format_runtime(proc['runtime_seconds'])}")
            await reply("\n".join(lines))
        return

    if command == "/kill":
        if not args:
            await reply("Error: /kill requires a PID.")
            return
        try:
            pid = int(args[0])
        except ValueError:
            await reply("Error: PID must be an integer.")
            return
        await reply(await process_registry.kill(pid))
        return

    if command == "/sysinfo":
        await reply(_format_system_snapshot())
        return

    if command == "/screenshot":
        class _TempCtx:
            def __init__(self, screenshot_args: list[str]) -> None:
                self.args = screenshot_args

        await cmd_fs.handle_screenshot(update, _TempCtx(args))
        return

    if command == "/ls":
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
        return

    if command == "/find":
        if not args:
            await reply("Error: /find requires a pattern.")
            return
        pattern = args[0]
        search_root = Path(args[1]) if len(args) > 1 else get_cwd()
        try:
            matches = []
            for match in search_root.rglob(pattern):
                matches.append(str(match))
                if len(matches) >= 50:
                    break
            await reply("\n".join(matches)[:4000] if matches else "No matches found.")
        except Exception as exc:
            await reply(f"Error: {exc}")
        return

    if command == "/tail":
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
        return

    if command == "/copy":
        if not args:
            await reply("Error: /copy requires text.")
            return
        import pyperclip

        text = " ".join(args)
        pyperclip.copy(text)
        await reply(f"Copied to clipboard ({len(text)} chars).")
        return

    if command == "/paste":
        import pyperclip

        content = pyperclip.paste()
        await reply(content[:4000] if content else "Clipboard is empty.")
        return

    if command == "/sleep":
        await reply("Putting PC to sleep...")
        os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
        return

    if command == "/shutdown":
        try:
            minutes = int(args[0]) if args else 0
        except ValueError:
            await reply("Error: minutes must be an integer.")
            return
        os.system(f"shutdown /s /t {minutes * 60}")
        await reply(f"Shutdown scheduled in {minutes} minutes.")
        return

    if command == "/reboot":
        try:
            minutes = int(args[0]) if args else 0
        except ValueError:
            await reply("Error: minutes must be an integer.")
            return
        os.system(f"shutdown /r /t {minutes * 60}")
        await reply(f"Reboot scheduled in {minutes} minutes.")
        return

    if command == "/remind":
        if len(args) < 2:
            await reply("Error: /remind requires <time> <message>.")
            return
        delay = cmd_schedule._parse_delay(args[0])
        if delay is None:
            await reply("Invalid time. Use Xm or Xh.")
            return
        msg = " ".join(args[1:])
        await reply(f"Reminder set for {args[0]} from now.")

        async def _fire() -> None:
            await asyncio.sleep(delay)
            from bot import notifier

            await notifier.send(f"Reminder: {msg}")

        asyncio.create_task(_fire())
        return

    if command == "/watches":
        items = watchdog.list_watches()
        if not items:
            await reply("No active watchers.")
        else:
            lines = ["Active watchers:\n"] + [f"{w['id']} — {w['type']}: {w['target']}" for w in items]
            await reply("\n".join(lines))
        return

    if command == "/schedules":
        items = scheduler.list_schedules()
        if not items:
            await reply("No pending scheduled runs.")
        else:
            lines = ["Pending schedules:\n"] + [f"{s['id']} — {s['alias']} at {s['run_at']}" for s in items]
            await reply("\n".join(lines))
        return

    if command == "/history":
        try:
            n = int(args[0]) if args else 20
        except ValueError:
            n = 20
        rows = await db.get_recent_conversations(n)
        if not rows:
            await reply("No conversation history yet.")
        else:
            lines = [f"Last {len(rows)} entries:\n"]
            for row in rows:
                ts = row["timestamp"][:16] if row.get("timestamp") else "?"
                lines.append(f"[{ts}] {row['role'].upper()}: {row['content'][:200]}")
            await reply("\n".join(lines)[:4000])
        return

    if command == "/runs":
        alias_arg = args[0] if args else None
        rows = await db.get_run_history(alias=alias_arg, limit=10)
        if not rows:
            await reply("No runs recorded yet.")
        else:
            lines = [f"Last {len(rows)} run(s):\n"]
            for row in rows:
                code = row.get("exit_code")
                icon = "✅" if code == 0 else "❌" if code is not None else "?"
                runtime = _format_runtime(row["runtime_seconds"]) if row.get("runtime_seconds") else "?"
                lines.append(f"{icon} {row['alias']} — {runtime}")
            await reply("\n".join(lines)[:4000])
        return

    await reply(f"Command '{command}' is not yet supported via /ask.")
