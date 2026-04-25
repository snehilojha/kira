"""Entry point for the telegram-runner bot.

Loads environment, configures logging, registers all command handlers,
and starts the long-polling event loop.
"""

import logging
import os
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv
from telegram import Update
from telegram.error import NetworkError, TimedOut
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters

from bot import db
from bot import handlers
from bot import job_monitor
from bot import local_voice
from bot import memory
from bot import mode
from bot import monitor
from bot import observer
from bot import notifier
from bot import overlay
from bot import ui_mode
from bot import scheduler
from bot import task_state
from bot import watchdog
from bot import world
from bot.auth import load_allowed_users

# ── Paths ─────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"
_LOG_PATH = _PROJECT_ROOT / "logs" / "bot.log"


def _setup_logging() -> None:
    """Configure rotating file + console logging."""
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    level = os.environ.get("LOG_LEVEL", "INFO").upper()

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Rotating file handler: 5 MB per file, keep 3 backups
    file_handler = RotatingFileHandler(
        _LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    root_logger.addHandler(console_handler)


def _wait_for_network(max_wait: int = 120, check_interval: int = 5) -> bool:
    """Block until a basic TCP connection to Telegram's API succeeds.

    Returns True once reachable, False if max_wait seconds elapse.
    """
    import socket
    deadline = time.monotonic() + max_wait
    logger = logging.getLogger(__name__)
    while time.monotonic() < deadline:
        try:
            socket.setdefaulttimeout(5)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(("api.telegram.org", 443))
            sock.close()
            return True
        except OSError:
            remaining = int(deadline - time.monotonic())
            logger.info(
                "Network not ready yet — retrying in %ds (%ds remaining)",
                check_interval, remaining,
            )
            time.sleep(check_interval)
    return False


async def _error_handler(update: object, context) -> None:
    """Log NetworkError and TimedOut silently; re-raise all other exceptions."""
    logger = logging.getLogger(__name__)
    exc = context.error
    if isinstance(exc, (NetworkError, TimedOut)):
        logger.warning("Transient network error (will retry): %s", exc)
        return
    logger.error("Unhandled exception in handler", exc_info=exc)


def _build_ptb_app(token: str):
    """Build and return the PTB Application with all handlers registered."""

    async def _post_init(application) -> None:
        await db.init_db()
        await handlers.reload_reminders()
        await scheduler.reload_from_db(handlers._scheduled_run_callback)
        await watchdog.reload_from_db()
        await job_monitor.reload_from_db()
        interrupted = task_state.mark_interrupted_tasks(
            "Bot restarted before the task reached a terminal state."
        )
        if interrupted:
            await notifier.send(
                "Kira restarted with unfinished complex task(s). "
                f"Marked {len(interrupted)} task(s) as interrupted; no actions were replayed. "
                "Use /tasks to inspect them."
            )
        application.create_task(monitor.start_monitor())
        application.create_task(memory.start_daily_summariser())
        application.create_task(observer.start())
        application.create_task(observer.start_fast_loop())
        application.create_task(job_monitor.start())
        application.create_task(mode.start())
        application.create_task(world.start())
        application.create_task(local_voice.start_as_task())

    app = ApplicationBuilder().token(token).post_init(_post_init).build()

    command_map = {
        "run": handlers.handle_run,
        "shell": handlers.handle_shell,
        "chain": handlers.handle_chain,
        "status": handlers.handle_status,
        "kill": handlers.handle_kill,
        "schedule": handlers.handle_schedule,
        "schedules": handlers.handle_schedules,
        "unschedule": handlers.handle_unschedule,
        "sysinfo": handlers.handle_sysinfo,
        "getfile": handlers.handle_getfile,
        "putfile": handlers.handle_putfile,
        "cd": handlers.handle_cd,
        "ls": handlers.handle_ls,
        "find": handlers.handle_find,
        "tail": handlers.handle_tail,
        "mkdir": handlers.handle_mkdir,
        "move": handlers.handle_move,
        "copy": handlers.handle_copy,
        "paste": handlers.handle_paste,
        "open": handlers.handle_open,
        "screenshot": handlers.handle_screenshot,
        "sleep": handlers.handle_sleep,
        "shutdown": handlers.handle_shutdown,
        "reboot": handlers.handle_reboot,
        "abort_shutdown": handlers.handle_abort_shutdown,
        "watch": handlers.handle_watch,
        "watches": handlers.handle_watches,
        "unwatch": handlers.handle_unwatch,
        "remind": handlers.handle_remind,
        "list_apps": handlers.handle_list_apps,
        "close_apps": handlers.handle_close_apps,
        "help": handlers.handle_help,
        "ask": handlers.handle_ask,
        "tasks": handlers.handle_tasks,
        "task": handlers.handle_task,
        "history": handlers.handle_history,
        "runs": handlers.handle_runs,
        "summarise": handlers.handle_summarise,
        "recall": handlers.handle_recall,
        "jobs": handlers.handle_jobs,
        "canceljob": handlers.handle_cancel_job,
        "pausejob": handlers.handle_pause_job,
        "resumejob": handlers.handle_resume_job,
        "mode": handlers.handle_mode,
    }
    for name, handler in command_map.items():
        app.add_handler(CommandHandler(name, handler))

    app.add_error_handler(_error_handler)

    async def unified_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        data = query.data or ""
        if data.startswith("ask_") or data.startswith("brain_"):
            await handlers.handle_ask_callback(update, context)
        else:
            await handlers.handle_callback_query(update, context)

    app.add_handler(CallbackQueryHandler(unified_callback_handler))

    _media_filter = (
        filters.Document.ALL | filters.PHOTO | filters.VIDEO
        | filters.AUDIO | filters.VOICE | filters.ANIMATION
    )
    app.add_handler(MessageHandler(
        _media_filter & filters.CaptionRegex(r"(?i)^/putfile"),
        handlers.handle_putfile,
    ))
    app.add_handler(MessageHandler(
        filters.VOICE & ~filters.CaptionRegex(r"(?i)^/putfile"),
        handlers.handle_voice,
    ))
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        handlers.handle_text,
    ))

    return app


def _run_bot(ptb_app) -> None:
    """Run PTB polling loop on a background thread with its own event loop."""
    import asyncio
    logger = logging.getLogger(__name__)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    logger.info("Bot is live — polling for updates")
    try:
        ptb_app.run_polling(drop_pending_updates=True)
    finally:
        loop.close()


def main() -> None:
    """Load config, start Qt overlay on main thread, run bot on background thread."""
    # 1. Load environment
    load_dotenv(_ENV_PATH)
    token = os.environ.get("BOT_TOKEN")
    if not token or token == "your_telegram_bot_token_here":
        print("ERROR: Set a valid BOT_TOKEN in .env before starting.")
        sys.exit(1)

    # 2. Logging
    _setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("telegram-runner starting up")

    # 3. Wait for network
    if not _wait_for_network():
        logger.error("Network unavailable after 120s — aborting startup")
        sys.exit(1)

    # 4. Init shared modules
    load_allowed_users()
    notifier.init()
    handlers.load_config()

    # 5. Build PTB app
    ptb_app = _build_ptb_app(token)

    # 6. Start bot on a background thread so main thread is free for Qt
    bot_thread = threading.Thread(
        target=_run_bot, args=(ptb_app,), daemon=True, name="kira-bot"
    )
    bot_thread.start()

    # 7. Start Qt overlay on the main thread (Qt requires this)
    overlay.start_on_main_thread()


if __name__ == "__main__":
    main()
