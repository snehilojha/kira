"""Entry point for the telegram-runner bot.

Loads environment, configures logging, registers all command handlers,
and starts the long-polling event loop.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, MessageHandler, filters

from bot import handlers
from bot import notifier
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


def main() -> None:
    """Load config, register handlers, start polling."""
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

    # 3. Init shared modules
    load_allowed_users()
    notifier.init()
    handlers.load_config()

    # 4. Build application
    app = ApplicationBuilder().token(token).build()

    # 5. Register command handlers
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
        "screenshot": handlers.handle_screenshot,
        "sleep": handlers.handle_sleep,
        "shutdown": handlers.handle_shutdown,
        "reboot": handlers.handle_reboot,
        "abort_shutdown": handlers.handle_abort_shutdown,
        "watch": handlers.handle_watch,
        "watches": handlers.handle_watches,
        "unwatch": handlers.handle_unwatch,
        "remind": handlers.handle_remind,
        "help": handlers.handle_help,
    }

    for name, handler in command_map.items():
        app.add_handler(CommandHandler(name, handler))

    # 6. Inline button callback handler (confirmations)
    app.add_handler(CallbackQueryHandler(handlers.handle_callback_query))

    # 7a. Media messages with /putfile caption — CommandHandler won't fire on these
    _media_filter = (
        filters.Document.ALL
        | filters.PHOTO
        | filters.VIDEO
        | filters.AUDIO
        | filters.VOICE
        | filters.ANIMATION
    )
    app.add_handler(
        MessageHandler(
            _media_filter & filters.CaptionRegex(r"(?i)^/putfile"),
            handlers.handle_putfile,
        )
    )

    # 8. Start polling
    logger.info("Bot is live — polling for updates")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
