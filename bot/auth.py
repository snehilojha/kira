"""Authentication decorator for Telegram bot handlers.

Loads allowed user IDs from environment and provides a decorator
that silently ignores messages from unauthorized users.
"""

import os
import logging
from functools import wraps
from typing import Callable, Any

from telegram import Update
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)

# Loaded once at import time from .env (already loaded by main.py)
ALLOWED_USER_IDS: set[int] = set()


def load_allowed_users() -> None:
    """Parse ALLOWED_USER_IDS from environment into the module-level set."""
    raw = os.environ.get("ALLOWED_USER_IDS", "")
    ALLOWED_USER_IDS.clear()
    for uid in raw.split(","):
        uid = uid.strip()
        if uid:
            try:
                ALLOWED_USER_IDS.add(int(uid))
            except ValueError:
                logger.warning("Invalid user ID in ALLOWED_USER_IDS: %s", uid)


def require_auth(handler: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that silently drops updates from non-whitelisted users.

    Unknown user IDs are logged but receive no response, so the bot's
    existence is not revealed to strangers.
    """

    @wraps(handler)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args: Any, **kwargs: Any) -> Any:
        user = update.effective_user
        if user is None or user.id not in ALLOWED_USER_IDS:
            user_id = user.id if user else "unknown"
            logger.warning(
                "Unauthorized access attempt from user_id=%s, message=%s",
                user_id,
                update.message.text if update.message else "<no message>",
            )
            return  # silent ignore
        return await handler(update, context, *args, **kwargs)

    return wrapper
