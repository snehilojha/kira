"""Outbound Telegram message helper.

Any module that needs to push a proactive message to the user calls
``notifier.send()``. This centralises all outbound logic in one place.
"""

import os
import logging

import httpx

logger = logging.getLogger(__name__)

# Populated by init() at bot startup
_BOT_TOKEN: str = ""
_CHAT_ID: str = ""

# Telegram message hard limit is 4096 characters
_MAX_MESSAGE_LENGTH = 4000


def init() -> None:
    """Load BOT_TOKEN and CHAT_ID from the environment.

    Must be called once after dotenv is loaded (i.e. from main.py).
    """
    global _BOT_TOKEN, _CHAT_ID
    _BOT_TOKEN = os.environ["BOT_TOKEN"]
    _CHAT_ID = os.environ["CHAT_ID"]


async def send(message: str, parse_mode: str | None = None) -> bool:
    """Send a Telegram message to CHAT_ID.

    Args:
        message: The text to send. Truncated to 4000 chars if longer.
        parse_mode: Optional Telegram parse mode (e.g. "HTML", "Markdown").

    Returns:
        True if the message was sent successfully, False otherwise.
    """
    if not _BOT_TOKEN or not _CHAT_ID:
        logger.error("notifier not initialised — call notifier.init() first")
        return False

    text = message[:_MAX_MESSAGE_LENGTH]
    payload: dict = {"chat_id": _CHAT_ID, "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode

    url = f"https://api.telegram.org/bot{_BOT_TOKEN}/sendMessage"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, timeout=15)
            if resp.status_code != 200:
                logger.error("Telegram API error %s: %s", resp.status_code, resp.text)
                return False
            return True
    except httpx.HTTPError as exc:
        logger.error("Failed to send Telegram message: %s", exc)
        return False
