"""Outbound Telegram message helper.

Any module that needs to push a proactive message to the user calls
``notifier.send()``. This centralises all outbound logic in one place.
"""

import asyncio
import os
import logging
import time

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


_VOICE_CONFIRM_TIMEOUT = 30  # seconds


async def confirm_via_telegram(command_preview: str) -> bool:
    """Send an inline Yes/No confirmation to Telegram and wait for the reply.

    Returns True if confirmed, False if cancelled or timed out.
    """
    from bot import handlers

    token = f"vc_{int(time.monotonic() * 1000)}"
    loop = asyncio.get_running_loop()
    future: asyncio.Future[bool] = loop.create_future()
    handlers.register_voice_confirm(token, future)

    keyboard = [
        [
            {"text": "✅ Yes", "callback_data": f"voice_confirm_{token}"},
            {"text": "❌ Cancel", "callback_data": f"voice_cancel_{token}"},
        ]
    ]
    payload = {
        "chat_id": _CHAT_ID,
        "text": f"Voice command wants to run:\n`{command_preview}`\n\nAllow it?",
        "parse_mode": "Markdown",
        "reply_markup": {"inline_keyboard": keyboard},
    }

    url = f"https://api.telegram.org/bot{_BOT_TOKEN}/sendMessage"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, timeout=15)
            if resp.status_code != 200:
                logger.error("Failed to send voice confirm message: %s", resp.text)
                return False
    except httpx.HTTPError as exc:
        logger.error("Failed to send voice confirm message: %s", exc)
        return False

    try:
        return await asyncio.wait_for(future, timeout=_VOICE_CONFIRM_TIMEOUT)
    except asyncio.TimeoutError:
        handlers._PENDING_VOICE_CONFIRMS.pop(token, None)
        await send("⏰ Voice confirmation timed out — command cancelled.")
        return False


async def send_photo(caption: str, png_bytes: bytes) -> int | None:
    """Send a screenshot (PNG bytes) with a caption to CHAT_ID.

    Returns the Telegram message_id on success, None on failure.
    """
    if not _BOT_TOKEN or not _CHAT_ID:
        logger.error("notifier not initialised — call notifier.init() first")
        return None

    url = f"https://api.telegram.org/bot{_BOT_TOKEN}/sendPhoto"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                data={"chat_id": _CHAT_ID, "caption": caption[:1024]},
                files={"photo": ("screenshot.png", png_bytes, "image/png")},
                timeout=20,
            )
            if resp.status_code != 200:
                logger.error("Telegram sendPhoto error %s: %s", resp.status_code, resp.text)
                return None
            return resp.json().get("result", {}).get("message_id")
    except httpx.HTTPError as exc:
        logger.error("Failed to send photo: %s", exc)
        return None


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
