"""Filesystem command handlers: cd, ls, find, tail, mkdir, move, copy/paste, screenshot, getfile, putfile."""

from __future__ import annotations

import asyncio
import io
import logging
import shutil
from pathlib import Path

import mss
import mss.tools
import pyperclip

from telegram import Update
from telegram.ext import ContextTypes

from bot.auth import require_auth

logger = logging.getLogger(__name__)

# Telegram bot API hard limit for file downloads is 20 MB for regular bots.
_TG_MAX_DOWNLOAD_MB = 20

# ── Bot CWD ───────────────────────────────────────────────────────

_CWD: Path = Path.cwd()


def get_cwd() -> Path:
    return _CWD


def set_cwd(path: Path) -> None:
    global _CWD
    _CWD = path


# ── Size formatter ────────────────────────────────────────────────

def _format_size(n_bytes: int) -> str:
    if n_bytes < 1024:
        return f"{n_bytes} B"
    if n_bytes < 1024 ** 2:
        return f"{n_bytes / 1024:.1f} KB"
    if n_bytes < 1024 ** 3:
        return f"{n_bytes / (1024 ** 2):.1f} MB"
    return f"{n_bytes / (1024 ** 3):.1f} GB"


# ── File transfer ─────────────────────────────────────────────────

@require_auth
async def handle_getfile(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/getfile <path> — send any file from the PC to your phone."""
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

    if size_bytes > 50 * 1024 * 1024:
        await update.message.reply_text(
            f"❌ File too large to send via Telegram bot API.\n"
            f"  Size: {size_str} (limit: 50 MB)\n"
            f"  File: {filepath.name}"
        )
        return

    await update.message.reply_text(f"📤 Sending {filepath.name} ({size_str})...")

    try:
        with open(filepath, "rb") as fh:
            await update.message.reply_document(
                document=fh,
                filename=filepath.name,
                caption=f"📄 {filepath.name}\n📦 {size_str}\n📁 {filepath.parent}",
            )
        logger.info("getfile: sent %s (%s) to user %s", filepath, size_str, update.effective_user.id)
    except Exception as exc:
        await update.message.reply_text(f"❌ Failed to send file: {exc}")


@require_auth
async def handle_putfile(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/putfile [path] — save a file sent from your phone to the PC."""
    msg = update.message

    attachment = (
        msg.document
        or msg.video
        or msg.audio
        or msg.voice
        or msg.animation
        or (msg.photo[-1] if msg.photo else None)
    )

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

    original_name: str
    file_size: int | None = None
    if hasattr(attachment, "file_name") and attachment.file_name:
        original_name = attachment.file_name
    elif hasattr(attachment, "file_unique_id"):
        ext = ""
        if hasattr(attachment, "mime_type") and attachment.mime_type:
            ext = "." + attachment.mime_type.split("/")[-1]
        original_name = f"received_{attachment.file_unique_id}{ext}"
    else:
        original_name = "received_file"

    if hasattr(attachment, "file_size"):
        file_size = attachment.file_size

    if file_size and file_size > _TG_MAX_DOWNLOAD_MB * 1024 * 1024:
        await msg.reply_text(
            f"❌ File too large to download via bot API.\n"
            f"  Size: {_format_size(file_size)} (limit: {_TG_MAX_DOWNLOAD_MB} MB)"
        )
        return

    path_str = ""
    if context.args:
        path_str = " ".join(context.args)
    elif source_msg.caption:
        cap = source_msg.caption.strip()
        if cap.lower().startswith("/putfile"):
            path_str = cap[len("/putfile"):].strip()

    if path_str:
        save_path = Path(path_str)
        if not save_path.suffix and not save_path.exists():
            save_path = save_path / original_name
        elif save_path.is_dir():
            save_path = save_path / original_name
    else:
        downloads_dir = Path(__file__).resolve().parent.parent / "downloads"
        save_path = downloads_dir / original_name

    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.exists():
        await msg.reply_text(f"⚠️ File already exists at {save_path} — overwriting.")

    size_str = _format_size(file_size) if file_size else "unknown size"
    await msg.reply_text(f"📥 Receiving {original_name} ({size_str})...")

    try:
        tg_file = await attachment.get_file()
        await tg_file.download_to_drive(str(save_path))
        final_size = _format_size(save_path.stat().st_size)
        await msg.reply_text(f"✅ Saved to:\n{save_path}\n\n📦 Size: {final_size}")
        logger.info("putfile: saved %s (%s) from user %s", save_path, final_size, update.effective_user.id)
    except Exception as exc:
        await msg.reply_text(f"❌ Failed to save file: {exc}")


# ── Directory commands ────────────────────────────────────────────

@require_auth
async def handle_cd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/cd <path> — change the bot's working directory."""
    if not context.args:
        await update.message.reply_text(f"📍 Current directory:\n{_CWD}")
        return

    raw = " ".join(context.args)

    if raw == "~" or raw.startswith("~/") or raw.startswith("~\\"):
        target = Path.home() / raw[2:].lstrip("/\\")
    else:
        candidate = Path(raw)
        target = candidate if candidate.is_absolute() else _CWD / candidate

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
    """/ls [path] — list directory contents."""
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

        all_entries = list(target.iterdir())
        if len(all_entries) > 100:
            lines.append(f"\n... and {len(all_entries) - 100} more")

        await update.message.reply_text("\n".join(lines)[:4000])
    except PermissionError:
        await update.message.reply_text(f"❌ Permission denied: {target}")


@require_auth
async def handle_find(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/find <pattern> [path] — find files matching a glob pattern."""
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
    """/tail <path> [n] — last N lines of a file (default 20)."""
    if not context.args:
        await update.message.reply_text("Usage: /tail <path> [n]")
        return

    n = 20
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
    """/mkdir <path> — create a directory (parents included)."""
    if not context.args:
        await update.message.reply_text("Usage: /mkdir <path>")
        return
    target = Path(" ".join(context.args))
    target.mkdir(parents=True, exist_ok=True)
    await update.message.reply_text(f"✅ Created: {target}")


@require_auth
async def handle_move(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/move <src> <dst> — move a file or directory."""
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
    """/copy <text> — set PC clipboard."""
    if not context.args:
        await update.message.reply_text("Usage: /copy <text>")
        return
    text = " ".join(context.args)
    pyperclip.copy(text)
    await update.message.reply_text(f"✅ Copied to clipboard ({len(text)} chars)")


@require_auth
async def handle_paste(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/paste — send back clipboard contents."""
    content = pyperclip.paste()
    if not content:
        await update.message.reply_text("Clipboard is empty.")
        return
    await update.message.reply_text(f"📋 Clipboard:\n{content[:4000]}")


# ── Screenshot ────────────────────────────────────────────────────

@require_auth
async def handle_screenshot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/screenshot [n] — capture screen and send as image."""
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
                monitor = sct.monitors[monitor_index + 1]
            else:
                monitor = sct.monitors[0]

            screenshot = sct.grab(monitor)
            png_bytes = mss.tools.to_png(screenshot.rgb, screenshot.size)

        bio = io.BytesIO(png_bytes)
        bio.name = "screenshot.png"
        await update.message.reply_photo(photo=bio)
        logger.info("Screenshot sent to user %s", update.effective_user.id)
    except Exception as exc:
        await update.message.reply_text(f"❌ Screenshot failed: {exc}")
