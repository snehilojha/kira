"""Scheduler, watchdog, and reminder command handlers."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta

import psutil

from telegram import Update
from telegram.ext import ContextTypes

from bot.auth import require_auth
from bot import db
from bot import notifier
from bot import scheduler
from bot import watchdog

logger = logging.getLogger(__name__)


# ── Time parsing ──────────────────────────────────────────────────

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


def _parse_time_spec(spec: str) -> datetime | None:
    """Parse a time spec like ``14:30``, ``30m``, ``2h`` into a datetime."""
    delay = _parse_delay(spec)
    if delay is not None:
        return datetime.now() + timedelta(seconds=delay)
    try:
        hour, minute = spec.split(":")
        target = datetime.now().replace(hour=int(hour), minute=int(minute), second=0, microsecond=0)
        if target < datetime.now():
            target += timedelta(days=1)
        return target
    except (ValueError, IndexError):
        return None


# ── Schedule commands ─────────────────────────────────────────────

@require_auth
async def handle_schedule(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/schedule <alias> <HH:MM|Xm|Xh> — queue a script for later."""
    if not context.args or len(context.args) < 2:
        await update.message.reply_text("Usage: /schedule <alias> <HH:MM|Xm|Xh>")
        return

    alias = context.args[0]
    time_spec = context.args[1]

    # Defer import to avoid circular at module load; cmd_process owns script config
    from bot import cmd_process
    if cmd_process._get_script(alias) is None:
        await update.message.reply_text(f"Unknown alias: {alias}")
        return

    run_at = _parse_time_spec(time_spec)
    if run_at is None:
        await update.message.reply_text("Invalid time. Use HH:MM, Xm, or Xh.")
        return

    sid = await scheduler.schedule(alias, run_at, cmd_process.scheduled_run_callback)
    await update.message.reply_text(f"✅ Scheduled {alias} at {run_at.strftime('%H:%M:%S')} (ID: {sid})")


@require_auth
async def handle_schedules(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/schedules — list pending scheduled runs."""
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
    """/unschedule <id> — cancel a pending scheduled run."""
    if not context.args:
        await update.message.reply_text("Usage: /unschedule <id>")
        return
    result = scheduler.cancel(context.args[0])
    await update.message.reply_text(result)


# ── Watchdog commands ─────────────────────────────────────────────

@require_auth
async def handle_watch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/watch pid <pid> or /watch file <path>."""
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
    """/watches — list active watchdog monitors."""
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
    """/unwatch <id> — remove a watchdog monitor."""
    if not context.args:
        await update.message.reply_text("Usage: /unwatch <id>")
        return
    result = watchdog.cancel(context.args[0])
    await update.message.reply_text(result)


# ── Reminders ─────────────────────────────────────────────────────

@require_auth
async def handle_remind(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/remind <Xm|Xh> <message> — send a reminder after delay."""
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
    """Restore pending reminders from the database after a restart."""
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
