"""Asyncio-based task scheduler for the ``/schedule`` command.

Schedules are persisted to SQLite so they survive a bot restart.
Each scheduled run is an ``asyncio.Task`` stored in a module-level dict
keyed by a short auto-incrementing ID.
"""

import asyncio
import logging
from datetime import datetime
from dataclasses import dataclass, field

from bot import notifier

logger = logging.getLogger(__name__)

_next_id: int = 1


@dataclass
class ScheduledRun:
    """Metadata for a pending scheduled execution."""

    id: str
    alias: str
    run_at: datetime
    task: asyncio.Task
    db_id: int | None = None   # rowid in the schedules table
    cancelled: bool = False


# Module-level mutable state
SCHEDULES: dict[str, ScheduledRun] = {}


async def schedule(
    alias: str,
    run_at: datetime,
    run_callback,
) -> str:
    """Register a script to run at a future time.

    Args:
        alias: Script alias from ``scripts.toml``.
        run_at: When to execute.
        run_callback: An async callable ``(alias) -> None`` that the
            scheduler calls when the time arrives.

    Returns:
        The schedule ID (for display and cancellation).
    """
    global _next_id
    from bot import db

    schedule_id = f"sched-{_next_id}"
    _next_id += 1

    # Persist to DB so restart can reload it
    try:
        db_id = await db.save_schedule(alias, run_at.isoformat())
    except Exception as exc:
        logger.warning("Failed to persist schedule to DB: %s", exc)
        db_id = None

    delay = (run_at - datetime.now()).total_seconds()
    if delay < 0:
        delay = 0

    async def _waiter() -> None:
        await asyncio.sleep(delay)
        entry = SCHEDULES.pop(schedule_id, None)
        if entry and not entry.cancelled:
            logger.info("Scheduled run firing: id=%s alias=%s", schedule_id, alias)
            # Mark fired in DB
            if entry.db_id is not None:
                try:
                    await db.mark_schedule_fired(entry.db_id)
                except Exception as exc:
                    logger.warning("Failed to mark schedule fired in DB: %s", exc)
            await notifier.send(f"⏰ Scheduled run starting: {alias}")
            try:
                await run_callback(alias)
            except Exception as exc:
                logger.error("Scheduled run failed: %s", exc)
                await notifier.send(f"❌ Scheduled run {alias} failed: {exc}")

    task = asyncio.create_task(_waiter())
    SCHEDULES[schedule_id] = ScheduledRun(
        id=schedule_id, alias=alias, run_at=run_at, task=task, db_id=db_id
    )
    logger.info("Scheduled id=%s alias=%s at=%s db_id=%s", schedule_id, alias, run_at, db_id)
    return schedule_id


async def reload_from_db(run_callback) -> int:
    """Reload unfired schedules from SQLite after a bot restart.

    Should be called once from ``main.py`` post_init, before polling starts.

    Args:
        run_callback: The same async callable passed to ``schedule()``.

    Returns:
        Number of schedules restored.
    """
    from bot import db

    rows = await db.get_pending_schedules()
    restored = 0
    for row in rows:
        try:
            run_at = datetime.fromisoformat(row["run_at"])
        except ValueError:
            logger.warning("Skipping schedule id=%d with invalid run_at=%s", row["id"], row["run_at"])
            continue

        if run_at <= datetime.now():
            # Already past — mark fired and skip
            await db.mark_schedule_fired(row["id"])
            logger.info("Dropping expired schedule db_id=%d alias=%s run_at=%s", row["id"], row["alias"], row["run_at"])
            continue

        global _next_id
        schedule_id = f"sched-{_next_id}"
        _next_id += 1

        alias = row["alias"]
        db_id = row["id"]
        delay = (run_at - datetime.now()).total_seconds()

        async def _waiter(sid=schedule_id, al=alias, did=db_id) -> None:
            await asyncio.sleep(delay)
            entry = SCHEDULES.pop(sid, None)
            if entry and not entry.cancelled:
                logger.info("Restored scheduled run firing: id=%s alias=%s", sid, al)
                try:
                    await db.mark_schedule_fired(did)
                except Exception as exc:
                    logger.warning("Failed to mark restored schedule fired: %s", exc)
                await notifier.send(f"⏰ Scheduled run starting: {al}")
                try:
                    await run_callback(al)
                except Exception as exc:
                    logger.error("Restored scheduled run failed: %s", exc)
                    await notifier.send(f"❌ Scheduled run {al} failed: {exc}")

        task = asyncio.create_task(_waiter())
        SCHEDULES[schedule_id] = ScheduledRun(
            id=schedule_id, alias=alias, run_at=run_at, task=task, db_id=db_id
        )
        logger.info("Restored schedule id=%s alias=%s at=%s from db_id=%d", schedule_id, alias, run_at, db_id)
        restored += 1

    return restored


def list_schedules() -> list[dict]:
    """Return all pending scheduled runs as serialisable dicts."""
    return [
        {"id": s.id, "alias": s.alias, "run_at": s.run_at.isoformat()}
        for s in SCHEDULES.values()
        if not s.cancelled
    ]


def cancel(schedule_id: str) -> str:
    """Cancel a pending scheduled run by ID.

    Returns:
        A human-readable result string.
    """
    entry = SCHEDULES.pop(schedule_id, None)
    if entry is None:
        return f"No scheduled run with ID {schedule_id}."
    entry.cancelled = True
    entry.task.cancel()

    # Fire-and-forget DB update — we're in a sync context here
    if entry.db_id is not None:
        asyncio.get_event_loop().create_task(_mark_fired_bg(entry.db_id))

    logger.info("Cancelled scheduled run id=%s alias=%s", schedule_id, entry.alias)
    return f"Cancelled scheduled run {schedule_id} ({entry.alias})."


async def _mark_fired_bg(db_id: int) -> None:
    """Background coroutine to mark a cancelled schedule as fired in DB."""
    from bot import db
    try:
        await db.mark_schedule_fired(db_id)
    except Exception as exc:
        logger.warning("Failed to mark cancelled schedule fired in DB: %s", exc)
