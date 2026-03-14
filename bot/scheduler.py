"""Asyncio-based task scheduler for the ``/schedule`` command.

Schedules are in-memory only — they do **not** survive a bot restart.
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
    schedule_id = f"sched-{_next_id}"
    _next_id += 1

    delay = (run_at - datetime.now()).total_seconds()
    if delay < 0:
        delay = 0

    async def _waiter() -> None:
        await asyncio.sleep(delay)
        entry = SCHEDULES.pop(schedule_id, None)
        if entry and not entry.cancelled:
            logger.info("Scheduled run firing: id=%s alias=%s", schedule_id, alias)
            await notifier.send(f"⏰ Scheduled run starting: {alias}")
            try:
                await run_callback(alias)
            except Exception as exc:
                logger.error("Scheduled run failed: %s", exc)
                await notifier.send(f"❌ Scheduled run {alias} failed: {exc}")

    task = asyncio.create_task(_waiter())
    SCHEDULES[schedule_id] = ScheduledRun(
        id=schedule_id, alias=alias, run_at=run_at, task=task
    )
    logger.info("Scheduled id=%s alias=%s at=%s", schedule_id, alias, run_at)
    return schedule_id


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
    logger.info("Cancelled scheduled run id=%s alias=%s", schedule_id, entry.alias)
    return f"Cancelled scheduled run {schedule_id} ({entry.alias})."
