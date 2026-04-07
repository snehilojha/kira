"""PID and file watchdog monitors.

Each watcher runs as its own ``asyncio.Task``, polling every 5 seconds.
When a watched process dies or a watched file changes, a Telegram alert
is sent via ``notifier.send()``.

Watchers are persisted to SQLite so they survive a bot restart.
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import psutil

from bot import notifier

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 5  # seconds
_next_id: int = 1


@dataclass
class WatchEntry:
    """Metadata for an active watchdog monitor."""

    id: str
    watch_type: str  # "pid" or "file"
    target: str  # PID as string, or file path
    label: str
    task: asyncio.Task
    db_id: int | None = None  # rowid in the watches table


# Module-level mutable state
WATCHES: dict[str, WatchEntry] = {}


async def watch_pid(pid: int, label: str = "") -> str:
    """Poll whether a process is alive; alert when it dies.

    Args:
        pid: System PID to monitor.
        label: Optional human-readable description.

    Returns:
        The monitor ID.
    """
    global _next_id
    from bot import db

    watch_id = f"watch-{_next_id}"
    _next_id += 1

    display_label = label or f"PID {pid}"

    try:
        db_id = await db.save_watch("pid", str(pid))
    except Exception as exc:
        logger.warning("Failed to persist watch to DB: %s", exc)
        db_id = None

    async def _poller() -> None:
        while True:
            await asyncio.sleep(_POLL_INTERVAL)
            if not psutil.pid_exists(pid):
                logger.info("Watchdog: PID %d is gone (%s)", pid, display_label)
                entry = WATCHES.pop(watch_id, None)
                if entry and entry.db_id is not None:
                    try:
                        await db.mark_watch_fired(entry.db_id)
                    except Exception as exc:
                        logger.warning("Failed to mark watch fired in DB: %s", exc)
                await notifier.send(f"⚠️ Watchdog: process {display_label} (PID {pid}) has died.")
                return

    task = asyncio.create_task(_poller())
    WATCHES[watch_id] = WatchEntry(
        id=watch_id, watch_type="pid", target=str(pid),
        label=display_label, task=task, db_id=db_id
    )
    logger.info("Watchdog registered: id=%s type=pid target=%d db_id=%s", watch_id, pid, db_id)
    return watch_id


async def watch_file(path: str, label: str = "") -> str:
    """Poll a file's mtime; alert when it changes.

    Args:
        path: Absolute path to the file to monitor.
        label: Optional human-readable description.

    Returns:
        The monitor ID.
    """
    global _next_id
    from bot import db

    watch_id = f"watch-{_next_id}"
    _next_id += 1

    display_label = label or path
    resolved = Path(path)

    if not resolved.exists():
        return f"File not found: {path}"

    initial_mtime = resolved.stat().st_mtime

    try:
        db_id = await db.save_watch("file", path)
    except Exception as exc:
        logger.warning("Failed to persist watch to DB: %s", exc)
        db_id = None

    async def _poller() -> None:
        last_mtime = initial_mtime
        while True:
            await asyncio.sleep(_POLL_INTERVAL)
            try:
                current_mtime = resolved.stat().st_mtime
            except FileNotFoundError:
                entry = WATCHES.pop(watch_id, None)
                if entry and entry.db_id is not None:
                    try:
                        await db.mark_watch_fired(entry.db_id)
                    except Exception as exc:
                        logger.warning("Failed to mark watch fired in DB: %s", exc)
                await notifier.send(f"⚠️ Watchdog: file {display_label} was deleted.")
                return

            if current_mtime != last_mtime:
                last_mtime = current_mtime
                await notifier.send(f"📝 Watchdog: file {display_label} was modified.")

    task = asyncio.create_task(_poller())
    WATCHES[watch_id] = WatchEntry(
        id=watch_id, watch_type="file", target=path,
        label=display_label, task=task, db_id=db_id
    )
    logger.info("Watchdog registered: id=%s type=file target=%s db_id=%s", watch_id, path, db_id)
    return watch_id


async def reload_from_db() -> int:
    """Reload unfired watches from SQLite after a bot restart.

    Should be called once from ``main.py`` post_init.

    Returns:
        Number of watches restored.
    """
    from bot import db

    rows = await db.get_pending_watches()
    restored = 0
    for row in rows:
        watch_type = row["type"]
        target = row["target"]
        db_id = row["id"]

        if watch_type == "pid":
            pid = int(target)
            if not psutil.pid_exists(pid):
                # Process already gone — mark fired and skip
                await db.mark_watch_fired(db_id)
                logger.info("Dropping dead-PID watch db_id=%d pid=%d", db_id, pid)
                continue
            watch_id = await _restore_pid_watch(pid, db_id)
        elif watch_type == "file":
            if not Path(target).exists():
                await db.mark_watch_fired(db_id)
                logger.info("Dropping missing-file watch db_id=%d path=%s", db_id, target)
                continue
            watch_id = await _restore_file_watch(target, db_id)
        else:
            logger.warning("Unknown watch type %r in DB — skipping db_id=%d", watch_type, db_id)
            continue

        logger.info("Restored watch id=%s type=%s target=%s from db_id=%d", watch_id, watch_type, target, db_id)
        restored += 1

    return restored


async def _restore_pid_watch(pid: int, db_id: int) -> str:
    """Re-create a PID watch from a DB row without writing a new DB entry."""
    global _next_id
    from bot import db

    watch_id = f"watch-{_next_id}"
    _next_id += 1
    display_label = f"PID {pid}"

    async def _poller() -> None:
        while True:
            await asyncio.sleep(_POLL_INTERVAL)
            if not psutil.pid_exists(pid):
                logger.info("Watchdog (restored): PID %d is gone", pid)
                WATCHES.pop(watch_id, None)
                try:
                    await db.mark_watch_fired(db_id)
                except Exception as exc:
                    logger.warning("Failed to mark restored watch fired: %s", exc)
                await notifier.send(f"⚠️ Watchdog: process {display_label} (PID {pid}) has died.")
                return

    task = asyncio.create_task(_poller())
    WATCHES[watch_id] = WatchEntry(
        id=watch_id, watch_type="pid", target=str(pid),
        label=display_label, task=task, db_id=db_id
    )
    return watch_id


async def _restore_file_watch(path: str, db_id: int) -> str:
    """Re-create a file watch from a DB row without writing a new DB entry."""
    global _next_id
    from bot import db

    watch_id = f"watch-{_next_id}"
    _next_id += 1
    resolved = Path(path)
    display_label = path

    try:
        initial_mtime = resolved.stat().st_mtime
    except FileNotFoundError:
        initial_mtime = 0.0

    async def _poller() -> None:
        last_mtime = initial_mtime
        while True:
            await asyncio.sleep(_POLL_INTERVAL)
            try:
                current_mtime = resolved.stat().st_mtime
            except FileNotFoundError:
                WATCHES.pop(watch_id, None)
                try:
                    await db.mark_watch_fired(db_id)
                except Exception as exc:
                    logger.warning("Failed to mark restored watch fired: %s", exc)
                await notifier.send(f"⚠️ Watchdog: file {display_label} was deleted.")
                return
            if current_mtime != last_mtime:
                last_mtime = current_mtime
                await notifier.send(f"📝 Watchdog: file {display_label} was modified.")

    task = asyncio.create_task(_poller())
    WATCHES[watch_id] = WatchEntry(
        id=watch_id, watch_type="file", target=path,
        label=display_label, task=task, db_id=db_id
    )
    return watch_id


def list_watches() -> list[dict]:
    """Return all active watchers as serialisable dicts."""
    return [
        {"id": w.id, "type": w.watch_type, "target": w.target, "label": w.label}
        for w in WATCHES.values()
    ]


def cancel(watch_id: str) -> str:
    """Remove a watchdog monitor by ID.

    Returns:
        A human-readable result string.
    """
    entry = WATCHES.pop(watch_id, None)
    if entry is None:
        return f"No watchdog with ID {watch_id}."
    entry.task.cancel()

    if entry.db_id is not None:
        asyncio.get_event_loop().create_task(_mark_fired_bg(entry.db_id))

    logger.info("Watchdog cancelled: id=%s type=%s target=%s", watch_id, entry.watch_type, entry.target)
    return f"Removed watchdog {watch_id} ({entry.watch_type}: {entry.target})."


async def _mark_fired_bg(db_id: int) -> None:
    """Background coroutine to mark a cancelled watch as fired in DB."""
    from bot import db
    try:
        await db.mark_watch_fired(db_id)
    except Exception as exc:
        logger.warning("Failed to mark cancelled watch fired in DB: %s", exc)
