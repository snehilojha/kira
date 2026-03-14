"""PID and file watchdog monitors.

Each watcher runs as its own ``asyncio.Task``, polling every 5 seconds.
When a watched process dies or a watched file changes, a Telegram alert
is sent via ``notifier.send()``.

Watchers are in-memory only — they do not survive a bot restart.
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
    watch_id = f"watch-{_next_id}"
    _next_id += 1

    display_label = label or f"PID {pid}"

    async def _poller() -> None:
        while True:
            await asyncio.sleep(_POLL_INTERVAL)
            if not psutil.pid_exists(pid):
                logger.info("Watchdog: PID %d is gone (%s)", pid, display_label)
                await notifier.send(f"⚠️ Watchdog: process {display_label} (PID {pid}) has died.")
                WATCHES.pop(watch_id, None)
                return

    task = asyncio.create_task(_poller())
    WATCHES[watch_id] = WatchEntry(
        id=watch_id, watch_type="pid", target=str(pid), label=display_label, task=task
    )
    logger.info("Watchdog registered: id=%s type=pid target=%d", watch_id, pid)
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
    watch_id = f"watch-{_next_id}"
    _next_id += 1

    display_label = label or path
    resolved = Path(path)

    if not resolved.exists():
        return f"File not found: {path}"

    initial_mtime = resolved.stat().st_mtime

    async def _poller() -> None:
        last_mtime = initial_mtime
        while True:
            await asyncio.sleep(_POLL_INTERVAL)
            try:
                current_mtime = resolved.stat().st_mtime
            except FileNotFoundError:
                await notifier.send(f"⚠️ Watchdog: file {display_label} was deleted.")
                WATCHES.pop(watch_id, None)
                return

            if current_mtime != last_mtime:
                last_mtime = current_mtime
                await notifier.send(f"📝 Watchdog: file {display_label} was modified.")

    task = asyncio.create_task(_poller())
    WATCHES[watch_id] = WatchEntry(
        id=watch_id, watch_type="file", target=path, label=display_label, task=task
    )
    logger.info("Watchdog registered: id=%s type=file target=%s", watch_id, path)
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
    logger.info("Watchdog cancelled: id=%s type=%s target=%s", watch_id, entry.watch_type, entry.target)
    return f"Removed watchdog {watch_id} ({entry.watch_type}: {entry.target})."
