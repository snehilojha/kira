"""PID and file watchdog monitors.

Each watcher runs as its own ``asyncio.Task`` and is event-driven:

- File watches use the ``watchdog`` package (OS filesystem events via a single
  shared observer thread), debounced so an editor's multi-event save fires one
  alert.
- PID watches block on ``psutil.Process.wait`` (a real OS wait, e.g.
  ``WaitForSingleObject`` on Windows) instead of polling.

When a watched process dies or a watched file changes, a Telegram alert is
sent via ``notifier.send()``. Watchers are persisted to SQLite so they survive
a bot restart.

Note: ``from watchdog...`` below refers to the PyPI *watchdog* package in
site-packages, not this module (``bot.watchdog``) — absolute imports don't
resolve to the current package.
"""

import asyncio
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import psutil
from watchdog.events import FileSystemEventHandler

from bot import notifier

logger = logging.getLogger(__name__)

# Only used as the fallback liveness-poll interval if an OS wait is unavailable
# for a PID (e.g. AccessDenied on a protected process).
_POLL_INTERVAL = 5  # seconds
_FILE_DEBOUNCE_SECONDS = 1.0
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
    observed_watch: object | None = None  # watchdog ObservedWatch (file watches)


# Module-level mutable state
WATCHES: dict[str, WatchEntry] = {}

# Single shared filesystem observer, started lazily on the first file watch.
_observer = None
_observer_lock = threading.Lock()


def _get_observer():
    """Return the shared watchdog Observer, starting it on first use."""
    global _observer
    if _observer is None:
        with _observer_lock:
            if _observer is None:
                from watchdog.observers import Observer
                obs = Observer()
                obs.daemon = True
                obs.start()
                _observer = obs
    return _observer


# ── PID watches ───────────────────────────────────────────────────

async def _await_process_exit(pid: int) -> None:
    """Block until the process exits. Returns when the PID is gone."""
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return  # already gone
    while True:
        try:
            await asyncio.to_thread(proc.wait, 30)
            return  # exited
        except psutil.TimeoutExpired:
            continue  # still alive — loop only to bound cancellation latency
        except psutil.NoSuchProcess:
            return
        except Exception as exc:
            # e.g. AccessDenied on a protected process — fall back to polling.
            logger.debug("proc.wait failed for PID %d (%s); polling instead", pid, exc)
            while psutil.pid_exists(pid):
                await asyncio.sleep(_POLL_INTERVAL)
            return


async def _pid_watch_loop(pid: int, watch_id: str, display_label: str, db_id: int | None) -> None:
    await _await_process_exit(pid)
    logger.info("Watchdog: PID %d is gone (%s)", pid, display_label)
    WATCHES.pop(watch_id, None)
    if db_id is not None:
        from bot import db
        try:
            await db.mark_watch_fired(db_id)
        except Exception as exc:
            logger.warning("Failed to mark watch fired in DB: %s", exc)
    await notifier.send(f"⚠️ Watchdog: process {display_label} (PID {pid}) has died.")


def _start_pid_watch(pid: int, watch_id: str, display_label: str, db_id: int | None) -> str:
    task = asyncio.create_task(_pid_watch_loop(pid, watch_id, display_label, db_id))
    WATCHES[watch_id] = WatchEntry(
        id=watch_id, watch_type="pid", target=str(pid),
        label=display_label, task=task, db_id=db_id,
    )
    return watch_id


async def watch_pid(pid: int, label: str = "") -> str:
    """Watch a process; alert when it dies.

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

    _start_pid_watch(pid, watch_id, display_label, db_id)
    logger.info("Watchdog registered: id=%s type=pid target=%d db_id=%s", watch_id, pid, db_id)
    return watch_id


# ── File watches ──────────────────────────────────────────────────

class _FileWatchHandler(FileSystemEventHandler):
    """Bridges filesystem events for one target file onto an asyncio queue.

    Runs on the observer thread, so it hands work back to the event loop via
    ``call_soon_threadsafe``. Emits "modified" or "deleted".
    """

    def __init__(self, target_path: str, loop: asyncio.AbstractEventLoop, queue: "asyncio.Queue[str]"):
        self._target = os.path.normcase(os.path.abspath(target_path))
        self._loop = loop
        self._queue = queue

    def _matches(self, path) -> bool:
        return bool(path) and os.path.normcase(os.path.abspath(path)) == self._target

    def _emit(self, kind: str) -> None:
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, kind)
        except RuntimeError:
            pass  # loop already closed during shutdown

    def on_modified(self, event) -> None:
        if not event.is_directory and self._matches(event.src_path):
            self._emit("modified")

    def on_created(self, event) -> None:
        # Editors that save atomically replace the file (delete + create);
        # treat a re-created target as a modification.
        if not event.is_directory and self._matches(event.src_path):
            self._emit("modified")

    def on_deleted(self, event) -> None:
        if not event.is_directory and self._matches(event.src_path):
            self._emit("deleted")

    def on_moved(self, event) -> None:
        # Moved onto the target = save; moved away from it = gone.
        if self._matches(getattr(event, "dest_path", "")):
            self._emit("modified")
        elif self._matches(event.src_path):
            self._emit("deleted")


def _teardown_observed_watch(entry: WatchEntry) -> None:
    if entry.observed_watch is not None and _observer is not None:
        try:
            _observer.unschedule(entry.observed_watch)
        except Exception as exc:
            logger.debug("Failed to unschedule observer watch: %s", exc)


async def _fire_file_deletion(watch_id: str, display_label: str, db_id: int | None) -> None:
    entry = WATCHES.pop(watch_id, None)
    if entry is not None:
        _teardown_observed_watch(entry)
    if db_id is not None:
        from bot import db
        try:
            await db.mark_watch_fired(db_id)
        except Exception as exc:
            logger.warning("Failed to mark watch fired in DB: %s", exc)
    await notifier.send(f"⚠️ Watchdog: file {display_label} was deleted.")


async def _file_watch_loop(watch_id: str, display_label: str, db_id: int | None, queue: "asyncio.Queue[str]") -> None:
    loop = asyncio.get_running_loop()
    while True:
        kind = await queue.get()
        if kind == "deleted":
            await _fire_file_deletion(watch_id, display_label, db_id)
            return

        # "modified": coalesce the burst of events a single save produces into
        # one alert, while still reacting to a deletion that lands mid-burst.
        deadline = loop.time() + _FILE_DEBOUNCE_SECONDS
        saw_deletion = False
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                nxt = await asyncio.wait_for(queue.get(), remaining)
            except asyncio.TimeoutError:
                break
            if nxt == "deleted":
                saw_deletion = True
                break

        if saw_deletion:
            await _fire_file_deletion(watch_id, display_label, db_id)
            return
        await notifier.send(f"📝 Watchdog: file {display_label} was modified.")


def _start_file_watch(path: str, watch_id: str, display_label: str, db_id: int | None) -> str:
    loop = asyncio.get_running_loop()
    queue: "asyncio.Queue[str]" = asyncio.Queue()
    parent = str(Path(path).resolve().parent)

    handler = _FileWatchHandler(path, loop, queue)
    observer = _get_observer()
    observed_watch = observer.schedule(handler, parent, recursive=False)

    task = asyncio.create_task(_file_watch_loop(watch_id, display_label, db_id, queue))
    WATCHES[watch_id] = WatchEntry(
        id=watch_id, watch_type="file", target=path,
        label=display_label, task=task, db_id=db_id, observed_watch=observed_watch,
    )
    return watch_id


async def watch_file(path: str, label: str = "") -> str:
    """Watch a file for changes/deletion via filesystem events.

    Args:
        path: Absolute path to the file to monitor.
        label: Optional human-readable description.

    Returns:
        The monitor ID, or an error string if the file does not exist.
    """
    global _next_id
    from bot import db

    watch_id = f"watch-{_next_id}"
    _next_id += 1
    display_label = label or path

    if not Path(path).exists():
        return f"File not found: {path}"

    try:
        db_id = await db.save_watch("file", path)
    except Exception as exc:
        logger.warning("Failed to persist watch to DB: %s", exc)
        db_id = None

    _start_file_watch(path, watch_id, display_label, db_id)
    logger.info("Watchdog registered: id=%s type=file target=%s db_id=%s", watch_id, path, db_id)
    return watch_id


# ── Restore-from-DB (after restart) ───────────────────────────────

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
    watch_id = f"watch-{_next_id}"
    _next_id += 1
    return _start_pid_watch(pid, watch_id, f"PID {pid}", db_id)


async def _restore_file_watch(path: str, db_id: int) -> str:
    """Re-create a file watch from a DB row without writing a new DB entry."""
    global _next_id
    watch_id = f"watch-{_next_id}"
    _next_id += 1
    return _start_file_watch(path, watch_id, path, db_id)


# ── Listing and cancellation ──────────────────────────────────────

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
    _teardown_observed_watch(entry)

    if entry.db_id is not None:
        asyncio.get_running_loop().create_task(_mark_fired_bg(entry.db_id))

    logger.info("Watchdog cancelled: id=%s type=%s target=%s", watch_id, entry.watch_type, entry.target)
    return f"Removed watchdog {watch_id} ({entry.watch_type}: {entry.target})."


async def _mark_fired_bg(db_id: int) -> None:
    """Background coroutine to mark a cancelled watch as fired in DB."""
    from bot import db
    try:
        await db.mark_watch_fired(db_id)
    except Exception as exc:
        logger.warning("Failed to mark cancelled watch fired in DB: %s", exc)
