"""Async SQLite persistence layer for Kira.

Stores conversation history, run metrics, and reminders so they survive
bot restarts.  Uses ``aiosqlite`` for non-blocking access from the
asyncio event loop.

Database location: ``data/kira.db`` relative to the project root.
The ``data/`` directory is created automatically on first call to
``init_db()``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

# Resolved once at module level; overridable via DB_PATH env var for tests.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DB_PATH = _PROJECT_ROOT / "data" / "kira.db"
_db_path: Path = Path(os.environ.get("KIRA_DB_PATH", str(_DEFAULT_DB_PATH)))

# Module-level connection — populated by init_db(), used by all helpers.
_conn: aiosqlite.Connection | None = None


# ── Initialisation ────────────────────────────────────────────────

async def init_db(db_path: Path | str | None = None) -> None:
    """Create the database and tables if they do not exist.

    Args:
        db_path: Override the default path.  Accepts ``":memory:"`` for
            in-memory databases (useful in tests).
    """
    global _conn, _db_path

    if db_path is not None:
        _db_path = Path(db_path) if db_path != ":memory:" else Path(":memory:")

    resolved = str(_db_path)

    # Ensure the parent directory exists (skip for in-memory DBs).
    if resolved != ":memory:":
        _db_path.parent.mkdir(parents=True, exist_ok=True)

    _conn = await aiosqlite.connect(resolved)
    _conn.row_factory = aiosqlite.Row

    await _conn.executescript(_SCHEMA)
    await _conn.commit()
    logger.info("Database initialised at %s", resolved)


async def close_db() -> None:
    """Close the database connection gracefully."""
    global _conn
    if _conn is not None:
        await _conn.close()
        _conn = None
        logger.info("Database connection closed")


def _get_conn() -> aiosqlite.Connection:
    """Return the active connection or raise if init_db() was not called."""
    if _conn is None:
        raise RuntimeError("Database not initialised — call init_db() first")
    return _conn


# ── Schema ────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversation_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp  TEXT    NOT NULL DEFAULT (datetime('now')),
    role       TEXT    NOT NULL,  -- 'user' or 'assistant'
    content    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS run_history (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    alias            TEXT    NOT NULL,
    started_at       TEXT    NOT NULL,
    finished_at      TEXT,
    exit_code        INTEGER,
    runtime_seconds  REAL,
    total_timesteps  INTEGER,
    reward           REAL,
    ep_len           REAL,
    loss             REAL
);

CREATE TABLE IF NOT EXISTS reminders (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    fire_at  TEXT    NOT NULL,
    message  TEXT    NOT NULL,
    fired    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT    NOT NULL,
    summary     TEXT    NOT NULL,
    raw_events  TEXT,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS schedules (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    alias       TEXT    NOT NULL,
    run_at      TEXT    NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    fired       INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS watches (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    type        TEXT    NOT NULL,
    target      TEXT    NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    fired       INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS observations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    observed_at     TEXT    NOT NULL DEFAULT (datetime('now')),
    active_projects TEXT,
    recent_files    TEXT,
    git_status      TEXT,
    running_procs   TEXT,
    screen_summary  TEXT
);

CREATE TABLE IF NOT EXISTS world_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    captured_at TEXT    NOT NULL DEFAULT (datetime('now')),
    btc_price   REAL,
    eth_price   REAL,
    fear_greed  INTEGER,
    top_news    TEXT
);
"""


# ── Conversation helpers ──────────────────────────────────────────

async def log_conversation(role: str, content: str) -> None:
    """Insert a conversation entry.

    Args:
        role: ``"user"`` or ``"assistant"``.
        content: The message text (truncated to 4000 chars for safety).
    """
    conn = _get_conn()
    await conn.execute(
        "INSERT INTO conversation_log (role, content) VALUES (?, ?)",
        (role, content[:4000]),
    )
    await conn.commit()


async def get_recent_conversations(n: int = 10) -> list[dict[str, Any]]:
    """Return the last *n* conversation entries, oldest first.

    Returns:
        List of dicts with keys: ``id``, ``timestamp``, ``role``, ``content``.
    """
    conn = _get_conn()
    cursor = await conn.execute(
        "SELECT id, timestamp, role, content "
        "FROM conversation_log ORDER BY id DESC LIMIT ?",
        (n,),
    )
    rows = await cursor.fetchall()
    # Reverse so oldest is first (chronological order for prompt injection).
    return [dict(r) for r in reversed(rows)]


# ── Run history helpers ───────────────────────────────────────────

async def log_run(
    alias: str,
    started_at: str,
    finished_at: str | None = None,
    exit_code: int | None = None,
    runtime_seconds: float | None = None,
    total_timesteps: int | None = None,
    reward: float | None = None,
    ep_len: float | None = None,
    loss: float | None = None,
) -> int:
    """Insert a completed (or crashed) run into history.

    Returns:
        The rowid of the new entry.
    """
    conn = _get_conn()
    cursor = await conn.execute(
        "INSERT INTO run_history "
        "(alias, started_at, finished_at, exit_code, runtime_seconds, "
        " total_timesteps, reward, ep_len, loss) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            alias, started_at, finished_at, exit_code, runtime_seconds,
            total_timesteps, reward, ep_len, loss,
        ),
    )
    await conn.commit()
    return cursor.lastrowid  # type: ignore[return-value]


async def get_run_history(
    alias: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Return recent runs, optionally filtered by alias.

    Returns:
        List of dicts, newest first.
    """
    conn = _get_conn()
    if alias:
        cursor = await conn.execute(
            "SELECT * FROM run_history WHERE alias = ? "
            "ORDER BY id DESC LIMIT ?",
            (alias, limit),
        )
    else:
        cursor = await conn.execute(
            "SELECT * FROM run_history ORDER BY id DESC LIMIT ?",
            (limit,),
        )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def get_previous_run_metrics(alias: str) -> dict[str, Any] | None:
    """Return metrics from the most recent *completed* run for comparison.

    Only considers runs with ``exit_code = 0`` so we compare against
    successful baselines, not crashes.

    Returns:
        A dict with metric fields, or ``None`` if no previous run exists.
    """
    conn = _get_conn()
    cursor = await conn.execute(
        "SELECT * FROM run_history "
        "WHERE alias = ? AND exit_code = 0 "
        "ORDER BY id DESC LIMIT 1",
        (alias,),
    )
    row = await cursor.fetchone()
    return dict(row) if row else None


# ── Reminder helpers ──────────────────────────────────────────────

async def save_reminder(fire_at: str, message: str) -> int:
    """Persist a reminder so it can be restored after a restart.

    Args:
        fire_at: ISO-format datetime string when the reminder should fire.
        message: The reminder text.

    Returns:
        The rowid of the new reminder.
    """
    conn = _get_conn()
    cursor = await conn.execute(
        "INSERT INTO reminders (fire_at, message) VALUES (?, ?)",
        (fire_at, message),
    )
    await conn.commit()
    return cursor.lastrowid  # type: ignore[return-value]


async def get_pending_reminders() -> list[dict[str, Any]]:
    """Return all reminders that have not yet fired.

    Returns:
        List of dicts with keys: ``id``, ``fire_at``, ``message``, ``fired``.
    """
    conn = _get_conn()
    cursor = await conn.execute(
        "SELECT id, fire_at, message, fired FROM reminders WHERE fired = 0",
    )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def mark_reminder_fired(reminder_id: int) -> None:
    """Mark a reminder as fired so it won't be reloaded on restart."""
    conn = _get_conn()
    await conn.execute(
        "UPDATE reminders SET fired = 1 WHERE id = ?",
        (reminder_id,),
    )
    await conn.commit()


# ── Session helpers ───────────────────────────────────────────────

async def save_session(date: str, summary: str, raw_events: str | None = None) -> int:
    """Persist a GPT-generated daily session summary.

    Args:
        date: ISO date string (YYYY-MM-DD).
        summary: GPT-generated paragraph summary.
        raw_events: Optional JSON string of raw event data.

    Returns:
        The rowid of the new session entry.
    """
    conn = _get_conn()
    cursor = await conn.execute(
        "INSERT INTO sessions (date, summary, raw_events) VALUES (?, ?, ?)",
        (date, summary, raw_events),
    )
    await conn.commit()
    return cursor.lastrowid  # type: ignore[return-value]


async def get_recent_sessions(n: int = 7) -> list[dict[str, Any]]:
    """Return the last *n* session summaries, newest first.

    Returns:
        List of dicts with keys: ``id``, ``date``, ``summary``, ``raw_events``, ``created_at``.
    """
    conn = _get_conn()
    cursor = await conn.execute(
        "SELECT id, date, summary, raw_events, created_at "
        "FROM sessions ORDER BY id DESC LIMIT ?",
        (n,),
    )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


# ── Schedule helpers ──────────────────────────────────────────────

async def save_schedule(alias: str, run_at: str) -> int:
    """Persist a scheduled run so it survives a bot restart.

    Args:
        alias: Script alias from scripts.toml.
        run_at: ISO-format datetime string.

    Returns:
        The rowid of the new schedule entry.
    """
    conn = _get_conn()
    cursor = await conn.execute(
        "INSERT INTO schedules (alias, run_at) VALUES (?, ?)",
        (alias, run_at),
    )
    await conn.commit()
    return cursor.lastrowid  # type: ignore[return-value]


async def mark_schedule_fired(schedule_id: int) -> None:
    """Mark a schedule as fired so it won't be reloaded on restart."""
    conn = _get_conn()
    await conn.execute(
        "UPDATE schedules SET fired = 1 WHERE id = ?",
        (schedule_id,),
    )
    await conn.commit()


async def get_pending_schedules() -> list[dict[str, Any]]:
    """Return all unfired schedules with a future run_at time.

    Returns:
        List of dicts with keys: ``id``, ``alias``, ``run_at``, ``created_at``.
    """
    conn = _get_conn()
    cursor = await conn.execute(
        "SELECT id, alias, run_at, created_at FROM schedules "
        "WHERE fired = 0 AND run_at > datetime('now')",
    )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


# ── Watch helpers ─────────────────────────────────────────────────

async def save_watch(watch_type: str, target: str) -> int:
    """Persist a watchdog entry so it survives a bot restart.

    Args:
        watch_type: ``"pid"`` or ``"file"``.
        target: PID as string, or file path.

    Returns:
        The rowid of the new watch entry.
    """
    conn = _get_conn()
    cursor = await conn.execute(
        "INSERT INTO watches (type, target) VALUES (?, ?)",
        (watch_type, target),
    )
    await conn.commit()
    return cursor.lastrowid  # type: ignore[return-value]


async def mark_watch_fired(watch_db_id: int) -> None:
    """Mark a watch as fired so it won't be reloaded on restart."""
    conn = _get_conn()
    await conn.execute(
        "UPDATE watches SET fired = 1 WHERE id = ?",
        (watch_db_id,),
    )
    await conn.commit()


async def get_pending_watches() -> list[dict[str, Any]]:
    """Return all unfired watch entries.

    Returns:
        List of dicts with keys: ``id``, ``type``, ``target``, ``created_at``.
    """
    conn = _get_conn()
    cursor = await conn.execute(
        "SELECT id, type, target, created_at FROM watches WHERE fired = 0",
    )
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


# ── Observation helpers ───────────────────────────────────────────

async def save_observation(snapshot: dict[str, Any]) -> int:
    """Persist an observer snapshot.

    Args:
        snapshot: Dict with optional keys: ``active_projects``, ``recent_files``,
            ``git_status``, ``running_procs``, ``screen_summary``.

    Returns:
        The rowid of the new observation entry.
    """
    import json as _json

    def _to_str(v: Any) -> str | None:
        if v is None:
            return None
        return v if isinstance(v, str) else _json.dumps(v)

    conn = _get_conn()
    cursor = await conn.execute(
        "INSERT INTO observations "
        "(active_projects, recent_files, git_status, running_procs, screen_summary) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            _to_str(snapshot.get("active_projects")),
            _to_str(snapshot.get("recent_files")),
            _to_str(snapshot.get("git_status")),
            _to_str(snapshot.get("running_procs")),
            _to_str(snapshot.get("screen_summary")),
        ),
    )
    await conn.commit()
    return cursor.lastrowid  # type: ignore[return-value]


# ── World snapshot helpers ────────────────────────────────────────

async def save_world_snapshot(snapshot: dict[str, Any]) -> int:
    """Persist a world data snapshot.

    Args:
        snapshot: Dict with optional keys: ``btc_price``, ``eth_price``,
            ``fear_greed``, ``top_news``.

    Returns:
        The rowid of the new snapshot entry.
    """
    import json as _json

    top_news = snapshot.get("top_news")
    if top_news is not None and not isinstance(top_news, str):
        top_news = _json.dumps(top_news)

    conn = _get_conn()
    cursor = await conn.execute(
        "INSERT INTO world_snapshots (btc_price, eth_price, fear_greed, top_news) "
        "VALUES (?, ?, ?, ?)",
        (
            snapshot.get("btc_price"),
            snapshot.get("eth_price"),
            snapshot.get("fear_greed"),
            top_news,
        ),
    )
    await conn.commit()
    return cursor.lastrowid  # type: ignore[return-value]


async def get_recent_world_snapshot() -> dict[str, Any] | None:
    """Return the most recently captured world snapshot, or None.

    Returns:
        Dict with keys: ``id``, ``captured_at``, ``btc_price``, ``eth_price``,
        ``fear_greed``, ``top_news``.
    """
    conn = _get_conn()
    cursor = await conn.execute(
        "SELECT * FROM world_snapshots ORDER BY id DESC LIMIT 1",
    )
    row = await cursor.fetchone()
    return dict(row) if row else None
