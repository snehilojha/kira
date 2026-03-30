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
