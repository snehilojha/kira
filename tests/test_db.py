"""Tests for the async SQLite persistence layer (bot/db.py)."""

from __future__ import annotations

import asyncio
import unittest

from bot import db


class DBTests(unittest.TestCase):
    """Verify database initialisation, inserts, and queries."""

    def setUp(self) -> None:
        """Create a fresh in-memory database for each test."""
        asyncio.run(db.init_db(":memory:"))

    def tearDown(self) -> None:
        """Close the database after each test."""
        asyncio.run(db.close_db())

    # ── init / close ──────────────────────────────────────────────

    def test_init_db_creates_tables(self) -> None:
        """All three tables should exist after init_db."""
        async def _check() -> list[str]:
            conn = db._get_conn()
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            rows = await cursor.fetchall()
            return [r["name"] for r in rows]

        tables = asyncio.run(_check())
        self.assertIn("conversation_log", tables)
        self.assertIn("run_history", tables)
        self.assertIn("reminders", tables)

    def test_init_db_idempotent(self) -> None:
        """Calling init_db twice on the same DB should not raise."""
        asyncio.run(db.init_db(":memory:"))

    def test_get_conn_raises_before_init(self) -> None:
        """_get_conn should raise RuntimeError if DB is not initialised."""
        asyncio.run(db.close_db())
        with self.assertRaises(RuntimeError):
            db._get_conn()

    # ── conversation_log ──────────────────────────────────────────

    def test_log_and_get_conversations(self) -> None:
        """Logged conversations should be retrievable in chronological order."""
        async def _run() -> list[dict]:
            await db.log_conversation("user", "hello")
            await db.log_conversation("assistant", "hi there")
            await db.log_conversation("user", "run training")
            return await db.get_recent_conversations(10)

        rows = asyncio.run(_run())
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["role"], "user")
        self.assertEqual(rows[0]["content"], "hello")
        self.assertEqual(rows[2]["role"], "user")
        self.assertEqual(rows[2]["content"], "run training")

    def test_get_recent_conversations_limit(self) -> None:
        """Only the last N entries should be returned."""
        async def _run() -> list[dict]:
            for i in range(20):
                await db.log_conversation("user", f"msg {i}")
            return await db.get_recent_conversations(5)

        rows = asyncio.run(_run())
        self.assertEqual(len(rows), 5)
        # Oldest of the 5 should be msg 15 (0-indexed)
        self.assertEqual(rows[0]["content"], "msg 15")

    def test_conversation_truncates_long_content(self) -> None:
        """Content longer than 4000 chars should be truncated on insert."""
        async def _run() -> list[dict]:
            await db.log_conversation("user", "x" * 5000)
            return await db.get_recent_conversations(1)

        rows = asyncio.run(_run())
        self.assertEqual(len(rows[0]["content"]), 4000)

    def test_get_recent_conversations_empty(self) -> None:
        """An empty DB should return an empty list, not raise."""
        rows = asyncio.run(db.get_recent_conversations(10))
        self.assertEqual(rows, [])

    # ── run_history ───────────────────────────────────────────────

    def test_log_and_get_run_history(self) -> None:
        """Logged runs should be retrievable."""
        async def _run() -> list[dict]:
            await db.log_run(
                alias="crypto_train", started_at="2026-03-30T10:00:00",
                finished_at="2026-03-30T12:00:00", exit_code=0,
                runtime_seconds=7200.0, total_timesteps=500000,
                reward=-0.12, ep_len=847.0, loss=0.0023,
            )
            return await db.get_run_history()

        rows = asyncio.run(_run())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["alias"], "crypto_train")
        self.assertEqual(rows[0]["exit_code"], 0)
        self.assertAlmostEqual(rows[0]["reward"], -0.12)

    def test_get_run_history_filtered_by_alias(self) -> None:
        """Filtering by alias should exclude other scripts."""
        async def _run() -> list[dict]:
            await db.log_run(alias="a", started_at="t1", exit_code=0)
            await db.log_run(alias="b", started_at="t2", exit_code=0)
            await db.log_run(alias="a", started_at="t3", exit_code=0)
            return await db.get_run_history(alias="a")

        rows = asyncio.run(_run())
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["alias"] == "a" for r in rows))

    def test_get_previous_run_metrics_skips_crashes(self) -> None:
        """Previous run lookup should only return exit_code=0 runs."""
        async def _run() -> dict | None:
            await db.log_run(alias="x", started_at="t1", exit_code=0, reward=1.0)
            await db.log_run(alias="x", started_at="t2", exit_code=1, reward=2.0)
            return await db.get_previous_run_metrics("x")

        prev = asyncio.run(_run())
        self.assertIsNotNone(prev)
        self.assertAlmostEqual(prev["reward"], 1.0)

    def test_get_previous_run_metrics_none_when_empty(self) -> None:
        """Should return None when there are no previous runs."""
        result = asyncio.run(db.get_previous_run_metrics("nonexistent"))
        self.assertIsNone(result)

    def test_log_run_returns_rowid(self) -> None:
        """log_run should return the auto-incremented row ID."""
        async def _run() -> tuple[int, int]:
            id1 = await db.log_run(alias="a", started_at="t1")
            id2 = await db.log_run(alias="b", started_at="t2")
            return id1, id2

        id1, id2 = asyncio.run(_run())
        self.assertEqual(id1, 1)
        self.assertEqual(id2, 2)

    # ── reminders ─────────────────────────────────────────────────

    def test_save_and_get_pending_reminders(self) -> None:
        """Saved reminders should appear in pending list."""
        async def _run() -> list[dict]:
            await db.save_reminder("2099-01-01T00:00:00", "future reminder")
            return await db.get_pending_reminders()

        rows = asyncio.run(_run())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["message"], "future reminder")
        self.assertEqual(rows[0]["fired"], 0)

    def test_mark_reminder_fired(self) -> None:
        """Fired reminders should no longer appear in pending list."""
        async def _run() -> list[dict]:
            rid = await db.save_reminder("2099-01-01T00:00:00", "test")
            await db.mark_reminder_fired(rid)
            return await db.get_pending_reminders()

        rows = asyncio.run(_run())
        self.assertEqual(len(rows), 0)

    def test_save_reminder_returns_rowid(self) -> None:
        """save_reminder should return the auto-incremented row ID."""
        async def _run() -> int:
            return await db.save_reminder("2099-01-01T00:00:00", "test")

        rid = asyncio.run(_run())
        self.assertIsInstance(rid, int)
        self.assertGreater(rid, 0)


if __name__ == "__main__":
    unittest.main()
