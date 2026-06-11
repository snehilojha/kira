"""Tests for the V1.5 mode state machine."""

from __future__ import annotations

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("KIRA_DB_PATH", ":memory:")

from bot import db
import bot.mode as mode_module
from bot.mode import (
    get_mode,
    get_last_input_seconds,
    is_autonomous,
    set_mode,
)


def _reset_mode() -> None:
    """Reset module-level state between tests."""
    mode_module._current_mode = "idle"
    mode_module._autonomous_since = None


class TestModeDefaults(unittest.TestCase):
    """Initial state before any transitions."""

    def setUp(self) -> None:
        _reset_mode()

    def test_initial_mode_is_idle(self) -> None:
        self.assertEqual(get_mode(), "idle")

    def test_is_autonomous_false_initially(self) -> None:
        self.assertFalse(is_autonomous())

    def test_get_last_input_seconds_returns_float(self) -> None:
        result = get_last_input_seconds()
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)


class TestStateTransitions(unittest.TestCase):
    """Async set_mode transitions with DB logging."""

    def setUp(self) -> None:
        _reset_mode()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(db.init_db(":memory:"))

    def tearDown(self) -> None:
        self.loop.run_until_complete(db.close_db())
        self.loop.close()
        asyncio.set_event_loop(None)
        _reset_mode()

    def _run(self, coro):
        return self.loop.run_until_complete(coro)

    def test_set_mode_changes_state(self) -> None:
        self._run(set_mode("autonomous", "test"))
        self.assertEqual(get_mode(), "autonomous")

    def test_is_autonomous_true_when_autonomous(self) -> None:
        self._run(set_mode("autonomous", "test"))
        self.assertTrue(is_autonomous())

    def test_set_mode_logs_to_db(self) -> None:
        self._run(set_mode("active_session", "input detected"))
        transitions = self._run(db.get_recent_mode_transitions(1))
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0]["to_mode"], "active_session")
        self.assertEqual(transitions[0]["from_mode"], "idle")

    def test_set_mode_noop_when_same(self) -> None:
        self._run(set_mode("idle", "already idle"))
        transitions = self._run(db.get_recent_mode_transitions(10))
        self.assertEqual(len(transitions), 0)

    def test_multiple_transitions_logged(self) -> None:
        self._run(set_mode("active_session", "present"))
        self._run(set_mode("autonomous", "went away"))
        transitions = self._run(db.get_recent_mode_transitions(10))
        self.assertEqual(len(transitions), 2)
        modes = [t["to_mode"] for t in transitions]
        self.assertIn("active_session", modes)
        self.assertIn("autonomous", modes)


class TestPresenceDetection(unittest.TestCase):
    """_tick() maps idle time to correct modes."""

    def setUp(self) -> None:
        _reset_mode()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(db.init_db(":memory:"))

    def tearDown(self) -> None:
        self.loop.run_until_complete(db.close_db())
        self.loop.close()
        asyncio.set_event_loop(None)
        _reset_mode()

    def _run(self, coro):
        return self.loop.run_until_complete(coro)

    def test_below_threshold_maps_to_active_session(self) -> None:
        with patch.object(mode_module, "_get_last_input_seconds", return_value=60.0):
            self._run(mode_module._tick(idle_threshold=180.0))
        self.assertEqual(get_mode(), "active_session")

    def test_above_threshold_maps_to_autonomous(self) -> None:
        with patch.object(mode_module, "_get_last_input_seconds", return_value=300.0):
            self._run(mode_module._tick(idle_threshold=180.0))
        self.assertEqual(get_mode(), "autonomous")

    def test_get_last_input_seconds_fallback_on_error(self) -> None:
        with patch("ctypes.windll", side_effect=AttributeError("no windll")):
            # Should not raise — returns 0.0 (treated as present)
            result = mode_module._get_last_input_seconds()
        self.assertIsInstance(result, float)

    def test_awaiting_confirmation_not_overridden_by_tick(self) -> None:
        """Tick must not overwrite awaiting_confirmation with active_session."""
        self._run(set_mode("awaiting_confirmation", "manual"))
        with patch.object(mode_module, "_get_last_input_seconds", return_value=10.0):
            self._run(mode_module._tick(idle_threshold=180.0))
        self.assertEqual(get_mode(), "awaiting_confirmation")


class TestReturnSummary(unittest.TestCase):
    """Return summary fires on autonomous → active_session transition."""

    def setUp(self) -> None:
        _reset_mode()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(db.init_db(":memory:"))

    def tearDown(self) -> None:
        # Cancel any pending return-summary tasks.
        pending = [t for t in asyncio.all_tasks(self.loop)
                   if t.get_name() == "kira-return-summary"]
        for t in pending:
            t.cancel()
        if pending:
            self.loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
        self.loop.run_until_complete(db.close_db())
        self.loop.close()
        asyncio.set_event_loop(None)
        _reset_mode()

    def _run(self, coro):
        return self.loop.run_until_complete(coro)

    def test_return_summary_task_created_on_transition(self) -> None:
        """Transitioning autonomous → active_session schedules the summary task."""
        sent_messages: list[str] = []

        async def _fake_send(msg: str) -> None:
            sent_messages.append(msg)

        async def _run():
            await set_mode("autonomous", "away")
            with patch("bot.notifier.send", side_effect=_fake_send):
                await set_mode("active_session", "back")
                # Drain all pending tasks created during the transition,
                # including the scheduled return-summary task.
                await asyncio.sleep(0.05)
                summary_tasks = [
                    t for t in asyncio.all_tasks()
                    if t.get_name() == "kira-return-summary"
                ]
                if summary_tasks:
                    await asyncio.gather(*summary_tasks, return_exceptions=True)

        self.loop.run_until_complete(_run())

        self.assertTrue(
            any("autonomous" in m.lower() or "welcome back" in m.lower()
                for m in sent_messages),
            msg=f"Expected return summary in: {sent_messages}",
        )


if __name__ == "__main__":
    unittest.main()
