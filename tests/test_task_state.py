"""Tests for Kira's operational task-state store."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from bot import task_state


class TaskStateTests(unittest.TestCase):
    """Verify JSON-backed task-state persistence."""

    def test_save_and_load_task_state(self) -> None:
        """Saving a task state should make it reloadable by task id."""
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {"KIRA_TASK_STATE_DIR": tmpdir},
            clear=False,
        ):
            task_state.save_task_state(
                "task-123",
                {
                    "status": "running",
                    "stage": "context_ready",
                    "last_message": "Execution context is ready.",
                },
            )

            saved = task_state.load_task_state("task-123")

        self.assertIsNotNone(saved)
        self.assertEqual(saved["task_id"], "task-123")
        self.assertEqual(saved["status"], "running")
        self.assertEqual(saved["stage"], "context_ready")
        self.assertIn("updated_at", saved)

    def test_list_recent_task_states_returns_newest_first(self) -> None:
        """Recent states should be returned newest first."""
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {"KIRA_TASK_STATE_DIR": tmpdir},
            clear=False,
        ):
            task_state.save_task_state("task-1", {"status": "running", "stage": "queued"})
            task_state.save_task_state("task-2", {"status": "completed", "stage": "completed"})

            states = task_state.list_recent_task_states(limit=2)

        self.assertEqual(len(states), 2)
        self.assertEqual(states[0]["task_id"], "task-2")
        self.assertEqual(states[1]["task_id"], "task-1")

    def test_mark_interrupted_tasks_updates_running_tasks_only(self) -> None:
        """Startup recovery should mark unfinished tasks interrupted."""
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {"KIRA_TASK_STATE_DIR": tmpdir},
            clear=False,
        ):
            task_state.save_task_state(
                "task-running",
                {
                    "status": "running",
                    "stage": "approval_pending",
                    "last_message": "Waiting for approval.",
                    "pending_approval": {"request_id": "approval-1"},
                },
            )
            task_state.save_task_state(
                "task-done",
                {
                    "status": "completed",
                    "stage": "completed",
                    "last_message": "Done.",
                },
            )

            interrupted = task_state.mark_interrupted_tasks("restart")
            running = task_state.load_task_state("task-running")
            done = task_state.load_task_state("task-done")

        self.assertEqual(len(interrupted), 1)
        self.assertEqual(running["status"], "interrupted")
        self.assertEqual(running["stage"], "interrupted")
        self.assertEqual(running["interrupted_reason"], "restart")
        self.assertEqual(done["status"], "completed")


if __name__ == "__main__":
    unittest.main()
