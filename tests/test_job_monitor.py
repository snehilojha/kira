"""Tests for the named-job monitor system."""

from __future__ import annotations

import asyncio
import os
import unittest
from datetime import datetime, timedelta, timezone

os.environ.setdefault("KIRA_DB_PATH", ":memory:")

from bot import db
from bot import job_monitor


class TestMonitorJobDataclass(unittest.TestCase):
    """MonitorJob fields and defaults."""

    def test_defaults_applied(self) -> None:
        job = job_monitor.MonitorJob(
            job_id="job-abc",
            name="test_job",
            subject="crypto_bot",
            condition="process exited",
            poll_interval_seconds=30.0,
            success_action="Done.",
        )
        self.assertEqual(job.status, "active")
        self.assertEqual(job.cooldown_seconds, 300.0)
        self.assertEqual(job.requires_model, "fast")
        self.assertIsNone(job.last_fired_at)
        self.assertIsNone(job.expiry_at)

    def test_created_at_is_set(self) -> None:
        job = job_monitor.MonitorJob(
            job_id="job-xyz",
            name="x",
            subject="s",
            condition="c",
            poll_interval_seconds=10.0,
            success_action="ok",
        )
        self.assertIsNotNone(job.created_at)


class TestHelpers(unittest.TestCase):
    """Unit tests for pure helper functions (no DB, no event loop needed)."""

    def test_is_expired_with_past_expiry(self) -> None:
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        job = job_monitor.MonitorJob(
            job_id="j", name="n", subject="s", condition="c",
            poll_interval_seconds=30.0, success_action="ok",
            expiry_at=past,
        )
        self.assertTrue(job_monitor._is_expired(job))

    def test_is_expired_with_future_expiry(self) -> None:
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        job = job_monitor.MonitorJob(
            job_id="j", name="n", subject="s", condition="c",
            poll_interval_seconds=30.0, success_action="ok",
            expiry_at=future,
        )
        self.assertFalse(job_monitor._is_expired(job))

    def test_is_not_expired_when_none(self) -> None:
        job = job_monitor.MonitorJob(
            job_id="j", name="n", subject="s", condition="c",
            poll_interval_seconds=30.0, success_action="ok",
        )
        self.assertFalse(job_monitor._is_expired(job))

    def test_cooldown_ok_when_never_fired(self) -> None:
        job = job_monitor.MonitorJob(
            job_id="j", name="n", subject="s", condition="c",
            poll_interval_seconds=30.0, success_action="ok",
            cooldown_seconds=300.0,
        )
        self.assertTrue(job_monitor._cooldown_ok(job))

    def test_cooldown_ok_after_elapsed(self) -> None:
        old = (datetime.now(timezone.utc) - timedelta(seconds=400)).isoformat()
        job = job_monitor.MonitorJob(
            job_id="j", name="n", subject="s", condition="c",
            poll_interval_seconds=30.0, success_action="ok",
            cooldown_seconds=300.0,
            last_fired_at=old,
        )
        self.assertTrue(job_monitor._cooldown_ok(job))

    def test_cooldown_not_ok_within_window(self) -> None:
        recent = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
        job = job_monitor.MonitorJob(
            job_id="j", name="n", subject="s", condition="c",
            poll_interval_seconds=30.0, success_action="ok",
            cooldown_seconds=300.0,
            last_fired_at=recent,
        )
        self.assertFalse(job_monitor._cooldown_ok(job))

    def test_build_condition_prompt_contains_condition_and_data(self) -> None:
        prompt = job_monitor._build_condition_prompt("loss < 0.2", "step=500 loss=0.15")
        self.assertIn("loss < 0.2", prompt)
        self.assertIn("step=500 loss=0.15", prompt)
        self.assertIn("yes or no", prompt)


class TestJobMonitorManager(unittest.TestCase):
    """Manager create / list / cancel / pause / resume."""

    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(db.init_db(":memory:"))
        self.manager = job_monitor.JobMonitorManager()

    def tearDown(self) -> None:
        # Cancel any armed polling tasks before closing the loop.
        for task in list(self.manager._tasks.values()):
            task.cancel()
        if self.manager._tasks:
            self.loop.run_until_complete(
                asyncio.gather(*self.manager._tasks.values(), return_exceptions=True)
            )
        self.loop.run_until_complete(db.close_db())
        self.loop.close()
        asyncio.set_event_loop(None)

    def _run(self, coro):
        return self.loop.run_until_complete(coro)

    def test_create_job_stores_and_returns(self) -> None:
        job = self._run(self.manager.create_job(
            name="test_create",
            subject="file.log",
            condition="loss < 0.2",
            poll_interval_seconds=60.0,
            success_action="Loss dropped.",
        ))
        self.assertEqual(job.name, "test_create")
        self.assertEqual(job.status, "active")
        self.assertIn(job.job_id, {j.job_id for j in self.manager.list_jobs()})

    def test_create_job_clamps_poll_interval(self) -> None:
        job = self._run(self.manager.create_job(
            name="fast_job",
            subject="s",
            condition="c",
            poll_interval_seconds=1.0,
            success_action="ok",
        ))
        self.assertGreaterEqual(job.poll_interval_seconds, 10.0)

    def test_cancel_job_returns_true(self) -> None:
        job = self._run(self.manager.create_job(
            name="to_cancel",
            subject="s",
            condition="c",
            poll_interval_seconds=30.0,
            success_action="ok",
        ))
        result = self._run(self.manager.cancel_job(job.job_id))
        self.assertTrue(result)
        self.assertEqual(self.manager.get_job(job.job_id).status, "cancelled")

    def test_cancel_unknown_job_returns_false(self) -> None:
        result = self._run(self.manager.cancel_job("job-doesnotexist"))
        self.assertFalse(result)

    def test_pause_and_resume(self) -> None:
        job = self._run(self.manager.create_job(
            name="pauseable",
            subject="s",
            condition="c",
            poll_interval_seconds=30.0,
            success_action="ok",
        ))
        paused = self._run(self.manager.pause_job(job.job_id))
        self.assertTrue(paused)
        self.assertEqual(self.manager.get_job(job.job_id).status, "paused")

        resumed = self._run(self.manager.resume_job(job.job_id))
        self.assertTrue(resumed)
        self.assertEqual(self.manager.get_job(job.job_id).status, "active")

    def test_pause_already_paused_returns_false(self) -> None:
        job = self._run(self.manager.create_job(
            name="double_pause",
            subject="s",
            condition="c",
            poll_interval_seconds=30.0,
            success_action="ok",
        ))
        self._run(self.manager.pause_job(job.job_id))
        result = self._run(self.manager.pause_job(job.job_id))
        self.assertFalse(result)

    def test_resume_active_job_returns_false(self) -> None:
        job = self._run(self.manager.create_job(
            name="active_resume",
            subject="s",
            condition="c",
            poll_interval_seconds=30.0,
            success_action="ok",
        ))
        result = self._run(self.manager.resume_job(job.job_id))
        self.assertFalse(result)

    def test_list_jobs_returns_all(self) -> None:
        self._run(self.manager.create_job(
            name="job_a", subject="s", condition="c",
            poll_interval_seconds=30.0, success_action="ok",
        ))
        self._run(self.manager.create_job(
            name="job_b", subject="s", condition="c",
            poll_interval_seconds=60.0, success_action="ok",
        ))
        jobs = self.manager.list_jobs()
        names = {j.name for j in jobs}
        self.assertIn("job_a", names)
        self.assertIn("job_b", names)


class TestDbRoundTrip(unittest.TestCase):
    """Monitor job DB persistence helpers."""

    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(db.init_db(":memory:"))

    def tearDown(self) -> None:
        self.loop.run_until_complete(db.close_db())
        self.loop.close()
        asyncio.set_event_loop(None)

    def _run(self, coro):
        return self.loop.run_until_complete(coro)

    def _make_job(self, job_id: str, name: str) -> job_monitor.MonitorJob:
        return job_monitor.MonitorJob(
            job_id=job_id,
            name=name,
            subject="s",
            condition="c",
            poll_interval_seconds=30.0,
            success_action="ok",
        )

    def test_save_and_retrieve_job(self) -> None:
        job = job_monitor.MonitorJob(
            job_id="job-db01",
            name="db_test",
            subject="train.log",
            condition="accuracy > 0.9",
            poll_interval_seconds=120.0,
            success_action="Training done.",
        )
        self._run(db.save_monitor_job(job_monitor._job_to_dict(job)))
        rows = self._run(db.get_active_monitor_jobs())
        ids = [r["job_id"] for r in rows]
        self.assertIn("job-db01", ids)

    def test_update_status_persists(self) -> None:
        job = self._make_job("job-db02", "status_test")
        self._run(db.save_monitor_job(job_monitor._job_to_dict(job)))
        self._run(db.update_monitor_job_status("job-db02", "paused"))
        rows = self._run(db.get_active_monitor_jobs())
        statuses = {r["job_id"]: r["status"] for r in rows}
        self.assertEqual(statuses.get("job-db02"), "paused")

    def test_cancelled_job_not_in_active(self) -> None:
        job = self._make_job("job-db03", "cancel_test")
        self._run(db.save_monitor_job(job_monitor._job_to_dict(job)))
        self._run(db.update_monitor_job_status("job-db03", "cancelled"))
        rows = self._run(db.get_active_monitor_jobs())
        ids = [r["job_id"] for r in rows]
        self.assertNotIn("job-db03", ids)

    def test_update_last_fired_persists(self) -> None:
        job = self._make_job("job-db04", "fired_test")
        self._run(db.save_monitor_job(job_monitor._job_to_dict(job)))
        fired_at = "2026-04-12T10:00:00+00:00"
        self._run(db.update_monitor_job_last_fired("job-db04", fired_at))
        rows = self._run(db.get_all_monitor_jobs())
        row = next((r for r in rows if r["job_id"] == "job-db04"), None)
        self.assertIsNotNone(row)
        self.assertEqual(row["last_fired_at"], fired_at)

    def test_reload_from_db_restores_jobs(self) -> None:
        job = self._make_job("job-db05", "reload_test")
        self._run(db.save_monitor_job(job_monitor._job_to_dict(job)))

        manager = job_monitor.JobMonitorManager()
        self._run(manager.reload_from_db())
        ids = {j.job_id for j in manager.list_jobs()}
        self.assertIn("job-db05", ids)


if __name__ == "__main__":
    unittest.main()
