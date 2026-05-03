"""Named-job monitor system for Kira V1.

Each monitor job polls a subject on a fixed interval and evaluates a
natural-language condition using the fast LLM model. When the condition
is met the job fires its success_action as a Telegram notification.

This is deliberately separate from ``bot.monitor`` (resource alerter) and
from the full Astra ``QueryEngine`` (complex tasks). The design goal is
minimal token cost: one short prompt → one yes/no answer per poll cycle.

Public API
----------
create_job(...)          create and persist a new job, returns MonitorJob
cancel_job(job_id)       mark a job cancelled and stop its loop
pause_job(job_id)        suspend polling without deleting the job
resume_job(job_id)       re-arm a paused job's polling loop
list_jobs()              return all known jobs (any status)
start()                  launch all active jobs on bot startup
reload_from_db()         restore active/paused jobs after a restart
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from bot import db
from bot import notifier
from bot import provider

logger = logging.getLogger(__name__)

# ── Dataclass ─────────────────────────────────────────────────────

@dataclass
class MonitorJob:
    """All state for one named monitor job."""

    job_id: str
    name: str
    subject: str
    condition: str
    poll_interval_seconds: float
    success_action: str
    failure_action: str | None = None
    expiry_at: str | None = None          # ISO datetime string, UTC
    requires_model: str = "fast"
    cooldown_seconds: float = 300.0
    status: str = "active"               # active | paused | cancelled | expired | fired
    created_at: str = field(default_factory=lambda: _utc_now())
    last_fired_at: str | None = None


# ── Manager ───────────────────────────────────────────────────────

class JobMonitorManager:
    """Manages the lifecycle of all named monitor jobs."""

    def __init__(self) -> None:
        self._jobs: dict[str, MonitorJob] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    # ── Public API ────────────────────────────────────────────────

    async def create_job(
        self,
        *,
        name: str,
        subject: str,
        condition: str,
        poll_interval_seconds: float = 60.0,
        success_action: str,
        failure_action: str | None = None,
        expiry_at: str | None = None,
        requires_model: str = "fast",
        cooldown_seconds: float = 300.0,
    ) -> MonitorJob:
        """Create, persist, and arm a new monitor job."""
        job = MonitorJob(
            job_id=f"job-{uuid.uuid4().hex[:12]}",
            name=name,
            subject=subject,
            condition=condition,
            poll_interval_seconds=max(poll_interval_seconds, 10.0),
            success_action=success_action,
            failure_action=failure_action,
            expiry_at=expiry_at,
            requires_model=requires_model,
            cooldown_seconds=cooldown_seconds,
            status="active",
        )
        await db.save_monitor_job(_job_to_dict(job))
        self._jobs[job.job_id] = job
        self._arm(job)
        logger.info("Monitor job created: %s (%s)", job.job_id, job.name)
        return job

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job and stop its loop. Returns False if not found."""
        job = self._jobs.get(job_id)
        if job is None:
            return False
        job.status = "cancelled"
        await db.update_monitor_job_status(job_id, "cancelled")
        self._cancel_task(job_id)
        logger.info("Monitor job cancelled: %s", job_id)
        return True

    async def pause_job(self, job_id: str) -> bool:
        """Suspend a job's polling loop without deleting it."""
        job = self._jobs.get(job_id)
        if job is None or job.status != "active":
            return False
        job.status = "paused"
        await db.update_monitor_job_status(job_id, "paused")
        self._cancel_task(job_id)
        logger.info("Monitor job paused: %s", job_id)
        return True

    async def resume_job(self, job_id: str) -> bool:
        """Re-arm a paused job."""
        job = self._jobs.get(job_id)
        if job is None or job.status != "paused":
            return False
        job.status = "active"
        await db.update_monitor_job_status(job_id, "active")
        self._arm(job)
        logger.info("Monitor job resumed: %s", job_id)
        return True

    def list_jobs(self) -> list[MonitorJob]:
        """Return all in-memory jobs (any status), newest created last."""
        return list(self._jobs.values())

    def get_job(self, job_id: str) -> MonitorJob | None:
        return self._jobs.get(job_id)

    async def start(self) -> None:
        """Arm all active jobs. Called once at bot startup after reload_from_db."""
        for job in self._jobs.values():
            if job.status == "active" and job.job_id not in self._tasks:
                self._arm(job)
        logger.info(
            "Job monitor started — %d active job(s)",
            sum(1 for j in self._jobs.values() if j.status == "active"),
        )

    async def reload_from_db(self) -> None:
        """Load active/paused jobs from the DB after a restart."""
        rows = await db.get_active_monitor_jobs()
        for row in rows:
            job = _job_from_dict(row)
            self._jobs[job.job_id] = job
        logger.info("Loaded %d monitor job(s) from DB", len(rows))

    # ── Internal helpers ──────────────────────────────────────────

    def _arm(self, job: MonitorJob) -> None:
        """Start the asyncio polling task for a job."""
        if job.job_id in self._tasks:
            return
        task = asyncio.create_task(
            self._run_job_loop(job),
            name=f"job-monitor-{job.job_id}",
        )
        self._tasks[job.job_id] = task
        task.add_done_callback(lambda t: self._tasks.pop(job.job_id, None))

    def _cancel_task(self, job_id: str) -> None:
        task = self._tasks.pop(job_id, None)
        if task and not task.done():
            task.cancel()

    async def _run_job_loop(self, job: MonitorJob) -> None:
        """Poll loop for a single job. Runs until cancelled, expired, or fired."""
        logger.info("Job loop started: %s (%s)", job.job_id, job.name)
        try:
            while True:
                await asyncio.sleep(job.poll_interval_seconds)

                if job.status != "active":
                    logger.info("Job %s is no longer active — stopping loop", job.job_id)
                    return

                if _is_expired(job):
                    job.status = "expired"
                    await db.update_monitor_job_status(job.job_id, "expired")
                    await notifier.send(
                        f"Monitor job *{job.name}* expired without triggering."
                    )
                    logger.info("Job %s expired", job.job_id)
                    return

                try:
                    triggered = await self._evaluate_condition(job)
                except Exception as exc:
                    logger.warning("Job %s condition eval failed: %s", job.job_id, exc)
                    continue

                if triggered:
                    if not _cooldown_ok(job):
                        logger.debug(
                            "Job %s triggered but cooldown not elapsed — skipping",
                            job.job_id,
                        )
                        continue

                    now = _utc_now()
                    job.last_fired_at = now
                    await db.update_monitor_job_last_fired(job.job_id, now)
                    await notifier.send(
                        f"Monitor job *{job.name}* triggered.\n{job.success_action}"
                    )
                    logger.info("Job %s fired — condition met", job.job_id)

        except asyncio.CancelledError:
            logger.info("Job loop cancelled: %s", job.job_id)
            raise

    async def _evaluate_condition(self, job: MonitorJob) -> bool:
        """Ask the fast LLM if the condition is met. Returns True/False."""
        subject_data = await _collect_subject_data(job.subject)
        prompt = _build_condition_prompt(job.condition, subject_data)

        cfg = provider.load_config()
        model = cfg.fast_model

        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0,
        )
        answer = (response.choices[0].message.content or "").strip().lower()
        logger.debug("Job %s condition answer: %r", job.job_id, answer)
        return answer.startswith("yes")


# ── Module-level singleton ─────────────────────────────────────────

_manager = JobMonitorManager()


# ── Public functions ───────────────────────────────────────────────

async def create_job(
    *,
    name: str,
    subject: str,
    condition: str,
    poll_interval_seconds: float = 60.0,
    success_action: str,
    failure_action: str | None = None,
    expiry_at: str | None = None,
    requires_model: str = "fast",
    cooldown_seconds: float = 300.0,
) -> MonitorJob:
    """Create and arm a new named monitor job."""
    return await _manager.create_job(
        name=name,
        subject=subject,
        condition=condition,
        poll_interval_seconds=poll_interval_seconds,
        success_action=success_action,
        failure_action=failure_action,
        expiry_at=expiry_at,
        requires_model=requires_model,
        cooldown_seconds=cooldown_seconds,
    )


async def cancel_job(job_id: str) -> bool:
    return await _manager.cancel_job(job_id)


async def pause_job(job_id: str) -> bool:
    return await _manager.pause_job(job_id)


async def resume_job(job_id: str) -> bool:
    return await _manager.resume_job(job_id)


def list_jobs() -> list[MonitorJob]:
    return _manager.list_jobs()


def get_job(job_id: str) -> MonitorJob | None:
    return _manager.get_job(job_id)


async def start() -> None:
    await _manager.start()


async def reload_from_db() -> None:
    await _manager.reload_from_db()


# ── Helpers ────────────────────────────────────────────────────────

async def _collect_subject_data(subject: str) -> str:
    """Gather current data for the subject to pass to the condition evaluator.

    Subjects can be:
    - A file path (reads last 100 lines)
    - A registered script name (reads from observer/process_registry)
    - A free-text description (falls back to observer machine context)
    """
    from pathlib import Path

    # File path subject
    path = Path(subject)
    if path.exists() and path.is_file():
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            if len(lines) > 100:
                lines = lines[-100:]
            return "\n".join(lines)
        except OSError:
            pass

    # Observer context as fallback
    try:
        from bot import observer
        ctx = observer.get_current_context()
        if ctx:
            return ctx
    except Exception:
        pass

    return f"Subject: {subject}\n(No live data available)"


def _build_condition_prompt(condition: str, subject_data: str) -> str:
    return (
        "You are evaluating a monitoring condition. "
        "Answer only 'yes' or 'no'.\n\n"
        f"Current data:\n{subject_data}\n\n"
        f"Condition: {condition}\n\n"
        "Is the condition currently met? Answer yes or no only."
    )


def _is_expired(job: MonitorJob) -> bool:
    if job.expiry_at is None:
        return False
    try:
        expiry = datetime.fromisoformat(job.expiry_at)
        now = datetime.now(timezone.utc)
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        return now >= expiry
    except ValueError:
        return False


def _cooldown_ok(job: MonitorJob) -> bool:
    if job.last_fired_at is None:
        return True
    try:
        last = datetime.fromisoformat(job.last_fired_at)
        now = datetime.now(timezone.utc)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        elapsed = (now - last).total_seconds()
        return elapsed >= job.cooldown_seconds
    except ValueError:
        return True


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_to_dict(job: MonitorJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "name": job.name,
        "subject": job.subject,
        "condition": job.condition,
        "poll_interval_seconds": job.poll_interval_seconds,
        "success_action": job.success_action,
        "failure_action": job.failure_action,
        "expiry_at": job.expiry_at,
        "requires_model": job.requires_model,
        "cooldown_seconds": job.cooldown_seconds,
        "status": job.status,
        "created_at": job.created_at,
        "last_fired_at": job.last_fired_at,
    }


def _job_from_dict(row: dict[str, Any]) -> MonitorJob:
    return MonitorJob(
        job_id=row["job_id"],
        name=row["name"],
        subject=row["subject"],
        condition=row["condition"],
        poll_interval_seconds=float(row["poll_interval_seconds"]),
        success_action=row["success_action"],
        failure_action=row.get("failure_action"),
        expiry_at=row.get("expiry_at"),
        requires_model=row.get("requires_model", "fast"),
        cooldown_seconds=float(row.get("cooldown_seconds", 300.0)),
        status=row.get("status", "active"),
        created_at=row.get("created_at", _utc_now()),
        last_fired_at=row.get("last_fired_at"),
    )
