"""Operational task state storage for Kira complex-task runs.

This module keeps restart-safe task checkpoints and recent task history in
JSON files under ``data/task_state``. The stored data is operational state,
not semantic memory, so it stays separate from the main SQLite database.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_STATE_DIR = _PROJECT_ROOT / "data" / "task_state"


def get_state_dir() -> Path:
    """Return the task-state directory, creating it on demand."""
    state_dir = Path(os.environ.get("KIRA_TASK_STATE_DIR", str(_DEFAULT_STATE_DIR)))
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def save_task_state(task_id: str, payload: dict[str, Any]) -> Path:
    """Write the latest operational state for one task."""
    path = _task_state_path(task_id)
    serializable = dict(payload)
    serializable.setdefault("task_id", task_id)
    serializable["updated_at"] = _utc_now()
    path.write_text(json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_task_state(task_id: str) -> dict[str, Any] | None:
    """Load one task state file if it exists."""
    path = _task_state_path(task_id)
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def list_recent_task_states(limit: int = 10) -> list[dict[str, Any]]:
    """Return recent task states, newest first."""
    states: list[dict[str, Any]] = []
    for path in sorted(
        get_state_dir().glob("*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    ):
        state = load_task_state(path.stem)
        if state is not None:
            states.append(state)
        if len(states) >= limit:
            break
    return states


def list_task_states() -> list[dict[str, Any]]:
    """Return all task states, newest first."""
    return list_recent_task_states(limit=10_000)


def mark_interrupted_tasks(reason: str) -> list[dict[str, Any]]:
    """Mark restart-unsafe in-flight tasks as interrupted.

    Kira does not replay complex actions after restart. Any task that was
    running or waiting for approval is marked interrupted so it is visible
    without pretending it completed.
    """
    interrupted: list[dict[str, Any]] = []
    unsafe_statuses = {"running"}
    unsafe_stages = {
        "queued",
        "context_ready",
        "analysis_running",
        "tool_running",
        "tool_result",
        "tool_error",
        "approval_pending",
        "approval_resolved",
    }

    for state in list_task_states():
        status = str(state.get("status", ""))
        stage = str(state.get("stage", ""))
        if status not in unsafe_statuses and stage not in unsafe_stages:
            continue

        updated = dict(state)
        updated["status"] = "interrupted"
        updated["stage"] = "interrupted"
        updated["interrupted_reason"] = reason
        updated["last_message"] = (
            "Kira restarted while this task was unfinished. "
            "No actions were replayed automatically."
        )
        task_id = str(updated.get("task_id") or _extract_task_id(updated))
        save_task_state(task_id, updated)
        interrupted.append(updated)

    return interrupted


def delete_task_state(task_id: str) -> None:
    """Remove one task state file if it exists."""
    path = _task_state_path(task_id)
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _task_state_path(task_id: str) -> Path:
    """Return the JSON path for one task id."""
    return get_state_dir() / f"{task_id}.json"


def _extract_task_id(state: dict[str, Any]) -> str:
    task_request = state.get("task_request")
    if isinstance(task_request, dict):
        task_id = task_request.get("task_id")
        if task_id:
            return str(task_id)
    return "task-unknown"


def _utc_now() -> str:
    """Return an ISO timestamp for persisted operational state."""
    return datetime.now(timezone.utc).isoformat()
