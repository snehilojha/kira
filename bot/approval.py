"""Approval request types for Kira's confirmation-gated actions."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ApprovalRequest:
    """A single action awaiting user confirmation."""

    task_id: str
    tool_name: str
    tool_input: dict[str, Any]
    reason: str
    timeout_seconds: int = 60
    request_id: str = field(default_factory=lambda: f"approval-{uuid.uuid4().hex[:12]}")

