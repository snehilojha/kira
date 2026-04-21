"""Explicit capability policy helpers for Kira V1.

This module holds the allow/confirm/deny policy table that governs what the
new brain runtime may do. In V1, actions that require confirmation are denied
by default because the Astra query loop does not yet pause for a live user
approval handshake.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from astra_node.permissions.manager import PermissionManager
from astra_node.permissions.types import PermissionDecision

_DEFAULT_POLICY = {
    "allow_without_confirmation": [
        "file_read",
        "grep",
        "glob",
        "process_status",
        "schedule_status",
        "watch_status",
        "summary_lookup",
        "registered_script_info",
    ],
    "require_confirmation": [
        "bash",
        "shell_command",
        "registered_script_run",
        "test_command_run",
        "file_write",
        "file_edit",
        "package_install",
        "git_commit",
        "git_push",
        "service_restart",
    ],
    "deny_or_manual_only": [
        "credential_read",
        "credential_write",
        "file_delete",
        "git_reset_hard",
        "git_rebase",
        "production_deploy",
    ],
}


@dataclass(frozen=True)
class CapabilityDecision:
    """Resolved decision for one tool or action name."""

    action: str
    category: str
    permission: PermissionDecision
    reason: str


def get_default_policy() -> dict[str, list[str]]:
    """Return a copy of Kira's V1 capability policy table."""
    return {category: list(actions) for category, actions in _DEFAULT_POLICY.items()}


def normalize_policy(policy: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
    """Normalize a capability policy into lowercase de-duplicated lists."""
    source = get_default_policy()
    if policy:
        for category, actions in policy.items():
            source[category] = list(actions)

    normalized: dict[str, list[str]] = {}
    for category in (
        "allow_without_confirmation",
        "require_confirmation",
        "deny_or_manual_only",
    ):
        items = source.get(category, [])
        unique = []
        seen: set[str] = set()
        for action in items:
            normalized_action = _normalize_action_name(action)
            if normalized_action and normalized_action not in seen:
                seen.add(normalized_action)
                unique.append(normalized_action)
        normalized[category] = unique
    return normalized


def decide_action(
    action_name: str,
    policy: dict[str, list[str]] | None = None,
    *,
    allow_confirmation_requests: bool = False,
) -> CapabilityDecision:
    """Resolve one action name against the V1 capability policy."""
    normalized = _normalize_action_name(action_name)
    policy_table = normalize_policy(policy)

    if normalized in policy_table["allow_without_confirmation"]:
        return CapabilityDecision(
            action=normalized,
            category="allow_without_confirmation",
            permission=PermissionDecision.ALLOW,
            reason="This action is explicitly read-only or status-only in V1.",
        )

    if normalized in policy_table["require_confirmation"]:
        permission = (
            PermissionDecision.ASK
            if allow_confirmation_requests
            else PermissionDecision.DENY
        )
        return CapabilityDecision(
            action=normalized,
            category="require_confirmation",
            permission=permission,
            reason=(
                "This action needs user confirmation before it can run."
                if allow_confirmation_requests
                else "This action needs user confirmation, and V1 does not yet have "
                "a live approval intercept inside the Astra loop."
            ),
        )

    if normalized in policy_table["deny_or_manual_only"]:
        return CapabilityDecision(
            action=normalized,
            category="deny_or_manual_only",
            permission=PermissionDecision.DENY,
            reason="This action is denied or manual-only in V1.",
        )

    return CapabilityDecision(
        action=normalized,
        category="deny_or_manual_only",
        permission=PermissionDecision.DENY,
        reason="Unclassified actions are denied by default in V1.",
    )


class KiraPermissionManager(PermissionManager):
    """Permission manager that enforces Kira's explicit V1 capability policy."""

    def __init__(
        self,
        capability_policy: dict[str, list[str]] | None = None,
        *,
        allow_confirmation_requests: bool = False,
    ) -> None:
        super().__init__()
        self._capability_policy = normalize_policy(capability_policy)
        self._allow_confirmation_requests = allow_confirmation_requests

    @property
    def capability_policy(self) -> dict[str, list[str]]:
        """Return the active normalized capability policy."""
        return {
            category: list(actions)
            for category, actions in self._capability_policy.items()
        }

    def check(self, tool_name: str, tool_input: dict[str, Any] | None = None) -> PermissionDecision:
        """Resolve a tool name directly against the Kira policy."""
        return self._resolve_permission(tool_name, tool_input)

    def check_level(self, tool_name: str, level, tool_input: dict[str, Any] | None = None) -> PermissionDecision:
        """Resolve a tool name while still respecting tool-level hard denies."""
        decision = decide_action(
            tool_name,
            self._capability_policy,
            allow_confirmation_requests=self._allow_confirmation_requests,
        )
        if decision.category != "allow_without_confirmation":
            return decision.permission

        # For explicitly allowed tools, still honor session overrides and the
        # tool's own declared permission level.
        return super().check_level(tool_name, level, tool_input)

    def _resolve_permission(
        self,
        tool_name: str,
        tool_input: dict[str, Any] | None = None,
    ) -> PermissionDecision:
        """Apply session overrides first, then the Kira capability policy."""
        override = super().check(tool_name, tool_input)
        if override in {PermissionDecision.ALLOW, PermissionDecision.DENY}:
            return override

        decision = decide_action(
            tool_name,
            self._capability_policy,
            allow_confirmation_requests=self._allow_confirmation_requests,
        )
        return decision.permission


def _normalize_action_name(action_name: str) -> str:
    """Normalize a tool or action name for policy matching."""
    return str(action_name or "").strip().lower()
