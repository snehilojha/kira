"""Tests for Kira's explicit V1 capability policy."""

from __future__ import annotations

import unittest

from astra_node.core.tool import PermissionLevel
from astra_node.permissions.types import PermissionDecision

from bot import capability_policy


class CapabilityPolicyTests(unittest.TestCase):
    """Verify allow/confirm/deny policy behavior for the brain runtime."""

    def test_default_policy_contains_expected_categories(self) -> None:
        """The V1 policy table should expose the three explicit classes."""
        policy = capability_policy.get_default_policy()

        self.assertIn("allow_without_confirmation", policy)
        self.assertIn("require_confirmation", policy)
        self.assertIn("deny_or_manual_only", policy)
        self.assertIn("file_read", policy["allow_without_confirmation"])
        self.assertIn("bash", policy["require_confirmation"])
        self.assertIn("file_delete", policy["deny_or_manual_only"])

    def test_decide_action_allows_read_only_tools(self) -> None:
        """Read-only tools should be allowed immediately."""
        decision = capability_policy.decide_action("file_read")

        self.assertEqual(decision.category, "allow_without_confirmation")
        self.assertEqual(decision.permission, PermissionDecision.ALLOW)

    def test_decide_action_denies_confirmation_tools_until_intercept_exists(self) -> None:
        """Confirmation-required actions should fail closed in V1."""
        decision = capability_policy.decide_action("bash")

        self.assertEqual(decision.category, "require_confirmation")
        self.assertEqual(decision.permission, PermissionDecision.DENY)
        self.assertIn("live approval intercept", decision.reason)

    def test_decide_action_requests_confirmation_when_enabled(self) -> None:
        """Confirmation-required actions should resolve to ASK in approval-aware mode."""
        decision = capability_policy.decide_action(
            "bash",
            allow_confirmation_requests=True,
        )

        self.assertEqual(decision.category, "require_confirmation")
        self.assertEqual(decision.permission, PermissionDecision.ASK)
        self.assertIn("needs user confirmation", decision.reason)

    def test_registered_script_run_requires_confirmation(self) -> None:
        """Running a registered script should pause for approval in aware mode."""
        decision = capability_policy.decide_action(
            "registered_script_run",
            allow_confirmation_requests=True,
        )

        self.assertEqual(decision.category, "require_confirmation")
        self.assertEqual(decision.permission, PermissionDecision.ASK)

    def test_test_command_run_requires_confirmation(self) -> None:
        """Running an approved check should still pause for approval."""
        decision = capability_policy.decide_action(
            "test_command_run",
            allow_confirmation_requests=True,
        )

        self.assertEqual(decision.category, "require_confirmation")
        self.assertEqual(decision.permission, PermissionDecision.ASK)

    def test_decide_action_denies_unknown_tools_by_default(self) -> None:
        """Unknown actions should be denied for safety."""
        decision = capability_policy.decide_action("mystery_tool")

        self.assertEqual(decision.category, "deny_or_manual_only")
        self.assertEqual(decision.permission, PermissionDecision.DENY)

    def test_kira_permission_manager_uses_policy_table(self) -> None:
        """The permission manager should enforce Kira's explicit policy."""
        manager = capability_policy.KiraPermissionManager()

        self.assertEqual(manager.check("file_read"), PermissionDecision.ALLOW)
        self.assertEqual(manager.check("bash"), PermissionDecision.DENY)
        self.assertEqual(manager.check("unknown_tool"), PermissionDecision.DENY)

    def test_permission_manager_can_emit_ask_for_confirmation_tools(self) -> None:
        """Approval-aware mode should allow confirmation-gated tools to pause."""
        manager = capability_policy.KiraPermissionManager(
            allow_confirmation_requests=True
        )

        self.assertEqual(
            manager.check_level("bash", PermissionLevel.ASK_USER, {}),
            PermissionDecision.ASK,
        )

    def test_check_level_keeps_tool_level_hard_deny(self) -> None:
        """Tool-level DENY should still win even for allowlisted actions."""
        manager = capability_policy.KiraPermissionManager()

        decision = manager.check_level("file_read", PermissionLevel.DENY, {})

        self.assertEqual(decision, PermissionDecision.DENY)


if __name__ == "__main__":
    unittest.main()
