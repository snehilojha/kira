"""Tests for Kira's internal complex-task brain runtime."""

from __future__ import annotations

import asyncio
import os
import tempfile
import types
import unittest
from unittest.mock import AsyncMock, patch

from astra_node.core.events import TextDelta, ToolStart, ToolResult, TurnEnd
from astra_node.core.tool import PermissionLevel, ToolResult as ToolExecResult
from pydantic import BaseModel

from bot import brain
from bot import task_state


class BrainTests(unittest.TestCase):
    """Verify request/context construction and complex task execution."""

    def test_build_task_request_sets_expected_fields(self) -> None:
        """Task requests should carry the structured contract fields."""
        request = brain.build_task_request(
            user_input="review the reward function",
            source="telegram",
            route="complex",
            conversation_id="chat-1",
        )

        self.assertEqual(request.user_input, "review the reward function")
        self.assertEqual(request.source, "telegram")
        self.assertEqual(request.route, "complex")
        self.assertEqual(request.conversation_id, "chat-1")
        self.assertEqual(request.requires_confirmation_policy, "read_only")
        self.assertTrue(request.task_id.startswith("task-"))

    def test_build_system_prompt_mentions_read_only_mode(self) -> None:
        """The complex system prompt should advertise the read-only tool boundary."""
        execution_context = brain.ExecutionContext(
            memory_context=["Recent session history:\n  [2026-04-07] worked on kira"],
            state_snapshot={"cwd": "D:/VS_adv_python/kira", "task_source": "telegram"},
            available_tools=["file_read", "grep", "glob"],
            capability_policy={"allow_without_confirmation": ["file_read", "grep", "glob"]},
            config={"max_turns": 6},
        )

        prompt = brain._build_system_prompt(execution_context)

        self.assertIn("read-only complex analysis mode", prompt)
        self.assertIn("file_read, grep, glob", prompt)
        self.assertIn("Recent session history", prompt)


class BrainAsyncTests(unittest.IsolatedAsyncioTestCase):
    """Verify async brain behavior with patched runtime dependencies."""

    async def test_build_execution_context_collects_kira_state(self) -> None:
        """Execution context should gather memory, observer, and live state."""
        request = brain.build_task_request(
            user_input="review the code",
            source="telegram",
        )

        with patch.object(
            brain.kira_memory,
            "get_recent_sessions",
            new=AsyncMock(return_value="Recent session history:\n  [2026-04-07] worked on kira"),
        ), patch.object(
            brain.observer,
            "get_current_context",
            return_value="Machine awareness:\nactive repo is kira",
        ), patch.object(
            brain.process_registry,
            "list_processes",
            return_value=[{"pid": 11, "alias": "train"}],
        ), patch.object(
            brain.scheduler,
            "list_schedules",
            return_value=[{"id": "sched-1", "alias": "nightly"}],
        ), patch.object(
            brain.watchdog,
            "list_watches",
            return_value=[{"id": "watch-1", "type": "pid", "target": "11"}],
        ), patch.object(
            brain,
            "_load_project_context",
            return_value="Project context text",
        ):
            context = await brain.build_execution_context(request)

        self.assertIn("Recent session history", context.memory_context[0])
        self.assertIn("Project context", context.memory_context[1])
        self.assertEqual(context.available_tools, ["file_read", "grep", "glob", "web_search"])
        self.assertIn("bash", context.capability_policy["require_confirmation"])
        self.assertIn("file_delete", context.capability_policy["deny_or_manual_only"])
        self.assertEqual(context.state_snapshot["running_processes"][0]["pid"], 11)

    async def test_build_execution_context_can_include_confirmation_tools(self) -> None:
        """Approval-aware complex tasks should expose confirmation-gated tools."""
        request = brain.build_task_request(
            user_input="run a shell command if needed",
            source="telegram",
        )

        with patch.object(
            brain.kira_memory,
            "get_recent_sessions",
            new=AsyncMock(return_value=""),
        ), patch.object(
            brain.observer,
            "get_current_context",
            return_value="",
        ), patch.object(
            brain.process_registry,
            "list_processes",
            return_value=[],
        ), patch.object(
            brain.scheduler,
            "list_schedules",
            return_value=[],
        ), patch.object(
            brain.watchdog,
            "list_watches",
            return_value=[],
        ), patch.object(
            brain,
            "_load_project_context",
            return_value="",
        ):
            context = await brain.build_execution_context(
                request,
                include_confirmation_tools=True,
            )

        self.assertIn("bash", context.available_tools)
        self.assertIn("registered_script_run", context.available_tools)
        self.assertIn("test_command_run", context.available_tools)

    async def test_run_complex_task_collects_astra_events(self) -> None:
        """Complex task execution should aggregate text and tool events into a result."""

        class FakeEngine:
            async def run(self, user_message: str):
                yield ToolStart(tool_name="glob", tool_input={"pattern": "**/*.py"}, tool_use_id="1")
                yield ToolResult(tool_use_id="1", tool_name="glob", output="bot/main.py", is_error=False)
                yield TextDelta(text="I found the relevant files. ")
                yield TextDelta(text="The main entrypoint is bot/main.py.")
                yield TurnEnd(stop_reason="end_turn")

        request = brain.build_task_request(
            user_input="find the main entrypoint",
            source="telegram",
        )

        with patch.object(
            brain,
            "build_execution_context",
            new=AsyncMock(
                return_value=brain.ExecutionContext(
                    memory_context=[],
                    state_snapshot={"cwd": "D:/VS_adv_python/kira", "task_source": "telegram"},
                    available_tools=["file_read", "grep", "glob"],
                    capability_policy={"allow_without_confirmation": ["file_read", "grep", "glob"]},
                    config={"max_turns": 6},
                )
            ),
        ), patch.object(brain, "_build_query_engine", return_value=FakeEngine()):
            result = await brain.run_complex_task(request)

        self.assertEqual(result.status, "completed")
        self.assertIn("main entrypoint is bot/main.py", result.summary)
        self.assertEqual(result.state_writes[0]["type"], "complex_task_run")

    async def test_run_complex_task_stream_persists_task_state(self) -> None:
        """Streamed execution should emit progress and persist task snapshots."""

        class FakeEngine:
            async def run(self, user_message: str):
                yield ToolStart(tool_name="glob", tool_input={"pattern": "**/*.py"}, tool_use_id="1")
                yield TextDelta(text="I found the relevant files. ")
                yield TextDelta(text="The main entrypoint is bot/main.py.")
                yield ToolResult(tool_use_id="1", tool_name="glob", output="bot/main.py", is_error=False)
                yield TurnEnd(stop_reason="end_turn")

        request = brain.build_task_request(
            user_input="find the main entrypoint",
            source="telegram",
        )

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {"KIRA_TASK_STATE_DIR": tmpdir},
            clear=False,
        ), patch.object(
            brain,
            "build_execution_context",
            new=AsyncMock(
                return_value=brain.ExecutionContext(
                    memory_context=["Project context:\nKira task state test"],
                    state_snapshot={"cwd": "D:/VS_adv_python/kira", "task_source": "telegram"},
                    available_tools=["file_read", "grep", "glob"],
                    capability_policy={"allow_without_confirmation": ["file_read", "grep", "glob"]},
                    config={"max_turns": 6},
                )
            ),
        ), patch.object(brain, "_build_query_engine", return_value=FakeEngine()):
            events = []
            async for event in brain.run_complex_task_stream(request):
                events.append(event)

            persisted = task_state.load_task_state(request.task_id)

        self.assertEqual(events[0].event_type, "status")
        self.assertEqual(events[1].event_type, "status")
        self.assertEqual(events[2].event_type, "tool")
        self.assertEqual(events[-1].event_type, "result")
        self.assertIsNotNone(events[-1].result)
        self.assertIsNotNone(persisted)
        self.assertEqual(persisted["status"], "completed")
        self.assertEqual(persisted["stage"], "completed")
        self.assertEqual(persisted["task_request"]["task_id"], request.task_id)

    async def test_run_complex_task_stream_can_pause_for_approval(self) -> None:
        """Approval-aware runs should emit approval events and continue after approval."""

        class FakeProvider:
            def __init__(self) -> None:
                self._call_count = 0
                self.last_response = None

            async def complete(self, messages, tools, system=""):
                self._call_count += 1
                if self._call_count == 1:
                    self.last_response = types.SimpleNamespace(
                        content="",
                        tool_calls=[
                            types.SimpleNamespace(
                                id="tool-1",
                                name="bash",
                                input={"command": "echo hello", "timeout": 5},
                            )
                        ],
                        stop_reason="tool_use",
                    )
                else:
                    self.last_response = types.SimpleNamespace(
                        content="I ran the requested command successfully.",
                        tool_calls=[],
                        stop_reason="end_turn",
                    )
                    yield TextDelta(text="I ran the requested command successfully.")

        class FakePermissionManager:
            def check_level(self, tool_name, level, tool_input):
                return __import__("astra_node.permissions.types", fromlist=["PermissionDecision"]).PermissionDecision.ASK

        request = brain.build_task_request(
            user_input="run a shell command and tell me the result",
            source="telegram",
        )

        execution_context = brain.ExecutionContext(
            memory_context=[],
            state_snapshot={"cwd": "D:/VS_adv_python/kira", "task_source": "telegram"},
            available_tools=["file_read", "grep", "glob", "bash"],
            capability_policy=brain.capability_policy.get_default_policy(),
            config={"max_turns": 6, "cwd": "D:/VS_adv_python/kira", "approval_timeout_seconds": 5},
        )

        async def _approve(_request):
            return True

        class FakeBashInput(BaseModel):
            command: str
            timeout: int

        class FakeBashTool:
            name = "bash"
            permission_level = PermissionLevel.ASK_USER
            input_schema = FakeBashInput

            def execute(self, validated_input, ctx):
                return ToolExecResult.ok("hello\n")

        fake_registry = types.SimpleNamespace(
            get=lambda name: FakeBashTool(),
            to_api_format=lambda provider_name: [],
        )

        with patch.object(
            brain,
            "_build_runtime_components",
            return_value=(FakeProvider(), fake_registry, FakePermissionManager()),
        ):
            events = []
            async for event in brain._run_agent_loop_with_approvals(
                request,
                execution_context,
                approval_callback=_approve,
            ):
                events.append(event)

        approval_events = [e for e in events if isinstance(e, brain.BrainEvent)]
        self.assertEqual(approval_events[0].event_type, "approval_request")
        self.assertEqual(approval_events[1].event_type, "approval_result")
        self.assertTrue(approval_events[1].details["approved"])


if __name__ == "__main__":
    unittest.main()
