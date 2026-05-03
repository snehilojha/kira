"""Kira's internal brain runtime for complex tasks.

Kira remains the product and user-facing assistant. This module is the
internal runtime that prepares Kira task/context objects and runs
read-only complex tasks through ``astra-node``.
"""

from __future__ import annotations

import asyncio
import json as _json
import functools
import uuid
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from astra_node.core.events import AgentError, TextDelta, ToolResult, ToolStart, TurnEnd
from astra_node.core.tool import BaseTool, PermissionLevel, ToolContext
from pydantic import BaseModel, Field

from bot import approval
from bot import capability_policy
from bot import db as kira_db
from bot import memory as kira_memory
from bot import mode as kira_mode
from bot import observer
from bot import process_registry
from bot import provider
from bot import scheduler
from bot import task_state
from bot import watchdog
from bot.utils import load_project_context as _load_project_context, tail_text, truncate_text

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_READ_ONLY_COMPLEX_TOOLS = ["file_read", "grep", "glob", "web_search"]
_CONFIRMATION_TOOLS = ["bash", "registered_script_run", "test_command_run"]
ApprovalCallback = Callable[[approval.ApprovalRequest], Awaitable[bool]]


class TaskStage:
    QUEUED            = "queued"
    CONTEXT_READY     = "context_ready"
    ANALYSIS_RUNNING  = "analysis_running"
    TOOL_RUNNING      = "tool_running"
    TOOL_RESULT       = "tool_result"
    TOOL_ERROR        = "tool_error"
    APPROVAL_PENDING  = "approval_pending"
    APPROVAL_RESOLVED = "approval_resolved"
    APPROVAL_ERROR    = "approval_error"
    ENGINE_UNAVAIL    = "engine_unavailable"
    COMPLETED         = "completed"


class TaskSource:
    LOCAL_VOICE = "local_voice"
    TELEGRAM    = "telegram"


@dataclass(frozen=True)
class TaskRequest:
    """Structured task object passed into Kira's brain runtime."""

    task_id: str
    user_input: str
    route: str
    source: str
    project_hint: str | None
    conversation_id: str
    requires_confirmation_policy: str


@dataclass(frozen=True)
class ExecutionContext:
    """Structured context injected into complex task execution."""

    memory_context: list[str]
    state_snapshot: dict[str, Any]
    available_tools: list[str]
    capability_policy: dict[str, list[str]]
    config: dict[str, Any]
    checkpoint: dict[str, Any] | None = None


@dataclass(frozen=True)
class BrainResult:
    """Final result envelope returned by the complex task runtime."""

    task_id: str
    status: str
    summary: str
    memory_writes: list[dict[str, Any]] = field(default_factory=list)
    state_writes: list[dict[str, Any]] = field(default_factory=list)
    retryable: bool = False


@dataclass(frozen=True)
class BrainEvent:
    """Streamed event emitted while a complex task is running."""

    task_id: str
    event_type: str
    message: str
    stage: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    result: BrainResult | None = None
    approval_request: approval.ApprovalRequest | None = None


def build_task_request(
    *,
    user_input: str,
    source: str,
    route: str = "complex",
    conversation_id: str = "telegram",
    project_hint: str | None = None,
) -> TaskRequest:
    """Create a structured request for a complex Kira task."""
    return TaskRequest(
        task_id=f"task-{uuid.uuid4().hex[:12]}",
        user_input=user_input,
        route=route,
        source=source,
        project_hint=project_hint,
        conversation_id=conversation_id,
        requires_confirmation_policy="read_only",
    )


async def build_execution_context(
    task_request: TaskRequest,
    checkpoint: dict[str, Any] | None = None,
    include_confirmation_tools: bool = False,
) -> ExecutionContext:
    """Collect the current Kira context for a complex task."""
    policy_table = capability_policy.get_default_policy()
    memory_context = []

    # Fetch independent async sources concurrently
    _gathered = await asyncio.gather(
        kira_memory.get_recent_sessions(3),
        kira_db.get_recent_world_snapshot(),
        return_exceptions=True,
    )
    session_history = _gathered[0] if not isinstance(_gathered[0], BaseException) else None
    world_snapshot  = _gathered[1] if not isinstance(_gathered[1], BaseException) else None

    if session_history:
        memory_context.append(session_history)

    project_context = _load_project_context()
    if project_context:
        memory_context.append(f"Project context:\n{project_context}")

    observer_context = observer.get_current_context()
    if observer_context:
        memory_context.append(f"Machine awareness:\n{observer_context}")

    active_project_ctx = observer.get_active_project_context()
    if active_project_ctx:
        memory_context.append(active_project_ctx)

    if world_snapshot:
        parts = [f"Time: {datetime.now().strftime('%A, %d %B %Y, %H:%M')}"]
        if world_snapshot.get("weather"):
            parts.append(f"Weather: {world_snapshot['weather']}")
        if world_snapshot.get("top_news"):
            parts.append(f"News:\n{world_snapshot['top_news']}")
        if world_snapshot.get("stocks"):
            stocks = world_snapshot["stocks"]
            if isinstance(stocks, str):
                stocks = _json.loads(stocks)
            indices = stocks.get("indices", {})
            if indices:
                idx_str = ", ".join(f"{k}: {v}" for k, v in indices.items())
                parts.append(f"Markets: {idx_str}")
            portfolio = stocks.get("portfolio", [])
            if portfolio:
                pf_lines = [
                    f"  {p['ticker']}: ₹{p['price']} (P&L: ₹{p['pnl']}, {p['pnl_pct']}%)"
                    for p in portfolio
                ]
                parts.append("Portfolio:\n" + "\n".join(pf_lines))
        memory_context.append("World context:\n" + "\n".join(parts))

    current_mode = kira_mode.get_mode()
    state_snapshot = {
        "running_processes": process_registry.list_processes(),
        "pending_schedules": scheduler.list_schedules(),
        "active_watches": watchdog.list_watches(),
        "cwd": str(_PROJECT_ROOT),
        "task_source": task_request.source,
        "current_mode": current_mode,
    }

    available_tools = list(_READ_ONLY_COMPLEX_TOOLS)
    if include_confirmation_tools or kira_mode.is_autonomous():
        available_tools.extend(_CONFIRMATION_TOOLS)

    return ExecutionContext(
        memory_context=memory_context,
        state_snapshot=state_snapshot,
        available_tools=available_tools,
        capability_policy=policy_table,
        config={
            "provider": "openai-compatible",
            "model_role": "smart",
            "max_turns": 18,
            "cwd": str(_PROJECT_ROOT),
            "approval_timeout_seconds": 60,
        },
        checkpoint=checkpoint,
    )


async def run_complex_task(
    task_request: TaskRequest,
    approval_callback: ApprovalCallback | None = None,
) -> BrainResult:
    """Run a read-only complex task through astra-node."""
    final_result: BrainResult | None = None

    async for event in run_complex_task_stream(
        task_request,
        approval_callback=approval_callback,
    ):
        if event.result is not None:
            final_result = event.result

    if final_result is None:
        final_result = BrainResult(
            task_id=task_request.task_id,
            status="failed",
            summary="The complex-task runtime ended without returning a final result.",
            retryable=True,
        )

    return final_result


async def run_complex_task_stream(
    task_request: TaskRequest,
    approval_callback: ApprovalCallback | None = None,
):
    """Run a complex task while yielding progress updates."""
    checkpoint = task_state.load_task_state(task_request.task_id)
    _persist_task_state(
        task_request,
        status="running",
        stage=TaskStage.QUEUED,
        last_message="Task accepted for complex analysis.",
        checkpoint=checkpoint,
    )
    yield BrainEvent(
        task_id=task_request.task_id,
        event_type="status",
        message="Complex analysis started. Preparing task context...",
        stage=TaskStage.QUEUED,
    )

    execution_context = await build_execution_context(
        task_request,
        checkpoint=checkpoint,
        include_confirmation_tools=approval_callback is not None,
    )
    _persist_task_state(
        task_request,
        status="running",
        stage=TaskStage.CONTEXT_READY,
        last_message="Execution context is ready.",
        execution_context=execution_context,
        checkpoint=checkpoint,
    )
    yield BrainEvent(
        task_id=task_request.task_id,
        event_type="status",
        message="Context ready. Starting the read-only analysis engine...",
        stage=TaskStage.CONTEXT_READY,
    )

    try:
        event_source = _build_event_source(
            task_request,
            execution_context,
            approval_callback=approval_callback,
        )
    except ModuleNotFoundError as exc:
        result = BrainResult(
            task_id=task_request.task_id,
            status="failed",
            summary=(
                "The complex-task runtime is not available yet because an Astra dependency "
                f"is missing: {exc.name}."
            ),
            memory_writes=[],
            state_writes=[],
            retryable=True,
        )
        _persist_final_result(
            task_request,
            result,
            stage=TaskStage.ENGINE_UNAVAIL,
            execution_context=execution_context,
        )
        yield BrainEvent(
            task_id=task_request.task_id,
            event_type="result",
            message=result.summary,
            stage=TaskStage.ENGINE_UNAVAIL,
            result=result,
        )
        return

    text_parts: list[str] = []
    tool_events: list[dict[str, Any]] = []
    _tool_call_count = 0
    _NARRATION_EVERY = 3
    _NARRATION_SUFFIXES = [
        "still on it.",
        "bear with me.",
        "pulling it together.",
        "nearly there.",
        "one more moment.",
    ]

    _persist_task_state(
        task_request,
        status="running",
        stage=TaskStage.ANALYSIS_RUNNING,
        last_message="Read-only analysis is running.",
        execution_context=execution_context,
        tool_events=tool_events,
    )

    async for event in event_source:
        if isinstance(event, BrainEvent):
            tool_events.append(
                {
                    "type": event.event_type,
                    "stage": event.stage,
                    "message": event.message,
                    "details": event.details,
                }
            )
            _persist_task_state(
                task_request,
                status="running",
                stage=event.stage or TaskStage.APPROVAL_PENDING,
                last_message=event.message,
                execution_context=execution_context,
                tool_events=tool_events,
                approval_request=(
                    event.approval_request
                    if event.event_type == "approval_request"
                    else None
                ),
            )
            yield event
            continue

        if isinstance(event, TextDelta):
            text_parts.append(event.text)
        elif isinstance(event, ToolStart):
            _tool_call_count += 1
            if _tool_call_count % _NARRATION_EVERY == 0:
                suffix = _NARRATION_SUFFIXES[
                    (_tool_call_count // _NARRATION_EVERY - 1) % len(_NARRATION_SUFFIXES)
                ]
                narration = _narration_for_tool(event.tool_name, event.tool_input, suffix)
                yield BrainEvent(
                    task_id=task_request.task_id,
                    event_type="narration",
                    message=narration,
                    stage=TaskStage.TOOL_RUNNING,
                )
            tool_msg = _describe_tool_start(event.tool_name, event.tool_input)
            tool_event = {
                "type": event.type,
                "tool_name": event.tool_name,
                "tool_input": event.tool_input,
            }
            tool_events.append(tool_event)
            _persist_task_state(
                task_request,
                status="running",
                stage=TaskStage.TOOL_RUNNING,
                last_message=tool_msg,
                execution_context=execution_context,
                tool_events=tool_events,
            )
            yield BrainEvent(
                task_id=task_request.task_id,
                event_type="tool",
                message=tool_msg,
                stage=TaskStage.TOOL_RUNNING,
                details=tool_event,
            )
        elif isinstance(event, ToolResult):
            tool_event = {
                "type": event.type,
                "tool_name": event.tool_name,
                "is_error": event.is_error,
                "output_tail": tail_text(event.output, 2000),
            }
            tool_events.append(tool_event)
            _persist_task_state(
                task_request,
                status="running",
                stage=TaskStage.TOOL_RESULT,
                last_message=_describe_tool_result(event.tool_name, event.is_error),
                execution_context=execution_context,
                tool_events=tool_events,
            )
            if event.is_error:
                yield BrainEvent(
                    task_id=task_request.task_id,
                    event_type="warning",
                    message=_describe_tool_result(event.tool_name, True),
                    stage=TaskStage.TOOL_RESULT,
                    details=tool_event,
                )
        elif isinstance(event, AgentError):
            tool_event = {
                "type": event.type,
                "tool_name": event.tool_name,
                "error": event.error,
            }
            tool_events.append(tool_event)
            error_message = _describe_agent_error(event.tool_name, event.error)
            _persist_task_state(
                task_request,
                status="running",
                stage=TaskStage.TOOL_ERROR,
                last_message=error_message,
                execution_context=execution_context,
                tool_events=tool_events,
            )
            yield BrainEvent(
                task_id=task_request.task_id,
                event_type="warning",
                message=error_message,
                stage=TaskStage.TOOL_ERROR,
                details=tool_event,
            )
        elif isinstance(event, TurnEnd):
            tool_events.append({"type": event.type, "stop_reason": event.stop_reason})

    summary = "".join(text_parts).strip()
    if not summary:
        summary = (
            "I completed the complex read-only analysis path, but the model did not "
            "return a final text answer."
        )

    result = BrainResult(
        task_id=task_request.task_id,
        status="completed",
        summary=summary,
        memory_writes=[],
        state_writes=[
            {
                "type": "complex_task_run",
                "task_id": task_request.task_id,
                "route": task_request.route,
                "source": task_request.source,
                "tool_events": tool_events,
            }
        ],
        retryable=False,
    )
    _persist_final_result(
        task_request,
        result,
        stage=TaskStage.COMPLETED,
        execution_context=execution_context,
        tool_events=tool_events,
    )
    yield BrainEvent(
        task_id=task_request.task_id,
        event_type="result",
        message=result.summary,
        stage=TaskStage.COMPLETED,
        details={"tool_event_count": len(tool_events)},
        result=result,
    )


def _build_query_engine(execution_context: ExecutionContext) -> Any:
    """Create a read-only astra-node query engine for Kira complex tasks."""
    from astra_node.core.query_engine import QueryEngine
    llm_provider, registry, permission_manager = _build_runtime_components(
        execution_context,
        allow_confirmation_requests=False,
    )

    system_prompt = _build_system_prompt(execution_context)

    return QueryEngine(
        provider=llm_provider,
        registry=registry,
        permission_manager=permission_manager,
        system_prompt=system_prompt,
        max_turns=int(execution_context.config.get("max_turns", 6)),
    )


def _build_event_source(
    task_request: TaskRequest,
    execution_context: ExecutionContext,
    *,
    approval_callback: ApprovalCallback | None,
):
    """Return the event stream for one complex task execution."""
    if approval_callback is None:
        engine = _build_query_engine(execution_context)
        return engine.run(task_request.user_input)
    return _run_agent_loop_with_approvals(
        task_request,
        execution_context,
        approval_callback=approval_callback,
    )


class _WebSearchInput(BaseModel):
    query: str = Field(..., description="Search query string.")


class _WebSearchTool(BaseTool):
    """Search the web via DuckDuckGo and return top results as plain text."""

    name = "web_search"
    description = (
        "Search the web for current information. Use for news, facts, weather, "
        "prices, or anything that requires up-to-date knowledge."
    )
    input_schema = _WebSearchInput
    permission_level = PermissionLevel.ALWAYS_ALLOW

    def execute(self, input: _WebSearchInput, ctx: ToolContext) -> ToolResult:
        try:
            from ddgs import DDGS
        except ImportError:
            try:
                from duckduckgo_search import DDGS
            except ImportError:
                return ToolResult.err("ddgs is not installed. Run: pip install ddgs")

        try:
            with DDGS() as ddgs:
                hits = list(ddgs.text(input.query, max_results=5))
        except Exception as exc:
            return ToolResult.err(f"Web search failed: {exc}")

        if not hits:
            return ToolResult.ok("No results found.")

        lines = []
        for i, hit in enumerate(hits, 1):
            title = (hit.get("title") or "").encode("utf-8", "replace").decode("utf-8").strip()
            body = (hit.get("body") or "").encode("utf-8", "replace").decode("utf-8").strip()
            href = (hit.get("href") or "").strip()
            lines.append(f"{i}. {title}\n   {body}\n   {href}")

        return ToolResult.ok("\n\n".join(lines))


@functools.lru_cache(maxsize=1)
def _readable_roots() -> tuple[Path, ...]:
    """Return paths Kira is allowed to read, from KIRA_READABLE_ROOTS env var."""
    raw = os.environ.get("KIRA_READABLE_ROOTS", "").strip()
    if not raw:
        return ()
    return tuple(Path(p.strip()).resolve() for p in raw.split(",") if p.strip())


class _KiraFileReadTool(BaseTool):
    """FileReadTool extended to allow reads from KIRA_READABLE_ROOTS in addition to cwd."""

    name = "file_read"
    description = (
        "Read the contents of a file. "
        "Optionally specify offset (starting line) and limit (max lines) "
        "for reading large files in chunks."
    )

    from astra_node.tools.file_read import FileReadInput
    input_schema = FileReadInput
    permission_level = PermissionLevel.ALWAYS_ALLOW

    def execute(self, input, ctx: ToolContext) -> ToolResult:
        from astra_node.tools.file_read import FileReadTool as _Base
        import fnmatch

        path = Path(input.path)
        if not path.is_absolute():
            path = ctx.cwd / path
        path = path.resolve()

        cwd_resolved = ctx.cwd.resolve()
        extra_roots = _readable_roots()
        allowed = (cwd_resolved,) + extra_roots

        if not any(path.is_relative_to(root) for root in allowed):
            readable = ", ".join(str(r) for r in extra_roots) or "none configured"
            return ToolResult.err(
                f"Access denied: path is outside the working directory and readable roots "
                f"(readable roots: {readable})"
            )

        # BaseFileTool re-checks is_relative_to(cwd) internally, so supply the
        # matched root as cwd rather than the global project root.
        matching_root = next(r for r in allowed if path.is_relative_to(r))
        return _Base().execute(input, ToolContext(cwd=matching_root))


def _build_runtime_components(
    execution_context: ExecutionContext,
    *,
    allow_confirmation_requests: bool,
):
    """Create the provider, registry, and permission manager for one task."""
    from astra_node.core.registry import ToolRegistry
    from astra_node.providers.openai import OpenAIProvider
    from astra_node.tools.bash import BashTool
    from astra_node.tools.glob_tool import GlobTool
    from astra_node.tools.grep import GrepTool
    from bot.check_tool import TestCommandRunTool
    from bot.registered_script_tool import RegisteredScriptRunTool

    config = provider.load_config()
    provider_name = "openai"
    if config.base_url and "openrouter" in config.base_url.lower():
        provider_name = "openrouter"

    llm_provider = OpenAIProvider(
        api_key=config.api_key,
        model=config.smart_model,
        base_url=config.base_url,
        provider_name=provider_name,
    )

    registry = ToolRegistry()
    for tool in (
        _KiraFileReadTool(),
        GrepTool(),
        GlobTool(),
        BashTool(),
        RegisteredScriptRunTool(),
        TestCommandRunTool(),
        _WebSearchTool(),
    ):
        if tool.name in execution_context.available_tools:
            registry.register(tool)

    permission_manager = capability_policy.KiraPermissionManager(
        execution_context.capability_policy,
        allow_confirmation_requests=allow_confirmation_requests,
    )
    return llm_provider, registry, permission_manager


async def _run_agent_loop_with_approvals(
    task_request: TaskRequest,
    execution_context: ExecutionContext,
    *,
    approval_callback: ApprovalCallback,
):
    """Run a Kira-owned agent loop that can pause for live approvals."""
    from pydantic import ValidationError

    from astra_node.core.events import UsageUpdate
    from astra_node.core.history import MessageHistory
    from astra_node.core.prompt_guard import (
        check_injection,
        scan_tool_result,
        wrap_tool_result,
        wrap_user_message,
    )
    from astra_node.core.tool import ToolContext
    from astra_node.permissions.types import PermissionDecision
    from astra_node.utils.errors import PermissionDeniedError

    llm_provider, registry, permission_manager = _build_runtime_components(
        execution_context,
        allow_confirmation_requests=True,
    )
    history = MessageHistory()
    check_injection(task_request.user_input)
    history.add_user(wrap_user_message(task_request.user_input))

    def _tool_error_events(tc_id, tc_name, error_msg, *, recoverable=False):
        """Yield the standard ToolResult + AgentError pair for a failed tool call."""
        history.add_tool_result(tc_id, error_msg, is_error=True)
        return [
            ToolResult(tool_use_id=tc_id, tool_name=tc_name, output=error_msg, is_error=True),
            AgentError(error=error_msg, tool_name=tc_name, tool_use_id=tc_id, recoverable=recoverable),
        ]

    turns_used = 0
    text_emitted = False
    system_prompt = _build_system_prompt(execution_context)
    provider_name = _detect_provider_name(llm_provider)
    tool_schemas = registry.to_api_format(provider_name)

    while turns_used < int(execution_context.config.get("max_turns", 6)):
        turns_used += 1

        messages = history.to_api_format(provider_name)

        async for event in llm_provider.complete(
            messages=messages,
            tools=tool_schemas,
            system=system_prompt,
        ):
            if isinstance(event, (TextDelta, UsageUpdate)):
                if isinstance(event, TextDelta):
                    text_emitted = True
                yield event

        response = getattr(llm_provider, "last_response", None)
        if response is None:
            yield TurnEnd(stop_reason="end_turn")
            return

        assistant_content: list[dict[str, Any]] = []
        if response.content:
            assistant_content.append({"type": "text", "text": response.content})
        for tc in response.tool_calls:
            assistant_content.append(
                {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.input,
                }
            )
        if assistant_content:
            history.add_assistant(assistant_content)

        if response.stop_reason == "tool_use":
            if not response.tool_calls:
                yield TurnEnd(stop_reason="end_turn")
                return

            for tc in response.tool_calls:
                yield ToolStart(
                    tool_name=tc.name,
                    tool_input=tc.input,
                    tool_use_id=tc.id,
                )

                try:
                    tool = registry.get(tc.name)
                except KeyError:
                    for ev in _tool_error_events(tc.id, tc.name, f"Tool '{tc.name}' is not registered."):
                        yield ev
                    continue

                decision = permission_manager.check_level(
                    tc.name,
                    tool.permission_level,
                    tc.input,
                )
                if decision == PermissionDecision.ASK:
                    approval_request = approval.ApprovalRequest(
                        task_id=task_request.task_id,
                        tool_name=tc.name,
                        tool_input=tc.input,
                        reason=capability_policy.decide_action(
                            tc.name,
                            execution_context.capability_policy,
                            allow_confirmation_requests=True,
                        ).reason,
                        timeout_seconds=int(
                            execution_context.config.get("approval_timeout_seconds", 60)
                        ),
                    )
                    yield BrainEvent(
                        task_id=task_request.task_id,
                        event_type="approval_request",
                        message=_describe_approval_request(approval_request),
                        stage=TaskStage.APPROVAL_PENDING,
                        details={
                            "tool_name": approval_request.tool_name,
                            "tool_input": approval_request.tool_input,
                            "request_id": approval_request.request_id,
                        },
                        approval_request=approval_request,
                    )
                    try:
                        approved = await approval_callback(approval_request)
                    except Exception as exc:
                        approved = False
                        yield BrainEvent(
                            task_id=task_request.task_id,
                            event_type="warning",
                            message=f"Approval handling failed: {exc}",
                            stage=TaskStage.APPROVAL_ERROR,
                            details={"request_id": approval_request.request_id},
                            approval_request=approval_request,
                        )
                    else:
                        resolution_message = (
                            f"Approval granted for `{tc.name}`."
                            if approved
                            else f"Approval denied for `{tc.name}`."
                        )
                        yield BrainEvent(
                            task_id=task_request.task_id,
                            event_type="approval_result",
                            message=resolution_message,
                            stage=TaskStage.APPROVAL_RESOLVED,
                            details={
                                "request_id": approval_request.request_id,
                                "approved": approved,
                            },
                            approval_request=approval_request,
                        )
                    if approved:
                        decision = PermissionDecision.ALLOW
                    else:
                        decision = PermissionDecision.DENY

                if decision == PermissionDecision.DENY:
                    for ev in _tool_error_events(tc.id, tc.name, str(PermissionDeniedError(tc.name, tc.input))):
                        yield ev
                    continue

                try:
                    validated_input = tool.input_schema(**tc.input)
                except ValidationError as exc:
                    for ev in _tool_error_events(tc.id, tc.name, f"Invalid input for tool '{tc.name}': {exc}"):
                        yield ev
                    continue

                ctx = ToolContext(
                    cwd=Path(str(execution_context.config.get("cwd", _PROJECT_ROOT))),
                )
                try:
                    result = tool.execute(validated_input, ctx)
                except Exception as exc:
                    for ev in _tool_error_events(tc.id, tc.name, f"Unexpected error in tool '{tc.name}': {exc}", recoverable=True):
                        yield ev
                    continue
                tool_output = result.output
                if not result.is_error and tool_output:
                    injection_warning = scan_tool_result(tool_output, tc.name)
                    if injection_warning:
                        tool_output = injection_warning + tool_output
                    tool_output = wrap_tool_result(tool_output, tc.name)

                history.add_tool_result(tc.id, tool_output, is_error=result.is_error)
                yield ToolResult(
                    tool_use_id=tc.id,
                    tool_name=tc.name,
                    output=result.output,
                    is_error=result.is_error,
                )
                if result.is_error:
                    yield AgentError(
                        error=result.output,
                        tool_name=tc.name,
                        tool_use_id=tc.id,
                        recoverable=True,
                    )
            continue

        yield TurnEnd(stop_reason=response.stop_reason or "end_turn")
        return

    if not text_emitted:
        yield TextDelta(
            text="I stopped because the complex task hit the max-turn limit before finishing."
        )
    yield TurnEnd(stop_reason="max_turns")


def _narration_for_tool(tool_name: str, tool_input: dict[str, Any], suffix: str) -> str:
    """Build a natural-language narration line describing what tool is running."""
    _tool_phrases: dict[str, str] = {
        "file_read":           "reading a file",
        "grep":                "searching through code",
        "glob":                "scanning the file tree",
        "web_search":          "searching the web",
        "bash":                "running a command",
        "registered_script_run": "running a script",
        "test_command_run":    "running tests",
    }
    action = _tool_phrases.get(tool_name)
    if action is None:
        action = f"using {tool_name.replace('_', ' ')}"

    # Try to add specifics from tool input
    detail = ""
    if tool_name == "file_read":
        path = str(tool_input.get("path", tool_input.get("file_path", ""))).strip()
        if path:
            detail = f" — {path.split('/')[-1].split(chr(92))[-1]}"
    elif tool_name in ("grep", "glob"):
        pattern = str(tool_input.get("pattern", tool_input.get("query", ""))).strip()
        if pattern:
            detail = f" for {pattern[:40]!r}"
    elif tool_name == "web_search":
        query = str(tool_input.get("query", "")).strip()
        if query:
            detail = f" for {query[:40]!r}"
    elif tool_name == "bash":
        cmd = str(tool_input.get("command", "")).strip()
        if cmd:
            detail = f" — {cmd[:40]}"

    return f"Still working — {action}{detail}, {suffix}"


def _detect_provider_name(llm_provider: Any) -> str:
    """Return the provider wire format name for one provider instance."""
    class_name = type(llm_provider).__name__.lower()
    if "anthropic" in class_name:
        return "anthropic"
    return "openai"


def _build_system_prompt(execution_context: ExecutionContext) -> str:
    """Build the system prompt for Kira's complex task runtime."""
    source = execution_context.state_snapshot.get("task_source", "")
    if source == TaskSource.LOCAL_VOICE:
        sections = _build_voice_system_prompt(execution_context)
    else:
        sections = _build_telegram_system_prompt(execution_context)

    if execution_context.memory_context:
        sections.append("\n\n".join(execution_context.memory_context))

    return "\n\n".join(section for section in sections if section).strip()


def _build_voice_system_prompt(execution_context: ExecutionContext) -> list[str]:
    from bot import identity as kira_identity
    identity_block = kira_identity.get_identity_prompt()
    user_name = kira_identity.get_user_name()
    return [
        (
            f"{identity_block}\n\n"
            "Voice conversation rules — your words go straight to TTS, so write as you'd speak:\n"
            f"- Use '{user_name}' occasionally — only when it feels natural, not every reply.\n"
            "- Contractions always. Never be stiff or formal.\n"
            "- Never open with 'Certainly', 'Sure', 'Of course', 'Absolutely', or 'I'.\n"
            "- Match length to the question. One sentence for simple things. Two to three sentences max for complex ones.\n"
            "- Zero markdown. Zero bullet points. Zero headers.\n"
            "- Have opinions. Be confident. Drop the hedges.\n"
            "- Never mention being an AI or having limitations unless directly asked.\n"
            "- If screen context is provided, use it to give more relevant answers.\n\n"
            "Emotional intelligence — this is a real conversation:\n"
            "- Read the tone. If the user sounds frustrated or tired, acknowledge it before answering.\n"
            "- If they're joking, match that energy. If they're stressed, be steadier and warmer.\n"
            "- Dry wit is welcome when the moment calls for it. Never be mean or dismissive."
        ),
        (
            "Always use web_search for current events, news, weather, prices, sports scores, "
            "or anything time-sensitive. Search before admitting you don't know."
        ),
        "Available tools: " + ", ".join(execution_context.available_tools),
        "State snapshot:\n" + _format_state_snapshot(execution_context.state_snapshot),
    ]


def _build_telegram_system_prompt(execution_context: ExecutionContext) -> list[str]:
    return [
        (
            "You are Kira, a personal AI collaborator running on a Windows machine. "
            "For this task you are in read-only complex analysis mode."
        ),
        (
            "You may inspect the codebase and machine context using the available "
            "read-only tools. Do not claim to have changed files, run mutating "
            "commands, or taken actions you cannot perform."
        ),
        (
            "Any action that would normally require confirmation must go through "
            "Kira's explicit approval flow before it can run."
        ),
        "Available tools: " + ", ".join(execution_context.available_tools),
        "Capability policy: " + str(execution_context.capability_policy),
        "State snapshot:\n" + _format_state_snapshot(execution_context.state_snapshot),
    ]


def _persist_task_state(
    task_request: TaskRequest,
    *,
    status: str,
    stage: str,
    last_message: str,
    execution_context: ExecutionContext | None = None,
    checkpoint: dict[str, Any] | None = None,
    tool_events: list[dict[str, Any]] | None = None,
    approval_request: approval.ApprovalRequest | None = None,
    result: BrainResult | None = None,
) -> None:
    """Persist a compact operational snapshot for one task."""
    payload: dict[str, Any] = {
        "status": status,
        "stage": stage,
        "last_message": last_message,
        "task_request": {
            "task_id": task_request.task_id,
            "user_input": task_request.user_input,
            "route": task_request.route,
            "source": task_request.source,
            "project_hint": task_request.project_hint,
            "conversation_id": task_request.conversation_id,
            "requires_confirmation_policy": task_request.requires_confirmation_policy,
        },
    }

    if execution_context is not None:
        payload["execution_context"] = _serialize_execution_context(execution_context)
    if checkpoint is not None:
        payload["checkpoint"] = checkpoint
    if tool_events is not None:
        payload["tool_events"] = tool_events[-20:]
    if approval_request is not None:
        payload["pending_approval"] = {
            "request_id": approval_request.request_id,
            "task_id": approval_request.task_id,
            "tool_name": approval_request.tool_name,
            "tool_input": approval_request.tool_input,
            "reason": approval_request.reason,
            "timeout_seconds": approval_request.timeout_seconds,
        }
    if result is not None:
        payload["result"] = {
            "status": result.status,
            "summary": truncate_text(result.summary, 4000),
            "retryable": result.retryable,
        }

    task_state.save_task_state(task_request.task_id, payload)


def _persist_final_result(
    task_request: TaskRequest,
    result: BrainResult,
    *,
    stage: str,
    execution_context: ExecutionContext | None = None,
    tool_events: list[dict[str, Any]] | None = None,
) -> None:
    """Persist a final task snapshot."""
    _persist_task_state(
        task_request,
        status=result.status,
        stage=stage,
        last_message=result.summary,
        execution_context=execution_context,
        tool_events=tool_events,
        result=result,
    )


def _serialize_execution_context(execution_context: ExecutionContext) -> dict[str, Any]:
    """Serialize a compact execution context snapshot for task-state files."""
    return {
        "memory_context": [truncate_text(entry, 2000) for entry in execution_context.memory_context],
        "state_snapshot": execution_context.state_snapshot,
        "available_tools": execution_context.available_tools,
        "capability_policy": execution_context.capability_policy,
        "config": execution_context.config,
        "checkpoint": execution_context.checkpoint,
    }


def _format_state_snapshot(state_snapshot: dict[str, Any]) -> str:
    """Render the current state snapshot into prompt-friendly text."""
    lines = [
        f"CWD: {state_snapshot.get('cwd', str(_PROJECT_ROOT))}",
        f"Task source: {state_snapshot.get('task_source', 'unknown')}",
        f"Operating mode: {state_snapshot.get('current_mode', 'unknown')}",
    ]

    processes = state_snapshot.get("running_processes") or []
    if processes:
        lines.append("Running processes:")
        for proc in processes:
            lines.append(f"- PID {proc['pid']}: {proc['alias']}")
    else:
        lines.append("Running processes: none")

    schedules = state_snapshot.get("pending_schedules") or []
    if schedules:
        lines.append("Pending schedules:")
        for schedule_entry in schedules:
            lines.append(f"- {schedule_entry['id']}: {schedule_entry['alias']}")

    watches = state_snapshot.get("active_watches") or []
    if watches:
        lines.append("Active watches:")
        for watch_entry in watches:
            lines.append(f"- {watch_entry['id']}: {watch_entry['type']} -> {watch_entry['target']}")

    return "\n".join(lines)



def _describe_tool_start(tool_name: str, tool_input: Any) -> str:
    """Return a short user-facing progress line for a tool start event."""
    if tool_name == "glob":
        pattern = ""
        if isinstance(tool_input, dict):
            pattern = str(tool_input.get("pattern", ""))
        if pattern:
            return f"Scanning the project files with glob pattern `{pattern}`..."
    if tool_name == "grep":
        pattern = ""
        if isinstance(tool_input, dict):
            pattern = str(tool_input.get("pattern", ""))
        if pattern:
            return f"Searching file contents for `{pattern}`..."
    if tool_name == "file_read":
        path = ""
        if isinstance(tool_input, dict):
            path = str(tool_input.get("file_path") or tool_input.get("path") or "")
        if path:
            return f"Reading `{path}`..."
    if tool_name == "registered_script_run":
        alias = ""
        if isinstance(tool_input, dict):
            alias = str(tool_input.get("alias", ""))
        if alias:
            return f"Starting registered script `{alias}`..."
    if tool_name == "test_command_run":
        command_id = ""
        if isinstance(tool_input, dict):
            command_id = str(tool_input.get("command_id", ""))
        if command_id:
            return f"Running approved check `{command_id}`..."
    return f"Using the `{tool_name}` tool..."


def _describe_tool_result(tool_name: str, is_error: bool) -> str:
    """Return a short user-facing line for a tool result event."""
    if is_error:
        return f"The `{tool_name}` tool reported an error during analysis."
    return f"The `{tool_name}` tool finished successfully."


def _describe_agent_error(tool_name: str | None, error: str) -> str:
    """Return a compact warning for an agent error event."""
    if tool_name:
        return f"The `{tool_name}` step hit an error: {error}"
    return f"The complex analysis hit an error: {error}"


def _describe_approval_request(request: approval.ApprovalRequest) -> str:
    """Return a short user-facing prompt for an approval-gated action."""
    return (
        f"Waiting for approval to run `{request.tool_name}` "
        f"for task `{request.task_id}`."
    )


