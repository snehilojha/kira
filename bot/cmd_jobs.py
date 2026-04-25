"""Brain/job/approval command handlers: history, runs, summarise, reflect, recall, jobs, tasks, mode."""

from __future__ import annotations

import asyncio
import json
import logging
import os

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from bot.auth import require_auth
from bot import approval
from bot import brain
from bot import db
from bot import job_monitor
from bot import mode
from bot import provider
from bot import router
from bot import task_state

logger = logging.getLogger(__name__)

_PENDING_BRAIN_APPROVALS: dict[str, asyncio.Future[bool]] = {}
_BRAIN_APPROVAL_TIMEOUT = 60


# ── Formatters ────────────────────────────────────────────────────

def _format_runtime(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m"


def _format_task_state_summary(state: dict) -> str:
    task_id = _task_state_id(state)
    request = state.get("task_request") or {}
    status = state.get("status", "unknown")
    stage = state.get("stage", "unknown")
    source = request.get("source", "unknown") if isinstance(request, dict) else "unknown"
    updated = str(state.get("updated_at", ""))[:19]
    message = str(state.get("last_message", "")).replace("\n", " ")
    if len(message) > 120:
        message = message[:117] + "..."
    return f"{task_id} | {status}/{stage} | {source} | {updated}\n  {message}"


def _format_task_state_detail(state: dict) -> str:
    task_id = _task_state_id(state)
    request = state.get("task_request") or {}
    lines = [
        f"Task: {task_id}",
        f"Status: {state.get('status', 'unknown')}",
        f"Stage: {state.get('stage', 'unknown')}",
        f"Updated: {state.get('updated_at', 'unknown')}",
    ]

    if isinstance(request, dict):
        lines.extend([
            f"Source: {request.get('source', 'unknown')}",
            f"Route: {request.get('route', 'unknown')}",
            f"Input: {request.get('user_input', '')}",
        ])

    if state.get("interrupted_reason"):
        lines.append(f"Interrupted reason: {state['interrupted_reason']}")

    pending = state.get("pending_approval")
    if isinstance(pending, dict):
        lines.extend([
            "",
            "Pending approval at interruption:",
            f"Request: {pending.get('request_id', '')}",
            f"Tool: {pending.get('tool_name', '')}",
            f"Reason: {pending.get('reason', '')}",
            f"Input: {json.dumps(pending.get('tool_input', {}), ensure_ascii=True)[:800]}",
        ])

    check_events = _extract_check_events(state)
    if check_events:
        lines.extend(["", "Verification checks:"])
        for event in check_events[-3:]:
            status = "failed" if event.get("is_error") else "passed"
            lines.append(f"- test_command_run {status}")
            output_tail = str(event.get("output_tail", "")).strip()
            if output_tail:
                lines.append(output_tail[:1200])

    result = state.get("result")
    if isinstance(result, dict):
        lines.extend(["", "Result:", str(result.get("summary", ""))[:1200]])
    else:
        lines.extend(["", "Last message:", str(state.get("last_message", ""))[:1200]])

    return "\n".join(lines)


def _extract_check_events(state: dict) -> list[dict]:
    events = state.get("tool_events")
    if not isinstance(events, list):
        return []
    return [
        event
        for event in events
        if isinstance(event, dict)
        and event.get("tool_name") == "test_command_run"
        and event.get("type") == "tool_result"
    ]


def _task_state_id(state: dict) -> str:
    if state.get("task_id"):
        return str(state["task_id"])
    request = state.get("task_request")
    if isinstance(request, dict) and request.get("task_id"):
        return str(request["task_id"])
    return "task-unknown"


# ── History & Runs ────────────────────────────────────────────────

@require_auth
async def handle_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/history [n] — show recent conversation history."""
    n = 20
    if context.args:
        try:
            n = int(context.args[0])
        except ValueError:
            await update.message.reply_text("Usage: /history [n]  (n = number of entries)")
            return

    try:
        rows = await db.get_recent_conversations(n)
    except Exception as exc:
        await update.message.reply_text(f"Failed to read history: {exc}")
        return

    if not rows:
        await update.message.reply_text("No conversation history yet.")
        return

    lines = [f"📜 Last {len(rows)} conversation entries:\n"]
    for row in rows:
        ts = row["timestamp"][:16] if row.get("timestamp") else "?"
        role = row["role"].upper()
        content = row["content"][:200]
        lines.append(f"[{ts}] {role}: {content}")

    await update.message.reply_text("\n".join(lines)[:4000])


@require_auth
async def handle_runs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/runs [alias] [n] — show recent run history with metrics."""
    alias = None
    limit = 10

    if context.args:
        args = list(context.args)
        if len(args) >= 1:
            try:
                limit = int(args[-1])
                args = args[:-1]
            except ValueError:
                pass
        if args:
            alias = args[0]

    try:
        rows = await db.get_run_history(alias=alias, limit=limit)
    except Exception as exc:
        await update.message.reply_text(f"Failed to read run history: {exc}")
        return

    if not rows:
        msg = f"No runs recorded for {alias}." if alias else "No runs recorded yet."
        await update.message.reply_text(msg)
        return

    header = f"📊 Last {len(rows)} run(s)"
    if alias:
        header += f" for {alias}"
    lines = [f"{header}:\n"]

    for row in rows:
        date = (row.get("finished_at") or row.get("started_at") or "?")[:16]
        code = row.get("exit_code")
        icon = "✅" if code == 0 else "❌" if code is not None else "?"
        runtime = row.get("runtime_seconds")
        runtime_str = _format_runtime(runtime) if runtime else "?"

        parts = [f"{icon} {row['alias']} ({date}) — {runtime_str}"]
        if (reward := row.get("reward")) is not None:
            parts.append(f"  reward={reward:.4f}")
        if (loss := row.get("loss")) is not None:
            parts.append(f"  loss={loss}")
        if (steps := row.get("total_timesteps")) is not None:
            parts.append(f"  steps={steps:,}")
        lines.append("  ".join(parts))

    await update.message.reply_text("\n".join(lines)[:4000])


# ── Memory commands ───────────────────────────────────────────────

@require_auth
async def handle_summarise(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/summarise — generate and save a GPT summary of today's activity."""
    from bot import memory
    await update.message.reply_text("⏳ Summarising today's activity...")
    try:
        summary = await memory.summarise_today()
        await update.message.reply_text(f"📋 Today's summary:\n\n{summary}")
    except Exception as exc:
        logger.error("handle_summarise failed: %s", exc)
        await update.message.reply_text(f"❌ Summarisation failed: {exc}")


@require_auth
async def handle_reflect(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/reflect — run weekly behavioral reflection immediately."""
    from bot import reflector
    await update.message.reply_text("Running behavioral reflection over the past 7 days...")
    try:
        facts = await reflector.reflect_now()
        facts_text = "\n".join(f"• {f}" for f in facts)
        await update.message.reply_text(f"Reflection complete. Updated facts:\n\n{facts_text}")
    except Exception as exc:
        logger.error("handle_reflect failed: %s", exc)
        await update.message.reply_text(f"Reflection failed: {exc}")


@require_auth
async def handle_recall(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/recall <query> — ask a NL question over recent session history."""
    query = " ".join(context.args) if context.args else ""
    if not query:
        await update.message.reply_text("Usage: /recall <your question about past sessions>")
        return

    await update.message.reply_text("🔍 Searching session history...")

    try:
        rows = await db.get_recent_sessions(7)
    except Exception as exc:
        await update.message.reply_text(f"❌ Failed to fetch session history: {exc}")
        return

    if not rows:
        await update.message.reply_text("No session history found yet. Run /summarise after some activity.")
        return

    session_block = "\n".join(
        f"[{r.get('date', '?')}] {r.get('summary', '')}"
        for r in rows
    )

    api_key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("KIRA_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
    )
    if not api_key:
        await update.message.reply_text("❌ OPENAI_API_KEY not set.")
        return

    try:
        response = await provider.create_chat_completion(
            role="fast",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are answering questions about a developer's past work sessions. "
                        "Answer concisely based only on the provided session summaries."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Session summaries:\n\n{session_block}\n\nQuestion: {query}",
                },
            ],
            max_tokens=300,
            temperature=0.3,
        )
        answer = response.choices[0].message.content.strip()
        await update.message.reply_text(answer)
    except Exception as exc:
        logger.error("handle_recall GPT call failed: %s", exc)
        await update.message.reply_text(f"❌ Recall failed: {exc}")


# ── Monitor jobs ──────────────────────────────────────────────────

@require_auth
async def handle_jobs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/jobs — list all monitor jobs."""
    jobs = job_monitor.list_jobs()
    if not jobs:
        await update.message.reply_text("No monitor jobs running.")
        return

    lines = ["**Monitor jobs:**\n"]
    for job in jobs:
        icon = {"active": "🟢", "paused": "⏸", "cancelled": "🔴", "expired": "⏰", "fired": "✅"}.get(
            job.status, "❓"
        )
        interval = int(job.poll_interval_seconds)
        fired_note = f" | last fired: {job.last_fired_at[:19]}" if job.last_fired_at else ""
        lines.append(
            f"{icon} `{job.job_id}` **{job.name}** ({job.status})\n"
            f"  Subject: {job.subject}\n"
            f"  Condition: {job.condition}\n"
            f"  Polling every {interval}s{fired_note}\n"
        )

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


@require_auth
async def handle_cancel_job(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/canceljob <job_id> — cancel a monitor job."""
    if not context.args:
        await update.message.reply_text("Usage: /canceljob <job_id>")
        return
    job_id = context.args[0].strip()
    ok = await job_monitor.cancel_job(job_id)
    if ok:
        await update.message.reply_text(f"Monitor job `{job_id}` cancelled.", parse_mode="Markdown")
    else:
        await update.message.reply_text(f"No active job found with ID `{job_id}`.", parse_mode="Markdown")


@require_auth
async def handle_pause_job(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/pausejob <job_id> — pause a monitor job."""
    if not context.args:
        await update.message.reply_text("Usage: /pausejob <job_id>")
        return
    job_id = context.args[0].strip()
    ok = await job_monitor.pause_job(job_id)
    if ok:
        await update.message.reply_text(f"Monitor job `{job_id}` paused.", parse_mode="Markdown")
    else:
        await update.message.reply_text(
            f"Could not pause `{job_id}` — not found or not active.", parse_mode="Markdown"
        )


@require_auth
async def handle_resume_job(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/resumejob <job_id> — resume a paused monitor job."""
    if not context.args:
        await update.message.reply_text("Usage: /resumejob <job_id>")
        return
    job_id = context.args[0].strip()
    ok = await job_monitor.resume_job(job_id)
    if ok:
        await update.message.reply_text(f"Monitor job `{job_id}` resumed.", parse_mode="Markdown")
    else:
        await update.message.reply_text(
            f"Could not resume `{job_id}` — not found or not paused.", parse_mode="Markdown"
        )


# ── Mode ──────────────────────────────────────────────────────────

@require_auth
async def handle_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/mode — show Kira's current operating mode and presence status."""
    current = mode.get_mode()
    idle_seconds = mode.get_last_input_seconds()

    mode_icons = {
        "idle": "⏸",
        "active_session": "🟢",
        "autonomous": "🤖",
        "awaiting_confirmation": "⏳",
        "recovering": "🔄",
    }
    icon = mode_icons.get(current, "❓")

    if idle_seconds < 60:
        input_label = f"{idle_seconds:.0f}s ago"
    elif idle_seconds < 3600:
        input_label = f"{idle_seconds / 60:.0f}m ago"
    else:
        input_label = f"{idle_seconds / 3600:.1f}h ago"

    lines = [
        f"{icon} *Mode:* `{current}`",
        f"*Last input:* {input_label}",
    ]

    try:
        transitions = await db.get_recent_mode_transitions(3)
        if transitions:
            lines.append("\n*Recent transitions:*")
            for t in transitions:
                arrow = f"{t['from_mode'] or 'start'} → {t['to_mode']}"
                lines.append(f"  `{arrow}` — {t['occurred_at']}")
    except Exception:
        pass

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


# ── Complex task state ────────────────────────────────────────────

@require_auth
async def handle_tasks(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/tasks [n] — list recent complex task states."""
    try:
        limit = int(context.args[0]) if context.args else 10
    except (ValueError, IndexError):
        limit = 10

    states = task_state.list_recent_task_states(max(1, min(limit, 25)))
    if not states:
        await update.message.reply_text("No complex task state recorded yet.")
        return

    lines = ["Recent complex tasks:\n"]
    for state in states:
        lines.append(_format_task_state_summary(state))

    await update.message.reply_text("\n".join(lines)[:4000])


@require_auth
async def handle_task(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/task <task_id> — show details for one complex task."""
    if not context.args:
        await update.message.reply_text("Usage: /task <task_id>")
        return
    task_id = context.args[0].strip()
    state = task_state.load_task_state(task_id)
    if state is None:
        await update.message.reply_text(f"No task state found for {task_id}.")
        return
    await update.message.reply_text(_format_task_state_detail(state)[:4000])


# ── Brain approval helpers ────────────────────────────────────────

async def run_complex_task_with_progress(reply, task_request) -> brain.BrainResult:
    """Run a complex task and stream meaningful progress updates to chat."""
    final_result: brain.BrainResult | None = None

    async def _approval_callback(request: approval.ApprovalRequest) -> bool:
        return await _await_brain_approval(reply, request)

    async for event in brain.run_complex_task_stream(
        task_request,
        approval_callback=_approval_callback,
    ):
        if event.result is not None:
            final_result = event.result
            continue
        if event.event_type in {"status", "tool", "warning"}:
            await reply(event.message[:4000])

    if final_result is None:
        final_result = brain.BrainResult(
            task_id=getattr(task_request, "task_id", "task-unknown"),
            status="failed",
            summary="The complex-task runtime ended without a final result.",
            retryable=True,
        )

    return final_result


async def _await_brain_approval(reply, request: approval.ApprovalRequest) -> bool:
    loop = asyncio.get_running_loop()
    future: asyncio.Future[bool] = loop.create_future()
    _PENDING_BRAIN_APPROVALS[request.request_id] = future

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Approve", callback_data=f"brain_yes|{request.request_id}"),
            InlineKeyboardButton("Deny", callback_data=f"brain_no|{request.request_id}"),
        ]
    ])
    await reply(
        _build_brain_approval_text(request),
        reply_markup=keyboard,
        parse_mode="Markdown",
    )

    try:
        decision = await asyncio.wait_for(
            future,
            timeout=min(request.timeout_seconds, _BRAIN_APPROVAL_TIMEOUT),
        )
        return bool(decision)
    except asyncio.TimeoutError:
        _PENDING_BRAIN_APPROVALS.pop(request.request_id, None)
        await reply(
            f"Approval timed out for `{request.tool_name}`. I treated it as denied.",
            parse_mode="Markdown",
        )
        return False


def _build_brain_approval_text(request: approval.ApprovalRequest) -> str:
    input_preview = json.dumps(request.tool_input, ensure_ascii=True)
    if len(input_preview) > 600:
        input_preview = input_preview[:600] + "...[truncated]"
    return (
        "Approval required for complex task:\n"
        f"`{request.task_id}`\n\n"
        f"Tool: `{request.tool_name}`\n"
        f"Reason: {request.reason}\n"
        f"Input: `{input_preview}`\n\n"
        "Approve this action?"
    )


# ── Monitor job parsing (used by /ask routing) ────────────────────

import re as _re


async def parse_monitor_job(user_message: str) -> dict | None:
    """Use the fast LLM to extract monitor job fields from natural language."""
    system_prompt = (
        "Extract a monitor job specification from the user's request. "
        "Return ONLY a JSON object — no prose, no markdown fences.\n\n"
        "Required fields:\n"
        "  name          - short identifier for this job (snake_case)\n"
        "  subject       - what to watch (file path, script name, or description)\n"
        "  condition     - natural-language condition that triggers the alert\n"
        "  poll_interval_seconds - how often to check (integer seconds, min 10)\n"
        "  success_action - message to send when condition is met\n\n"
        "Optional fields:\n"
        "  failure_action - message to send if monitoring fails (or null)\n"
        "  expiry_at      - ISO datetime when to stop monitoring (or null)\n\n"
        "Example output:\n"
        '{"name": "crypto_bot_exit", "subject": "crypto_bot", '
        '"condition": "the process has exited", "poll_interval_seconds": 30, '
        '"success_action": "crypto_bot has exited, sir.", '
        '"failure_action": null, "expiry_at": null}'
    )

    try:
        response = await provider.create_chat_completion(
            role="fast",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            max_tokens=300,
        )
    except Exception as exc:
        logger.warning("parse_monitor_job: API call failed: %s", exc)
        return None

    raw = (response.choices[0].message.content or "").strip()
    raw = _re.sub(r"^```[a-z]*\n?", "", raw)
    raw = _re.sub(r"\n?```$", "", raw).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = _re.search(r'\{.*\}', raw, _re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                logger.warning("parse_monitor_job: unparseable response: %r", raw)
                return None
        else:
            logger.warning("parse_monitor_job: no JSON in response: %r", raw)
            return None

    required = ("name", "subject", "condition", "poll_interval_seconds", "success_action")
    if not all(k in parsed for k in required):
        logger.warning("parse_monitor_job: missing required fields: %s", parsed)
        return None

    return parsed


async def create_monitor_job_from_message(user_message: str, reply) -> bool:
    """Parse, create, and confirm a monitor job from natural language."""
    spec = await parse_monitor_job(user_message)
    if spec is None:
        await reply(_build_route_stub_text(
            router.RoutingDecision(
                route="monitor", source="rule", confidence=0.9,
                reason="could not parse job spec",
            )
        ))
        return False

    try:
        job = await job_monitor.create_job(
            name=spec["name"],
            subject=spec["subject"],
            condition=spec["condition"],
            poll_interval_seconds=float(spec.get("poll_interval_seconds", 60)),
            success_action=spec["success_action"],
            failure_action=spec.get("failure_action"),
            expiry_at=spec.get("expiry_at"),
        )
    except Exception as exc:
        logger.error("create_monitor_job_from_message: create failed: %s", exc)
        await reply(f"Failed to create monitor job: {exc}")
        return False

    interval = int(job.poll_interval_seconds)
    expiry_note = f"\nExpires: {job.expiry_at}" if job.expiry_at else ""
    await reply(
        f"Monitor job created.\n"
        f"ID: `{job.job_id}`\n"
        f"Name: {job.name}\n"
        f"Subject: {job.subject}\n"
        f"Condition: {job.condition}\n"
        f"Polling every {interval}s{expiry_note}\n\n"
        f"I'll notify you when the condition is met."
    )
    return True


def _build_route_stub_text(decision: router.RoutingDecision) -> str:
    """Return a user-facing message for routes that are not live yet."""
    if decision.needs_clarification:
        return (
            "I could not confidently route that request yet. "
            "Please be more specific or use a direct Kira command."
        )

    if decision.route == "complex":
        return (
            f"Complex route selected ({decision.source}). "
            "Multi-step execution is not wired in yet, so please use a direct "
            "command or break the request into smaller steps for now."
        )

    if decision.route == "monitor":
        return (
            f"Monitor route selected ({decision.source}). "
            "Could not parse the job details — please be more specific, e.g. "
            "'monitor crypto_bot until it exits' or 'watch training loss until it drops below 0.2'."
        )

    return (
        "I could not route that request safely yet. "
        "Please try a direct command."
    )
