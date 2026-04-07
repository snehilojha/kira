"""Daily session summariser for Kira.

Runs as a background asyncio Task started from ``main.py``.
At 23:55 each day (or on manual ``/summarise``), pulls today's run history,
sends it to GPT-4o Mini for a 3-4 sentence summary, and stores the result
in the ``sessions`` table.

Public API
----------
- ``start_daily_summariser()``  — background loop, called from main.py
- ``summarise_today()``         — callable from /summarise handler
- ``get_recent_sessions(n)``    — formatted string for prompt injection
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, date, timedelta
from typing import Any

logger = logging.getLogger(__name__)

_SUMMARISE_HOUR = 23
_SUMMARISE_MINUTE = 55

_GPT_MODEL = "gpt-4o-mini"
_MAX_HISTORY_ENTRIES = 50       # cap sent to GPT
_MAX_SUMMARY_CHARS = 600        # cap per session summary in prompt injection


async def summarise_today() -> str:
    """Build and persist a GPT summary of today's activity.

    Returns:
        The summary text, or an error message string.
    """
    from bot import db

    today_str = date.today().isoformat()

    # Pull today's run history
    rows = await db.get_run_history(limit=_MAX_HISTORY_ENTRIES)
    today_rows = [r for r in rows if r.get("started_at", "").startswith(today_str)]

    if not today_rows:
        return "No runs recorded today — nothing to summarise."

    raw_events = _format_runs_for_gpt(today_rows)
    summary = await _call_gpt_summarise(raw_events)

    import json
    await db.save_session(
        date=today_str,
        summary=summary,
        raw_events=json.dumps(today_rows),
    )

    logger.info("Session summary saved for %s", today_str)
    return summary


async def get_recent_sessions(n: int = 3) -> str:
    """Return a formatted string of the last *n* session summaries for prompt injection.

    Returns an empty string if no sessions exist yet.
    """
    from bot import db

    rows = await db.get_recent_sessions(n)
    if not rows:
        return ""

    lines = ["Recent session history:"]
    for row in rows:
        date_str = row.get("date", "unknown date")
        summary = row.get("summary", "")[:_MAX_SUMMARY_CHARS]
        lines.append(f"  [{date_str}] {summary}")

    return "\n".join(lines)


async def start_daily_summariser() -> None:
    """Background loop that triggers a session summary each day at 23:55."""
    logger.info("Daily session summariser started (fires at %02d:%02d)", _SUMMARISE_HOUR, _SUMMARISE_MINUTE)

    while True:
        try:
            delay = _seconds_until_next_summarise()
            logger.debug("Session summariser sleeping %.0fs until next run", delay)
            await asyncio.sleep(delay)

            logger.info("Triggering daily session summary")
            summary = await summarise_today()
            from bot import notifier
            await notifier.send(f"📋 Daily summary:\n{summary}")

        except asyncio.CancelledError:
            logger.info("Daily session summariser cancelled")
            raise
        except Exception as exc:
            logger.exception("Daily session summariser error: %s", exc)
            # Back off 5 minutes before retrying so we don't spam on persistent errors
            await asyncio.sleep(300)


# ── Internal helpers ──────────────────────────────────────────────

def _seconds_until_next_summarise() -> float:
    """Return seconds until the next 23:55 trigger."""
    now = datetime.now()
    target = now.replace(hour=_SUMMARISE_HOUR, minute=_SUMMARISE_MINUTE, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def _format_runs_for_gpt(rows: list[dict[str, Any]]) -> str:
    """Format run history rows into a compact text block for the GPT prompt."""
    lines = []
    for r in rows:
        alias = r.get("alias", "unknown")
        started = r.get("started_at", "?")
        exit_code = r.get("exit_code")
        runtime = r.get("runtime_seconds")
        reward = r.get("reward")
        timesteps = r.get("total_timesteps")

        parts = [f"- {alias}  started={started}"]
        if exit_code is not None:
            parts.append(f"exit={exit_code}")
        if runtime is not None:
            parts.append(f"runtime={runtime:.1f}s")
        if timesteps is not None:
            parts.append(f"steps={timesteps}")
        if reward is not None:
            parts.append(f"reward={reward:.4f}")

        lines.append("  ".join(parts))

    return "\n".join(lines)


async def _call_gpt_summarise(raw_events: str) -> str:
    """Send run history to GPT-4o Mini and return a 3-4 sentence summary."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "(Summary unavailable — OPENAI_API_KEY not set)"

    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)

        response = await client.chat.completions.create(
            model=_GPT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are summarising a developer's machine activity for the day. "
                        "Be concise. Write 3-4 sentences covering what ran, "
                        "whether it succeeded or failed, and any notable metrics like "
                        "training reward or step count. Do not use bullet points."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Today's run history:\n\n{raw_events}\n\nSummarise.",
                },
            ],
            max_tokens=200,
            temperature=0.3,
        )

        return response.choices[0].message.content.strip()

    except Exception as exc:
        logger.error("GPT summarisation failed: %s", exc)
        return f"(Summary failed: {exc})"
