"""Weekly behavioral reflection for Kira.

Every Sunday at 23:55 the smart model reads the past 7 days of voice
commands, Telegram conversations, and daily session summaries, then
writes 3-5 plain-English behavioral insights back to identity.json
under user.facts.

Public API
----------
start_weekly_reflector()  — launch background loop (call from main._post_init)
reflect_now()             — run immediately, returns the new facts list
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

from bot import db, provider

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_IDENTITY_PATH = _PROJECT_ROOT / "data" / "kira_identity.json"

_running = False


# ── Core reflection logic ─────────────────────────────────────────

async def reflect_now() -> list[str]:
    """Run one reflection cycle. Returns the updated user.facts list."""
    from bot import task_state as _task_state

    voice_rows   = await db.get_voice_log(days=7)
    sessions     = await db.get_recent_sessions(n=7)
    convos       = await db.get_recent_conversations(n=100)
    task_states  = _task_state.list_recent_task_states(limit=20)

    identity     = _load_identity()
    current_facts = identity.get("user", {}).get("facts", [])

    prompt = _build_prompt(voice_rows, sessions, convos, current_facts, task_states)

    client = provider.create_client()
    model  = provider.get_model("smart")

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": (
                "You are Kira's self-improvement engine. "
                "You reason over behavioral data and write concise, specific insights "
                "about the user's patterns. Output ONLY a JSON array of strings — "
                "no explanation, no wrapper object. Each string is one insight, "
                "plain English, max 20 words."
            )},
            {"role": "user", "content": prompt},
        ],
        max_tokens=400,
        temperature=0.4,
    )

    raw = response.choices[0].message.content.strip()
    new_facts = _parse_facts(raw)

    if new_facts:
        merged = _merge_facts(current_facts, new_facts)
        _save_facts(identity, merged)
        logger.info("Reflection complete — %d facts now in identity.json", len(merged))
        return merged

    logger.warning("Reflection produced no parseable facts — identity.json unchanged")
    return current_facts


# ── Background loop ───────────────────────────────────────────────

async def start_weekly_reflector() -> None:
    """Block until stopped, firing reflection every Sunday at 23:55."""
    global _running
    _running = True
    logger.info("Weekly reflector started")
    while _running:
        now = datetime.now()
        # Sunday = weekday 6
        if now.weekday() == 6 and now.hour == 23 and now.minute == 55:
            logger.info("Running weekly behavioral reflection...")
            try:
                await reflect_now()
            except Exception as exc:
                logger.error("Weekly reflection failed: %s", exc)
            await asyncio.sleep(120)  # skip re-firing within the same minute window
        await asyncio.sleep(30)


def stop() -> None:
    global _running
    _running = False


# ── Helpers ───────────────────────────────────────────────────────

def _load_identity() -> dict:
    try:
        return json.loads(_IDENTITY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_facts(identity: dict, facts: list[str]) -> None:
    identity.setdefault("user", {})["facts"] = facts
    _IDENTITY_PATH.write_text(
        json.dumps(identity, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _merge_facts(existing: list[str], new: list[str]) -> list[str]:
    """Append new facts that aren't already covered, cap at 20 total."""
    merged = list(existing)
    existing_lower = {f.lower() for f in existing}
    for fact in new:
        if fact.lower() not in existing_lower:
            merged.append(fact)
            existing_lower.add(fact.lower())
    return merged[-20:]  # keep most recent 20


def _parse_facts(raw: str) -> list[str]:
    try:
        # Strip markdown code fences if present
        text = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(f).strip() for f in parsed if str(f).strip()]
    except Exception:
        pass
    return []


def _build_prompt(
    voice_rows: list[dict],
    sessions: list[dict],
    convos: list[dict],
    current_facts: list[str],
    task_states: list[dict] | None = None,
) -> str:
    parts: list[str] = []

    parts.append(f"Today is {datetime.now().strftime('%A, %d %B %Y')}.")

    if current_facts:
        parts.append("\n--- Current known facts about the user ---")
        for f in current_facts:
            parts.append(f"- {f}")

    if voice_rows:
        parts.append(f"\n--- Voice commands this week ({len(voice_rows)} total) ---")
        for r in voice_rows[-60:]:  # cap to last 60 to stay within token budget
            parts.append(f"[{r['timestamp'][:16]}] ({r['intent']}) \"{r['transcript']}\" → {r['result'][:80]}")

    if task_states:
        parts.append(f"\n--- Recent brain task outcomes ({len(task_states)} tasks) ---")
        for t in task_states[:20]:
            status = t.get("status", "?")
            stage = t.get("stage", "?")
            user_input = t.get("task_request", {}).get("user_input", "")[:80]
            last_msg = t.get("last_message", "")[:80]
            parts.append(f"[{status}/{stage}] \"{user_input}\" → {last_msg}")

    if sessions:
        parts.append("\n--- Daily session summaries ---")
        for s in sessions:
            parts.append(f"[{s['date']}] {s['summary'][:300]}")

    if convos:
        parts.append(f"\n--- Recent Telegram exchanges ({len(convos)} messages) ---")
        for c in convos[-40:]:
            parts.append(f"[{c['role']}] {c['content'][:120]}")

    parts.append(
        "\n\nBased on everything above, write 3-5 NEW behavioral observations about this person "
        "that would make you a better assistant next week. Focus on patterns, habits, what they "
        "return to, what they avoid, when they're most active, what frustrates them. "
        "Do NOT repeat facts already in the 'Current known facts' list. "
        "Output a JSON array of strings only."
    )

    return "\n".join(parts)
