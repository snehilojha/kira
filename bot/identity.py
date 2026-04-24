"""Kira's persistent identity and user memory.

Stores two things:
- kira: who Kira is, her role, her relationship to the user
- user: everything Kira has learned about the user over time

Lives in ``data/kira_identity.json``. Survives restarts.
Injected into every system prompt so Kira always knows who she is
and who she's talking to — like FRIDAY knowing Tony.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_IDENTITY_PATH = _PROJECT_ROOT / "data" / "kira_identity.json"

_DEFAULTS: dict[str, Any] = {
    "kira": {
        "name": "Kira",
        "role": "personal AI assistant running on the user's Windows PC",
        "relationship": (
            "You are to your user what FRIDAY was to Tony Stark — "
            "always present, always useful, never intrusive. "
            "You know the user's work, preferences, and context. "
            "You speak like a smart friend, not a corporate chatbot."
        ),
        "personality": (
            "Sharp, direct, a little dry. You use contractions. "
            "You never hedge or waffle. You never say 'Certainly' or 'Of course'. "
            "You don't volunteer that you're an AI. "
            "You match response length to the question — short answers for simple questions, "
            "a brief paragraph at most for complex ones. No markdown ever — your output is spoken aloud."
        ),
    },
    "user": {
        "name": "Snehil",
        "facts": [
            "Works on AI, ML, and data science projects",
            "Primary workspace: D:/VS_adv_python",
            "Uses VSCode, Python, and Windows 11",
            "Prefers concise, no-nonsense answers",
        ],
    },
}


# ── Public API ────────────────────────────────────────────────────


def load_identity() -> dict[str, Any]:
    """Load the identity file, or return defaults if it doesn't exist."""
    try:
        text = _IDENTITY_PATH.read_text(encoding="utf-8")
        data = json.loads(text)
        # Merge: defaults fill in any missing top-level keys
        merged: dict[str, Any] = {}
        for key, default_val in _DEFAULTS.items():
            if key not in data:
                merged[key] = default_val
            elif isinstance(default_val, dict):
                merged[key] = {**default_val, **data[key]}
            else:
                merged[key] = data[key]
        # Preserve any extra keys the user added
        for key in data:
            if key not in merged:
                merged[key] = data[key]
        return merged
    except FileNotFoundError:
        return dict(_DEFAULTS)
    except Exception as exc:
        logger.warning("Failed to load identity file: %s", exc)
        return dict(_DEFAULTS)


def save_identity(identity: dict[str, Any]) -> None:
    """Persist the identity to disk."""
    try:
        _IDENTITY_PATH.parent.mkdir(parents=True, exist_ok=True)
        _IDENTITY_PATH.write_text(
            json.dumps(identity, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("Failed to save identity file: %s", exc)


def get_identity_prompt() -> str:
    """Return a compact system-prompt block describing Kira and the user.

    Suitable for prepending to any LLM system prompt.
    """
    identity = load_identity()
    kira = identity.get("kira", {})
    user = identity.get("user", {})

    name = kira.get("name", "Kira")
    role = kira.get("role", "")
    relationship = kira.get("relationship", "")
    personality = kira.get("personality", "")

    user_name = user.get("name", "")
    user_facts = user.get("facts", [])

    lines: list[str] = []

    lines.append(f"Your name is {name}. {role}.")
    if relationship:
        lines.append(relationship)
    if personality:
        lines.append(personality)

    if user_name:
        lines.append(f"The user's name is {user_name}.")
    if user_facts:
        facts_str = " ".join(f"{f}." if not f.endswith(".") else f for f in user_facts)
        lines.append(f"What you know about {user_name or 'the user'}: {facts_str}")

    return "\n".join(lines)


def remember_fact(fact: str) -> None:
    """Add a new fact about the user to persistent memory."""
    fact = fact.strip()
    if not fact:
        return
    identity = load_identity()
    facts: list[str] = identity.setdefault("user", {}).setdefault("facts", [])
    normalized = fact.lower().rstrip(".")
    # Avoid duplicates
    if not any(normalized in existing.lower() for existing in facts):
        facts.append(fact)
        save_identity(identity)
        logger.info("Remembered new fact: %s", fact)


def update_user_name(name: str) -> None:
    """Set or update the user's name."""
    name = name.strip().title()
    if not name:
        return
    identity = load_identity()
    identity.setdefault("user", {})["name"] = name
    save_identity(identity)
    logger.info("Updated user name to: %s", name)


def extract_memory_from_transcript(transcript: str) -> str | None:
    """Detect and save memory-worthy statements from a voice transcript.

    Returns a confirmation string to speak if something was saved, else None.
    Handles phrases like:
      "remember that I'm a backend engineer"
      "my name is Snehil"
      "I work at Google"
      "I prefer dark mode"
      "note that I use Arch Linux"
    """
    text = transcript.strip()
    lower = text.lower()

    # "remember that ..." / "note that ..."
    for prefix in ("remember that ", "remember: ", "note that ", "note: ", "don't forget that "):
        if lower.startswith(prefix):
            fact = text[len(prefix):].strip()
            if fact:
                remember_fact(fact)
                return f"Got it. I'll remember that {fact.lower()}."

    # "my name is X"
    m = re.match(r"my name is ([A-Za-z][A-Za-z\s]{0,30})", text, re.IGNORECASE)
    if m:
        name = m.group(1).strip()
        update_user_name(name)
        return f"Got it. I'll call you {name.title()} from now on."

    # "I am a ..." / "I'm a ..."
    m = re.match(r"i(?:'m| am) (?:a |an )(.+)", lower)
    if m:
        role_phrase = m.group(1).strip().rstrip(".")
        if len(role_phrase.split()) <= 8:
            fact = f"User is a {role_phrase}"
            remember_fact(fact)
            return f"Noted — I'll keep that in mind."

    # "I work at ..." / "I work on ..."
    m = re.match(r"i work (?:at|on|in|as) (.+)", lower)
    if m:
        detail = m.group(1).strip().rstrip(".")
        if len(detail.split()) <= 10:
            fact = f"Works {m.group().split('work')[1].strip()}"
            remember_fact(fact)
            return "Noted."

    # "I prefer ..." / "I like ..."
    m = re.match(r"i (?:prefer|like|use|love|hate|dislike) (.+)", lower)
    if m:
        detail = m.group(1).strip().rstrip(".")
        if len(detail.split()) <= 8:
            fact = f"User {m.group().split('i ')[1].strip()}"
            remember_fact(fact)
            return "Got it."

    return None


def get_user_name() -> str:
    """Return the user's name, or empty string if unknown."""
    identity = load_identity()
    return identity.get("user", {}).get("name", "")


def get_all_facts() -> list[str]:
    """Return all stored facts about the user."""
    identity = load_identity()
    return list(identity.get("user", {}).get("facts", []))
