"""Intent-detection predicates, phrase tables, and model/filler selection.

Pure routing logic: given a transcript, decide what kind of request it is and
how to handle it. No I/O beyond reading identity facts and shared history.
"""

from __future__ import annotations

from bot import identity
from bot.voice_runtime import session
from bot.voice_runtime.util import _normalize_text


_IDENTITY_QUERIES = {
    "who am i",
    "what do you know about me",
    "what do you know about me?",
    "tell me what you know about me",
    "what have you remembered",
    "what have you remembered about me",
    "what do you remember about me",
    "do you know who i am",
    "who are you",
    "what are you",
    "introduce yourself",
}


def _is_identity_query(text: str) -> bool:
    return _normalize_text(text) in _IDENTITY_QUERIES


_SCREEN_PHRASES = {
    "what's on my screen",
    "what is on my screen",
    "what do you see",
    "what am i working on",
    "what's on screen",
    "what is on screen",
    "look at my screen",
    "what's open",
    "what is open",
    "what's on the screen",
    "describe my screen",
    "describe the screen",
}


def _is_screen_query(text: str) -> bool:
    normalized = _normalize_text(text)
    return normalized in _SCREEN_PHRASES


_MULTISTEP_SIGNALS = (
    " and search ", " and go to ", " and navigate ", " and open ", " and play ",
    " then search ", " then go ", " then open ",
)

_MULTISTEP_EXACT = {
    "play the first video",
    "play first video",
    "click the first video",
    "click first result",
    "play the first result",
    "open the first result",
    "open the first video",
}


def _is_multistep(text: str) -> bool:
    """Return True for phrases that describe a sequence of actions."""
    normalized = _normalize_text(text)
    if normalized in _MULTISTEP_EXACT:
        return True
    return any(signal in normalized for signal in _MULTISTEP_SIGNALS)


# Conversational phrases that Kira answers instantly — no filler needed
_CONVERSATIONAL = {
    "how are you", "how are you doing", "what's up", "whats up",
    "hey", "hello", "hi", "yo", "sup",
    "who are you", "what are you", "introduce yourself",
    "who am i", "what do you know about me",
    "good morning", "good afternoon", "good evening", "good night",
    "thanks", "thank you", "cheers", "cool", "okay", "ok",
}

# Prefixes that signal a slow external lookup is needed
_SEARCH_PREFIXES = (
    "search", "look up", "find out", "google",
    "what's the latest", "what is the latest", "any news",
    "what's happening", "what is happening",
)

# Prefixes that suggest a factual / time-sensitive question needing the smart model
_FACTUAL_PREFIXES = (
    "what's the", "what is the", "who is", "who was", "when did",
    "when is", "where is", "where was", "how much", "how many",
    "how long", "what happened", "tell me about", "explain",
)


def _pick_filler(transcript: str) -> str:
    """Return a filler only when processing will take noticeable time.

    Screen captures, web searches, and multi-step commands get a filler.
    Conversational replies and instant desktop commands get nothing.
    """
    normalized = _normalize_text(transcript)

    # Conversational — Kira knows these instantly
    if normalized in _CONVERSATIONAL:
        return ""

    # Desktop commands — instant
    instant_prefixes = ("open ", "close ", "press ", "click", "scroll", "type ", "volume", "play pause", "mute")
    if any(normalized.startswith(p) for p in instant_prefixes):
        return ""
    if normalized in {"status", "system status", "sysinfo", "system info"}:
        return ""

    # Screen queries — need a screenshot + vision call
    if _is_screen_query(transcript):
        return "Let me take a look."

    # Multi-step actions — involve several sequential steps
    if _is_multistep(transcript):
        return "On it."

    # Explicit search / lookup requests
    if any(normalized.startswith(p) for p in _SEARCH_PREFIXES):
        return "On it."

    # Everything else goes to the LLM — only filler if it looks like
    # a factual/current-data query, not a conversational one.
    if any(normalized.startswith(p) for p in _FACTUAL_PREFIXES):
        return "Let me check."

    return ""


def _pick_model(transcript: str, config) -> str:
    """Return fast_model or smart_model based on query complexity.

    fast  — conversational, follow-ups, simple opinion questions
    smart — explicit searches, time-sensitive data, complex factual queries
    """
    normalized = _normalize_text(transcript)

    # Clearly conversational — fast model is more than enough
    if normalized in _CONVERSATIONAL:
        return config.fast_model

    # Explicit search / web lookup — needs smart for quality synthesis
    if any(normalized.startswith(p) for p in _SEARCH_PREFIXES):
        return config.smart_model

    # Complex factual questions — smart
    if any(normalized.startswith(p) for p in _FACTUAL_PREFIXES):
        return config.smart_model

    # Follow-up questions (short, referencing session history) — fast is fine
    if session.get_session_history() and len(normalized.split()) <= 8:
        return config.fast_model

    # Default: smart for anything ambiguous
    return config.smart_model


def _history_context() -> str:
    """Return the last few commands as a plain-text context string."""
    command_history = session.get_command_history()
    if not command_history:
        return ""
    lines = [f"{i + 1}. User: {t!r} → {r}" for i, (t, r) in enumerate(command_history)]
    return "Recent commands:\n" + "\n".join(lines)


def _build_identity_reply(transcript: str) -> str:
    """Return a spoken summary of Kira's identity or what she knows about the user."""
    normalized = _normalize_text(transcript)
    user_name = identity.get_user_name()
    facts = identity.get_all_facts()

    if normalized in {"who are you", "what are you", "introduce yourself"}:
        name_part = f", {user_name}" if user_name else ""
        return (
            f"I'm Kira{name_part} — your personal AI. "
            "Think of me as your FRIDAY. I run on your PC, I know your setup, "
            "and I'm here whenever you need me."
        )

    # "who am I", "what do you know about me", etc.
    if not facts:
        return (
            f"Honestly, I don't know much about you yet{', ' + user_name if user_name else ''}. "
            "Tell me things and I'll remember them."
        )
    facts_spoken = ". ".join(f[:100] for f in facts[:6])
    name_part = user_name or "you"
    return f"Here's what I know about {name_part}: {facts_spoken}."


# Prefixes that mark a query as informational — skip desktop-action routing entirely.
_QUESTION_PREFIXES = (
    "how", "what", "why", "when", "who", "where", "which",
    "tell me", "explain", "is ", "are ", "was ", "were ", "do ", "does ",
    "did ", "can ", "could ", "would ", "should ", "has ", "have ",
)


def _is_desktop_action_candidate(text: str) -> bool:
    """Return True only for requests that plausibly require clicking/typing on screen.

    Informational questions (how, what, why, tell me…) always go to brain.
    Action verbs without an explicit UI target also go to brain.
    """
    normalized = _normalize_text(text)
    if any(normalized.startswith(p) for p in _QUESTION_PREFIXES):
        return False
    # Explicit action verbs that imply UI manipulation
    _ACTION_VERBS = (
        "click", "press", "scroll", "drag", "select", "highlight",
        "copy", "paste", "close", "minimize", "maximize", "resize",
        "move the", "switch to", "go to", "navigate to", "open ",
        "type ", "fill in", "submit", "right-click",
    )
    return any(normalized.startswith(v) or f" {v}" in normalized for v in _ACTION_VERBS)


_CORRECTION_PHRASES = (
    "that's wrong", "thats wrong", "that is wrong",
    "not what i meant", "not what i said",
    "ignore that", "forget that", "never mind",
    "that's not right", "thats not right", "that is not right",
    "wrong answer", "incorrect", "you misunderstood",
    "that's incorrect", "thats incorrect",
)


def _is_correction(text: str) -> bool:
    normalized = _normalize_text(text)
    return any(phrase in normalized for phrase in _CORRECTION_PHRASES)
