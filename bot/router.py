"""Intent routing helpers for Kira V1.

This module classifies natural-language requests into one of the V1 routes:
`simple`, `complex`, or `monitor`.

It is intentionally standalone so the routing behavior can be tested before
being wired into `/ask` and the future Astra task path.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from bot import provider

Route = Literal["simple", "complex", "monitor"]
RouteSource = Literal["slash", "rule", "model"]

_SIMPLE_PREFIXES = (
    "open ",
    "close ",
    "run ",
    "show ",
    "status",
    "what's ",
    "whats ",
)
_MONITOR_PHRASES = (
    "watch until",
    "alert when",
    "until ",
    "monitor ",
    "keep ",
)
_COMPLEX_PHRASES = (
    " and then ",
    " then ",
    " after that ",
    " once you ",
    " review ",
    " fix ",
    " debug ",
)


@dataclass(frozen=True)
class RoutingDecision:
    """Classification result for one natural-language request."""

    route: Route
    source: RouteSource
    confidence: float
    reason: str
    needs_clarification: bool = False


def classify_by_rule(user_message: str) -> RoutingDecision | None:
    """Return a routing decision for only the clearest deterministic cases."""
    text = _normalize(user_message)
    if not text:
        return None

    if text.startswith("/"):
        return RoutingDecision(
            route="simple",
            source="slash",
            confidence=1.0,
            reason="Explicit slash command bypasses natural-language routing.",
        )

    if _contains_monitor_phrase(text):
        return RoutingDecision(
            route="monitor",
            source="rule",
            confidence=0.92,
            reason="Contains an explicit monitoring/waiting condition.",
        )

    if _contains_complex_phrase(text):
        return RoutingDecision(
            route="complex",
            source="rule",
            confidence=0.88,
            reason="Contains an explicit multi-step or debugging signal.",
        )

    if _is_obvious_simple(text):
        return RoutingDecision(
            route="simple",
            source="rule",
            confidence=0.9,
            reason="Matches an obvious single-action request.",
        )

    return None


async def classify_request(user_message: str) -> RoutingDecision:
    """Classify a request using rules first, then the configured fast model."""
    rule_decision = classify_by_rule(user_message)
    if rule_decision is not None:
        return rule_decision

    return await _classify_with_model(user_message)


async def _classify_with_model(user_message: str) -> RoutingDecision:
    """Use the fast model as a fallback classifier for ambiguous requests."""
    response = await provider.create_chat_completion(
        role="fast",
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the user's request into exactly one route: "
                    "simple, complex, or monitor.\n"
                    "simple = one immediate action with a direct result\n"
                    "complex = multi-step reasoning or deciding what to do next\n"
                    "monitor = waiting, watching, polling, or condition checking over time\n"
                    "Return JSON only: {\"route\": \"simple|complex|monitor\"}"
                ),
            },
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
        max_tokens=30,
    )

    raw_text = response.choices[0].message.content.strip()
    route = _parse_model_route(raw_text)
    if route is None:
        return RoutingDecision(
            route="complex",
            source="model",
            confidence=0.0,
            reason="Model output could not be parsed cleanly.",
            needs_clarification=True,
        )

    return RoutingDecision(
        route=route,
        source="model",
        confidence=0.6,
        reason="Fell back to model classification because no clear rule matched.",
    )


def _normalize(text: str) -> str:
    """Normalize input for routing without losing intent-bearing words."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _contains_monitor_phrase(text: str) -> bool:
    """Return True for requests that obviously describe monitoring over time."""
    if "keep " in text and ("running" in text or "watching" in text):
        return True
    return any(phrase in text for phrase in _MONITOR_PHRASES)


def _contains_complex_phrase(text: str) -> bool:
    """Return True for requests that obviously need multi-step reasoning."""
    if text.startswith(("review ", "debug ", "fix ")):
        return True
    return any(phrase in text for phrase in _COMPLEX_PHRASES)


def _is_obvious_simple(text: str) -> bool:
    """Return True for clear single-action requests only."""
    if any(text.startswith(prefix) for prefix in _SIMPLE_PREFIXES):
        return " until " not in text and " then " not in text
    return False


def _parse_model_route(raw_text: str) -> Route | None:
    """Parse a route from the model response."""
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            return None

    route = str(parsed.get("route", "")).strip().lower()
    if route in {"simple", "complex", "monitor"}:
        return route
    return None
