"""Dataclasses and shared type aliases for the voice runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

ConfirmCallback = Callable[[str], Awaitable[bool]]


@dataclass(frozen=True)
class ParsedCommand:
    """Local voice command after deterministic or LLM parsing."""

    command: str
    args: list[str]
    source: str
    risky: bool = False


@dataclass(frozen=True)
class LocalVoiceResult:
    """Execution result for one local voice command."""

    ok: bool
    message: str
    spoken: str
