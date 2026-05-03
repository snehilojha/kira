"""Shared text and context utilities for Kira bot modules."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_CONTEXT_PATH = Path(
    os.environ.get("PROJECT_CONTEXT_PATH", str(_PROJECT_ROOT / "context.md"))
)
_PROJECT_CONTEXT_MAX_CHARS = 12_000


def load_project_context(path: Path | None = None) -> str:
    """Return project context file content, capped for prompt safety."""
    target = path if path is not None else _PROJECT_CONTEXT_PATH
    try:
        text = target.read_text(encoding="utf-8", errors="replace").strip()
    except FileNotFoundError:
        logger.warning("Project context file not found at %s", target)
        return ""
    except OSError as exc:
        logger.warning("Could not read project context file %s: %s", target, exc)
        return ""
    if len(text) > _PROJECT_CONTEXT_MAX_CHARS:
        return text[:_PROJECT_CONTEXT_MAX_CHARS] + "\n\n[...project context truncated...]"
    return text


def truncate_text(text: str, max_chars: int) -> str:
    """Truncate from the head, preserving a clear overflow marker."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[...truncated...]"


def tail_text(text: str, max_chars: int) -> str:
    """Return a bounded tail of text, preserving a clear overflow marker."""
    if len(text) <= max_chars:
        return text.strip()
    return "[...output truncated...]\n" + text[-max_chars:].strip()
