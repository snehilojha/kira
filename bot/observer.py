"""Machine awareness layer for Kira.

Runs as a background asyncio Task started from ``main.py``.
Every OBSERVER_INTERVAL seconds (default 900 = 15 min), takes a snapshot of:
  - Recently modified files in PROJECT_ROOTS (last 48h)
  - Git log + status for each project root
  - Active processes from process_registry
  - Log tails for running processes

The raw snapshot is persisted to ``db.observations`` and summarised by
GPT-4o Mini into a short string available via ``get_current_context()``.
That string is injected into the /ask system prompt for situational awareness.

Public API
----------
- ``start()``               — background loop, called from main.py
- ``get_current_context()`` — returns latest GPT summary string (or raw fallback)
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_OBSERVER_INTERVAL = int(os.environ.get("OBSERVER_INTERVAL", "40000")) 
_RECENT_FILE_HOURS = 48
_MAX_FILES_PER_ROOT = 20
_MAX_LOG_TAIL_LINES = 30
_GPT_MODEL = "gpt-4o-mini"

# Module-level cached context string
_CURRENT_CONTEXT_SUMMARY: str = ""


def get_current_context() -> str:
    """Return the latest GPT-summarised machine context, or empty string if not ready."""
    return _CURRENT_CONTEXT_SUMMARY


async def start() -> None:
    """Background loop: snapshot machine state every OBSERVER_INTERVAL seconds."""
    logger.info("Observer started (interval=%ds)", _OBSERVER_INTERVAL)

    while True:
        try:
            await _run_cycle()
        except asyncio.CancelledError:
            logger.info("Observer cancelled")
            raise
        except Exception as exc:
            logger.exception("Observer cycle failed: %s", exc)

        await asyncio.sleep(_OBSERVER_INTERVAL)


async def _run_cycle() -> None:
    """Collect a snapshot, persist it, and update the cached summary."""
    global _CURRENT_CONTEXT_SUMMARY

    snapshot = await asyncio.to_thread(_collect_snapshot)

    try:
        from bot import db
        await db.save_observation(snapshot)
    except Exception as exc:
        logger.warning("Failed to persist observation: %s", exc)

    summary = await _summarise_snapshot(snapshot)
    _CURRENT_CONTEXT_SUMMARY = summary
    logger.debug("Observer context updated (%d chars)", len(summary))


# ── Snapshot collection ───────────────────────────────────────────

def _collect_snapshot() -> dict[str, Any]:
    """Collect all machine-state data synchronously (runs in a thread)."""
    project_roots = _get_project_roots()

    recent_files = _collect_recent_files(project_roots)
    git_statuses = _collect_git_statuses(project_roots)
    running_procs = _collect_running_procs()
    log_tails = _collect_log_tails()

    return {
        "observed_at": datetime.now().isoformat(timespec="seconds"),
        "active_projects": _format_git_statuses(git_statuses),
        "recent_files": _format_recent_files(recent_files),
        "git_status": _format_git_statuses(git_statuses),
        "running_procs": _format_running_procs(running_procs),
        "screen_summary": _format_log_tails(log_tails),
    }


def _get_project_roots() -> list[Path]:
    """Read PROJECT_ROOTS from env, fall back to sensible defaults."""
    raw = os.environ.get("PROJECT_ROOTS", "")
    if raw.strip():
        roots = [Path(p.strip()) for p in raw.split(",") if p.strip()]
    else:
        roots = [
            Path("D:/VS_adv_python"),
            Path("D:/AI_tools"),
            Path("D:/DS_ML_AI_journey"),
        ]
    return [r for r in roots if r.exists()]


def _collect_recent_files(roots: list[Path]) -> dict[str, list[str]]:
    """Find files modified in the last 48h under each project root."""
    cutoff = datetime.now() - timedelta(hours=_RECENT_FILE_HOURS)
    result: dict[str, list[str]] = {}

    # Extensions worth tracking
    _TRACK_EXTS = {
        ".py", ".ipynb", ".yaml", ".yml", ".toml", ".json",
        ".sh", ".bat", ".md", ".txt", ".cfg", ".ini",
    }
    # Directories to skip
    _SKIP_DIRS = {
        ".git", "__pycache__", ".venv", "venv", "env", "node_modules",
        ".mypy_cache", ".pytest_cache", "dist", "build", ".tox",
    }

    for root in roots:
        files: list[str] = []
        try:
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                # Skip ignored dirs
                if any(part in _SKIP_DIRS for part in path.parts):
                    continue
                if path.suffix.lower() not in _TRACK_EXTS:
                    continue
                try:
                    mtime = datetime.fromtimestamp(path.stat().st_mtime)
                    if mtime >= cutoff:
                        files.append(str(path.relative_to(root)))
                except OSError:
                    continue

                if len(files) >= _MAX_FILES_PER_ROOT:
                    break
        except Exception as exc:
            logger.warning("Error scanning %s: %s", root, exc)

        if files:
            result[str(root)] = sorted(files)

    return result


def _collect_git_statuses(roots: list[Path]) -> dict[str, dict[str, str]]:
    """Run git log + git status for each root that is a git repo."""
    result: dict[str, dict[str, str]] = {}

    for root in roots:
        if not (root / ".git").exists():
            # Try to find git repos one level deep
            try:
                sub_roots = [p for p in root.iterdir() if p.is_dir() and (p / ".git").exists()]
            except OSError:
                continue
        else:
            sub_roots = [root]

        for repo in sub_roots:
            try:
                log_out = subprocess.run(
                    ["git", "-C", str(repo), "log", "--oneline", "-5"],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip()

                status_out = subprocess.run(
                    ["git", "-C", str(repo), "status", "--short"],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip()

                if log_out or status_out:
                    result[str(repo)] = {
                        "log": log_out or "(no commits)",
                        "status": status_out or "(clean)",
                    }
            except Exception as exc:
                logger.debug("git failed for %s: %s", repo, exc)

    return result


def _collect_running_procs() -> list[dict]:
    """Get active processes from process_registry."""
    try:
        from bot import process_registry
        return process_registry.list_processes()
    except Exception as exc:
        logger.debug("Failed to read process_registry: %s", exc)
        return []


def _collect_log_tails() -> dict[str, str]:
    """Tail the last N lines of log files for active processes."""
    try:
        from bot import process_registry
        procs = process_registry.list_processes()
    except Exception:
        return {}

    tails: dict[str, str] = {}
    for proc in procs:
        log_path = proc.get("log_path")
        alias = proc.get("alias", "unknown")
        if not log_path:
            continue
        try:
            p = Path(log_path)
            if not p.exists():
                continue
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = "\n".join(lines[-_MAX_LOG_TAIL_LINES:])
            tails[alias] = tail
        except Exception as exc:
            logger.debug("Failed to tail log for %s: %s", alias, exc)

    return tails


# ── Formatters ────────────────────────────────────────────────────

def _format_recent_files(data: dict[str, list[str]]) -> str:
    if not data:
        return "No recently modified files."
    lines = ["Recently modified files (last 48h):"]
    for root, files in data.items():
        lines.append(f"  [{root}]")
        for f in files[:10]:
            lines.append(f"    {f}")
        if len(files) > 10:
            lines.append(f"    ... and {len(files) - 10} more")
    return "\n".join(lines)


def _format_git_statuses(data: dict[str, dict[str, str]]) -> str:
    if not data:
        return "No git repositories found."
    lines = ["Git status:"]
    for repo, info in data.items():
        lines.append(f"  [{repo}]")
        lines.append(f"    log: {info['log']}")
        lines.append(f"    status: {info['status']}")
    return "\n".join(lines)


def _format_running_procs(procs: list[dict]) -> str:
    if not procs:
        return "No active processes."
    lines = ["Active processes:"]
    for p in procs:
        rc = p.get("returncode")
        state = "running" if rc is None else f"exited({rc})"
        runtime = p.get("runtime_seconds", 0)
        lines.append(f"  PID {p.get('pid')}: {p.get('alias')} — {state}, {runtime:.0f}s")
    return "\n".join(lines)


def _format_log_tails(tails: dict[str, str]) -> str:
    if not tails:
        return ""
    lines = ["Recent log tails:"]
    for alias, tail in tails.items():
        lines.append(f"  [{alias}]")
        for line in tail.splitlines()[-10:]:
            lines.append(f"    {line}")
    return "\n".join(lines)


# ── GPT summarisation ─────────────────────────────────────────────

async def _summarise_snapshot(snapshot: dict[str, Any]) -> str:
    """Call GPT-4o Mini to produce a concise context summary.

    Falls back to the raw formatted snapshot if GPT is unavailable.
    """
    raw_text = "\n\n".join(filter(None, [
        snapshot.get("active_projects", ""),
        snapshot.get("recent_files", ""),
        snapshot.get("running_procs", ""),
        snapshot.get("screen_summary", ""),
    ]))

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return raw_text

    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)

        response = await client.chat.completions.create(
            model=_GPT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are summarising a developer's machine state for use as context "
                        "in a Telegram bot assistant. Write 3-5 sentences covering: what projects "
                        "are actively being worked on, what is currently running, and any notable "
                        "recent changes. Be concise and factual. No bullet points."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Current machine state:\n\n{raw_text}\n\nSummarise.",
                },
            ],
            max_tokens=250,
            temperature=0.2,
        )

        return response.choices[0].message.content.strip()

    except Exception as exc:
        logger.warning("Observer GPT summarisation failed, using raw snapshot: %s", exc)
        return raw_text
