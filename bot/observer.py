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
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from bot import provider

logger = logging.getLogger(__name__)

_OBSERVER_INTERVAL = int(os.environ.get("OBSERVER_INTERVAL", "40000"))
_RECENT_FILE_HOURS = 48
_MAX_FILES_PER_ROOT = 20
_MAX_LOG_TAIL_LINES = 30
_GPT_MODEL = "gpt-4o-mini"

# Seconds with no new log output before a process is considered potentially stalled.
_STALL_THRESHOLD_SECONDS = float(os.environ.get("KIRA_STALL_THRESHOLD_SECONDS", "300"))
_FAST_LOOP_INTERVAL = 30  # seconds
_HIGH_CPU_THRESHOLD = 90.0  # percent
_HIGH_CPU_DURATION = 600    # seconds sustained before alerting

# Module-level cached context string and raw snapshot.
_CURRENT_CONTEXT_SUMMARY: str = ""
_raw_snapshot_cache: dict = {}

# Fast-loop state — tracks what has already been notified to avoid repeat pings
_notified: set[str] = set()
_high_cpu_since: dict[int, float] = {}
_low_cpu_since: dict[int, float] = {}
_LOW_CPU_STDIN_DURATION = int(os.environ.get("KIRA_STDIN_DETECT_SECONDS", "120"))

# Absence log — records escalation events that fired during autonomous mode.
# Cleared when the user returns. Read by mode.py for the return summary.
_absence_log: list[str] = []

# Maps Telegram message_id → alert description so handle_text can attach
# context when the user replies to an escalation message.
_escalation_context: dict[int, str] = {}
_ESCALATION_CONTEXT_MAX = 20


def get_escalation_context(message_id: int) -> str | None:
    """Return the alert description for a Telegram message_id, or None."""
    return _escalation_context.get(message_id)


def _store_escalation_context(message_id: int, description: str) -> None:
    if len(_escalation_context) >= _ESCALATION_CONTEXT_MAX:
        oldest = next(iter(_escalation_context))
        del _escalation_context[oldest]
    _escalation_context[message_id] = description


def get_absence_log() -> list[str]:
    """Return escalation events that fired during the current autonomous period."""
    return list(_absence_log)


def clear_absence_log() -> None:
    """Clear the absence log. Called by mode.py when user returns."""
    _absence_log.clear()


def get_current_context() -> str:
    """Return the latest GPT-summarised machine context, or empty string if not ready."""
    return _CURRENT_CONTEXT_SUMMARY


# ── Active project context cache ──────────────────────────────────

# Keyed by project folder path string → summary string
_project_context_cache: dict[str, str] = {}
# Last window title we parsed — avoid recomputing on every brain call
_last_foreground_title: str = ""
_active_project_context: str = ""
_last_active_folder: str = ""

# Optional callback fired when the user switches to a different project.
# Signature: (project_summary: str) -> None  (sync, called via ensure_future)
_project_switch_callback: "Callable[[str], Any] | None" = None


def get_active_project_context() -> str:
    """Return the pre-emptive context for the project currently in focus."""
    return _active_project_context


def register_project_switch_callback(fn: "Callable[[str], Any]") -> None:
    """Register a callback invoked when the active project changes.

    ``fn`` receives the project summary string. It may be a coroutine function;
    if so it is scheduled with ``asyncio.ensure_future``.
    """
    global _project_switch_callback
    _project_switch_callback = fn


def get_pending_triggers() -> list[str]:
    """Return screen-vision trigger types that are currently active.

    Inspects the cached raw snapshot for ambiguity conditions. Called by the
    observer after each cycle to decide whether to fire screen_vision.

    Returns:
        List of TriggerType strings (may be empty).
    """
    triggers: list[str] = []
    snapshot = _raw_snapshot_cache

    if not snapshot:
        return triggers

    dialog = snapshot.get("dialog_detected")
    if dialog:
        triggers.append("dialog_appeared")

    stalled = snapshot.get("stalled_processes", [])
    for proc in stalled:
        label = proc.get("alias", "")
        if "cursor" in label.lower():
            triggers.append("cursor_ai_stalled")
        else:
            triggers.append("process_frozen")

    stdin_blocked = snapshot.get("stdin_blocked_processes", [])
    if stdin_blocked:
        triggers.append("stdin_silent")

    return triggers


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


async def start_fast_loop() -> None:
    """Fast escalation loop — checks urgent conditions every 30 seconds."""
    logger.info("Observer fast loop started (interval=%ds)", _FAST_LOOP_INTERVAL)
    while True:
        try:
            await _run_fast_cycle()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Observer fast cycle failed")
        await asyncio.sleep(_FAST_LOOP_INTERVAL)


async def _run_fast_cycle() -> None:
    """Check urgent conditions and ping Telegram if something needs attention."""
    from bot import mode as kira_mode
    from bot import notifier

    # Always update low-cpu timers regardless of autonomous state,
    # so the stdin detection clock starts from when the process went idle,
    # not from when the user walked away.
    await asyncio.to_thread(_update_process_timers)

    # Keep active project context fresh on every fast tick
    title = await asyncio.to_thread(_collect_foreground_window)
    if title:
        await _refresh_active_project_context(title)

    autonomous = kira_mode.is_autonomous()
    logger.debug("Fast cycle: autonomous=%s", autonomous)
    if not autonomous:
        # User is back — clear notifications so they can re-fire next absence.
        _notified.clear()
        return

    # Remove stale notified keys for pids that no longer exist.
    live_pids = {p["pid"] for p in _collect_watched_procs()}
    stale = {k for k in _notified if "_" in k and k.split("_")[1].isdigit()
             and int(k.split("_")[1]) not in live_pids}
    _notified.difference_update(stale)

    events = await asyncio.to_thread(_collect_fast_snapshot)
    logger.debug("Fast cycle: %d events found", len(events))
    for event_key, message in events:
        if event_key in _notified:
            continue
        _notified.add(event_key)
        logger.info("Escalating to Telegram: %s", event_key)
        _absence_log.append(message)
        try:
            png = await asyncio.to_thread(_get_screenshot_bytes)
            caption = f"Kira: {message}"
            if png:
                msg_id = await notifier.send_photo(caption, png)
            else:
                msg_id = await notifier.send(caption)
            if msg_id:
                _store_escalation_context(msg_id, message)
        except Exception as exc:
            logger.warning("Escalation notify failed: %s", exc)


_WATCHED_PROCESS_NAMES = {"python.exe", "pythonw.exe", "code.exe"}
_MIN_RUNTIME_SECONDS = 60  # ignore quick one-off processes


def _collect_watched_procs() -> list[dict]:
    """Return user-owned Python and VSCode processes running longer than 60s."""
    try:
        import psutil
    except ImportError:
        return []

    procs = []
    current_user = None
    try:
        current_user = psutil.Process().username()
    except Exception:
        pass

    for ps in psutil.process_iter(["pid", "name", "username", "create_time", "cmdline", "status"]):
        try:
            info = ps.info
            if info["name"] not in _WATCHED_PROCESS_NAMES:
                continue
            if current_user and info.get("username") != current_user:
                continue
            runtime = time.time() - (info["create_time"] or time.time())
            if runtime < _MIN_RUNTIME_SECONDS:
                continue
            cmdline = " ".join(info.get("cmdline") or [])
            procs.append({
                "pid": info["pid"],
                "name": info["name"],
                "cmdline": cmdline[:120],
                "runtime_seconds": runtime,
                "status": info.get("status", ""),
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    return procs


def _update_process_timers() -> None:
    """Update low/high CPU timers for all watched processes. Called every cycle."""
    import psutil

    watched = _collect_watched_procs()
    live_pids = {p["pid"] for p in watched}

    for pid in list(_high_cpu_since):
        if pid not in live_pids:
            _high_cpu_since.pop(pid, None)
    for pid in list(_low_cpu_since):
        if pid not in live_pids:
            _low_cpu_since.pop(pid, None)

    for proc in watched:
        pid = proc["pid"]
        runtime = proc["runtime_seconds"]
        try:
            ps = psutil.Process(pid)
            cpu = ps.cpu_percent(interval=0.5)
            if cpu < 1.0 and runtime > 60:
                _low_cpu_since.setdefault(pid, time.monotonic())
            else:
                _low_cpu_since.pop(pid, None)

            if cpu >= _HIGH_CPU_THRESHOLD:
                _high_cpu_since.setdefault(pid, time.monotonic())
            else:
                _high_cpu_since.pop(pid, None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            _low_cpu_since.pop(pid, None)
            _high_cpu_since.pop(pid, None)


def _collect_fast_snapshot() -> list[tuple[str, str]]:
    """Collect urgent events to ping about. Timers already updated by _update_process_timers."""
    watched = _collect_watched_procs()
    events: list[tuple[str, str]] = []
    now = time.monotonic()

    for proc in watched:
        pid = proc["pid"]
        label = proc["cmdline"] or proc["name"]

        # Stdin blocked
        since = _low_cpu_since.get(pid)
        if since and now - since >= _LOW_CPU_STDIN_DURATION:
            events.append((f"stdin_{pid}", f"A script may be waiting for input: {label}"))

        # High CPU sustained
        since = _high_cpu_since.get(pid)
        if since and now - since >= _HIGH_CPU_DURATION:
            events.append((f"highcpu_{pid}", f"A script has been pegging CPU for over {_HIGH_CPU_DURATION // 60} minutes: {label}"))

    # Dialog / UAC
    dialog = _collect_dialog_state()
    if dialog:
        events.append(("dialog", f"A dialog appeared on screen: {dialog}"))

    return events


def _get_screenshot_bytes() -> bytes:
    from bot.screen_vision import take_screenshot_png
    return take_screenshot_png()


async def _run_cycle() -> None:
    """Collect a snapshot, persist it, update the cached summary, dispatch triggers."""
    global _CURRENT_CONTEXT_SUMMARY, _raw_snapshot_cache

    snapshot = await asyncio.to_thread(_collect_snapshot)
    _raw_snapshot_cache = snapshot

    try:
        from bot import db
        await db.save_observation(snapshot)
    except Exception as exc:
        logger.warning("Failed to persist observation: %s", exc)

    summary = await _summarise_snapshot(snapshot)
    _CURRENT_CONTEXT_SUMMARY = summary
    logger.debug("Observer context updated (%d chars)", len(summary))

    # Refresh active project context from current foreground window
    title = snapshot.get("foreground_window", "")
    if title:
        await _refresh_active_project_context(title)

    await _dispatch_triggers()
    await _maybe_proactive_notify(summary)


# ── Snapshot collection ───────────────────────────────────────────

def _collect_snapshot() -> dict[str, Any]:
    """Collect all machine-state data synchronously (runs in a thread)."""
    project_roots = _get_project_roots()

    recent_files = _collect_recent_files(project_roots)
    git_statuses = _collect_git_statuses(project_roots)
    running_procs = _collect_running_procs()
    log_tails = _collect_log_tails()
    foreground_window = _collect_foreground_window()
    dialog_detected = _collect_dialog_state()
    stalled_processes = _check_process_stall_state(running_procs)
    stdin_blocked = _check_stdin_blocked(running_procs)

    return {
        "observed_at": datetime.now().isoformat(timespec="seconds"),
        "active_projects": _format_git_statuses(git_statuses),
        "recent_files": _format_recent_files(recent_files),
        "git_status": _format_git_statuses(git_statuses),
        "running_procs": _format_running_procs(running_procs),
        "screen_summary": _format_log_tails(log_tails),
        # V1.5 additions
        "foreground_window": foreground_window,
        "dialog_detected": dialog_detected,
        "stalled_processes": stalled_processes,
        "stdin_blocked_processes": stdin_blocked,
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


# ── V1.5 awareness collectors ─────────────────────────────────────

def _collect_foreground_window() -> str:
    """Return the title of the currently focused window (Windows only).

    Uses ctypes ``GetForegroundWindow`` + ``GetWindowTextW``. Returns empty
    string on non-Windows or if the call fails.
    """
    import platform
    import ctypes

    if platform.system() != "Windows":
        return ""

    try:
        hwnd = ctypes.windll.user32.GetForegroundWindow()  # type: ignore[attr-defined]
        if not hwnd:
            return ""
        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)  # type: ignore[attr-defined]
        if length == 0:
            return ""
        buf = ctypes.create_unicode_buffer(length + 1)
        ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)  # type: ignore[attr-defined]
        return buf.value.strip()
    except Exception as exc:
        logger.debug("GetForegroundWindow failed: %s", exc)
        return ""


def _collect_dialog_state() -> str | None:
    """Detect the presence of a modal dialog or UAC prompt (Windows only).

    Enumerates top-level windows looking for the Windows dialog class
    ``#32770`` (common for message boxes and UAC prompts) and other known
    modal class names. Returns a short description or None.
    """
    import platform
    import ctypes

    if platform.system() != "Windows":
        return None

    found_dialogs: list[str] = []

    _DIALOG_CLASSES = {"#32770", "ApplicationFrameWindow"}

    try:
        def _enum_callback(hwnd, _):
            if not ctypes.windll.user32.IsWindowVisible(hwnd):  # type: ignore[attr-defined]
                return True
            buf = ctypes.create_unicode_buffer(256)
            ctypes.windll.user32.GetClassNameW(hwnd, buf, 256)  # type: ignore[attr-defined]
            cls = buf.value.strip()
            if cls in _DIALOG_CLASSES:
                title_buf = ctypes.create_unicode_buffer(256)
                ctypes.windll.user32.GetWindowTextW(hwnd, title_buf, 256)  # type: ignore[attr-defined]
                title = title_buf.value.strip()
                if title:
                    found_dialogs.append(f"{cls}: {title}")
            return True

        _EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)  # type: ignore[attr-defined]
        ctypes.windll.user32.EnumWindows(_EnumWindowsProc(_enum_callback), 0)  # type: ignore[attr-defined]
    except Exception as exc:
        logger.debug("EnumWindows failed: %s", exc)
        return None

    if found_dialogs:
        return "; ".join(found_dialogs)
    return None


def _check_process_stall_state(running_procs: list[dict]) -> list[dict]:
    """Return processes that appear stalled (running too long with no log output).

    A process is considered stalled if:
    - It has been running for more than _STALL_THRESHOLD_SECONDS, AND
    - Its log file has not been modified in the last _STALL_THRESHOLD_SECONDS.

    Args:
        running_procs: List of process dicts from process_registry.

    Returns:
        List of stalled process dicts (subset of running_procs).
    """
    import time

    stalled: list[dict] = []
    now = time.time()

    for proc in running_procs:
        # Skip processes that have already exited.
        if proc.get("returncode") is not None:
            continue

        runtime = proc.get("runtime_seconds", 0) or 0
        if runtime < _STALL_THRESHOLD_SECONDS:
            continue

        log_path = proc.get("log_path")
        if not log_path:
            continue

        try:
            mtime = Path(log_path).stat().st_mtime
            log_age = now - mtime
            if log_age >= _STALL_THRESHOLD_SECONDS:
                stalled.append({
                    "alias": proc.get("alias", "unknown"),
                    "pid": proc.get("pid"),
                    "runtime_seconds": runtime,
                    "log_idle_seconds": log_age,
                })
        except OSError:
            pass

    return stalled


def _check_stdin_blocked(running_procs: list[dict]) -> list[dict]:
    """Detect Kira-launched processes that may be blocked waiting for stdin.

    Uses psutil to check whether a process is in the 'stopped' state or
    has an open stdin pipe with no recent stdout. This is a best-effort
    heuristic — false positives are tolerable since screen_vision will
    confirm before notifying.

    Args:
        running_procs: List of process dicts from process_registry.

    Returns:
        List of possibly-blocked process dicts.
    """
    blocked: list[dict] = []

    try:
        import psutil
    except ImportError:
        return blocked

    for proc in running_procs:
        if proc.get("returncode") is not None:
            continue
        pid = proc.get("pid")
        if not pid:
            continue
        try:
            ps = psutil.Process(pid)
            status = ps.status()
            if status in (psutil.STATUS_STOPPED, "stopped"):
                blocked.append({
                    "alias": proc.get("alias", "unknown"),
                    "pid": pid,
                    "status": status,
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    return blocked


async def _dispatch_triggers() -> None:
    """Fire screen-vision checks for any active ambiguity triggers."""
    triggers = get_pending_triggers()
    if not triggers:
        return

    from bot import screen_vision

    snapshot = _raw_snapshot_cache

    for trigger in triggers:
        process_label = ""
        if trigger in ("process_frozen", "cursor_ai_stalled"):
            stalled = snapshot.get("stalled_processes", [])
            if stalled:
                process_label = stalled[0].get("alias", "")
        elif trigger == "stdin_silent":
            blocked = snapshot.get("stdin_blocked_processes", [])
            if blocked:
                process_label = blocked[0].get("alias", "")
        elif trigger == "dialog_appeared":
            process_label = snapshot.get("dialog_detected", "") or ""

        try:
            await screen_vision.notify_if_actionable(trigger, process_label)
        except Exception as exc:
            logger.warning("Screen vision dispatch failed for %s: %s", trigger, exc)


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


# ── Active project detection ──────────────────────────────────────

import re as _re

# VSCode window title patterns (covers most common variants):
#   "file.py — folder — Visual Studio Code"
#   "● file.py — folder — Visual Studio Code"
#   "folder — Visual Studio Code"
_VSCODE_TITLE_RE = _re.compile(
    r"^●?\s*(?:.+?\s+—\s+)?(.+?)\s+—\s+Visual Studio Code",
    _re.IGNORECASE,
)


def _parse_project_from_title(title: str) -> Path | None:
    """Extract a matching project root from a window title."""
    m = _VSCODE_TITLE_RE.match(title)
    if not m:
        return None

    folder_hint = m.group(1).strip()
    roots = _get_project_roots()

    # Try exact subfolder match first, then name-contains match
    for root in roots:
        try:
            for candidate in [root, *root.iterdir()]:
                if candidate.is_dir() and candidate.name.lower() == folder_hint.lower():
                    return candidate
        except OSError:
            pass

    # Fallback: check if any root name appears in the hint
    for root in roots:
        if root.name.lower() in folder_hint.lower():
            return root

    return None


def _build_project_summary(folder: Path) -> str:
    """Build a compact context string for a project folder."""
    lines: list[str] = [f"Active project: {folder.name} ({folder})"]

    # Recent git log
    try:
        log = subprocess.run(
            ["git", "-C", str(folder), "log", "--oneline", "-5"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "-C", str(folder), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if branch:
            lines.append(f"Branch: {branch}")
        if log:
            lines.append(f"Recent commits:\n{log}")
    except Exception:
        pass

    # Recent modified files (last 6h)
    try:
        cutoff = datetime.now() - timedelta(hours=6)
        _TRACK_EXTS = {".py", ".ipynb", ".yaml", ".yml", ".toml", ".json", ".md"}
        _SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules"}
        recent: list[str] = []
        for p in folder.rglob("*"):
            if not p.is_file():
                continue
            if any(part in _SKIP_DIRS for part in p.parts):
                continue
            if p.suffix.lower() not in _TRACK_EXTS:
                continue
            try:
                if datetime.fromtimestamp(p.stat().st_mtime) >= cutoff:
                    recent.append(str(p.relative_to(folder)))
            except OSError:
                continue
            if len(recent) >= 10:
                break
        if recent:
            lines.append("Recently modified:\n" + "\n".join(f"  {f}" for f in recent))
    except Exception:
        pass

    # context.md if present
    ctx_file = folder / "context.md"
    if ctx_file.exists():
        try:
            text = ctx_file.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                lines.append(f"Project context:\n{text[:800]}")
        except OSError:
            pass

    return "\n".join(lines)


async def _refresh_active_project_context(title: str) -> None:
    """Detect active project from window title and update the context cache."""
    global _last_foreground_title, _active_project_context, _last_active_folder

    if title == _last_foreground_title:
        return
    _last_foreground_title = title

    folder = await asyncio.to_thread(_parse_project_from_title, title)
    if folder is None:
        _active_project_context = ""
        return

    folder_key = str(folder)
    is_new_project = folder_key != _last_active_folder

    if folder_key in _project_context_cache:
        _active_project_context = _project_context_cache[folder_key]
    else:
        summary = await asyncio.to_thread(_build_project_summary, folder)
        _project_context_cache[folder_key] = summary
        _active_project_context = summary
        logger.info("Active project context loaded: %s", folder.name)

    if is_new_project:
        _last_active_folder = folder_key
        if _project_switch_callback is not None:
            try:
                result = _project_switch_callback(_active_project_context)
                if asyncio.iscoroutine(result):
                    asyncio.ensure_future(result)
            except Exception as exc:
                logger.debug("Project switch callback failed: %s", exc)


# ── Proactive notification ────────────────────────────────────────

_prev_summary: str = ""


async def _maybe_proactive_notify(current_summary: str) -> None:
    """Ask GPT if the new observer summary is notable enough to ping the user.

    Only fires during autonomous mode. Compares against the previous summary
    so repeated identical state doesn't keep pinging.
    """
    global _prev_summary

    from bot import mode as kira_mode

    if not kira_mode.is_autonomous():
        _prev_summary = current_summary
        return

    if not current_summary or current_summary == _prev_summary:
        return

    prev = _prev_summary
    _prev_summary = current_summary

    if not prev:
        return  # First cycle — no baseline to compare against

    try:
        response = await provider.create_chat_completion(
            role="fast",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are deciding whether to interrupt a user who is away from their computer. "
                        "You will be shown a before/after snapshot of their machine state. "
                        "Reply with ONLY 'yes' if something genuinely notable changed that warrants interrupting them "
                        "(e.g. a process crashed, a new unexpected file appeared, a long-running task finished, a git conflict). "
                        "Reply with ONLY 'no' if the change is routine or unimportant. "
                        "No explanation. One word."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Before:\n{prev}\n\nAfter:\n{current_summary}",
                },
            ],
            max_tokens=5,
            temperature=0.0,
        )
        verdict = response.choices[0].message.content.strip().lower()
        logger.debug("Proactive notify verdict: %r", verdict)

        if verdict.startswith("yes"):
            from bot import notifier
            _absence_log.append(f"Observer noticed: {current_summary[:200]}")
            await notifier.send(f"Kira (proactive): {current_summary[:800]}")

    except Exception as exc:
        logger.debug("Proactive notify check failed: %s", exc)


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

    try:
        response = await provider.create_chat_completion(
            role="fast",
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

    except RuntimeError:
        return raw_text
    except Exception as exc:
        logger.warning("Observer GPT summarisation failed, using raw snapshot: %s", exc)
        return raw_text
