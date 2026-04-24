"""Local application and mode control for Kira.

Loads trusted app/mode definitions from ``config/apps.toml`` and provides
small execution helpers shared by Telegram commands and local voice.
"""

from __future__ import annotations

import logging
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil
import toml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "apps.toml"


@dataclass(frozen=True)
class AppDefinition:
    """One configured application Kira can open or close."""

    name: str
    open_command: str
    close_names: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ModeDefinition:
    """One configured multi-app local mode."""

    name: str
    aliases: list[str] = field(default_factory=list)
    open_apps: list[str] = field(default_factory=list)
    close_apps: list[str] = field(default_factory=list)
    say: str = ""


@dataclass(frozen=True)
class AppsConfig:
    """Resolved local app/mode configuration."""

    apps: dict[str, AppDefinition] = field(default_factory=dict)
    modes: dict[str, ModeDefinition] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionResult:
    """User-facing result from one local app/mode action."""

    ok: bool
    message: str
    spoken: str


def load_apps_config(path: Path | None = None) -> AppsConfig:
    """Load local app and mode definitions from TOML."""
    config_path = path or _DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return AppsConfig()

    loaded = toml.load(config_path)
    raw_apps = loaded.get("apps", {})
    raw_modes = loaded.get("modes", {})

    apps: dict[str, AppDefinition] = {}
    if isinstance(raw_apps, dict):
        for app_name, raw in raw_apps.items():
            if not isinstance(raw, dict):
                continue
            key = _normalize_key(app_name)
            open_command = str(raw.get("open", "")).strip()
            close_names = _string_list(raw.get("close", []))
            if key and open_command:
                apps[key] = AppDefinition(
                    name=key,
                    open_command=open_command,
                    close_names=close_names,
                )

    modes: dict[str, ModeDefinition] = {}
    if isinstance(raw_modes, dict):
        for mode_name, raw in raw_modes.items():
            if not isinstance(raw, dict):
                continue
            key = _normalize_key(mode_name)
            if not key:
                continue
            modes[key] = ModeDefinition(
                name=key,
                aliases=_string_list(raw.get("aliases", [])),
                open_apps=[_normalize_key(item) for item in _string_list(raw.get("open", []))],
                close_apps=[_normalize_key(item) for item in _string_list(raw.get("close", []))],
                say=str(raw.get("say", "")).strip(),
            )

    return AppsConfig(apps=apps, modes=modes)


def find_app(name: str, config: AppsConfig | None = None) -> AppDefinition | None:
    """Find an app by configured key or close/open-friendly name."""
    apps_config = config or load_apps_config()
    key = _normalize_key(name)
    if key in apps_config.apps:
        return apps_config.apps[key]

    for app in apps_config.apps.values():
        candidates = {app.name, Path(app.open_command.split()[0]).stem.lower()}
        candidates.update(Path(item).stem.lower() for item in app.close_names)
        if key in candidates:
            return app
    return None


def find_mode(phrase: str, config: AppsConfig | None = None) -> ModeDefinition | None:
    """Find a mode by name or alias."""
    apps_config = config or load_apps_config()
    normalized = _normalize_phrase(phrase)
    key = _normalize_key(phrase)
    if key in apps_config.modes:
        return apps_config.modes[key]

    for mode in apps_config.modes.values():
        if normalized == _normalize_phrase(mode.name):
            return mode
        if any(normalized == _normalize_phrase(alias) for alias in mode.aliases):
            return mode
    return None


def open_app(app_name: str, config: AppsConfig | None = None) -> ActionResult:
    """Open a configured app, falling back to a sanitized app name.

    If the app is already running, focuses its window instead of launching
    a second instance.
    """
    apps_config = config or load_apps_config()
    app = find_app(app_name, apps_config)

    close_terms = app.close_names if app and app.close_names else [app_name]
    running = _matching_processes(close_terms)
    if running:
        focused = _focus_process(running[0])
        label = app.name if app else app_name
        if focused:
            return ActionResult(ok=True, message=f"{label} is already running — focused.", spoken=f"{label} is already open.")
        return ActionResult(ok=True, message=f"{label} is already running.", spoken=f"{label} is already open.")

    if app is not None:
        return _launch_command(app.open_command, app.name)

    normalized = normalize_app_name(app_name)
    if normalized is None:
        return ActionResult(
            ok=False,
            message="Usage: /open <app name>\nExample: /open notepad",
            spoken="I need an app name to open.",
        )
    return _launch_name_only(normalized)


def close_apps(app_names: list[str], config: AppsConfig | None = None) -> ActionResult:
    """Close configured apps or running apps matching the given names."""
    if not app_names:
        return ActionResult(
            ok=False,
            message="Usage: /close_apps <app1> [app2]...",
            spoken="I need at least one app name to close.",
        )

    apps_config = config or load_apps_config()
    closed_count = 0
    failed_count = 0
    lines = ["Closing applications:\n"]

    for app_name in app_names:
        app = find_app(app_name, apps_config)
        close_terms = app.close_names if app and app.close_names else [app_name]
        matched = _matching_processes(close_terms)
        if not matched:
            lines.append(f"- {app_name}: no running instances found")
            failed_count += 1
            continue

        for proc in matched:
            try:
                proc.terminate()
                proc_name = proc.info.get("name") or "process"
                lines.append(f"- {app_name}: closed {proc_name} (PID {proc.pid})")
                closed_count += 1
            except psutil.NoSuchProcess:
                lines.append(f"- {app_name}: process already terminated")
            except psutil.AccessDenied:
                lines.append(f"- {app_name}: access denied")
                failed_count += 1
            except Exception as exc:
                lines.append(f"- {app_name}: error: {exc}")
                failed_count += 1

    lines.append(f"\nSummary: {closed_count} closed, {failed_count} failed")
    ok = closed_count > 0 and failed_count == 0
    spoken = (
        f"Closed {closed_count} app instance{'s' if closed_count != 1 else ''}."
        if closed_count
        else "I did not find any matching apps to close."
    )
    return ActionResult(ok=ok, message="\n".join(lines), spoken=spoken)


def run_mode(mode_name: str, config: AppsConfig | None = None) -> ActionResult:
    """Run a configured local mode."""
    apps_config = config or load_apps_config()
    mode = find_mode(mode_name, apps_config)
    if mode is None:
        return ActionResult(
            ok=False,
            message=f"Unknown mode: {mode_name}",
            spoken=f"I do not know the mode {mode_name}.",
        )

    parts: list[str] = [f"Mode: {mode.name}"]
    ok = True

    if mode.close_apps:
        close_result = close_apps(mode.close_apps, apps_config)
        parts.append(close_result.message)
        ok = ok and close_result.ok

    for app_name in mode.open_apps:
        result = open_app(app_name, apps_config)
        parts.append(result.message)
        ok = ok and result.ok

    spoken = mode.say or f"{mode.name.title()} mode is ready."
    parts.append(spoken)
    return ActionResult(ok=ok, message="\n".join(parts), spoken=spoken)


def normalize_app_name(raw_name: str) -> str | None:
    """Return a sanitized app name, or ``None`` if the value looks like a path."""
    cleaned = raw_name.strip().strip('"').strip("'").strip()
    if not cleaned:
        return None
    if any(sep in cleaned for sep in ("\\", "/", ":")):
        return None
    return cleaned


def is_system_process(process_name: str, executable_path: str) -> bool:
    """Return True for Windows system processes Kira should not terminate."""
    system_names = {
        "explorer.exe",
        "winlogon.exe",
        "csrss.exe",
        "lsass.exe",
        "services.exe",
        "svchost.exe",
        "system",
        "idle",
        "smss.exe",
        "dwm.exe",
        "conhost.exe",
        "spoolsv.exe",
        "taskmgr.exe",
    }
    system_paths = {
        "c:\\windows\\system32\\",
        "c:\\windows\\syswow64\\",
        "c:\\windows\\",
        "c:\\program files\\windows defender\\",
        "c:\\program files (x86)\\windows defender\\",
    }

    if process_name in system_names:
        return True
    return any(executable_path.startswith(path) for path in system_paths)


def _launch_name_only(app_name: str) -> ActionResult:
    try:
        subprocess.Popen(
            ["cmd", "/c", "start", "", app_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("open: launched app by name %r", app_name)
        return ActionResult(
            ok=True,
            message=f"Opening {app_name}...",
            spoken=f"Opening {app_name}.",
        )
    except Exception as exc:
        logger.error("open: failed to launch %r: %s", app_name, exc, exc_info=True)
        return ActionResult(
            ok=False,
            message=f"Failed to open {app_name}: {exc}",
            spoken=f"I could not open {app_name}.",
        )


def _launch_command(command: str, label: str) -> ActionResult:
    try:
        command_parts = shlex.split(command, posix=False)
        subprocess.Popen(
            ["cmd", "/c", "start", "", *command_parts],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("open: launched configured app %r via %r", label, command)
        return ActionResult(ok=True, message=f"Opening {label}...", spoken=f"Opening {label}.")
    except Exception as exc:
        logger.error("open: failed to launch configured app %r: %s", label, exc, exc_info=True)
        return ActionResult(
            ok=False,
            message=f"Failed to open {label}: {exc}",
            spoken=f"I could not open {label}.",
        )


def _matching_processes(terms: list[str]) -> list[psutil.Process]:
    normalized_terms = [_normalize_process_term(term) for term in terms if term.strip()]
    matches = []
    seen: set[int] = set()
    for proc in psutil.process_iter(["pid", "name", "exe"]):
        proc_name = (proc.info.get("name") or "").lower()
        proc_exe = (proc.info.get("exe") or "").lower()
        if is_system_process(proc_name, proc_exe):
            continue
        if any(_process_matches(term, proc_name, proc_exe) for term in normalized_terms):
            if proc.pid not in seen:
                seen.add(proc.pid)
                matches.append(proc)
    return matches


def _process_matches(term: str, proc_name: str, proc_exe: str) -> bool:
    if not term:
        return False
    stem = Path(term).stem.lower()
    return term == proc_name or stem == Path(proc_name).stem.lower() or term in proc_exe


def _normalize_process_term(term: str) -> str:
    return term.strip().strip('"').strip("'").lower()


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _focus_process(proc: psutil.Process) -> bool:
    """Bring the main window of a process to the foreground. Returns True on success."""
    try:
        import win32gui
        import win32con

        pid = proc.pid

        def _callback(hwnd: int, found: list[int]) -> bool:
            if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowThreadProcessId(hwnd)[1] == pid:
                found.append(hwnd)
            return True

        hwnds: list[int] = []
        win32gui.EnumWindows(_callback, hwnds)
        if not hwnds:
            return False
        hwnd = hwnds[0]
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
        return True
    except Exception as exc:
        logger.debug("Could not focus window for PID %d: %s", proc.pid, exc)
        return False


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.strip().lower()).strip("_")


def _normalize_phrase(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())
