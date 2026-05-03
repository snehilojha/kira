"""Application management command handlers: list_apps, open, close_apps."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import psutil

from telegram import Update
from telegram.ext import ContextTypes

from bot.auth import require_auth

logger = logging.getLogger(__name__)

_SYSTEM_NAMES = {
    'explorer.exe', 'winlogon.exe', 'csrss.exe', 'lsass.exe',
    'services.exe', 'svchost.exe', 'system', 'idle', 'smss.exe',
    'dwm.exe', 'conhost.exe', 'spoolsv.exe', 'taskmgr.exe',
}

_SYSTEM_PATHS = {
    'c:\\windows\\system32\\',
    'c:\\windows\\syswow64\\',
    'c:\\windows\\',
    'c:\\program files\\windows defender\\',
    'c:\\program files (x86)\\windows defender\\',
}


def _is_system_process(process_name: str, executable_path: str) -> bool:
    if process_name in _SYSTEM_NAMES:
        return True
    if executable_path:
        for sys_path in _SYSTEM_PATHS:
            if executable_path.startswith(sys_path):
                return True
    return False


def _normalize_app_name(raw_name: str) -> str | None:
    """Return a sanitized app name, or None if the value looks like a path."""
    cleaned = raw_name.strip().strip('"').strip("'").strip()
    if not cleaned:
        return None
    if any(sep in cleaned for sep in ("\\", "/", ":")):
        return None
    return cleaned


def _open_app_by_name(app_name: str) -> str:
    normalized = _normalize_app_name(app_name)
    if normalized is None:
        return "Usage: /open <app name>\nExample: /open notepad"
    try:
        subprocess.Popen(
            ["cmd", "/c", "start", "", normalized],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("open: launched app by name %r", normalized)
        return f"✅ Opening {normalized}..."
    except Exception as exc:
        logger.error("open: failed to launch %r: %s", normalized, exc, exc_info=True)
        return f"❌ Failed to open {normalized}: {exc}"


@require_auth
async def handle_list_apps(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/list_apps — list running installed applications."""
    apps: dict[str, list[dict]] = {}

    for proc in psutil.process_iter(['pid', 'name', 'exe']):
        proc_name = proc.info['name'].lower()
        proc_exe = proc.info.get('exe', '') or ''

        if _is_system_process(proc_name, proc_exe.lower()):
            continue

        app_name = Path(proc_exe).stem.lower() if proc_exe else proc_name
        apps.setdefault(app_name, []).append({'pid': proc.info['pid'], 'name': proc.info['name']})

    lines = ["📋 Running applications:\n"]
    for app_name, processes in sorted(apps.items()):
        if len(processes) == 1:
            lines.append(f"• {app_name} (PID {processes[0]['pid']})")
        else:
            pids = [p['pid'] for p in processes]
            lines.append(f"• {app_name} ({len(processes)} instances: {', '.join(map(str, pids))})")

    if not apps:
        lines.append("No user applications running.")

    await update.message.reply_text("\n".join(lines)[:4000])


@require_auth
async def handle_open(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/open <app> — open an application by name only."""
    if not context.args:
        await update.message.reply_text("Usage: /open <app name>\nExample: /open notepad")
        return
    app_name = " ".join(context.args)
    result = _open_app_by_name(app_name)
    await update.message.reply_text(result)


@require_auth
async def handle_close_apps(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/close_apps <app1> [app2]... — close specified applications."""
    if not context.args:
        await update.message.reply_text("Usage: /close_apps <app1> [app2]...")
        return

    apps_to_close = context.args
    closed_count = 0
    failed_count = 0
    lines = ["🔄 Closing applications:\n"]

    for app_name in apps_to_close:
        try:
            matching_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'exe']):
                proc_name = proc.info['name'].lower()
                proc_exe = (proc.info.get('exe', '') or '').lower()

                if (app_name.lower() in proc_name or app_name.lower() in proc_exe):
                    if _is_system_process(proc_name, proc_exe):
                        continue
                    matching_processes.append(proc)

            if not matching_processes:
                lines.append(f"❌ {app_name} - no running instances found")
                failed_count += 1
                continue

            for proc in matching_processes:
                proc.terminate()
                lines.append(f"✅ {app_name} - closed {proc.info['name']} (PID {proc.pid})")
                closed_count += 1

        except psutil.NoSuchProcess:
            lines.append(f"⚠️ {app_name} - process already terminated")
        except psutil.AccessDenied:
            lines.append(f"❌ {app_name} - access denied")
            failed_count += 1
        except Exception as exc:
            lines.append(f"❌ {app_name} - error: {exc}")
            failed_count += 1

    lines.append(f"\n📊 Summary: {closed_count} closed, {failed_count} failed")
    await update.message.reply_text("\n".join(lines))
