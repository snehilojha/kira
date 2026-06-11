"""Execution of a parsed local command (app control, desktop control, status)."""

from __future__ import annotations

import asyncio
import os

from bot import app_control
from bot import desktop_control
from bot.voice_runtime import session
from bot.voice_runtime.models import ConfirmCallback, LocalVoiceResult, ParsedCommand
from bot.voice_runtime.util import (
    _format_command,
    _format_process_status,
    _format_sysinfo,
    _from_action_result,
)
from bot.voice_runtime.windows_focus import _restore_foreground

# Raw mouse/keyboard commands guarded by the desktop-control safety gate.
_DESKTOP_COMMANDS = {
    "/mouse_move", "/click", "/scroll", "/type", "/press", "/hotkey", "/copy", "/paste",
}


async def execute_command(
    parsed: ParsedCommand,
    *,
    confirm: ConfirmCallback | None = None,
    config: app_control.AppsConfig | None = None,
) -> LocalVoiceResult:
    """Execute one parsed local command."""
    if parsed.risky:
        if confirm is None or not await confirm(_format_command(parsed)):
            return LocalVoiceResult(
                ok=False,
                message=f"Confirmation denied for {_format_command(parsed)}.",
                spoken="I need confirmation before doing that.",
            )

    apps_config = config or app_control.load_apps_config()
    command = parsed.command.strip().lower()

    if command == "/mode_run":
        result = await asyncio.to_thread(app_control.run_mode, " ".join(parsed.args), apps_config)
        return _from_action_result(result)

    if command == "/open":
        result = await asyncio.to_thread(app_control.open_app, " ".join(parsed.args), apps_config)
        return _from_action_result(result)

    if command == "/close_apps":
        result = await asyncio.to_thread(app_control.close_apps, parsed.args, apps_config)
        return _from_action_result(result)

    if command == "/status":
        message = _format_process_status()
        return LocalVoiceResult(ok=True, message=message, spoken="Status is ready.")

    if command == "/sysinfo":
        message = await asyncio.to_thread(_format_sysinfo)
        return LocalVoiceResult(ok=True, message=message, spoken="System info is ready.")

    if command == "/desktop_arm":
        arm_seconds = float(os.environ.get("KIRA_DESKTOP_ARM_SECONDS", "5"))
        session.DESKTOP_ARM_STATE.arm(arm_seconds)
        return LocalVoiceResult(
            ok=True,
            message=f"Desktop control armed for {arm_seconds:.0f} seconds.",
            spoken="Desktop control armed.",
        )

    # Raw desktop control is gated: reject mouse/keyboard commands unless the
    # user armed the gate (said "arm desktop control") within the window.
    if command in _DESKTOP_COMMANDS and not session.DESKTOP_ARM_STATE.is_armed():
        return LocalVoiceResult(
            ok=False,
            message=f"Desktop control safety gate blocked {parsed.command}. Say 'arm desktop control' first.",
            spoken="Desktop control is locked. Say arm desktop control first.",
        )

    # Restore focus to the user's window before sending input so keystrokes/
    # scroll/clicks land on the right app instead of the Kira terminal.
    await asyncio.to_thread(_restore_foreground, session.get_last_user_hwnd())
    desktop_result = await asyncio.to_thread(desktop_control.execute_command, parsed.command, parsed.args)
    if desktop_result is not None:
        return _from_action_result(desktop_result)

    return LocalVoiceResult(
        ok=False,
        message=f"Command {parsed.command} is not supported by local voice yet.",
        spoken="That command is not supported locally yet.",
    )
