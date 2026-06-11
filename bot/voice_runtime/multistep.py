"""Decomposition and execution of multi-step voice commands."""

from __future__ import annotations

import asyncio
import json
import re

from bot import app_control
from bot import provider
from bot.voice_runtime import routing
from bot.voice_runtime import session
from bot.voice_runtime.brain_fallback import _handle_with_brain
from bot.voice_runtime.executor import execute_command
from bot.voice_runtime.models import LocalVoiceResult, ParsedCommand
from bot.voice_runtime.parsing import is_risky_command
from bot.voice_runtime.windows_focus import _find_foreground_after_open


async def _handle_multistep(transcript: str) -> LocalVoiceResult:
    """Decompose a multi-step voice command into single commands and execute them."""
    config = app_control.load_apps_config()
    app_names = ", ".join(sorted(config.apps)) or "(none configured)"
    history = routing._history_context()

    system_prompt = (
        "You decompose a multi-step PC voice command into an ordered list of single Kira commands. "
        "Return ONLY a JSON array of command objects: [{\"command\": string, \"args\": [strings]}, ...]\n\n"
        "Available commands:\n"
        "- /open <app>\n"
        "- /close_apps <app>\n"
        "- /type <text>\n"
        "- /press <key>  (e.g. enter, space, playpause)\n"
        "- /hotkey <key1> <key2>\n"
        "- /click <button> <count>\n"
        "- /scroll <amount>\n"
        "- /wait  (inserts a short pause between steps)\n\n"
        f"Configured apps: {app_names}\n\n"
        "IMPORTANT rules:\n"
        "- Always insert a /wait after /open to let the app focus before typing.\n"
        "- For any search in a browser, always open a new tab first with /hotkey ctrl t, then type the URL.\n"
        "- This ensures searches never interfere with the current page.\n\n"
        "Example 1: 'open chrome and search youtube for lo-fi music' ->\n"
        "[{\"command\": \"/open\", \"args\": [\"chrome\"]},\n"
        " {\"command\": \"/wait\", \"args\": []},\n"
        " {\"command\": \"/hotkey\", \"args\": [\"ctrl\", \"t\"]},\n"
        " {\"command\": \"/wait\", \"args\": []},\n"
        " {\"command\": \"/type\", \"args\": [\"youtube.com/results?search_query=lo-fi+music\"]},\n"
        " {\"command\": \"/press\", \"args\": [\"enter\"]}]\n\n"
        "Example 2: 'play the first video' (on a YouTube search results page) ->\n"
        "[{\"command\": \"/press\", \"args\": [\"tab\"]},\n"
        " {\"command\": \"/wait\", \"args\": []},\n"
        " {\"command\": \"/press\", \"args\": [\"enter\"]}]\n\n"
        + (f"{history}\n\n" if history else "")
    )

    response = await provider.create_chat_completion(
        role="fast",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript},
        ],
        temperature=0,
        max_tokens=400,
    )
    raw = (response.choices[0].message.content or "").strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw).strip()

    try:
        steps = json.loads(raw)
        if not isinstance(steps, list):
            raise ValueError("not a list")
    except (json.JSONDecodeError, ValueError):
        return await _handle_with_brain(transcript)

    # If steps mention browser actions but no /open, prepend a focus of chrome
    commands_in_steps = [str(s.get("command", "")).strip().lower() for s in steps]
    has_open = any(c == "/open" for c in commands_in_steps)
    has_browser_action = any(
        c in ("/hotkey", "/type") for c in commands_in_steps
    )
    if not has_open and has_browser_action:
        steps = [{"command": "/open", "args": ["chrome"]}] + steps

    # The user explicitly issued this multi-step command, so arm the desktop
    # safety gate for its duration — the individual /type, /press, /hotkey
    # steps would otherwise be blocked by the gate in execute_command.
    session.DESKTOP_ARM_STATE.arm(60)
    try:
        messages = []
        for step in steps:
            command = str(step.get("command", "")).strip().lower()
            args = [str(a) for a in step.get("args", [])]

            if command == "/wait":
                await asyncio.sleep(4.0)
                continue

            parsed = ParsedCommand(command=command, args=args, source="multistep", risky=is_risky_command(command, args))
            result = await execute_command(parsed, config=config)
            messages.append(result.message)

            if command == "/open":
                await asyncio.sleep(4.0)
                # After opening an app, update _last_user_hwnd to the new window
                # so subsequent steps target the freshly opened app.
                new_hwnd = _find_foreground_after_open(args[0] if args else "")
                if new_hwnd:
                    session.set_last_user_hwnd(new_hwnd)
            else:
                await asyncio.sleep(0.5)
    finally:
        session.DESKTOP_ARM_STATE.disarm()

    spoken = "Done." if messages else "I could not complete that."
    return LocalVoiceResult(ok=True, message="\n".join(messages), spoken=spoken)
