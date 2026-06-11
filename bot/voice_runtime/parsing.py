"""Deterministic and LLM-based parsing of a transcript into a ParsedCommand."""

from __future__ import annotations

import json
import re

from bot import app_control
from bot import provider
from bot.voice_runtime.models import ParsedCommand
from bot.voice_runtime.util import _normalize_text


def parse_deterministic(
    transcript: str,
    config: app_control.AppsConfig | None = None,
) -> ParsedCommand | None:
    """Parse cheap local app/mode commands without calling an LLM."""
    text = _normalize_text(transcript)
    if not text:
        return None

    apps_config = config or app_control.load_apps_config()

    # Configured natural-language shortcuts (intents) take priority over the
    # generic open/launch prefix handling, so e.g. "open browser" can map to a
    # specific configured command instead of literally opening "browser".
    intent = app_control.find_intent(text, apps_config)
    if intent is not None:
        return ParsedCommand(command=intent.command, args=list(intent.args), source="intent")

    mode = app_control.find_mode(text, apps_config)
    if mode is not None:
        return ParsedCommand(command="/mode_run", args=[mode.name], source="deterministic")

    for prefix in ("open ", "launch ", "start "):
        if text.startswith(prefix):
            app_name = text[len(prefix):].strip()
            # Only match simple single-word or configured app names.
            # Multi-word phrases with conjunctions ("and", "then") are complex
            # commands and should fall through to the LLM parser.
            if app_name and not any(w in app_name.split() for w in ("and", "then", "after", "search", "go", "navigate")):
                if app_name in (apps_config.apps or {}) or len(app_name.split()) <= 3:
                    return ParsedCommand(command="/open", args=[app_name], source="deterministic")

    for prefix in ("close ", "quit ", "exit "):
        if text.startswith(prefix):
            app_name = text[len(prefix):].strip()
            if app_name and not any(w in app_name.split() for w in ("and", "then", "after")):
                return ParsedCommand(command="/close_apps", args=[app_name], source="deterministic")

    if text in {"status", "system status", "what is running"}:
        return ParsedCommand(command="/status", args=[], source="deterministic")

    if text in {"sysinfo", "system info", "system information"}:
        return ParsedCommand(command="/sysinfo", args=[], source="deterministic")

    # Desktop-control grammar — deterministic mouse/keyboard commands that would
    # otherwise fall through to the slow, LLM-driven desktop agent.
    desktop = _parse_desktop_command(transcript, text)
    if desktop is not None:
        return desktop

    return None


# Click phrase → (button, count). Matched against the normalized transcript.
_CLICK_VARIANTS = {
    "click": ("left", "1"),
    "left click": ("left", "1"),
    "double click": ("left", "2"),
    "triple click": ("left", "3"),
    "right click": ("right", "1"),
    "middle click": ("middle", "1"),
}


def _parse_desktop_command(transcript: str, text: str) -> ParsedCommand | None:
    """Parse a deterministic desktop-control command, or return None.

    ``text`` is the normalized (lowercased) transcript used for matching;
    ``transcript`` is the raw transcript, used to preserve case for the
    payloads of /type and /copy.
    """
    if text in ("arm desktop control", "arm desktop"):
        return ParsedCommand(command="/desktop_arm", args=[], source="deterministic")

    if text == "paste":
        return ParsedCommand(command="/paste", args=[], source="deterministic")

    if text in _CLICK_VARIANTS:
        button, count = _CLICK_VARIANTS[text]
        return ParsedCommand(command="/click", args=[button, count], source="deterministic")

    m = re.match(r"^move (?:the )?mouse (-?\d+)[ ,]+(-?\d+)$", text)
    if m:
        return ParsedCommand(command="/mouse_move", args=[m.group(1), m.group(2)], source="deterministic")

    m = re.match(r"^scroll (up|down)(?: (\d+))?$", text)
    if m:
        amount = int(m.group(2)) if m.group(2) else 300
        if m.group(1) == "down":
            amount = -amount
        return ParsedCommand(command="/scroll", args=[str(amount)], source="deterministic")

    m = re.match(r"^press ([a-z0-9]+)$", text)
    if m:
        return ParsedCommand(command="/press", args=[m.group(1)], source="deterministic")

    if text.startswith("hotkey "):
        keys = [k for k in re.split(r"[+\s]+", text[len("hotkey "):].strip()) if k]
        if len(keys) >= 2:
            return ParsedCommand(command="/hotkey", args=keys, source="deterministic")

    # /type and /copy preserve the user's original casing, so slice the payload
    # from the raw transcript rather than the lowercased ``text``.
    m = re.match(r"^\s*(type|copy)\s+(.+)$", transcript, re.IGNORECASE)
    if m:
        payload = " ".join(m.group(2).split())
        command = "/type" if m.group(1).lower() == "type" else "/copy"
        return ParsedCommand(command=command, args=[payload], source="deterministic")

    return None


async def parse_with_llm(transcript: str) -> ParsedCommand | None:
    """Fall back to a small local command translator."""
    config = app_control.load_apps_config()
    app_names = ", ".join(sorted(config.apps)) or "(none configured)"
    mode_names = ", ".join(sorted(config.modes)) or "(none configured)"
    system_prompt = (
        "You translate local PC voice requests into one Kira command. "
        "Return ONLY JSON: {\"command\": string, \"args\": [strings]}.\n\n"
        "Supported safe commands:\n"
        "- /open <app>  (works for any app, not just configured ones)\n"
        "- /close_apps <app>\n"
        "- /status\n"
        "- /sysinfo\n"
        "- /mode_run <mode>\n"
        "- /click <button> <count>  (button: left/right/middle, count: integer)\n"
        "- /mouse_move <x> <y>\n"
        "- /scroll <amount>  (positive=up, negative=down)\n"
        "- /type <text to type>\n"
        "- /press <key>  (e.g. space, enter, playpause, volumeup, volumedown)\n"
        "- /hotkey <key1> <key2> ...  (e.g. ctrl alt delete)\n"
        "- /copy <text>\n"
        "- /paste\n\n"
        "Supported risky commands, which will require terminal confirmation:\n"
        "- /shell <command>\n"
        "- /sleep\n"
        "- /shutdown <minutes>\n"
        "- /reboot <minutes>\n"
        "- /kill <pid>\n\n"
        f"Configured apps: {app_names}\n"
        f"Configured modes: {mode_names}\n"
        "Examples:\n"
        "  'click' -> {\"command\": \"/click\", \"args\": [\"left\", \"1\"]}\n"
        "  'press space' -> {\"command\": \"/press\", \"args\": [\"space\"]}\n"
        "  'play pause' -> {\"command\": \"/press\", \"args\": [\"playpause\"]}\n"
        "  'volume up' -> {\"command\": \"/press\", \"args\": [\"volumeup\"]}\n"
        "  'type hello world' -> {\"command\": \"/type\", \"args\": [\"hello world\"]}\n"
        "  'open notepad' -> {\"command\": \"/open\", \"args\": [\"notepad\"]}\n"
        "If the request is not about local PC control, return "
        "{\"command\":\"\",\"args\":[]}."
    )

    response = await provider.create_chat_completion(
        role="fast",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript},
        ],
        temperature=0,
        max_tokens=120,
    )
    raw = (response.choices[0].message.content or "").strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            return None

    command = str(parsed.get("command", "")).strip()
    args = [str(item) for item in parsed.get("args", []) if str(item).strip()]
    if not command:
        return None
    return ParsedCommand(
        command=command,
        args=args,
        source="llm",
        risky=is_risky_command(command, args),
    )


def is_risky_command(command: str, args: list[str]) -> bool:
    """Return True for commands that need local terminal confirmation."""
    normalized = command.strip().lower()
    if normalized in {"/shell", "/sleep", "/shutdown", "/reboot", "/kill"}:
        return True
    if normalized == "/close_apps":
        return False
    if normalized == "/run":
        return True
    return False
