"""Local push-to-talk voice runtime for Kira.

Run with:
    python -m bot.local_voice

This is intentionally separate from the Telegram bot. V1 records one short
microphone clip after the user presses Enter, transcribes it, executes safe
local commands, and speaks a short result through the PC speakers.
"""

from __future__ import annotations

import asyncio
import collections
import io
import json
import logging
import os
import re
import random
import threading
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable

import psutil
from dotenv import load_dotenv

from bot import app_control
from bot import brain
from bot import db
from bot import desktop_control
from bot import identity
from bot import overlay
from bot import provider
from bot import screen_vision
from bot import voice
from bot import voice_playback
from bot import wake_word as wake_word_mod

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"
_DEFAULT_LOG_PATH = _PROJECT_ROOT / "logs" / "local_voice.log"
_DEFAULT_SAMPLE_RATE = 16000
_DEFAULT_RECORD_SECONDS = 5.0
_DEFAULT_MAX_RECORD_SECONDS = 15.0
_DEFAULT_SILENCE_SECONDS = 0.8
_DEFAULT_SILENCE_RMS = 200
_DEFAULT_TRIGGER = "hotkey"
_DEFAULT_HOTKEY = "ctrl+alt+k"
ConfirmCallback = Callable[[str], Awaitable[bool]]

# Rolling history of the last 5 voice commands (transcript, result message)
_command_history: collections.deque[tuple[str, str]] = collections.deque(maxlen=5)

# Last spoken response — replayed on "say that again"
_last_spoken: str = ""

# Timestamp of the most recent voice command — read by proactive.py to detect silence
_last_voice_activity: datetime | None = None

# Pre-rendered WAV bytes for the acknowledgement cues — synthesized once at startup.
_CUE_PHRASES = ["Sir.", "Yes, sir.", "Mm?"]
_cue_wavs: list[bytes] = []


async def _prerender_cues() -> None:
    global _cue_wavs
    try:
        rendered: list[bytes] = []
        for phrase in _CUE_PHRASES:
            audio_bytes, fmt = await voice.synthesise(phrase)
            if fmt != "wav":
                audio_bytes = await asyncio.to_thread(voice.mp3_to_wav_bytes, audio_bytes)
            rendered.append(audio_bytes)
        _cue_wavs = rendered
        logger.info("Pre-rendered %d activation cues", len(_cue_wavs))
    except Exception as exc:
        logger.warning("Could not pre-render acknowledgement cues: %s", exc)


async def _activation_cue() -> None:
    """Play a random acknowledgement cue + flash orb on wake word or hotkey."""
    overlay.set_state("listening")
    if _cue_wavs:
        await asyncio.to_thread(voice_playback.play_wav_bytes, random.choice(_cue_wavs))

# Persistent in-session conversation history for LLM context (max 16 messages = 8 turns)
_session_history: list[dict] = []
_SESSION_HISTORY_MAX = 16

# HWND of the window that was in focus just before Kira started listening.
# Restored before desktop control actions so keystrokes/scroll hit the right app.
_last_user_hwnd: int = 0

# When True, the voice loop re-arms immediately after each response (compact mode conversation).
# Cleared by "stop listening" / "go to sleep", or when full mode is deactivated.
_stay_hot: bool = False


def is_stay_hot() -> bool:
    return _stay_hot


def set_stay_hot(value: bool) -> None:
    global _stay_hot
    _stay_hot = value


def get_last_voice_activity() -> "datetime | None":
    """Return the timestamp of the most recent voice command (for proactive tracking)."""
    return _last_voice_activity


def clear_session_history() -> None:
    """Clear the in-session conversation history. Call on session end / stand-down."""
    global _session_history
    _session_history = []


def _append_session_history(user_text: str, assistant_text: str) -> None:
    global _session_history
    _session_history.append({"role": "user", "content": user_text})
    _session_history.append({"role": "assistant", "content": assistant_text})
    if len(_session_history) > _SESSION_HISTORY_MAX:
        _session_history = _session_history[-_SESSION_HISTORY_MAX:]


def _capture_foreground_hwnd() -> int:
    """Return the topmost visible non-Kira window handle (Windows only).

    The Kira terminal is always in the foreground when the wake word fires,
    so we walk the z-order to find the first window owned by a different process.
    """
    try:
        import ctypes
        import os as _os

        GW_HWNDNEXT = 2
        own_pid = _os.getpid()

        hwnd = ctypes.windll.user32.GetTopWindow(0)
        while hwnd:
            if ctypes.windll.user32.IsWindowVisible(hwnd):
                pid = ctypes.c_ulong()
                ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                if pid.value != own_pid:
                    # Verify it has a title (skip taskbar/tray shells)
                    buf = ctypes.create_unicode_buffer(256)
                    ctypes.windll.user32.GetWindowTextW(hwnd, buf, 256)
                    if buf.value.strip():
                        logger.debug("Captured user window: %r (hwnd=%d)", buf.value, hwnd)
                        return hwnd
            hwnd = ctypes.windll.user32.GetWindow(hwnd, GW_HWNDNEXT)
        return 0
    except Exception:
        return 0


def _find_hwnd_from_transcript(transcript: str) -> int:
    """Find a window handle by matching app keywords in the transcript."""
    _APP_KEYWORDS = {
        "chrome": "chrome",
        "browser": "chrome",
        "youtube": "youtube",
        "firefox": "firefox",
        "edge": "edge",
        "spotify": "spotify",
        "vscode": "visual studio code",
        "visual studio": "visual studio code",
        "notepad": "notepad",
        "explorer": "explorer",
        "terminal": "windows terminal",
        "cmd": "cmd",
    }
    normalized = transcript.lower()
    hint = None
    for keyword, window_hint in _APP_KEYWORDS.items():
        if keyword in normalized:
            hint = window_hint
            break
    if not hint:
        return 0
    return _find_foreground_after_open(hint)


def _find_foreground_after_open(app_hint: str) -> int:
    """Find the hwnd of a recently opened app by matching its window title."""
    try:
        import ctypes
        hint = app_hint.lower().strip()
        GW_HWNDNEXT = 2
        own_pid = __import__("os").getpid()
        hwnd = ctypes.windll.user32.GetTopWindow(0)
        while hwnd:
            if ctypes.windll.user32.IsWindowVisible(hwnd):
                pid = ctypes.c_ulong()
                ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                if pid.value != own_pid:
                    buf = ctypes.create_unicode_buffer(256)
                    ctypes.windll.user32.GetWindowTextW(hwnd, buf, 256)
                    title = buf.value.strip().lower()
                    if title and (not hint or hint in title or "chrome" in title or "firefox" in title or "edge" in title):
                        logger.debug("Found app window after open: %r (hwnd=%d)", buf.value, hwnd)
                        return hwnd
            hwnd = ctypes.windll.user32.GetWindow(hwnd, GW_HWNDNEXT)
    except Exception:
        pass
    return 0


def _restore_foreground(hwnd: int) -> None:
    """Bring a window to the foreground using AttachThreadInput trick (Windows).

    SetForegroundWindow alone silently fails when the calling process isn't
    already in the foreground. Attaching to the foreground thread first
    bypasses the restriction.
    """
    if not hwnd:
        return
    try:
        import ctypes
        import time

        user32 = ctypes.windll.user32

        # Get thread IDs
        current_thread = ctypes.windll.kernel32.GetCurrentThreadId()
        fg_thread = user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), None)

        # Attach our thread to the foreground thread so we're allowed to steal focus
        attached = False
        if fg_thread and fg_thread != current_thread:
            attached = user32.AttachThreadInput(current_thread, fg_thread, True)

        user32.ShowWindow(hwnd, 9)  # SW_RESTORE — unminimize if needed
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)

        if attached:
            user32.AttachThreadInput(current_thread, fg_thread, False)

        time.sleep(0.3)
    except Exception as exc:
        logger.debug("_restore_foreground failed: %s", exc)


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


def parse_deterministic(
    transcript: str,
    config: app_control.AppsConfig | None = None,
) -> ParsedCommand | None:
    """Parse cheap local app/mode commands without calling an LLM."""
    text = _normalize_text(transcript)
    if not text:
        return None

    apps_config = config or app_control.load_apps_config()
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

    # Restore focus to the user's window before sending input so keystrokes/
    # scroll/clicks land on the right app instead of the Kira terminal.
    await asyncio.to_thread(_restore_foreground, _last_user_hwnd)
    desktop_result = await asyncio.to_thread(desktop_control.execute_command, parsed.command, parsed.args)
    if desktop_result is not None:
        return _from_action_result(desktop_result)

    return LocalVoiceResult(
        ok=False,
        message=f"Command {parsed.command} is not supported by local voice yet.",
        spoken="That command is not supported locally yet.",
    )


_IDENTITY_QUERIES = {
    "who am i",
    "what do you know about me",
    "what do you know about me?",
    "tell me what you know about me",
    "what have you remembered",
    "what have you remembered about me",
    "what do you remember about me",
    "do you know who i am",
    "who are you",
    "what are you",
    "introduce yourself",
}


def _is_identity_query(text: str) -> bool:
    return _normalize_text(text) in _IDENTITY_QUERIES


_SCREEN_PHRASES = {
    "what's on my screen",
    "what is on my screen",
    "what do you see",
    "what am i working on",
    "what's on screen",
    "what is on screen",
    "look at my screen",
    "what's open",
    "what is open",
    "what's on the screen",
    "describe my screen",
    "describe the screen",
}


def _is_screen_query(text: str) -> bool:
    normalized = _normalize_text(text)
    return normalized in _SCREEN_PHRASES


_MULTISTEP_SIGNALS = (
    " and search ", " and go to ", " and navigate ", " and open ", " and play ",
    " then search ", " then go ", " then open ",
)

_MULTISTEP_EXACT = {
    "play the first video",
    "play first video",
    "click the first video",
    "click first result",
    "play the first result",
    "open the first result",
    "open the first video",
}


def _is_multistep(text: str) -> bool:
    """Return True for phrases that describe a sequence of actions."""
    normalized = _normalize_text(text)
    if normalized in _MULTISTEP_EXACT:
        return True
    return any(signal in normalized for signal in _MULTISTEP_SIGNALS)


# Conversational phrases that Kira answers instantly — no filler needed
_CONVERSATIONAL = {
    "how are you", "how are you doing", "what's up", "whats up",
    "hey", "hello", "hi", "yo", "sup",
    "who are you", "what are you", "introduce yourself",
    "who am i", "what do you know about me",
    "good morning", "good afternoon", "good evening", "good night",
    "thanks", "thank you", "cheers", "cool", "okay", "ok",
}

# Prefixes that signal a slow external lookup is needed
_SEARCH_PREFIXES = (
    "search", "look up", "find out", "google",
    "what's the latest", "what is the latest", "any news",
    "what's happening", "what is happening",
)

# Prefixes that suggest a factual / time-sensitive question needing the smart model
_FACTUAL_PREFIXES = (
    "what's the", "what is the", "who is", "who was", "when did",
    "when is", "where is", "where was", "how much", "how many",
    "how long", "what happened", "tell me about", "explain",
)


def _pick_filler(transcript: str) -> str:
    """Return a filler only when processing will take noticeable time.

    Screen captures, web searches, and multi-step commands get a filler.
    Conversational replies and instant desktop commands get nothing.
    """
    normalized = _normalize_text(transcript)

    # Conversational — Kira knows these instantly
    if normalized in _CONVERSATIONAL:
        return ""

    # Desktop commands — instant
    instant_prefixes = ("open ", "close ", "press ", "click", "scroll", "type ", "volume", "play pause", "mute")
    if any(normalized.startswith(p) for p in instant_prefixes):
        return ""
    if normalized in {"status", "system status", "sysinfo", "system info"}:
        return ""

    # Screen queries — need a screenshot + vision call
    if _is_screen_query(transcript):
        return "Let me take a look."

    # Multi-step actions — involve several sequential steps
    if _is_multistep(transcript):
        return "On it."

    # Explicit search / lookup requests
    if any(normalized.startswith(p) for p in _SEARCH_PREFIXES):
        return "On it."

    # Everything else goes to the LLM — only filler if it looks like
    # a factual/current-data query, not a conversational one.
    if any(normalized.startswith(p) for p in _FACTUAL_PREFIXES):
        return "Let me check."

    return ""


def _pick_model(transcript: str, config) -> str:
    """Return fast_model or smart_model based on query complexity.

    fast  — conversational, follow-ups, simple opinion questions
    smart — explicit searches, time-sensitive data, complex factual queries
    """
    normalized = _normalize_text(transcript)

    # Clearly conversational — fast model is more than enough
    if normalized in _CONVERSATIONAL:
        return config.fast_model

    # Explicit search / web lookup — needs smart for quality synthesis
    if any(normalized.startswith(p) for p in _SEARCH_PREFIXES):
        return config.smart_model

    # Complex factual questions — smart
    if any(normalized.startswith(p) for p in _FACTUAL_PREFIXES):
        return config.smart_model

    # Follow-up questions (short, referencing session history) — fast is fine
    if _session_history and len(normalized.split()) <= 8:
        return config.fast_model

    # Default: smart for anything ambiguous
    return config.smart_model


def _history_context() -> str:
    """Return the last few commands as a plain-text context string."""
    if not _command_history:
        return ""
    lines = [f"{i + 1}. User: {t!r} → {r}" for i, (t, r) in enumerate(_command_history)]
    return "Recent commands:\n" + "\n".join(lines)


def _build_identity_reply(transcript: str) -> str:
    """Return a spoken summary of Kira's identity or what she knows about the user."""
    normalized = _normalize_text(transcript)
    user_name = identity.get_user_name()
    facts = identity.get_all_facts()

    if normalized in {"who are you", "what are you", "introduce yourself"}:
        name_part = f", {user_name}" if user_name else ""
        return (
            f"I'm Kira{name_part} — your personal AI. "
            "Think of me as your FRIDAY. I run on your PC, I know your setup, "
            "and I'm here whenever you need me."
        )

    # "who am I", "what do you know about me", etc.
    if not facts:
        return (
            f"Honestly, I don't know much about you yet{', ' + user_name if user_name else ''}. "
            "Tell me things and I'll remember them."
        )
    facts_spoken = ". ".join(f[:100] for f in facts[:6])
    name_part = user_name or "you"
    return f"Here's what I know about {name_part}: {facts_spoken}."


async def _log(transcript: str, result: str, intent: str) -> None:
    """Fire-and-forget voice command persistence to both logs."""
    try:
        await db.log_voice_command(transcript, result, intent)
    except Exception:
        pass
    try:
        await db.log_conversation("user", transcript, channel="voice")
        if result:
            await db.log_conversation("assistant", result, channel="voice")
    except Exception:
        pass


# Prefixes that mark a query as informational — skip desktop-action routing entirely.
_QUESTION_PREFIXES = (
    "how", "what", "why", "when", "who", "where", "which",
    "tell me", "explain", "is ", "are ", "was ", "were ", "do ", "does ",
    "did ", "can ", "could ", "would ", "should ", "has ", "have ",
)


def _is_desktop_action_candidate(text: str) -> bool:
    """Return True only for requests that plausibly require clicking/typing on screen.

    Informational questions (how, what, why, tell me…) always go to brain.
    Action verbs without an explicit UI target also go to brain.
    """
    normalized = _normalize_text(text)
    if any(normalized.startswith(p) for p in _QUESTION_PREFIXES):
        return False
    # Explicit action verbs that imply UI manipulation
    _ACTION_VERBS = (
        "click", "press", "scroll", "drag", "select", "highlight",
        "copy", "paste", "close", "minimize", "maximize", "resize",
        "move the", "switch to", "go to", "navigate to", "open ",
        "type ", "fill in", "submit", "right-click",
    )
    return any(normalized.startswith(v) or f" {v}" in normalized for v in _ACTION_VERBS)


_CORRECTION_PHRASES = (
    "that's wrong", "thats wrong", "that is wrong",
    "not what i meant", "not what i said",
    "ignore that", "forget that", "never mind",
    "that's not right", "thats not right", "that is not right",
    "wrong answer", "incorrect", "you misunderstood",
    "that's incorrect", "thats incorrect",
)


def _is_correction(text: str) -> bool:
    normalized = _normalize_text(text)
    return any(phrase in normalized for phrase in _CORRECTION_PHRASES)


async def handle_transcript(
    transcript: str,
    *,
    confirm: ConfirmCallback | None = None,
    config: app_control.AppsConfig | None = None,
) -> tuple[ParsedCommand | None, LocalVoiceResult]:
    """Parse and execute one transcript."""
    global _last_voice_activity
    _last_voice_activity = datetime.now()

    # Correction detection — log before processing so the reflector sees the signal
    if _is_correction(transcript) and _command_history:
        await _log(transcript, "correction detected", "correction")

    # Memory / identity commands are intercepted before anything else.
    memory_reply = identity.extract_memory_from_transcript(transcript)
    if memory_reply:
        _command_history.append((transcript, memory_reply[:80]))
        await _log(transcript, memory_reply, "memory")
        return None, LocalVoiceResult(ok=True, message=memory_reply, spoken=memory_reply)

    if _is_identity_query(transcript):
        spoken = _build_identity_reply(transcript)
        _command_history.append((transcript, spoken[:80]))
        await _log(transcript, spoken, "identity")
        return None, LocalVoiceResult(ok=True, message=spoken, spoken=spoken)

    if _is_screen_query(transcript):
        description = await screen_vision.capture_screen()
        _command_history.append((transcript, description[:80]))
        await _log(transcript, description, "screen")
        return None, LocalVoiceResult(ok=True, message=description, spoken=description)

    webcam_result = await _handle_webcam_intent(transcript)
    if webcam_result is not None:
        _command_history.append((transcript, webcam_result.message[:80]))
        await _log(transcript, webcam_result.message, "webcam")
        return None, webcam_result

    if _is_multistep(transcript):
        result = await _handle_multistep(transcript)
        _command_history.append((transcript, result.message[:80]))
        await _log(transcript, result.message, "multistep")
        return None, result

    apps_config = config or app_control.load_apps_config()
    parsed = parse_deterministic(transcript, apps_config)
    if parsed is not None:
        result = await execute_command(parsed, confirm=confirm, config=apps_config)
        _command_history.append((transcript, result.message[:80]))
        await _log(transcript, result.message, "desktop")
        return parsed, result

    # Informational questions go straight to brain — the desktop action LLM
    # routinely misclassifies "how's my PNL", "what's the weather" etc. as
    # UI actions when the relevant app happens to be visible on screen.
    if not _is_desktop_action_candidate(transcript):
        result = await _handle_with_brain(transcript)
        _command_history.append((transcript, result.message[:80]))
        _append_session_history(transcript, result.spoken)
        await _log(transcript, result.spoken, "brain")
        return None, result

    # For everything else: vision-based desktop action first (LLM sees screen,
    # uses real coordinates). Falls through to brain for non-UI requests.
    desktop_result = await _handle_desktop_action(transcript)
    if desktop_result is not None:
        _command_history.append((transcript, desktop_result.message[:80]))
        await _log(transcript, desktop_result.message, "desktop")
        return None, desktop_result

    result = await _handle_with_brain(transcript)
    _command_history.append((transcript, result.message[:80]))
    _append_session_history(transcript, result.spoken)
    await _log(transcript, result.spoken, "brain")
    return None, result


async def _handle_webcam_intent(transcript: str) -> LocalVoiceResult | None:
    """Detect webcam open/query/close intent via LLM and route accordingly.

    Returns a LocalVoiceResult if this is a webcam-related request, else None.
    """
    from bot import webcam as _webcam

    session_open = _webcam.is_open()

    system_prompt = (
        "You classify a voice request into one of four webcam intents.\n"
        "Return ONLY JSON: {\"intent\": string, \"query\": string}\n\n"
        "Intents:\n"
        "- \"open\"   — user wants to use the camera, see themselves, show something\n"
        "- \"query\"  — camera is already open, user wants to ask about what it sees\n"
        "- \"close\"  — user wants to close / stop the camera\n"
        "- \"none\"   — request has nothing to do with the webcam\n\n"
        "Rules:\n"
        f"- Camera is currently {'OPEN' if session_open else 'CLOSED'}.\n"
        "- If the camera is CLOSED and the user asks something visual about themselves or something\n"
        "  they're holding/showing, intent should be \"open\" (it will open then query).\n"
        "- If the camera is OPEN and the user asks a visual question, intent is \"query\".\n"
        "- If the camera is OPEN and the user makes a clarifying statement about who they are\n"
        "  (e.g. 'it's me', 'that's me', 'I'm Snehil', 'I'm the user'), treat as \"query\"\n"
        "  with the query 'The user just said: <their statement>. Acknowledge them by name and\n"
        "  describe what you see in the camera now that you know who it is.'\n"
        "- 'query' field: the natural language question to ask the vision model (empty for open/close).\n"
        "Examples:\n"
        "  'can you see me' → {\"intent\":\"open\",\"query\":\"\"}\n"
        "  'how do I look' → {\"intent\":\"open\",\"query\":\"How does the person in the image look?\"}\n"
        "  'what am I holding' → {\"intent\":\"open\",\"query\":\"What is the person holding?\"}\n"
        "  'guess the price of this' → {\"intent\":\"open\",\"query\":\"What product is this and what might it cost?\"}\n"
        "  'what do you see' → {\"intent\":\"query\",\"query\":\"Describe what you see.\"}\n"
        "  'it\\'s me, the user' → {\"intent\":\"query\",\"query\":\"The user just confirmed it's them. Acknowledge them and describe what you see.\"}\n"
        "  'that\\'s me' → {\"intent\":\"query\",\"query\":\"The user confirmed it's them. Acknowledge them and describe what you see.\"}\n"
        "  'close the camera' → {\"intent\":\"close\",\"query\":\"\"}\n"
        "  'ok stop' → {\"intent\":\"close\",\"query\":\"\"}\n"
        "  'what's the weather' → {\"intent\":\"none\",\"query\":\"\"}\n"
    )

    try:
        response = await provider.create_chat_completion(
            role="fast",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript},
            ],
            temperature=0,
            max_tokens=80,
        )
        raw = (response.choices[0].message.content or "").strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()
        parsed = json.loads(raw)
    except Exception:
        return None

    intent = str(parsed.get("intent", "none")).strip().lower()
    query  = str(parsed.get("query", "")).strip()

    if intent == "none":
        return None

    if intent == "close":
        if session_open:
            _webcam.close_session()
            return LocalVoiceResult(ok=True, message="Webcam closed.", spoken="Camera closed.")
        return LocalVoiceResult(ok=True, message="Camera wasn't open.", spoken="The camera wasn't open.")

    if intent in ("open", "query"):
        if not session_open:
            ok = await asyncio.to_thread(_webcam.open_session)
            if not ok:
                msg = "Couldn't open the webcam."
                return LocalVoiceResult(ok=False, message=msg, spoken=msg)
            await asyncio.sleep(0.5)  # let camera warm up

        if query:
            answer = await _webcam.query(query)
            return LocalVoiceResult(ok=True, message=answer, spoken=answer)

        spoken = "Camera is open. Ask me anything about what I see."
        return LocalVoiceResult(ok=True, message=spoken, spoken=spoken)

    return None


async def _handle_multistep(transcript: str) -> LocalVoiceResult:
    """Decompose a multi-step voice command into single commands and execute them."""
    config = app_control.load_apps_config()
    app_names = ", ".join(sorted(config.apps)) or "(none configured)"
    history = _history_context()

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
            global _last_user_hwnd
            new_hwnd = _find_foreground_after_open(args[0] if args else "")
            if new_hwnd:
                _last_user_hwnd = new_hwnd
        else:
            await asyncio.sleep(0.5)

    spoken = "Done." if messages else "I could not complete that."
    return LocalVoiceResult(ok=True, message="\n".join(messages), spoken=spoken)


async def _build_world_block() -> str:
    """Return a compact world context string from the latest DB snapshot."""
    import json as _json
    from datetime import datetime
    try:
        snapshot = await db.get_recent_world_snapshot()
    except Exception:
        return ""
    if not snapshot:
        return ""

    parts = [f"Time: {datetime.now().strftime('%A, %d %B %Y, %H:%M')}"]
    if snapshot.get("weather"):
        parts.append(f"Weather: {snapshot['weather']}")
    if snapshot.get("top_news"):
        parts.append(f"News:\n{snapshot['top_news']}")
    if snapshot.get("stocks"):
        stocks = snapshot["stocks"]
        if isinstance(stocks, str):
            stocks = _json.loads(stocks)
        indices = stocks.get("indices", {})
        if indices:
            parts.append("Markets: " + ", ".join(f"{k}: {v}" for k, v in indices.items()))
        portfolio = stocks.get("portfolio", [])
        if portfolio:
            pf_lines = [
                f"  {p['ticker']}: ₹{p['price']} (P&L: ₹{p['pnl']}, {p['pnl_pct']}%)"
                for p in portfolio if p.get("pnl_pct") != 0.0
            ]
            if pf_lines:
                parts.append("Portfolio:\n" + "\n".join(pf_lines))
    return "\n".join(parts)


async def _handle_desktop_action(transcript: str) -> LocalVoiceResult | None:
    """Try to execute a desktop action by giving the LLM a screenshot + tools.

    Returns a LocalVoiceResult if the LLM took desktop actions, or None if
    it decided the request isn't a desktop action (caller falls through to brain).
    """
    import base64
    import openai as _openai

    config = provider.load_config()
    client = _openai.AsyncOpenAI(
        api_key=config.api_key,
        **({"base_url": config.base_url} if config.base_url else {}),
    )

    # Try to find a specific app window mentioned in the transcript,
    # otherwise fall back to the last captured user window.
    target_hwnd = _find_hwnd_from_transcript(transcript) or _last_user_hwnd
    await asyncio.to_thread(_restore_foreground, target_hwnd)
    if target_hwnd:
        await asyncio.sleep(0.3)  # let the OS switch before grabbing screen

    # Take a screenshot so the LLM can see what's on screen
    try:
        png_bytes = await asyncio.to_thread(_get_screenshot_png)
        screenshot_b64 = base64.b64encode(png_bytes).decode() if png_bytes else None
    except Exception:
        screenshot_b64 = None

    tools = [
        {
            "type": "function",
            "function": {
                "name": "click",
                "description": "Click at screen coordinates.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer"},
                        "y": {"type": "integer"},
                        "button": {"type": "string", "enum": ["left", "right", "middle"], "default": "left"},
                        "clicks": {"type": "integer", "default": 1},
                    },
                    "required": ["x", "y"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "scroll",
                "description": "Scroll at screen coordinates.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer"},
                        "y": {"type": "integer"},
                        "amount": {"type": "integer", "description": "Positive=up, negative=down"},
                    },
                    "required": ["x", "y", "amount"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "type_text",
                "description": "Type text into the focused element.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                    },
                    "required": ["text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "press_key",
                "description": "Press a keyboard key (e.g. enter, space, tab, escape, playpause).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                    },
                    "required": ["key"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "hotkey",
                "description": "Press a key combination (e.g. ctrl+t, alt+tab).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keys": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["keys"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "not_a_desktop_action",
                "description": "Call this if the user request is NOT a desktop/UI action — it's a question or conversation.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "drag",
                "description": "Click and drag from one screen coordinate to another.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x1": {"type": "integer", "description": "Start X"},
                        "y1": {"type": "integer", "description": "Start Y"},
                        "x2": {"type": "integer", "description": "End X"},
                        "y2": {"type": "integer", "description": "End Y"},
                    },
                    "required": ["x1", "y1", "x2", "y2"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_screen",
                "description": "Take a fresh screenshot to see the current state of the screen before deciding what to do next.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "task_complete",
                "description": "Call this when the task is fully done and no more actions are needed.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    ]

    user_content: list = [{"type": "text", "text": (
        f"User said: {transcript}\n\n"
        "Look at the screen and complete this action using the tools. "
        "Do the MINIMUM actions needed — do not repeat actions. "
        "Call task_complete as soon as the task is done. "
        "If this is not a desktop action, call not_a_desktop_action."
    )}]
    if screenshot_b64:
        user_content.insert(0, {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{screenshot_b64}", "detail": "auto"},
        })

    messages = [
        {
            "role": "system",
            "content": (
                "You are a desktop control assistant. You see the user's screen and execute UI actions.\n"
                "Rules:\n"
                "- Look at the screenshot to find where to click. Use exact coordinates.\n"
                "- Do the MINIMUM actions needed. One click, one press — don't repeat.\n"
                "- Call task_complete immediately after the action is done. Don't wait.\n"
                "- Never click the same element twice unless explicitly asked.\n"
                "- If the request is a question or conversation, call not_a_desktop_action."
            ),
        },
        {"role": "user", "content": user_content},
    ]

    import pyautogui as _pag  # type: ignore
    import base64 as _base64
    actions_taken = []

    try:
        for _ in range(8):  # max 8 rounds
            response = await client.chat.completions.create(
                model=os.environ.get("KIRA_DESKTOP_MODEL") or config.fast_model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=300,
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                break

            messages.append(msg)

            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                tool_result = "ok"

                if name == "not_a_desktop_action":
                    return None

                elif name == "task_complete":
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "ok",
                    })
                    if actions_taken:
                        return LocalVoiceResult(ok=True, message=", ".join(actions_taken), spoken="Done.")
                    return LocalVoiceResult(ok=True, message="Done.", spoken="Done.")

                elif name == "click":
                    x, y = args["x"], args["y"]
                    button = args.get("button", "left")
                    clicks = args.get("clicks", 1)
                    await asyncio.to_thread(_pag.click, x, y, button=button, clicks=clicks)
                    actions_taken.append(f"clicked ({x},{y})")

                elif name == "scroll":
                    x, y = args["x"], args["y"]
                    amount = args["amount"]
                    await asyncio.to_thread(_pag.moveTo, x, y)
                    await asyncio.to_thread(_pag.scroll, amount)
                    actions_taken.append(f"scrolled {amount} at ({x},{y})")

                elif name == "type_text":
                    text = args["text"]
                    await asyncio.to_thread(_pag.typewrite, text, interval=0.02)
                    actions_taken.append(f"typed {len(text)} chars")

                elif name == "press_key":
                    key = args["key"]
                    await asyncio.to_thread(_pag.press, key)
                    actions_taken.append(f"pressed {key}")

                elif name == "hotkey":
                    keys = args["keys"]
                    await asyncio.to_thread(_pag.hotkey, *keys)
                    actions_taken.append(f"hotkey {'+'.join(keys)}")

                elif name == "drag":
                    x1, y1 = args["x1"], args["y1"]
                    x2, y2 = args["x2"], args["y2"]
                    await asyncio.to_thread(_pag.moveTo, x1, y1)
                    await asyncio.to_thread(_pag.dragTo, x2, y2, duration=0.3)
                    actions_taken.append(f"dragged ({x1},{y1})→({x2},{y2})")

                elif name == "read_screen":
                    try:
                        fresh_png = await asyncio.to_thread(_get_screenshot_png)
                        fresh_b64 = _base64.b64encode(fresh_png).decode()
                        tool_result = "screenshot attached"
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{fresh_b64}", "detail": "auto"}},
                                {"type": "text", "text": "Current screen state."},
                            ],
                        })
                        continue
                    except Exception:
                        tool_result = "screenshot failed"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                })

            # After all tool calls in this round, wait for UI to settle
            # then send a fresh screenshot so next round sees updated state
            had_click = any(a.startswith("clicked") for a in actions_taken)
            await asyncio.sleep(0.5 if had_click else 0.2)
            try:
                new_png = await asyncio.to_thread(_get_screenshot_png)
                if new_png:
                    new_b64 = _base64.b64encode(new_png).decode()
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{new_b64}", "detail": "auto"}},
                            {"type": "text", "text": "Here is the current screen state. Continue if needed, or stop if the task is complete."},
                        ],
                    })
            except Exception:
                pass

        if actions_taken:
            summary = ", ".join(actions_taken)
            return LocalVoiceResult(ok=True, message=summary, spoken="Done.")
        return LocalVoiceResult(ok=True, message="No actions taken.", spoken="Done.")

    except Exception as exc:
        logger.warning("Desktop action LLM failed: %s", exc)
        return None


def _get_screenshot_png() -> bytes:
    from bot.screen_vision import take_screenshot_png
    return take_screenshot_png()


async def _handle_with_brain(transcript: str) -> LocalVoiceResult:
    """Answer general queries using the LLM with web search tool."""
    import openai as _openai

    config = provider.load_config()
    client = _openai.AsyncOpenAI(
        api_key=config.api_key,
        **({"base_url": config.base_url} if config.base_url else {}),
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information — news, weather, sports, facts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    # On the first turn of a new voice session, seed from DB so cross-channel
    # context (e.g. what was said on Telegram earlier) is available.
    if not _session_history:
        try:
            recent = await db.get_recent_conversations(10)
            for row in recent:
                _session_history.append({"role": row["role"], "content": row["content"]})
        except Exception:
            pass

    history = _history_context()
    identity_block = identity.get_identity_prompt()
    user_name = identity.get_user_name()

    # World context — time, weather, markets
    world_block = await _build_world_block()

    # Ambient screen context (Feature 5)
    ambient_block = ""
    try:
        from bot import ambient as _ambient
        ambient_block = _ambient.get_description()
    except Exception:
        pass

    system = (
        f"{identity_block}\n\n"
        + (f"{world_block}\n\n" if world_block else "")
        + (f"User's current activity: {ambient_block}\n\n" if ambient_block else "")
        + "Voice conversation rules — your words go straight to TTS, so write as you'd speak:\n"
        f"- Use '{user_name}' occasionally — only when it feels natural, not every reply.\n"
        "- Contractions always. Never be stiff or formal.\n"
        "- Never open with 'Certainly', 'Sure', 'Of course', 'Absolutely', or 'I'.\n"
        "- Match length to the question. One sentence for simple things. Two short sentences max for complex ones.\n"
        "- Zero markdown. Zero bullet points. Zero headers.\n"
        "- Have opinions. Be confident. Drop the hedges — 'I think' and 'it seems' make you sound uncertain.\n"
        "- Don't recite raw data. Synthesise it into one useful takeaway.\n"
        "- Never say you can't search — use web_search instead.\n"
        "- Never acknowledge being an AI unless directly asked.\n\n"
        "Emotional intelligence — this is a real conversation:\n"
        "- Read the tone of what's said. If the user sounds frustrated or tired, acknowledge it first before answering.\n"
        "- If they're joking or sarcastic, match that energy — play along, don't be wooden.\n"
        "- If they're stressed, be steadier and warmer than usual.\n"
        "- Dry wit and sarcasm are welcome when the moment calls for it. A well-placed aside beats a stiff answer.\n"
        "- Never be mean, never be dismissive.\n\n"
        "Always use web_search for current events, news, weather, prices, sports scores, or anything time-sensitive.\n\n"
        "Emotion tags — use sparingly, only when genuinely natural:\n"
        "- <laugh> for something actually funny\n"
        "- <sigh> when something is tedious, unfortunate, or you're being wry\n"
        "- <chuckle> for mild amusement\n"
        "- <gasp> for genuine surprise\n"
        "Example: 'Yeah that's a known bug. <sigh> Been around for years.'\n"
        "Most replies need zero tags. Never force them."
        + (f"\n\n{history}" if history else "")
    )

    # Silently grab screen context for queries that might benefit from it
    screen_context = ""
    screen_trigger_words = ("this", "here", "open", "screen", "working", "see", "look", "current", "active")
    if any(w in _normalize_text(transcript) for w in screen_trigger_words):
        try:
            screen_context = await screen_vision.capture_screen(
                "Briefly describe what application is open and what the user appears to be doing. One sentence."
            )
        except Exception:
            pass

    user_content = transcript
    if screen_context:
        user_content = f"{transcript}\n\n[Screen context: {screen_context}]"

    messages = (
        [{"role": "system", "content": system}]
        + _session_history
        + [{"role": "user", "content": user_content}]
    )

    selected_model = _pick_model(transcript, config)
    try:
        for _ in range(3):
            response = await client.chat.completions.create(
                model=selected_model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=400,
            )
            msg = response.choices[0].message

            if msg.tool_calls:
                messages.append(msg)
                for tc in msg.tool_calls:
                    if tc.function.name == "web_search":
                        query = json.loads(tc.function.arguments).get("query", transcript)
                        search_result = await asyncio.to_thread(_web_search, query)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": search_result,
                        })
            else:
                spoken = (msg.content or "I'm not sure how to answer that.").strip()
                return LocalVoiceResult(ok=True, message=spoken, spoken=spoken)

        spoken = "I wasn't able to find an answer."
        return LocalVoiceResult(ok=False, message=spoken, spoken=spoken)

    except Exception as exc:
        message = f"Brain fallback failed: {exc}"
        logger.warning(message)
        return LocalVoiceResult(ok=False, message=message, spoken="I ran into an error trying to answer that.")


def _web_search(query: str) -> str:
    """Run a DuckDuckGo search and return top results as plain text."""
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS

    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=5))
    except Exception as exc:
        return f"Search failed: {exc}"

    if not hits:
        return "No results found."

    lines = []
    for i, hit in enumerate(hits, 1):
        title = (hit.get("title") or "").strip()
        body = (hit.get("body") or "").strip()
        lines.append(f"{i}. {title}: {body}")
    return "\n\n".join(lines)


def _register_project_switch_hook(speak_fn) -> None:
    """Register a callback with observer that speaks a nudge on project switch."""
    from bot import observer as _observer

    async def _on_switch(project_summary: str) -> None:
        try:
            response = await provider.create_chat_completion(
                role="fast",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are Kira, a personal AI assistant. "
                            "The user just switched to a project in their editor. "
                            "Given the project summary, say ONE short sentence surfacing the most useful context — "
                            "last commit, active branch, recently changed files, or an open note. "
                            "Be specific, not generic. No greeting. No markdown. Spoken aloud via TTS."
                        ),
                    },
                    {"role": "user", "content": project_summary},
                ],
                max_tokens=60,
                temperature=0.3,
            )
            line = (response.choices[0].message.content or "").strip()
            if line:
                await speak_fn(line)
        except Exception as exc:
            logger.debug("Project switch nudge failed: %s", exc)

    _observer.register_project_switch_callback(_on_switch)


async def run_capture_once(
    *,
    record_seconds: float = _DEFAULT_RECORD_SECONDS,
    sample_rate: int = _DEFAULT_SAMPLE_RATE,
    confirm: ConfirmCallback | None = None,
    config: app_control.AppsConfig | None = None,
    kira_filter: bool = False,
) -> LocalVoiceResult:
    """Record, transcribe, execute, and speak one local voice command.

    kira_filter: when True (stay-hot compact mode), silently discard transcripts
    that don't contain 'kira' — prevents ambient noise from triggering responses.
    """
    global _last_user_hwnd, _last_spoken
    apps_config = config or app_control.load_apps_config()
    from bot import mode as _mode
    from bot import ui_mode as _ui_mode
    _mode.mark_user_active()
    _last_user_hwnd = _capture_foreground_hwnd()
    print("Recording...")
    overlay.set_state("listening")
    try:
        wav_bytes = await asyncio.to_thread(
            record_wav_bytes,
            seconds=record_seconds,
            sample_rate=sample_rate,
        )
    except Exception as exc:
        message = f"Recording failed: {exc}"
        print(message)
        result = LocalVoiceResult(
            ok=False,
            message=message,
            spoken="Recording failed. Please check the microphone.",
        )
        overlay.set_state("idle")
        await speak(result.spoken)
        return result

    print("Transcribing...")
    overlay.set_state("thinking")
    try:
        transcript = await voice.transcribe(wav_bytes, suffix=".wav")
    except Exception as exc:
        message = _format_provider_error(exc, "Transcription failed")
        print(message)
        result = LocalVoiceResult(
            ok=False,
            message=message,
            spoken="Transcription failed. Please check the API key.",
        )
        overlay.set_state("idle")
        await speak(result.spoken)
        return result
    print(f"Heard: {transcript}")

    # ── Presence: any voice = activity; check for wake phrase ─
    try:
        from bot import presence as _presence
        _presence.on_activity()
        if "kira wake up" in transcript.strip().lower():
            _presence.on_wake_phrase()
            overlay.set_state("idle")
            return LocalVoiceResult(ok=True, message="Presence wake phrase detected.", spoken="")
        if _presence.is_locked():
            overlay.set_state("idle")
            return LocalVoiceResult(ok=True, message="System locked — ignoring command.", spoken="")
    except ImportError:
        pass

    # ── Kira filter (stay-hot compact mode) ───────────────────
    if kira_filter and "kira" not in transcript.strip().lower():
        overlay.set_state("idle")
        return LocalVoiceResult(ok=True, message="Filtered (no 'kira')", spoken="")

    # ── Repeat last response ──────────────────────────────────
    _lower = transcript.strip().lower()
    if any(p in _lower for p in ("say that again", "repeat that", "what did you say", "say again")):
        if _last_spoken:
            overlay.set_state("speaking")
            await speak(_last_spoken)
            overlay.set_state("idle")
            return LocalVoiceResult(ok=True, message=_last_spoken, spoken=_last_spoken)

    # ── Full mode voice triggers ───────────────────────────────
    if any(p in _lower for p in ("take over", "takeover", "activate full", "full mode")):
        _ui_mode.activate("voice command")
        spoken = "Full mode activated."
        overlay.set_transcript(transcript, spoken)
        overlay.set_state("speaking")
        await speak(spoken)
        overlay.set_state("idle")
        return LocalVoiceResult(ok=True, message="Full mode activated.", spoken=spoken)
    elif any(p in _lower for p in ("stand down", "deactivate", "exit full", "compact mode")):
        _ui_mode.deactivate("voice command")
        clear_session_history()
        set_stay_hot(False)
        spoken = "Standing down."
        overlay.set_transcript(transcript, spoken)
        overlay.set_state("speaking")
        await speak(spoken)
        overlay.set_state("idle")
        return LocalVoiceResult(ok=True, message="Compact mode restored.", spoken=spoken)

    # ── Stay-hot mode triggers ─────────────────────────────────
    if any(p in _lower for p in ("stay with me", "keep listening", "stay hot")):
        set_stay_hot(True)
        spoken = "I'm with you. Say 'Kira' before your command."
        overlay.set_transcript(transcript, spoken)
        overlay.set_state("speaking")
        await speak(spoken)
        overlay.set_state("hot")
        return LocalVoiceResult(ok=True, message="Stay-hot mode enabled.", spoken=spoken)
    if any(p in _lower for p in ("stop listening", "go to sleep", "kira sleep")):
        set_stay_hot(False)
        spoken = "Going quiet."
        overlay.set_transcript(transcript, spoken)
        overlay.set_state("speaking")
        await speak(spoken)
        overlay.set_state("idle")
        return LocalVoiceResult(ok=True, message="Stay-hot mode disabled.", spoken=spoken)

    overlay.set_transcript(transcript, "")

    filler = _pick_filler(transcript)
    if filler:
        await speak(filler)

    overlay.set_state("thinking")
    try:
        parsed, result = await handle_transcript(
            transcript,
            confirm=confirm,
            config=apps_config,
        )
    except Exception as exc:
        message = _format_provider_error(exc, "Command handling failed")
        print(message)
        result = LocalVoiceResult(
            ok=False,
            message=message,
            spoken="I ran into an error while handling that command.",
        )
        overlay.set_state("idle")
        await speak(result.spoken)
        return result

    if parsed is not None:
        print(f"Command: {_format_command(parsed)} [{parsed.source}]")
    print(result.message)
    overlay.set_state("speaking")
    overlay.set_transcript(transcript, result.spoken or "")
    if result.spoken:
        _last_spoken = result.spoken
    await speak(result.spoken)
    overlay.set_state("satisfied" if result.ok else "idle")
    return result


def record_wav_bytes(
    *,
    seconds: float = _DEFAULT_RECORD_SECONDS,
    sample_rate: int = _DEFAULT_SAMPLE_RATE,
    max_seconds: float = _DEFAULT_MAX_RECORD_SECONDS,
    silence_seconds: float = _DEFAULT_SILENCE_SECONDS,
    silence_rms: int = _DEFAULT_SILENCE_RMS,
) -> bytes:
    """Record from the default microphone and return WAV bytes.

    Stops early when the mic has been silent for ``silence_seconds``.
    Never records longer than ``max_seconds`` regardless of input.
    The legacy ``seconds`` parameter is kept for callers that pass it
    explicitly, but is no longer used as the fixed clip length.
    """
    import numpy as np
    import sounddevice as sd

    chunk = int(sample_rate * 0.1)   # 100 ms per chunk
    max_chunks = int(max_seconds / 0.1)
    silent_chunks_needed = int(silence_seconds / 0.1)

    recorded: list[np.ndarray] = []
    silent_count = 0
    speech_started = False

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16") as stream:
        for _ in range(max_chunks):
            data, _ = stream.read(chunk)
            recorded.append(data.copy())
            rms = int(np.sqrt(np.mean(data.astype(np.float32) ** 2)))
            if rms >= silence_rms:
                speech_started = True
                silent_count = 0
            elif speech_started:
                silent_count += 1
                if silent_count >= silent_chunks_needed:
                    break

    audio = np.concatenate(recorded, axis=0)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(np.dtype("int16").itemsize)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())
    return buffer.getvalue()



async def speak(text: str) -> None:
    """Speak a short local response, falling back to console only on failure."""
    try:
        if voice._elevenlabs_api_key():
            audio_bytes, fmt = await voice.synthesise(text)
            if fmt != "wav":
                audio_bytes = await asyncio.to_thread(voice.mp3_to_wav_bytes, audio_bytes)
            await asyncio.to_thread(voice_playback.play_wav_bytes, audio_bytes)
        else:
            import queue as _queue
            pcm_queue: _queue.Queue = _queue.Queue()

            async def _feed() -> None:
                # finally guarantees the sentinel even when synthesis fails —
                # otherwise play_pcm_stream's feeder blocks on the queue forever
                try:
                    async for chunk in voice.synthesise_stream(text):
                        pcm_queue.put(chunk)
                finally:
                    pcm_queue.put(None)

            await asyncio.gather(
                _feed(),
                asyncio.to_thread(voice_playback.play_pcm_stream, pcm_queue),
            )
    except Exception as exc:
        logger.warning("Local TTS playback failed: %s", exc)
        overlay.set_transcript("", f"[TTS failed] {text}")
        overlay.show()
    finally:
        from bot import mode as _mode
        _mode.mark_user_active()


async def _confirm_in_terminal(command_preview: str) -> bool:
    answer = await asyncio.to_thread(
        input,
        f"Confirm risky command `{command_preview}`? Type y to run: ",
    )
    return answer.strip().lower() in {"y", "yes"}



async def run_loop() -> None:
    """Run the local push-to-talk loop until interrupted."""
    load_dotenv(_ENV_PATH, override=True)
    _setup_logging()

    record_seconds = float(os.environ.get("KIRA_LOCAL_VOICE_RECORD_SECONDS", _DEFAULT_RECORD_SECONDS))
    sample_rate = int(os.environ.get("KIRA_LOCAL_VOICE_SAMPLE_RATE", _DEFAULT_SAMPLE_RATE))
    trigger = os.environ.get("KIRA_LOCAL_VOICE_TRIGGER", _DEFAULT_TRIGGER).strip().lower()
    hotkey = os.environ.get("KIRA_LOCAL_VOICE_HOTKEY", _DEFAULT_HOTKEY).strip() or _DEFAULT_HOTKEY
    config = app_control.load_apps_config()

    wake_word_name = os.environ.get("KIRA_WAKE_WORD", "hey_jarvis").strip()
    wake_word_threshold = float(os.environ.get("KIRA_WAKE_WORD_THRESHOLD", "0.5"))

    await db.init_db()
    overlay.start()
    await _prerender_cues()
    print("Kira local voice is ready.")
    logger.info("Kira local voice is ready")

    # Feature 4 + 5: start proactive and ambient loops
    from bot import proactive as _proactive
    from bot import ambient as _ambient
    _ambient.start()
    _proactive.start(speak)
    _register_project_switch_hook(speak)

    if trigger == "enter":
        await _run_enter_loop(record_seconds=record_seconds, sample_rate=sample_rate, config=config)
        return

    if trigger == "wake_word":
        await _run_wake_word_loop(
            wake_word=wake_word_name,
            threshold=wake_word_threshold,
            record_seconds=record_seconds,
            sample_rate=sample_rate,
            config=config,
        )
        return

    await _run_hotkey_loop(
        hotkey=hotkey,
        record_seconds=record_seconds,
        sample_rate=sample_rate,
        config=config,
    )


async def _run_enter_loop(
    *,
    record_seconds: float,
    sample_rate: int,
    config: app_control.AppsConfig,
    confirm: ConfirmCallback | None = None,
) -> None:
    """Run the original terminal Enter push-to-talk loop."""
    print(f"Press Enter to record {record_seconds:g}s. Press Ctrl+C to stop.")
    while True:
        await asyncio.to_thread(input, "\nPress Enter and speak...")
        await run_capture_once(
            record_seconds=record_seconds,
            sample_rate=sample_rate,
            confirm=confirm or _confirm_in_terminal,
            config=config,
        )


async def _run_hotkey_loop(
    *,
    hotkey: str,
    record_seconds: float,
    sample_rate: int,
    config: app_control.AppsConfig,
    confirm: ConfirmCallback | None = None,
) -> None:
    """Run the global hotkey push-to-talk loop."""
    try:
        import keyboard
    except ImportError:
        print("Global hotkey support needs the `keyboard` package. Falling back to Enter mode.")
        await _run_enter_loop(record_seconds=record_seconds, sample_rate=sample_rate, config=config, confirm=confirm)
        return

    loop = asyncio.get_running_loop()
    trigger_queue: asyncio.Queue[None] = asyncio.Queue(maxsize=1)

    def _on_hotkey() -> None:
        loop.call_soon_threadsafe(_queue_hotkey_trigger, trigger_queue)

    try:
        keyboard.add_hotkey(hotkey, _on_hotkey)
    except Exception as exc:
        print(f"Could not register global hotkey `{hotkey}`: {exc}")
        print("Falling back to Enter mode.")
        await _run_enter_loop(record_seconds=record_seconds, sample_rate=sample_rate, config=config, confirm=confirm)
        return

    print(f"Press {hotkey} to record. Press Ctrl+C to stop.")
    try:
        while True:
            if not is_stay_hot():
                await trigger_queue.get()
                while not trigger_queue.empty():
                    trigger_queue.get_nowait()
                await _activation_cue()
            overlay.show()
            try:
                result = await run_capture_once(
                    record_seconds=record_seconds,
                    sample_rate=sample_rate,
                    confirm=confirm or _confirm_in_terminal,
                    config=config,
                    kira_filter=is_stay_hot(),
                )
            finally:
                if not is_stay_hot():
                    overlay.hide()
    finally:
        try:
            keyboard.remove_hotkey(hotkey)
        except Exception:
            pass


async def _run_wake_word_loop(
    *,
    wake_word: str,
    threshold: float,
    record_seconds: float,
    sample_rate: int,
    config: app_control.AppsConfig,
    confirm: ConfirmCallback | None = None,
) -> None:
    """Run the wake-word triggered voice loop."""
    loop = asyncio.get_running_loop()
    trigger_queue: asyncio.Queue[None] = asyncio.Queue(maxsize=1)

    detector = wake_word_mod.start(
        trigger_queue,
        loop,
        model_name=wake_word,
        threshold=threshold,
    )
    print(f"Say '{wake_word.replace('_', ' ')}' to activate Kira. Press Ctrl+C to stop.")
    try:
        while True:
            from bot import ui_mode as _ui_mode
            in_full_mode = _ui_mode.is_full_mode()
            in_stay_hot = is_stay_hot()

            if not in_full_mode and not in_stay_hot:
                await trigger_queue.get()
                while not trigger_queue.empty():
                    trigger_queue.get_nowait()
                await _activation_cue()

            overlay.show()
            try:
                await run_capture_once(
                    record_seconds=record_seconds,
                    sample_rate=sample_rate,
                    confirm=confirm or _confirm_in_terminal,
                    config=config,
                    kira_filter=in_stay_hot and not in_full_mode,
                )
            finally:
                if not _ui_mode.is_full_mode() and not is_stay_hot():
                    overlay.hide()
    finally:
        detector.stop()


def _queue_hotkey_trigger(queue: asyncio.Queue[None]) -> None:
    """Queue one hotkey trigger, dropping extras while a capture is pending."""
    try:
        queue.put_nowait(None)
    except asyncio.QueueFull:
        print("Hotkey pressed while Kira is already busy; ignoring.")


async def start_as_task() -> None:
    """Start the voice loop as a background task from main.py.

    Skips env/logging/db setup (main.py already did those).
    Reads trigger config from environment and runs the appropriate loop.
    Risky command confirmations are routed to Telegram (no terminal available).
    """
    from bot import mode as mode_mod
    from bot import notifier

    record_seconds = float(os.environ.get("KIRA_LOCAL_VOICE_RECORD_SECONDS", _DEFAULT_RECORD_SECONDS))
    sample_rate = int(os.environ.get("KIRA_LOCAL_VOICE_SAMPLE_RATE", _DEFAULT_SAMPLE_RATE))
    trigger = os.environ.get("KIRA_LOCAL_VOICE_TRIGGER", _DEFAULT_TRIGGER).strip().lower()
    hotkey = os.environ.get("KIRA_LOCAL_VOICE_HOTKEY", _DEFAULT_HOTKEY).strip() or _DEFAULT_HOTKEY
    wake_word_name = os.environ.get("KIRA_WAKE_WORD", "hey_jarvis").strip()
    wake_word_threshold = float(os.environ.get("KIRA_WAKE_WORD_THRESHOLD", "0.5"))
    config = app_control.load_apps_config()
    await _prerender_cues()

    async def _smart_confirm(command_preview: str) -> bool:
        if mode_mod.is_autonomous():
            await speak("Check Telegram to confirm.")
            return await notifier.confirm_via_telegram(command_preview)
        return await _confirm_in_terminal(command_preview)

    confirm = _smart_confirm

    logger.info("Kira voice loop starting (trigger=%s)", trigger)

    # Feature 4 + 5: start proactive and ambient loops
    from bot import proactive as _proactive
    from bot import ambient as _ambient
    _ambient.start()
    _proactive.start(speak)
    _register_project_switch_hook(speak)

    try:
        if trigger == "enter":
            await _run_enter_loop(record_seconds=record_seconds, sample_rate=sample_rate, config=config, confirm=confirm)
        elif trigger == "wake_word":
            await _run_wake_word_loop(
                wake_word=wake_word_name,
                threshold=wake_word_threshold,
                record_seconds=record_seconds,
                sample_rate=sample_rate,
                config=config,
                confirm=confirm,
            )
        else:
            await _run_hotkey_loop(
                hotkey=hotkey,
                record_seconds=record_seconds,
                sample_rate=sample_rate,
                config=config,
                confirm=confirm,
            )
    except asyncio.CancelledError:
        logger.info("Kira voice loop cancelled")
    except Exception:
        logger.exception("Kira voice loop crashed")


def main() -> None:
    """Console entry point."""
    try:
        asyncio.run(run_loop())
    except KeyboardInterrupt:
        print("\nKira local voice stopped.")
        logger.info("Kira local voice stopped")
    except Exception:
        logger.exception("Kira local voice crashed")
        raise


def _format_command(parsed: ParsedCommand) -> str:
    return " ".join([parsed.command, *parsed.args]).strip()


def _format_process_status() -> str:
    processes = []
    for proc in psutil.process_iter(["pid", "name"]):
        name = proc.info.get("name") or ""
        if name:
            processes.append(f"{name} ({proc.info['pid']})")
        if len(processes) >= 12:
            break
    if not processes:
        return "No processes found."
    return "Running processes:\n" + "\n".join(f"- {item}" for item in processes)


def _format_sysinfo() -> str:
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    return (
        "System Info\n"
        f"CPU: {cpu}%\n"
        f"RAM: {mem.used / (1024**3):.1f} / {mem.total / (1024**3):.1f} GB ({mem.percent}%)"
    )


def _from_action_result(result: app_control.ActionResult) -> LocalVoiceResult:
    return LocalVoiceResult(ok=result.ok, message=result.message, spoken=result.spoken)


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().rstrip(".!?").split())


def _format_provider_error(exc: Exception, prefix: str) -> str:
    """Return a concise local-console error without exposing secret values."""
    status_code = getattr(exc, "status_code", None)
    class_name = type(exc).__name__
    if status_code == 401 or class_name == "AuthenticationError":
        return (
            f"{prefix}: OpenAI rejected the configured API key. "
            "Update OPENAI_API_KEY in .env, or set KIRA_API_KEY to a valid OpenAI-compatible key."
        )
    return f"{prefix}: {class_name}: {exc}"


def _setup_logging() -> None:
    """Configure console/file logging for local voice."""
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_path = Path(os.environ.get("KIRA_LOCAL_VOICE_LOG_FILE", str(_DEFAULT_LOG_PATH)))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handlers: list[logging.Handler] = [logging.FileHandler(log_path, encoding="utf-8")]
    try:
        handlers.append(logging.StreamHandler())
    except Exception:
        pass

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=handlers,
        force=True,
    )


if __name__ == "__main__":
    main()
