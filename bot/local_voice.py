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
import wave
from dataclasses import dataclass
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
        result = app_control.run_mode(" ".join(parsed.args), apps_config)
        return _from_action_result(result)

    if command == "/open":
        result = app_control.open_app(" ".join(parsed.args), apps_config)
        return _from_action_result(result)

    if command == "/close_apps":
        result = app_control.close_apps(parsed.args, apps_config)
        return _from_action_result(result)

    if command == "/status":
        message = _format_process_status()
        return LocalVoiceResult(ok=True, message=message, spoken="Status is ready.")

    if command == "/sysinfo":
        message = _format_sysinfo()
        return LocalVoiceResult(ok=True, message=message, spoken="System info is ready.")

    desktop_result = desktop_control.execute_command(parsed.command, parsed.args)
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


def _pick_filler(transcript: str) -> str:
    """Return a short spoken filler for queries that need processing time, or empty string for instant commands."""
    normalized = _normalize_text(transcript)

    # Instant commands — no filler, they respond in under a second
    instant_prefixes = ("open ", "close ", "press ", "click", "scroll", "type ", "volume", "play pause", "mute")
    if any(normalized.startswith(p) for p in instant_prefixes):
        return ""
    if normalized in {"status", "system status", "sysinfo", "system info"}:
        return ""

    # Screen queries
    if _is_screen_query(transcript):
        return "Let me take a look."

    # Multi-step actions
    if _is_multistep(transcript):
        return "On it."

    # General knowledge / brain queries
    question_words = ("what", "who", "when", "where", "why", "how", "is ", "are ", "can ", "will ", "tell me", "search")
    if any(normalized.startswith(w) for w in question_words):
        return "Let me check."

    return ""


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


async def handle_transcript(
    transcript: str,
    *,
    confirm: ConfirmCallback | None = None,
    config: app_control.AppsConfig | None = None,
) -> tuple[ParsedCommand | None, LocalVoiceResult]:
    """Parse and execute one transcript."""
    # Memory / identity commands are intercepted before anything else.
    memory_reply = identity.extract_memory_from_transcript(transcript)
    if memory_reply:
        _command_history.append((transcript, memory_reply[:80]))
        return None, LocalVoiceResult(ok=True, message=memory_reply, spoken=memory_reply)

    if _is_identity_query(transcript):
        spoken = _build_identity_reply(transcript)
        _command_history.append((transcript, spoken[:80]))
        return None, LocalVoiceResult(ok=True, message=spoken, spoken=spoken)

    if _is_screen_query(transcript):
        description = await screen_vision.capture_screen()
        _command_history.append((transcript, description[:80]))
        return None, LocalVoiceResult(ok=True, message=description, spoken=description)

    if _is_multistep(transcript):
        result = await _handle_multistep(transcript)
        _command_history.append((transcript, result.message[:80]))
        return None, result

    apps_config = config or app_control.load_apps_config()
    parsed = parse_deterministic(transcript, apps_config)
    if parsed is None:
        parsed = await parse_with_llm(transcript)

    if parsed is None:
        result = await _handle_with_brain(transcript)
        _command_history.append((transcript, result.message[:80]))
        return None, result

    result = await execute_command(parsed, confirm=confirm, config=apps_config)
    _command_history.append((transcript, result.message[:80]))
    return parsed, result


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
            await asyncio.sleep(2.5)
            continue

        parsed = ParsedCommand(command=command, args=args, source="multistep", risky=is_risky_command(command, args))
        result = await execute_command(parsed, config=config)
        messages.append(result.message)

        if command == "/open":
            await asyncio.sleep(2.5)
        else:
            await asyncio.sleep(0.4)

    spoken = "Done." if messages else "I could not complete that."
    return LocalVoiceResult(ok=True, message="\n".join(messages), spoken=spoken)


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

    history = _history_context()
    identity_block = identity.get_identity_prompt()
    user_name = identity.get_user_name()
    system = (
        f"{identity_block}\n\n"
        "Rules for how you talk:\n"
        f"- Address the user as '{user_name}' occasionally but not every response — keep it natural.\n"
        "- Speak naturally, like a real person. Use contractions. Don't be formal.\n"
        "- Never start a sentence with 'Certainly', 'Sure', 'Of course', 'Absolutely', or 'I'.\n"
        "- Match response length to the question. Simple = one or two sentences. Complex = short paragraph max.\n"
        "- No markdown. No bullet points. No headers. Your response is read aloud via TTS.\n"
        "- Be direct and confident. Don't hedge with 'I think' or 'it seems like' unless genuinely uncertain.\n"
        "- If you don't know something or it requires current data, use web_search — don't say you can't look it up.\n"
        "- Never mention that you're an AI or that you have limitations unless directly asked.\n"
        "- If screen context is provided, use it to give more relevant answers.\n"
        "Always use web_search for current events, news, weather, prices, sports scores, or anything time-sensitive."
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

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]

    try:
        for _ in range(3):
            response = await client.chat.completions.create(
                model=config.smart_model,
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


async def run_capture_once(
    *,
    record_seconds: float = _DEFAULT_RECORD_SECONDS,
    sample_rate: int = _DEFAULT_SAMPLE_RATE,
    confirm: ConfirmCallback | None = None,
    config: app_control.AppsConfig | None = None,
) -> LocalVoiceResult:
    """Record, transcribe, execute, and speak one local voice command."""
    apps_config = config or app_control.load_apps_config()
    print("Recording...")
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
        await speak(result.spoken)
        return result

    print("Transcribing...")
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
        await speak(result.spoken)
        return result
    print(f"Heard: {transcript}")
    filler = _pick_filler(transcript)
    if filler:
        await speak(filler)

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
        await speak(result.spoken)
        return result

    if parsed is not None:
        print(f"Command: {_format_command(parsed)} [{parsed.source}]")
    print(result.message)
    await speak(result.spoken)
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


def play_wav_bytes(audio_bytes: bytes) -> None:
    """Play WAV bytes through the default output device."""
    import numpy as np
    import sounddevice as sd

    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    audio = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels)
    sd.play(audio, sample_rate)
    sd.wait()


async def speak(text: str) -> None:
    """Speak a short local response, falling back to console only on failure."""
    try:
        audio = await voice.synthesise(text, response_format="wav")
        play_wav_bytes(audio)
    except Exception as exc:
        logger.warning("Local TTS playback failed: %s", exc)


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

    await db.init_db()
    overlay.start()
    print("Kira local voice is ready.")
    logger.info("Kira local voice is ready")
    if trigger == "enter":
        await _run_enter_loop(record_seconds=record_seconds, sample_rate=sample_rate, config=config)
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
) -> None:
    """Run the original terminal Enter push-to-talk loop."""
    print(f"Press Enter to record {record_seconds:g}s. Press Ctrl+C to stop.")
    while True:
        await asyncio.to_thread(input, "\nPress Enter and speak...")
        await run_capture_once(
            record_seconds=record_seconds,
            sample_rate=sample_rate,
            confirm=_confirm_in_terminal,
            config=config,
        )


async def _run_hotkey_loop(
    *,
    hotkey: str,
    record_seconds: float,
    sample_rate: int,
    config: app_control.AppsConfig,
) -> None:
    """Run the global hotkey push-to-talk loop."""
    try:
        import keyboard
    except ImportError:
        print("Global hotkey support needs the `keyboard` package. Falling back to Enter mode.")
        await _run_enter_loop(record_seconds=record_seconds, sample_rate=sample_rate, config=config)
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
        await _run_enter_loop(record_seconds=record_seconds, sample_rate=sample_rate, config=config)
        return

    print(f"Press {hotkey} to record. Press Ctrl+C to stop.")
    try:
        while True:
            await trigger_queue.get()
            while not trigger_queue.empty():
                trigger_queue.get_nowait()
            overlay.show()
            try:
                await run_capture_once(
                    record_seconds=record_seconds,
                    sample_rate=sample_rate,
                    confirm=_confirm_in_terminal,
                    config=config,
                )
            finally:
                overlay.hide()
    finally:
        try:
            keyboard.remove_hotkey(hotkey)
        except Exception:
            pass


def _queue_hotkey_trigger(queue: asyncio.Queue[None]) -> None:
    """Queue one hotkey trigger, dropping extras while a capture is pending."""
    try:
        queue.put_nowait(None)
    except asyncio.QueueFull:
        print("Hotkey pressed while Kira is already busy; ignoring.")


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
