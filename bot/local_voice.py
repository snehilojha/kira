"""Local push-to-talk voice runtime for Kira.

Run with:
    python -m bot.local_voice

This is intentionally separate from the Telegram bot. V1 records one short
microphone clip after the user presses Enter, transcribes it, executes safe
local commands, and speaks a short result through the PC speakers.
"""

from __future__ import annotations

import asyncio
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
from bot import provider
from bot import voice

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"
_DEFAULT_LOG_PATH = _PROJECT_ROOT / "logs" / "local_voice.log"
_DEFAULT_SAMPLE_RATE = 16000
_DEFAULT_RECORD_SECONDS = 5.0
_DEFAULT_TRIGGER = "hotkey"
_DEFAULT_HOTKEY = "ctrl+alt+k"
ConfirmCallback = Callable[[str], Awaitable[bool]]


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
            if app_name:
                return ParsedCommand(command="/open", args=[app_name], source="deterministic")

    for prefix in ("close ", "quit ", "exit "):
        if text.startswith(prefix):
            app_name = text[len(prefix):].strip()
            if app_name:
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
        "- /open <app>\n"
        "- /close_apps <app>\n"
        "- /status\n"
        "- /sysinfo\n"
        "- /mode_run <mode>\n\n"
        "Supported risky commands, which will require terminal confirmation:\n"
        "- /shell <command>\n"
        "- /sleep\n"
        "- /shutdown <minutes>\n"
        "- /reboot <minutes>\n"
        "- /kill <pid>\n\n"
        f"Configured apps: {app_names}\n"
        f"Configured modes: {mode_names}\n"
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

    return LocalVoiceResult(
        ok=False,
        message=f"Command {parsed.command} is not supported by local voice yet.",
        spoken="That command is not supported locally yet.",
    )


async def handle_transcript(
    transcript: str,
    *,
    confirm: ConfirmCallback | None = None,
    config: app_control.AppsConfig | None = None,
) -> tuple[ParsedCommand | None, LocalVoiceResult]:
    """Parse and execute one transcript."""
    apps_config = config or app_control.load_apps_config()
    parsed = parse_deterministic(transcript, apps_config)
    if parsed is None:
        parsed = await parse_with_llm(transcript)

    if parsed is None:
        return None, LocalVoiceResult(
            ok=False,
            message="I could not map that to a local command.",
            spoken="I could not map that to a local command.",
        )

    result = await execute_command(parsed, confirm=confirm, config=apps_config)
    return parsed, result


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
) -> bytes:
    """Record from the default microphone and return WAV bytes."""
    import numpy as np
    import sounddevice as sd

    frames = int(seconds * sample_rate)
    audio = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()

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

    print(f"Press {hotkey} to record {record_seconds:g}s. Press Ctrl+C to stop.")
    try:
        while True:
            await trigger_queue.get()
            while not trigger_queue.empty():
                trigger_queue.get_nowait()
            await run_capture_once(
                record_seconds=record_seconds,
                sample_rate=sample_rate,
                confirm=_confirm_in_terminal,
                config=config,
            )
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
