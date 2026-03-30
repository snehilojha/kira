"""Tests for `/ask` prompt context injection helpers."""

from __future__ import annotations

import asyncio
import sys
import types
import tempfile
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

_fake_telegram = types.ModuleType("telegram")
_fake_telegram_ext = types.ModuleType("telegram.ext")


class _FakeUpdate:  # pragma: no cover - test shim
    """Minimal Update placeholder for importing handlers in tests."""


class _FakeInlineKeyboardButton:  # pragma: no cover - test shim
    """Minimal InlineKeyboardButton placeholder for importing handlers in tests."""

    def __init__(self, *args, **kwargs) -> None:
        pass


class _FakeInlineKeyboardMarkup:  # pragma: no cover - test shim
    """Minimal InlineKeyboardMarkup placeholder for importing handlers in tests."""

    def __init__(self, *args, **kwargs) -> None:
        pass


class _FakeContextTypes:  # pragma: no cover - test shim
    """Minimal ContextTypes placeholder for importing handlers in tests."""

    DEFAULT_TYPE = object()


_fake_telegram.Update = _FakeUpdate
_fake_telegram.InlineKeyboardButton = _FakeInlineKeyboardButton
_fake_telegram.InlineKeyboardMarkup = _FakeInlineKeyboardMarkup
_fake_telegram_ext.ContextTypes = _FakeContextTypes

_fake_mss = types.ModuleType("mss")
_fake_mss_tools = types.ModuleType("mss.tools")
_fake_mss.tools = _fake_mss_tools
_fake_mss_tools.to_png = lambda *args, **kwargs: b""  # pragma: no cover - test shim

_fake_pyperclip = types.ModuleType("pyperclip")
_fake_pyperclip.copy = lambda text: None  # pragma: no cover - test shim
_fake_pyperclip.paste = lambda: ""  # pragma: no cover - test shim

_fake_toml = types.ModuleType("toml")
_fake_toml.load = lambda path: {}  # pragma: no cover - test shim

_fake_httpx = types.ModuleType("httpx")


class _FakeHTTPError(Exception):  # pragma: no cover - test shim
    """Minimal httpx.HTTPError replacement."""


class _FakeAsyncClient:  # pragma: no cover - test shim
    """Minimal httpx.AsyncClient replacement."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        return types.SimpleNamespace(status_code=200, text="OK")


_fake_httpx.HTTPError = _FakeHTTPError
_fake_httpx.AsyncClient = _FakeAsyncClient

_fake_openai = types.ModuleType("openai")


class _FakeAsyncOpenAI:  # pragma: no cover - test shim
    """Minimal AsyncOpenAI placeholder used only for imports."""

    def __init__(self, *args, **kwargs) -> None:
        self.audio = types.SimpleNamespace(
            transcriptions=types.SimpleNamespace(create=self._noop_async),
            speech=types.SimpleNamespace(create=self._noop_async),
        )
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._noop_async)
        )

    async def _noop_async(self, *args, **kwargs):
        return types.SimpleNamespace(text="", choices=[])


_fake_openai.AsyncOpenAI = _FakeAsyncOpenAI

sys.modules.setdefault("telegram", _fake_telegram)
sys.modules.setdefault("telegram.ext", _fake_telegram_ext)
sys.modules.setdefault("mss", _fake_mss)
sys.modules.setdefault("mss.tools", _fake_mss_tools)
sys.modules.setdefault("pyperclip", _fake_pyperclip)
sys.modules.setdefault("toml", _fake_toml)
sys.modules.setdefault("httpx", _fake_httpx)
sys.modules.setdefault("openai", _fake_openai)

from bot import handlers


class AskContextTests(unittest.TestCase):
    """Verify live session state and project context are injected correctly."""

    def setUp(self) -> None:
        """Reset module-level context buffers before each test."""
        self._original_context_path = handlers._PROJECT_CONTEXT_PATH
        self._original_recent_output = handlers._RECENT_OUTPUT_LINES
        handlers._RECENT_OUTPUT_LINES = deque(maxlen=20)

    def tearDown(self) -> None:
        """Restore module-level state after each test."""
        handlers._PROJECT_CONTEXT_PATH = self._original_context_path
        handlers._RECENT_OUTPUT_LINES = self._original_recent_output

    def test_load_project_context_reads_and_trims_file(self) -> None:
        """Project context should be loaded from the configured file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "context.md"
            path.write_text("project notes\n", encoding="utf-8")
            handlers._PROJECT_CONTEXT_PATH = path

            self.assertEqual(handlers._load_project_context(), "project notes")

    def test_build_ask_system_prompt_includes_project_and_live_context(self) -> None:
        """The system prompt should include project context and live state snapshots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "context.md"
            path.write_text("active project: kira\npriority: test", encoding="utf-8")
            handlers._PROJECT_CONTEXT_PATH = path
            handlers._RECENT_OUTPUT_LINES.append("training started")

            with patch.object(handlers.process_registry, "list_processes", return_value=[{"pid": 11, "alias": "train", "runtime_seconds": 65, "returncode": None}]), patch.object(handlers.scheduler, "list_schedules", return_value=[{"id": "sched-1", "alias": "eval", "run_at": "2026-03-26T12:00:00"}]), patch.object(handlers.watchdog, "list_watches", return_value=[{"id": "watch-1", "type": "pid", "target": "11", "label": "train"}]), patch.object(handlers.psutil, "cpu_percent", return_value=12.5), patch.object(handlers.psutil, "virtual_memory", return_value=type("vm", (), {"percent": 33.3})()), patch("bot.handlers._format_runtime", return_value="1m 5s"):
                prompt = handlers._build_ask_system_prompt()

        self.assertIn("active project: kira", prompt)
        self.assertIn("Running processes:", prompt)
        self.assertIn("PID 11: train", prompt)
        self.assertIn("Pending schedules:", prompt)
        self.assertIn("watch-1: pid -> 11", prompt)
        self.assertIn("Recent command output:", prompt)
        self.assertIn("training started", prompt)
        self.assertIn("System snapshot: CPU 12.5% | RAM 33.3%", prompt)

    def test_recent_output_tail_truncates_old_content(self) -> None:
        """Recent output should be capped so the prompt stays bounded."""
        handlers._RECENT_OUTPUT_LINES.extend(["x" * 1000, "y" * 1000, "z" * 1000])
        tail = handlers._get_recent_output_tail()

        self.assertIn("[...recent output truncated...]", tail)
        self.assertLessEqual(len(tail), handlers._RECENT_OUTPUT_MAX_CHARS + 100)

    def test_build_ask_confirmation_text(self) -> None:
        """Confirmation text should preserve the exact command and args."""
        text = handlers._build_ask_confirmation_text("/run", ["crypto_train_full", "--fee_mult", "10"])
        self.assertEqual(text, "Proposed command:\n`/run crypto_train_full --fee_mult 10`\n\nExecute?")

    def test_normalize_app_name_rejects_path_like_input(self) -> None:
        """Path-like values should be rejected because /open is app-name only."""
        self.assertEqual(handlers._normalize_app_name("  notepad  "), "notepad")
        self.assertIsNone(handlers._normalize_app_name(r"C:\\Windows\\notepad.exe"))
        self.assertIsNone(handlers._normalize_app_name(r"..\\notepad"))

    def test_open_app_by_name_launches_name_only(self) -> None:
        """The launcher should call subprocess with the sanitized app name."""
        with patch.object(handlers.subprocess, "Popen") as popen_mock:
            result = handlers._open_app_by_name(" notepad ")

        popen_mock.assert_called_once_with(
            ["cmd", "/c", "start", "", "notepad"],
            stdout=handlers.subprocess.DEVNULL,
            stderr=handlers.subprocess.DEVNULL,
        )
        self.assertEqual(result, "✅ Opening notepad...")

    def test_open_app_by_name_rejects_path_like_input(self) -> None:
        """Path-like inputs should return usage text instead of launching anything."""
        with patch.object(handlers.subprocess, "Popen") as popen_mock:
            result = handlers._open_app_by_name(r"C:\\Windows\\notepad.exe")

        popen_mock.assert_not_called()
        self.assertEqual(result, "Usage: /open <app name>\nExample: /open notepad")

    def test_handle_ask_callback_executes_open(self) -> None:
        """The /ask confirmation callback should support the /open command."""

        replies: list[str] = []

        class _FakeMessage:
            async def reply_text(self, text: str, **kwargs):
                replies.append(text)

        class _FakeQuery:
            def __init__(self) -> None:
                self.data = handlers._encode_ask_cb("/open", ["notepad"])
                self.message = _FakeMessage()

            async def answer(self):
                return None

            async def edit_message_text(self, text: str, **kwargs):
                replies.append(text)

        update = types.SimpleNamespace(callback_query=_FakeQuery())
        context = types.SimpleNamespace()

        with patch.object(handlers, "_open_app_by_name", return_value="✅ Opening notepad...") as open_mock:
            asyncio.run(handlers.handle_ask_callback(update, context))

        open_mock.assert_called_once_with("notepad")
        self.assertIn("Executing: `/open notepad`", replies[0])
        self.assertIn("✅ Opening notepad...", replies[-1])


if __name__ == "__main__":
    unittest.main()
