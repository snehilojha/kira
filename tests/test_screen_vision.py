"""Tests for the V1.5 trigger-based screen vision module."""

from __future__ import annotations

import asyncio
import os
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("KIRA_DB_PATH", ":memory:")

from bot import db
import bot.screen_vision as sv_module
from bot.screen_vision import (
    _build_trigger_prompt,
    _is_actionable,
    capture_and_analyse,
    notify_if_actionable,
)


def _reset_cooldown() -> None:
    """Clear per-trigger cooldown state between tests."""
    sv_module._last_fired.clear()


class TestTriggerPrompts(unittest.TestCase):
    """_build_trigger_prompt returns focused questions per trigger type."""

    def test_stdin_silent_mentions_input(self) -> None:
        prompt = _build_trigger_prompt("stdin_silent", "python train.py")
        self.assertIn("waiting for", prompt.lower())
        self.assertIn("input", prompt.lower())

    def test_dialog_appeared_mentions_dialog(self) -> None:
        prompt = _build_trigger_prompt("dialog_appeared", "pip install")
        self.assertTrue(
            "dialog" in prompt.lower() or "modal" in prompt.lower()
        )

    def test_process_frozen_mentions_frozen(self) -> None:
        prompt = _build_trigger_prompt("process_frozen", "crypto_bot")
        self.assertTrue(
            "frozen" in prompt.lower() or "stuck" in prompt.lower()
        )

    def test_cursor_ai_stalled_mentions_cursor(self) -> None:
        prompt = _build_trigger_prompt("cursor_ai_stalled", "")
        self.assertIn("cursor", prompt.lower())

    def test_process_label_included_in_prompt(self) -> None:
        prompt = _build_trigger_prompt("stdin_silent", "my_special_script")
        self.assertIn("my_special_script", prompt)


class TestIsActionable(unittest.TestCase):
    """_is_actionable classifies vision model responses correctly."""

    def test_yes_prefix_is_actionable(self) -> None:
        self.assertTrue(_is_actionable("Yes, there is a dialog waiting."))

    def test_no_prefix_is_not_actionable(self) -> None:
        self.assertFalse(_is_actionable("No issues detected."))

    def test_waiting_keyword_is_actionable(self) -> None:
        self.assertTrue(_is_actionable("The terminal appears to be waiting for input."))

    def test_empty_string_is_not_actionable(self) -> None:
        self.assertFalse(_is_actionable(""))

    def test_frozen_keyword_is_actionable(self) -> None:
        self.assertTrue(_is_actionable("The application looks frozen."))


class TestCaptureAndAnalyse(unittest.TestCase):
    """capture_and_analyse integrates screenshot + vision model."""

    def setUp(self) -> None:
        _reset_cooldown()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(db.init_db(":memory:"))

    def tearDown(self) -> None:
        self.loop.run_until_complete(db.close_db())
        self.loop.close()
        asyncio.set_event_loop(None)

    def _run(self, coro):
        return self.loop.run_until_complete(coro)

    def test_returns_model_interpretation(self) -> None:
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Yes, there is a dialog waiting."

        with patch.object(sv_module, "_take_screenshot", return_value="fakeb64data"):
            with patch("bot.provider.create_vision_completion", new_callable=AsyncMock,
                       return_value=mock_response):
                result = self._run(capture_and_analyse("dialog_appeared", "pip"))

        self.assertEqual(result, "Yes, there is a dialog waiting.")

    def test_logs_to_db(self) -> None:
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Yes, frozen."

        with patch.object(sv_module, "_take_screenshot", return_value="fakeb64"):
            with patch("bot.provider.create_vision_completion", new_callable=AsyncMock,
                       return_value=mock_response):
                self._run(capture_and_analyse("process_frozen", "train"))

        triggers = self._run(db.get_recent_vision_triggers(1))
        self.assertEqual(len(triggers), 1)
        self.assertEqual(triggers[0]["trigger_type"], "process_frozen")
        self.assertIn("frozen", triggers[0]["interpretation"].lower())

    def test_screenshot_unavailable_returns_message(self) -> None:
        with patch.object(sv_module, "_take_screenshot", return_value=""):
            result = self._run(capture_and_analyse("stdin_silent", ""))
        self.assertIn("unavailable", result.lower())


class TestNotifyIfActionable(unittest.TestCase):
    """notify_if_actionable sends or skips notifications correctly."""

    def setUp(self) -> None:
        _reset_cooldown()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(db.init_db(":memory:"))

    def tearDown(self) -> None:
        self.loop.run_until_complete(db.close_db())
        self.loop.close()
        asyncio.set_event_loop(None)
        _reset_cooldown()

    def _run(self, coro):
        return self.loop.run_until_complete(coro)

    def test_notifies_when_actionable(self) -> None:
        sent: list[str] = []

        async def _fake_capture(trigger, process_label=""):
            return "Yes, there is a dialog."

        async def _fake_send(msg):
            sent.append(msg)

        with patch.object(sv_module, "capture_and_analyse", side_effect=_fake_capture):
            with patch("bot.notifier.send", side_effect=_fake_send):
                self._run(notify_if_actionable("dialog_appeared", "pip"))

        self.assertEqual(len(sent), 1)
        self.assertIn("dialog_appeared", sent[0])

    def test_no_notify_when_not_actionable(self) -> None:
        sent: list[str] = []

        async def _fake_capture(trigger, process_label=""):
            return "No issues detected on screen."

        async def _fake_send(msg):
            sent.append(msg)

        with patch.object(sv_module, "capture_and_analyse", side_effect=_fake_capture):
            with patch("bot.notifier.send", side_effect=_fake_send):
                self._run(notify_if_actionable("process_frozen", "crypto_bot"))

        self.assertEqual(len(sent), 0)

    def test_cooldown_prevents_repeat_notification(self) -> None:
        sent: list[str] = []

        async def _fake_capture(trigger, process_label=""):
            return "Yes, dialog present."

        async def _fake_send(msg):
            sent.append(msg)

        sv_module._last_fired["dialog_appeared"] = time.monotonic()

        with patch.object(sv_module, "capture_and_analyse", side_effect=_fake_capture):
            with patch("bot.notifier.send", side_effect=_fake_send):
                self._run(notify_if_actionable("dialog_appeared", "pip"))

        self.assertEqual(len(sent), 0)


if __name__ == "__main__":
    unittest.main()
