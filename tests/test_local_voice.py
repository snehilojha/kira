import asyncio
import os
import threading
import unittest
from unittest.mock import AsyncMock, patch

from bot import app_control
from bot import local_voice


class LocalVoiceTests(unittest.TestCase):
    def test_parse_deterministic_open_close_and_mode(self) -> None:
        config = app_control.AppsConfig(
            apps={"chrome": app_control.AppDefinition("chrome", "chrome", ["chrome.exe"])},
            modes={
                "work": app_control.ModeDefinition(
                    name="work",
                    aliases=["work mode"],
                    open_apps=["chrome"],
                )
            },
            intents={
                "open_browser": app_control.IntentDefinition(
                    name="open_browser",
                    aliases=["open browser"],
                    command="/open",
                    args=["chrome"],
                )
            }
        )

        self.assertEqual(
            local_voice.parse_deterministic("open browser", config),
            local_voice.ParsedCommand("/open", ["chrome"], "intent"),
        )
        self.assertEqual(
            local_voice.parse_deterministic("open Chrome", config),
            local_voice.ParsedCommand("/open", ["chrome"], "deterministic"),
        )
        self.assertEqual(
            local_voice.parse_deterministic("close Chrome", config),
            local_voice.ParsedCommand("/close_apps", ["chrome"], "deterministic"),
        )
        self.assertEqual(
            local_voice.parse_deterministic("work mode", config),
            local_voice.ParsedCommand("/mode_run", ["work"], "deterministic"),
        )

    def test_execute_safe_open_runs_without_confirmation(self) -> None:
        parsed = local_voice.ParsedCommand("/open", ["chrome"], "deterministic")
        confirm = AsyncMock(return_value=False)

        with patch.object(
            local_voice.app_control,
            "open_app",
            return_value=app_control.ActionResult(True, "Opening chrome...", "Opening chrome."),
        ) as open_mock:
            result = asyncio.run(local_voice.execute_command(parsed, confirm=confirm))

        open_mock.assert_called_once()
        confirm.assert_not_awaited()
        self.assertTrue(result.ok)
        self.assertEqual(result.spoken, "Opening chrome.")

    def test_parse_deterministic_desktop_control_commands(self) -> None:
        self.assertEqual(
            local_voice.parse_deterministic("arm desktop control"),
            local_voice.ParsedCommand("/desktop_arm", [], "deterministic"),
        )
        self.assertEqual(
            local_voice.parse_deterministic("move mouse 500 300"),
            local_voice.ParsedCommand("/mouse_move", ["500", "300"], "deterministic"),
        )
        self.assertEqual(
            local_voice.parse_deterministic("double click"),
            local_voice.ParsedCommand("/click", ["left", "2"], "deterministic"),
        )
        self.assertEqual(
            local_voice.parse_deterministic("scroll down 250"),
            local_voice.ParsedCommand("/scroll", ["-250"], "deterministic"),
        )
        self.assertEqual(
            local_voice.parse_deterministic("type Hello World"),
            local_voice.ParsedCommand("/type", ["Hello World"], "deterministic"),
        )
        self.assertEqual(
            local_voice.parse_deterministic("press enter"),
            local_voice.ParsedCommand("/press", ["enter"], "deterministic"),
        )
        self.assertEqual(
            local_voice.parse_deterministic("hotkey ctrl+shift+n"),
            local_voice.ParsedCommand("/hotkey", ["ctrl", "shift", "n"], "deterministic"),
        )
        self.assertEqual(
            local_voice.parse_deterministic("copy hello there"),
            local_voice.ParsedCommand("/copy", ["hello there"], "deterministic"),
        )
        self.assertEqual(
            local_voice.parse_deterministic("paste"),
            local_voice.ParsedCommand("/paste", [], "deterministic"),
        )

    def test_execute_risky_command_requires_confirmation(self) -> None:
        parsed = local_voice.ParsedCommand("/shutdown", ["1"], "llm", risky=True)
        confirm = AsyncMock(return_value=False)

        result = asyncio.run(local_voice.execute_command(parsed, confirm=confirm))

        confirm.assert_awaited_once()
        self.assertFalse(result.ok)
        self.assertIn("Confirmation denied", result.message)

    def test_handle_transcript_uses_llm_fallback_when_needed(self) -> None:
        config = app_control.AppsConfig()

        async def _run():
            with patch.object(
                local_voice,
                "parse_with_llm",
                new=AsyncMock(return_value=local_voice.ParsedCommand("/sysinfo", [], "llm")),
            ) as llm_mock:
                parsed, result = await local_voice.handle_transcript(
                    "tell me about this machine",
                    config=config,
                )
            return parsed, result, llm_mock

        parsed, result, llm_mock = asyncio.run(_run())

        llm_mock.assert_awaited_once()
        self.assertEqual(parsed.command, "/sysinfo")
        self.assertTrue(result.ok)

    def test_handle_transcript_deterministic_control_bypasses_llm(self) -> None:
        async def _run():
            with patch.object(
                local_voice,
                "parse_with_llm",
                new=AsyncMock(return_value=local_voice.ParsedCommand("/sysinfo", [], "llm")),
            ) as llm_mock, patch.object(
                local_voice.desktop_control,
                "execute_command",
                return_value=app_control.ActionResult(True, "Mouse moved", "Mouse moved."),
            ) as control_mock:
                local_voice._DESKTOP_ARM_STATE.arm(5)
                parsed, result = await local_voice.handle_transcript("move mouse 50 90")
            return parsed, result, llm_mock, control_mock

        parsed, result, llm_mock, control_mock = asyncio.run(_run())
        llm_mock.assert_not_awaited()
        control_mock.assert_called_once_with("/mouse_move", ["50", "90"])
        self.assertEqual(parsed.command, "/mouse_move")
        self.assertTrue(result.ok)

    def test_desktop_command_requires_arm(self) -> None:
        local_voice._DESKTOP_ARM_STATE.disarm()

        async def _run():
            return await local_voice.handle_transcript("click")

        parsed, result = asyncio.run(_run())
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.command, "/click")
        self.assertFalse(result.ok)
        self.assertIn("safety gate blocked", result.message)

    def test_desktop_arm_allows_next_desktop_command(self) -> None:
        local_voice._DESKTOP_ARM_STATE.disarm()

        async def _run():
            with patch.object(
                local_voice.desktop_control,
                "execute_command",
                return_value=app_control.ActionResult(True, "Clicked", "Clicked."),
            ) as control_mock:
                _, arm_result = await local_voice.handle_transcript("arm desktop control")
                parsed, action_result = await local_voice.handle_transcript("click")
            return arm_result, parsed, action_result, control_mock

        arm_result, parsed, action_result, control_mock = asyncio.run(_run())
        self.assertTrue(arm_result.ok)
        self.assertEqual(parsed.command, "/click")
        control_mock.assert_called_once_with("/click", ["left", "1"])
        self.assertTrue(action_result.ok)

    def test_handle_transcript_llm_disabled_returns_rephrase_help(self) -> None:
        async def _run():
            env = {"KIRA_LLM_FALLBACK_ENABLED": "false", "KIRA_AI_FALLBACK_ENABLED": "false"}
            with patch.dict(os.environ, env, clear=False):
                with patch.object(
                    local_voice,
                    "parse_with_llm",
                    new=AsyncMock(return_value=local_voice.ParsedCommand("/sysinfo", [], "llm")),
                ) as llm_mock:
                    parsed, result = await local_voice.handle_transcript("tell me about this machine")
            return parsed, result, llm_mock

        parsed, result, llm_mock = asyncio.run(_run())
        llm_mock.assert_not_awaited()
        self.assertIsNone(parsed)
        self.assertFalse(result.ok)
        self.assertIn("Try a deterministic command", result.message)

    def test_queue_hotkey_trigger_drops_overlapping_press(self) -> None:
        queue: asyncio.Queue[None] = asyncio.Queue(maxsize=1)

        local_voice._queue_hotkey_trigger(queue)
        local_voice._queue_hotkey_trigger(queue)

        self.assertEqual(queue.qsize(), 1)

    def test_queue_hold_trigger_drops_overlapping_press(self) -> None:
        queue: asyncio.Queue[threading.Event] = asyncio.Queue(maxsize=1)
        first = threading.Event()
        second = threading.Event()

        local_voice._queue_hold_trigger(queue, first)
        local_voice._queue_hold_trigger(queue, second)

        self.assertEqual(queue.qsize(), 1)
        self.assertTrue(second.is_set())

    def test_default_trigger_and_hotkey_are_configurable_constants(self) -> None:
        self.assertEqual(local_voice._DEFAULT_TRIGGER, "hotkey")
        self.assertEqual(local_voice._DEFAULT_HOTKEY, "ctrl+alt+k")

    def test_resolve_hotkey_behavior_falls_back_for_combo_hold(self) -> None:
        self.assertEqual(local_voice._resolve_hotkey_behavior("ctrl+alt+k", "hold"), "tap")
        self.assertEqual(local_voice._resolve_hotkey_behavior("f8", "hold"), "hold")
        self.assertEqual(local_voice._resolve_hotkey_behavior("f8", "unknown"), "tap")


if __name__ == "__main__":
    unittest.main()
