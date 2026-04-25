import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bot import app_control


class AppControlTests(unittest.TestCase):
    def test_load_apps_config_reads_apps_and_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "apps.toml"
            path.write_text(
                """
[apps.chrome]
open = "chrome"
close = ["chrome.exe"]

[modes.work]
aliases = ["work mode"]
open = ["chrome"]
close = []
say = "Work mode is ready."

[intents.open_browser]
aliases = ["open browser"]
command = "/open"
args = ["chrome"]
""".strip(),
                encoding="utf-8",
            )

            config = app_control.load_apps_config(path)

        self.assertEqual(config.apps["chrome"].open_command, "chrome")
        self.assertEqual(config.apps["chrome"].close_names, ["chrome.exe"])
        self.assertEqual(config.modes["work"].aliases, ["work mode"])
        self.assertEqual(config.modes["work"].open_apps, ["chrome"])
        self.assertEqual(config.intents["open_browser"].command, "/open")
        self.assertEqual(config.intents["open_browser"].args, ["chrome"])

    def test_find_intent_matches_alias(self) -> None:
        config = app_control.AppsConfig(
            intents={
                "work_setup": app_control.IntentDefinition(
                    name="work_setup",
                    aliases=["start coding setup"],
                    command="/mode_run",
                    args=["work"],
                )
            }
        )

        intent = app_control.find_intent("start coding setup", config)

        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.command, "/mode_run")
        self.assertEqual(intent.args, ["work"])

    def test_open_configured_app_launches_command(self) -> None:
        config = app_control.AppsConfig(
            apps={
                "vscode": app_control.AppDefinition(
                    name="vscode",
                    open_command="code D:/VS_adv_python/kira",
                    close_names=["Code.exe"],
                )
            }
        )

        with patch.object(app_control, "_matching_processes", return_value=[]), patch.object(
            app_control.subprocess, "Popen"
        ) as popen_mock:
            result = app_control.open_app("vscode", config)

        popen_mock.assert_called_once_with(
            ["cmd", "/c", "start", "", "code", "D:/VS_adv_python/kira"],
            stdout=app_control.subprocess.DEVNULL,
            stderr=app_control.subprocess.DEVNULL,
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.spoken, "Opening vscode.")

    def test_open_configured_app_focuses_existing_window(self) -> None:
        config = app_control.AppsConfig(
            apps={
                "chrome": app_control.AppDefinition(
                    name="chrome",
                    open_command="chrome",
                    close_names=["chrome.exe"],
                )
            }
        )

        class _Proc:
            pid = 777
            info = {"name": "chrome.exe"}

        with patch.object(app_control, "_matching_processes", return_value=[_Proc()]) as match_mock, patch.object(
            app_control, "_focus_window_for_pid", return_value=True
        ) as focus_mock, patch.object(app_control.subprocess, "Popen") as popen_mock:
            result = app_control.open_app("chrome", config)

        match_mock.assert_called_once_with(["chrome.exe"])
        focus_mock.assert_called_once_with(777)
        popen_mock.assert_not_called()
        self.assertTrue(result.ok)
        self.assertIn("already open", result.message)

    def test_close_configured_app_uses_close_names(self) -> None:
        config = app_control.AppsConfig(
            apps={
                "chrome": app_control.AppDefinition(
                    name="chrome",
                    open_command="chrome",
                    close_names=["chrome.exe"],
                )
            }
        )

        class _Proc:
            pid = 123
            info = {"name": "chrome.exe"}

            def terminate(self):
                return None

        with patch.object(app_control, "_matching_processes", return_value=[_Proc()]) as match_mock:
            result = app_control.close_apps(["chrome"], config)

        match_mock.assert_called_once_with(["chrome.exe"])
        self.assertTrue(result.ok)
        self.assertIn("closed chrome.exe", result.message)

    def test_run_mode_opens_configured_apps(self) -> None:
        config = app_control.AppsConfig(
            apps={
                "chrome": app_control.AppDefinition("chrome", "chrome", ["chrome.exe"]),
                "claude": app_control.AppDefinition("claude", "claude", ["Claude.exe"]),
            },
            modes={
                "work": app_control.ModeDefinition(
                    name="work",
                    aliases=["work mode"],
                    open_apps=["chrome", "claude"],
                    say="Work mode is ready.",
                )
            },
        )

        with patch.object(app_control, "_matching_processes", return_value=[]), patch.object(
            app_control.subprocess, "Popen"
        ) as popen_mock:
            result = app_control.run_mode("work mode", config)

        self.assertEqual(popen_mock.call_count, 2)
        self.assertTrue(result.ok)
        self.assertEqual(result.spoken, "Work mode is ready.")

    def test_run_mode_chain_executes_chained_modes(self) -> None:
        config = app_control.AppsConfig(
            apps={
                "chrome": app_control.AppDefinition("chrome", "chrome", ["chrome.exe"]),
                "claude": app_control.AppDefinition("claude", "claude", ["Claude.exe"]),
            },
            modes={
                "work": app_control.ModeDefinition(
                    name="work",
                    aliases=["work mode"],
                    open_apps=["chrome"],
                    chain_modes=["assistant"],
                    say="Work mode is ready.",
                ),
                "assistant": app_control.ModeDefinition(
                    name="assistant",
                    aliases=["assistant mode"],
                    open_apps=["claude"],
                    say="Assistant mode is ready.",
                ),
            },
        )

        with patch.object(app_control, "_matching_processes", return_value=[]), patch.object(
            app_control.subprocess, "Popen"
        ) as popen_mock:
            result = app_control.run_mode("work", config)

        self.assertEqual(popen_mock.call_count, 2)
        self.assertTrue(result.ok)
        self.assertIn("Mode: assistant", result.message)


if __name__ == "__main__":
    unittest.main()
