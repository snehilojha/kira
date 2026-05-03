import unittest
from unittest.mock import Mock, patch

from bot import desktop_control


class DesktopControlTests(unittest.TestCase):
    def test_move_mouse_dispatches_to_pyautogui(self) -> None:
        pyauto = Mock()
        with patch.object(desktop_control, "_require_pyautogui", return_value=pyauto):
            result = desktop_control.execute_command("/mouse_move", ["100", "200"])
        assert result is not None
        pyauto.moveTo.assert_called_once_with(100, 200)
        self.assertTrue(result.ok)

    def test_click_dispatches_to_pyautogui(self) -> None:
        pyauto = Mock()
        with patch.object(desktop_control, "_require_pyautogui", return_value=pyauto):
            result = desktop_control.execute_command("/click", ["right", "1"])
        assert result is not None
        pyauto.click.assert_called_once_with(button="right", clicks=1)
        self.assertTrue(result.ok)

    def test_hotkey_requires_multiple_keys(self) -> None:
        result = desktop_control.execute_command("/hotkey", ["ctrl"])
        assert result is not None
        self.assertFalse(result.ok)
        self.assertIn("Usage: hotkey", result.message)

    def test_copy_uses_clipboard(self) -> None:
        with patch("pyperclip.copy") as copy_mock:
            result = desktop_control.execute_command("/copy", ["hello world"])
        assert result is not None
        copy_mock.assert_called_once_with("hello world")
        self.assertTrue(result.ok)


if __name__ == "__main__":
    unittest.main()
