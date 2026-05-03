"""Deterministic local desktop control helpers for Kira local voice."""

from __future__ import annotations

from bot import app_control


def execute_command(command: str, args: list[str]) -> app_control.ActionResult | None:
    """Execute one deterministic desktop control command."""
    normalized = command.strip().lower()

    if normalized == "/mouse_move":
        return _move_mouse(args)
    if normalized == "/click":
        return _click(args)
    if normalized == "/scroll":
        return _scroll(args)
    if normalized == "/type":
        return _type_text(args)
    if normalized == "/press":
        return _press(args)
    if normalized == "/hotkey":
        return _hotkey(args)
    if normalized == "/copy":
        return _copy(args)
    if normalized == "/paste":
        return _paste(args)
    return None


def _move_mouse(args: list[str]) -> app_control.ActionResult:
    if len(args) != 2:
        return app_control.ActionResult(
            ok=False,
            message="Usage: move mouse <x> <y>",
            spoken="I need X and Y coordinates.",
        )
    try:
        x = int(args[0])
        y = int(args[1])
    except ValueError:
        return app_control.ActionResult(
            ok=False,
            message="Mouse coordinates must be integers.",
            spoken="Mouse coordinates must be numbers.",
        )

    pyautogui = _require_pyautogui()
    pyautogui.moveTo(x, y)
    return app_control.ActionResult(
        ok=True,
        message=f"Moved mouse to ({x}, {y}).",
        spoken="Mouse moved.",
    )


def _click(args: list[str]) -> app_control.ActionResult:
    button = args[0] if args else "left"
    clicks_raw = args[1] if len(args) > 1 else "1"
    if button not in {"left", "right", "middle"}:
        return app_control.ActionResult(
            ok=False,
            message=f"Unsupported click button: {button}",
            spoken="That click button is not supported.",
        )
    try:
        clicks = int(clicks_raw)
    except ValueError:
        return app_control.ActionResult(
            ok=False,
            message="Click count must be an integer.",
            spoken="Click count must be a number.",
        )
    if clicks < 1:
        return app_control.ActionResult(
            ok=False,
            message="Click count must be at least 1.",
            spoken="Click count must be at least one.",
        )

    pyautogui = _require_pyautogui()
    pyautogui.click(button=button, clicks=clicks)
    return app_control.ActionResult(
        ok=True,
        message=f"Clicked {button} button ({clicks}x).",
        spoken="Click complete.",
    )


def _scroll(args: list[str]) -> app_control.ActionResult:
    if len(args) != 1:
        return app_control.ActionResult(
            ok=False,
            message="Usage: scroll <amount>",
            spoken="I need a scroll amount.",
        )
    try:
        amount = int(args[0])
    except ValueError:
        return app_control.ActionResult(
            ok=False,
            message="Scroll amount must be an integer.",
            spoken="Scroll amount must be a number.",
        )
    pyautogui = _require_pyautogui()
    pyautogui.scroll(amount)
    direction = "up" if amount > 0 else "down"
    return app_control.ActionResult(
        ok=True,
        message=f"Scrolled {direction} by {abs(amount)}.",
        spoken="Scroll complete.",
    )


def _type_text(args: list[str]) -> app_control.ActionResult:
    text = " ".join(args).strip()
    if not text:
        return app_control.ActionResult(
            ok=False,
            message="Usage: type <text>",
            spoken="I need text to type.",
        )
    pyautogui = _require_pyautogui()
    pyautogui.typewrite(text)
    return app_control.ActionResult(
        ok=True,
        message=f"Typed {len(text)} characters.",
        spoken="Typed.",
    )


def _press(args: list[str]) -> app_control.ActionResult:
    if len(args) != 1:
        return app_control.ActionResult(
            ok=False,
            message="Usage: press <key>",
            spoken="I need one key to press.",
        )
    key = args[0].strip().lower()
    if not key:
        return app_control.ActionResult(
            ok=False,
            message="Key cannot be empty.",
            spoken="I need one key to press.",
        )
    pyautogui = _require_pyautogui()
    pyautogui.press(key)
    return app_control.ActionResult(
        ok=True,
        message=f"Pressed {key}.",
        spoken="Key pressed.",
    )


def _hotkey(args: list[str]) -> app_control.ActionResult:
    keys = [key.strip().lower() for key in args if key.strip()]
    if len(keys) < 2:
        return app_control.ActionResult(
            ok=False,
            message="Usage: hotkey <key1>+<key2>[+key3...]",
            spoken="I need at least two keys for a hotkey.",
        )
    pyautogui = _require_pyautogui()
    pyautogui.hotkey(*keys)
    return app_control.ActionResult(
        ok=True,
        message=f"Sent hotkey: {'+'.join(keys)}.",
        spoken="Hotkey sent.",
    )


def _copy(args: list[str]) -> app_control.ActionResult:
    text = " ".join(args)
    import pyperclip

    pyperclip.copy(text)
    return app_control.ActionResult(
        ok=True,
        message=f"Copied {len(text)} characters to clipboard.",
        spoken="Copied.",
    )


def _paste(args: list[str]) -> app_control.ActionResult:
    pyautogui = _require_pyautogui()
    pyautogui.hotkey("ctrl", "v")
    return app_control.ActionResult(
        ok=True,
        message="Pasted clipboard contents.",
        spoken="Pasted.",
    )


def _require_pyautogui():
    try:
        import pyautogui  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pyautogui is not installed. Run: pip install pyautogui") from exc
    return pyautogui
