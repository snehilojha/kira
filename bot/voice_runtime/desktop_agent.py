"""Vision-driven desktop control: the LLM sees a screenshot and drives pyautogui."""

from __future__ import annotations

import asyncio
import json
import logging
import os

from bot import provider
from bot.voice_runtime import session
from bot.voice_runtime.models import LocalVoiceResult
from bot.voice_runtime.windows_focus import _find_hwnd_from_transcript, _restore_foreground

logger = logging.getLogger(__name__)


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
    target_hwnd = _find_hwnd_from_transcript(transcript) or session.get_last_user_hwnd()
    await asyncio.to_thread(_restore_foreground, target_hwnd)
    if target_hwnd:
        await asyncio.sleep(0.3)  # let the OS switch before grabbing screen

    # Take a screenshot so the LLM can see what's on screen
    try:
        png_bytes = await asyncio.to_thread(_get_screenshot_png)
        screenshot_b64 = base64.b64encode(png_bytes).decode() if png_bytes else None
    except Exception:
        logger.warning("Initial screenshot failed — desktop agent is running blind", exc_info=True)
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
                        logger.debug("read_screen screenshot failed", exc_info=True)
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
                logger.debug("End-of-round screenshot failed", exc_info=True)

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
