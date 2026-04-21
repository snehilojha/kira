"""Trigger-based screen vision for Kira V1.5.

Only fires when the observer detects a specific ambiguity signal — never runs
continuously. Each trigger takes a screenshot, asks the vision model a
focused yes/no question, and notifies via Telegram if action is needed.

Triggers
--------
stdin_silent       — terminal process is open but has produced no output for a
                     while (stdin may be blocked waiting for input)
dialog_appeared    — a new modal window / dialog was detected by the observer
process_frozen     — a tracked process has been alive but silent too long
cursor_ai_stalled  — a Cursor AI task appears to have stopped mid-response

Spam protection
---------------
Each trigger type has a 10-minute cooldown (module-level dict). The observer
is responsible for calling ``notify_if_actionable`` — it should not call it
more often than the observer cycle.

Public API
----------
capture_and_analyse(trigger, process_label)  — screenshot → vision model → str
notify_if_actionable(trigger, process_label) — wraps capture_and_analyse,
                                               sends notifier.send() if needed
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from typing import Literal

logger = logging.getLogger(__name__)

# ── Types ─────────────────────────────────────────────────────────

TriggerType = Literal[
    "stdin_silent",
    "dialog_appeared",
    "process_frozen",
    "cursor_ai_stalled",
]

# ── Cooldown state ────────────────────────────────────────────────

_COOLDOWN_SECONDS = 600.0  # 10 minutes per trigger type
_last_fired: dict[str, float] = {}


# ── Public API ────────────────────────────────────────────────────

async def capture_and_analyse(
    trigger: TriggerType,
    process_label: str = "",
) -> str:
    """Take a screenshot and ask the vision model if action is needed.

    Args:
        trigger: The ambiguity signal that caused this check.
        process_label: Human-readable label for the process or context.

    Returns:
        The vision model's plain-text interpretation.
    """
    image_b64 = await asyncio.to_thread(_take_screenshot)
    if not image_b64:
        return "Screenshot unavailable."

    prompt = _build_trigger_prompt(trigger, process_label)

    try:
        from bot import provider
        response = await provider.create_vision_completion(
            prompt=prompt,
            image_b64=image_b64,
            max_tokens=200,
        )
        interpretation = (response.choices[0].message.content or "").strip()
    except RuntimeError:
        # Provider not configured (e.g. in tests without env vars)
        interpretation = "Vision model unavailable."
    except Exception as exc:
        logger.warning("Vision model call failed for trigger %s: %s", trigger, exc)
        interpretation = f"Vision analysis failed: {exc}"

    # Persist the trigger event regardless of whether we notify.
    try:
        from bot import db
        await db.log_vision_trigger(
            trigger_type=trigger,
            process_label=process_label,
            interpretation=interpretation,
            notified=False,
        )
    except Exception as exc:
        logger.debug("Failed to log vision trigger to DB: %s", exc)

    logger.info(
        "Screen vision [%s] process=%r → %r",
        trigger, process_label, interpretation[:120],
    )
    return interpretation


async def notify_if_actionable(
    trigger: TriggerType,
    process_label: str = "",
) -> None:
    """Capture, analyse, and send a Telegram notification if action is needed.

    Respects the per-trigger 10-minute cooldown so the user is not spammed if
    the observer keeps detecting the same condition every cycle.

    Args:
        trigger: The ambiguity signal that caused this check.
        process_label: Human-readable label for the process or context.
    """
    now = time.monotonic()
    last = _last_fired.get(trigger, 0.0)
    if now - last < _COOLDOWN_SECONDS:
        remaining = int(_COOLDOWN_SECONDS - (now - last))
        logger.debug(
            "Screen vision cooldown active for %s — %ds remaining",
            trigger, remaining,
        )
        return

    interpretation = await capture_and_analyse(trigger, process_label)

    if _is_actionable(interpretation):
        _last_fired[trigger] = time.monotonic()
        label_part = f" ({process_label})" if process_label else ""
        message = (
            f"Screen vision triggered — *{trigger}*{label_part}\n\n"
            f"{interpretation}"
        )
        try:
            from bot import notifier
            await notifier.send(message)

            # Update the DB record to mark it as notified.
            try:
                from bot import db
                triggers = await db.get_recent_vision_triggers(1)
                if triggers:
                    # Re-log with notified=True (simpler than UPDATE by rowid).
                    await db.log_vision_trigger(
                        trigger_type=trigger,
                        process_label=process_label,
                        interpretation=interpretation,
                        notified=True,
                    )
            except Exception:
                pass

        except Exception as exc:
            logger.warning("Failed to send vision notification: %s", exc)
    else:
        logger.debug("Screen vision [%s] — not actionable, no notification sent", trigger)


# ── Internal helpers ──────────────────────────────────────────────

def _take_screenshot() -> str:
    """Capture the primary monitor and return a base64-encoded PNG string.

    Returns empty string on failure.
    """
    try:
        import mss
        import mss.tools

        with mss.mss() as sct:
            monitor = sct.monitors[1]  # Primary monitor (index 0 = all monitors)
            screenshot = sct.grab(monitor)
            png_bytes = mss.tools.to_png(screenshot.rgb, screenshot.size)
        return base64.b64encode(png_bytes).decode("ascii")
    except Exception as exc:
        logger.warning("Screenshot capture failed: %s", exc)
        return ""


def _build_trigger_prompt(trigger: TriggerType, process_label: str) -> str:
    """Return a focused yes/no question for the given trigger type."""
    label = process_label or "the tracked process"
    prompts: dict[str, str] = {
        "stdin_silent": (
            f"You are checking whether a terminal is waiting for user input. "
            f"The process '{label}' has produced no new output for an unusually long time. "
            "Look at the screenshot. Is there a terminal or console visible that appears "
            "to be waiting for keyboard input (e.g. a prompt, cursor blinking, or question)? "
            "Answer in one or two sentences. Start with 'yes' or 'no'."
        ),
        "dialog_appeared": (
            f"You are checking whether a dialog or modal window has appeared. "
            f"Context: '{label}'. "
            "Look at the screenshot. Is there a dialog box, modal window, UAC prompt, "
            "or pop-up that requires user interaction? "
            "Answer in one or two sentences. Start with 'yes' or 'no'."
        ),
        "process_frozen": (
            f"You are checking whether a process appears frozen or stuck. "
            f"The process '{label}' has been running with no new output for a long time. "
            "Look at the screenshot. Does the application appear frozen, crashed, or stuck "
            "(e.g. spinning cursor, unresponsive UI, error dialog, no progress indicator)? "
            "Answer in one or two sentences. Start with 'yes' or 'no'."
        ),
        "cursor_ai_stalled": (
            "You are checking whether a Cursor AI task has stalled. "
            "Look at the screenshot. Is the Cursor IDE visible? If so, does the AI panel "
            "appear to have stopped mid-response or be waiting (e.g. no streaming text, "
            "a permission prompt, or an idle state with no progress)? "
            "Answer in one or two sentences. Start with 'yes' or 'no'."
        ),
    }
    return prompts.get(trigger, f"Describe what is visible on screen. Context: {label}.")


def _is_actionable(interpretation: str) -> bool:
    """Return True if the vision model's response suggests action is needed."""
    if not interpretation:
        return False
    lower = interpretation.lower()
    if lower.startswith("no"):
        return False
    actionable_signals = ("yes", "waiting", "blocked", "requires", "frozen", "stalled", "dialog", "prompt")
    return any(signal in lower for signal in actionable_signals)
