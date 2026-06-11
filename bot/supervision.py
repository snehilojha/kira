"""Crash supervision for Kira's always-on background tasks.

main.py launches ~9 long-running asyncio tasks (observer, mode, world,
local voice, …). Bare ``create_task`` lets any of them die silently on an
unhandled exception — the feature just stops with no log surfaced and no
alert. ``supervise`` wraps each one so crashes are logged, reported to the
user over Telegram, and the task is restarted with capped exponential
backoff.

Usage::

    application.create_task(supervise(observer.start, name="observer"))
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

# Factory = zero-arg callable returning a fresh coroutine each invocation.
# We take the factory (not a coroutine) so each restart gets a new awaitable.
TaskFactory = Callable[[], Awaitable[None]]


async def supervise(
    factory: TaskFactory,
    *,
    name: str,
    restart: bool = True,
    max_restarts: int = 5,
    base_delay: float = 5.0,
    max_delay: float = 300.0,
    stable_seconds: float = 600.0,
) -> None:
    """Run ``factory()`` forever, restarting it if it crashes.

    Args:
        factory: Zero-arg callable returning the coroutine to run. Called
            afresh on every (re)start so the coroutine is never reused.
        name: Human-readable task name for logs and alerts.
        restart: If False, a crash is logged/alerted but not restarted.
        max_restarts: Consecutive crashes tolerated before giving up.
        base_delay: First backoff delay in seconds (doubles each crash).
        max_delay: Backoff ceiling in seconds.
        stable_seconds: A run lasting at least this long resets the
            consecutive-failure counter — an occasional hiccup after hours
            of healthy operation shouldn't count toward ``max_restarts``.
    """
    failures = 0
    loop = asyncio.get_event_loop()

    while True:
        started = loop.time()
        try:
            await factory()
        except asyncio.CancelledError:
            # Clean shutdown — propagate so the task actually stops.
            logger.info("Supervised task %r cancelled", name)
            raise
        except Exception as exc:
            ran_for = loop.time() - started
            if ran_for >= stable_seconds:
                failures = 0  # healthy long run — forgive prior failures
            failures += 1
            logger.exception("Supervised task %r crashed (failure %d)", name, failures)

            if not restart:
                await _alert(f"⚠️ Background task {name} crashed: {exc!r}. Not restarting.")
                return

            if failures > max_restarts:
                await _alert(
                    f"⛔ Background task {name} crashed {failures} times and won't be "
                    f"restarted. Last error: {exc!r}. Restart Kira to recover."
                )
                return

            delay = min(base_delay * 2 ** (failures - 1), max_delay)
            await _alert(
                f"⚠️ Background task {name} crashed: {exc!r}. "
                f"Restarting in {delay:.0f}s ({failures}/{max_restarts})."
            )
            await asyncio.sleep(delay)
            continue
        else:
            # Normal completion — the task was meant to finish (one-shot).
            logger.info("Supervised task %r completed", name)
            return


async def _alert(message: str) -> None:
    """Best-effort Telegram alert. Never let a notifier failure mask the crash."""
    try:
        from bot import notifier
        await notifier.send(message)
    except Exception:
        logger.warning("Failed to send supervision alert for: %s", message, exc_info=True)
