"""Tests for bot.supervision — crash supervision of background tasks."""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from bot import supervision


class SupervisionTests(unittest.TestCase):
    def setUp(self) -> None:
        # Patch the alert path so tests never touch Telegram, and so we can
        # assert on what would have been sent.
        self._alert_patcher = patch.object(supervision, "_alert", new=AsyncMock())
        self.alert = self._alert_patcher.start()
        self.addCleanup(self._alert_patcher.stop)

    def test_normal_completion_returns_without_restart(self) -> None:
        calls = []

        async def factory():
            calls.append(1)

        asyncio.run(supervision.supervise(factory, name="oneshot"))

        self.assertEqual(len(calls), 1)
        self.alert.assert_not_awaited()

    def test_restarts_with_backoff_until_success(self) -> None:
        calls = []

        async def factory():
            calls.append(len(calls))
            if len(calls) < 3:
                raise RuntimeError("boom")
            # third call completes normally

        async def _run():
            with patch.object(supervision.asyncio, "sleep", new=AsyncMock()) as sleep_mock:
                await supervision.supervise(factory, name="flaky", base_delay=1.0)
            return sleep_mock

        sleep_mock = asyncio.run(_run())

        self.assertEqual(len(calls), 3)              # two crashes, then success
        self.assertEqual(self.alert.await_count, 2)  # one alert per crash
        # Exponential backoff: 1s then 2s
        delays = [c.args[0] for c in sleep_mock.await_args_list]
        self.assertEqual(delays, [1.0, 2.0])

    def test_gives_up_after_max_restarts(self) -> None:
        calls = []

        async def factory():
            calls.append(1)
            raise RuntimeError("always")

        async def _run():
            with patch.object(supervision.asyncio, "sleep", new=AsyncMock()):
                await supervision.supervise(
                    factory, name="broken", max_restarts=3, base_delay=1.0
                )

        asyncio.run(_run())

        # 3 restarts attempted + 1 final failing run that exceeds the cap = 4 calls
        self.assertEqual(len(calls), 4)
        # Final alert is the give-up message
        last_msg = self.alert.await_args.args[0]
        self.assertIn("won't be", last_msg)

    def test_restart_false_alerts_once_and_stops(self) -> None:
        calls = []

        async def factory():
            calls.append(1)
            raise RuntimeError("boom")

        asyncio.run(supervision.supervise(factory, name="noretry", restart=False))

        self.assertEqual(len(calls), 1)
        self.alert.assert_awaited_once()
        self.assertIn("Not restarting", self.alert.await_args.args[0])

    def test_cancellation_propagates(self) -> None:
        async def factory():
            raise asyncio.CancelledError()

        async def _run():
            await supervision.supervise(factory, name="cancelled")

        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(_run())
        self.alert.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
