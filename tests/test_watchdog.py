"""Tests for bot.watchdog — event-driven file and PID watches.

These exercise the event-handling logic (path matching, debounce coalescing,
deletion handling, PID-exit detection) directly, without depending on
filesystem-event timing, so they're fast and deterministic.
"""

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

from bot import watchdog


class _Evt:
    """Minimal stand-in for a watchdog filesystem event."""

    def __init__(self, src_path, is_directory=False, dest_path=None):
        self.src_path = src_path
        self.is_directory = is_directory
        if dest_path is not None:
            self.dest_path = dest_path


class WatchdogTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        watchdog.WATCHES.clear()
        self._np = patch.object(watchdog.notifier, "send", new=AsyncMock())
        self.send = self._np.start()
        self._dp = patch.object(watchdog, "_FILE_DEBOUNCE_SECONDS", 0.1)
        self._dp.start()
        self.path = os.path.abspath("C:/tmp/watched.txt")
        self.other = os.path.abspath("C:/tmp/other.txt")

    async def asyncTearDown(self):
        self._np.stop()
        self._dp.stop()
        watchdog.WATCHES.clear()

    def _alerts(self):
        return [c.args[0] for c in self.send.await_args_list]

    # ── handler path matching / emission ──────────────────────────
    async def test_handler_emits_modified_only_for_target(self):
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        h = watchdog._FileWatchHandler(self.path, loop, q)
        h.on_modified(_Evt(self.path))
        h.on_modified(_Evt(self.other))      # different file — ignored
        h.on_modified(_Evt(self.path, is_directory=True))  # dir event — ignored
        await asyncio.sleep(0.05)
        self.assertEqual(q.qsize(), 1)
        self.assertEqual(await q.get(), "modified")

    async def test_handler_created_counts_as_modified(self):
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        h = watchdog._FileWatchHandler(self.path, loop, q)
        h.on_created(_Evt(self.path))
        await asyncio.sleep(0.05)
        self.assertEqual(await q.get(), "modified")

    async def test_handler_moved_away_is_deletion(self):
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        h = watchdog._FileWatchHandler(self.path, loop, q)
        h.on_moved(_Evt(self.path, dest_path=self.other))   # target moved away
        await asyncio.sleep(0.05)
        self.assertEqual(await q.get(), "deleted")

    async def test_handler_moved_onto_target_is_modified(self):
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        h = watchdog._FileWatchHandler(self.path, loop, q)
        h.on_moved(_Evt(self.other, dest_path=self.path))   # atomic-save rename onto target
        await asyncio.sleep(0.05)
        self.assertEqual(await q.get(), "modified")

    # ── debounce / deletion in the consumer loop ──────────────────
    async def test_burst_of_modifications_coalesces_to_one_alert(self):
        q: asyncio.Queue = asyncio.Queue()
        for _ in range(5):
            q.put_nowait("modified")
        task = asyncio.create_task(watchdog._file_watch_loop("watch-x", "label", None, q))
        await asyncio.sleep(0.3)   # past the 0.1s debounce window
        task.cancel()
        mods = [m for m in self._alerts() if "was modified" in m]
        self.assertEqual(len(mods), 1, self._alerts())

    async def test_deletion_fires_alert_and_removes_watch(self):
        watchdog.WATCHES["watch-x"] = watchdog.WatchEntry(
            id="watch-x", watch_type="file", target=self.path, label="L",
            task=asyncio.current_task(), db_id=None,
        )
        q: asyncio.Queue = asyncio.Queue()
        q.put_nowait("deleted")
        await asyncio.wait_for(watchdog._file_watch_loop("watch-x", "L", None, q), timeout=2)
        self.assertNotIn("watch-x", watchdog.WATCHES)
        self.assertTrue(any("was deleted" in m for m in self._alerts()))

    async def test_deletion_mid_burst_wins(self):
        q: asyncio.Queue = asyncio.Queue()
        q.put_nowait("modified")
        q.put_nowait("deleted")
        watchdog.WATCHES["watch-x"] = watchdog.WatchEntry(
            id="watch-x", watch_type="file", target=self.path, label="L",
            task=asyncio.current_task(), db_id=None,
        )
        await asyncio.wait_for(watchdog._file_watch_loop("watch-x", "L", None, q), timeout=2)
        self.assertTrue(any("was deleted" in m for m in self._alerts()))
        self.assertFalse(any("was modified" in m for m in self._alerts()))

    # ── PID exit detection ─────────────────────────────────────────
    async def test_await_process_exit_returns_for_dead_pid(self):
        # A PID that almost certainly doesn't exist → psutil.Process raises
        # NoSuchProcess and the helper returns immediately.
        await asyncio.wait_for(watchdog._await_process_exit(2_000_000_000), timeout=2)

    async def test_pid_watch_loop_fires_and_removes(self):
        watchdog.WATCHES["watch-p"] = watchdog.WatchEntry(
            id="watch-p", watch_type="pid", target="2000000000", label="PID 2000000000",
            task=asyncio.current_task(), db_id=None,
        )
        await asyncio.wait_for(
            watchdog._pid_watch_loop(2_000_000_000, "watch-p", "PID 2000000000", None),
            timeout=2,
        )
        self.assertNotIn("watch-p", watchdog.WATCHES)
        self.assertTrue(any("has died" in m for m in self._alerts()))


if __name__ == "__main__":
    unittest.main()
