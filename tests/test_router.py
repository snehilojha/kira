"""Tests for the standalone intent router."""

from __future__ import annotations

import types
import unittest
from unittest.mock import AsyncMock, patch

from bot import router


def _fake_response(text: str):
    """Build the minimal chat completion object the router expects."""
    return types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(content=text),
            )
        ]
    )


class RouterRuleTests(unittest.TestCase):
    """Verify deterministic routing for obvious inputs."""

    def test_slash_command_bypasses_nl_routing(self) -> None:
        decision = router.classify_by_rule("/status")
        self.assertIsNotNone(decision)
        self.assertEqual(decision.route, "simple")
        self.assertEqual(decision.source, "slash")

    def test_monitor_phrase_routes_to_monitor(self) -> None:
        decision = router.classify_by_rule("monitor this training run until loss drops below 0.2")
        self.assertIsNotNone(decision)
        self.assertEqual(decision.route, "monitor")

    def test_complex_phrase_routes_to_complex(self) -> None:
        decision = router.classify_by_rule("review my code and fix the bug")
        self.assertIsNotNone(decision)
        self.assertEqual(decision.route, "complex")

    def test_obvious_simple_routes_to_simple(self) -> None:
        decision = router.classify_by_rule("open spotify")
        self.assertIsNotNone(decision)
        self.assertEqual(decision.route, "simple")


class RouterModelTests(unittest.IsolatedAsyncioTestCase):
    """Verify model fallback behavior for ambiguous inputs."""

    async def test_model_fallback_returns_parsed_route(self) -> None:
        with patch.object(
            router.provider,
            "create_chat_completion",
            new=AsyncMock(return_value=_fake_response('{"route":"complex"}')),
        ) as mock_create:
            decision = await router.classify_request("should I revisit the last training failure?")

        self.assertEqual(decision.route, "complex")
        self.assertEqual(decision.source, "model")
        self.assertFalse(decision.needs_clarification)
        mock_create.assert_awaited_once()

    async def test_invalid_model_output_requests_clarification(self) -> None:
        with patch.object(
            router.provider,
            "create_chat_completion",
            new=AsyncMock(return_value=_fake_response("not valid json")),
        ):
            decision = await router.classify_request("did something change?")

        self.assertEqual(decision.route, "complex")
        self.assertTrue(decision.needs_clarification)
        self.assertEqual(decision.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
