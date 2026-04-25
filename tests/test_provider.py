"""Tests for the shared provider helpers."""

from __future__ import annotations

import os
import sys
import types
import unittest
from unittest.mock import patch

from bot import provider


class ProviderTests(unittest.TestCase):
    """Verify provider config resolution and client construction."""

    def test_load_config_prefers_kira_settings(self) -> None:
        """KIRA_* settings should take precedence over compatibility fallbacks."""
        with patch.dict(
            os.environ,
            {
                "KIRA_API_KEY": "kira-key",
                "OPENAI_API_KEY": "openai-key",
                "KIRA_API_BASE_URL": "https://example.test/v1",
                "KIRA_FAST_MODEL": "fast-model",
                "KIRA_SMART_MODEL": "smart-model",
                "KIRA_VOICE_TRANSCRIBE_MODEL": "stt-model",
                "KIRA_VOICE_SYNTHESISE_MODEL": "tts-model",
            },
            clear=False,
        ):
            config = provider.load_config()

        self.assertEqual(config.api_key, "kira-key")
        self.assertEqual(config.base_url, "https://example.test/v1")
        self.assertEqual(config.fast_model, "fast-model")
        self.assertEqual(config.smart_model, "smart-model")
        self.assertEqual(config.voice_transcribe_model, "stt-model")
        self.assertEqual(config.voice_synthesise_model, "tts-model")

    def test_load_config_supports_openrouter_fallback(self) -> None:
        """OpenRouter env vars should work when explicit KIRA_* vars are absent."""
        env = dict(os.environ)
        for key in [
            "KIRA_API_KEY",
            "OPENAI_API_KEY",
            "KIRA_API_BASE_URL",
            "OPENAI_BASE_URL",
        ]:
            env.pop(key, None)
        env["OPENROUTER_API_KEY"] = "router-key"
        env["OPENROUTER_BASE_URL"] = "https://openrouter.ai/api/v1"

        with patch.dict(os.environ, env, clear=True):
            config = provider.load_config()

        self.assertEqual(config.api_key, "router-key")
        self.assertEqual(config.base_url, "https://openrouter.ai/api/v1")

    def test_get_model_uses_role_mapping(self) -> None:
        """Known roles should resolve to the configured model names."""
        with patch.dict(
            os.environ,
            {
                "KIRA_API_KEY": "kira-key",
                "KIRA_FAST_MODEL": "fast-model",
                "KIRA_SMART_MODEL": "smart-model",
            },
            clear=False,
        ):
            self.assertEqual(provider.get_model("fast"), "fast-model")
            self.assertEqual(provider.get_model("smart"), "smart-model")

    def test_create_client_passes_base_url_when_configured(self) -> None:
        """The provider should create one OpenAI-compatible client with the resolved config."""

        class FakeAsyncOpenAI:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

        fake_openai = types.ModuleType("openai")
        fake_openai.AsyncOpenAI = FakeAsyncOpenAI

        with patch.dict(
            os.environ,
            {
                "KIRA_API_KEY": "kira-key",
                "KIRA_API_BASE_URL": "https://example.test/v1",
            },
            clear=False,
        ), patch.dict(sys.modules, {"openai": fake_openai}):
            client = provider.create_client()

        self.assertEqual(client.kwargs["api_key"], "kira-key")
        self.assertEqual(client.kwargs["base_url"], "https://example.test/v1")

    def test_load_config_raises_without_any_api_key(self) -> None:
        """A missing API key should fail fast with a clear error."""
        env = dict(os.environ)
        for key in [
            "KIRA_API_KEY",
            "OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
        ]:
            env.pop(key, None)

        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(RuntimeError):
                provider.load_config()


if __name__ == "__main__":
    unittest.main()
