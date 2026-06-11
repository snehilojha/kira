import asyncio
import types
import unittest
from unittest.mock import AsyncMock, patch

from bot import voice


class VoiceIOTests(unittest.TestCase):
    def test_transcribe_defaults_to_ogg_and_accepts_wav_suffix(self) -> None:
        async def _run() -> tuple[str, str]:
            seen_names: list[str] = []

            async def _fake_transcribe_audio(*, file, language="en", **kwargs):
                seen_names.append(file.name)
                return types.SimpleNamespace(text="open chrome")

            with patch.object(
                voice.provider,
                "transcribe_audio",
                new=AsyncMock(side_effect=_fake_transcribe_audio),
            ):
                default_text = await voice.transcribe(b"audio")
                wav_text = await voice.transcribe(b"audio", suffix=".wav")

            self.assertTrue(seen_names[0].endswith(".ogg"))
            self.assertTrue(seen_names[1].endswith(".wav"))
            return default_text, wav_text

        default_text, wav_text = asyncio.run(_run())

        self.assertEqual(default_text, "open chrome")
        self.assertEqual(wav_text, "open chrome")

    def test_synthesise_accepts_response_format(self) -> None:
        async def _run() -> tuple[bytes, str]:
            response = types.SimpleNamespace(read=lambda: b"wav-bytes")
            with patch.object(
                voice.provider,
                "synthesise_speech",
                new=AsyncMock(return_value=response),
            ) as synth_mock:
                audio, fmt = await voice.synthesise("hello", response_format="wav")

            synth_mock.assert_awaited_once()
            self.assertEqual(synth_mock.await_args.kwargs["response_format"], "wav")
            return audio, fmt

        audio, fmt = asyncio.run(_run())
        self.assertEqual(audio, b"wav-bytes")
        self.assertEqual(fmt, "wav")


if __name__ == "__main__":
    unittest.main()
