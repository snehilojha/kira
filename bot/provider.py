"""Shared LLM provider helpers for Kira.

Centralises client creation, model role lookup, and provider configuration
so handlers and background tasks do not construct API clients directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

_DEFAULT_FAST_MODEL = "gpt-4o-mini"
_DEFAULT_SMART_MODEL = "gpt-4o-mini"
_DEFAULT_VISION_MODEL = "gpt-4o-mini"
_DEFAULT_VOICE_TRANSCRIBE_MODEL = "whisper-1"
_DEFAULT_VOICE_SYNTHESISE_MODEL = "tts-1"


@dataclass(frozen=True)
class ProviderConfig:
    """Resolved provider configuration from environment variables."""

    api_key: str
    base_url: str | None
    fast_model: str
    smart_model: str
    vision_model: str
    voice_transcribe_model: str
    voice_synthesise_model: str


def load_config() -> ProviderConfig:
    """Resolve provider settings from environment variables.

    `OPENAI_API_KEY` remains the default for backward compatibility.
    `KIRA_*` settings allow the bot to move to a single provider abstraction
    without changing call sites again later.
    """
    api_key = (
        os.environ.get("KIRA_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "No provider API key configured. Set KIRA_API_KEY, OPENROUTER_API_KEY, or OPENAI_API_KEY."
        )

    base_url = (
        os.environ.get("KIRA_API_BASE_URL")
        or os.environ.get("OPENROUTER_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or None
    )

    return ProviderConfig(
        api_key=api_key,
        base_url=base_url,
        fast_model=os.environ.get("KIRA_FAST_MODEL", _DEFAULT_FAST_MODEL),
        smart_model=os.environ.get("KIRA_SMART_MODEL", _DEFAULT_SMART_MODEL),
        vision_model=os.environ.get("KIRA_VISION_MODEL", _DEFAULT_VISION_MODEL),
        voice_transcribe_model=os.environ.get(
            "KIRA_VOICE_TRANSCRIBE_MODEL", _DEFAULT_VOICE_TRANSCRIBE_MODEL
        ),
        voice_synthesise_model=os.environ.get(
            "KIRA_VOICE_SYNTHESISE_MODEL", _DEFAULT_VOICE_SYNTHESISE_MODEL
        ),
    )


def get_model(role: str) -> str:
    """Return the configured model name for a known role."""
    config = load_config()
    role_map = {
        "fast": config.fast_model,
        "smart": config.smart_model,
        "vision": config.vision_model,
        "voice_transcribe": config.voice_transcribe_model,
        "voice_synthesise": config.voice_synthesise_model,
    }
    try:
        return role_map[role]
    except KeyError as exc:
        raise ValueError(f"Unknown model role: {role}") from exc


def create_client() -> Any:
    """Create an async OpenAI-compatible client using the configured provider."""
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise ImportError("openai package not installed. Run: pip install openai") from exc

    config = load_config()
    kwargs: dict[str, Any] = {"api_key": config.api_key}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return AsyncOpenAI(**kwargs)


async def create_chat_completion(
    *,
    role: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    **kwargs: Any,
) -> Any:
    """Create a chat completion for a configured model role."""
    client = create_client()
    return await client.chat.completions.create(
        model=get_model(role),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )


async def transcribe_audio(*, file: Any, language: str = "en", **kwargs: Any) -> Any:
    """Run speech-to-text using the configured transcription model."""
    client = create_client()
    return await client.audio.transcriptions.create(
        model=get_model("voice_transcribe"),
        file=file,
        language=language,
        **kwargs,
    )


async def synthesise_speech(*, text: str, voice: str, response_format: str = "mp3", **kwargs: Any) -> Any:
    """Run text-to-speech using the configured speech model."""
    client = create_client()
    return await client.audio.speech.create(
        model=get_model("voice_synthesise"),
        voice=voice,
        input=text,
        response_format=response_format,
        **kwargs,
    )


async def create_vision_completion(
    *,
    prompt: str,
    image_b64: str,
    max_tokens: int = 300,
    image_format: str = "png",
) -> Any:
    """Send an image to the vision model and return the completion response.

    Args:
        prompt: Text instruction for the vision model.
        image_b64: Base64-encoded image bytes (no data-URI prefix needed).
        max_tokens: Maximum tokens for the response.
        image_format: MIME sub-type — "png" or "jpeg". Defaults to "png".

    Returns:
        The raw OpenAI-compatible completion response object.
    """
    client = create_client()
    mime = f"image/{image_format}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                },
            ],
        }
    ]
    return await client.chat.completions.create(
        model=get_model("vision"),
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.1,
    )
