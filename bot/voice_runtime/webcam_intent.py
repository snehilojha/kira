"""LLM classification of webcam open/query/close intent."""

from __future__ import annotations

import asyncio
import json
import logging
import re

from bot import provider
from bot.voice_runtime.models import LocalVoiceResult

logger = logging.getLogger(__name__)

# Words that suggest the request might be about the webcam. The LLM intent
# classifier (a network call) only runs when one of these is present or a
# webcam session is already open — so ordinary commands like "tell me about
# this machine" never pay for a wasted classification round-trip.
_WEBCAM_KEYWORDS = (
    "camera", "webcam", "see me", "look at me", "holding",
    "how do i look", "how do i look like", "can you see",
)


def _mentions_webcam(transcript: str) -> bool:
    """Cheap prefilter: should we bother running the webcam intent classifier?"""
    lowered = transcript.strip().lower()
    if any(keyword in lowered for keyword in _WEBCAM_KEYWORDS):
        return True
    try:
        from bot import webcam as _webcam
        return _webcam.is_open()
    except Exception:
        return False


async def _handle_webcam_intent(transcript: str) -> LocalVoiceResult | None:
    """Detect webcam open/query/close intent via LLM and route accordingly.

    Returns a LocalVoiceResult if this is a webcam-related request, else None.
    """
    from bot import webcam as _webcam

    session_open = _webcam.is_open()

    system_prompt = (
        "You classify a voice request into one of four webcam intents.\n"
        "Return ONLY JSON: {\"intent\": string, \"query\": string}\n\n"
        "Intents:\n"
        "- \"open\"   — user wants to use the camera, see themselves, show something\n"
        "- \"query\"  — camera is already open, user wants to ask about what it sees\n"
        "- \"close\"  — user wants to close / stop the camera\n"
        "- \"none\"   — request has nothing to do with the webcam\n\n"
        "Rules:\n"
        f"- Camera is currently {'OPEN' if session_open else 'CLOSED'}.\n"
        "- If the camera is CLOSED and the user asks something visual about themselves or something\n"
        "  they're holding/showing, intent should be \"open\" (it will open then query).\n"
        "- If the camera is OPEN and the user asks a visual question, intent is \"query\".\n"
        "- If the camera is OPEN and the user makes a clarifying statement about who they are\n"
        "  (e.g. 'it's me', 'that's me', 'I'm Snehil', 'I'm the user'), treat as \"query\"\n"
        "  with the query 'The user just said: <their statement>. Acknowledge them by name and\n"
        "  describe what you see in the camera now that you know who it is.'\n"
        "- 'query' field: the natural language question to ask the vision model (empty for open/close).\n"
        "Examples:\n"
        "  'can you see me' → {\"intent\":\"open\",\"query\":\"\"}\n"
        "  'how do I look' → {\"intent\":\"open\",\"query\":\"How does the person in the image look?\"}\n"
        "  'what am I holding' → {\"intent\":\"open\",\"query\":\"What is the person holding?\"}\n"
        "  'guess the price of this' → {\"intent\":\"open\",\"query\":\"What product is this and what might it cost?\"}\n"
        "  'what do you see' → {\"intent\":\"query\",\"query\":\"Describe what you see.\"}\n"
        "  'it\\'s me, the user' → {\"intent\":\"query\",\"query\":\"The user just confirmed it's them. Acknowledge them and describe what you see.\"}\n"
        "  'that\\'s me' → {\"intent\":\"query\",\"query\":\"The user confirmed it's them. Acknowledge them and describe what you see.\"}\n"
        "  'close the camera' → {\"intent\":\"close\",\"query\":\"\"}\n"
        "  'ok stop' → {\"intent\":\"close\",\"query\":\"\"}\n"
        "  'what's the weather' → {\"intent\":\"none\",\"query\":\"\"}\n"
    )

    try:
        response = await provider.create_chat_completion(
            role="fast",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript},
            ],
            temperature=0,
            max_tokens=80,
        )
        raw = (response.choices[0].message.content or "").strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()
        parsed = json.loads(raw)
    except Exception:
        logger.debug("Webcam intent classification failed or returned non-JSON", exc_info=True)
        return None

    intent = str(parsed.get("intent", "none")).strip().lower()
    query  = str(parsed.get("query", "")).strip()

    if intent == "none":
        return None

    if intent == "close":
        if session_open:
            _webcam.close_session()
            return LocalVoiceResult(ok=True, message="Webcam closed.", spoken="Camera closed.")
        return LocalVoiceResult(ok=True, message="Camera wasn't open.", spoken="The camera wasn't open.")

    if intent in ("open", "query"):
        if not session_open:
            ok = await asyncio.to_thread(_webcam.open_session)
            if not ok:
                msg = "Couldn't open the webcam."
                return LocalVoiceResult(ok=False, message=msg, spoken=msg)
            await asyncio.sleep(0.5)  # let camera warm up

        if query:
            answer = await _webcam.query(query)
            return LocalVoiceResult(ok=True, message=answer, spoken=answer)

        spoken = "Camera is open. Ask me anything about what I see."
        return LocalVoiceResult(ok=True, message=spoken, spoken=spoken)

    return None
