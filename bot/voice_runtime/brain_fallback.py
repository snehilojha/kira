"""General-question answering via the LLM with a web-search tool."""

from __future__ import annotations

import asyncio
import json
import logging

from bot import db
from bot import identity
from bot import provider
from bot import screen_vision
from bot.voice_runtime import routing
from bot.voice_runtime import session
from bot.voice_runtime.models import LocalVoiceResult
from bot.voice_runtime.util import _normalize_text

logger = logging.getLogger(__name__)


async def _build_world_block() -> str:
    """Return a compact world context string from the latest DB snapshot."""
    import json as _json
    from datetime import datetime
    try:
        snapshot = await db.get_recent_world_snapshot()
    except Exception:
        logger.debug("World snapshot unavailable for voice brain context", exc_info=True)
        return ""
    if not snapshot:
        return ""

    parts = [f"Time: {datetime.now().strftime('%A, %d %B %Y, %H:%M')}"]
    if snapshot.get("weather"):
        parts.append(f"Weather: {snapshot['weather']}")
    if snapshot.get("top_news"):
        parts.append(f"News:\n{snapshot['top_news']}")
    if snapshot.get("stocks"):
        stocks = snapshot["stocks"]
        if isinstance(stocks, str):
            stocks = _json.loads(stocks)
        indices = stocks.get("indices", {})
        if indices:
            parts.append("Markets: " + ", ".join(f"{k}: {v}" for k, v in indices.items()))
        portfolio = stocks.get("portfolio", [])
        if portfolio:
            pf_lines = [
                f"  {p['ticker']}: ₹{p['price']} (P&L: ₹{p['pnl']}, {p['pnl_pct']}%)"
                for p in portfolio if p.get("pnl_pct") != 0.0
            ]
            if pf_lines:
                parts.append("Portfolio:\n" + "\n".join(pf_lines))
    return "\n".join(parts)


async def _handle_with_brain(transcript: str) -> LocalVoiceResult:
    """Answer general queries using the LLM with web search tool."""
    import openai as _openai

    config = provider.load_config()
    client = _openai.AsyncOpenAI(
        api_key=config.api_key,
        **({"base_url": config.base_url} if config.base_url else {}),
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information — news, weather, sports, facts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    # On the first turn of a new voice session, seed from DB so cross-channel
    # context (e.g. what was said on Telegram earlier) is available.
    if not session.get_session_history():
        try:
            recent = await db.get_recent_conversations(10)
            for row in recent:
                session.get_session_history().append({"role": row["role"], "content": row["content"]})
        except Exception:
            pass

    history = routing._history_context()
    identity_block = identity.get_identity_prompt()
    user_name = identity.get_user_name()

    # World context — time, weather, markets
    world_block = await _build_world_block()

    # Ambient screen context (Feature 5)
    ambient_block = ""
    try:
        from bot import ambient as _ambient
        ambient_block = _ambient.get_description()
    except Exception:
        pass

    system = (
        f"{identity_block}\n\n"
        + (f"{world_block}\n\n" if world_block else "")
        + (f"User's current activity: {ambient_block}\n\n" if ambient_block else "")
        + "Voice conversation rules — your words go straight to TTS, so write as you'd speak:\n"
        f"- Use '{user_name}' occasionally — only when it feels natural, not every reply.\n"
        "- Contractions always. Never be stiff or formal.\n"
        "- Never open with 'Certainly', 'Sure', 'Of course', 'Absolutely', or 'I'.\n"
        "- Match length to the question. One sentence for simple things. Two short sentences max for complex ones.\n"
        "- Zero markdown. Zero bullet points. Zero headers.\n"
        "- Have opinions. Be confident. Drop the hedges — 'I think' and 'it seems' make you sound uncertain.\n"
        "- Don't recite raw data. Synthesise it into one useful takeaway.\n"
        "- Never say you can't search — use web_search instead.\n"
        "- Never acknowledge being an AI unless directly asked.\n\n"
        "Emotional intelligence — this is a real conversation:\n"
        "- Read the tone of what's said. If the user sounds frustrated or tired, acknowledge it first before answering.\n"
        "- If they're joking or sarcastic, match that energy — play along, don't be wooden.\n"
        "- If they're stressed, be steadier and warmer than usual.\n"
        "- Dry wit and sarcasm are welcome when the moment calls for it. A well-placed aside beats a stiff answer.\n"
        "- Never be mean, never be dismissive.\n\n"
        "Always use web_search for current events, news, weather, prices, sports scores, or anything time-sensitive.\n\n"
        "Emotion tags — use sparingly, only when genuinely natural:\n"
        "- <laugh> for something actually funny\n"
        "- <sigh> when something is tedious, unfortunate, or you're being wry\n"
        "- <chuckle> for mild amusement\n"
        "- <gasp> for genuine surprise\n"
        "Example: 'Yeah that's a known bug. <sigh> Been around for years.'\n"
        "Most replies need zero tags. Never force them."
        + (f"\n\n{history}" if history else "")
    )

    # Silently grab screen context for queries that might benefit from it
    screen_context = ""
    screen_trigger_words = ("this", "here", "open", "screen", "working", "see", "look", "current", "active")
    if any(w in _normalize_text(transcript) for w in screen_trigger_words):
        try:
            screen_context = await screen_vision.capture_screen(
                "Briefly describe what application is open and what the user appears to be doing. One sentence."
            )
        except Exception:
            pass

    user_content = transcript
    if screen_context:
        user_content = f"{transcript}\n\n[Screen context: {screen_context}]"

    messages = (
        [{"role": "system", "content": system}]
        + session.get_session_history()
        + [{"role": "user", "content": user_content}]
    )

    selected_model = routing._pick_model(transcript, config)
    try:
        for _ in range(3):
            response = await client.chat.completions.create(
                model=selected_model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=400,
            )
            msg = response.choices[0].message

            if msg.tool_calls:
                messages.append(msg)
                for tc in msg.tool_calls:
                    if tc.function.name == "web_search":
                        query = json.loads(tc.function.arguments).get("query", transcript)
                        search_result = await asyncio.to_thread(_web_search, query)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": search_result,
                        })
            else:
                spoken = (msg.content or "I'm not sure how to answer that.").strip()
                return LocalVoiceResult(ok=True, message=spoken, spoken=spoken)

        spoken = "I wasn't able to find an answer."
        return LocalVoiceResult(ok=False, message=spoken, spoken=spoken)

    except Exception as exc:
        message = f"Brain fallback failed: {exc}"
        logger.warning(message)
        return LocalVoiceResult(ok=False, message=message, spoken="I ran into an error trying to answer that.")


def _web_search(query: str) -> str:
    """Run a DuckDuckGo search and return top results as plain text."""
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS

    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=5))
    except Exception as exc:
        return f"Search failed: {exc}"

    if not hits:
        return "No results found."

    lines = []
    for i, hit in enumerate(hits, 1):
        title = (hit.get("title") or "").strip()
        body = (hit.get("body") or "").strip()
        lines.append(f"{i}. {title}: {body}")
    return "\n\n".join(lines)
