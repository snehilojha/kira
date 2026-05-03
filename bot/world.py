"""World snapshot writer for Kira.

Fetches weather, news, Indian market indices, and portfolio data every 30
minutes and persists them to the world_snapshots table.

Sources (all free, no API key):
- Weather    : wttr.in (HTTP JSON)
- News       : DuckDuckGo news search
- Indices    : yfinance (^NSEI, ^BSESN, NIFTYMIDCAP150.NS, NIFTYSMLCAP250.NS)
- Portfolio  : data/portfolio.json (user-maintained) + yfinance prices

Portfolio file format (data/portfolio.json):
    [
        {"ticker": "RELIANCE.NS", "qty": 10, "avg_price": 2400.0},
        {"ticker": "TCS.NS",      "qty":  5, "avg_price": 3500.0}
    ]
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import httpx

from bot import db

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PORTFOLIO_PATH = _PROJECT_ROOT / "data" / "portfolio.json"
_INTERVAL_SECONDS = 30 * 60  # 30 minutes

# Indices to always track
_INDICES = {
    "Nifty 50":       "^NSEI",
    "Sensex":         "^BSESN",
    "Nifty Midcap":   "NIFTYMIDCAP150.NS",
    "Nifty Smallcap": "NIFTYSMLCAP250.NS",
}


async def start() -> None:
    """Background loop — fetch and save a world snapshot every 30 minutes."""
    logger.info("World snapshot writer started (interval=%ds)", _INTERVAL_SECONDS)
    while True:
        try:
            snapshot = await _build_snapshot()
            await db.save_world_snapshot(snapshot)
            logger.info("World snapshot saved")
        except Exception:
            logger.exception("World snapshot failed")
        await asyncio.sleep(_INTERVAL_SECONDS)


async def _build_snapshot() -> dict:
    weather, news, stocks = await asyncio.gather(
        _fetch_weather(),
        _fetch_news(),
        asyncio.to_thread(_fetch_stocks),
        return_exceptions=True,
    )
    return {
        "weather": weather if isinstance(weather, str) else None,
        "top_news": news if isinstance(news, str) else None,
        "stocks": stocks if isinstance(stocks, dict) else None,
    }


async def _fetch_weather() -> str:
    """Fetch current weather from wttr.in as a one-line summary."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get("https://wttr.in/?format=3")
            resp.raise_for_status()
            return resp.text.strip()
    except Exception as exc:
        logger.warning("Weather fetch failed: %s", exc)
        return ""


async def _fetch_news() -> str:
    """Fetch top Indian market + general news via DuckDuckGo."""
    return await asyncio.to_thread(_ddg_news)


def _ddg_news() -> str:
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS

    queries = [
        ("AI artificial intelligence news today", 4),
        ("global financial markets news today", 3),
        ("world news today", 3),
        ("India stock market today", 3),
    ]
    headlines: list[str] = []
    seen: set[str] = set()
    try:
        with DDGS() as ddgs:
            for query, n in queries:
                for hit in ddgs.news(query, max_results=n):
                    title = (hit.get("title") or "").strip()
                    if title and title.lower() not in seen:
                        headlines.append(title)
                        seen.add(title.lower())
    except Exception as exc:
        logger.warning("News fetch failed: %s", exc)

    return "\n".join(f"- {h}" for h in headlines[:15])


def _fetch_stocks() -> dict:
    """Fetch index prices and portfolio P&L via yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed — skipping stock fetch")
        return {}

    result: dict = {"indices": {}, "portfolio": []}

    # Indices
    tickers = list(_INDICES.values())
    portfolio_tickers: list[str] = []
    holdings: list[dict] = []

    if _PORTFOLIO_PATH.exists():
        try:
            holdings = json.loads(_PORTFOLIO_PATH.read_text(encoding="utf-8"))
            portfolio_tickers = [h["ticker"] for h in holdings if "ticker" in h]
        except Exception as exc:
            logger.warning("Failed to load portfolio: %s", exc)

    all_tickers = tickers + portfolio_tickers
    if not all_tickers:
        return result

    try:
        data = yf.download(
            all_tickers,
            period="1d",
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        closes = data["Close"] if "Close" in data else data
    except Exception as exc:
        logger.warning("yfinance download failed: %s", exc)
        return result

    # Extract index prices
    for name, ticker in _INDICES.items():
        try:
            price = float(closes[ticker].dropna().iloc[-1])
            result["indices"][name] = round(price, 2)
        except Exception:
            pass

    # Portfolio P&L
    for holding in holdings:
        ticker = holding.get("ticker", "")
        qty = holding.get("qty", 0)
        avg = holding.get("avg_price", 0.0)
        try:
            price = float(closes[ticker].dropna().iloc[-1])
            invested = qty * avg
            current = qty * price
            pnl = current - invested
            pnl_pct = (pnl / invested * 100) if invested else 0.0
            result["portfolio"].append({
                "ticker": ticker,
                "price": round(price, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
            })
        except Exception:
            pass

    return result
