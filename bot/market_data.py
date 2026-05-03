"""Market data fetcher for Kira overlay.

Fetches:
  - Market indices: NIFTY 50, SENSEX, NIFTY BANK, BTC/INR, TAO/INR
  - Portfolio: tickers from data/portfolio.json, P&L in INR

Uses yfinance. Falls back gracefully if unavailable.
USD→INR conversion fetched from USDINR=X ticker.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_PORTFOLIO_PATH = Path(__file__).parent.parent / "data" / "portfolio.json"

_MARKET_TICKERS = {
    "NIFTY 50":  "^NSEI",
    "SENSEX":    "^BSESN",
    "BANK NIFTY": "^NSEBANK",
}

_CRYPTO_TICKERS = {
    "BTC":  "BTC-USD",
    "TAO":  "TAO22974-USD",
}

_USD_INR_TICKER = "USDINR=X"


@dataclass
class TickerResult:
    name: str
    price: float           # in INR
    change_pct: float      # daily % change


@dataclass
class HoldingResult:
    ticker: str
    qty: int
    avg_price: float       # INR
    current_price: float   # INR
    current_value: float   # INR
    invested: float        # INR
    pnl: float             # INR
    pnl_pct: float         # %


@dataclass
class MarketSnapshot:
    indices: list[TickerResult] = field(default_factory=list)
    crypto: list[TickerResult] = field(default_factory=list)
    holdings: list[HoldingResult] = field(default_factory=list)
    total_invested: float = 0.0
    total_value: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    usd_inr: float = 0.0
    error: Optional[str] = None


def _load_portfolio() -> list[dict]:
    try:
        with open(_PORTFOLIO_PATH) as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Could not load portfolio.json: %s", exc)
        return []


def fetch_snapshot() -> MarketSnapshot:
    """Synchronous fetch — run on a background thread, not Qt thread."""
    try:
        import yfinance as yf
    except ImportError:
        return MarketSnapshot(error="yfinance not installed (pip install yfinance)")

    snap = MarketSnapshot()

    # ── USD/INR rate ─────────────────────────────────────────────────
    try:
        usdinr_data = yf.Ticker(_USD_INR_TICKER).fast_info
        snap.usd_inr = float(usdinr_data.last_price or 84.0)
    except Exception:
        snap.usd_inr = 84.0  # fallback

    # ── Market indices ────────────────────────────────────────────────
    try:
        idx_tickers = yf.Tickers(" ".join(_MARKET_TICKERS.values()))
        for name, sym in _MARKET_TICKERS.items():
            try:
                t = idx_tickers.tickers[sym]
                info = t.fast_info
                price = float(info.last_price or 0)
                prev  = float(info.previous_close or price)
                chg   = ((price - prev) / prev * 100) if prev else 0.0
                snap.indices.append(TickerResult(name=name, price=price, change_pct=chg))
            except Exception as exc:
                logger.debug("Index fetch failed %s: %s", sym, exc)
    except Exception as exc:
        logger.warning("Indices fetch failed: %s", exc)

    # ── Crypto (kept in USD) ──────────────────────────────────────────
    try:
        crypto_tickers = yf.Tickers(" ".join(_CRYPTO_TICKERS.values()))
        for name, sym in _CRYPTO_TICKERS.items():
            try:
                t = crypto_tickers.tickers[sym]
                info = t.fast_info
                price_usd = float(info.last_price or 0)
                prev_usd  = float(info.previous_close or price_usd)
                chg       = ((price_usd - prev_usd) / prev_usd * 100) if prev_usd else 0.0
                snap.crypto.append(TickerResult(name=name, price=price_usd, change_pct=chg))
            except Exception as exc:
                logger.debug("Crypto fetch failed %s: %s", sym, exc)
    except Exception as exc:
        logger.warning("Crypto fetch failed: %s", exc)

    # ── Portfolio holdings ────────────────────────────────────────────
    portfolio = _load_portfolio()
    if portfolio:
        syms = [h["ticker"] for h in portfolio]
        try:
            port_tickers = yf.Tickers(" ".join(syms))
            for entry in portfolio:
                sym = entry["ticker"]
                qty = int(entry["qty"])
                avg = float(entry["avg_price"])
                try:
                    t = port_tickers.tickers[sym]
                    info = t.fast_info
                    cur_price = float(info.last_price or avg)
                    cur_val   = cur_price * qty
                    invested  = avg * qty
                    pnl       = cur_val - invested
                    pnl_pct   = (pnl / invested * 100) if invested else 0.0
                    short_name = sym.replace(".NS", "").replace(".BO", "")
                    snap.holdings.append(HoldingResult(
                        ticker=short_name,
                        qty=qty,
                        avg_price=avg,
                        current_price=cur_price,
                        current_value=cur_val,
                        invested=invested,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                    ))
                except Exception as exc:
                    logger.debug("Holding fetch failed %s: %s", sym, exc)
        except Exception as exc:
            logger.warning("Portfolio fetch failed: %s", exc)

    if snap.holdings:
        snap.total_invested = sum(h.invested for h in snap.holdings)
        snap.total_value    = sum(h.current_value for h in snap.holdings)
        snap.total_pnl      = snap.total_value - snap.total_invested
        snap.total_pnl_pct  = (snap.total_pnl / snap.total_invested * 100) if snap.total_invested else 0.0

    return snap
