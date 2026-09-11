#!/usr/bin/env python3
"""
intraday_quotes.py — Live / pre-market / post-market quote layer for the
Intraday dashboard (``pages/14_Intraday.py``).

Everything is derived from Yahoo Finance's v8 chart endpoint, one request per
symbol.  We shell out to ``curl`` via ``create_returns._fetch_url`` for the HTTP
layer: Python's own TLS fingerprint is 429-blocked by Yahoo, while curl's
browser-like fingerprint is not (same reason ``create_returns`` uses curl).

From a single chart response per symbol we take:
  - latest price   — last non-null close in the (pre/regular/post) intraday
                     series, so pre- and post-market prints are picked up
  - previous close — ``meta.previousClose`` (yesterday's regular close)
  - pct change     — ``price / previous_close - 1`` (during pre-market this is
                     the *implied open* move; during regular hours the day move)
  - market state   — derived from ``meta.currentTradingPeriod`` vs now

No API key required; no dependency on the IBKR gateway.
"""

from __future__ import annotations

import concurrent.futures as cf
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd

from config import UNIVERSE_DB
from pipeline.create_returns import YAHOO_TICKER_ALIAS, _fetch_url
from utils import get_db, get_logger

log = get_logger("intraday_quotes")

_CHART_URL = (
    "https://query2.finance.yahoo.com/v8/finance/chart/{sym}"
    "?interval=5m&range=1d&includePrePost=true"
)

# ---------------------------------------------------------------------------
# Display config — which market-context tickers the dashboard shows at the top.
# This is VIEW configuration (what context to render), not a universe / identity
# mapping, so it lives here rather than in universe.db's reference tables.  The
# style / factor ETFs, by contrast, ARE mapped (index_registry) and are pulled
# live via get_style_etfs().  Order here is the display order top-to-bottom.
# ---------------------------------------------------------------------------
INDEX_TICKERS: list[tuple[str, str]] = [
    ("^GSPC", "S&P 500"),
    ("^IXIC", "Nasdaq"),
    ("^DJI", "Dow Jones"),
    ("^RUI", "Russell 1000"),
    ("^RUT", "Russell 2000"),
    ("^VIX", "VIX"),
]

# 11 GICS sectors via the State Street sector SPDR ETFs.
SECTOR_ETFS: list[tuple[str, str]] = [
    ("XLK", "Technology"),
    ("XLF", "Financials"),
    ("XLV", "Health Care"),
    ("XLY", "Cons. Discretionary"),
    ("XLP", "Cons. Staples"),
    ("XLE", "Energy"),
    ("XLI", "Industrials"),
    ("XLB", "Materials"),
    ("XLU", "Utilities"),
    ("XLRE", "Real Estate"),
    ("XLC", "Comm. Services"),
]

_MARKET_STATE_LABELS = {
    "PRE": "Pre-Market",
    "REGULAR": "Market Open",
    "POST": "After Hours",
    "CLOSED": "Market Closed",
}


@dataclass
class Quote:
    """A single symbol's live snapshot, normalised across pre/regular/post."""

    symbol: str
    price: float | None
    prev_close: float | None
    pct: float | None
    state: str
    currency: str | None
    name: str | None
    session_hint: str | None = None  # e.g. "pre-market opens 04:00 EDT" when CLOSED


def get_style_etfs() -> list[tuple[str, str]]:
    """(ticker, label) for the MSCI USA single-factor ETFs from index_registry.

    Pulled from the DB (not hardcoded) so the style row tracks whatever factor
    ETFs are registered.  Labels drop the shared "MSCI USA " prefix for a
    compact axis (e.g. "MSCI USA Momentum" -> "Momentum").
    """
    if not UNIVERSE_DB.exists():
        return []
    with get_db(UNIVERSE_DB) as conn:
        rows = conn.execute(
            r"""
            SELECT etf_ticker, COALESCE(display_name, etf_name, etf_ticker)
            FROM index_registry
            WHERE index_name LIKE 'msci\_usa\_%' ESCAPE '\'
              AND etf_ticker IS NOT NULL AND etf_ticker != ''
            ORDER BY display_name
            """
        ).fetchall()
    return [(t, str(n).replace("MSCI USA ", "").strip()) for t, n in rows]


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------
def _derive_state(meta: dict) -> str:
    """Map Yahoo's currentTradingPeriod windows to PRE / REGULAR / POST / CLOSED."""
    now = datetime.now(timezone.utc).timestamp()
    cp = meta.get("currentTradingPeriod") or {}
    reg, pre, post = cp.get("regular"), cp.get("pre"), cp.get("post")
    if reg and reg["start"] <= now < reg["end"]:
        return "REGULAR"
    if pre and pre["start"] <= now < pre["end"]:
        return "PRE"
    if post and post["start"] <= now < post["end"]:
        return "POST"
    return "CLOSED"


def _session_hint(state: str, meta: dict) -> str | None:
    """Human hint about the next live session, for the banner.

    CLOSED → "pre-market opens HH:MM TZ" (+" next session" once today's window
    has already passed); PRE → "market opens HH:MM TZ". Times use the exchange's
    own gmtoffset from the trading-period window, so DST needs no special-casing.
    """
    cp = meta.get("currentTradingPeriod") or {}

    def at(window: str, key: str) -> tuple[str, str] | None:
        w = cp.get(window)
        if not w or key not in w:
            return None
        tz = timezone(timedelta(seconds=w.get("gmtoffset", 0)))
        return datetime.fromtimestamp(w[key], tz).strftime("%H:%M"), w.get("timezone", "ET")

    if state == "CLOSED":
        pre = cp.get("pre")
        got = at("pre", "start")
        if not (pre and got):
            return None
        t, tzabbr = got
        passed = datetime.now(timezone.utc).timestamp() >= pre["start"]
        return f"pre-market opens {t} {tzabbr}" + (" next session" if passed else "")
    if state == "PRE":
        got = at("regular", "start")
        if got:
            return f"market opens {got[0]} {got[1]}"
    return None


def _parse(symbol: str, data: dict) -> Quote | None:
    try:
        result = data["chart"]["result"][0]
        meta = result["meta"]
    except (KeyError, IndexError, TypeError):
        return None

    prev_close = meta.get("previousClose") or meta.get("chartPreviousClose")

    # Latest print = last non-null close in the intraday series. With
    # includePrePost=true this series carries pre- and post-market bars, so this
    # captures the extended-hours price without any special-casing.
    price: float | None = None
    try:
        closes = result["indicators"]["quote"][0]["close"]
        for c in reversed(closes):
            if c is not None:
                price = float(c)
                break
    except (KeyError, IndexError, TypeError):
        pass
    if price is None:
        price = meta.get("regularMarketPrice")

    pct = (price / prev_close - 1.0) if (price and prev_close) else None
    name = meta.get("shortName") or meta.get("longName")
    state = _derive_state(meta)
    return Quote(symbol, price, prev_close, pct, state,
                 meta.get("currency"), name, _session_hint(state, meta))


def fetch_quote(symbol: str) -> Quote | None:
    """Fetch one symbol. Returns None on network failure / rate-limit / bad data."""
    raw = symbol.upper().strip()
    # IBKR stores class shares with a space ("BRK B"); Yahoo wants a hyphen
    # ("BRK-B"). The alias table covers the no-space form ("BRKB").
    yahoo_sym = YAHOO_TICKER_ALIAS.get(raw, raw).replace(" ", "-")
    url = _CHART_URL.format(sym=urllib.parse.quote(yahoo_sym, safe=""))
    data = _fetch_url(url, retries=1, fast_fail_on_429=True)
    if not data:  # None (failure) or {} (404) — nothing to show
        return None
    return _parse(symbol, data)


def fetch_quotes(symbols: list[str], max_workers: int = 8) -> dict[str, Quote]:
    """Fetch many symbols concurrently. Returns {symbol: Quote} (misses omitted).

    Curl subprocesses are IO-bound, so a modest thread pool keeps a full
    dashboard refresh (~70 tickers) to a few seconds. Duplicates are de-duped
    while preserving first-seen order.
    """
    seen = list(dict.fromkeys(s for s in symbols if s))
    out: dict[str, Quote] = {}
    if not seen:
        return out
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_quote, s): s for s in seen}
        for fut in cf.as_completed(futures):
            sym = futures[fut]
            try:
                quote = fut.result()
            except Exception as exc:  # noqa: BLE001 — one bad ticker must not sink the batch
                log.warning("quote fetch failed for %s: %s", sym, exc)
                quote = None
            if quote is not None:
                out[sym] = quote
    return out


def market_state(quotes: dict[str, Quote], probe: str = "^GSPC") -> str:
    """Overall market state, taken from the S&P 500 probe (fallback: any quote)."""
    q = quotes.get(probe)
    if q is not None:
        return q.state
    for q in quotes.values():
        return q.state
    return "CLOSED"


def state_label(state: str) -> str:
    return _MARKET_STATE_LABELS.get(state, state.title())


def quotes_to_frame(
    quotes: dict[str, Quote], spec: list[tuple[str, str]]
) -> pd.DataFrame:
    """Build a display frame for a labelled ticker set (indexes/styles/sectors).

    Columns: ticker, label, price, prev_close, pct, state. Rows with no quote are
    kept with NaN so the dashboard can show "—" rather than silently dropping a
    tile the user expects to see.
    """
    rows = []
    for ticker, label in spec:
        q = quotes.get(ticker)
        rows.append(
            {
                "ticker": ticker,
                "label": label,
                "price": q.price if q else None,
                "prev_close": q.prev_close if q else None,
                "pct": q.pct if q else None,
                "state": q.state if q else None,
            }
        )
    return pd.DataFrame(rows)
