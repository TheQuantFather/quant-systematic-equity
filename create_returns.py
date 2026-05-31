#!/usr/bin/env python3
"""
create_returns.py — Build and maintain the daily price / returns database.

Data source: Yahoo Finance via yfinance (no API key required).
Historical data was loaded via SimFin CSV — that path is complete and removed.

Column mapping:
  close     — raw closing price (used for market-cap / value factor calculations)
  adj_close — fully adjusted for splits and dividends (total return; used for momentum)

Usage:
  python create_returns.py --update   # pull new/missing prices from Yahoo Finance
  python create_returns.py --check    # run integrity checks only
"""

import argparse
import json
import shutil
import sqlite3
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from config import DATA_DIR, RETURNS_DB, UNIVERSE_DB
from utils import get_db, get_logger

log = get_logger("create_returns")

HISTORY_START     = "2020-01-01"   # earliest date fetched for universe stocks
ETF_HISTORY_START = "2010-01-01"   # ETFs: more history for momentum strategy backtesting
YAHOO_DELAY       = 0.15           # seconds between per-ticker requests (avoids rate-limiting)

# ---------------------------------------------------------------------------
# ETF universe defaults
# Seeded into etf_universe table on first run; DB is the source of truth after that.
# To add a new ETF:
#   INSERT INTO etf_universe (ticker, name, asset_class, region) VALUES ('XYZ', ...);
# To disable without deleting: UPDATE etf_universe SET active = 0 WHERE ticker = 'XYZ';
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

DDL = """
CREATE TABLE IF NOT EXISTS returns (
    isin         TEXT    NOT NULL,
    date         DATE    NOT NULL,
    total_return REAL,
    close        REAL,
    volume       INTEGER,
    ccy          TEXT    NOT NULL DEFAULT 'USD',
    PRIMARY KEY (isin, date)
);
CREATE INDEX IF NOT EXISTS idx_returns_isin ON returns (isin);
CREATE INDEX IF NOT EXISTS idx_returns_date ON returns (date);

CREATE TABLE IF NOT EXISTS benchmark_returns (
    index_name   TEXT NOT NULL,
    date         TEXT NOT NULL,
    close        REAL,
    total_return REAL,
    PRIMARY KEY (index_name, date)
);
CREATE INDEX IF NOT EXISTS idx_bench_date ON benchmark_returns (date);

CREATE TABLE IF NOT EXISTS etf_dividends (
    ticker  TEXT NOT NULL,
    ex_date TEXT NOT NULL,
    amount  REAL NOT NULL,
    PRIMARY KEY (ticker, ex_date)
);
CREATE INDEX IF NOT EXISTS idx_etfdiv_ticker ON etf_dividends (ticker);

CREATE TABLE IF NOT EXISTS metadata (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(RETURNS_DB)
    conn.execute("PRAGMA journal_mode=WAL")
    for stmt in DDL.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Yahoo Finance update
# ---------------------------------------------------------------------------

def update_from_yahoo(
    conn: sqlite3.Connection,
    history_start: str = HISTORY_START,
) -> None:
    """
    Pull daily returns from Yahoo Finance for all universe tickers, one at a time.
    Writes pre-computed total_return rows into the `returns` table (keyed by ISIN).
    Uses per-isin last date so each ticker fetches only its own gap.
    Progress is committed every 50 tickers so re-runs resume from where they left off.
    """
    today_str = date.today().strftime("%Y-%m-%d")

    # Load ticker → ISIN mapping for universe tickers
    ticker_to_isin: dict[str, str] = {}
    if UNIVERSE_DB.exists():
        with get_db(UNIVERSE_DB) as uc:
            # delisted_date IS NULL → live / still-trading names (incl. those that
            # merely dropped out of the index). Truly-delisted names (delisted_date
            # set) have no Yahoo data and are sourced from FMP via --backfill-delisted;
            # fetching them here would only rack up failures and risk the abort guard.
            has_col = any(r[1] == "delisted_date"
                          for r in uc.execute("PRAGMA table_info(companies)").fetchall())
            where_live = "AND delisted_date IS NULL" if has_col else ""
            ticker_to_isin = dict(uc.execute(
                "SELECT ticker, isin FROM companies "
                f"WHERE ticker IS NOT NULL AND ticker != '' AND isin IS NOT NULL {where_live}"
            ).fetchall())

    # Per-isin last date in returns table
    per_isin_last: dict[str, str] = dict(conn.execute(
        "SELECT isin, MAX(date) FROM returns GROUP BY isin"
    ).fetchall())

    # Build work list: (ticker, isin, fetch_from_date)
    # fetch_from = last date in returns (to use as adj_close anchor) or history_start
    work: list[tuple[str, str, str]] = []
    for ticker, isin in sorted(ticker_to_isin.items()):
        last = per_isin_last.get(isin)
        from_str = last if last else history_start
        if from_str <= today_str:
            work.append((ticker, isin, from_str))

    already_current = len(ticker_to_isin) - len(work)
    if already_current:
        log.info("%s tickers already current — skipping", f"{already_current:,}")
    log.info("%s tickers to update", f"{len(work):,}")

    # Preflight: wait until Yahoo Finance is actually responding before burning
    # through retries on every ticker.  Uses exponential backoff up to 30 minutes.
    log.info("Checking Yahoo Finance connectivity ...")
    _wait_for_yahoo()

    total_inserted = 0
    errors         = 0
    consec_none    = 0   # consecutive None returns — used to detect active rate-limit

    for i, (ticker, isin, from_str) in enumerate(work):
        raw_rows = _yahoo_ticker(ticker, from_str, today_str, fast_fail_on_429=True)

        if raw_rows is None:
            errors      += 1
            consec_none += 1
            if consec_none >= 20:
                conn.commit()
                log.error(
                    "[ABORT] 20 consecutive failures — Yahoo Finance is rate-limiting. "
                    "Committed %s rows so far. Re-run --update later.",
                    f"{total_inserted:,}",
                )
                return
        else:
            consec_none = 0
            if raw_rows:
                # raw_rows: list of (ticker, date, open, high, low, close, adj_close, volume, ...)
                # Compute total_return = adj_close[i] / adj_close[i-1] - 1
                ret_rows: list[tuple] = []
                for j, row in enumerate(raw_rows):
                    d, c, ac, v = row[1], row[5], row[6], row[7]
                    if j == 0:
                        total_return = None  # anchor row — will be skipped by INSERT OR IGNORE
                    else:
                        prev_ac = raw_rows[j - 1][6]
                        total_return = (
                            float(ac) / float(prev_ac) - 1
                            if prev_ac and prev_ac > 0 and ac and ac > 0
                            else None
                        )
                    ret_rows.append((isin, d, total_return, c, v, "USD"))

                conn.executemany(
                    "INSERT OR IGNORE INTO returns "
                    "(isin, date, total_return, close, volume, ccy) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    ret_rows,
                )
                # Count genuinely new rows (dates after from_str)
                new_count = sum(1 for r in ret_rows if r[1] > from_str)
                total_inserted += new_count

        # Commit and print progress every 50 tickers
        if (i + 1) % 50 == 0:
            conn.commit()
            pct = (i + 1) / len(work) * 100
            log.info("%s/%s (%.0f%%) | %s rows | %d errors",
                     f"{i+1:,}", f"{len(work):,}", pct, f"{total_inserted:,}", errors)

        time.sleep(YAHOO_DELAY)

    conn.commit()
    conn.execute(
        "INSERT OR REPLACE INTO metadata VALUES ('last_yahoo_update', ?)",
        (today_str,),
    )
    conn.commit()
    log.info("Yahoo update complete — %s rows | %d errors", f"{total_inserted:,}", errors)


# ---------------------------------------------------------------------------
# Index / benchmark returns + dividends (unified)
# ---------------------------------------------------------------------------

def update_index_returns(
    conn: sqlite3.Connection,
    history_start: str = ETF_HISTORY_START,
) -> None:
    """
    Pull daily prices and dividend events for every entry in universe.db
    index_registry (benchmarks and investable ETFs alike).

    Prices     → benchmark_returns  keyed by index_name (e.g. 'sp500', 'efa').
    Dividends  → etf_dividends      keyed by etf_ticker (the Yahoo Finance ticker).

    All entries use the same history_start so the table is uniform.
    The is_investable flag in index_registry is for strategy code only — price
    and dividend fetching is identical regardless of that flag.
    """
    today_str = date.today().strftime("%Y-%m-%d")

    if not UNIVERSE_DB.exists():
        log.warning("universe.db not found — skipping index update")
        return

    with get_db(UNIVERSE_DB) as uc:
        registry: list[tuple[str, str]] = uc.execute(
            "SELECT index_name, etf_ticker FROM index_registry "
            "WHERE etf_ticker IS NOT NULL ORDER BY index_name"
        ).fetchall()

    if not registry:
        log.warning("index_registry is empty — nothing to fetch")
        return

    index_names = [r[0] for r in registry]
    etf_tickers = [r[1] for r in registry]

    ph_i = ",".join("?" * len(index_names))
    ph_t = ",".join("?" * len(etf_tickers))

    per_price_last: dict[str, str] = dict(conn.execute(
        f"SELECT index_name, MAX(date) FROM benchmark_returns "
        f"WHERE index_name IN ({ph_i}) GROUP BY index_name",
        index_names,
    ).fetchall())
    per_div_last: dict[str, str] = dict(conn.execute(
        f"SELECT ticker, MAX(ex_date) FROM etf_dividends "
        f"WHERE ticker IN ({ph_t}) GROUP BY ticker",
        etf_tickers,
    ).fetchall())

    log.info("Updating %d index(es) ...", len(registry))
    total_price_rows = 0
    total_div_rows   = 0

    for index_name, etf_ticker in registry:
        # ── prices ────────────────────────────────────────────────────────
        from_price = per_price_last.get(index_name) or history_start
        raw_rows   = _yahoo_ticker(etf_ticker, from_price, today_str, retries=3)

        if raw_rows is None:
            log.warning("[%-30s] %s  price fetch failed", index_name, etf_ticker)
        elif raw_rows:
            price_rows: list[tuple] = []
            for j, row in enumerate(raw_rows):
                d, c, ac = row[1], row[5], row[6]
                if j == 0:
                    tr = None
                else:
                    prev_ac = raw_rows[j - 1][6]
                    tr = (
                        float(ac) / float(prev_ac) - 1
                        if prev_ac and prev_ac > 0 and ac and ac > 0
                        else None
                    )
                price_rows.append((index_name, d, float(c) if c else None, tr))

            conn.executemany(
                "INSERT OR IGNORE INTO benchmark_returns "
                "(index_name, date, close, total_return) VALUES (?, ?, ?, ?)",
                price_rows,
            )
            new_count = sum(1 for r in price_rows if r[1] > from_price)
            total_price_rows += new_count
            if new_count:
                log.info("[%-30s] %-6s  +%d rows", index_name, etf_ticker, new_count)

        time.sleep(YAHOO_DELAY)

        # ── dividends ─────────────────────────────────────────────────────
        from_div = per_div_last.get(etf_ticker) or history_start
        div_rows = _yahoo_dividends(etf_ticker, from_div, today_str, retries=3)

        if div_rows is None:
            log.warning("[%-30s] %s  dividend fetch failed", index_name, etf_ticker)
        elif div_rows:
            new_divs = [(t, d, a) for t, d, a in div_rows if d > from_div]
            if new_divs:
                conn.executemany(
                    "INSERT OR IGNORE INTO etf_dividends (ticker, ex_date, amount) "
                    "VALUES (?, ?, ?)",
                    new_divs,
                )
                total_div_rows += len(new_divs)
                log.info("[%-30s] %-6s  dividends +%d events", index_name, etf_ticker, len(new_divs))

        time.sleep(YAHOO_DELAY)

    conn.commit()
    log.info(
        "Index update complete — %d price rows | %d dividend events",
        total_price_rows, total_div_rows,
    )


def _yahoo_dividends(
    ticker: str,
    from_date: str,
    to_date: str,
    retries: int = 2,
) -> list[tuple[str, str, float]] | None:
    """
    Fetch dividend ex-dates and amounts for one ETF from the Yahoo Finance v8
    chart API.  Returns a list of (ticker, ex_date, amount) tuples sorted by
    ex_date, None on permanent failure, or [] if no dividends in the window.
    """
    t1 = int(datetime.strptime(from_date, "%Y-%m-%d")
             .replace(tzinfo=timezone.utc).timestamp())
    t2 = int((datetime.strptime(to_date, "%Y-%m-%d")
              .replace(tzinfo=timezone.utc) + timedelta(days=1)).timestamp())

    url = (
        f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?interval=1d&period1={t1}&period2={t2}&events=div"
    )

    data = _fetch_url(url, retries=retries, fast_fail_on_429=False)
    if data is None:
        return None
    if data == {}:
        return []

    try:
        events    = data["chart"]["result"][0].get("events", {})
        dividends = events.get("dividends", {})
    except (KeyError, IndexError, TypeError):
        return []

    rows: list[tuple[str, str, float]] = []
    for div_data in dividends.values():
        ts     = div_data.get("date")
        amount = div_data.get("amount")
        if ts is None or not amount or amount <= 0:
            continue
        ex_date = datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")
        rows.append((ticker, ex_date, float(amount)))

    return sorted(rows, key=lambda x: x[1])


def _wait_for_yahoo(max_wait_minutes: int = 30) -> None:
    """
    Block until Yahoo Finance returns a valid response for SPY or the timeout expires.
    Uses exponential backoff: 30s → 60s → 120s → 300s → 300s → ...
    Prints a success message when Yahoo responds, or a warning on timeout.
    """
    deadline = time.monotonic() + max_wait_minutes * 60
    wait = 30
    while True:
        rows = _yahoo_ticker("SPY", "2026-01-01", "2026-01-02", retries=0)
        if rows is not None:   # None = permanent error; [] = no data; list = success
            log.info("Yahoo Finance is responding — starting update")
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            log.warning("Yahoo Finance still rate-limited after %dm — proceeding anyway", max_wait_minutes)
            return
        actual_wait = min(wait, remaining)
        log.info("Rate-limited — waiting %.0fs (up to %dm total)...", actual_wait, max_wait_minutes)
        time.sleep(actual_wait)
        wait = min(wait * 2, 300)


_YF_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


_CURL_BIN = shutil.which("curl")   # None if curl not on PATH


def _fetch_url(url: str, retries: int, fast_fail_on_429: bool) -> dict | None:
    """
    Fetch a URL and return parsed JSON, or None on unrecoverable failure.
    Prefers curl (bypasses Python TLS fingerprinting blocks) over urllib.
    """
    if _CURL_BIN:
        # curl impersonates a browser TLS fingerprint; -w writes HTTP code to stdout
        # after the body, separated by a sentinel we strip.
        for attempt in range(retries + 1):
            result = subprocess.run(
                [
                    _CURL_BIN, "-sS", "--max-time", "20",
                    "-A", _YF_HEADERS["User-Agent"],
                    "-H", f"Accept: {_YF_HEADERS['Accept']}",
                    "-w", "\n__STATUS__%{http_code}",
                    url,
                ],
                capture_output=True, text=True,
            )
            raw = result.stdout
            if "__STATUS__" in raw:
                body, status_str = raw.rsplit("__STATUS__", 1)
                status = int(status_str.strip())
            else:
                body, status = raw, 0

            if status == 200:
                try:
                    return json.loads(body)
                except json.JSONDecodeError:
                    return None
            if status == 404:
                return {}   # sentinel: ticker not found
            if status == 429:
                if fast_fail_on_429:
                    return None
                time.sleep(60.0 * (2 ** attempt))
            elif attempt < retries:
                time.sleep(2.0 * (attempt + 1))
            else:
                return None
        return None

    # Fallback: urllib (may be blocked by TLS fingerprinting)
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=_YF_HEADERS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {}
            if e.code == 429:
                if fast_fail_on_429:
                    return None
                time.sleep(60.0 * (2 ** attempt))
            elif attempt < retries:
                time.sleep(2.0 * (attempt + 1))
            else:
                return None
        except Exception:
            if attempt == retries:
                return None
            time.sleep(2.0)
    return None


def _yahoo_ticker(
    ticker: str,
    from_date: str,
    to_date: str,
    retries: int = 2,
    fast_fail_on_429: bool = False,
) -> list | None:
    """
    Fetch daily OHLCV + adj_close for one ticker from Yahoo Finance v8 chart API.
    Returns a list of row tuples on success, None on permanent failure, [] if no data.

    adj_close is Yahoo's fully adjusted price (splits + dividends), used for total-
    return momentum.  close is the raw closing price, used for market-cap calculations.
    """
    t1 = int(datetime.strptime(from_date, "%Y-%m-%d")
             .replace(tzinfo=timezone.utc).timestamp())
    t2 = int((datetime.strptime(to_date, "%Y-%m-%d")
              .replace(tzinfo=timezone.utc) + timedelta(days=1)).timestamp())

    url = (
        f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?interval=1d&period1={t1}&period2={t2}"
        f"&events=history&includeAdjustedClose=true"
    )

    data = _fetch_url(url, retries=retries, fast_fail_on_429=fast_fail_on_429)
    if data is None:
        return None
    if data == {}:
        return []   # 404 / ticker not found

    try:
        result     = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        q          = result["indicators"]["quote"][0]
        adj_close  = result["indicators"]["adjclose"][0]["adjclose"]
    except (KeyError, IndexError, TypeError):
        return []

    rows = []
    for i, ts in enumerate(timestamps):
        close = q["close"][i] if q["close"][i] is not None else None
        if close is None or close <= 0:
            continue
        dt_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        rows.append((
            ticker,
            dt_str,
            q["open"][i],
            q["high"][i],
            q["low"][i],
            float(close),
            float(adj_close[i]) if adj_close[i] is not None else None,
            int(q["volume"][i]) if q["volume"][i] is not None else None,
            None,   # dividend (baked into adj_close)
            None,   # shares_outstanding (sourced from constituents.db in factors)
            "yahoo",
        ))
    return rows


# ---------------------------------------------------------------------------
# Integrity checks
# ---------------------------------------------------------------------------

def run_checks(conn: sqlite3.Connection) -> bool:
    log.info("Running integrity checks ...")
    passed = True

    total = conn.execute("SELECT COUNT(*) FROM returns").fetchone()[0]
    _check("Row count > 0 (returns)", total > 0, f"{total:,} rows")

    null_close = conn.execute(
        "SELECT COUNT(*) FROM returns WHERE close IS NULL"
    ).fetchone()[0]
    _check("No NULL close prices (returns)", null_close == 0, f"{null_close:,} nulls")

    neg = conn.execute("SELECT COUNT(*) FROM returns WHERE close < 0").fetchone()[0]
    _check("No negative close prices (returns)", neg == 0, f"{neg:,} rows with close < 0")

    min_d, max_d = conn.execute("SELECT MIN(date), MAX(date) FROM returns").fetchone()
    _check(
        "Date range looks valid (returns)",
        min_d is not None and max_d is not None and min_d < max_d,
        f"{min_d} → {max_d}",
    )

    n_isins = conn.execute(
        "SELECT COUNT(DISTINCT isin) FROM returns"
    ).fetchone()[0]
    _check("ISIN count > 100 (returns)", n_isins > 100, f"{n_isins:,} ISINs")

    dupes = conn.execute(
        "SELECT COUNT(*) FROM (SELECT isin, date, COUNT(*) c "
        "FROM returns GROUP BY isin, date HAVING c > 1)"
    ).fetchone()[0]
    _check("No duplicate (isin, date) pairs", dupes == 0, f"{dupes:,} dupes found")

    meta_rows = conn.execute("SELECT key, value FROM metadata").fetchall()
    _check("Metadata table populated", len(meta_rows) > 0, str(dict(meta_rows)))

    # Index checks
    n_index = conn.execute(
        "SELECT COUNT(DISTINCT index_name) FROM benchmark_returns"
    ).fetchone()[0]
    _check("benchmark_returns populated", n_index > 0, f"{n_index} indexes")

    n_div_tickers = conn.execute(
        "SELECT COUNT(DISTINCT ticker) FROM etf_dividends"
    ).fetchone()[0]
    _check("etf_dividends populated", n_div_tickers > 0, f"{n_div_tickers} tickers with dividend history")

    sp500_min = conn.execute(
        "SELECT MIN(date) FROM benchmark_returns WHERE index_name = 'sp500'"
    ).fetchone()[0]
    _check(
        "Index history reaches 2010 (sp500)",
        sp500_min is not None and sp500_min <= "2011-01-01",
        f"sp500 starts {sp500_min}",
    )

    return passed


def _check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    msg = f"[{status}] {label}" + (f"  ({detail})" if detail else "")
    if condition:
        log.info(msg)
    else:
        log.error(msg)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(conn: sqlite3.Connection) -> None:
    total    = conn.execute("SELECT COUNT(*) FROM returns").fetchone()[0]
    n_isins  = conn.execute("SELECT COUNT(DISTINCT isin) FROM returns").fetchone()[0]
    min_d, max_d = conn.execute("SELECT MIN(date), MAX(date) FROM returns").fetchone()
    meta     = dict(conn.execute("SELECT key, value FROM metadata").fetchall())

    log.info("Returns DB: %s rows | %s ISINs | %s → %s",
             f"{total:,}", f"{n_isins:,}", min_d, max_d)
    for k, v in meta.items():
        log.info("  %s: %s", k, v)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _float(val) -> float | None:
    try:
        return float(val) if val not in (None, "", "nan") else None
    except (ValueError, TypeError):
        return None


def _int(val) -> int | None:
    try:
        return int(float(val)) if val not in (None, "", "nan") else None
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Delisted-name price backfill from FMP  (--backfill-delisted)
#
# Truly-delisted names (companies.delisted_date IS NOT NULL) have no Yahoo data —
# Yahoo purges delisted tickers and recycles them.  FMP retains dividend-adjusted
# history for delisted symbols, so they are sourced here, keyed by their (dead)
# ISIN so universe_snapshots membership resolves.  Resumable: only names still
# missing from returns are fetched, and FMP's daily quota simply caps how many
# land per run — re-run on subsequent days to finish.
# ---------------------------------------------------------------------------

_FMP_BASE = "https://financialmodelingprep.com/stable"

# FMP keys that hit their daily cap (429) this run — skipped thereafter.
_FMP_EXHAUSTED: set[str] = set()


def _load_fmp_api_keys() -> list[str]:
    """All FMP keys from .env (FMP_API_KEY, FMP_API_KEY_SECOND, …) — free keys
    multiply the daily quota; rotation falls through when one hits its cap."""
    env = Path(".env")
    if not env.exists():
        return []
    keys: list[str] = []
    for line in env.read_text().splitlines():
        s = line.strip()
        if s.startswith("FMP_API_KEY") and "=" in s:
            val = s.split("=", 1)[1].strip()
            if val and val not in keys:
                keys.append(val)
    return keys


def _fmp_eod_adjusted(ticker: str, from_date: str, to_date: str, keys: list[str]) -> list | None:
    """Fetch FMP dividend-adjusted daily EOD for one ticker, rotating across keys.

    Returns [(date, adj_close, volume), ...] oldest-first, [] if no data, or None
    when all keys are exhausted / error (caller treats None as 'resume next run').
    """
    sym = urllib.parse.quote(ticker.replace("/", "-"))
    data = None
    for k in keys:
        if k in _FMP_EXHAUSTED:
            continue
        url = (f"{_FMP_BASE}/historical-price-eod/dividend-adjusted"
               f"?symbol={sym}&from={from_date}&to={to_date}&apikey={k}")
        broke = False
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    data = json.load(resp)
                broke = True
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    if attempt < 2:
                        time.sleep(15)  # transient per-minute throttle — wait, retry
                        continue
                    _FMP_EXHAUSTED.add(k)  # sustained 429 → daily cap; next key
                    break
                return None
            except Exception as e:
                log.warning("FMP fetch failed for %s: %s", ticker, e)
                return None
        if broke:
            break
    if data is None:
        return None  # all keys exhausted
    if not isinstance(data, list) or not data:
        return []
    rows = []
    for d in data:
        ac = d.get("adjClose")
        if ac is None or ac <= 0:
            continue
        rows.append((d["date"], float(ac), d.get("volume")))
    rows.sort(key=lambda r: r[0])
    return rows


# Tiingo is the working free source for *delisted* daily history (FMP free
# premium-locks delisted symbols; Yahoo purges them). Free tier ≈ 50 symbols/hour,
# so the backfill paces conservatively and is resumable.
_TIINGO_BASE = "https://api.tiingo.com/tiingo/daily"


def _load_tiingo_token() -> str | None:
    env = Path(".env")
    if not env.exists():
        return None
    for line in env.read_text().splitlines():
        s = line.strip()
        if s.startswith("TIINGO_API_KEY") and "=" in s:
            return s.split("=", 1)[1].strip() or None
    return None


def _tiingo_eod(ticker: str, from_date: str, to_date: str, token: str) -> list | None:
    """Fetch Tiingo split/dividend-adjusted daily EOD for one ticker.

    Returns [(date, adj_close, volume), ...] oldest-first, [] if no data, or None
    on rate-limit (429) / hard error so the caller can stop and resume next run.
    """
    sym = urllib.parse.quote(ticker.replace("/", "-"))
    url = f"{_TIINGO_BASE}/{sym}/prices?startDate={from_date}&endDate={to_date}&token={token}"
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        if e.code == 429:
            log.warning("Tiingo rate-limit (429) on %s — stopping; re-run later to resume", ticker)
            return None
        log.warning("Tiingo %s for %s", e.code, ticker)
        return []          # 404 / not found — treat as no data, keep going
    except Exception as e:
        log.warning("Tiingo fetch failed for %s: %s", ticker, e)
        return None
    if not isinstance(data, list) or not data:
        return []
    rows = []
    for d in data:
        ac = d.get("adjClose")
        if ac is None or ac <= 0:
            continue
        rows.append((d["date"][:10], float(ac), d.get("adjVolume") or d.get("volume")))
    rows.sort(key=lambda r: r[0])
    return rows


def backfill_delisted(conn: sqlite3.Connection, history_start: str = HISTORY_START) -> None:
    """Backfill delisted-name daily prices (Tiingo) for names missing from returns."""
    token = _load_tiingo_token()
    if not token:
        log.error("TIINGO_API_KEY not found in .env — cannot backfill delisted prices.")
        return

    today_str = date.today().strftime("%Y-%m-%d")
    have = {r[0] for r in conn.execute("SELECT DISTINCT isin FROM returns").fetchall()}
    with get_db(UNIVERSE_DB) as uc:
        targets = [
            (r[0], r[1]) for r in uc.execute(
                "SELECT isin, ticker FROM companies "
                "WHERE delisted_date IS NOT NULL AND ticker IS NOT NULL AND ticker != ''"
            ).fetchall()
            if r[0] not in have
        ]
    log.info("Delisted price backfill (Tiingo): %d names missing from returns", len(targets))

    n_done = n_rows = n_empty = 0
    for isin, ticker in targets:
        eod = _tiingo_eod(ticker, history_start, today_str, token)
        if eod is None:
            conn.commit()
            log.warning("Stopped after %d names (Tiingo unavailable). Committed %d rows. Re-run to resume.",
                        n_done, n_rows)
            return
        if not eod:
            n_empty += 1
            continue
        ret_rows = []
        for j, (d, ac, vol) in enumerate(eod):
            tr = None if j == 0 else (ac / eod[j - 1][1] - 1 if eod[j - 1][1] > 0 else None)
            # close stored = adjusted close (Tiingo adjClose; delisted names feed no
            # market-cap factors, and the backtest uses total_return).
            ret_rows.append((isin, d, tr, ac, vol, "USD"))
        conn.executemany(
            "INSERT OR IGNORE INTO returns (isin, date, total_return, close, volume, ccy) "
            "VALUES (?, ?, ?, ?, ?, ?)", ret_rows,
        )
        n_rows += len(ret_rows)
        n_done += 1
        conn.commit()  # commit each name — slow paced run, keep progress durable
        if n_done % 10 == 0:
            log.info("  %d / %d names | %s rows", n_done, len(targets), f"{n_rows:,}")
        time.sleep(95)  # ~38/hour — comfortably under Tiingo free 50/hour (rolling window)

    conn.commit()
    log.info("Delisted backfill done: %d names, %s rows (%d had no FMP data)",
             n_done, f"{n_rows:,}", n_empty)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build and maintain the daily prices database."
    )
    parser.add_argument("--update",        action="store_true",
                        help="Pull latest/missing prices from Yahoo Finance")
    parser.add_argument("--check",         action="store_true",
                        help="Run integrity checks only")
    parser.add_argument("--backfill-delisted", action="store_true",
                        help="Backfill FMP dividend-adjusted prices for delisted names "
                             "(companies.delisted_date set) missing from returns")
    parser.add_argument("--history-start", default=HISTORY_START,
                        help=f"Start date for tickers with no existing data (default {HISTORY_START})")
    args = parser.parse_args()

    if not any([args.update, args.check, args.backfill_delisted]):
        parser.print_help()
        return

    conn = connect()
    try:
        if args.update:
            update_from_yahoo(conn, history_start=args.history_start)
            log.info("Updating index returns ...")
            update_index_returns(conn)

        if args.backfill_delisted:
            backfill_delisted(conn, history_start=args.history_start)

        if args.check or args.update:
            run_checks(conn)

        print_summary(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
