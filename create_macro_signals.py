"""
create_macro_signals.py — Fetch and store US macro signals.

Fetches Treasury yields, spreads, commodities, VIX, and economic indicators from
FRED, Yahoo Finance, and CBOE. Respects publication lags to prevent lookahead bias.

Usage:
  python create_macro_signals.py                    # today's date
  python create_macro_signals.py --date 2025-04-01  # single date (repeatable)
  python create_macro_signals.py --backfill         # backfill MACRO_BACKFILL_START → today
  python create_macro_signals.py --clean --backfill # rebuild from scratch
"""

import argparse
import os
import sys
import sqlite3
import time
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

from config import MACRO_DB, MACRO_BACKFILL_START
from macro_db import load_signals_reference
from utils import get_db, get_logger

log = get_logger("create_macro_signals")

FRED_BASE_URL = "https://api.stlouisfed.org/fred"


def _load_fred_api_key() -> str:
    """Read FRED_API_KEY from env or .env file (same pattern as other scripts)."""
    key = os.getenv("FRED_API_KEY", "")
    if key:
        return key
    env_path = Path(".env")
    if not env_path.exists():
        return ""
    for line in env_path.read_text().splitlines():
        s = line.strip()
        if s.startswith("FRED_API_KEY") and "=" in s:
            return s.split("=", 1)[1].strip()
    return ""


FRED_API_KEY = _load_fred_api_key()

# HTTP status codes that warrant a retry (rate limit + transient server errors)
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
# Backoff delays in seconds between successive retry attempts
_RETRY_DELAYS = [5, 15, 45]


def fetch_fred_series(
    series_id: str,
    start_date: str,
    end_date: str,
    fred_units: str | None = None,
) -> dict[str, float]:
    """
    Fetch FRED series observations over a date range. Returns {date_str: value}.

    fred_units: optional FRED transformation (e.g. "pc1" for percent change from year ago).
    Retries up to 3 times on rate-limit (429) and transient server errors (5xx)
    with increasing backoff. Non-retryable HTTP errors (e.g. 400, 401) fail fast.
    Call once per series over the full range — never per-day.
    """
    if not FRED_API_KEY:
        log.warning("FRED_API_KEY not set — skipping FRED series %s", series_id)
        return {}

    url = f"{FRED_BASE_URL}/series/observations"
    params: dict = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date,
    }
    if fred_units:
        params["units"] = fred_units

    last_exc: Exception | None = None
    max_attempts = len(_RETRY_DELAYS) + 1

    for attempt in range(max_attempts):
        if attempt > 0:
            delay = _RETRY_DELAYS[attempt - 1]
            log.warning("Retrying FRED %s in %ds (attempt %d/%d)",
                        series_id, delay, attempt + 1, max_attempts)
            time.sleep(delay)

        try:
            resp = requests.get(url, params=params, timeout=30)
        except requests.RequestException as e:
            log.warning("Connection error for FRED %s: %s", series_id, e)
            last_exc = e
            continue

        if resp.status_code in _RETRYABLE_STATUS:
            last_exc = Exception(f"HTTP {resp.status_code} on {series_id}")
            continue

        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            # Non-retryable client error (e.g. bad series ID, invalid API key)
            log.error("Non-retryable HTTP error for FRED %s: %s", series_id, e)
            return {}

        try:
            data = resp.json()
        except ValueError as e:
            log.error("Failed to parse JSON for FRED series %s: %s", series_id, e)
            return {}

        result: dict[str, float] = {}
        for obs in data.get("observations", []):
            date_str = obs.get("date", "")
            value_str = obs.get("value", ".")
            if not date_str or value_str == ".":
                continue
            try:
                result[date_str] = float(value_str)
            except ValueError:
                continue

        log.debug("Fetched %d observations for FRED %s", len(result), series_id)
        return result

    log.error("All %d attempts exhausted for FRED %s: %s", max_attempts, series_id, last_exc)
    return {}


def fetch_yahoo_ticker(ticker: str, start_date: str, end_date: str) -> dict[str, float]:
    """
    Fetch Yahoo Finance ticker data. Returns {date_str: close_price}.

    yfinance's end date is exclusive, so 1 day is added internally.
    """
    # yfinance end is exclusive — add 1 day so the requested date is included
    end_exclusive = (
        datetime.fromisoformat(end_date) + timedelta(days=1)
    ).strftime("%Y-%m-%d")

    try:
        df = yf.download(
            ticker, start=start_date, end=end_exclusive,
            progress=False, auto_adjust=True,
        )
    except Exception as e:
        log.error("Failed to download Yahoo ticker %s: %s", ticker, e)
        return {}

    if df.empty:
        log.warning("No data for Yahoo ticker %s between %s and %s",
                    ticker, start_date, end_date)
        return {}

    # Flatten multi-level columns (yfinance ≥ 0.2 returns MultiIndex for some calls)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close_series = df["Close"].dropna()
    result = {idx.strftime("%Y-%m-%d"): float(v) for idx, v in close_series.items()}
    log.debug("Fetched %d prices for Yahoo %s", len(result), ticker)
    return result


def fetch_signal(
    signal_id: str, signal_meta: dict, start_date: str, end_date: str
) -> dict[str, float]:
    """
    Dispatch to the correct data source. Returns {date_str: value}.
    """
    source = signal_meta.get("source", "")
    endpoint = signal_meta.get("api_endpoint", "")

    if not source or not endpoint:
        log.error("Missing source or endpoint for signal %s", signal_id)
        return {}

    if source == "FRED":
        return fetch_fred_series(endpoint, start_date, end_date, signal_meta.get("fred_units"))
    elif source in ("Yahoo", "CBOE"):
        return fetch_yahoo_ticker(endpoint, start_date, end_date)

    log.error("Unknown source '%s' for signal %s", source, signal_id)
    return {}


def _bulk_insert(rows: list[tuple]) -> int:
    """
    Bulk-insert (published_date, signal_id, value, update_date) rows.
    Returns count inserted.
    """
    if not rows:
        return 0
    try:
        with get_db(MACRO_DB) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO daily_signals "
                "(published_date, signal_id, value, update_date) VALUES (?, ?, ?, ?)",
                rows,
            )
            conn.commit()
        return len(rows)
    except sqlite3.Error as e:
        log.error("Database error inserting %d rows: %s", len(rows), e)
        return 0


def _rows_from_data(
    data: dict[str, float], signal_id: str, update_ts: str
) -> list[tuple]:
    """Convert a {date: value} dict to insert-ready row tuples, skipping NaN/None."""
    rows = []
    for d, v in data.items():
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        rows.append((d, signal_id, float(v), update_ts))
    return rows


def process_date(date_str: str) -> None:
    """
    Fetch and insert all signals for a single date.
    Used for incremental updates (--date or default today).
    """
    signals_ref = load_signals_reference()
    if not signals_ref:
        log.error("No signals loaded from signals_reference")
        return

    update_ts = datetime.now().isoformat()
    rows: list[tuple] = []

    for signal_id, meta in signals_ref.items():
        if meta.get("source") not in ("FRED", "Yahoo", "CBOE"):
            continue
        try:
            data = fetch_signal(signal_id, meta, date_str, date_str)
            rows.extend(_rows_from_data(data, signal_id, update_ts))
        except Exception as e:
            log.error("Unexpected error fetching signal %s on %s: %s", signal_id, date_str, e)

    inserted = _bulk_insert(rows)
    log.info("Inserted %d signal values for %s", inserted, date_str)


def backfill_signals() -> None:
    """
    Backfill all signals from MACRO_BACKFILL_START to today.

    Fetches each signal once over the full date range (one FRED/Yahoo call per series),
    then bulk-inserts. This avoids per-day API hammering that triggers FRED rate limits.
    """
    signals_ref = load_signals_reference()
    if not signals_ref:
        log.error("No signals loaded from signals_reference")
        return

    start = MACRO_BACKFILL_START
    end = date.today().strftime("%Y-%m-%d")
    update_ts = datetime.now().isoformat()

    log.info("Backfilling %d signals from %s to %s", len(signals_ref), start, end)

    total_inserted = 0
    for signal_id, meta in signals_ref.items():
        if meta.get("source") not in ("FRED", "Yahoo", "CBOE"):
            log.debug("Skipping signal %s with unsupported source %s",
                      signal_id, meta.get("source"))
            continue

        log.info("Fetching %s [%s / %s]", signal_id, meta["source"], meta["api_endpoint"])
        try:
            data = fetch_signal(signal_id, meta, start, end)
        except Exception as e:
            log.error("Unexpected error fetching signal %s: %s", signal_id, e)
            continue

        if not data:
            log.warning("No data returned for signal %s", signal_id)
            continue

        rows = _rows_from_data(data, signal_id, update_ts)
        inserted = _bulk_insert(rows)
        log.info("  %s: %d rows inserted", signal_id, inserted)
        total_inserted += inserted

    log.info("Backfill complete. Total rows inserted: %d", total_inserted)


def clean_signals() -> None:
    """Drop and recreate daily_signals table."""
    log.info("Cleaning signals table...")
    try:
        with get_db(MACRO_DB) as conn:
            conn.execute("DROP TABLE IF EXISTS daily_signals")
            conn.execute("""
                CREATE TABLE daily_signals (
                    published_date TEXT NOT NULL,
                    signal_id      TEXT NOT NULL,
                    value          REAL NOT NULL,
                    update_date    TEXT NOT NULL,
                    PRIMARY KEY (published_date, signal_id),
                    FOREIGN KEY (signal_id) REFERENCES signals_reference(signal_id)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_signals_published "
                "ON daily_signals(published_date)"
            )
            conn.commit()
        log.info("Signals table rebuilt.")
    except sqlite3.Error as e:
        log.error("Database error while rebuilding signals table: %s", e)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and store US macro signals")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--date", metavar="YYYY-MM-DD", action="append", dest="dates",
                     help="Specific date (repeatable; default=today)")
    grp.add_argument("--backfill", action="store_true",
                     help="Backfill from MACRO_BACKFILL_START to today")
    parser.add_argument("--clean", action="store_true",
                        help="Drop and rebuild daily_signals table (use with --backfill)")

    args = parser.parse_args()

    try:
        if args.clean:
            clean_signals()

        if args.backfill:
            backfill_signals()
        elif args.dates:
            for date_str in args.dates:
                try:
                    datetime.fromisoformat(date_str)
                except ValueError as e:
                    log.error("Invalid date format '%s': %s", date_str, e)
                    sys.exit(1)
                log.info("Processing date: %s", date_str)
                process_date(date_str)
        else:
            today = date.today().strftime("%Y-%m-%d")
            log.info("Processing today: %s", today)
            process_date(today)

        log.info("Done.")

    except KeyboardInterrupt:
        log.warning("Interrupted by user.")
        sys.exit(1)
    except Exception as e:
        log.error("Unexpected error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
