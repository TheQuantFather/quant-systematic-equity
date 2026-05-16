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
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from config import DATA_DIR, RETURNS_DB, UNIVERSE_DB
from utils import get_db

HISTORY_START = "2020-01-01"   # earliest date fetched for tickers with no existing data
YAHOO_DELAY   = 0.15           # seconds between per-ticker requests (avoids rate-limiting)

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
            ticker_to_isin = dict(uc.execute(
                "SELECT ticker, isin FROM companies "
                "WHERE ticker IS NOT NULL AND ticker != '' AND isin IS NOT NULL"
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
        print(f"  {already_current:,} tickers already current — skipping")
    print(f"  {len(work):,} tickers to update")

    # Preflight: wait until Yahoo Finance is actually responding before burning
    # through retries on every ticker.  Uses exponential backoff up to 30 minutes.
    print("  Checking Yahoo Finance connectivity ...")
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
                print(
                    f"\n  [ABORT] 20 consecutive failures — Yahoo Finance is rate-limiting.\n"
                    f"  Committed {total_inserted:,} rows so far. Re-run --update later."
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
            print(f"  {i+1:,}/{len(work):,} ({pct:.0f}%) | {total_inserted:,} rows | {errors} errors")

        time.sleep(YAHOO_DELAY)

    conn.commit()
    conn.execute(
        "INSERT OR REPLACE INTO metadata VALUES ('last_yahoo_update', ?)",
        (today_str,),
    )
    conn.commit()
    print(f"\nYahoo update complete — {total_inserted:,} rows | {errors} errors")


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
            print("  Yahoo Finance is responding — starting update")
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(f"  [WARN] Yahoo Finance still rate-limited after {max_wait_minutes}m — proceeding anyway")
            return
        actual_wait = min(wait, remaining)
        print(f"  Rate-limited — waiting {actual_wait:.0f}s (up to {max_wait_minutes}m total)...")
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
    print("\nRunning integrity checks …")
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

    return passed


def _check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    detail_str = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{detail_str}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(conn: sqlite3.Connection) -> None:
    total    = conn.execute("SELECT COUNT(*) FROM returns").fetchone()[0]
    n_isins  = conn.execute("SELECT COUNT(DISTINCT isin) FROM returns").fetchone()[0]
    min_d, max_d = conn.execute("SELECT MIN(date), MAX(date) FROM returns").fetchone()
    meta     = dict(conn.execute("SELECT key, value FROM metadata").fetchall())

    print("\n── Returns DB Summary ──────────────────────────")
    print(f"  returns table rows:  {total:,}")
    print(f"  ISINs:               {n_isins:,}")
    print(f"  Date range:          {min_d} → {max_d}")
    for k, v in meta.items():
        print(f"  {k}: {v}")
    print("────────────────────────────────────────────────")


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
    parser.add_argument("--history-start", default=HISTORY_START,
                        help=f"Start date for tickers with no existing data (default {HISTORY_START})")
    args = parser.parse_args()

    if not any([args.update, args.check]):
        parser.print_help()
        return

    conn = connect()
    try:
        if args.update:
            update_from_yahoo(conn, history_start=args.history_start)

        if args.check or args.update:
            run_checks(conn)

        print_summary(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
