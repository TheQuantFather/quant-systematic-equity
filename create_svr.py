#!/usr/bin/env python3
"""
create_svr.py — Fetch FINRA regShoDaily short volume data and store in returns.db.

Short Volume Ratio (SVR) = shortParQuantity / totalParQuantity per stock per day.
Aggregated across all FINRA reporting facilities before storage.

The FINRA API has ~6.6M records split into two sequential blocks:
  Block 1 (offsets 0 → ~4.24M): historical data, date-descending (old → early 2026)
  Block 2 (offsets ~4.24M → end): recent data, date-ascending (early 2026 → today)
Recent data always lives in the last ~2.5M records. We probe the record-total
header and start from (total - BACKFILL_BUFFER) for backfill, or
(total - INCREMENTAL_BUFFER) for daily updates.

Usage:
  python create_svr.py                   # incremental update (last date → today)
  python create_svr.py --backfill        # fetch last ~90 trading days from scratch
  python create_svr.py --deep-backfill   # walk full FINRA API from offset 0 (~12-14 months)
  python create_svr.py --check           # print coverage stats, no fetch
"""

import argparse
import sqlite3
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config import RETURNS_DB, UNIVERSE_DB
from utils import get_db, get_logger

log = get_logger("create_svr")

FINRA_URL = "https://api.finra.org/data/group/otcMarket/name/regShoDaily"
PAGE_SIZE = 5000

# Records per trading day ≈ 25,000 (all NMS securities × reporting facilities).
# Backfill buffer covers ~108 trading days; incremental covers ~8 days.
BACKFILL_BUFFER     = 2_700_000
INCREMENTAL_BUFFER  =   200_000

DDL = """
CREATE TABLE IF NOT EXISTS svr_daily (
    isin       TEXT NOT NULL,
    date       DATE NOT NULL,
    svr        REAL NOT NULL,
    short_vol  INTEGER,
    total_vol  INTEGER,
    PRIMARY KEY (isin, date)
);
CREATE INDEX IF NOT EXISTS idx_svr_date ON svr_daily (date);
CREATE INDEX IF NOT EXISTS idx_svr_isin ON svr_daily (isin);
"""


# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------

def setup_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    for stmt in DDL.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)
    conn.commit()


# ---------------------------------------------------------------------------
# FINRA API helpers
# ---------------------------------------------------------------------------

def _get_total_records() -> int:
    """Probe the API with limit=1 to read the record-total header."""
    resp = requests.get(
        FINRA_URL,
        params={"limit": 1, "offset": 0},
        headers={"Accept": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    return int(resp.headers.get("record-total", 0))


def _fetch_page(offset: int, retries: int = 5) -> list[dict]:
    for attempt in range(retries):
        try:
            resp = requests.get(
                FINRA_URL,
                params={"limit": PAGE_SIZE, "offset": offset},
                headers={"Accept": "application/json"},
                timeout=30,
            )
            if resp.status_code == 400:
                # FINRA returns 400 when offset exceeds actual record count
                # (record-total header can be stale). Treat as end of data.
                return []
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                log.warning("Retry %d/%d at offset %s (%s) — waiting %ds", attempt + 1, retries, f"{offset:,}", e, wait)
                time.sleep(wait)
            else:
                raise



def _detect_fields(sample: dict) -> tuple[str, str, str, str]:
    fields = list(sample.keys())
    date_f   = next(f for f in fields if "date"   in f.lower())
    ticker_f = next(f for f in fields if "symbol" in f.lower() or "ticker" in f.lower())
    short_f  = next(f for f in fields if "short"  in f.lower() and "quantity" in f.lower())
    total_f  = next(f for f in fields if "total"  in f.lower() and "quantity" in f.lower())
    return date_f, ticker_f, short_f, total_f


# ---------------------------------------------------------------------------
# Fetch and aggregate
# ---------------------------------------------------------------------------

def _aggregate(raw_rows: list[dict]) -> pd.DataFrame:
    """Aggregate raw FINRA rows across reporting facilities → one row per (isin, date)."""
    if not raw_rows:
        return pd.DataFrame()
    df = pd.DataFrame(raw_rows)
    df["short_vol"] = pd.to_numeric(df["short_vol"], errors="coerce")
    df["total_vol"] = pd.to_numeric(df["total_vol"], errors="coerce")
    agg = (
        df.groupby(["isin", "date"])
        .agg(short_vol=("short_vol", "sum"), total_vol=("total_vol", "sum"))
        .reset_index()
    )
    agg["svr"] = agg["short_vol"] / agg["total_vol"].clip(lower=1)
    return agg


def fetch_svr(
    start_offset: int,
    cutoff_date: str,
    ticker_to_isin: dict[str, str],
    flush_conn: sqlite3.Connection | None = None,
    flush_every_pages: int = 100,
) -> pd.DataFrame:
    """
    Page through FINRA from start_offset, keep rows with date > cutoff_date
    that match our universe tickers. Returns an aggregated DataFrame with
    columns [isin, date, short_vol, total_vol, svr].

    If `flush_conn` is provided, accumulated rows are aggregated and upserted
    every `flush_every_pages` pages so progress survives crashes/kills.  The
    returned DataFrame then contains only the tail buffer (committed by the
    caller's final upsert).  PK is (isin, date) so re-flushing is idempotent.
    """
    # Probe first page at start_offset to detect field names
    probe = _fetch_page(start_offset, retries=5)
    if not probe:
        log.warning("No data at start_offset — nothing to fetch.")
        return pd.DataFrame()

    date_f, ticker_f, short_f, total_f = _detect_fields(probe[0])

    raw_rows: list[dict] = []
    offset = start_offset
    pages  = 0

    while True:
        page = probe if pages == 0 else _fetch_page(offset)
        if not page:
            break

        for row in page:
            row_date = str(row.get(date_f, ""))
            if row_date <= cutoff_date:
                continue
            ticker = str(row.get(ticker_f, "")).upper().split(".")[0]
            isin   = ticker_to_isin.get(ticker)
            if isin is None:
                continue
            raw_rows.append({
                "isin":      isin,
                "date":      row_date,
                "short_vol": row.get(short_f),
                "total_vol": row.get(total_f),
            })

        pages  += 1
        offset += PAGE_SIZE

        if pages % 20 == 0:
            latest = max(str(r.get(date_f, "")) for r in page)
            log.info("Page %d, offset %s | buffer rows: %s | latest date: %s",
                     pages, f"{offset:,}", f"{len(raw_rows):,}", latest)

        if flush_conn is not None and pages > 0 and pages % flush_every_pages == 0 and raw_rows:
            batch = _aggregate(raw_rows)
            n = upsert_svr(flush_conn, batch)
            log.info("Flushed checkpoint — %s rows committed (offset %s)",
                     f"{n:,}", f"{offset:,}")
            raw_rows.clear()

        if len(page) < PAGE_SIZE:
            break  # last page

        time.sleep(0.05)

    return _aggregate(raw_rows)


# ---------------------------------------------------------------------------
# Write to DB
# ---------------------------------------------------------------------------

def upsert_svr(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    rows = [
        (row.isin, row.date, float(row.svr), int(row.short_vol), int(row.total_vol))
        for row in df.itertuples()
        if pd.notna(row.svr) and pd.notna(row.short_vol) and pd.notna(row.total_vol)
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO svr_daily (isin, date, svr, short_vol, total_vol) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------

def print_coverage(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT isin), COUNT(DISTINCT date), "
        "MIN(date), MAX(date) FROM svr_daily"
    ).fetchone()
    total, n_isins, n_dates, min_d, max_d = row
    meta = dict(conn.execute("SELECT key, value FROM metadata WHERE key LIKE 'last_svr%'").fetchall())
    log.info("SVR coverage: %s rows | %s ISINs | %s trading days (%s → %s)",
             f"{total:,}", f"{n_isins:,}", f"{n_dates:,}", min_d, max_d)
    for k, v in meta.items():
        log.info("  %s: %s", k, v)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch FINRA SVR data into returns.db")
    parser.add_argument("--backfill", action="store_true",
                        help="Fetch last ~90 trading days (ignores existing data)")
    parser.add_argument("--deep-backfill", action="store_true", dest="deep_backfill",
                        help="Walk full FINRA API from offset 0 (~12-14 months, 45-60 min)")
    parser.add_argument("--check",   action="store_true",
                        help="Print coverage stats only, no fetch")
    args = parser.parse_args()

    # Load universe ticker → ISIN map (always re-fetch — tickers can change)
    with get_db(UNIVERSE_DB) as u:
        rows = u.execute(
            "SELECT ticker, isin FROM companies WHERE ticker IS NOT NULL AND ticker != ''"
        ).fetchall()
    ticker_to_isin: dict[str, str] = {t.upper(): i for t, i in rows}
    log.info("Universe: %s tickers", f"{len(ticker_to_isin):,}")

    with get_db(RETURNS_DB) as conn:
        setup_db(conn)

        if args.check:
            print_coverage(conn)
            return

        # Determine start offset and cutoff date
        log.info("Probing FINRA record count ...")
        total_records = _get_total_records()
        log.info("Total FINRA records: %s", f"{total_records:,}")

        if args.deep_backfill:
            # Block 1 is date-descending; INSERT OR REPLACE on (isin,date) PK
            # handles overlap with Block 2 cleanly.  Permissive cutoff so every
            # row that matches a universe ticker is kept.
            start_offset = 0
            cutoff_date  = "1900-01-01"
            log.info("Deep backfill mode: walking full FINRA API from offset 0 (%s records)",
                     f"{total_records:,}")
        elif args.backfill:
            start_offset = max(0, total_records - BACKFILL_BUFFER)
            cutoff_date  = (date.today() - timedelta(days=95)).isoformat()
            log.info("Backfill mode: offset %s, cutoff %s", f"{start_offset:,}", cutoff_date)
        else:
            # Incremental: find last date already in DB
            last_row = conn.execute("SELECT MAX(date) FROM svr_daily").fetchone()
            last_date = last_row[0] if last_row and last_row[0] else None

            if last_date is None:
                log.error("No existing SVR data — run with --backfill first.")
                return

            start_offset = max(0, total_records - INCREMENTAL_BUFFER)
            cutoff_date  = last_date
            log.info("Incremental mode: offset %s, cutoff %s (last date in DB)",
                     f"{start_offset:,}", cutoff_date)

        log.info("Fetching SVR data ...")
        # Deep backfill is the only mode long enough to need periodic flush.
        flush_conn = conn if args.deep_backfill else None
        df = fetch_svr(start_offset, cutoff_date, ticker_to_isin, flush_conn=flush_conn)

        if df.empty:
            log.info("No new rows to insert.")
        else:
            n = upsert_svr(conn, df)
            today_str = date.today().isoformat()
            conn.execute(
                "INSERT OR REPLACE INTO metadata VALUES ('last_svr_update', ?)",
                (today_str,),
            )
            conn.commit()
            log.info("Inserted %s rows | dates %s → %s", f"{n:,}", df['date'].min(), df['date'].max())

        print_coverage(conn)


if __name__ == "__main__":
    main()
