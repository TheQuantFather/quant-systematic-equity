"""
create_risk.py — Build covariance-based risk model for the optimizer.

Currently: Ledoit-Wolf shrunk sample covariance from trailing 252d daily returns.
Future: drop-in replacement with Barra-style Σ = BFB' + D without touching the optimizer.

Usage:
    python create_risk.py                    # latest snapshot date
    python create_risk.py --date 2025-04-01  # specific date
    python create_risk.py --backfill         # all snapshot dates
"""

import argparse
import io
import json
import sqlite3
import zlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from config import (
    RETURNS_DB, UNIVERSE_DB, RISK_DB, FACTORS_DB,
    LW_LOOKBACK_DAYS as LOOKBACK_DAYS,
    LW_MIN_HISTORY   as MIN_HISTORY,
    LW_WINSOR_CLIP   as WINSOR_CLIP,
)
from utils import get_db


def _get_snapshot_dates() -> list[str]:
    """Discover snapshot dates from factors.db — snapshot_dates table, then factors table."""
    try:
        with get_db(FACTORS_DB) as conn:
            rows = conn.execute(
                "SELECT data_date FROM snapshot_dates ORDER BY data_date"
            ).fetchall()
            if rows:
                return [r[0] for r in rows]
            # Fallback: derive from factors table itself (handles pre-snapshot_dates era)
            rows = conn.execute(
                "SELECT DISTINCT data_date FROM factors ORDER BY data_date"
            ).fetchall()
            if rows:
                return [r[0] for r in rows]
    except Exception:
        pass
    raise RuntimeError(
        "No snapshot dates found in factors.db. Run create_factors.py first."
    )


# ── DB init ──────────────────────────────────────────────────────────────────

def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS covariance_matrix (
            data_date        TEXT    NOT NULL PRIMARY KEY,
            n_stocks         INTEGER NOT NULL,
            shrinkage_coeff  REAL,
            lookback_days    INTEGER NOT NULL,
            matrix_blob      BLOB    NOT NULL,
            isin_list        TEXT    NOT NULL,
            computation_date TEXT    NOT NULL
        )
    """)
    conn.commit()


# ── Data loading ─────────────────────────────────────────────────────────────

def get_universe_isins() -> list[str]:
    with get_db(UNIVERSE_DB) as conn:
        isins = [r[0] for r in conn.execute(
            "SELECT isin FROM companies WHERE isin IS NOT NULL"
        ).fetchall()]
    return isins


def load_returns_window(data_date: str, lookback: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Load a (dates × isins) daily return matrix for the trailing `lookback`
    trading days ending on data_date (inclusive).
    """
    end_dt    = datetime.strptime(data_date, "%Y-%m-%d")
    # Fetch a wider window to guarantee enough trading days after weekends/holidays
    start_str = (end_dt - timedelta(days=lookback + 90)).strftime("%Y-%m-%d")
    end_str   = data_date

    universe_isins = get_universe_isins()

    with get_db(RETURNS_DB) as conn:
        df = pd.read_sql(
            """
            SELECT isin, date, total_return
            FROM returns
            WHERE date > ? AND date <= ?
              AND isin IN ({placeholders})
            ORDER BY date
            """.format(placeholders=",".join("?" * len(universe_isins))),
            conn,
            params=[start_str, end_str] + universe_isins,
        )

    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    wide = df.pivot(index="date", columns="isin", values="total_return")

    # Keep exactly the last `lookback` trading days
    return wide.tail(lookback)


# ── Covariance estimation ─────────────────────────────────────────────────────

def compute_covariance(
    returns_wide: pd.DataFrame,
    min_history: int = MIN_HISTORY,
) -> tuple[np.ndarray, list[str], float]:
    """
    Compute annualised Ledoit-Wolf shrunk covariance matrix.

    Returns (cov_annual, isins, shrinkage_coeff).
    Only stocks with >= min_history non-NaN observations are included.
    Missing days are filled with 0 (no-trade assumption) after the coverage filter.
    """
    valid_cols = returns_wide.columns[returns_wide.notna().sum() >= min_history]
    df = returns_wide[valid_cols].fillna(0.0).clip(-WINSOR_CLIP, WINSOR_CLIP)

    lw = LedoitWolf(assume_centered=False)
    lw.fit(df.values)  # shape (T, N)

    # Scale daily → annualised
    cov_annual = lw.covariance_ * LOOKBACK_DAYS

    return cov_annual, list(df.columns), float(lw.shrinkage_)


# ── Serialisation ─────────────────────────────────────────────────────────────

def pack_matrix(matrix: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, matrix.astype(np.float32))
    return zlib.compress(buf.getvalue(), level=6)


def unpack_matrix(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(zlib.decompress(blob)))


# ── Per-date processing ───────────────────────────────────────────────────────

def process_date(conn: sqlite3.Connection, data_date: str) -> None:
    print(f"  {data_date}: loading returns ...", end=" ", flush=True)

    wide = load_returns_window(data_date)
    if wide.empty:
        print("no returns data — skipping")
        return

    n_raw = wide.shape[1]
    print(f"{n_raw} stocks × {wide.shape[0]} days", end="  →  ", flush=True)

    cov, isins, shrinkage = compute_covariance(wide)

    blob = pack_matrix(cov)

    conn.execute(
        """
        INSERT OR REPLACE INTO covariance_matrix
            (data_date, n_stocks, shrinkage_coeff, lookback_days,
             matrix_blob, isin_list, computation_date)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            data_date,
            len(isins),
            shrinkage,
            LOOKBACK_DAYS,
            blob,
            json.dumps(isins),
            datetime.now().strftime("%Y-%m-%d"),
        ),
    )
    conn.commit()

    print(
        f"cov {len(isins)}×{len(isins)}, "
        f"shrinkage={shrinkage:.3f}, "
        f"blob={len(blob) / 1024:.0f} KB"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build risk covariance model")
    parser.add_argument("--date",     help="Snapshot date YYYY-MM-DD (default: latest)")
    parser.add_argument("--backfill", action="store_true", help="All snapshot dates")
    args = parser.parse_args()

    if args.backfill:
        dates = _get_snapshot_dates()
    elif args.date:
        dates = [args.date]
    else:
        dates = [_get_snapshot_dates()[-1]]

    with get_db(RISK_DB) as conn:
        init_db(conn)

        print(f"Building covariance risk model for {len(dates)} date(s) "
              f"(lookback={LOOKBACK_DAYS}d, Ledoit-Wolf shrinkage) ...\n")

        for d in dates:
            process_date(conn, d)

    print("\nDone.")


if __name__ == "__main__":
    main()
