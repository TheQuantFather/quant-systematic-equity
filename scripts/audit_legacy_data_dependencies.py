#!/usr/bin/env python3
"""Audit remaining legacy CSV/SimFin dependencies.

Read-only. This focuses on infrastructure dependencies that matter for moving
the universe pipeline to an N-PORT-first design.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONSTITUENTS_DB, DATA_DIR, FACTORS_DB, PARAMS_FILE, UNIVERSE_DB  # noqa: E402


def _conn(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(str(path))


def _q(path: Path, sql: str, params: tuple = ()) -> pd.DataFrame:
    with _conn(path) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def _print(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def _print_df(df: pd.DataFrame, max_rows: int = 40) -> None:
    if df.empty:
        print("  OK")
        return
    print(df.head(max_rows).to_string(index=False))
    if len(df) > max_rows:
        print(f"  ... {len(df) - max_rows:,} more")


def audit_files() -> None:
    _print("Legacy Local Input Files")
    for rel in ["universe_index", "simfin"]:
        path = DATA_DIR / rel
        files = sorted(p.resolve().relative_to(ROOT) for p in path.glob("*") if p.is_file()) if path.exists() else []
        print(f"{rel}: {len(files)} file(s)")
        for p in files[:30]:
            print(f"  {p}")
        if len(files) > 30:
            print(f"  ... {len(files) - 30:,} more")


def audit_strategy_sources() -> None:
    _print("Active Strategy Sources")
    if not PARAMS_FILE.exists():
        print(f"{PARAMS_FILE} not found.")
        return
    df = pd.read_excel(PARAMS_FILE, sheet_name="Strategies", dtype=str).fillna("")
    active = df[df["active"].str.strip().str.upper().eq("TRUE")].copy()
    cols = [c for c in ["strategy_id", "benchmark_file", "benchmark_index", "universe_index"] if c in active.columns]
    _print_df(active[cols], max_rows=50)
    if "benchmark_file" in active:
        refs = int(active["benchmark_file"].str.strip().ne("").sum())
        print(f"\nactive benchmark_file references: {refs}")


def audit_universe_sources() -> None:
    _print("Universe Snapshot Source Coverage")
    non_nport = _q(
        UNIVERSE_DB,
        """
        SELECT us.index_name, us.snapshot_date, COUNT(*) AS rows,
               SUM(us.weight) AS weight_sum
        FROM universe_snapshots us
        LEFT JOIN nport_accessions na
          ON na.index_name = us.index_name
         AND na.snapshot_date = us.snapshot_date
        WHERE na.accession IS NULL
        GROUP BY us.index_name, us.snapshot_date
        ORDER BY us.snapshot_date DESC, us.index_name
        """,
    )
    print("Snapshots not backed by N-PORT accession")
    _print_df(non_nport)

    nport = _q(
        UNIVERSE_DB,
        """
        SELECT us.index_name, COUNT(DISTINCT us.snapshot_date) AS dates,
               COUNT(*) AS rows, MIN(us.snapshot_date) AS first_date,
               MAX(us.snapshot_date) AS last_date
        FROM universe_snapshots us
        JOIN nport_accessions na
          ON na.index_name = us.index_name
         AND na.snapshot_date = us.snapshot_date
        GROUP BY us.index_name
        ORDER BY us.index_name
        """,
    )
    print("\nN-PORT-backed snapshots")
    _print_df(nport)


def audit_simfin_usage() -> None:
    _print("SimFin Footprint")
    companies = _q(
        UNIVERSE_DB,
        """
        SELECT COUNT(*) AS companies,
               SUM(simfin_id IS NOT NULL) AS with_simfin_id,
               SUM(cik IS NOT NULL AND cik != '') AS with_cik,
               SUM(simfin_id IS NULL) AS without_simfin_id
        FROM companies
        """,
    )
    print("companies")
    _print_df(companies)

    if CONSTITUENTS_DB.exists():
        constituents = _q(
            CONSTITUENTS_DB,
            """
            SELECT COUNT(*) AS rows,
                   COUNT(DISTINCT security_id) AS securities,
                   SUM(security_id GLOB '[0-9]*') AS numeric_simfin_like,
                   SUM(length(security_id)=12 AND substr(security_id,1,2) GLOB '[A-Z][A-Z]') AS isin_like
            FROM constituents
            """,
        )
        print("\nconstituents security_id profile")
        _print_df(constituents)

    if FACTORS_DB.exists():
        factors = _q(
            FACTORS_DB,
            """
            SELECT COUNT(*) AS rows,
                   COUNT(DISTINCT security_id) AS securities,
                   SUM(security_id GLOB '[0-9]*') AS numeric_simfin_like,
                   SUM(length(security_id)=12 AND substr(security_id,1,2) GLOB '[A-Z][A-Z]') AS isin_like
            FROM factors
            """,
        )
        print("\nfactors security_id profile")
        _print_df(factors)


def main() -> None:
    audit_files()
    audit_strategy_sources()
    audit_universe_sources()
    audit_simfin_usage()


if __name__ == "__main__":
    main()
