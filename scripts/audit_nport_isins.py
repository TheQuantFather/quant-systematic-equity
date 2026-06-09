#!/usr/bin/env python3
"""Audit EDGAR N-PORT ISIN coverage and identifier quality.

Read-only. This checks whether universe_snapshots rows backed by
nport_accessions look reliable enough to become the production identifier
source for index membership.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import UNIVERSE_DB  # noqa: E402

ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")


def _conn() -> sqlite3.Connection:
    return sqlite3.connect(str(UNIVERSE_DB))


def _q(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)


def _print_df(df: pd.DataFrame, *, max_rows: int = 30) -> None:
    if df.empty:
        print("  OK")
        return
    print(df.head(max_rows).to_string(index=False))
    if len(df) > max_rows:
        print(f"  ... {len(df) - max_rows:,} more")


def _fmt(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col.endswith("weight") or col in {"weight", "weight_sum", "market_value"}:
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6f}")
    return out


def _filter_clause(indexes: list[str], snapshot: str | None, alias: str = "us") -> tuple[str, list[str]]:
    filters: list[str] = []
    params: list[str] = []
    if indexes:
        filters.append(f"{alias}.index_name IN ({','.join('?' * len(indexes))})")
        params.extend(indexes)
    if snapshot:
        filters.append(f"{alias}.snapshot_date = ?")
        params.append(snapshot)
    clause = " AND " + " AND ".join(filters) if filters else ""
    return clause, params


def audit_nport_rows(conn: sqlite3.Connection, indexes: list[str], snapshot: str | None) -> pd.DataFrame:
    clause, params = _filter_clause(indexes, snapshot)
    df = _q(
        conn,
        f"""
        SELECT us.index_name, us.snapshot_date, na.accession, na.period_ending,
               COUNT(*) AS rows,
               COUNT(DISTINCT us.isin) AS distinct_isins,
               SUM(us.weight) AS weight_sum,
               SUM(us.weight IS NULL) AS null_weights,
               SUM(us.market_value IS NULL) AS null_market_values,
               SUM(c.isin IS NULL) AS missing_company,
               SUM(c.isin IS NOT NULL) AS has_company
        FROM universe_snapshots us
        JOIN nport_accessions na
          ON na.index_name = us.index_name
         AND na.snapshot_date = us.snapshot_date
        LEFT JOIN companies c ON c.isin = us.isin
        WHERE 1=1 {clause}
        GROUP BY us.index_name, us.snapshot_date, na.accession, na.period_ending
        ORDER BY us.snapshot_date DESC, us.index_name
        """,
        tuple(params),
    )
    print("\nN-PORT-backed snapshot coverage")
    _print_df(_fmt(df), max_rows=60)
    return df


def audit_quality(conn: sqlite3.Connection, indexes: list[str], snapshot: str | None, top: int) -> None:
    clause, params = _filter_clause(indexes, snapshot)
    df = _q(
        conn,
        f"""
        SELECT us.index_name, us.snapshot_date, us.isin, us.weight,
               us.market_value, c.ticker, c.company_name
        FROM universe_snapshots us
        JOIN nport_accessions na
          ON na.index_name = us.index_name
         AND na.snapshot_date = us.snapshot_date
        LEFT JOIN companies c ON c.isin = us.isin
        WHERE 1=1 {clause}
        """,
        tuple(params),
    )
    if df.empty:
        print("\nNo N-PORT-backed rows found for selection.")
        return

    df["valid_isin_format"] = df["isin"].fillna("").map(lambda x: bool(ISIN_RE.match(x)))
    print("\nN-PORT ISIN quality summary")
    print(f"rows:                 {len(df):,}")
    print(f"invalid ISIN format:  {int((~df['valid_isin_format']).sum()):,}")
    print(f"duplicate rows:       {int(df.duplicated(['index_name', 'snapshot_date', 'isin']).sum()):,}")
    print(f"missing companies:    {int(df['ticker'].isna().sum()):,}")
    print(f"null weights:         {int(df['weight'].isna().sum()):,}")
    print(f"null market values:   {int(df['market_value'].isna().sum()):,}")

    invalid = df[~df["valid_isin_format"]]
    print("\nInvalid ISIN examples")
    _print_df(_fmt(invalid), max_rows=top)

    missing = df[df["ticker"].isna()].sort_values(["snapshot_date", "index_name", "weight"], ascending=[False, True, False])
    print("\nN-PORT ISINs missing from companies")
    _print_df(_fmt(missing), max_rows=top)


def audit_non_nport_snapshots(conn: sqlite3.Connection, indexes: list[str], snapshot: str | None) -> None:
    clause, params = _filter_clause(indexes, snapshot)
    df = _q(
        conn,
        f"""
        SELECT us.index_name, us.snapshot_date, COUNT(*) AS rows,
               SUM(us.weight) AS weight_sum
        FROM universe_snapshots us
        LEFT JOIN nport_accessions na
          ON na.index_name = us.index_name
         AND na.snapshot_date = us.snapshot_date
        WHERE na.accession IS NULL {clause}
        GROUP BY us.index_name, us.snapshot_date
        ORDER BY us.snapshot_date DESC, us.index_name
        """,
        tuple(params),
    )
    print("\nUniverse snapshots not backed by N-PORT accession")
    _print_df(_fmt(df), max_rows=60)


def audit_staged_metadata(conn: sqlite3.Connection, indexes: list[str], snapshot: str | None, top: int) -> None:
    print("\nN-PORT staged security metadata")
    exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='nport_security_metadata'"
    ).fetchone()
    if not exists:
        print("nport_security_metadata not found.")
        return

    clause, params = _filter_clause(indexes, snapshot)
    coverage = _q(
        conn,
        f"""
        SELECT us.index_name, us.snapshot_date, COUNT(*) AS rows,
               SUM(nm.isin IS NOT NULL) AS staged_rows,
               SUM(nm.security_name IS NOT NULL AND nm.security_name != '') AS with_name,
               SUM(nm.cusip IS NOT NULL AND nm.cusip != '') AS with_cusip,
               SUM(nm.lei IS NOT NULL AND nm.lei != '') AS with_lei,
               SUM(c.isin IS NULL AND nm.isin IS NOT NULL) AS staged_missing_companies
        FROM universe_snapshots us
        JOIN nport_accessions na
          ON na.index_name = us.index_name
         AND na.snapshot_date = us.snapshot_date
        LEFT JOIN nport_security_metadata nm
          ON nm.accession = na.accession
         AND nm.isin = us.isin
        LEFT JOIN companies c ON c.isin = us.isin
        WHERE 1=1 {clause}
        GROUP BY us.index_name, us.snapshot_date
        ORDER BY us.snapshot_date DESC, us.index_name
        """,
        tuple(params),
    )
    _print_df(coverage, max_rows=60)

    missing_companies = _q(
        conn,
        f"""
        SELECT us.index_name, us.snapshot_date, us.isin, us.weight,
               nm.security_name, nm.security_title, nm.cusip, nm.lei,
               nm.currency, nm.investment_country
        FROM universe_snapshots us
        JOIN nport_accessions na
          ON na.index_name = us.index_name
         AND na.snapshot_date = us.snapshot_date
        JOIN nport_security_metadata nm
          ON nm.accession = na.accession
         AND nm.isin = us.isin
        LEFT JOIN companies c ON c.isin = us.isin
        WHERE c.isin IS NULL {clause}
        ORDER BY us.snapshot_date DESC, us.index_name, us.weight DESC
        LIMIT ?
        """,
        tuple(params + [top]),
    )
    print("\nStaged N-PORT metadata for ISINs missing from companies")
    _print_df(_fmt(missing_companies), max_rows=top)


def audit_company_candidates(conn: sqlite3.Connection, indexes: list[str], snapshot: str | None, top: int) -> None:
    print("\nN-PORT company candidates")
    exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='nport_company_candidates'"
    ).fetchone()
    if not exists:
        print("nport_company_candidates not found.")
        return
    cols = {row[1] for row in conn.execute("PRAGMA table_info(nport_company_candidates)").fetchall()}
    has_resolution = {
        "resolution_status", "resolved_ticker", "resolved_cik", "resolution_confidence"
    }.issubset(cols)

    filters = ["1=1"]
    params: list[object] = []
    if indexes:
        filters.append("(" + " OR ".join("seen_indexes LIKE ?" for _ in indexes) + ")")
        params.extend([f"%{idx}%" for idx in indexes])
    if snapshot:
        filters.append("last_snapshot_date = ?")
        params.append(snapshot)
    where = " AND ".join(filters)

    summary = _q(
        conn,
        f"""
        SELECT company_status, COUNT(*) AS rows,
               SUM(security_name IS NOT NULL AND security_name != '') AS with_name,
               SUM(cusip IS NOT NULL AND cusip != '') AS with_cusip,
               SUM(lei IS NOT NULL AND lei != '') AS with_lei,
               MIN(first_snapshot_date) AS first_snapshot,
               MAX(last_snapshot_date) AS last_snapshot,
               MAX(staged_at) AS staged_at
        FROM nport_company_candidates
        WHERE {where}
        GROUP BY company_status
        ORDER BY company_status
        """,
        tuple(params),
    )
    _print_df(summary, max_rows=20)

    if has_resolution:
        resolution = _q(
            conn,
            f"""
            SELECT COALESCE(resolution_status, 'not_resolved') AS resolution_status,
                   COUNT(*) AS rows,
                   SUM(resolved_ticker IS NOT NULL AND resolved_ticker != '') AS with_ticker,
                   SUM(resolved_cik IS NOT NULL AND resolved_cik != '') AS with_cik,
                   ROUND(AVG(resolution_confidence), 3) AS avg_confidence,
                   MAX(resolved_at) AS resolved_at
            FROM nport_company_candidates
            WHERE {where}
              AND company_status = 'missing_from_companies'
            GROUP BY COALESCE(resolution_status, 'not_resolved')
            ORDER BY rows DESC, resolution_status
            """,
            tuple(params),
        )
        print("\nCandidate resolution status")
        _print_df(resolution, max_rows=20)

        select_resolution_cols = """
               resolved_ticker, resolved_cik, resolved_company_name,
               resolved_exchange, resolution_status, resolution_confidence,
        """
    else:
        select_resolution_cols = ""

    missing = _q(
        conn,
        f"""
        SELECT isin, security_name, security_title, cusip, lei, currency,
               investment_country, {select_resolution_cols}
               max_weight, seen_indexes,
               first_snapshot_date, last_snapshot_date
        FROM nport_company_candidates
        WHERE {where}
          AND company_status = 'missing_from_companies'
        ORDER BY max_weight DESC, isin
        LIMIT ?
        """,
        tuple(params + [top]),
    )
    print("\nCandidate ISINs missing from companies")
    _print_df(_fmt(missing), max_rows=top)


def audit_accession_reuse(conn: sqlite3.Connection, indexes: list[str]) -> None:
    idx_filter = ""
    params: list[str] = []
    if indexes:
        idx_filter = f"WHERE index_name IN ({','.join('?' * len(indexes))})"
        params.extend(indexes)
    df = _q(
        conn,
        f"""
        SELECT index_name, accession, period_ending,
               COUNT(*) AS snapshot_dates,
               MIN(snapshot_date) AS first_snapshot,
               MAX(snapshot_date) AS last_snapshot
        FROM nport_accessions
        {idx_filter}
        GROUP BY index_name, accession, period_ending
        ORDER BY index_name, last_snapshot DESC
        """,
        tuple(params),
    )
    print("\nN-PORT accession reuse")
    _print_df(df, max_rows=80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", action="append", dest="indexes", default=[], help="Index to audit. Repeatable.")
    parser.add_argument("--snapshot", help="Restrict to one snapshot date.")
    parser.add_argument("--top", type=int, default=25, help="Examples per issue section.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with _conn() as conn:
        audit_nport_rows(conn, args.indexes, args.snapshot)
        audit_quality(conn, args.indexes, args.snapshot, args.top)
        audit_non_nport_snapshots(conn, args.indexes, args.snapshot)
        audit_staged_metadata(conn, args.indexes, args.snapshot, args.top)
        audit_company_candidates(conn, args.indexes, args.snapshot, args.top)
        audit_accession_reuse(conn, args.indexes)


if __name__ == "__main__":
    main()
