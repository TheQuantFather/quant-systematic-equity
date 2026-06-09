#!/usr/bin/env python3
"""Read-only integrity audit for universe snapshots and security metadata."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FACTORS_DB, MODELS_DB, PARAMS_FILE, RETURNS_DB, RISK_DB, UNIVERSE_DB  # noqa: E402
from universe_loader import load_clean_universe  # noqa: E402


def _conn(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(str(path))


def _q(path: Path, sql: str, params: tuple = ()) -> pd.DataFrame:
    with _conn(path) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def _scalar(path: Path, sql: str, params: tuple = ()) -> object:
    with _conn(path) as conn:
        row = conn.execute(sql, params).fetchone()
    return row[0] if row else None


def _table_cols(path: Path, table: str) -> set[str]:
    with _conn(path) as conn:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {r[1] for r in rows}


def _print_df(df: pd.DataFrame, *, max_rows: int = 20) -> None:
    if df.empty:
        print("  OK")
        return
    shown = df.head(max_rows).copy()
    print(shown.to_string(index=False))
    if len(df) > max_rows:
        print(f"  ... {len(df) - max_rows:,} more")


def _fmt_weight_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col.endswith("weight") or col in {"weight", "weight_sum"}:
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6f}")
    return out


def snapshot_arg(value: str | None) -> str:
    if not value or value.lower() == "latest":
        latest = _scalar(UNIVERSE_DB, "SELECT MAX(snapshot_date) FROM universe_snapshots")
        if not latest:
            raise SystemExit("No universe snapshots found.")
        return str(latest)
    return value


def print_header(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def audit_snapshot_spine() -> None:
    print_header("Snapshot Spine")

    schedule = _q(
        UNIVERSE_DB,
        """
        SELECT cadence, COUNT(*) AS n_dates, MIN(data_date) AS first_date,
               MAX(data_date) AS last_date,
               SUM(factors_computed_at IS NOT NULL) AS factors_marked
        FROM snapshot_schedule
        GROUP BY cadence
        ORDER BY MIN(data_date)
        """,
    )
    _print_df(schedule)

    coverage = pd.DataFrame(
        [
            {
                "dataset": "factors",
                "n_dates": _scalar(FACTORS_DB, "SELECT COUNT(DISTINCT data_date) FROM factors"),
                "latest": _scalar(FACTORS_DB, "SELECT MAX(data_date) FROM factors"),
            },
            {
                "dataset": "models",
                "n_dates": _scalar(MODELS_DB, "SELECT COUNT(DISTINCT data_date) FROM models"),
                "latest": _scalar(MODELS_DB, "SELECT MAX(data_date) FROM models"),
            },
            {
                "dataset": "lw_risk",
                "n_dates": _scalar(RISK_DB, "SELECT COUNT(*) FROM covariance_matrix"),
                "latest": _scalar(RISK_DB, "SELECT MAX(data_date) FROM covariance_matrix"),
            },
            {
                "dataset": "barra",
                "n_dates": _scalar(RISK_DB, "SELECT COUNT(DISTINCT snapshot_date) FROM factor_covariance"),
                "latest": _scalar(RISK_DB, "SELECT MAX(snapshot_date) FROM factor_covariance"),
            },
        ]
    )
    print("\nModel/risk coverage")
    _print_df(coverage)


def audit_universe_coverage(top: int) -> None:
    print_header("Universe Snapshot Coverage")

    summary = _q(
        UNIVERSE_DB,
        """
        SELECT index_name, COUNT(DISTINCT snapshot_date) AS n_dates,
               MIN(snapshot_date) AS first_date, MAX(snapshot_date) AS last_date,
               COUNT(*) AS rows
        FROM universe_snapshots
        GROUP BY index_name
        ORDER BY index_name
        """,
    )
    _print_df(summary, max_rows=50)

    missing = _q(
        UNIVERSE_DB,
        """
        WITH indexes AS (
            SELECT DISTINCT index_name FROM universe_snapshots
        ),
        grid AS (
            SELECT i.index_name, s.data_date
            FROM indexes i CROSS JOIN snapshot_schedule s
        ),
        have AS (
            SELECT DISTINCT index_name, snapshot_date FROM universe_snapshots
        )
        SELECT g.index_name, COUNT(*) AS missing_dates,
               MIN(g.data_date) AS first_missing,
               MAX(g.data_date) AS last_missing
        FROM grid g
        LEFT JOIN have h
          ON h.index_name = g.index_name AND h.snapshot_date = g.data_date
        WHERE h.snapshot_date IS NULL
        GROUP BY g.index_name
        ORDER BY missing_dates DESC, g.index_name
        """,
    )
    print("\nScheduled dates without direct universe_snapshots rows")
    _print_df(missing, max_rows=50)

    examples = _q(
        UNIVERSE_DB,
        """
        WITH indexes AS (
            SELECT DISTINCT index_name FROM universe_snapshots
        ),
        grid AS (
            SELECT i.index_name, s.data_date
            FROM indexes i CROSS JOIN snapshot_schedule s
        ),
        have AS (
            SELECT DISTINCT index_name, snapshot_date FROM universe_snapshots
        )
        SELECT g.index_name, g.data_date
        FROM grid g
        LEFT JOIN have h
          ON h.index_name = g.index_name AND h.snapshot_date = g.data_date
        WHERE h.snapshot_date IS NULL
        ORDER BY g.data_date DESC, g.index_name
        LIMIT ?
        """,
        (top,),
    )
    print("\nRecent missing schedule/index pairs")
    _print_df(examples, max_rows=top)

    if "nport_accessions" in {
        r[0] for r in _q(
            UNIVERSE_DB,
            "SELECT name FROM sqlite_master WHERE type='table'",
        ).itertuples(index=False, name=None)
    }:
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
        print("\nUniverse snapshots not backed by N-PORT accession")
        _print_df(_fmt_weight_cols(non_nport), max_rows=50)


def audit_latest_snapshot(snapshot_date: str, indexes: list[str], top: int) -> None:
    print_header(f"Snapshot Health: {snapshot_date}")

    idx_filter = ""
    params: list[object] = [snapshot_date]
    if indexes:
        idx_filter = f"AND us.index_name IN ({','.join('?' * len(indexes))})"
        params.extend(indexes)

    weights = _q(
        UNIVERSE_DB,
        f"""
        SELECT us.index_name, COUNT(*) AS n_members,
               COUNT(DISTINCT us.isin) AS n_isins,
               COUNT(DISTINCT c.ticker) AS n_tickers,
               SUM(us.weight) AS weight_sum,
               SUM(c.delisted_date IS NOT NULL AND c.delisted_date <= us.snapshot_date) AS delisted_members,
               SUM(us.isin LIKE 'NOISN_%') AS placeholder_isins,
               SUM(c.gics_sector IS NULL OR c.gics_sector = '') AS missing_sector,
               SUM(c.gics_industry IS NULL OR c.gics_industry = '') AS missing_gics_industry,
               SUM(c.simfin_industry IS NULL OR c.simfin_industry = '') AS missing_simfin_industry
        FROM universe_snapshots us
        JOIN companies c ON c.isin = us.isin
        WHERE us.snapshot_date = ? {idx_filter}
        GROUP BY us.index_name
        ORDER BY us.index_name
        """,
        tuple(params),
    )
    _print_df(_fmt_weight_cols(weights), max_rows=50)

    latest_returns = _q(
        RETURNS_DB,
        "SELECT isin, MAX(date) AS last_return_date FROM returns GROUP BY isin",
    )
    latest_day = _scalar(RETURNS_DB, "SELECT MAX(date) FROM returns")
    day_returns = _q(
        RETURNS_DB,
        "SELECT isin, close, volume FROM returns WHERE date = ?",
        (latest_day,),
    )
    members = _q(
        UNIVERSE_DB,
        f"""
        SELECT us.index_name, us.snapshot_date, us.isin, us.weight,
               c.ticker, c.company_name, c.gics_sector, c.simfin_industry,
               c.delisted_date
        FROM universe_snapshots us
        JOIN companies c ON c.isin = us.isin
        WHERE us.snapshot_date = ? {idx_filter}
        """,
        tuple(params),
    )
    members = members.merge(latest_returns, on="isin", how="left")
    members = members.merge(day_returns, on="isin", how="left")

    tradability = (
        members.assign(
            delisted_now=lambda d: d["delisted_date"].notna() & (d["delisted_date"] <= d["snapshot_date"]),
            missing_returns=lambda d: d["last_return_date"].isna(),
            stale_returns=lambda d: d["last_return_date"].notna() & (d["last_return_date"] < d["snapshot_date"]),
            no_latest_volume=lambda d: d["volume"].fillna(0) <= 0,
        )
        .groupby("index_name")
        .agg(
            delisted_now=("delisted_now", "sum"),
            missing_returns=("missing_returns", "sum"),
            stale_returns=("stale_returns", "sum"),
            no_latest_volume=("no_latest_volume", "sum"),
        )
        .reset_index()
    )
    print(f"\nReturn/tradability coverage using latest returns date {latest_day}")
    _print_df(tradability, max_rows=50)

    bad = members[
        (members["delisted_date"].notna() & (members["delisted_date"] <= members["snapshot_date"]))
        | members["last_return_date"].isna()
        | (members["last_return_date"].notna() & (members["last_return_date"] < members["snapshot_date"]))
        | (members["volume"].fillna(0) <= 0)
    ].copy()
    bad = bad.sort_values(["index_name", "delisted_date", "last_return_date", "weight"], ascending=[True, False, True, False])
    cols = ["index_name", "ticker", "isin", "company_name", "weight", "delisted_date", "last_return_date", "close", "volume"]
    print("\nProblem members: delisted/stale/no latest volume")
    _print_df(_fmt_weight_cols(bad[cols]), max_rows=top)


def audit_identity(snapshot_date: str, indexes: list[str], top: int) -> None:
    print_header("Security Identity")

    duplicates = _q(
        UNIVERSE_DB,
        """
        SELECT ticker, COUNT(*) AS n_isins,
               GROUP_CONCAT(isin, ' | ') AS isins,
               GROUP_CONCAT(COALESCE(update_date, ''), ' | ') AS update_dates,
               GROUP_CONCAT(COALESCE(delisted_date, ''), ' | ') AS delisted_dates
        FROM companies
        WHERE ticker IS NOT NULL AND ticker <> ''
        GROUP BY ticker
        HAVING COUNT(*) > 1
        ORDER BY n_isins DESC, ticker
        """,
    )
    print("Company tickers with multiple ISIN rows")
    _print_df(duplicates, max_rows=top)

    idx_filter = ""
    params: list[object] = [snapshot_date]
    if indexes:
        idx_filter = f"AND us.index_name IN ({','.join('?' * len(indexes))})"
        params.extend(indexes)

    drift = _q(
        UNIVERSE_DB,
        f"""
        WITH members AS (
            SELECT us.index_name, us.snapshot_date, us.isin AS snapshot_isin, us.weight,
                   c.ticker, c.company_name, c.update_date AS snapshot_update_date,
                   c.delisted_date AS snapshot_delisted_date
            FROM universe_snapshots us
            JOIN companies c ON c.isin = us.isin
            WHERE us.snapshot_date = ? {idx_filter}
        ),
        ranked AS (
            SELECT c.ticker, c.isin AS current_isin, c.company_name AS current_name,
                   c.update_date AS current_update_date, c.delisted_date AS current_delisted_date,
                   ROW_NUMBER() OVER (
                       PARTITION BY c.ticker
                       ORDER BY (c.delisted_date IS NULL) DESC,
                                COALESCE(c.update_date, '') DESC,
                                COALESCE(c.data_date, '') DESC
                   ) AS rn
            FROM companies c
            WHERE c.ticker IS NOT NULL AND c.ticker <> ''
        )
        SELECT m.index_name, m.ticker, m.snapshot_isin, r.current_isin,
               m.company_name AS snapshot_name, r.current_name,
               m.weight, m.snapshot_update_date, r.current_update_date,
               m.snapshot_delisted_date, r.current_delisted_date
        FROM members m
        JOIN ranked r ON r.ticker = m.ticker AND r.rn = 1
        WHERE m.snapshot_isin <> r.current_isin
        ORDER BY m.index_name, m.weight DESC
        """,
        tuple(params),
    )
    print("\nLatest snapshot ISIN differs from newest security-master row for same ticker")
    _print_df(_fmt_weight_cols(drift), max_rows=top)

    patches = _q(
        UNIVERSE_DB,
        """
        SELECT
            (SELECT COUNT(*) FROM isin_patch) AS isin_patch_rows,
            (SELECT COUNT(*) FROM ticker_alias) AS ticker_alias_rows,
            (SELECT COUNT(*) FROM simfin_exclude) AS simfin_exclude_rows
        """,
    )
    print("\nManual mapping tables")
    _print_df(patches)


def audit_classification(snapshot_date: str, indexes: list[str], top: int) -> None:
    print_header("Classification")

    cols = _table_cols(UNIVERSE_DB, "companies")
    sic_cols = sorted(c for c in cols if c.lower().startswith("sic"))
    if sic_cols:
        print(f"SEC SIC columns present: {', '.join(sic_cols)}")
    else:
        print("SEC SIC columns present: none")

    idx_filter = ""
    params: list[object] = [snapshot_date]
    if indexes:
        idx_filter = f"AND us.index_name IN ({','.join('?' * len(indexes))})"
        params.extend(indexes)

    missing = _q(
        UNIVERSE_DB,
        f"""
        SELECT us.index_name,
               SUM(c.gics_sector IS NULL OR c.gics_sector = '') AS missing_gics_sector,
               SUM(c.gics_industry_group IS NULL OR c.gics_industry_group = '') AS missing_gics_industry_group,
               SUM(c.gics_industry IS NULL OR c.gics_industry = '') AS missing_gics_industry,
               SUM(c.gics_sub_industry IS NULL OR c.gics_sub_industry = '') AS missing_gics_sub_industry,
               SUM(c.simfin_industry IS NULL OR c.simfin_industry = '') AS missing_simfin_industry
        FROM universe_snapshots us
        JOIN companies c ON c.isin = us.isin
        WHERE us.snapshot_date = ? {idx_filter}
        GROUP BY us.index_name
        ORDER BY us.index_name
        """,
        tuple(params),
    )
    print("\nMissing classification in selected snapshot")
    _print_df(missing, max_rows=50)

    examples = _q(
        UNIVERSE_DB,
        f"""
        SELECT us.index_name, c.ticker, c.isin, c.company_name,
               c.gics_sector, c.gics_industry_group, c.gics_industry,
               c.simfin_sector, c.simfin_industry, us.weight
        FROM universe_snapshots us
        JOIN companies c ON c.isin = us.isin
        WHERE us.snapshot_date = ? {idx_filter}
          AND (
              c.gics_industry IS NULL OR c.gics_industry = ''
              OR c.simfin_industry IS NULL OR c.simfin_industry = ''
          )
        ORDER BY us.index_name, us.weight DESC
        LIMIT ?
        """,
        tuple(params + [top]),
    )
    print("\nClassification gap examples")
    _print_df(_fmt_weight_cols(examples), max_rows=top)


def audit_strategy_csv_dependencies() -> None:
    print_header("Strategy Universe/Benchmark Sources")
    if not PARAMS_FILE.exists():
        print(f"{PARAMS_FILE} not found.")
        return
    try:
        df = pd.read_excel(PARAMS_FILE, sheet_name="Strategies", dtype=str)
    except Exception as exc:
        print(f"Could not read {PARAMS_FILE}: {exc}")
        return
    cols = [
        c for c in [
            "strategy_id", "name", "active", "benchmark_file",
            "benchmark_index", "universe_index", "alpha_date", "risk_date",
        ]
        if c in df.columns
    ]
    if not cols:
        print("Strategies sheet has no expected columns.")
        return
    dep = df[cols].copy()
    filters = []
    for col in ("benchmark_file", "benchmark_index", "universe_index"):
        if col in dep.columns:
            filters.append(dep[col].fillna("").str.strip() != "")
    if filters:
        mask = filters[0]
        for f in filters[1:]:
            mask = mask | f
        dep = dep[mask]
    _print_df(dep, max_rows=50)


def audit_clean_loader_preview(snapshot_date: str, indexes: list[str], top: int) -> None:
    print_header("Clean Universe Loader Preview")
    selected = indexes or _q(
        UNIVERSE_DB,
        "SELECT DISTINCT index_name FROM universe_snapshots ORDER BY index_name",
    )["index_name"].tolist()

    rows = []
    blocked_frames = []
    mapped_frames = []
    for index_name in selected:
        try:
            result = load_clean_universe(
                index_name,
                snapshot_date,
                benchmark_index="sp500",
                mode="live",
            )
        except Exception as exc:
            rows.append({
                "index_name": index_name,
                "error": str(exc),
            })
            continue

        df = result.members
        reasons = (
            df.loc[~df["is_tradable"], "exclude_reason"]
            .str.get_dummies(sep="|")
            .sum()
            .sort_values(ascending=False)
        )
        rows.append({
            "index_name": index_name,
            "source_snapshot": result.source_snapshot_date,
            "members": len(df),
            "tradable": int(df["is_tradable"].sum()),
            "mapped_to_current_isin": int(df["identity_status"].eq("mapped_to_current_isin").sum()),
            "low_identity_conf": int((df["identity_confidence"] < 1.0).sum()),
            "blocked": int((~df["is_tradable"]).sum()),
            "top_block_reason": reasons.index[0] if not reasons.empty else "",
            "top_block_count": int(reasons.iloc[0]) if not reasons.empty else 0,
        })
        blocked_frames.append(
            df.loc[
                ~df["is_tradable"],
                ["index_name", "ticker", "original_isin", "isin", "company_name",
                 "weight", "identity_status", "identity_rule",
                 "identity_confidence", "exclude_reason"],
            ].head(top)
        )
        mapped = df[df["identity_status"].eq("mapped_to_current_isin")].copy()
        if not mapped.empty:
            mapped_frames.append(
                mapped[
                    ["index_name", "ticker", "original_isin", "isin", "company_name",
                     "weight", "identity_status", "identity_rule", "identity_confidence"]
                ].head(top)
            )

    _print_df(_fmt_weight_cols(pd.DataFrame(rows)), max_rows=50)
    if blocked_frames:
        print("\nBlocked-name examples after clean loader")
        _print_df(_fmt_weight_cols(pd.concat(blocked_frames, ignore_index=True)), max_rows=top)
    if mapped_frames:
        print("\nIdentifier mapping examples after clean loader")
        _print_df(_fmt_weight_cols(pd.concat(mapped_frames, ignore_index=True)), max_rows=top)


def audit_materialized_clean_snapshots(snapshot_date: str, indexes: list[str], top: int) -> None:
    print_header("Materialized Clean Universe Snapshots")

    with _conn(UNIVERSE_DB) as conn:
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='clean_universe_snapshots'"
        ).fetchone()
    if not exists:
        print("clean_universe_snapshots not found.")
        return

    cols = _table_cols(UNIVERSE_DB, "clean_universe_snapshots")
    required = {"identity_rule", "identity_confidence", "canonical_isin"}
    missing = required - cols
    if missing:
        print(f"Missing identity audit columns: {', '.join(sorted(missing))}")
        return

    idx_filter = ""
    params: list[object] = [snapshot_date]
    if indexes:
        idx_filter = f"AND index_name IN ({','.join('?' * len(indexes))})"
        params.extend(indexes)

    summary = _q(
        UNIVERSE_DB,
        f"""
        SELECT mode, index_name, COUNT(*) AS rows,
               SUM(is_tradable) AS tradable,
               SUM(identity_status = 'mapped_to_current_isin') AS mapped_to_current_isin,
               SUM(identity_confidence < 1.0) AS low_identity_conf,
               SUM(canonical_isin != isin) AS canonical_mismatch,
               MAX(materialized_at) AS materialized_at
        FROM clean_universe_snapshots
        WHERE requested_snapshot_date = ? {idx_filter}
        GROUP BY mode, index_name
        ORDER BY mode, index_name
        """,
        tuple(params),
    )
    _print_df(summary, max_rows=50)

    rules = _q(
        UNIVERSE_DB,
        f"""
        SELECT mode, index_name, identity_rule, COUNT(*) AS rows,
               SUM(is_tradable) AS tradable
        FROM clean_universe_snapshots
        WHERE requested_snapshot_date = ? {idx_filter}
        GROUP BY mode, index_name, identity_rule
        ORDER BY mode, index_name, rows DESC
        """,
        tuple(params),
    )
    print("\nIdentity rules in materialized table")
    _print_df(rules, max_rows=50)

    examples = _q(
        UNIVERSE_DB,
        f"""
        SELECT mode, index_name, ticker, original_isin, isin, mapped_from_isin,
               mapped_to_isin, identity_status, identity_rule,
               identity_confidence, is_tradable, exclude_reason
        FROM clean_universe_snapshots
        WHERE requested_snapshot_date = ? {idx_filter}
          AND (identity_confidence < 1.0 OR is_tradable = 0)
        ORDER BY mode, index_name, identity_confidence, weight DESC
        LIMIT ?
        """,
        tuple(params + [top]),
    )
    print("\nMaterialized identity/tradability examples")
    _print_df(_fmt_weight_cols(examples), max_rows=top)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default="latest", help="Universe snapshot date to audit, or 'latest'.")
    parser.add_argument(
        "--index",
        action="append",
        dest="indexes",
        default=[],
        help="Index name to include. Repeat for multiple. Default: all indexes in the snapshot.",
    )
    parser.add_argument("--top", type=int, default=25, help="Rows to show per issue section.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot_date = snapshot_arg(args.snapshot)

    print(f"Universe integrity audit | snapshot={snapshot_date} | indexes={args.indexes or 'all'}")
    audit_snapshot_spine()
    audit_universe_coverage(args.top)
    audit_latest_snapshot(snapshot_date, args.indexes, args.top)
    audit_identity(snapshot_date, args.indexes, args.top)
    audit_classification(snapshot_date, args.indexes, args.top)
    audit_clean_loader_preview(snapshot_date, args.indexes, args.top)
    audit_materialized_clean_snapshots(snapshot_date, args.indexes, args.top)
    audit_strategy_csv_dependencies()


if __name__ == "__main__":
    main()
