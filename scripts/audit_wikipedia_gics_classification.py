#!/usr/bin/env python3
"""Audit Wikipedia GICS labels against the local universe security master.

Read-only. This is intended as a first pass before replacing SimFin-derived
industry metadata. It pulls:

  * GICS hierarchy from Wikipedia's Global Industry Classification Standard page
  * Russell 1000 constituent labels from Wikipedia's Russell 1000 Index page

and joins them to universe.db companies by normalized ticker.
"""

from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path
import sqlite3
import urllib.request

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import UNIVERSE_DB

USER_AGENT = "quant-classification-audit shivam3125@gmail.com"
GICS_URL = "https://en.wikipedia.org/wiki/Global_Industry_Classification_Standard"
RUSSELL_1000_URL = "https://en.wikipedia.org/wiki/Russell_1000_Index"
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def _fetch_tables(url: str) -> list[pd.DataFrame]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode("utf-8")
    return pd.read_html(StringIO(html))


def _clean(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).replace("\xa0", " ").strip().split())


def _norm_label(value: object) -> str:
    raw = _clean(value).upper()
    for ch in ["&", ",", ".", "-", "/", "(", ")"]:
        raw = raw.replace(ch, " ")
    return " ".join(raw.split())


def _norm_ticker(value: object) -> str:
    raw = _clean(value).upper()
    return "".join(ch for ch in raw if ch.isalnum())


def load_gics_hierarchy() -> pd.DataFrame:
    tables = _fetch_tables(GICS_URL)
    hierarchy = tables[0].copy()
    expected = {
        "Sector.1", "Industry Group.1", "Industry.1", "Sub-Industry.1",
    }
    missing = expected - set(hierarchy.columns)
    if missing:
        raise RuntimeError(f"GICS hierarchy table missing columns: {sorted(missing)}")
    out = pd.DataFrame({
        "hierarchy_sector": hierarchy["Sector.1"].map(_clean),
        "gics_industry_group": hierarchy["Industry Group.1"].map(_clean),
        "gics_industry": hierarchy["Industry.1"].map(_clean),
        "gics_sub_industry": hierarchy["Sub-Industry.1"].map(_clean),
    })
    out["sub_industry_key"] = out["gics_sub_industry"].map(_norm_label)
    out = out[out["sub_industry_key"] != ""].drop_duplicates("sub_industry_key")
    return out


def load_russell_1000_labels() -> pd.DataFrame:
    for table in _fetch_tables(RUSSELL_1000_URL):
        cols = set(map(str, table.columns))
        if {"Company", "Symbol", "GICS Sector", "GICS Sub-Industry"}.issubset(cols):
            labels = table.copy()
            break
    else:
        raise RuntimeError("Could not find Russell 1000 components table on Wikipedia.")

    out = labels.rename(columns={
        "Company": "wiki_company",
        "Symbol": "wiki_ticker",
        "GICS Sector": "wiki_sector",
        "GICS Sub-Industry": "wiki_sub_industry",
    })[["wiki_company", "wiki_ticker", "wiki_sector", "wiki_sub_industry"]].copy()
    for col in out.columns:
        out[col] = out[col].map(_clean)
    out["wiki_label_source"] = "wikipedia_russell_1000"
    out["wiki_cik"] = ""
    out["ticker_key"] = out["wiki_ticker"].map(_norm_ticker)
    out["sub_industry_key"] = out["wiki_sub_industry"].map(_norm_label)
    out = out[out["ticker_key"] != ""].drop_duplicates("ticker_key", keep="first")
    return out


def load_sp500_labels() -> pd.DataFrame:
    for table in _fetch_tables(SP500_URL):
        cols = set(map(str, table.columns))
        if {"Symbol", "Security", "GICS Sector", "GICS Sub-Industry", "CIK"}.issubset(cols):
            labels = table.copy()
            break
    else:
        raise RuntimeError("Could not find S&P 500 components table on Wikipedia.")

    out = labels.rename(columns={
        "Security": "wiki_company",
        "Symbol": "wiki_ticker",
        "GICS Sector": "wiki_sector",
        "GICS Sub-Industry": "wiki_sub_industry",
        "CIK": "wiki_cik",
    })[["wiki_company", "wiki_ticker", "wiki_sector", "wiki_sub_industry", "wiki_cik"]].copy()
    for col in out.columns:
        out[col] = out[col].map(_clean)
    out["wiki_label_source"] = "wikipedia_sp500"
    out["ticker_key"] = out["wiki_ticker"].map(_norm_ticker)
    out["sub_industry_key"] = out["wiki_sub_industry"].map(_norm_label)
    out = out[out["ticker_key"] != ""].drop_duplicates("ticker_key", keep="first")
    return out


def load_wikipedia_labels() -> pd.DataFrame:
    labels = pd.concat([load_sp500_labels(), load_russell_1000_labels()], ignore_index=True)
    labels["_has_sub"] = labels["wiki_sub_industry"].fillna("").ne("")
    labels["_source_priority"] = labels["wiki_label_source"].map({
        "wikipedia_sp500": 0,
        "wikipedia_russell_1000": 1,
    }).fillna(9)

    conflict_keys = set()
    for ticker_key, group in labels.groupby("ticker_key"):
        pairs = {
            (
                _norm_label(row["wiki_sector"]),
                _norm_label(row["wiki_sub_industry"]),
            )
            for _, row in group.iterrows()
        }
        if len(pairs) > 1:
            conflict_keys.add(ticker_key)

    labels = labels.sort_values(
        ["ticker_key", "_has_sub", "_source_priority"],
        ascending=[True, False, True],
    )
    source_rollup = (
        labels.groupby("ticker_key", as_index=False)
        .agg(wiki_label_sources=("wiki_label_source", lambda s: "|".join(sorted(set(map(str, s))))))
    )
    best = labels.drop_duplicates("ticker_key", keep="first").drop(columns=["_has_sub", "_source_priority"])
    best = best.merge(source_rollup, on="ticker_key", how="left")
    best["wiki_label_conflict"] = best["ticker_key"].isin(conflict_keys)
    return best


def _selected_snapshot_dates(conn: sqlite3.Connection, indexes: list[str], snapshot: str) -> dict[str, str]:
    selected: dict[str, str] = {}
    for index_name in indexes:
        if snapshot == "latest":
            row = conn.execute(
                "SELECT MAX(snapshot_date) FROM universe_snapshots WHERE index_name = ?",
                (index_name,),
            ).fetchone()
            if row and row[0]:
                selected[index_name] = row[0]
        else:
            selected[index_name] = snapshot
    return selected


def load_companies(indexes: list[str] | None = None, snapshot: str = "latest") -> pd.DataFrame:
    with sqlite3.connect(str(UNIVERSE_DB)) as conn:
        if indexes:
            selected = _selected_snapshot_dates(conn, indexes, snapshot)
            if not selected:
                raise RuntimeError(f"No universe snapshots found for indexes={indexes}")
            clauses = []
            params: list[str] = []
            for index_name, snapshot_date in selected.items():
                clauses.append("(us.index_name = ? AND us.snapshot_date = ?)")
                params.extend([index_name, snapshot_date])
            members = pd.read_sql_query(
                f"""
                SELECT us.index_name, us.snapshot_date, us.isin, us.weight,
                       c.ticker, c.company_name, c.cik,
                       c.gics_sector, c.gics_industry_group, c.gics_industry,
                       c.gics_sub_industry, c.simfin_sector, c.simfin_industry
                FROM universe_snapshots us
                LEFT JOIN companies c ON c.isin = us.isin
                WHERE {' OR '.join(clauses)}
                """,
                conn,
                params=tuple(params),
            )
            index_meta = (
                members.groupby("isin", as_index=False)
                .agg(
                    index_names=("index_name", lambda s: "|".join(sorted(set(map(str, s))))),
                    snapshot_dates=("snapshot_date", lambda s: "|".join(sorted(set(map(str, s))))),
                    max_index_weight=("weight", "max"),
                )
            )
            companies = (
                members.drop(columns=["index_name", "snapshot_date", "weight"])
                .drop_duplicates("isin")
                .merge(index_meta, on="isin", how="left")
            )
        else:
            companies = pd.read_sql_query(
                """
                SELECT isin, ticker, company_name, cik,
                       gics_sector, gics_industry_group, gics_industry,
                       gics_sub_industry, simfin_sector, simfin_industry
                FROM companies
                WHERE ticker IS NOT NULL AND ticker != ''
                """,
                conn,
            )
            companies["index_names"] = ""
            companies["snapshot_dates"] = ""
            companies["max_index_weight"] = None
    companies["ticker_key"] = companies["ticker"].map(_norm_ticker)
    return companies


def build_audit_frame(indexes: list[str] | None = None, snapshot: str = "latest") -> pd.DataFrame:
    hierarchy = load_gics_hierarchy()
    labels = load_wikipedia_labels()
    companies = load_companies(indexes=indexes, snapshot=snapshot)
    audit = companies.merge(labels, on="ticker_key", how="left")
    audit = audit.merge(hierarchy, on="sub_industry_key", how="left")

    has_label = audit["wiki_ticker"].notna()
    has_sector = audit["wiki_sector"].fillna("").ne("")
    has_sub = audit["wiki_sub_industry"].fillna("").ne("")
    has_hierarchy = audit["hierarchy_sector"].fillna("").ne("")
    hierarchy_sector_ok = (
        audit["wiki_sector"].map(_norm_label)
        == audit["hierarchy_sector"].map(_norm_label)
    )
    existing_sector_conflict = (
        audit["gics_sector"].fillna("").ne("")
        & has_sector
        & (audit["gics_sector"].map(_norm_label) != audit["wiki_sector"].map(_norm_label))
    )

    audit["wiki_gics_status"] = "no_wiki_label"
    audit.loc[has_label & has_sector & ~has_sub, "wiki_gics_status"] = "sector_only"
    audit.loc[has_label & has_sector & has_sub & ~has_hierarchy, "wiki_gics_status"] = "subindustry_not_in_hierarchy"
    audit.loc[has_label & has_sector & has_sub & has_hierarchy & hierarchy_sector_ok, "wiki_gics_status"] = "complete"
    audit.loc[has_label & has_sector & has_sub & has_hierarchy & ~hierarchy_sector_ok, "wiki_gics_status"] = "hierarchy_sector_conflict"
    audit["existing_sector_conflict"] = existing_sector_conflict
    return audit


def _print_df(df: pd.DataFrame, max_rows: int) -> None:
    if df.empty:
        print("  OK")
        return
    print(df.head(max_rows).to_string(index=False))
    if len(df) > max_rows:
        print(f"  ... {len(df) - max_rows} more")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        action="append",
        dest="indexes",
        default=[],
        help="Restrict to an index snapshot. Repeatable. Default: all companies.",
    )
    parser.add_argument(
        "--snapshot",
        default="latest",
        help="Snapshot date for --index filters, or 'latest'.",
    )
    parser.add_argument("--top", type=int, default=25, help="Rows to show per issue section.")
    parser.add_argument("--output", help="Optional CSV path for the full joined audit frame.")
    args = parser.parse_args()

    audit = build_audit_frame(indexes=args.indexes or None, snapshot=args.snapshot)
    russell_labels = load_russell_1000_labels()
    sp500_labels = load_sp500_labels()
    labels = load_wikipedia_labels()
    hierarchy = load_gics_hierarchy()

    print("\nWikipedia source coverage")
    print(f"  Russell 1000 label rows: {len(russell_labels):,}")
    print(f"  S&P 500 label rows: {len(sp500_labels):,}")
    print(f"  Combined unique ticker labels: {len(labels):,}")
    print(f"  GICS hierarchy sub-industries: {len(hierarchy):,}")
    scope = (
        f"index snapshots {args.indexes} @ {args.snapshot}"
        if args.indexes else "all companies"
    )
    print(f"  Local audit scope: {scope}")
    print(f"  Local rows: {len(audit):,}")

    print("\nWiki GICS status")
    status = (
        audit.groupby("wiki_gics_status", dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values("rows", ascending=False)
    )
    _print_df(status, max_rows=20)

    print("\nUnique combined Wikipedia labels")
    unique = labels.merge(hierarchy, on="sub_industry_key", how="left")
    label_status = pd.DataFrame({
        "labels": [len(labels)],
        "with_sector": [int(labels["wiki_sector"].fillna("").ne("").sum())],
        "with_sub_industry": [int(labels["wiki_sub_industry"].fillna("").ne("").sum())],
        "sub_industry_in_hierarchy": [int(unique["hierarchy_sector"].fillna("").ne("").sum())],
        "source_conflicts": [int(labels["wiki_label_conflict"].sum())],
    })
    _print_df(label_status, max_rows=5)

    print("\nRows with Wikipedia sector but missing sub-industry")
    sector_only = audit[audit["wiki_gics_status"].eq("sector_only")][
        ["index_names", "ticker", "company_name", "wiki_company", "wiki_sector", "wiki_label_sources", "simfin_industry", "max_index_weight"]
    ].sort_values(["wiki_sector", "ticker"])
    _print_df(sector_only, args.top)

    print("\nRows where Wikipedia sub-industry is not in parsed GICS hierarchy")
    sub_miss = audit[audit["wiki_gics_status"].eq("subindustry_not_in_hierarchy")][
        ["index_names", "ticker", "company_name", "wiki_sector", "wiki_sub_industry", "wiki_label_sources", "max_index_weight"]
    ].sort_values(["wiki_sector", "wiki_sub_industry", "ticker"])
    _print_df(sub_miss, args.top)

    print("\nExisting company sector conflicts vs Wikipedia")
    conflicts = audit[audit["existing_sector_conflict"]][
        ["index_names", "ticker", "company_name", "gics_sector", "wiki_sector", "wiki_sub_industry", "wiki_label_sources", "max_index_weight"]
    ].sort_values(["ticker"])
    _print_df(conflicts, args.top)

    print("\nWikipedia source label conflicts")
    source_conflict_mask = audit["wiki_label_conflict"].eq(True)
    source_conflicts = audit[source_conflict_mask][
        ["index_names", "ticker", "company_name", "wiki_sector", "wiki_sub_industry", "wiki_label_sources"]
    ].sort_values(["ticker"])
    _print_df(source_conflicts, args.top)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(out, index=False)
        print(f"\nWrote audit CSV: {out}")


if __name__ == "__main__":
    main()
