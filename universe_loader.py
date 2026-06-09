"""Clean point-in-time universe loader for optimizers and diagnostics.

The intent is to keep messy security-master decisions out of optimizer code.
This module is read-only: it does not mutate universe snapshots or mappings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Literal

import pandas as pd

from config import RETURNS_DB, UNIVERSE_DB
from utils import get_logger

log = get_logger(__name__)

UniverseMode = Literal["live", "point_in_time"]


@dataclass(frozen=True)
class CleanUniverseResult:
    """Cleaned universe plus summary metadata."""

    index_name: str
    requested_snapshot_date: str
    source_snapshot_date: str
    benchmark_index: str | None
    benchmark_source_snapshot_date: str | None
    mode: UniverseMode
    members: pd.DataFrame

    @property
    def tradable(self) -> pd.DataFrame:
        return self.members[self.members["is_tradable"]].copy()


def _connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(str(path))


def _query(path: Path, sql: str, params: tuple = ()) -> pd.DataFrame:
    with _connect(path) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def nearest_universe_snapshot(index_name: str, snapshot_date: str) -> str | None:
    """Most recent universe snapshot for index_name on or before snapshot_date."""

    with _connect(UNIVERSE_DB) as conn:
        row = conn.execute(
            """
            SELECT MAX(snapshot_date)
            FROM universe_snapshots
            WHERE index_name = ? AND snapshot_date <= ?
            """,
            (index_name, snapshot_date),
        ).fetchone()
    return str(row[0]) if row and row[0] else None


def latest_universe_snapshot(index_name: str) -> str | None:
    """Latest universe snapshot for index_name."""

    with _connect(UNIVERSE_DB) as conn:
        row = conn.execute(
            "SELECT MAX(snapshot_date) FROM universe_snapshots WHERE index_name = ?",
            (index_name,),
        ).fetchone()
    return str(row[0]) if row and row[0] else None


def _load_snapshot_members(index_name: str, snapshot_date: str) -> pd.DataFrame:
    return _query(
        UNIVERSE_DB,
        """
        SELECT us.index_name, us.snapshot_date AS source_snapshot_date,
               us.isin AS original_isin, us.weight AS raw_weight,
               us.market_value,
               c.isin IS NOT NULL AS snapshot_has_security_master,
               c.ticker AS snapshot_ticker,
               c.company_name AS snapshot_company_name,
               c.gics_sector AS snapshot_gics_sector,
               c.gics_industry_group AS snapshot_gics_industry_group,
               c.gics_industry AS snapshot_gics_industry,
               c.gics_sub_industry AS snapshot_gics_sub_industry,
               c.simfin_sector AS snapshot_simfin_sector,
               c.simfin_industry AS snapshot_simfin_industry,
               c.country AS snapshot_country,
               c.exchange AS snapshot_exchange,
               c.currency AS snapshot_currency,
               c.cik AS snapshot_cik,
               c.cusip AS snapshot_cusip,
               c.data_date AS snapshot_company_data_date,
               c.update_date AS snapshot_company_update_date,
               c.delisted_date AS snapshot_delisted_date
        FROM universe_snapshots us
        LEFT JOIN companies c ON c.isin = us.isin
        WHERE us.index_name = ? AND us.snapshot_date = ?
        """,
        (index_name, snapshot_date),
    )


def _load_security_master(mode: UniverseMode, snapshot_date: str) -> pd.DataFrame:
    df = _query(
        UNIVERSE_DB,
        """
        SELECT isin, ticker, company_name, gics_sector, gics_industry_group,
               gics_industry, gics_sub_industry, simfin_sector, simfin_industry,
               country, exchange, currency, cik, cusip, data_date, update_date,
               delisted_date
        FROM companies
        WHERE ticker IS NOT NULL AND ticker <> ''
        """,
    )
    return df


def _normalise_name(value: object) -> str:
    raw = str(value or "").upper()
    for token in [
        " INCORPORATED", " INC", " CORPORATION", " CORP", " COMPANY", " CO",
        " PLC", " LTD", " LIMITED", " CLASS A", " CLASS B", " CLASS C",
        ".", ",", "'", "\"", " /DE/",
    ]:
        raw = raw.replace(token, " ")
    return " ".join(raw.split())


def _name_matches(snapshot_name: object, current_name: object) -> bool:
    left = _normalise_name(snapshot_name)
    right = _normalise_name(current_name)
    if not left or not right:
        return False
    return left == right or left in right or right in left


def _identity_evidence(row: pd.Series) -> str:
    parts = [
        f"snapshot_ticker={row.get('snapshot_ticker') or ''}",
        f"snapshot_name={row.get('snapshot_company_name') or ''}",
        f"original_isin={row.get('original_isin') or ''}",
        f"current_ticker={row.get('current_ticker') or ''}",
        f"current_name={row.get('current_company_name') or ''}",
        f"current_isin={row.get('current_isin') or ''}",
    ]
    return "; ".join(parts)


def _canonical_by_ticker(master: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    if master.empty:
        return master
    df = master.copy()
    df["_not_delisted"] = df["delisted_date"].isna() | (df["delisted_date"].astype(str) > snapshot_date)
    df["_update_sort"] = df["update_date"].fillna("")
    df["_data_sort"] = df["data_date"].fillna("")
    df = df.sort_values(
        ["ticker", "_not_delisted", "_update_sort", "_data_sort"],
        ascending=[True, False, False, False],
    )
    return df.groupby("ticker", as_index=False).head(1).drop(columns=["_not_delisted", "_update_sort", "_data_sort"])


def _apply_security_identity(
    members: pd.DataFrame,
    *,
    mode: UniverseMode,
    snapshot_date: str,
    normalize_live_isin: bool,
) -> pd.DataFrame:
    out = members.copy()
    master = _load_security_master(mode, snapshot_date)
    canonical = _canonical_by_ticker(master, snapshot_date)
    canonical = canonical.add_prefix("current_")
    out = out.merge(
        canonical,
        left_on="snapshot_ticker",
        right_on="current_ticker",
        how="left",
    )

    out["ticker"] = out["snapshot_ticker"]
    out["isin"] = out["original_isin"]
    out["company_name"] = out["snapshot_company_name"]
    out["gics_sector"] = out["snapshot_gics_sector"]
    out["gics_industry_group"] = out["snapshot_gics_industry_group"]
    out["gics_industry"] = out["snapshot_gics_industry"]
    out["gics_sub_industry"] = out["snapshot_gics_sub_industry"]
    out["simfin_sector"] = out["snapshot_simfin_sector"]
    out["simfin_industry"] = out["snapshot_simfin_industry"]
    out["country"] = out["snapshot_country"]
    out["exchange"] = out["snapshot_exchange"]
    out["currency"] = out["snapshot_currency"]
    out["cik"] = out["snapshot_cik"]
    out["cusip"] = out["snapshot_cusip"]
    out["delisted_date"] = out["snapshot_delisted_date"]
    if "snapshot_has_security_master" not in out.columns:
        out["snapshot_has_security_master"] = out["snapshot_company_name"].notna()
    has_snapshot_master = out["snapshot_has_security_master"].fillna(False).astype(bool)
    out["identity_status"] = "snapshot_isin"
    out["canonical_isin"] = out["isin"]
    out["mapped_from_isin"] = ""
    out["mapped_to_isin"] = ""
    out["identity_rule"] = "point_in_time_snapshot_isin" if mode == "point_in_time" else "snapshot_isin"
    out["identity_confidence"] = 1.0

    has_current = out["current_isin"].notna()
    different_isin = has_current & (out["current_isin"] != out["original_isin"])
    name_ok = out.apply(
        lambda r: _name_matches(r.get("snapshot_company_name"), r.get("current_company_name")),
        axis=1,
    )
    can_map = normalize_live_isin & (mode == "live") & different_isin & name_ok

    # A live remap should only fire for a genuine rename. If the target current
    # ISIN is already present in this snapshot under its own ISIN, the two are
    # co-listed distinct securities (e.g. dual share classes sharing a ticker,
    # like Clearway CWEN class A/C) — not a rename. Suppress the remap so we
    # never collapse two securities onto one ISIN.
    present_isins = set(out["original_isin"].dropna().astype(str))
    ticker_collision = can_map & out["current_isin"].astype(str).isin(present_isins)
    can_map = can_map & ~ticker_collision

    unresolved = different_isin & ~name_ok

    copy_cols = [
        "company_name", "gics_sector", "gics_industry_group", "gics_industry",
        "gics_sub_industry", "simfin_sector", "simfin_industry", "country",
        "exchange", "currency", "cik", "cusip", "delisted_date",
    ]
    for col in copy_cols:
        current_col = f"current_{col}"
        if current_col in out:
            out.loc[can_map, col] = out.loc[can_map, current_col]

    out.loc[can_map, "isin"] = out.loc[can_map, "current_isin"]
    out.loc[can_map, "canonical_isin"] = out.loc[can_map, "current_isin"]
    out.loc[can_map, "mapped_from_isin"] = out.loc[can_map, "original_isin"]
    out.loc[can_map, "mapped_to_isin"] = out.loc[can_map, "current_isin"]
    out.loc[can_map, "identity_status"] = "mapped_to_current_isin"
    out.loc[can_map, "identity_rule"] = "live_same_ticker_same_name_current_isin"
    out.loc[can_map, "identity_confidence"] = 0.9
    out.loc[unresolved, "identity_status"] = "unresolved_ticker_reuse"
    out.loc[unresolved, "identity_rule"] = "same_ticker_name_mismatch"
    out.loc[unresolved, "identity_confidence"] = 0.0
    out.loc[has_current & ~different_isin, "identity_status"] = "current_isin"
    out.loc[has_current & ~different_isin, "identity_rule"] = "current_isin_exact_match"
    out.loc[has_current & ~different_isin, "identity_confidence"] = 1.0
    no_current_ticker_match = has_snapshot_master & ~has_current
    out.loc[no_current_ticker_match, "identity_status"] = "snapshot_isin"
    out.loc[no_current_ticker_match, "identity_rule"] = "snapshot_isin_no_current_ticker_match"
    out.loc[no_current_ticker_match, "identity_confidence"] = 0.8
    out.loc[~has_snapshot_master, "identity_status"] = "missing_security_master"
    out.loc[~has_snapshot_master, "identity_rule"] = "missing_security_master"
    out.loc[~has_snapshot_master, "identity_confidence"] = 0.0
    out.loc[ticker_collision, "identity_status"] = "ticker_collision_kept_snapshot_isin"
    out.loc[ticker_collision, "identity_rule"] = "live_remap_suppressed_target_already_present"
    out.loc[ticker_collision, "identity_confidence"] = 0.7
    out["canonical_isin"] = out["isin"]
    out["identity_evidence"] = out.apply(_identity_evidence, axis=1)
    return out


def _latest_return_coverage() -> tuple[str | None, pd.DataFrame]:
    latest_date = None
    with _connect(RETURNS_DB) as conn:
        row = conn.execute("SELECT MAX(date) FROM returns").fetchone()
        latest_date = str(row[0]) if row and row[0] else None
    if latest_date is None:
        return None, pd.DataFrame(columns=["isin", "last_return_date", "latest_close", "latest_volume"])

    coverage = _query(
        RETURNS_DB,
        """
        WITH last_by_isin AS (
            SELECT isin, MAX(date) AS last_return_date
            FROM returns
            GROUP BY isin
        )
        SELECT l.isin, l.last_return_date,
               r.close AS latest_close,
               r.volume AS latest_volume
        FROM last_by_isin l
        JOIN returns r
          ON r.isin = l.isin
         AND r.date = l.last_return_date
        """,
    )
    return latest_date, coverage


def _add_exclusion_reasons(
    df: pd.DataFrame,
    *,
    snapshot_date: str,
    min_return_date: str | None,
    require_latest_volume: bool,
) -> pd.DataFrame:
    out = df.copy()
    reasons: list[list[str]] = [[] for _ in range(len(out))]

    def add(mask: pd.Series, reason: str) -> None:
        for i in out.index[mask.fillna(False)]:
            reasons[out.index.get_loc(i)].append(reason)

    add(out["isin"].isna(), "missing_isin")
    add(out["isin"].fillna("").str.startswith("NOISN_"), "placeholder_isin")
    add(out["identity_status"].eq("missing_security_master"), "missing_security_master")
    add(out["identity_status"].eq("unresolved_ticker_reuse"), "unresolved_ticker_reuse")
    add(out["delisted_date"].notna() & (out["delisted_date"].astype(str) <= snapshot_date), "delisted")
    add(out["last_return_date"].isna(), "missing_returns")
    if min_return_date:
        add(out["last_return_date"].notna() & (out["last_return_date"].astype(str) < min_return_date), "stale_returns")
    if require_latest_volume:
        add(out["latest_volume"].fillna(0) <= 0, "no_latest_volume")

    out["exclude_reason"] = ["|".join(r) for r in reasons]
    out["is_tradable"] = out["exclude_reason"].eq("")
    return out


def _normalise_weights(df: pd.DataFrame, weight_col: str = "raw_weight") -> pd.Series:
    weights = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0)
    total = float(weights.sum())
    if total <= 0:
        return pd.Series(0.0, index=df.index)
    return weights / total


def _benchmark_weights(
    benchmark_index: str,
    snapshot_date: str,
    *,
    mode: UniverseMode,
    normalize_live_isin: bool,
    min_return_date: str | None,
    require_latest_volume: bool,
) -> tuple[str | None, pd.DataFrame]:
    source = nearest_universe_snapshot(benchmark_index, snapshot_date)
    if source is None:
        return None, pd.DataFrame(columns=["isin", "benchmark_weight"])

    raw = _load_snapshot_members(benchmark_index, source)
    if raw.empty:
        return source, pd.DataFrame(columns=["isin", "benchmark_weight"])

    cleaned = _apply_security_identity(
        raw,
        mode=mode,
        snapshot_date=snapshot_date,
        normalize_live_isin=normalize_live_isin,
    )
    _, coverage = _latest_return_coverage()
    cleaned = cleaned.merge(coverage, on="isin", how="left")
    cleaned = _add_exclusion_reasons(
        cleaned,
        snapshot_date=snapshot_date,
        min_return_date=min_return_date,
        require_latest_volume=require_latest_volume,
    )
    cleaned = cleaned[cleaned["is_tradable"]].copy()
    if cleaned.empty:
        return source, pd.DataFrame(columns=["isin", "benchmark_weight"])
    cleaned["benchmark_weight"] = _normalise_weights(cleaned, "raw_weight")
    bench = cleaned.groupby("isin", as_index=False)["benchmark_weight"].sum()
    return source, bench


def load_clean_universe(
    index_name: str,
    snapshot_date: str,
    *,
    benchmark_index: str | None = None,
    mode: UniverseMode = "live",
    normalize_live_isin: bool = True,
    min_return_date: str | None = None,
    require_latest_volume: bool = True,
    tradable_only: bool = False,
) -> CleanUniverseResult:
    """Return a clean member table for one index and date.

    `snapshot_date` is the requested rebalance date. If there is no exact
    universe snapshot, the most recent universe snapshot on or before that date
    is used and exposed as `source_snapshot_date`.

    In `live` mode, same-name ticker rows may be mapped from a stale snapshot
    ISIN to the newest current ISIN. In `point_in_time` mode, future
    security-master rows are ignored and ISINs remain point-in-time.
    """

    if mode not in ("live", "point_in_time"):
        raise ValueError("mode must be 'live' or 'point_in_time'")

    source_snapshot = nearest_universe_snapshot(index_name, snapshot_date)
    if source_snapshot is None:
        raise ValueError(f"No universe snapshot for {index_name!r} on or before {snapshot_date}.")

    raw = _load_snapshot_members(index_name, source_snapshot)
    if raw.empty:
        raise ValueError(f"No members for {index_name!r} at {source_snapshot}.")

    members = _apply_security_identity(
        raw,
        mode=mode,
        snapshot_date=snapshot_date,
        normalize_live_isin=normalize_live_isin,
    )
    latest_return_date, coverage = _latest_return_coverage()
    members = members.merge(coverage, on="isin", how="left")
    if min_return_date is None and mode == "live":
        min_return_date = snapshot_date
        if latest_return_date is not None and str(latest_return_date) < str(snapshot_date):
            log.warning(
                "Returns data lags requested snapshot: latest return date %s < snapshot %s "
                "for %s. Most members will be marked stale_returns and non-tradable — "
                "run `create_returns --update` before loading this date.",
                latest_return_date, snapshot_date, index_name,
            )
    members = _add_exclusion_reasons(
        members,
        snapshot_date=snapshot_date,
        min_return_date=min_return_date,
        require_latest_volume=require_latest_volume,
    )
    members["weight"] = _normalise_weights(members, "raw_weight")
    members["latest_return_snapshot"] = latest_return_date

    benchmark_source = None
    if benchmark_index:
        benchmark_source, bench = _benchmark_weights(
            benchmark_index,
            snapshot_date,
            mode=mode,
            normalize_live_isin=normalize_live_isin,
            min_return_date=min_return_date,
            require_latest_volume=require_latest_volume,
        )
        members = members.merge(bench, on="isin", how="left")
        members["benchmark_weight"] = members["benchmark_weight"].fillna(0.0)
    else:
        members["benchmark_weight"] = 0.0

    if tradable_only:
        members = members[members["is_tradable"]].copy()
        members["weight"] = _normalise_weights(members, "raw_weight")

    sort_cols = ["is_tradable", "weight", "ticker"]
    members = members.sort_values(sort_cols, ascending=[False, False, True]).reset_index(drop=True)
    return CleanUniverseResult(
        index_name=index_name,
        requested_snapshot_date=snapshot_date,
        source_snapshot_date=source_snapshot,
        benchmark_index=benchmark_index,
        benchmark_source_snapshot_date=benchmark_source,
        mode=mode,
        members=members,
    )
