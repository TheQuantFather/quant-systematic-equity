"""
db.py — cached data access layer for the Quant dashboard.
All public functions return DataFrames and are cached with st.cache_data.
"""

import io
import json
import sqlite3
import zlib
import numpy as np
import pandas as pd
import streamlit as st

from config import (
    UNIVERSE_DB, FACTORS_DB, MODELS_DB, CONSTITUENTS_DB, RETURNS_DB, RISK_DB,
    FACTORS_REF, MODELS_REF, CONSTITUENTS_REF,
)
from utils import get_db

# ---------------------------------------------------------------------------
# Reference / metadata
# ---------------------------------------------------------------------------

@st.cache_data
def get_ticker_map() -> dict:
    """Returns {isin (str): ticker (str)} from universe.db."""
    with get_db(UNIVERSE_DB) as conn:
        rows = conn.execute("SELECT isin, ticker FROM companies WHERE ticker IS NOT NULL").fetchall()
    return {isin: ticker for isin, ticker in rows if ticker}


@st.cache_data
def get_factor_metadata() -> pd.DataFrame:
    """Returns factor_id, factor_name, category, description, direction."""
    return pd.read_csv(FACTORS_REF)[["factor_id", "factor_name", "category", "description", "direction"]]


@st.cache_data
def get_model_metadata() -> pd.DataFrame:
    """Returns unique Model, ModelID, IsComposite rows from models_reference.csv."""
    df = pd.read_csv(MODELS_REF)
    df["IsComposite"] = df["IsComposite"].astype(int)
    return df[["Model", "ModelID", "IsComposite"]].drop_duplicates()


@st.cache_data
def get_constituents_metadata() -> pd.DataFrame:
    """Returns constituent_id, constituent_name, statement_type."""
    return pd.read_csv(CONSTITUENTS_REF)[["constituent_id", "constituent_name", "statement_type"]]


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

@st.cache_data
def get_universe() -> pd.DataFrame:
    with get_db(UNIVERSE_DB) as conn:
        df = pd.read_sql(
            "SELECT isin, ticker, company_name, gics_sector, simfin_sector, simfin_industry, "
            "       country, exchange, cik, simfin_id "
            "FROM companies",
            conn,
        )
    df["ticker"]       = df["ticker"].fillna("")
    df["sector"]       = df["gics_sector"].fillna(df["simfin_sector"]).fillna("")
    df["industry"]     = df["simfin_industry"].fillna("")
    df["security_id"]  = df["isin"]
    df["display_name"] = df.apply(
        lambda r: f"{r['ticker']} — {r['company_name']}" if r["ticker"] else r["company_name"],
        axis=1,
    )
    return df


# ---------------------------------------------------------------------------
# Factors (cross-sectional — latest date only)
# ---------------------------------------------------------------------------

@st.cache_data
def get_factors_long() -> pd.DataFrame:
    """
    All factor rows from factors.db (full time series), joined with factor names.
    Includes both raw values and cross-sectional z-scores.
    """
    with get_db(FACTORS_DB) as conn:
        df = pd.read_sql(
            "SELECT data_date, factor_id, security_id, "
            "       factor_value, factor_value_z FROM factors",
            conn,
        )
    df["factor_value"]   = pd.to_numeric(df["factor_value"],   errors="coerce")
    df["factor_value_z"] = pd.to_numeric(df["factor_value_z"], errors="coerce")
    df["security_id"]    = df["security_id"].astype(str)
    # Keep only known factors
    meta = get_factor_metadata()
    known_ids = set(meta["factor_id"])
    df = df[df["factor_id"].isin(known_ids)]
    # Attach human-readable name and category
    df = df.merge(meta[["factor_id", "factor_name", "category"]], on="factor_id", how="left")
    return df


@st.cache_data
def get_factors_wide() -> pd.DataFrame:
    """
    Wide pivot using the latest data_date per security.
    One row per security, one column per factor_name (raw values).
    """
    long = get_factors_long()
    # Keep only each security's most recent snapshot
    latest = long.groupby("security_id")["data_date"].max().reset_index()
    long = long.merge(latest.rename(columns={"data_date": "max_date"}), on="security_id")
    long = long[long["data_date"] == long["max_date"]]
    wide = long.pivot_table(
        index="security_id", columns="factor_name", values="factor_value", aggfunc="first"
    )
    wide.columns.name = None
    wide.reset_index(inplace=True)
    return wide


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@st.cache_data
def get_models_wide() -> pd.DataFrame:
    """Wide pivot: one row per security, columns named by model name (e.g. 'Quality Model')."""
    with get_db(MODELS_DB) as conn:
        df = pd.read_sql(
            "SELECT data_date, model_id, security_id, model_value, model_value_z FROM models",
            conn,
        )
    df["model_value"]   = pd.to_numeric(df["model_value"],   errors="coerce")
    df["model_value_z"] = pd.to_numeric(df["model_value_z"], errors="coerce")
    df["security_id"]   = df["security_id"].astype(str)
    # Keep only each security's most recent snapshot
    latest = df.groupby("security_id")["data_date"].max().reset_index()
    df = df.merge(latest.rename(columns={"data_date": "max_date"}), on="security_id")
    df = df[df["data_date"] == df["max_date"]]
    # Use z-scored values as the canonical model score
    wide = df.pivot_table(
        index="security_id", columns="model_id", values="model_value_z", aggfunc="first"
    )
    wide.columns.name = None
    wide.reset_index(inplace=True)
    # Rename model_id columns to human-readable "<Model> Model" names
    meta = get_model_metadata()
    id_to_name = dict(zip(meta["ModelID"], meta["Model"].map(lambda m: f"{m} Model")))
    wide.rename(columns=id_to_name, inplace=True)
    return wide


# ---------------------------------------------------------------------------
# Screener — merged table
# ---------------------------------------------------------------------------

@st.cache_data
def get_screener_df() -> pd.DataFrame:
    """Universe + factors (wide) + models (wide), merged on ISIN (security_id)."""
    universe = get_universe().copy()   # already has security_id = isin
    factors  = get_factors_wide()
    models   = get_models_wide()

    df = universe.merge(factors, on="security_id", how="inner")
    df = df.merge(models, on="security_id", how="left")
    return df


# ---------------------------------------------------------------------------
# Single-security detail
# ---------------------------------------------------------------------------

@st.cache_data
def get_company_info(isin: str) -> dict:
    """Company-level metadata for the Deep Dive header — name, sectors, business summary, etc."""
    with get_db(UNIVERSE_DB) as conn:
        row = conn.execute(
            "SELECT ticker, company_name, gics_sector, gics_industry, simfin_industry, "
            "       country, exchange, num_employees, business_summary "
            "FROM companies WHERE isin = ?",
            (isin,),
        ).fetchone()
    if not row:
        return {}
    keys = ["ticker", "company_name", "gics_sector", "gics_industry", "simfin_industry",
            "country", "exchange", "num_employees", "business_summary"]
    return dict(zip(keys, row))


@st.cache_data
def get_constituents_for_security(security_id: str) -> pd.DataFrame:
    # Historical SimFin data is stored under simfin_id; EDGAR updates use ISIN.
    # Query both so the full history is returned for every company.
    with get_db(UNIVERSE_DB) as conn:
        row = conn.execute(
            "SELECT simfin_id FROM companies WHERE isin = ? OR CAST(simfin_id AS TEXT) = ?",
            (security_id, security_id),
        ).fetchone()
    simfin_str = str(row[0]) if row and row[0] is not None else None
    ids = list({security_id, simfin_str} - {None})
    placeholders = ",".join("?" * len(ids))

    with get_db(CONSTITUENTS_DB) as conn:
        df = pd.read_sql(
            f"SELECT constituent_id, constituent_value, fiscal_year, fiscal_period, "
            f"report_date, publish_date "
            f"FROM constituents WHERE security_id IN ({placeholders})",
            conn, params=ids,
        )
    const_meta = get_constituents_metadata()
    df = df.merge(const_meta, on="constituent_id", how="left")
    df["constituent_value"] = pd.to_numeric(df["constituent_value"], errors="coerce")
    df["fiscal_year"] = pd.to_numeric(df["fiscal_year"], errors="coerce").astype("Int64")
    df["report_date"]  = pd.to_datetime(df["report_date"],  errors="coerce")
    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")

    # Deduplicate: for the same (fiscal_year, fiscal_period, constituent_id), keep the
    # row with the latest publish_date.  EDGAR rows are generally more recent than SimFin
    # for overlapping periods, so this naturally prefers EDGAR data.
    df = (
        df.sort_values("publish_date", ascending=True)
          .drop_duplicates(subset=["constituent_id", "fiscal_year", "fiscal_period"],
                           keep="last")
    )

    # Derive Q4 = FY − Q1 − Q2 − Q3 for income statement and cash flow concepts where
    # Q4 is absent.  Without this, the Deep Dive rolling LTM window hits a NaN at every
    # Q4 position (EDGAR only has FY annual, not a separate Q4 10-Q) and the LTM chart
    # breaks for any metric that isn't also provided by SimFin for Q4.
    flow_stmts = {"Income Statement", "Cash Flow Statement"}
    flow_df = df[df["statement_type"].isin(flow_stmts)].copy()
    if not flow_df.empty:
        fy_vals = (
            flow_df[flow_df["fiscal_period"] == "FY"]
            .set_index(["constituent_id", "fiscal_year"])["constituent_value"]
        )
        q_sums = (
            flow_df[flow_df["fiscal_period"].isin(["Q1", "Q2", "Q3"])]
            .groupby(["constituent_id", "fiscal_year"])["constituent_value"]
            .sum()
        )
        q4_rows = []
        for (cid, fy), fy_val in fy_vals.items():
            if pd.isna(fy_val):
                continue
            q_sum = q_sums.get((cid, fy))
            if q_sum is None or pd.isna(q_sum):
                continue
            # Skip if Q4 already exists
            existing_q4 = df[
                (df["constituent_id"] == cid) &
                (df["fiscal_year"] == fy) &
                (df["fiscal_period"] == "Q4")
            ]
            if not existing_q4.empty:
                continue
            # Only derive if all 3 quarterly summands are present
            n_quarters = flow_df[
                (flow_df["constituent_id"] == cid) &
                (flow_df["fiscal_year"] == fy) &
                (flow_df["fiscal_period"].isin(["Q1", "Q2", "Q3"]))
            ]["fiscal_period"].nunique()
            if n_quarters < 3:
                continue
            meta = const_meta[const_meta["constituent_id"] == cid].iloc[0] if len(const_meta[const_meta["constituent_id"] == cid]) else None
            q4_rows.append({
                "constituent_id":    cid,
                "constituent_value": float(fy_val) - float(q_sum),
                "fiscal_year":       fy,
                "fiscal_period":     "Q4",
                "report_date":       pd.NaT,
                "publish_date":      pd.NaT,
                "constituent_name":  meta["constituent_name"] if meta is not None else None,
                "statement_type":    meta["statement_type"]   if meta is not None else None,
            })
        if q4_rows:
            df = pd.concat([df, pd.DataFrame(q4_rows)], ignore_index=True)

    df = df.sort_values(["statement_type", "constituent_name", "fiscal_year", "fiscal_period"])
    return df


@st.cache_data
def get_industry_composite(industry: str) -> pd.DataFrame:
    """Equal-weighted daily total-return series for all stocks in `industry`
    (industry as stored in companies.simfin_industry). Returns columns date, total_return."""
    if not industry:
        return pd.DataFrame(columns=["date", "total_return"])
    with get_db(UNIVERSE_DB) as conn:
        rows = conn.execute(
            "SELECT isin FROM companies WHERE simfin_industry = ?", (industry,)
        ).fetchall()
    isins = [r[0] for r in rows if r[0]]
    if not isins:
        return pd.DataFrame(columns=["date", "total_return"])
    ph = ",".join("?" * len(isins))
    with get_db(RETURNS_DB) as conn:
        df = pd.read_sql(
            f"SELECT date, isin, total_return FROM returns WHERE isin IN ({ph})",
            conn, params=isins,
        )
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    composite = (
        df.dropna(subset=["total_return"])
          .groupby("date")["total_return"].mean()
          .reset_index()
          .rename(columns={"total_return": "total_return"})
    )
    return composite


@st.cache_data
def get_returns_for_security(isin: str) -> pd.DataFrame:
    """Daily total_return and close for one ISIN from returns.db, sorted by date."""
    if not RETURNS_DB.exists():
        return pd.DataFrame()
    with get_db(RETURNS_DB) as conn:
        df = pd.read_sql(
            "SELECT date, total_return, close FROM returns WHERE isin = ? ORDER BY date",
            conn, params=(isin,)
        )
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["total_return"] = pd.to_numeric(df["total_return"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df


@st.cache_data
def get_factors_for_security(security_id: str) -> pd.DataFrame:
    """Factor values for one company (with names and categories)."""
    long = get_factors_long()
    return long[long["security_id"] == str(security_id)].copy()


@st.cache_data
def get_models_for_security(security_id: str) -> pd.DataFrame:
    with get_db(MODELS_DB) as conn:
        df = pd.read_sql(
            "SELECT data_date, model_id, model_value, model_value_z FROM models WHERE security_id = ?",
            conn, params=(str(security_id),)
        )
    df["model_value"]   = pd.to_numeric(df["model_value"],   errors="coerce")
    df["model_value_z"] = pd.to_numeric(df["model_value_z"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Sector-level aggregates
# ---------------------------------------------------------------------------

@st.cache_data
def get_sector_factor_medians() -> pd.DataFrame:
    """Median factor value per sector for benchmarking."""
    screener = get_screener_df()
    factor_names = get_factor_metadata()["factor_name"].tolist()
    cols = ["sector"] + [c for c in factor_names if c in screener.columns]
    return screener[cols].groupby("sector").median(numeric_only=True).reset_index()


@st.cache_data
def get_sector_model_medians() -> pd.DataFrame:
    """Median model score per sector."""
    screener = get_screener_df()
    meta = get_model_metadata()
    model_col_names = [f"{m} Model" for m in meta["Model"]]
    cols = ["sector"] + [c for c in model_col_names if c in screener.columns]
    return screener[cols].groupby("sector").median(numeric_only=True).reset_index()


def load_covariance(data_date: str | None = None) -> tuple[np.ndarray, list[str]] | tuple[None, None]:
    """
    Load the covariance matrix from risk.db for the given snapshot date.
    If data_date is None, returns the most recent available.

    Returns (cov_matrix, isins) where isins[i] is the ticker/ISIN for row/col i.
    Returns (None, None) if risk.db doesn't exist or has no data for that date.
    """
    if not RISK_DB.exists():
        return None, None

    with get_db(RISK_DB) as conn:
        if data_date is None:
            row = conn.execute(
                "SELECT matrix_blob, isin_list FROM covariance_matrix ORDER BY data_date DESC LIMIT 1"
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT matrix_blob, isin_list FROM covariance_matrix WHERE data_date = ?",
                (data_date,),
            ).fetchone()

    if row is None:
        return None, None

    cov = np.load(io.BytesIO(zlib.decompress(row[0]))).astype(np.float64)
    isins = json.loads(row[1])
    return cov, isins


def get_risk_metadata() -> pd.DataFrame:
    """Return metadata for all stored covariance matrices."""
    if not RISK_DB.exists():
        return pd.DataFrame()
    with get_db(RISK_DB) as conn:
        df = pd.read_sql(
            "SELECT data_date, n_stocks, shrinkage_coeff, lookback_days, computation_date "
            "FROM covariance_matrix ORDER BY data_date",
            conn,
        )
    return df


# ---------------------------------------------------------------------------
# Benchmark returns
# ---------------------------------------------------------------------------

@st.cache_data
def get_benchmark_returns(index_name: str) -> pd.Series:
    """
    Daily total_return series for one index from the benchmark_returns table.
    Returns an empty Series if returns.db does not exist or has no data.
    """
    if not RETURNS_DB.exists():
        return pd.Series(dtype=float, name=index_name)
    with get_db(RETURNS_DB) as conn:
        df = pd.read_sql(
            "SELECT date, total_return FROM benchmark_returns "
            "WHERE index_name = ? ORDER BY date",
            conn, params=(index_name,),
        )
    if df.empty:
        return pd.Series(dtype=float, name=index_name)
    df["date"]         = pd.to_datetime(df["date"])
    df["total_return"] = pd.to_numeric(df["total_return"], errors="coerce")
    return df.set_index("date")["total_return"].rename(index_name)


@st.cache_data
def get_available_benchmark_indices() -> list[str]:
    """Return list of index_name values present in benchmark_returns."""
    if not RETURNS_DB.exists():
        return []
    with get_db(RETURNS_DB) as conn:
        rows = conn.execute(
            "SELECT DISTINCT index_name FROM benchmark_returns ORDER BY index_name"
        ).fetchall()
    return [r[0] for r in rows]


@st.cache_data
def get_sp500_isins_at_date(snapshot_date: str) -> list[str]:
    """Return ISINs in the S&P 500 universe at the given snapshot_date."""
    return get_universe_isins_at_date("sp500", snapshot_date)


@st.cache_data
def get_sp500_weights_at_date(snapshot_date: str) -> dict[str, float]:
    """Return {isin: weight} for S&P 500 at snapshot_date, normalised to sum=1."""
    return get_universe_weights_at_date("sp500", snapshot_date)


@st.cache_data
def get_universe_isins_at_date(index_name: str, snapshot_date: str) -> list[str]:
    """Return ISINs for any tracked index at the given snapshot_date."""
    with get_db(UNIVERSE_DB) as conn:
        rows = conn.execute(
            "SELECT isin FROM universe_snapshots WHERE snapshot_date = ? AND index_name = ?",
            (snapshot_date, index_name),
        ).fetchall()
    return [r[0] for r in rows]


@st.cache_data
def get_universe_weights_at_date(index_name: str, snapshot_date: str) -> dict[str, float]:
    """
    Return {isin: weight} for any tracked index at snapshot_date, normalised to sum=1.
    Weights come from universe_snapshots.weight (iShares N-PORT-P % weights).
    """
    with get_db(UNIVERSE_DB) as conn:
        rows = conn.execute(
            "SELECT isin, weight FROM universe_snapshots "
            "WHERE snapshot_date = ? AND index_name = ? AND weight IS NOT NULL",
            (snapshot_date, index_name),
        ).fetchall()
    if not rows:
        return {}
    raw   = {r[0]: float(r[1]) for r in rows}
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {isin: w / total for isin, w in raw.items()}


@st.cache_data
def get_available_universe_indices() -> list[str]:
    """Return index_name values that have universe snapshots (usable as optimizer universe)."""
    if not UNIVERSE_DB.exists():
        return []
    with get_db(UNIVERSE_DB) as conn:
        rows = conn.execute(
            "SELECT DISTINCT index_name FROM universe_snapshots ORDER BY index_name"
        ).fetchall()
    return [r[0] for r in rows]
