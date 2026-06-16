"""
10_Data_Quality.py — Pipeline data validation and quality checks.
"""

import sys
from pathlib import Path
from datetime import datetime, date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    UNIVERSE_DB, RETURNS_DB, FACTORS_DB, MODELS_DB,
    CONSTITUENTS_DB, RISK_DB,
    FACTORS_REF, MODELS_REF,
)
from utils import get_db, inject_css

st.set_page_config(page_title="Data Quality", layout="wide")
inject_css()
st.title("Data Quality & Pipeline Health")
st.caption("Validation checks across all pipeline databases. All queries run live against the local DBs.")

# ---------------------------------------------------------------------------
# Cached data-loading helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def _db_meta() -> pd.DataFrame:
    dbs = {
        "universe":     UNIVERSE_DB,
        "constituents": CONSTITUENTS_DB,
        "returns":      RETURNS_DB,
        "factors":      FACTORS_DB,
        "models":       MODELS_DB,
        "risk":         RISK_DB,
        "barra":        RISK_DB,
    }
    rows = []
    for name, path in dbs.items():
        p = Path(path)
        if p.exists():
            stat = p.stat()
            rows.append({
                "DB": name,
                "Size (MB)": round(stat.st_size / 1_048_576, 1),
                "Last Modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            })
        else:
            rows.append({"DB": name, "Size (MB)": None, "Last Modified": "missing"})
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def _snapshot_coverage() -> pd.DataFrame:
    """Security counts per snapshot date across factors, models, barra."""
    with get_db(FACTORS_DB) as conn:
        f = pd.read_sql(
            "SELECT data_date, COUNT(DISTINCT security_id) AS factors "
            "FROM factors GROUP BY data_date ORDER BY data_date",
            conn,
        )
    with get_db(MODELS_DB) as conn:
        m = pd.read_sql(
            "SELECT data_date, COUNT(DISTINCT security_id) AS models "
            "FROM models GROUP BY data_date ORDER BY data_date",
            conn,
        )
    with get_db(RISK_DB) as conn:
        b = pd.read_sql(
            "SELECT snapshot_date AS data_date, COUNT(DISTINCT security_id) AS barra "
            "FROM factor_exposures GROUP BY snapshot_date ORDER BY snapshot_date",
            conn,
        )
    cov = f.merge(m, on="data_date", how="outer").merge(b, on="data_date", how="outer")
    cov = cov.sort_values("data_date").fillna(0).astype({"factors": int, "models": int, "barra": int})
    return cov


@st.cache_data(ttl=300)
def _factor_fill(snapshot_date: str) -> pd.DataFrame:
    ref = pd.read_csv(FACTORS_REF)[["factor_id", "factor_name", "category", "direction"]]
    with get_db(FACTORS_DB) as conn:
        raw = pd.read_sql(
            "SELECT factor_id, "
            "  COUNT(*) AS total, "
            "  COUNT(factor_value) AS filled, "
            "  AVG(factor_value_z) AS avg_z, "
            "  MIN(factor_value_z) AS min_z, "
            "  MAX(factor_value_z) AS max_z "
            "FROM factors WHERE data_date = ? GROUP BY factor_id",
            conn,
            params=(snapshot_date,),
        )
    with get_db(FACTORS_DB) as conn:
        extremes = pd.read_sql(
            "SELECT factor_id, COUNT(*) AS n_extreme "
            "FROM factors WHERE data_date = ? AND ABS(factor_value_z) > 4 "
            "GROUP BY factor_id",
            conn,
            params=(snapshot_date,),
        )
    df = raw.merge(ref, on="factor_id", how="left").merge(extremes, on="factor_id", how="left")
    df["filled"]    = pd.to_numeric(df["filled"], errors="coerce")
    df["total"]     = pd.to_numeric(df["total"],  errors="coerce")
    df["n_extreme"] = df["n_extreme"].fillna(0).astype(int)
    df["fill_pct"]  = (df["filled"] / df["total"] * 100).round(1)
    return df.sort_values("fill_pct")


# --- Factor anomaly detection (two unsupervised lenses) --------------------
# Surfaces individual suspect (security, factor) cells likely caused by XBRL
# mis-tagging or near-zero denominators — values that distort cross-sectional
# z-scores but pass the aggregate fill-rate / |z|>4 checks above.
#
#   cross_sectional — robust-z (median/MAD) on the RAW factor_value within each
#                     factor's peer group; catches extremes that winsorization
#                     hides once they become factor_value_z.
#   time_series     — robust-z of a value vs the security's own factor history,
#                     with the own-history MAD floored by the factor's pooled
#                     cross-sectional MAD so a genuine small change in a low-
#                     dispersion factor isn't mistaken for a mis-tag.
XSECTION_Z     = 10.0   # raw value this many MADs from peer median
TIMESERIES_Z   = 6.0    # value this many (floored) MADs from own history
MIN_HISTORY    = 5      # snapshots required before the time-series lens applies
MAD_FLOOR_FRAC = 0.5    # own-history MAD floored at this × the factor's pooled MAD


@st.cache_data(ttl=300)
def _security_map() -> pd.DataFrame:
    """factors.db security_id (ISIN or simfin_id) → ticker, sector."""
    with get_db(UNIVERSE_DB) as conn:
        comp = pd.read_sql(
            "SELECT isin, ticker, gics_sector, simfin_id FROM companies", conn
        )
    comp["simfin_str"] = comp["simfin_id"].apply(
        lambda x: str(int(x)) if pd.notna(x) else None
    )
    by_isin   = comp[["isin", "ticker", "gics_sector"]].rename(
        columns={"isin": "security_id", "gics_sector": "sector"})
    by_simfin = (comp.dropna(subset=["simfin_str"])[["simfin_str", "ticker", "gics_sector"]]
                 .rename(columns={"simfin_str": "security_id", "gics_sector": "sector"}))
    return pd.concat([by_isin, by_simfin], ignore_index=True).drop_duplicates("security_id")


def _add_med_mad(df: pd.DataFrame, by: list[str], col: str, prefix: str) -> pd.DataFrame:
    """Attach group median and MAD columns (two vectorised transforms, no lambda)."""
    med = df.groupby(by, sort=False)[col].transform("median")
    df[f"{prefix}_med"] = med
    df[f"{prefix}_mad"] = (df[col] - med).abs().groupby(
        [df[c] for c in by], sort=False
    ).transform("median")
    return df


@st.cache_data(ttl=300)
def _factor_anomalies(snapshot_date: str) -> pd.DataFrame:
    """
    Ranked suspect cells for one snapshot. Both lenses share a robust-z scale;
    the merged table is ordered by within-lens percentile so neither lens
    dominates, and cells flagged by both lenses are marked high-confidence.
    """
    with get_db(FACTORS_DB) as conn:
        panel = pd.read_sql(
            "SELECT data_date, factor_id, security_id, factor_value FROM factors "
            "WHERE factor_value IS NOT NULL", conn
        )
    panel["security_id"] = panel["security_id"].astype(str)
    names = dict(pd.read_csv(FACTORS_REF)[["factor_id", "factor_name"]].values)

    # --- Lens 1: cross-sectional, on the selected snapshot only ---
    xs = panel[panel["data_date"] == snapshot_date].copy()
    xs = _add_med_mad(xs, ["factor_id"], "factor_value", "xs")
    xs["rz"] = 0.6745 * (xs["factor_value"] - xs["xs_med"]) / xs["xs_mad"].replace(0, np.nan)
    xs_hit = xs[xs["rz"].abs() >= XSECTION_Z].copy()
    xs_hit["lens"]     = "cross_sectional"
    xs_hit["severity"] = xs_hit["rz"].abs()
    xs_hit["detail"]   = xs_hit.apply(
        lambda r: f"raw {r['factor_value']:.3g} is {abs(r['rz']):.0f}×MAD "
                  f"from peer median {r['xs_med']:.3g}", axis=1)

    # --- Lens 2: time-series, full history, reported on the snapshot ---
    ts = _add_med_mad(panel.copy(), ["security_id", "factor_id"], "factor_value", "own")
    pooled_mad = (panel["factor_value"] - panel.groupby("factor_id", sort=False)["factor_value"]
                  .transform("median")).abs().groupby(panel["factor_id"], sort=False).transform("median")
    ts["mad_eff"] = np.maximum(ts["own_mad"], MAD_FLOOR_FRAC * pooled_mad)
    ts["n_hist"]  = ts.groupby(["security_id", "factor_id"], sort=False)["factor_value"].transform("size")
    ts["rz"] = 0.6745 * (ts["factor_value"] - ts["own_med"]) / ts["mad_eff"].replace(0, np.nan)
    ts_hit = ts[(ts["rz"].abs() >= TIMESERIES_Z)
                & (ts["n_hist"] >= MIN_HISTORY)
                & (ts["data_date"] == snapshot_date)].copy()
    ts_hit["lens"]     = "time_series"
    ts_hit["severity"] = ts_hit["rz"].abs()
    ts_hit["detail"]   = ts_hit.apply(
        lambda r: f"{r['factor_value']:.3g} is {abs(r['rz']):.0f}×MAD "
                  f"from own-history median {r['own_med']:.3g}", axis=1)

    cols = ["lens", "security_id", "factor_id", "factor_value", "severity", "detail"]
    hits = pd.concat([xs_hit[cols], ts_hit[cols]], ignore_index=True)
    if hits.empty:
        return hits.assign(ticker=None, sector=None, factor_name=None,
                           rank_pct=None, confidence=None)

    hits["factor_name"] = hits["factor_id"].map(names)
    hits = hits.merge(_security_map(), on="security_id", how="left")
    hits["ticker"] = hits["ticker"].fillna(hits["security_id"])

    # within-lens percentile so the two robust-z scales interleave fairly
    hits["rank_pct"] = hits.groupby("lens")["severity"].rank(pct=True)
    dup = hits.groupby(["security_id", "factor_id"])["lens"].transform("nunique")
    hits["confidence"] = np.where(dup > 1, "both lenses", "single")
    return hits.sort_values(["rank_pct", "severity"], ascending=False, ignore_index=True)


@st.cache_data(ttl=300)
def _factor_coverage_over_time(model_id: str) -> pd.DataFrame:
    """% of the PIT Russell 1000 with a value for each factor in `model_id`, per date.

    Same universe denominator as _model_coverage_over_time, so absolute coverage is
    comparable. This is the view that catches a silent data regression: per-factor
    fill % is ~100% by construction (it divides by rows that exist), so it can read
    green while a factor actually covers a fraction of the universe.
    Sector-gated factors (bank/REIT-only) sit structurally low by design.
    """
    mref = pd.read_csv(MODELS_REF)
    fids = mref.loc[mref["ModelID"] == model_id, "Factors"].dropna().astype(str).tolist()
    if not fids:
        return pd.DataFrame()
    fref = pd.read_csv(FACTORS_REF)[["factor_id", "factor_name"]]
    names = dict(zip(fref["factor_id"], fref["factor_name"]))
    placeholders = ",".join("?" * len(fids))
    with get_db(FACTORS_DB) as conn:
        cov = pd.read_sql(
            f"SELECT data_date, factor_id, COUNT(DISTINCT security_id) AS scored "
            f"FROM factors WHERE factor_id IN ({placeholders}) AND factor_value IS NOT NULL "
            f"GROUP BY data_date, factor_id",
            conn, params=fids,
        )
    with get_db(UNIVERSE_DB) as conn:
        uni = pd.read_sql(
            "SELECT snapshot_date, COUNT(DISTINCT isin) AS uni_n "
            "FROM universe_snapshots WHERE index_name = 'russell_1000' "
            "GROUP BY snapshot_date ORDER BY snapshot_date",
            conn,
        )
    if cov.empty or uni.empty:
        return pd.DataFrame()
    uni = uni.sort_values("snapshot_date")
    uni["_ts"] = pd.to_datetime(uni["snapshot_date"])
    mdates = pd.DataFrame({"data_date": sorted(cov["data_date"].unique())})
    mdates["_ts"] = pd.to_datetime(mdates["data_date"])
    mapped = pd.merge_asof(mdates, uni[["_ts", "uni_n"]], on="_ts", direction="backward").drop(columns="_ts")
    df = cov.merge(mapped, on="data_date", how="left")
    df["coverage_pct"] = (df["scored"] / df["uni_n"] * 100).round(1)
    df["Factor"] = df["factor_id"].map(names).fillna(df["factor_id"])
    return df.sort_values("data_date")


@st.cache_data(ttl=300)
def _model_coverage_over_time() -> pd.DataFrame:
    """% of the point-in-time universe scored by each model, per snapshot date.

    Denominator is the russell_1000 membership at the nearest universe snapshot
    on or before each model date (the true PIT investable universe). This is the
    coverage signal that matters — null_pct is ~0 everywhere since models never
    store NULL z-scores, so it tells us nothing.
    """
    ref = pd.read_csv(MODELS_REF)[["ModelID", "Model"]].drop_duplicates()
    with get_db(MODELS_DB) as conn:
        scored = pd.read_sql(
            "SELECT data_date, model_id, COUNT(DISTINCT security_id) AS scored "
            "FROM models WHERE model_value_z IS NOT NULL "
            "GROUP BY data_date, model_id",
            conn,
        )
    with get_db(UNIVERSE_DB) as conn:
        uni = pd.read_sql(
            "SELECT snapshot_date, COUNT(DISTINCT isin) AS uni_n "
            "FROM universe_snapshots WHERE index_name = 'russell_1000' "
            "GROUP BY snapshot_date ORDER BY snapshot_date",
            conn,
        )
    if scored.empty or uni.empty:
        return pd.DataFrame()

    # Map each model date to the nearest universe snapshot on or before it.
    # merge_asof needs datetime keys, so map on a parsed column then drop it.
    uni = uni.sort_values("snapshot_date")
    uni["_ts"] = pd.to_datetime(uni["snapshot_date"])
    model_dates = pd.DataFrame({"data_date": sorted(scored["data_date"].unique())})
    model_dates["_ts"] = pd.to_datetime(model_dates["data_date"])
    mapped = pd.merge_asof(
        model_dates, uni[["_ts", "uni_n"]], on="_ts", direction="backward",
    ).drop(columns="_ts")
    df = scored.merge(mapped, on="data_date", how="left")
    df["coverage_pct"] = (df["scored"] / df["uni_n"] * 100).round(1)
    df = df.merge(ref.rename(columns={"ModelID": "model_id"}), on="model_id", how="left")
    return df.sort_values("data_date")


@st.cache_data(ttl=300)
def _model_summary(snapshot_date: str) -> pd.DataFrame:
    """Per-model N, mean z, std z at a given snapshot — used for centring check."""
    ref = pd.read_csv(MODELS_REF)[["ModelID", "Model"]].drop_duplicates()
    with get_db(MODELS_DB) as conn:
        df = pd.read_sql(
            "SELECT model_id, "
            "  COUNT(*) AS N, "
            "  AVG(model_value_z) AS mean_z, "
            "  AVG(model_value_z * model_value_z) - AVG(model_value_z)*AVG(model_value_z) AS var_z "
            "FROM models WHERE data_date = ? AND model_value_z IS NOT NULL "
            "GROUP BY model_id",
            conn,
            params=(snapshot_date,),
        )
    df["std_z"] = df["var_z"].pow(0.5).round(3)
    df["mean_z"] = df["mean_z"].round(3)
    df = df.merge(ref.rename(columns={"ModelID": "model_id"}), on="model_id", how="left")
    return df[["Model", "N", "mean_z", "std_z"]].sort_values("Model")


@st.cache_data(ttl=300)
def _barra_snapshot_stats() -> pd.DataFrame:
    """Per-snapshot: n_stocks, avg/max idio_var from risk.db Barra factor snapshots."""
    with get_db(RISK_DB) as conn:
        exp = pd.read_sql(
            "SELECT snapshot_date, COUNT(DISTINCT security_id) AS n_stocks "
            "FROM factor_exposures GROUP BY snapshot_date ORDER BY snapshot_date",
            conn,
        )
        idio = pd.read_sql(
            "SELECT snapshot_date, AVG(idio_var) AS avg_idio, MAX(idio_var) AS max_idio "
            "FROM idiosyncratic_vars GROUP BY snapshot_date ORDER BY snapshot_date",
            conn,
        )
        fr = pd.read_sql(
            "SELECT MIN(trade_date) AS min_date, MAX(trade_date) AS max_date, COUNT(*) AS n_rows "
            "FROM factor_returns",
            conn,
        ).iloc[0]
    df = exp.merge(idio, on="snapshot_date", how="left")
    df["avg_idio"] = df["avg_idio"].round(5)
    df["max_idio"] = df["max_idio"].round(5)
    return df, fr


@st.cache_data(ttl=300)
def _returns_coverage() -> pd.DataFrame:
    """Last price date and days of history per ISIN."""
    with get_db(RETURNS_DB) as conn:
        df = pd.read_sql(
            "SELECT isin, MIN(date) AS first_date, MAX(date) AS last_date, COUNT(*) AS n_days "
            "FROM returns GROUP BY isin",
            conn,
        )
    df["first_date"] = pd.to_datetime(df["first_date"])
    df["last_date"]  = pd.to_datetime(df["last_date"])
    today = pd.Timestamp.today().normalize()
    df["days_stale"] = (today - df["last_date"]).dt.days
    return df


@st.cache_data(ttl=300)
def _constituent_coverage() -> pd.DataFrame:
    """Per-company: distinct fiscal years, first/last year, latest publish date."""
    with get_db(CONSTITUENTS_DB) as conn:
        df = pd.read_sql(
            "SELECT security_id, "
            "  COUNT(DISTINCT fiscal_year) AS n_years, "
            "  MIN(fiscal_year) AS first_fy, "
            "  MAX(fiscal_year) AS last_fy, "
            "  MAX(publish_date) AS latest_publish "
            "FROM constituents GROUP BY security_id",
            conn,
        )
    return df


@st.cache_data(ttl=300)
def _ltm_gap_check(threshold: int) -> pd.DataFrame:
    """
    Universe companies whose latest quarterly sort_key is below threshold.

    sort_key = fiscal_year*10 + period_num (Q1=1, Q2=2, Q3=3).  Takes the MAX
    across both ISIN-keyed (EDGAR) and SimFin-keyed records so dual-source
    companies aren't incorrectly flagged.
    """
    with get_db(CONSTITUENTS_DB) as conn:
        latest_qtrs = pd.read_sql(
            """
            SELECT security_id,
                   MAX(CASE fiscal_period
                       WHEN 'Q1' THEN fiscal_year * 10 + 1
                       WHEN 'Q2' THEN fiscal_year * 10 + 2
                       WHEN 'Q3' THEN fiscal_year * 10 + 3
                   END) AS latest_qtr_sk
            FROM constituents
            WHERE fiscal_period IN ('Q1','Q2','Q3')
            GROUP BY security_id
            """,
            conn,
        )
    with get_db(UNIVERSE_DB) as conn:
        companies = pd.read_sql(
            "SELECT isin, ticker, company_name, gics_sector, simfin_id FROM companies "
            "WHERE isin IS NOT NULL",
            conn,
        )
    companies["simfin_str"] = companies["simfin_id"].apply(
        lambda x: str(int(x)) if pd.notna(x) else None
    )
    isin_sk  = latest_qtrs.rename(columns={"security_id": "isin",       "latest_qtr_sk": "edgar_sk"})
    sfin_sk  = latest_qtrs.rename(columns={"security_id": "simfin_str", "latest_qtr_sk": "simfin_sk"})
    df = companies.merge(isin_sk, on="isin", how="left")
    df = df.merge(sfin_sk, on="simfin_str", how="left")
    df["latest_sk"] = df[["edgar_sk", "simfin_sk"]].max(axis=1)
    stale = df[df["latest_sk"].isna() | (df["latest_sk"] < threshold)].copy()
    return stale[["ticker", "company_name", "gics_sector", "latest_sk"]].sort_values(
        "latest_sk", ascending=True, na_position="first"
    )


def _sk_label(sk: int) -> str:
    year, q = divmod(int(sk), 10)
    ql = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}.get(q, str(q))
    return f"{ql} FY{year}"


def _expected_sk_today() -> int:
    """Sort_key of the most recent quarter that should be available by today (2-month filing lag)."""
    today = date.today()
    best = 0
    for yr in [today.year, today.year - 1]:
        for q_num, q_month in [(3, 9), (2, 6), (1, 3)]:
            deadline_month = q_month + 2
            deadline_yr    = yr + (1 if deadline_month > 12 else 0)
            deadline_month = deadline_month % 12 or 12
            if date(deadline_yr, deadline_month, 1) <= today:
                best = max(best, yr * 10 + q_num)
    return best


@st.cache_data(ttl=300)
def _quarter_coverage() -> pd.DataFrame:
    """
    For each quarter (fiscal_year × fiscal_period) in the last 2 years, return
    the count of active universe companies that have data (EDGAR or SimFin).
    """
    today  = date.today()
    min_fy = today.year - 2

    with get_db(UNIVERSE_DB) as conn:
        snap = conn.execute("SELECT MAX(snapshot_date) FROM universe_snapshots").fetchone()[0]
        universe = pd.read_sql(
            "SELECT DISTINCT c.isin, c.simfin_id "
            "FROM companies c JOIN universe_snapshots us ON us.isin = c.isin "
            "WHERE us.snapshot_date = ?", conn, params=(snap,)
        )

    universe["simfin_str"] = universe["simfin_id"].apply(
        lambda x: str(int(x)) if pd.notna(x) else None
    )
    isin_set   = set(universe["isin"].dropna())
    sid_to_isin = dict(zip(universe["simfin_str"].dropna(), universe["isin"].dropna()))

    with get_db(CONSTITUENTS_DB) as conn:
        raw = pd.read_sql(
            "SELECT security_id, fiscal_year, fiscal_period "
            "FROM constituents "
            "WHERE fiscal_period IN ('Q1','Q2','Q3') AND fiscal_year >= ? "
            "GROUP BY security_id, fiscal_year, fiscal_period",
            conn, params=(min_fy,)
        )

    raw["isin"] = raw["security_id"].apply(lambda s: s if s in isin_set else sid_to_isin.get(s))
    raw = raw.dropna(subset=["isin"])
    raw = raw[raw["isin"].isin(isin_set)]

    counts = (
        raw.groupby(["fiscal_year", "fiscal_period"])["isin"]
        .nunique()
        .reset_index(name="companies_with_data")
    )
    counts["quarter"]      = counts["fiscal_period"] + " FY" + counts["fiscal_year"].astype(str)
    counts["sort_key"]     = counts["fiscal_year"] * 10 + counts["fiscal_period"].map({"Q1": 1, "Q2": 2, "Q3": 3})
    counts["universe_size"] = len(isin_set)
    counts["pct_covered"]  = (counts["companies_with_data"] / len(isin_set) * 100).round(1)
    return counts.sort_values("sort_key")


@st.cache_data(ttl=300)
def _gap_detail() -> pd.DataFrame:
    """
    Per-company gap matrix: which quarters in FY2024/FY2025/FY2026 are present,
    whether FY2025 annual exists, and a rollup status label.
    """
    FPI_ISINS = {
        "BMG611881019","BMG93A5A1010","CA1130041058","CA11285B1085","CH1134540470",
        "JE00BS44BN30","KYG0260P1028","KYG169101204","KYG393871085","KYG6683N1034",
        "KYG982391099","LU0038705702","LU0974299876","LU1778762911","NL0010545661",
        "NL0015002CX3","GB0022569080","GB00BRXH2664","IE00028FXN24","IE000R94NGM2",
    }

    with get_db(UNIVERSE_DB) as conn:
        snap = conn.execute("SELECT MAX(snapshot_date) FROM universe_snapshots").fetchone()[0]
        universe = pd.read_sql(
            "SELECT DISTINCT c.ticker, c.isin, c.simfin_id, c.fiscal_year_end, c.gics_sector "
            "FROM companies c JOIN universe_snapshots us ON us.isin = c.isin "
            "WHERE us.snapshot_date = ?", conn, params=(snap,)
        )

    universe["simfin_str"] = universe["simfin_id"].apply(
        lambda x: str(int(x)) if pd.notna(x) else None
    )

    with get_db(CONSTITUENTS_DB) as conn:
        qraw = pd.read_sql(
            "SELECT security_id, fiscal_year, fiscal_period FROM constituents "
            "WHERE fiscal_period IN ('Q1','Q2','Q3') AND fiscal_year >= 2024 "
            "GROUP BY security_id, fiscal_year, fiscal_period", conn
        )
        araw = pd.read_sql(
            "SELECT security_id, fiscal_year FROM constituents "
            "WHERE fiscal_period IN ('FY','Q4') AND fiscal_year >= 2024 "
            "GROUP BY security_id, fiscal_year", conn
        )

    sid_to_isin = dict(zip(universe["simfin_str"].dropna(), universe["isin"].dropna()))
    isin_set    = set(universe["isin"].dropna())

    def _resolve(sid: str) -> str | None:
        return sid if sid in isin_set else sid_to_isin.get(sid)

    qraw["isin"] = qraw["security_id"].apply(_resolve)
    araw["isin"] = araw["security_id"].apply(_resolve)
    qraw = qraw.dropna(subset=["isin"])
    araw = araw.dropna(subset=["isin"])

    q_set = qraw.groupby("isin").apply(
        lambda g: set(zip(g["fiscal_year"], g["fiscal_period"]))
    ).to_dict()
    a_set = araw.groupby("isin")["fiscal_year"].apply(set).to_dict()

    rows = []
    for _, row in universe.iterrows():
        isin = row["isin"]
        if isin in FPI_ISINS:
            continue
        qs = q_set.get(isin, set())
        an = a_set.get(isin, set())

        def _has(fy: int, p: str) -> bool:
            return (fy, p) in qs

        gap24 = [p for p in ("Q1","Q2","Q3") if not _has(2024, p)]
        gap25 = [p for p in ("Q1","Q2","Q3") if not _has(2025, p)]
        has_ann25 = 2025 in an

        if not qs and not an:
            status = "No data"
        elif gap24 == ["Q1","Q2","Q3"] and gap25 == ["Q1","Q2","Q3"]:
            status = "No quarterly history"
        elif gap25:
            status = "FY2025 gaps"
        elif gap24:
            status = "FY2024 gaps only"
        elif not has_ann25:
            status = "Missing FY2025 annual"
        else:
            status = "OK"

        rows.append({
            "ticker":        row["ticker"],
            "sector":        row["gics_sector"],
            "fye":           int(row["fiscal_year_end"]) if pd.notna(row["fiscal_year_end"]) else 12,
            "FY2024 Q1":     "✓" if _has(2024,"Q1") else "✗",
            "FY2024 Q2":     "✓" if _has(2024,"Q2") else "✗",
            "FY2024 Q3":     "✓" if _has(2024,"Q3") else "✗",
            "FY2025 Q1":     "✓" if _has(2025,"Q1") else "✗",
            "FY2025 Q2":     "✓" if _has(2025,"Q2") else "✗",
            "FY2025 Q3":     "✓" if _has(2025,"Q3") else "✗",
            "FY2026 Q1":     "✓" if _has(2026,"Q1") else "✗",
            "Annual FY2025": "✓" if has_ann25 else "✗",
            "Status":        status,
        })

    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def _recent_filings(n_days: int = 90) -> pd.DataFrame:
    """Filings published in the last N days (by publish_date)."""
    cutoff = (pd.Timestamp.today() - pd.Timedelta(days=n_days)).strftime("%Y-%m-%d")
    with get_db(CONSTITUENTS_DB) as conn:
        df = pd.read_sql(
            "SELECT security_id, fiscal_year, fiscal_period, statement_type, publish_date "
            "FROM constituents WHERE publish_date >= ? "
            "GROUP BY security_id, fiscal_year, fiscal_period "
            "ORDER BY publish_date DESC LIMIT 500",
            conn,
            params=(cutoff,),
        )
    return df


@st.cache_data(ttl=300)
def _universe_summary() -> dict:
    with get_db(UNIVERSE_DB) as conn:
        total       = conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
        with_ticker = conn.execute("SELECT COUNT(*) FROM companies WHERE ticker IS NOT NULL").fetchone()[0]
        with_isin   = conn.execute("SELECT COUNT(*) FROM companies WHERE isin IS NOT NULL").fetchone()[0]
        snap_dates  = conn.execute(
            "SELECT COUNT(DISTINCT snapshot_date) FROM universe_snapshots WHERE index_name='russell_1000'"
        ).fetchone()[0]
        latest_snap = conn.execute(
            "SELECT MAX(snapshot_date) FROM universe_snapshots WHERE index_name='russell_1000'"
        ).fetchone()[0]
    return {
        "total": total, "with_ticker": with_ticker, "with_isin": with_isin,
        "snap_dates": snap_dates, "latest_snap": latest_snap,
    }


@st.cache_data(ttl=300)
def _risk_summary() -> pd.DataFrame:
    if not RISK_DB.exists():
        return pd.DataFrame()
    with get_db(RISK_DB) as conn:
        return pd.read_sql(
            "SELECT data_date, n_stocks, shrinkage_coeff, computation_date "
            "FROM covariance_matrix ORDER BY data_date",
            conn,
        )


# ---------------------------------------------------------------------------
# TOP-LEVEL KPIs
# ---------------------------------------------------------------------------

with st.spinner("Loading pipeline status…"):
    db_meta      = _db_meta()
    snap_cov     = _snapshot_coverage()
    univ_summary = _universe_summary()

latest_factor_date = snap_cov["data_date"].max() if not snap_cov.empty else "N/A"
latest_factor_n    = int(snap_cov.loc[snap_cov["data_date"] == latest_factor_date, "factors"].iloc[0]) if not snap_cov.empty else 0
n_snap_dates       = len(snap_cov)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Universe companies",  f"{univ_summary['total']:,}")
c2.metric("Snapshot dates",      f"{n_snap_dates}")
c3.metric("Latest snapshot",     latest_factor_date)
c4.metric("Companies (latest)",  f"{latest_factor_n:,}")
c5.metric("Russell 1000 snaps",  f"{univ_summary['snap_dates']}")

st.divider()

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------

tabs = st.tabs([
    "🗄️ Pipeline Health",
    "📅 Snapshot Coverage",
    "🔢 Factor Quality",
    "📊 Model Quality",
    "⚡ Barra Quality",
    "📈 Return Coverage",
    "📋 Constituents",
])
tab_health, tab_snap, tab_factor, tab_model, tab_barra, tab_ret, tab_const = tabs


# ============================================================
# TAB 1: Pipeline Health
# ============================================================
with tab_health:
    st.subheader("Database file health")

    col_tbl, col_sync = st.columns([2, 1])

    with col_tbl:
        st.dataframe(db_meta, use_container_width=True, hide_index=True)

    with col_sync:
        st.markdown("**Sync status**")
        if not snap_cov.empty:
            synced = snap_cov[snap_cov["barra"] > 0]
            latest_synced = synced.iloc[-1] if not synced.empty else snap_cov.iloc[-1]

            f_n = latest_synced["factors"]
            m_n = latest_synced["models"]
            b_n = latest_synced["barra"]

            def _check(label, val, ref):
                diff = abs(val - ref)
                if val == 0:
                    st.error(f"❌ {label}: 0 (missing)")
                elif diff <= 2:
                    st.success(f"✅ {label}: {val:,}")
                else:
                    st.warning(f"⚠️ {label}: {val:,} (Δ{diff} vs factors)")

            st.caption(f"At latest Barra-synced snapshot **{latest_synced['data_date']}**")
            _check("Factors", f_n, f_n)
            _check("Models",  m_n, f_n)
            _check("Barra",   b_n, f_n)

            extra = snap_cov[
                (snap_cov["data_date"] > latest_synced["data_date"]) &
                (snap_cov["factors"] > 0)
            ]
            if not extra.empty:
                dates_str = ", ".join(extra["data_date"].tolist())
                st.caption(f"ℹ️ {dates_str}: factors/models exist but no Barra yet")
        else:
            st.info("No snapshot data found.")

    st.divider()
    st.subheader("Row counts per table")

    row_counts = []
    db_map = {
        "universe":     (UNIVERSE_DB,     ["companies", "universe_snapshots", "isin_patch", "ticker_alias", "index_registry", "nport_accessions"]),
        "constituents": (CONSTITUENTS_DB, ["constituents"]),
        "returns":      (RETURNS_DB,      ["returns", "svr_daily"]),
        "factors":      (FACTORS_DB,      ["factors"]),
        "models":       (MODELS_DB,       ["models"]),
        "risk":         (RISK_DB,         ["covariance_matrix"]),
        "barra":        (RISK_DB,         ["factor_returns", "factor_covariance", "idiosyncratic_vars", "factor_exposures"]),
    }

    for db_name, (db_path, tables) in db_map.items():
        if not Path(db_path).exists():
            row_counts.append({"Database": db_name, "Table": "(missing)", "Rows": None})
            continue
        with get_db(db_path) as conn:
            existing_tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
        for tbl in tables:
            if tbl not in existing_tables:
                row_counts.append({"Database": db_name, "Table": tbl, "Rows": None})
                continue
            with get_db(db_path) as conn:
                n = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            row_counts.append({"Database": db_name, "Table": tbl, "Rows": n})

    rc_df = pd.DataFrame(row_counts)
    rc_df["Rows"] = rc_df["Rows"].apply(lambda x: f"{x:,}" if pd.notna(x) else "—")
    st.dataframe(rc_df, hide_index=True, use_container_width=True)


# ============================================================
# TAB 2: Snapshot Coverage
# ============================================================
with tab_snap:
    st.subheader("Snapshot coverage")

    last_barra_date = snap_cov.loc[snap_cov["barra"] > 0, "data_date"].max() if (snap_cov["barra"] > 0).any() else ""

    def _status(row) -> str:
        if row["factors"] == 0:
            return "❌ Missing"
        if row["models"] != row["factors"]:
            return "⚠️ Models gap"
        if row["barra"] == 0:
            return "ℹ️ No Barra (expected)" if row["data_date"] > last_barra_date else "⚠️ No Barra"
        if abs(row["barra"] - row["factors"]) > 2:
            return "⚠️ Barra gap"
        return "✅ Synced"

    cov_display = snap_cov.copy()
    cov_display["Status"] = cov_display.apply(_status, axis=1)

    synced_count    = (cov_display["Status"] == "✅ Synced").sum()
    issues_count    = cov_display["Status"].str.startswith("⚠️").sum()
    latest_synced_n = int(snap_cov.loc[snap_cov["data_date"] == last_barra_date, "factors"].iloc[0]) if last_barra_date else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total snapshot dates",    len(cov_display))
    k2.metric("Fully synced dates",      synced_count)
    k3.metric("Dates with issues",       issues_count)
    k4.metric("Latest synced companies", f"{latest_synced_n:,}")

    st.divider()

    # Universe growth over time — useful for spotting coverage drops at a date
    fig_growth = go.Figure()
    fig_growth.add_trace(go.Scatter(
        x=cov_display["data_date"],
        y=cov_display["factors"],
        mode="lines+markers",
        name="Companies with factor data",
        line=dict(color="#4C8BF5", width=2),
        marker=dict(size=7),
        hovertemplate="<b>%{x}</b><br>%{y} companies<extra></extra>",
    ))
    fig_growth.update_layout(
        height=300,
        xaxis_title=None,
        yaxis_title="Companies in snapshot",
        margin=dict(l=0, r=0, t=10, b=40),
        showlegend=False,
        xaxis=dict(tickangle=-40),
        hovermode="x unified",
    )
    st.plotly_chart(fig_growth, use_container_width=True)

    tbl = cov_display[["data_date", "factors", "models", "barra", "Status"]].copy()
    tbl.columns = ["Date", "Factors", "Models", "Barra", "Status"]
    tbl = tbl.sort_values("Date", ascending=False).reset_index(drop=True)
    st.dataframe(tbl, hide_index=True, use_container_width=True, height=550)


# ============================================================
# TAB 3: Factor Quality
# ============================================================
with tab_factor:
    st.subheader("Factor coverage over time")
    st.caption(
        "Share of the PIT Russell 1000 with a value for each factor in the selected "
        "model. **This is the regression-catching view** — per-factor *fill %* (table "
        "below) divides by rows that exist, so it reads ~100% even when a factor covers "
        "a fraction of the universe. A drop or a structurally low line here is the real "
        "signal. Sector-gated factors (bank/REIT-only) sit low by design."
    )

    _mref = pd.read_csv(MODELS_REF)
    _base = _mref[_mref["IsComposite"] == 0][["ModelID", "Model"]].drop_duplicates()
    _model_opts = dict(zip(_base["Model"] + " (" + _base["ModelID"] + ")", _base["ModelID"]))
    sel_model_label = st.selectbox("Model", list(_model_opts), index=0, key="fq_model")
    sel_model_id = _model_opts[sel_model_label]

    with st.spinner("Loading factor coverage…"):
        fcov = _factor_coverage_over_time(sel_model_id)

    if fcov.empty:
        st.info("No factor coverage data for this model.")
    else:
        # Regression detector: flag a factor whose CURRENT coverage dropped materially
        # from its own historical peak. This ignores structurally-low sector-gated
        # factors (always low ≠ regression) and catches the silent drops that the
        # all-green fill % missed.
        peak = fcov.groupby("Factor")["coverage_pct"].max().rename("peak")
        cur  = (fcov[fcov["data_date"] == fcov["data_date"].max()]
                .set_index("Factor")["coverage_pct"].rename("cur"))
        chk  = pd.concat([peak, cur], axis=1).dropna()
        drops = chk[(chk["peak"] >= 40) & (chk["cur"] < chk["peak"] - 15)]
        if not drops.empty:
            st.warning(
                "⚠️ Coverage regression vs each factor's own peak: "
                + ", ".join(f"{f} {r.cur:.0f}% (peak {r.peak:.0f}%)" for f, r in drops.iterrows())
            )
        else:
            st.success("✅ No factor coverage regressions (each factor near its historical peak).")
        fig_fcov = px.line(
            fcov, x="data_date", y="coverage_pct", color="Factor", markers=False,
            labels={"data_date": "", "coverage_pct": "Coverage (%)", "Factor": "Factor"},
        )
        fig_fcov.update_layout(
            height=440, margin=dict(l=0, r=0, t=10, b=10),
            yaxis=dict(range=[0, 102], ticksuffix="%"),
            legend=dict(orientation="h", y=-0.2), hovermode="x unified",
        )
        st.plotly_chart(fig_fcov, use_container_width=True)

    st.divider()

    # Per-date detail (fill %, z health, anomalies) — needs a specific snapshot.
    factor_dates = snap_cov["data_date"].tolist()
    sel_date_f   = st.selectbox("Snapshot date (for the detail below)", factor_dates,
                                index=len(factor_dates) - 1, key="fq_date")

    with st.spinner("Loading factor data…"):
        fill_df = _factor_fill(sel_date_f)

    # Compact quality table: fill %, avg z, extreme count
    st.markdown("**Factor quality summary**")
    tbl_f = fill_df[["factor_name", "category", "fill_pct", "avg_z", "n_extreme"]].copy()
    tbl_f.columns = ["Factor", "Category", "Fill %", "Avg Z", "|Z|>4 count"]
    tbl_f["Avg Z"] = tbl_f["Avg Z"].round(3)

    low_fill = tbl_f["Fill %"] < 80
    extreme  = tbl_f["|Z|>4 count"] > 0
    if low_fill.any():
        st.warning(f"⚠️ {low_fill.sum()} factor(s) below 80% fill: {', '.join(tbl_f.loc[low_fill, 'Factor'].tolist())}")
    if extreme.any():
        st.warning(f"⚠️ {extreme.sum()} factor(s) with extreme z-scores (|z|>4): {', '.join(tbl_f.loc[extreme, 'Factor'].tolist())}")
    if not low_fill.any() and not extreme.any():
        st.success("✅ All factors have ≥ 80% fill and no extreme z-scores.")

    st.dataframe(tbl_f.sort_values("Fill %"), hide_index=True, use_container_width=True)

    # --- Factor value anomalies ---------------------------------------
    st.subheader("Factor value anomalies")
    st.caption(
        "Suspect individual values from two unsupervised lenses — cross-sectional "
        "outliers (raw value vs peers) and time-series jumps (value vs the company's "
        "own history). Catches XBRL mis-tags and near-zero-denominator blow-ups that "
        "distort z-scores but slip past the aggregate fill/|z| checks above. "
        "“Both lenses” = highest confidence."
    )

    with st.spinner("Scanning factor values…"):
        anom = _factor_anomalies(sel_date_f)

    if anom.empty:
        st.success("✅ No factor value anomalies detected at this snapshot.")
    else:
        n_xs   = int((anom["lens"] == "cross_sectional").sum())
        n_ts   = int((anom["lens"] == "time_series").sum())
        n_both = int((anom["confidence"] == "both lenses").sum()) // 2  # counted per lens row
        c1, c2, c3 = st.columns(3)
        c1.metric("Suspect cells", f"{len(anom):,}")
        c2.metric("Cross-sectional / time-series", f"{n_xs} / {n_ts}")
        c3.metric("Flagged by both lenses", f"{n_both}")
        st.warning(
            f"⚠️ {len(anom):,} suspect value(s). Top offenders below — review the "
            "highest-severity and “both lenses” rows first."
        )

        show = anom.head(50)[
            ["lens", "ticker", "sector", "factor_name", "factor_value",
             "severity", "confidence", "detail"]
        ].rename(columns={
            "lens": "Lens", "ticker": "Ticker", "sector": "Sector",
            "factor_name": "Factor", "factor_value": "Value",
            "severity": "Robust-z", "confidence": "Confidence", "detail": "Detail",
        })
        show["Value"]    = show["Value"].map(lambda v: f"{v:.3g}")
        show["Robust-z"] = show["Robust-z"].round(0).astype(int)
        st.dataframe(show, hide_index=True, use_container_width=True, height=460)


# ============================================================
# TAB 4: Model Quality
# ============================================================
with tab_model:
    st.subheader("Model score quality")

    with st.spinner("Loading model data…"):
        coverage = _model_coverage_over_time()

    model_dates = sorted(coverage["data_date"].unique(), reverse=True)
    sel_date_m  = st.selectbox("Snapshot date", model_dates, index=0, key="mq_date")

    # Coverage over time: % of the PIT russell_1000 universe each model scores.
    st.markdown("**Model coverage over time (% of point-in-time Russell 1000 scored)**")
    st.caption(
        "Share of the PIT universe with a model score at each snapshot. Sector-gated "
        "models (e.g. Growth, Short Interest) sit structurally lower; watch for "
        "*drops* over time, which signal a data-coverage regression."
    )
    fig_cov = px.line(
        coverage, x="data_date", y="coverage_pct", color="Model",
        markers=False,
        labels={"data_date": "", "coverage_pct": "Coverage (%)", "Model": "Model"},
    )
    fig_cov.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=10, b=10),
        yaxis=dict(range=[0, 102], ticksuffix="%"),
        legend=dict(orientation="h", y=-0.18),
        hovermode="x unified",
    )
    st.plotly_chart(fig_cov, use_container_width=True)

    st.divider()

    with st.spinner("Loading score summary…"):
        summary_m = _model_summary(sel_date_m)

    st.markdown(f"**Score centring at {sel_date_m}** — z-scores should be well-centred (|mean| ≤ 0.1)")
    off_centre = summary_m[summary_m["mean_z"].abs() > 0.1]
    if not off_centre.empty:
        st.warning(f"⚠️ Off-centre models: {', '.join(off_centre['Model'].tolist())}")
    else:
        st.success("✅ All model z-scores are well-centred.")

    summary_m.columns = ["Model", "N scored", "Mean Z", "Std Z"]
    st.dataframe(summary_m, hide_index=True, use_container_width=True)


# ============================================================
# TAB 5: Barra Quality
# ============================================================
with tab_barra:
    st.subheader("Barra factor risk model quality")

    with st.spinner("Loading Barra data…"):
        barra_stats, fr_info = _barra_snapshot_stats()
        risk_df = _risk_summary()

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Factor return rows",   f"{fr_info['n_rows']:,}")
    col_b.metric("Factor returns start", fr_info["min_date"])
    col_c.metric("Factor returns end",   fr_info["max_date"])
    col_d.metric("Barra snapshots",      f"{len(barra_stats):,}")

    st.divider()

    # Factor-snapshot dates: those with full-universe n_stocks
    q75          = barra_stats["n_stocks"].quantile(0.75)
    factor_snaps = barra_stats[barra_stats["n_stocks"] >= q75 * 0.9].copy()

    st.markdown(f"**Factor-snapshot dates ({len(factor_snaps)} dates)**")
    st.caption("Align with factors.db/models.db. n_stocks drop = universe coverage issue.")

    fig_fsnap = px.bar(
        factor_snaps, x="snapshot_date", y="n_stocks",
        labels={"snapshot_date": "", "n_stocks": "Securities"},
        text="n_stocks",
        color_discrete_sequence=["#4C8BF5"],
    )
    fig_fsnap.update_traces(textposition="outside")
    fig_fsnap.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=20, b=60),
        xaxis_tickangle=-45,
        yaxis=dict(range=[0, factor_snaps["n_stocks"].max() * 1.15]),
    )
    st.plotly_chart(fig_fsnap, use_container_width=True)

    tbl_barra = factor_snaps[["snapshot_date", "n_stocks", "avg_idio", "max_idio"]].copy()
    tbl_barra.columns = ["Date", "Securities", "Avg Idio Var", "Max Idio Var"]
    st.dataframe(tbl_barra, hide_index=True, use_container_width=True)

    st.divider()

    # Ledoit-Wolf risk snapshots
    st.markdown("**Ledoit-Wolf covariance matrices (risk.db)**")
    if risk_df.empty:
        st.info("risk.db not found or empty.")
    else:
        st.dataframe(risk_df, hide_index=True, use_container_width=True)


# ============================================================
# TAB 6: Return Coverage
# ============================================================
with tab_ret:
    st.subheader("Price / return data coverage")

    with st.spinner("Loading returns metadata…"):
        ret_cov = _returns_coverage()

    stale_threshold = st.slider("Stale threshold (days since last price)", 5, 60, 10, key="stale_thresh")

    stale      = ret_cov[ret_cov["days_stale"] > stale_threshold]
    current    = ret_cov[ret_cov["days_stale"] <= stale_threshold]
    short_hist = ret_cov[ret_cov["n_days"] < 252]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ISINs with prices",          f"{len(ret_cov):,}")
    c2.metric("Current (≤ threshold days)", f"{len(current):,}")
    c3.metric("Stale (> threshold days)",   f"{len(stale):,}")
    c4.metric("Short history (< 1yr)",      f"{len(short_hist):,}")

    if not stale.empty:
        st.markdown(f"**Stale companies — last price > {stale_threshold} days ago**")
        stale_disp = stale.sort_values("last_date")[["isin", "first_date", "last_date", "n_days", "days_stale"]].copy()
        stale_disp["last_date"]  = stale_disp["last_date"].dt.strftime("%Y-%m-%d")
        stale_disp["first_date"] = stale_disp["first_date"].dt.strftime("%Y-%m-%d")
        with get_db(UNIVERSE_DB) as conn:
            tickers = pd.read_sql("SELECT isin, ticker, company_name FROM companies", conn)
        stale_disp = stale_disp.merge(tickers, on="isin", how="left")
        stale_disp = stale_disp[["ticker", "company_name", "isin", "first_date", "last_date", "n_days", "days_stale"]]
        st.dataframe(stale_disp.head(100), hide_index=True, use_container_width=True)
    else:
        st.success(f"✅ All ISINs have prices within the last {stale_threshold} days.")

    st.divider()
    st.markdown("**Companies in universe but missing from returns.db**")

    with get_db(UNIVERSE_DB) as conn:
        universe_isins = set(r[0] for r in conn.execute("SELECT isin FROM companies WHERE isin IS NOT NULL").fetchall())
    missing_from_returns = universe_isins - set(ret_cov["isin"].tolist())

    if missing_from_returns:
        with get_db(UNIVERSE_DB) as conn:
            missing_df = pd.read_sql(
                f"SELECT isin, ticker, company_name, gics_sector FROM companies "
                f"WHERE isin IN ({','.join('?' * len(missing_from_returns))})",
                conn,
                params=list(missing_from_returns),
            )
        st.warning(f"⚠️ {len(missing_from_returns)} companies in universe.db have no price data:")
        st.dataframe(missing_df, hide_index=True, use_container_width=True)
    else:
        st.success("✅ All universe companies have price data in returns.db.")


# ============================================================
# TAB 7: Constituents
# ============================================================
with tab_const:
    st.subheader("Constituent (financial statement) data quality")

    # Companies with no constituent data at all
    with st.spinner("Loading constituent coverage…"):
        const_cov = _constituent_coverage()

    const_sids = set(const_cov["security_id"].astype(str).tolist())
    with get_db(UNIVERSE_DB) as conn:
        companies_all = pd.read_sql(
            "SELECT isin, ticker, company_name, gics_sector, simfin_id, cik FROM companies", conn
        )
    companies_all["simfin_str"] = companies_all["simfin_id"].dropna().apply(lambda x: str(int(x)))
    companies_all["has_const"]  = (
        companies_all["isin"].isin(const_sids) |
        companies_all["simfin_str"].isin(const_sids)
    )
    no_const_df = companies_all[~companies_all["has_const"]].copy()

    c1, c2 = st.columns(2)
    c1.metric("Companies with any constituent data", f"{len(const_cov):,}")
    c2.metric("Universe companies with NO data",     f"{len(no_const_df):,}")

    if not no_const_df.empty:
        nc_disp = no_const_df[["ticker", "company_name", "gics_sector", "simfin_id", "cik"]].copy()
        nc_disp["simfin_id"] = nc_disp["simfin_id"].apply(lambda x: str(int(x)) if pd.notna(x) else "—")
        nc_disp["cik"] = nc_disp["cik"].fillna("—")
        st.warning(f"⚠️ {len(no_const_df)} universe companies have no constituent data:")
        st.dataframe(nc_disp, hide_index=True, use_container_width=True)
    else:
        st.success("✅ All universe companies have constituent data.")

    # Companies with sparse historical coverage
    st.divider()
    st.markdown("**Companies with sparse history (< 3 fiscal years)**")
    sparse = const_cov[const_cov["n_years"] < 3].sort_values("n_years")
    if sparse.empty:
        st.success("✅ All companies have at least 3 fiscal years of data.")
    else:
        with get_db(UNIVERSE_DB) as conn:
            tickers = pd.read_sql("SELECT isin, ticker, company_name FROM companies", conn)
        sparse = sparse.merge(tickers.rename(columns={"isin": "security_id"}), on="security_id", how="left")
        sparse_disp = sparse[["ticker", "company_name", "security_id", "n_years", "first_fy", "last_fy", "latest_publish"]]
        st.warning(f"⚠️ {len(sparse)} companies have fewer than 3 fiscal years:")
        st.dataframe(sparse_disp.head(50), hide_index=True, use_container_width=True)

    # LTM window gaps
    st.divider()
    st.subheader("LTM window gaps")
    _exp_sk    = _expected_sk_today()
    _exp_label = _sk_label(_exp_sk) if _exp_sk else "N/A"
    st.caption(
        f"Universe companies whose latest quarterly sort_key is below the expected minimum "
        f"({_exp_label}, 2-month filing lag applied). "
        "Should drop to near-zero once the EDGAR quarterly backfill completes."
    )

    with st.spinner("Checking LTM gaps…"):
        gap_df = _ltm_gap_check(_exp_sk)

    if gap_df.empty:
        st.success(f"✅ All universe companies have quarterly data through at least {_exp_label}.")
    else:
        gap_df["Latest Quarter"] = gap_df["latest_sk"].apply(
            lambda x: _sk_label(x) if pd.notna(x) else "No quarterly data"
        )
        no_data   = gap_df["latest_sk"].isna().sum()
        thru_2024 = ((gap_df["latest_sk"] < 20250) & gap_df["latest_sk"].notna()).sum()
        thru_q1   = ((gap_df["latest_sk"] >= 20250) & (gap_df["latest_sk"] < 20252) & gap_df["latest_sk"].notna()).sum()
        thru_q2   = ((gap_df["latest_sk"] >= 20252) & (gap_df["latest_sk"] < 20253) & gap_df["latest_sk"].notna()).sum()

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric(f"Below {_exp_label}",  f"{len(gap_df):,}")
        k2.metric("No quarterly data",    f"{no_data:,}")
        k3.metric("Latest ≤ FY2024",      f"{thru_2024:,}")
        k4.metric("Latest = FY2025 Q1",   f"{thru_q1:,}")
        k5.metric("Latest = FY2025 Q2",   f"{thru_q2:,}")

        col_tbl, col_chart = st.columns([3, 2])
        with col_tbl:
            st.dataframe(
                gap_df[["ticker", "company_name", "gics_sector", "Latest Quarter"]],
                hide_index=True, use_container_width=True, height=420,
            )
        with col_chart:
            sector_counts = (
                gap_df.groupby("gics_sector").size()
                .reset_index(name="count")
                .sort_values("count", ascending=True)
            )
            fig_sec = px.bar(
                sector_counts, x="count", y="gics_sector", orientation="h",
                labels={"count": "# companies with gaps", "gics_sector": ""},
                color="count", color_continuous_scale="Reds",
            )
            fig_sec.update_layout(
                height=420, margin=dict(l=0, r=0, t=20, b=20),
                showlegend=False, coloraxis_showscale=False,
            )
            st.plotly_chart(fig_sec, use_container_width=True)

    # Quarter-by-quarter coverage
    st.divider()
    st.subheader("Quarter-by-quarter constituent coverage")
    st.caption(
        "Percentage of active universe companies with at least one filing for each quarter. "
        "FY2026 Q1 is expected to be lower until 10-Q filings are published (May–June 2026)."
    )

    with st.spinner("Loading quarter coverage…"):
        qcov = _quarter_coverage()

    if qcov.empty:
        st.info("No quarterly constituent data found.")
    else:
        fig_qcov = px.bar(
            qcov,
            x="quarter",
            y="pct_covered",
            text=qcov.apply(lambda r: f"{r['companies_with_data']:,} ({r['pct_covered']:.0f}%)", axis=1),
            labels={"quarter": "Quarter", "pct_covered": "% of universe covered"},
            color="pct_covered",
            color_continuous_scale="RdYlGn",
            range_color=[50, 100],
        )
        fig_qcov.add_hline(y=90, line_dash="dash", line_color="orange", annotation_text="90% target")
        fig_qcov.update_traces(textposition="outside")
        fig_qcov.update_layout(
            height=340,
            margin=dict(l=0, r=0, t=20, b=40),
            yaxis=dict(range=[0, 115], title="% of universe"),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_qcov, use_container_width=True)

        _kc1, _kc2, _kc3 = st.columns(3)
        _kc1.metric("Universe size", f"{qcov['universe_size'].iloc[0]:,}")
        _exp_row = qcov[qcov["sort_key"] == _exp_sk]
        if not _exp_row.empty:
            _kc2.metric(
                f"Expected quarter ({_exp_label})",
                f"{_exp_row['companies_with_data'].iloc[0]:,} ({_exp_row['pct_covered'].iloc[0]:.0f}%)",
            )
        _latest_row = qcov.iloc[-1]
        _kc3.metric(
            f"Latest available ({_latest_row['quarter']})",
            f"{_latest_row['companies_with_data']:,} ({_latest_row['pct_covered']:.0f}%)",
        )

    # Per-company gap matrix
    st.divider()
    st.subheader("Per-company coverage matrix")
    st.caption(
        "FY2024/FY2025/FY2026 quarter presence per universe company. "
        "FPIs (foreign private issuers filing 20-F/40-F annually) excluded. "
        "✓ = filing present in constituents.db; ✗ = missing."
    )

    with st.spinner("Building coverage matrix…"):
        gap_detail_df = _gap_detail()

    _statuses = ["OK", "Missing FY2025 annual", "FY2025 gaps", "FY2024 gaps only", "No quarterly history", "No data"]
    _gs_cols = st.columns(len(_statuses))
    for _col, _st in zip(_gs_cols, _statuses):
        _col.metric(_st, f"{(gap_detail_df['Status'] == _st).sum():,}")

    _sel_status = st.selectbox("Filter by status", ["All"] + _statuses, key="gap_detail_status")
    _disp_df = (
        gap_detail_df[gap_detail_df["Status"] == _sel_status]
        if _sel_status != "All"
        else gap_detail_df
    ).sort_values(["Status", "ticker"]).copy()

    st.dataframe(
        _disp_df,
        hide_index=True,
        use_container_width=True,
        height=520,
        column_config={
            "ticker":        st.column_config.TextColumn("Ticker",    width="small"),
            "sector":        st.column_config.TextColumn("Sector"),
            "fye":           st.column_config.NumberColumn("FYE", format="%d", width="small"),
            "FY2024 Q1":     st.column_config.TextColumn("24 Q1",    width="small"),
            "FY2024 Q2":     st.column_config.TextColumn("24 Q2",    width="small"),
            "FY2024 Q3":     st.column_config.TextColumn("24 Q3",    width="small"),
            "FY2025 Q1":     st.column_config.TextColumn("25 Q1",    width="small"),
            "FY2025 Q2":     st.column_config.TextColumn("25 Q2",    width="small"),
            "FY2025 Q3":     st.column_config.TextColumn("25 Q3",    width="small"),
            "FY2026 Q1":     st.column_config.TextColumn("26 Q1",    width="small"),
            "Annual FY2025": st.column_config.TextColumn("Ann 25",   width="small"),
            "Status":        st.column_config.TextColumn("Status"),
        },
    )

    # Recent filings
    st.divider()
    st.subheader("Recent filings")
    days_window = st.slider("Window (days)", 30, 180, 90, step=30, key="const_window")
    with st.spinner("Fetching recent filings…"):
        recent = _recent_filings(n_days=days_window)

    if recent.empty:
        st.info(f"No filings with publish_date in the last {days_window} days.")
    else:
        with get_db(UNIVERSE_DB) as conn:
            tickers = pd.read_sql("SELECT isin, ticker, company_name FROM companies", conn)
        recent = recent.merge(tickers.rename(columns={"isin": "security_id"}), on="security_id", how="left")
        st.metric("Recent filings", f"{len(recent):,}")
        st.dataframe(
            recent[["publish_date", "ticker", "company_name", "fiscal_year", "fiscal_period", "statement_type"]]
            .sort_values("publish_date", ascending=False)
            .head(200),
            hide_index=True,
            use_container_width=True,
        )
