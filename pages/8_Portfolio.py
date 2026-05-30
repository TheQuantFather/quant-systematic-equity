"""
8_Portfolio.py — Portfolio optimisation results dashboard.

Run optimize_portfolio.py to generate results, then view here.
"""

import io
import json
import sqlite3
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    OUTPUT_DIR, PARAMS_FILE, MODELS_DB, RISK_DB,
    UNIVERSE_DB as UNIV_DB, BENCHMARK_DIR,
    FACTORS_REF, MODELS_REF,
    BARRA_SECTORS as _BARRA_SECTORS,
)
from utils import get_db, inject_css

# Build BARRA_GROUPS dynamically from factors_reference.csv.
# Layout: [market | sectors | styles | beta | fundamentals]  (K = 30).
_ref_csv  = pd.read_csv(str(FACTORS_REF))
_N_MARKET = 1
_N_SECTOR = len(_BARRA_SECTORS)
_N_STYLE  = len(_ref_csv[_ref_csv["barra_factor_type"] == "style"])
_N_BETA   = 1
_N_FUND   = len(_ref_csv[_ref_csv["barra_factor_type"] == "fundamental"])
_BARRA_GROUPS = {
    "Market":      0,
    "Sector":      slice(_N_MARKET, _N_MARKET + _N_SECTOR),
    "Style":       slice(_N_MARKET + _N_SECTOR,
                         _N_MARKET + _N_SECTOR + _N_STYLE),
    "Beta":        _N_MARKET + _N_SECTOR + _N_STYLE,
    "Fundamental": slice(_N_MARKET + _N_SECTOR + _N_STYLE + _N_BETA,
                         _N_MARKET + _N_SECTOR + _N_STYLE + _N_BETA + _N_FUND),
}

# Base model IDs and display names — read from models_reference.csv, not hardcoded.
_mref = pd.read_csv(MODELS_REF)
_mref["IsComposite"] = _mref["IsComposite"].astype(int)
BASE_MODELS = dict(zip(
    _mref.loc[_mref["IsComposite"] == 0, "ModelID"],
    _mref.loc[_mref["IsComposite"] == 0, "Model"],
))

st.set_page_config(page_title="Portfolio", layout="wide")
inject_css()
st.title("Portfolio Optimiser")


# ── Helpers ───────────────────────────────────────────────────────────────────

def list_strategies() -> list[str]:
    if not PARAMS_FILE.exists():
        return []
    xl = pd.ExcelFile(PARAMS_FILE)
    df = pd.read_excel(xl, sheet_name="Strategies", dtype=str)
    return list(df[df["active"].str.strip().str.upper() == "TRUE"]["strategy_id"].str.strip())


def load_latest(strategy_id: str) -> tuple[pd.DataFrame | None, dict | None]:
    results_path = OUTPUT_DIR / f"{strategy_id}_latest.csv"
    summary_path = OUTPUT_DIR / f"{strategy_id}_latest_summary.json"
    if not results_path.exists():
        return None, None
    df      = pd.read_csv(results_path)
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return df, summary


def run_optimizer(strategy_id: str) -> tuple[bool, pd.DataFrame | None, dict | None, str]:
    """
    Run optimization in-process (no subprocess). Returns (ok, df, summary, error_log).
    Results are also persisted to portfolio_output/ so they survive page refreshes.
    """
    from optimize_portfolio import (
        run_optimization, load_strategy_params, save_results,
    )
    try:
        strategies = load_strategy_params(strategy_id)
        if not strategies:
            return False, None, None, f"Strategy '{strategy_id}' not found in params file."
        results_df, summary = run_optimization(strategies[0])
        save_results(results_df, summary)
        return True, results_df, summary, ""
    except Exception as exc:
        return False, None, None, str(exc)


@st.cache_data(ttl=300)
def load_factor_scores(alpha_date: str) -> pd.DataFrame:
    if not MODELS_DB.exists():
        return pd.DataFrame()
    model_ids    = list(BASE_MODELS.keys())
    placeholders = ",".join("?" * len(model_ids))
    with get_db(MODELS_DB) as conn:
        df = pd.read_sql(
            f"SELECT security_id, model_id, model_value_z FROM models "
            f"WHERE data_date=? AND model_id IN ({placeholders})",
            conn, params=[alpha_date] + model_ids,
        )
    if df.empty:
        return pd.DataFrame()
    return df.pivot(index="security_id", columns="model_id", values="model_value_z")


def compute_factor_tilts(df: pd.DataFrame, scores_wide: pd.DataFrame,
                         has_benchmark: bool) -> pd.DataFrame:
    merged    = df.merge(scores_wide, left_on="isin", right_index=True, how="left")
    ref_label = "Benchmark" if has_benchmark else "Univ. Avg"
    rows = []
    for model_id, factor_name in BASE_MODELS.items():
        if model_id not in merged.columns:
            continue
        col      = merged[model_id].fillna(0)
        port_w   = merged["portfolio_weight"]
        port_avg = float((col * port_w).sum() / port_w.sum()) if port_w.sum() > 0 else 0.0
        if has_benchmark and merged["benchmark_weight"].sum() > 0:
            bm_w    = merged["benchmark_weight"]
            ref_avg = float((col * bm_w).sum() / bm_w.sum())
        else:
            ref_avg = float(col.mean())
        rows.append({
            "Factor":      factor_name,
            "Portfolio":   port_avg,
            ref_label:     ref_avg,
            "Active Tilt": port_avg - ref_avg,
            "ref_label":   ref_label,
        })
    return pd.DataFrame(rows).sort_values("Active Tilt", ascending=True)


@st.cache_data(ttl=300)
def load_risk_contributions(risk_date: str, isins: tuple,
                             weights: tuple) -> pd.DataFrame | None:
    """
    Compute per-stock risk contributions from the stored covariance matrix.

    Returns DataFrame with columns:
        isin, pct_of_risk, vol_contribution
    where:
        pct_of_risk      = w_i*(Σw)_i / (w'Σw)  — fraction of total variance
        vol_contribution = w_i*(Σw)_i / σ_p      — contribution to total vol
    """
    if not RISK_DB.exists():
        return None
    with get_db(RISK_DB) as conn:
        row = conn.execute(
            "SELECT matrix_blob, isin_list FROM covariance_matrix WHERE data_date=?",
            (risk_date,),
        ).fetchone()
    if row is None:
        return None

    cov        = np.load(io.BytesIO(zlib.decompress(row[0]))).astype(np.float64)
    isins_full = json.loads(row[1])
    isin_idx   = {isin: i for i, isin in enumerate(isins_full)}

    valid_isins   = [s for s in isins   if s in isin_idx]
    valid_weights = [weights[isins.index(s)] for s in valid_isins]
    if not valid_isins:
        return None

    idx   = [isin_idx[s] for s in valid_isins]
    Sigma = cov[np.ix_(idx, idx)]
    w     = np.array(valid_weights, dtype=np.float64)
    w    /= w.sum()

    sigma_p     = float(np.sqrt(w @ Sigma @ w))
    Sigma_w     = Sigma @ w
    vol_contrib = w * Sigma_w / sigma_p        # contribution to total vol (sums to sigma_p)
    pct_risk    = w * Sigma_w / (sigma_p ** 2) # fraction of total risk (sums to 1)

    return pd.DataFrame({
        "isin":             valid_isins,
        "pct_of_risk":      pct_risk * 100,
        "vol_contribution": vol_contrib * 100,
        "sigma_p":          sigma_p,
    })


@st.cache_data(ttl=300)
def load_risk_contributions_barra(barra_date: str, isins: tuple,
                                   weights: tuple) -> pd.DataFrame | None:
    """Compute per-stock risk contributions from Barra Σ = XFX' + Δ."""
    try:
        with get_db(RISK_DB) as conn:
            row = conn.execute(
                "SELECT factor_names, cov_blob FROM factor_covariance WHERE snapshot_date=?",
                (barra_date,),
            ).fetchone()
            if row is None:
                return None
            fnames = json.loads(row[0])
            K = len(fnames)
            F = np.frombuffer(zlib.decompress(row[1]), dtype=np.float32).reshape(K, K).astype(np.float64)
            x_rows = conn.execute(
                "SELECT security_id, factor_id, exposure FROM factor_exposures WHERE snapshot_date=?",
                (barra_date,),
            ).fetchall()
            d_rows = conn.execute(
                "SELECT security_id, idio_var FROM idiosyncratic_vars WHERE snapshot_date=?",
                (barra_date,),
            ).fetchall()
    except Exception:
        return None

    x_data: dict = {}
    for sec_id, fac_id, exp in x_rows:
        x_data.setdefault(sec_id, {})[fac_id] = float(exp)
    X_df = (
        pd.DataFrame.from_dict(x_data, orient="index")
        .reindex(columns=fnames, fill_value=0.0)
        .fillna(0.0)
    )
    delta_s = pd.Series({r[0]: float(r[1]) for r in d_rows})

    valid_isins   = [s for s in isins if s in X_df.index]
    valid_weights = [weights[isins.index(s)] for s in valid_isins]
    if not valid_isins:
        return None

    X_sub   = X_df.loc[valid_isins].values
    delta_v = delta_s.reindex(valid_isins, fill_value=0.04).values
    Sigma   = X_sub @ F @ X_sub.T + np.diag(delta_v)
    w       = np.array(valid_weights, dtype=np.float64)
    w      /= w.sum()

    sigma_p     = float(np.sqrt(w @ Sigma @ w))
    Sigma_w     = Sigma @ w
    pct_risk    = w * Sigma_w / (sigma_p ** 2)
    vol_contrib = w * Sigma_w / sigma_p

    return pd.DataFrame({
        "isin":             valid_isins,
        "pct_of_risk":      pct_risk * 100,
        "vol_contribution": vol_contrib * 100,
        "sigma_p":          sigma_p,
    })


@st.cache_data(ttl=300)
def load_barra_components(barra_date: str) -> tuple | None:
    """
    Return (F, fnames, X_df, delta_s) from risk.db (Barra tables) for a snapshot date.
    Shared across portfolio risk and active risk attribution.
    """
    try:
        with get_db(RISK_DB) as conn:
            row = conn.execute(
                "SELECT factor_names, cov_blob FROM factor_covariance WHERE snapshot_date=?",
                (barra_date,),
            ).fetchone()
            if row is None:
                return None
            fnames = json.loads(row[0])
            K      = len(fnames)
            F      = np.frombuffer(zlib.decompress(row[1]), dtype=np.float32).reshape(K, K).astype(np.float64)
            x_rows = conn.execute(
                "SELECT security_id, factor_id, exposure FROM factor_exposures WHERE snapshot_date=?",
                (barra_date,),
            ).fetchall()
            d_rows = conn.execute(
                "SELECT security_id, idio_var FROM idiosyncratic_vars WHERE snapshot_date=?",
                (barra_date,),
            ).fetchall()
    except Exception:
        return None
    x_data: dict = {}
    for sec_id, fac_id, exp in x_rows:
        x_data.setdefault(sec_id, {})[fac_id] = float(exp)
    X_df = (
        pd.DataFrame.from_dict(x_data, orient="index")
        .reindex(columns=fnames, fill_value=0.0)
        .fillna(0.0)
    )
    delta_s = pd.Series({r[0]: float(r[1]) for r in d_rows})
    return F, fnames, X_df, delta_s


@st.cache_data(ttl=300)
def load_lw_raw(risk_date: str) -> tuple | None:
    """Return (Sigma: N×N float64, isins: list[str]) from risk.db."""
    if not RISK_DB.exists():
        return None
    with get_db(RISK_DB) as conn:
        row = conn.execute(
            "SELECT matrix_blob, isin_list FROM covariance_matrix WHERE data_date=?",
            (risk_date,),
        ).fetchone()
    if row is None:
        return None
    Sigma = np.load(io.BytesIO(zlib.decompress(row[0]))).astype(np.float64)
    isins = json.loads(row[1])
    return Sigma, isins


@st.cache_data
def load_benchmark_isins(benchmark_file: str) -> pd.Series | None:
    """Return Series[isin → weight] by mapping benchmark tickers via universe.db."""
    path = BENCHMARK_DIR / benchmark_file
    if not path.exists():
        return None
    df = pd.read_csv(path, skiprows=2).dropna(subset=["Ticker"])
    df = df[df["Asset Class"].str.strip() == "Equity"].copy()
    df["weight"] = pd.to_numeric(df["Weight (%)"], errors="coerce") / 100.0
    df = df.dropna(subset=["weight"])
    df["weight"] /= df["weight"].sum()
    tickers = df["Ticker"].str.strip().tolist()
    placeholders = ",".join("?" * len(tickers))
    with get_db(UNIV_DB) as conn:
        rows = conn.execute(
            f"SELECT ticker, isin FROM companies WHERE ticker IN ({placeholders})", tickers
        ).fetchall()
    tick_to_isin = {r[0]: r[1] for r in rows}
    df["isin"] = df["Ticker"].str.strip().map(tick_to_isin)
    df = df.dropna(subset=["isin"])
    s = df.groupby("isin")["weight"].sum()
    return s / s.sum()   # renormalize after ticker→isin matching


def _get_benchmark_file(strategy_id: str) -> str | None:
    if not PARAMS_FILE.exists():
        return None
    xl  = pd.ExcelFile(PARAMS_FILE)
    sdf = pd.read_excel(xl, "Strategies", dtype=str)
    row = sdf[sdf["strategy_id"].str.strip() == strategy_id]
    if row.empty:
        return None
    val = str(row.iloc[0].get("benchmark_file", "") or "").strip()
    return val or None


def _compute_attribution(isins: list[str], weights: np.ndarray,
                         barra_comps: tuple | None,
                         lw_raw: tuple | None) -> pd.DataFrame | None:
    """
    Compute per-stock risk attribution. Weights may be signed (active weights).
    Returns DataFrame[isin, pct_of_risk, vol_contribution, sigma_p].
    sigma_p = sqrt(w' Σ w) — total or tracking error depending on weights passed.
    """
    if barra_comps is not None:
        F, fnames, X_df, delta_s = barra_comps
        valid = [(s, w) for s, w in zip(isins, weights) if s in X_df.index]
    elif lw_raw is not None:
        Sigma_full, isins_full = lw_raw
        isin_idx = {s: i for i, s in enumerate(isins_full)}
        valid = [(s, w) for s, w in zip(isins, weights) if s in isin_idx]
    else:
        return None
    if not valid:
        return None
    valid_isins, valid_weights = zip(*valid)
    w = np.array(valid_weights, dtype=np.float64)

    if barra_comps is not None:
        X_sub   = X_df.loc[list(valid_isins)].values
        delta_v = delta_s.reindex(list(valid_isins), fill_value=0.04).values
        Sigma   = X_sub @ F @ X_sub.T + np.diag(delta_v)
    else:
        idx   = [isin_idx[s] for s in valid_isins]
        Sigma = Sigma_full[np.ix_(idx, idx)]

    var_p = float(w @ Sigma @ w)
    if var_p <= 0:
        return None
    sigma_p     = float(np.sqrt(var_p))
    Sigma_w     = Sigma @ w
    pct_risk    = w * Sigma_w / var_p
    vol_contrib = w * Sigma_w / sigma_p
    return pd.DataFrame({
        "isin":             list(valid_isins),
        "pct_of_risk":      pct_risk * 100,
        "vol_contribution": vol_contrib * 100,
        "sigma_p":          sigma_p,
    })


def _barra_factor_te(x_active: np.ndarray, F: np.ndarray,
                      delta_v: np.ndarray, w_a: np.ndarray) -> dict:
    """
    Decompose active variance into factor-group contributions.
    x_active = X_sub.T @ w_a  (K,) active factor tilt.
    Returns {group_name: variance_contribution}.
    """
    groups: dict = {}
    for g_name, g_slice in _BARRA_GROUPS.items():
        if isinstance(g_slice, int):
            xi = x_active[[g_slice]]
            Fi = F[np.ix_([g_slice], [g_slice])]
        else:
            xi = x_active[g_slice]
            Fi = F[g_slice, g_slice]
        groups[g_name] = float(xi @ Fi @ xi)
    groups["Idiosyncratic"] = float(np.sum(delta_v * w_a ** 2))
    return groups


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.header("Strategy")

strategies = list_strategies()
if not strategies:
    st.warning("No strategies found. Run `python create_strategy_params.py` first.")
    st.stop()

selected = st.sidebar.selectbox("Select strategy", strategies)

if st.sidebar.button("▶  Run Optimisation", use_container_width=True, type="primary"):
    with st.spinner("Optimising…"):
        ok, _df, _summary, _log = run_optimizer(selected)
    if ok:
        st.session_state[f"port_{selected}"] = (_df, _summary)
        st.success("Optimisation complete.")
    else:
        st.error("Optimisation failed.")
        st.code(_log)

st.sidebar.markdown("---")
st.sidebar.caption("Re-run after changing `strategy_params.xlsx`.")

# ── Load results — session state first, then file fallback ───────────────────

if f"port_{selected}" in st.session_state:
    df, summary = st.session_state[f"port_{selected}"]
else:
    df, summary = load_latest(selected)

if df is None:
    st.info(f"No results yet for **{selected}**. Click **▶ Run Optimisation** in the sidebar.")
    st.stop()

# ── Summary metrics ───────────────────────────────────────────────────────────

objective  = summary.get("objective", "maximize_alpha")
is_alpha   = objective == "maximize_alpha"
is_sharpe  = objective == "maximize_sharpe"
is_min_var = objective == "minimize_variance"
has_bm     = df["benchmark_weight"].sum() > 0

st.subheader(summary.get("name", selected))
st.caption(
    f"Run: {summary.get('run_date', '—')}  |  "
    f"Status: {summary.get('status', '—')}  |  "
    f"Objective: {objective}"
)

c1, c2, c3, c4, c5 = st.columns(5)
if is_min_var:
    c1.metric("Expected Alpha", "—")
    c2.metric("Portfolio Vol",  f"{summary.get('portfolio_vol', 0):.2%}")
    c3.metric("Sharpe Ratio",   "—")
elif is_sharpe:
    c1.metric("Exp Return (α)", f"{summary.get('expected_alpha', 0):+.4f}")
    c2.metric("Portfolio Vol",  f"{summary.get('portfolio_vol', 0):.2%}")
    c3.metric("Sharpe Ratio",   f"{summary.get('sharpe_ratio', 0):.2f}")
else:
    c1.metric("Expected Alpha", f"{summary.get('expected_alpha', 0):+.4f}")
    c2.metric("Active Risk",    f"{summary.get('active_risk', 0):.2%}")
    c3.metric("Info Ratio",     f"{summary.get('info_ratio', 0):.2f}")
c4.metric("Positions",        summary.get("n_positions", "—"))
c5.metric("Benchmark Stocks", summary.get("n_benchmark", "—") if is_alpha else "—")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Weights", "Sector & Industry", "Factor Tilts", "Risk Attribution", "Raw Data"
])

# ── Tab 1: Weights ────────────────────────────────────────────────────────────

with tab1:
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Top 20 Overweights**")
        top = df[df["active_weight"] > 0].nlargest(20, "active_weight")
        fig = go.Figure(go.Bar(
            x=top["active_weight"] * 100, y=top["company_name"],
            orientation="h", marker_color="#2196F3",
            text=[f"{v:.2f}%" for v in top["active_weight"] * 100],
            textposition="outside",
            customdata=top["ticker"],
            hovertemplate="%{customdata}<br>Active: %{x:.2f}%<extra></extra>",
        ))
        fig.update_layout(height=500, margin=dict(l=10, r=60, t=10, b=10),
                          xaxis_title="Active Weight (%)",
                          yaxis=dict(autorange="reversed", automargin=True),
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("**Top 20 Underweights**")
        bot = df[df["active_weight"] < 0].nsmallest(20, "active_weight")
        fig = go.Figure(go.Bar(
            x=bot["active_weight"] * 100, y=bot["company_name"],
            orientation="h", marker_color="#F44336",
            text=[f"{v:.2f}%" for v in bot["active_weight"] * 100],
            textposition="outside",
            customdata=bot["ticker"],
            hovertemplate="%{customdata}<br>Active: %{x:.2f}%<extra></extra>",
        ))
        fig.update_layout(height=500, margin=dict(l=10, r=60, t=10, b=10),
                          xaxis_title="Active Weight (%)",
                          yaxis=dict(autorange="reversed", automargin=True),
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Active Weight Distribution**")
    fig = go.Figure(go.Histogram(
        x=df["active_weight"] * 100, nbinsx=60,
        marker_color="#1F4E79", opacity=0.8,
    ))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=30),
                      xaxis_title="Active Weight (%)", yaxis_title="# Stocks",
                      plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


# ── Tab 2: Sector & Industry ──────────────────────────────────────────────────

with tab2:
    sec = (
        df.groupby("gics_sector")
        .agg(portfolio_weight=("portfolio_weight", "sum"),
             benchmark_weight=("benchmark_weight", "sum"))
        .reset_index()
    )
    sec["active_weight"] = sec["portfolio_weight"] - sec["benchmark_weight"]
    sec = sec.sort_values("active_weight", ascending=True)

    ref_lbl = "Benchmark" if has_bm else "Universe avg"
    st.markdown(f"**Sector Exposure — Portfolio vs {ref_lbl}**")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=ref_lbl, x=sec["benchmark_weight"] * 100, y=sec["gics_sector"],
        orientation="h", marker_color="#90A4AE",
    ))
    fig.add_trace(go.Bar(
        name="Portfolio", x=sec["portfolio_weight"] * 100, y=sec["gics_sector"],
        orientation="h", marker_color="#1F4E79",
    ))
    fig.update_layout(barmode="group", height=420, margin=dict(l=10, r=10, t=10, b=30),
                      xaxis_title="Weight (%)", legend=dict(orientation="h", y=1.05),
                      plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Sector Active Weights**")
    colors = ["#F44336" if v < 0 else "#2196F3" for v in sec["active_weight"]]
    fig = go.Figure(go.Bar(
        x=sec["active_weight"] * 100, y=sec["gics_sector"],
        orientation="h", marker_color=colors,
        text=[f"{v:+.2f}%" for v in sec["active_weight"] * 100],
        textposition="outside",
    ))
    fig.update_layout(height=350, margin=dict(l=10, r=80, t=10, b=30),
                      xaxis_title="Active Weight (%)", plot_bgcolor="rgba(0,0,0,0)",
                      shapes=[dict(type="line", x0=0, x1=0, y0=-0.5, y1=len(sec)-0.5,
                                   line=dict(color="black", width=1))])
    st.plotly_chart(fig, use_container_width=True)

    ind = (
        df.groupby("industry")
        .agg(portfolio_weight=("portfolio_weight", "sum"),
             benchmark_weight=("benchmark_weight", "sum"))
        .reset_index()
    )
    ind["active_weight"] = ind["portfolio_weight"] - ind["benchmark_weight"]
    ind_top = pd.concat([
        ind.nlargest(10, "active_weight"),
        ind.nsmallest(10, "active_weight"),
    ]).drop_duplicates().sort_values("active_weight", ascending=True)

    st.markdown("**Industry Active Weights (top 10 over/under)**")
    colors = ["#F44336" if v < 0 else "#2196F3" for v in ind_top["active_weight"]]
    fig = go.Figure(go.Bar(
        x=ind_top["active_weight"] * 100, y=ind_top["industry"],
        orientation="h", marker_color=colors,
        text=[f"{v:+.2f}%" for v in ind_top["active_weight"] * 100],
        textposition="outside",
    ))
    fig.update_layout(height=500, margin=dict(l=10, r=80, t=10, b=30),
                      xaxis_title="Active Weight (%)", plot_bgcolor="rgba(0,0,0,0)",
                      shapes=[dict(type="line", x0=0, x1=0, y0=-0.5, y1=len(ind_top)-0.5,
                                   line=dict(color="black", width=1))])
    st.plotly_chart(fig, use_container_width=True)


# ── Tab 3: Factor Tilts ───────────────────────────────────────────────────────

with tab3:
    alpha_date  = summary.get("alpha_date", "2026-04-01")
    scores_wide = load_factor_scores(alpha_date)

    if scores_wide.empty:
        st.info("Factor scores not available — run `create_models.py` first.")
    else:
        tilts   = compute_factor_tilts(df, scores_wide, has_bm)
        ref_lbl = tilts["ref_label"].iloc[0] if not tilts.empty else ("Benchmark" if has_bm else "Univ. Avg")

        st.markdown(f"**Active Factor Tilts — Portfolio vs {ref_lbl}**")
        st.caption("Positive = portfolio overweights that factor vs reference.")

        colors = ["#2196F3" if v >= 0 else "#F44336" for v in tilts["Active Tilt"]]
        fig = go.Figure(go.Bar(
            x=tilts["Active Tilt"], y=tilts["Factor"],
            orientation="h", marker_color=colors,
            text=[f"{v:+.3f}" for v in tilts["Active Tilt"]],
            textposition="outside",
        ))
        fig.add_vline(x=0, line_color="black", line_width=1)
        fig.update_layout(height=340, margin=dict(l=10, r=80, t=10, b=30),
                          xaxis_title="Active Tilt (z-score units)",
                          plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"**Factor Scores — Portfolio vs {ref_lbl}**")
        fig = go.Figure()
        fig.add_trace(go.Bar(name=ref_lbl, x=tilts["Factor"], y=tilts[ref_lbl],
                             marker_color="#90A4AE"))
        fig.add_trace(go.Bar(name="Portfolio", x=tilts["Factor"], y=tilts["Portfolio"],
                             marker_color="#1F4E79"))
        fig.update_layout(barmode="group", height=320, margin=dict(l=10, r=10, t=10, b=30),
                          yaxis_title="Weighted avg z-score",
                          legend=dict(orientation="h", y=1.05), plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Composite alpha distribution
    st.markdown("**Composite Alpha Score Distribution**")
    port_alpha = float(
        (df["alpha_score"] * df["portfolio_weight"]).sum() / df["portfolio_weight"].sum()
    ) if df["portfolio_weight"].sum() > 0 else 0.0
    bm_alpha  = float(
        (df["alpha_score"] * df["benchmark_weight"]).sum() / df["benchmark_weight"].sum()
    ) if has_bm else float(df["alpha_score"].mean())
    bm_label  = "Benchmark" if has_bm else "Universe avg"

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["alpha_score"], nbinsx=50, name="All stocks",
        marker_color="#CFD8DC", opacity=0.6, histnorm="probability density",
    ))
    fig.add_vline(x=port_alpha, line_color="#1F4E79", line_width=2,
                  annotation_text=f"Portfolio: {port_alpha:+.3f}",
                  annotation_position="top right")
    fig.add_vline(x=bm_alpha, line_color="#90A4AE", line_width=2, line_dash="dash",
                  annotation_text=f"{bm_label}: {bm_alpha:+.3f}",
                  annotation_position="top left")
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=30),
                      xaxis_title="Alpha Score (z-score)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    df_contrib = df.copy()
    df_contrib["alpha_contribution"] = df_contrib["alpha_score"] * df_contrib["portfolio_weight"]
    top_contrib = df_contrib.nlargest(15, "alpha_contribution")

    st.markdown("**Top 15 Alpha Contributors**")
    fig = go.Figure(go.Bar(
        x=top_contrib["alpha_contribution"] * 100, y=top_contrib["company_name"],
        orientation="h", marker_color="#1F4E79",
        text=[f"{v:.3f}%" for v in top_contrib["alpha_contribution"] * 100],
        textposition="outside",
        customdata=top_contrib["ticker"],
        hovertemplate="%{customdata}<br>Alpha contribution: %{x:.3f}%<extra></extra>",
    ))
    fig.update_layout(height=420, margin=dict(l=10, r=80, t=10, b=30),
                      xaxis_title="Alpha Contribution (weight × score, %)",
                      yaxis=dict(autorange="reversed", automargin=True),
                      plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


# ── Tab 4: Risk Attribution ───────────────────────────────────────────────────

with tab4:
    risk_date  = summary.get("risk_date", "2026-04-01")
    barra_date = summary.get("barra_date")

    # Load risk components once — reused for both portfolio and active attribution
    if barra_date:
        barra_comps = load_barra_components(barra_date)
        lw_raw      = None
        risk_model_label = f"Barra ({barra_date})"
    else:
        barra_comps = None
        lw_raw      = load_lw_raw(risk_date)
        risk_model_label = f"Ledoit-Wolf ({risk_date})"

    if barra_comps is None and lw_raw is None:
        st.info("Risk attribution requires risk.db. Run the relevant pipeline first.")
    else:
        # ── Portfolio weight vector (held stocks only, normalised) ───────────
        held         = df[df["portfolio_weight"] > 1e-5].copy()
        port_isins   = held["isin"].tolist()
        port_weights = held["portfolio_weight"].values / held["portfolio_weight"].sum()

        # ── Active weight vector — only meaningful for benchmark-aware strategies ─
        # For maximize_sharpe/minimize_variance there is no benchmark, so
        # active_weight == portfolio_weight and "TE" would just equal portfolio vol.
        has_benchmark = is_alpha and (df["benchmark_weight"] > 1e-6).any()
        if has_benchmark:
            active_isins   = df["isin"].tolist()
            active_weights = df["active_weight"].values
        else:
            active_isins   = port_isins
            active_weights = port_weights

        # ── Compute attributions ──────────────────────────────────────────────
        rc_port   = _compute_attribution(port_isins, port_weights, barra_comps, lw_raw)
        rc_active = _compute_attribution(active_isins, active_weights, barra_comps, lw_raw) \
                    if has_benchmark else None

        sub_port, sub_active = st.tabs(["Portfolio Risk", "Active Risk (TE)"])

        # ════════════════════════════════════════════════════════════════════
        # Portfolio Risk sub-tab
        # ════════════════════════════════════════════════════════════════════
        with sub_port:
            if rc_port is None:
                st.info("Could not compute portfolio risk attribution.")
            else:
                sigma_p = float(rc_port["sigma_p"].iloc[0])
                st.caption(
                    f"Total portfolio volatility: **{sigma_p:.2%}**  |  "
                    f"Risk model: {risk_model_label}  |  "
                    f"Stocks: {len(rc_port)}"
                )
                rc = rc_port.merge(
                    df[["isin", "ticker", "company_name", "gics_sector", "industry",
                        "portfolio_weight", "active_weight", "benchmark_weight"]],
                    on="isin", how="left"
                )

                st.markdown("**Top 20 Stocks by Risk Contribution**")
                top_rc = rc.nlargest(20, "pct_of_risk")
                fig = go.Figure(go.Bar(
                    x=top_rc["pct_of_risk"], y=top_rc["company_name"],
                    orientation="h", marker_color="#1F4E79",
                    text=[f"{v:.1f}%" for v in top_rc["pct_of_risk"]],
                    textposition="outside",
                    customdata=top_rc["ticker"],
                    hovertemplate="%{customdata}<br>Risk contribution: %{x:.1f}%<extra></extra>",
                ))
                fig.update_layout(height=500, margin=dict(l=10, r=70, t=10, b=10),
                                  xaxis_title="% of Total Portfolio Risk",
                                  yaxis=dict(autorange="reversed", automargin=True),
                                  plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("**Weight vs Risk Contribution**")
                st.caption("Stocks above the diagonal contribute disproportionately to risk relative to their weight.")
                fig = go.Figure(go.Scatter(
                    x=rc["portfolio_weight"] * 100, y=rc["pct_of_risk"],
                    mode="markers+text", text=rc["ticker"], textposition="top center",
                    customdata=rc["company_name"],
                    hovertemplate="%{customdata} (%{text})<br>Weight: %{x:.2f}%<br>Risk: %{y:.1f}%<extra></extra>",
                    marker=dict(size=7, color="#1F4E79", opacity=0.7),
                ))
                max_val = max(rc["portfolio_weight"].max() * 100, rc["pct_of_risk"].max()) * 1.05
                fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                              line=dict(color="#90A4AE", width=1, dash="dash"))
                fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=40),
                                  xaxis_title="Portfolio Weight (%)",
                                  yaxis_title="% of Total Risk", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

                col_sec, col_ind = st.columns(2)
                with col_sec:
                    st.markdown("**Risk by Sector**")
                    sec_rc = (
                        rc.groupby("gics_sector")
                        .agg(pct_of_risk=("pct_of_risk", "sum"),
                             portfolio_weight=("portfolio_weight", "sum"))
                        .reset_index().sort_values("pct_of_risk", ascending=True)
                    )
                    sec_rc["risk_per_weight"] = sec_rc["pct_of_risk"] / (sec_rc["portfolio_weight"] * 100)
                    colors = ["#F44336" if r > 1.15 else "#FFA726" if r > 1.0 else "#2196F3"
                              for r in sec_rc["risk_per_weight"]]
                    fig = go.Figure(go.Bar(
                        x=sec_rc["pct_of_risk"], y=sec_rc["gics_sector"],
                        orientation="h", marker_color=colors,
                        text=[f"{v:.1f}%" for v in sec_rc["pct_of_risk"]],
                        textposition="outside",
                    ))
                    fig.update_layout(height=380, margin=dict(l=10, r=60, t=10, b=30),
                                      xaxis_title="% of Total Risk", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Red = risk contribution > 15% above weight share; orange = slightly elevated.")

                with col_ind:
                    st.markdown("**Risk by Industry (top 15)**")
                    ind_rc = (
                        rc.groupby("industry")
                        .agg(pct_of_risk=("pct_of_risk", "sum"),
                             portfolio_weight=("portfolio_weight", "sum"))
                        .reset_index().nlargest(15, "pct_of_risk")
                        .sort_values("pct_of_risk", ascending=True)
                    )
                    fig = go.Figure(go.Bar(
                        x=ind_rc["pct_of_risk"], y=ind_rc["industry"],
                        orientation="h", marker_color="#1F4E79",
                        text=[f"{v:.1f}%" for v in ind_rc["pct_of_risk"]],
                        textposition="outside",
                    ))
                    fig.update_layout(height=380, margin=dict(l=10, r=60, t=10, b=30),
                                      xaxis_title="% of Total Risk", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("**Sector Risk vs Weight Summary**")
                tbl = sec_rc[["gics_sector", "portfolio_weight", "pct_of_risk", "risk_per_weight"]].copy()
                tbl["portfolio_weight"] = (tbl["portfolio_weight"] * 100).round(2)
                tbl["pct_of_risk"]      = tbl["pct_of_risk"].round(2)
                tbl["risk_per_weight"]  = tbl["risk_per_weight"].round(2)
                tbl.columns = ["Sector", "Weight %", "Risk %", "Risk/Weight"]
                st.dataframe(tbl.sort_values("Risk %", ascending=False),
                             use_container_width=True, hide_index=True)

        # ════════════════════════════════════════════════════════════════════
        # Active Risk (TE) sub-tab
        # ════════════════════════════════════════════════════════════════════
        with sub_active:
            if not has_benchmark:
                st.info(
                    f"Tracking error is not applicable for **{objective}** strategies "
                    "(no benchmark). See the Portfolio Risk tab for absolute volatility."
                )
            elif rc_active is None:
                st.info("Could not compute active risk attribution.")
            else:
                # Use optimizer's TE — active weights come directly from the CSV
                # (pre-computed with the optimizer's normalization) so they match exactly.
                te_computed = float(rc_active["sigma_p"].iloc[0])
                te          = summary.get("active_risk") or te_computed

                has_bm_underweights = (df["benchmark_weight"] > 1e-6).any()
                bm_note = (
                    "Benchmark stocks not held included as underweights  |  "
                    if has_bm_underweights else
                    "Benchmark underweights not available — held positions only  |  "
                )
                st.caption(
                    f"Tracking error: **{te:.2%}**  |  "
                    f"Risk model: {risk_model_label}  |  "
                    + bm_note +
                    f"Stocks: {len(rc_active)}"
                )

                # All investable stocks (held + underweights) are already in df
                rc_a = rc_active.merge(
                    df[["isin", "ticker", "company_name", "gics_sector", "industry",
                        "active_weight", "portfolio_weight"]],
                    on="isin", how="left"
                )
                rc_a["active_weight_pct"] = rc_a["active_weight"] * 100
                rc_a["is_overweight"]     = rc_a["active_weight"] >= 0

                # ── Barra: Factor Group TE Decomposition ─────────────────────
                if barra_comps is not None:
                    F, fnames, X_df, delta_s = barra_comps
                    valid_a = [s for s in active_isins if s in X_df.index]
                    w_a_valid = np.array([
                        active_weights[active_isins.index(s)] for s in valid_a
                    ])
                    X_sub_a = X_df.loc[valid_a].values
                    delta_v_a = delta_s.reindex(valid_a, fill_value=0.04).values
                    x_active_tilt = X_sub_a.T @ w_a_valid
                    grp_te = _barra_factor_te(x_active_tilt, F, delta_v_a, w_a_valid)
                    total_te_var = sum(grp_te.values())

                    st.subheader("Factor Group Contribution to Tracking Error")
                    grp_df = pd.DataFrame([
                        {"Group": g, "TE Variance": v,
                         "TE Contribution (%)": round(v / max(total_te_var, 1e-12) * 100, 1),
                         "Ann. TE (bps)": round(np.sqrt(max(v, 0)) * 10000, 1)}
                        for g, v in grp_te.items()
                    ]).sort_values("TE Contribution (%)", ascending=False)

                    col_pie, col_tbl = st.columns([1, 1])
                    with col_pie:
                        fig_pie = px.pie(
                            grp_df, values="TE Contribution (%)", names="Group",
                            title="TE Decomposition by Factor Group",
                            color_discrete_map={
                                "Sector": "#4C78A8", "Style": "#F58518",
                                "Beta": "#E45756", "Fundamental": "#72B7B2",
                                "Idiosyncratic": "#B279A2",
                            },
                        )
                        fig_pie.update_layout(height=320)
                        st.plotly_chart(fig_pie, use_container_width=True)
                    with col_tbl:
                        st.dataframe(
                            grp_df[["Group", "TE Contribution (%)", "Ann. TE (bps)"]],
                            use_container_width=True, hide_index=True,
                        )

                # ── Top 20 TE contributors ───────────────────────────────────
                st.markdown("**Top 20 Stocks by |TE Contribution|**")
                st.caption("Blue = overweight (adds TE), orange = underweight (offsets TE).")
                top_te = rc_a.reindex(rc_a["pct_of_risk"].abs().nlargest(20).index)
                colors_te = ["#1F4E79" if ow else "#F28E2B" for ow in top_te["is_overweight"]]
                fig = go.Figure(go.Bar(
                    x=top_te["pct_of_risk"],
                    y=top_te["company_name"].fillna(top_te["ticker"]),
                    orientation="h", marker_color=colors_te,
                    text=[f"{v:+.1f}%" for v in top_te["pct_of_risk"]],
                    textposition="outside",
                    customdata=top_te["ticker"].fillna(top_te["isin"]),
                    hovertemplate="%{customdata}<br>TE contribution: %{x:+.1f}%<extra></extra>",
                ))
                fig.update_layout(height=500, margin=dict(l=10, r=70, t=10, b=10),
                                  xaxis_title="% of Total TE Variance",
                                  yaxis=dict(autorange="reversed", automargin=True),
                                  plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

                # ── Active weight vs TE contribution scatter ─────────────────
                st.markdown("**Active Weight vs TE Contribution**")
                fig = go.Figure(go.Scatter(
                    x=rc_a["active_weight_pct"], y=rc_a["pct_of_risk"],
                    mode="markers", text=rc_a["ticker"].fillna(rc_a["isin"]),
                    customdata=rc_a["company_name"].fillna(rc_a["ticker"]),
                    hovertemplate="%{customdata} (%{text})<br>Active: %{x:.2f}%<br>TE contribution: %{y:.2f}%<extra></extra>",
                    marker=dict(
                        size=7, opacity=0.7,
                        color=rc_a["active_weight_pct"],
                        colorscale="RdBu", cmid=0,
                        colorbar=dict(title="Active wt %"),
                    ),
                ))
                fig.add_hline(y=0, line_color="#90A4AE", line_dash="dash", line_width=1)
                fig.add_vline(x=0, line_color="#90A4AE", line_dash="dash", line_width=1)
                fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=40),
                                  xaxis_title="Active Weight (%)",
                                  yaxis_title="% of Total TE Variance", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

                # ── Sector TE attribution ─────────────────────────────────────
                st.markdown("**Sector TE Attribution**")
                sec_te = (
                    rc_a.dropna(subset=["gics_sector"])
                    .groupby("gics_sector")
                    .agg(te_pct=("pct_of_risk", "sum"),
                         active_w=("active_weight", "sum"))
                    .reset_index().sort_values("te_pct")
                )
                colors_sec = ["#1F4E79" if v >= 0 else "#F28E2B" for v in sec_te["te_pct"]]
                fig = go.Figure(go.Bar(
                    x=sec_te["te_pct"], y=sec_te["gics_sector"],
                    orientation="h", marker_color=colors_sec,
                    text=[f"{v:+.1f}%" for v in sec_te["te_pct"]],
                    textposition="outside",
                ))
                fig.update_layout(height=380, margin=dict(l=10, r=60, t=10, b=30),
                                  xaxis_title="% of Total TE Variance", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("**Sector TE Summary**")
                sec_te_tbl = sec_te[["gics_sector", "active_w", "te_pct"]].copy()
                sec_te_tbl["active_w"] = (sec_te_tbl["active_w"] * 100).round(2)
                sec_te_tbl["te_pct"]   = sec_te_tbl["te_pct"].round(2)
                sec_te_tbl.columns     = ["Sector", "Active Weight %", "TE Contribution %"]
                st.dataframe(sec_te_tbl.sort_values("TE Contribution %", ascending=False),
                             use_container_width=False, hide_index=True, width=500)


# ── Tab 5: Raw Data ───────────────────────────────────────────────────────────

with tab5:
    st.markdown("**Full Portfolio Holdings**")

    display_df = df[[
        "ticker", "company_name", "gics_sector", "industry",
        "benchmark_weight", "portfolio_weight", "active_weight", "alpha_score",
    ]].copy()
    display_df["benchmark_weight"] = (display_df["benchmark_weight"] * 100).round(3)
    display_df["portfolio_weight"] = (display_df["portfolio_weight"] * 100).round(3)
    display_df["active_weight"]    = (display_df["active_weight"]    * 100).round(3)
    display_df.columns = [
        "Ticker", "Name", "Sector", "Industry",
        "BM Weight %", "Port Weight %", "Active Weight %", "Alpha Score",
    ]

    filter_col, _ = st.columns([2, 5])
    sector_filter = filter_col.multiselect(
        "Filter by sector", sorted(display_df["Sector"].unique()), default=[]
    )
    if sector_filter:
        display_df = display_df[display_df["Sector"].isin(sector_filter)]

    st.dataframe(
        display_df.sort_values("Active Weight %", ascending=False),
        use_container_width=True, height=500,
    )
    csv = display_df.to_csv(index=False).encode()
    st.download_button("Download CSV", csv, f"{selected}_portfolio.csv", "text/csv")
