"""
6_Backtester.py — Factor backtest and walk-forward optimised backtest.

Tab 1 — Factor Backtest  : rank stocks by model z-score, hold equal-weight top-N.
Tab 2 — Optimised Backtest: CVXPY walk-forward with quarterly rebalancing,
         one-way turnover constraint, Barra risk model, S&P 500 universe,
         and per-trade EUR transaction costs.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
import db
from config import MODELS_DB, PARAMS_FILE, RETURNS_DB, RISK_DB, UNIVERSE_DB
from utils import get_db, inject_css

st.set_page_config(page_title="Backtester", layout="wide")
inject_css()
st.title("Factor Backtester")

RISK_FREE   = 0.04  # annualised, used for Sharpe
N_QUINTILES = 5
TC_EUR      = 2.0   # €2 per trade (DeGiro US stocks)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_all_model_scores() -> pd.DataFrame:
    with get_db(MODELS_DB) as conn:
        df = pd.read_sql(
            "SELECT data_date, model_id, security_id, model_value_z FROM models", conn
        )
    df["model_value_z"] = pd.to_numeric(df["model_value_z"], errors="coerce")
    return df


@st.cache_data
def load_returns_matrix(min_isin_coverage: int = 200) -> pd.DataFrame:
    with get_db(RETURNS_DB) as conn:
        df = pd.read_sql(
            "SELECT isin, date, total_return FROM returns WHERE total_return IS NOT NULL", conn
        )
    df["date"] = pd.to_datetime(df["date"])
    matrix = df.pivot_table(index="date", columns="isin", values="total_return").sort_index()
    matrix.columns.name = None
    coverage   = matrix.notna().sum(axis=1)
    last_valid = coverage[coverage >= min_isin_coverage].index.max()
    if pd.notna(last_valid):
        matrix = matrix.loc[:last_valid]
    return matrix


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def perf_metrics(ret: pd.Series) -> dict:
    ret = ret.dropna()
    if len(ret) < 10:
        return {}
    total   = (1 + ret).prod() - 1
    n_years = len(ret) / 252
    ann_ret = (1 + total) ** (1 / max(n_years, 1e-6)) - 1
    ann_vol = ret.std() * 252 ** 0.5
    sharpe  = (ann_ret - RISK_FREE) / ann_vol if ann_vol > 0 else np.nan
    cum     = (1 + ret).cumprod()
    max_dd  = (cum / cum.cummax() - 1).min()
    return {
        "Total return":    f"{total:+.1%}",
        "Ann. return":     f"{ann_ret:+.1%}",
        "Ann. volatility": f"{ann_vol:.1%}",
        "Sharpe ratio":    f"{sharpe:.2f}",
        "Max drawdown":    f"{max_dd:.1%}",
        "Daily win rate":  f"{(ret > 0).mean():.1%}",
    }


def active_metrics(ret: pd.Series, bench: pd.Series, turnover_pct: float | None) -> dict:
    common = ret.index.intersection(bench.index)
    if len(common) < 63:
        return {}
    r = ret.loc[common].dropna()
    b = bench.loc[common].dropna()
    active  = r - b
    ann_act = active.mean() * 252
    te      = active.std() * np.sqrt(252)
    ir      = ann_act / te if te > 0 else np.nan
    beta    = r.cov(b) / b.var() if b.var() > 0 else np.nan
    out = {
        "Active return (ann.)": f"{ann_act:+.1%}",
        "Tracking error":       f"{te:.1%}",
        "Information ratio":    f"{ir:.2f}" if pd.notna(ir) else "—",
        "Beta (vs benchmark)":  f"{beta:.2f}" if pd.notna(beta) else "—",
    }
    if turnover_pct is not None:
        out["Avg rebal. turnover"] = f"{turnover_pct:.0f}%"
    return out


# ---------------------------------------------------------------------------
# Chart helpers  (shared by both tabs)
# ---------------------------------------------------------------------------

def _cum_return_chart(
    traces: list[dict],
    title: str = "Cumulative return (base = 1.0)",
    height: int = 420,
) -> go.Figure:
    """
    Build a cumulative-return line chart.
    Each trace dict: {series (raw returns), name, color, width=2, dash="solid"}.
    """
    fig = go.Figure()
    for t in traces:
        cum = (1 + t["series"]).cumprod()
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values, name=t["name"],
            line=dict(color=t["color"], width=t.get("width", 2), dash=t.get("dash", "solid")),
        ))
    fig.update_layout(
        title=title, height=height, yaxis_title="Portfolio value",
        hovermode="x unified", legend=dict(orientation="h", y=-0.15),
        margin=dict(l=0, r=0, t=40, b=10),
    )
    return fig


def _drawdown_chart(
    traces: list[dict],
    height: int = 220,
) -> go.Figure:
    """
    Build a drawdown chart.
    Each trace dict: {series (raw returns), name, color, dash="solid", fill_color=None}.
    """
    fig = go.Figure()
    for t in traces:
        cum  = (1 + t["series"]).cumprod()
        dd   = cum / cum.cummax() - 1
        kw: dict = {}
        if t.get("fill_color"):
            kw = {"fill": "tozeroy", "fillcolor": t["fill_color"]}
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values, name=t["name"],
            line=dict(color=t["color"], width=t.get("width", 1.5), dash=t.get("dash", "solid")),
            **kw,
        ))
    fig.update_layout(
        title="Drawdown", height=height,
        yaxis_tickformat=".0%", hovermode="x unified",
        legend=dict(orientation="h", y=-0.3),
        margin=dict(l=0, r=0, t=40, b=10),
    )
    return fig


def _annual_bar_chart(
    traces: list[dict],
    height: int = 320,
) -> go.Figure:
    """
    Build a grouped annual-returns bar chart.
    Each trace dict: {series (raw returns), name, color}.
    """
    fig = go.Figure()
    for t in traces:
        ann = t["series"].resample("YE").apply(lambda x: (1 + x).prod() - 1)
        fig.add_trace(go.Bar(
            x=ann.index.year.astype(str), y=ann.values,
            name=t["name"], marker_color=t["color"],
            text=[f"{v:+.1%}" if pd.notna(v) else "" for v in ann.values],
            textposition="outside",
        ))
    fig.update_layout(
        barmode="group", height=height, yaxis_tickformat=".0%",
        margin=dict(l=0, r=0, t=20, b=20),
    )
    return fig


def _metrics_table(metrics: dict) -> None:
    st.dataframe(
        pd.DataFrame(metrics.items(), columns=["Metric", "Value"]),
        hide_index=True, use_container_width=True,
    )


def _sector_chart(df: pd.DataFrame, title: str, color: str) -> go.Figure:
    counts = (
        df.groupby("Sector").size().reset_index(name="Count")
        .sort_values("Count", ascending=True)
    )
    fig = go.Figure(go.Bar(
        x=counts["Count"], y=counts["Sector"],
        orientation="h", marker_color=color,
        text=counts["Count"], textposition="outside",
    ))
    fig.update_layout(
        title=title, height=max(200, len(counts) * 30 + 60),
        margin=dict(l=0, r=40, t=40, b=10),
        xaxis_title="# stocks", yaxis_title="",
        xaxis=dict(showgrid=False),
    )
    return fig


# ---------------------------------------------------------------------------
# Factor backtest engine
# ---------------------------------------------------------------------------

def run_backtest(
    model_id, n_long, include_short, n_short, sel_sectors
) -> tuple[pd.Series | None, pd.Series | None, pd.Series | None, list, str | None]:
    if not RETURNS_DB.exists():
        return None, None, None, [], "returns.db not found — run `create_returns.py --update`."
    if not MODELS_DB.exists():
        return None, None, None, [], "models.db not found — run `create_models.py`."

    scores     = load_all_model_scores()
    ret_matrix = load_returns_matrix()
    ticker_map = db.get_ticker_map()

    uni = db.get_universe()[["security_id", "sector", "company_name"]].copy()
    uni["security_id"] = uni["security_id"].astype(str)
    sector_map = dict(zip(uni["security_id"], uni["sector"]))
    name_map   = dict(zip(uni["security_id"], uni["company_name"]))

    model_df = scores[scores["model_id"] == model_id].copy().dropna(subset=["model_value_z"])
    if sel_sectors:
        valid = set(uni[uni["sector"].isin(sel_sectors)]["security_id"])
        model_df = model_df[model_df["security_id"].isin(valid)]

    snapshot_dates = sorted(model_df["data_date"].unique())
    if len(snapshot_dates) < 2:
        return None, None, None, [], "Need at least 2 snapshot dates for a backtest."

    trading_index = ret_matrix.index

    def next_td(d_str: str):
        pos = trading_index.searchsorted(pd.Timestamp(d_str))
        return trading_index[pos] if pos < len(trading_index) else None

    def holdings_df(isins: list, score_lkp: dict, price_cols: set) -> pd.DataFrame:
        return pd.DataFrame([{
            "Rank":       rank,
            "Ticker":     ticker_map.get(isin, isin),
            "Company":    name_map.get(isin, ""),
            "Sector":     sector_map.get(isin, "Unknown"),
            "Score":      round(score_lkp.get(isin, np.nan), 3),
            "Price data": "✓" if isin in price_cols else "—",
        } for rank, isin in enumerate(isins, 1)])

    long_parts, bench_parts, short_parts, holdings_log = [], [], [], []

    for i, snap in enumerate(snapshot_dates):
        next_snap = (
            snapshot_dates[i + 1] if i + 1 < len(snapshot_dates)
            else trading_index[-1].strftime("%Y-%m-%d")
        )
        t_start = next_td(snap)
        t_end   = next_td(next_snap)
        if t_start is None or t_end is None or t_start >= t_end:
            continue

        snap_df     = model_df[model_df["data_date"] == snap].dropna(subset=["model_value_z"])
        score_lkp   = dict(zip(snap_df["security_id"], snap_df["model_value_z"]))
        long_isins  = snap_df.nlargest(n_long, "model_value_z")["security_id"].tolist()
        short_isins = (
            snap_df.nsmallest(n_short, "model_value_z")["security_id"].tolist()
            if include_short else []
        )
        period     = ret_matrix.loc[(ret_matrix.index >= t_start) & (ret_matrix.index < t_end)]
        price_cols = set(period.columns)

        def ew(isins):
            cols = [s for s in isins if s in price_cols]
            return period[cols].mean(axis=1) if cols else pd.Series(0.0, index=period.index)

        long_parts.append(ew(long_isins))
        bench_parts.append(ew(snap_df["security_id"].tolist()))
        if include_short:
            short_parts.append(ew(short_isins))

        holdings_log.append({
            "label":       f"{snap[:10]}  →  {next_snap[:10]}",
            "long":        holdings_df(long_isins,  score_lkp, price_cols),
            "short":       holdings_df(short_isins, score_lkp, price_cols) if include_short else None,
            "long_isins":  long_isins,
            "short_isins": short_isins,
        })

    if not long_parts:
        return None, None, None, [], "No overlapping price data found for this model and date range."

    return (
        pd.concat(long_parts).sort_index(),
        pd.concat(short_parts).sort_index() if include_short and short_parts else None,
        pd.concat(bench_parts).sort_index(),
        holdings_log,
        None,
    )


# ---------------------------------------------------------------------------
# Quintile analysis
# ---------------------------------------------------------------------------

def run_quintile_analysis(model_id: str, sel_sectors: list) -> list[pd.Series]:
    scores     = load_all_model_scores()
    ret_matrix = load_returns_matrix()
    model_df   = scores[scores["model_id"] == model_id].dropna(subset=["model_value_z"]).copy()

    if sel_sectors:
        uni = db.get_universe()[["security_id", "sector"]].copy()
        uni["security_id"] = uni["security_id"].astype(str)
        valid    = set(uni[uni["sector"].isin(sel_sectors)]["security_id"])
        model_df = model_df[model_df["security_id"].isin(valid)]

    snapshot_dates = sorted(model_df["data_date"].unique())
    trading_index  = ret_matrix.index

    def next_td(d):
        pos = trading_index.searchsorted(pd.Timestamp(d))
        return trading_index[pos] if pos < len(trading_index) else None

    q_parts = [[] for _ in range(N_QUINTILES)]
    for i, snap in enumerate(snapshot_dates):
        next_snap = (
            snapshot_dates[i + 1] if i + 1 < len(snapshot_dates)
            else trading_index[-1].strftime("%Y-%m-%d")
        )
        t_start = next_td(snap)
        t_end   = next_td(next_snap)
        if t_start is None or t_end is None or t_start >= t_end:
            continue

        snap_df = model_df[model_df["data_date"] == snap].sort_values(
            "model_value_z", ascending=False
        ).reset_index(drop=True)
        n      = len(snap_df)
        period = ret_matrix.loc[(ret_matrix.index >= t_start) & (ret_matrix.index < t_end)]
        for q in range(N_QUINTILES):
            isins = snap_df.iloc[int(q * n / N_QUINTILES):int((q + 1) * n / N_QUINTILES)]["security_id"].tolist()
            cols  = [s for s in isins if s in period.columns]
            q_parts[q].append(
                period[cols].mean(axis=1) if cols else pd.Series(0.0, index=period.index)
            )

    return [pd.concat(parts).sort_index() if parts else pd.Series(dtype=float) for parts in q_parts]


# ---------------------------------------------------------------------------
# Factor backtest helpers
# ---------------------------------------------------------------------------

def _compute_turnover(holdings_log: list) -> list[dict]:
    rows, prev = [], set()
    for p in holdings_log:
        curr = set(p.get("long_isins", []))
        if not curr:
            continue
        to = len(curr - prev) / len(curr) * 100 if prev else 100.0
        rows.append({"Period": p["label"].split("→")[0].strip(), "Turnover (%)": round(to, 1)})
        prev = curr
    return rows


# ---------------------------------------------------------------------------
# Optimised backtest helpers
# ---------------------------------------------------------------------------

def _find_nearest_before(target: str, dates: list[str]) -> str | None:
    candidates = [d for d in dates if d <= target]
    return max(candidates) if candidates else None


def _run_optimised_backtest(
    strategy_id: str,
    portfolio_eur: float,
    max_turnover: float,
    tc_per_trade_eur: float,
    benchmark_name: str,
    rebal_freq: str = "quarterly",
    min_weight: float = 0.0,
) -> dict:
    """
    Walk-forward optimised backtest.

    rebal_freq="quarterly": rebalance only at model snapshot dates (alpha + Barra both fresh).
    rebal_freq="monthly":   rebalance monthly; alpha held constant from the most recent
                            quarterly snapshot; Barra risk updated from the nearest weekly
                            snapshot — reduces ex-ante/realized TE gap caused by drift.

    Returns a results dict or {"error": str} on failure.
    """
    from optimize_portfolio import load_strategy_params, optimize_for_backtest

    if not PARAMS_FILE.exists():
        return {"error": "strategy_params.xlsx not found. Run create_strategy_params.py first."}
    try:
        strategies = load_strategy_params(strategy_id)
    except Exception as exc:
        return {"error": str(exc)}
    if not strategies:
        return {"error": f"Strategy '{strategy_id}' not found or inactive."}

    sp            = strategies[0]
    alpha_weights = sp["alpha_weights"]
    objective     = sp["objective"]
    constraints   = dict(sp["constraints"])

    # All available model dates and S&P 500 universe dates (kept separate for monthly lookups)
    with get_db(MODELS_DB) as conn:
        model_dates = sorted(r[0] for r in conn.execute("SELECT DISTINCT data_date FROM models").fetchall())
    with get_db(UNIVERSE_DB) as conn:
        sp500_dates = sorted(r[0] for r in conn.execute(
            "SELECT DISTINCT snapshot_date FROM universe_snapshots WHERE index_name = 'sp500'"
        ).fetchall())
    if not model_dates or not sp500_dates:
        return {"error": "Need at least 1 model date and 1 S&P 500 universe snapshot."}

    # Available Barra and LW risk dates
    barra_dates: list[str] = []
    try:
        with get_db(RISK_DB) as conn:
            barra_dates = sorted(r[0] for r in conn.execute(
                "SELECT DISTINCT snapshot_date FROM factor_covariance"
            ).fetchall())
    except Exception:
        barra_dates = []
    with get_db(RISK_DB) as conn:
        risk_dates = sorted(r[0] for r in conn.execute(
            "SELECT data_date FROM covariance_matrix"
        ).fetchall())
    if not risk_dates:
        return {"error": "No Ledoit-Wolf covariance matrices found. Run create_risk.py first."}

    ret_matrix    = load_returns_matrix()
    trading_index = ret_matrix.index

    def next_td(d_str: str):
        pos = trading_index.searchsorted(pd.Timestamp(d_str))
        return trading_index[pos] if pos < len(trading_index) else None

    # ── Build rebalancing schedule ────────────────────────────────────────────
    if rebal_freq == "monthly":
        # Monthly calendar dates → nearest trading day on or after each
        first_alpha = pd.Timestamp(model_dates[0])
        last_td     = trading_index[-1]
        anchors     = pd.date_range(start=first_alpha, end=last_td, freq="MS")
        rebal_dates: list[str] = []
        for anchor in anchors:
            pos = trading_index.searchsorted(anchor)
            if pos < len(trading_index):
                rebal_dates.append(trading_index[pos].strftime("%Y-%m-%d"))
        rebal_dates = sorted(set(rebal_dates))
    else:
        # Quarterly: intersection of model dates and S&P 500 universe dates (existing behaviour)
        rebal_dates = sorted(set(model_dates) & set(sp500_dates))

    if len(rebal_dates) < 2:
        return {"error": "Not enough rebalancing dates in the backtest window."}

    meta_df    = db.get_universe()[["security_id", "sector", "ticker", "company_name"]].copy()
    sector_map = dict(zip(meta_df["security_id"], meta_df["sector"]))
    ticker_map = dict(zip(meta_df["security_id"], meta_df["ticker"]))
    name_map   = dict(zip(meta_df["security_id"], meta_df["company_name"]))

    prev_weights: dict[str, float] | None = None
    period_log:   list[dict]              = []
    return_parts: list[pd.Series]         = []
    warnings:     list[str]               = []

    for i, snap_date in enumerate(rebal_dates):
        next_snap = (
            rebal_dates[i + 1] if i + 1 < len(rebal_dates)
            else trading_index[-1].strftime("%Y-%m-%d")
        )
        t_start = next_td(snap_date)
        t_end   = next_td(next_snap)
        if t_start is None or t_end is None or t_start >= t_end:
            continue

        # For monthly rebalancing, alpha and S&P 500 universe come from the most recent
        # quarterly snapshot; Barra risk is updated from the nearest weekly snapshot.
        alpha_date   = _find_nearest_before(snap_date, model_dates)
        sp500_snap   = _find_nearest_before(snap_date, sp500_dates)
        barra_date   = _find_nearest_before(snap_date, barra_dates)
        risk_date    = _find_nearest_before(snap_date, risk_dates)

        if alpha_date is None or sp500_snap is None:
            warnings.append(f"{snap_date}: no alpha or universe snapshot available — skipped.")
            continue
        if risk_date is None:
            warnings.append(f"{snap_date}: no LW risk date available — skipped.")
            continue

        sp500_isins = db.get_sp500_isins_at_date(sp500_snap)
        bm_weights  = db.get_sp500_weights_at_date(sp500_snap) if objective == "maximize_alpha" else {}
        if len(sp500_isins) < 50:
            warnings.append(f"{snap_date}: only {len(sp500_isins)} S&P 500 stocks — skipped.")
            continue

        opt_result = optimize_for_backtest(
            alpha_weights=alpha_weights,
            objective=objective,
            constraints=constraints,
            alpha_date=alpha_date,
            barra_date=barra_date,
            risk_date=risk_date,
            sp500_isins=sp500_isins,
            bm_weights=bm_weights,
            prev_weights=prev_weights,
            max_turnover=max_turnover,
            solver="CLARABEL",
            min_weight=min_weight,
        )

        if opt_result is None:
            warnings.append(f"{snap_date}: optimization failed — carrying forward previous weights.")
            new_weights: dict[str, float] = (
                prev_weights if prev_weights is not None
                else {isin: 1.0 / min(100, len(sp500_isins)) for isin in sp500_isins[:100]}
            )
            opt_metrics: dict = {}
        else:
            new_weights, opt_metrics = opt_result

        # Transaction costs: count a trade only when the EUR value of the order meets a
        # minimum order size. Monthly rebalancing produces many small weight tweaks that
        # would not be executed in practice (placing a €30 order at €2 commission = 6.7%).
        # Threshold scales with portfolio so TC% is independent of portfolio size.
        min_order_eur   = max(tc_per_trade_eur / 0.01, 200.0)  # cap TC at ~1% of order value
        trade_threshold = min_order_eur / portfolio_eur
        if prev_weights is None:
            n_trades = sum(1 for w in new_weights.values() if w >= trade_threshold)
        else:
            n_trades = sum(
                1 for isin in set(new_weights) | set(prev_weights)
                if abs(new_weights.get(isin, 0.0) - prev_weights.get(isin, 0.0)) >= trade_threshold
            )
        tc_pct = n_trades * tc_per_trade_eur / portfolio_eur

        # Hold-period return simulation
        period = ret_matrix.loc[(ret_matrix.index >= t_start) & (ret_matrix.index < t_end)]
        avail  = [isin for isin in new_weights if isin in period.columns]
        if avail:
            w_arr        = np.array([new_weights[isin] for isin in avail])
            w_arr       /= w_arr.sum()
            port_returns = pd.Series(
                period[avail].fillna(0.0).values @ w_arr, index=period.index
            )
        else:
            port_returns = pd.Series(0.0, index=period.index)

        if len(port_returns) > 0:
            port_returns.iloc[0] -= tc_pct

        # One-way turnover (fractional)
        if prev_weights is not None and new_weights:
            actual_to = sum(
                abs(new_weights.get(isin, 0.0) - prev_weights.get(isin, 0.0))
                for isin in set(new_weights) | set(prev_weights)
            ) / 2
        else:
            actual_to = 1.0

        sector_weights: dict[str, float] = {}
        for isin, w in new_weights.items():
            sec = sector_map.get(isin, "Unknown")
            sector_weights[sec] = sector_weights.get(sec, 0.0) + w

        period_log.append({
            "snap_date":       snap_date,
            "next_snap":       next_snap[:10],
            "alpha_date":      alpha_date,
            "weights":         new_weights,
            "n_trades":        n_trades,
            "tc_pct":          tc_pct,
            "turnover":        actual_to,
            "sector_weights":  sector_weights,
            "metrics":         opt_metrics,
            "used_barra":      opt_metrics.get("used_barra", False),
            "relaxed_integer": opt_metrics.get("relaxed_integer", False),
            "n_positions":     opt_metrics.get("n_positions", len(new_weights)),
        })
        return_parts.append(port_returns)
        prev_weights = new_weights

    if not return_parts:
        return {"error": "No valid backtest periods found."}

    return {
        "port_series":   pd.concat(return_parts).sort_index(),
        "period_log":    period_log,
        "warnings":      warnings,
        "strategy_name": sp["name"],
        "objective":     objective,
        "rebal_freq":    rebal_freq,
        "sector_map":    sector_map,
        "ticker_map":    ticker_map,
        "name_map":      name_map,
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Strategy settings")

    model_meta    = db.get_model_metadata()
    model_options = {f"{r['Model']} ({r['ModelID']})": r["ModelID"]
                     for _, r in model_meta.iterrows()}
    sel_model_label = st.selectbox("Model", list(model_options.keys()))
    sel_model_id    = model_options[sel_model_label]

    n_long = st.slider("Long portfolio size (top N)", 10, 200, 50, step=10)

    include_short = st.toggle("Add short leg", value=False)
    n_short = st.slider("Short portfolio size (bottom N)", 10, 200, 50, step=10) if include_short else 0

    st.divider()

    all_sectors = sorted(db.get_universe()["sector"].dropna().unique())
    sel_sectors = st.multiselect("Sector filter", all_sectors, placeholder="All sectors")

    st.divider()

    date_range = st.date_input(
        "Date range",
        value=(pd.Timestamp("2021-04-01").date(), pd.Timestamp.today().date()),
        min_value=pd.Timestamp("2020-01-01").date(),
        max_value=pd.Timestamp.today().date(),
    )

    st.divider()
    st.caption(
        f"Risk-free rate: {RISK_FREE:.0%} (Sharpe).  \n"
        "Equal-weight within each leg.  \n"
        "Rebalances at each annual factor snapshot."
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2 = st.tabs(["Factor Backtest", "Optimised Backtest"])


# ===========================================================================
# Tab 1 — Factor Backtest
# ===========================================================================

with tab1:
    st.caption(
        "Annual rebalancing: rank stocks by model score at each snapshot, hold equal-weight "
        "top N. Pre-computed daily total returns; no transaction costs."
    )

    with st.spinner("Running backtest…"):
        long_s, short_s, benchmark, holdings_log, err = run_backtest(
            sel_model_id, n_long, include_short, n_short, sel_sectors
        )

    if err:
        st.warning(err)
        st.stop()

    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        d_start   = pd.Timestamp(date_range[0])
        d_end     = pd.Timestamp(date_range[1])
        long_s    = long_s.loc[d_start:d_end]
        benchmark = benchmark.loc[d_start:d_end]
        if short_s is not None:
            short_s = short_s.loc[d_start:d_end]
    else:
        d_start = long_s.index[0]
        d_end   = long_s.index[-1]

    long_label  = f"Long (top {n_long}) — {sel_model_label.split(' (')[0]}"
    short_label = f"Short basket (bottom {n_short})"
    cum_bench   = (1 + benchmark).cumprod()   # reused in quintile chart

    turnover_rows = _compute_turnover(holdings_log)
    avg_turnover  = (
        np.mean([r["Turnover (%)"] for r in turnover_rows[1:]])
        if len(turnover_rows) > 1 else None
    )

    # Cumulative return chart
    cum_traces = [{"series": long_s, "name": long_label, "color": "#2563EB"}]
    if short_s is not None:
        cum_traces.append({"series": short_s, "name": short_label,
                           "color": "#DC2626", "dash": "dash"})
    cum_traces.append({"series": benchmark, "name": "EW universe",
                       "color": "#94A3B8", "width": 1.5, "dash": "dot"})
    st.plotly_chart(_cum_return_chart(cum_traces), use_container_width=True)

    # Drawdown chart
    dd_traces = [
        {"series": long_s,  "name": long_label, "color": "#2563EB",
         "fill_color": "rgba(37,99,235,0.08)"},
    ]
    if short_s is not None:
        dd_traces.append({"series": short_s, "name": short_label, "color": "#DC2626",
                          "dash": "dash", "fill_color": "rgba(220,38,38,0.05)"})
    dd_traces.append({"series": benchmark, "name": "EW universe",
                      "color": "#94A3B8", "width": 1, "dash": "dot"})
    st.plotly_chart(_drawdown_chart(dd_traces), use_container_width=True)

    # Metrics
    st.divider()
    n_cols = 3 if short_s is not None else 2
    cols   = st.columns(n_cols)
    with cols[0]:
        st.subheader("Long basket")
        _metrics_table({**perf_metrics(long_s), **active_metrics(long_s, benchmark, avg_turnover)})
    if short_s is not None:
        with cols[1]:
            st.subheader("Short basket (held long)")
            st.caption("↑ outperformance here = headwind for short position")
            _metrics_table(perf_metrics(short_s))
    with cols[-1]:
        st.subheader("EW benchmark")
        _metrics_table(perf_metrics(benchmark))

    # Returns chart
    st.divider()
    st.subheader("Returns")
    ret_view = st.segmented_control(
        "View", ["Annual", "Monthly heatmap"], default="Annual", key="ret_view"
    )

    if ret_view == "Annual":
        annual_traces = [{"series": long_s, "name": "Long basket", "color": "#2563EB"}]
        if short_s is not None:
            annual_traces.append({"series": short_s, "name": "Short basket", "color": "#DC2626"})
        annual_traces.append({"series": benchmark, "name": "EW universe", "color": "#94A3B8"})
        st.plotly_chart(_annual_bar_chart(annual_traces), use_container_width=True)
    else:
        MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly = long_s.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        mdf = monthly.reset_index()
        mdf.columns = ["date", "ret"]
        mdf["Year"]  = mdf["date"].dt.year.astype(str)
        mdf["Month"] = mdf["date"].dt.strftime("%b")
        pivot = mdf.pivot(index="Year", columns="Month", values="ret").reindex(columns=MONTH_ORDER)
        text  = pivot.map(lambda v: f"{v:+.1%}" if pd.notna(v) else "")
        fig_heat = go.Figure(go.Heatmap(
            z=pivot.values, x=MONTH_ORDER, y=pivot.index.tolist(),
            colorscale="RdYlGn", zmid=0,
            text=text.values, texttemplate="%{text}",
            hovertemplate="%{y} %{x}: %{text}<extra></extra>",
            showscale=True,
        ))
        fig_heat.update_layout(
            height=max(280, len(pivot) * 36 + 80),
            margin=dict(l=0, r=0, t=20, b=10),
            yaxis=dict(autorange="reversed"),
            xaxis_title="", yaxis_title="",
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption("Long basket monthly returns. Green = positive, red = negative.")

    # Rolling metrics
    st.divider()
    st.subheader("Rolling metrics")
    roll_choice = st.selectbox(
        "Metric",
        ["Rolling Sharpe (1Y)", "Rolling Information Ratio (1Y)", "Turnover by period"],
        key="roll_choice",
    )
    ROLL_WINDOW = 252

    if roll_choice == "Rolling Sharpe (1Y)":
        def _rolling_sharpe(s):
            return (s.rolling(ROLL_WINDOW).mean() * ROLL_WINDOW - RISK_FREE) / \
                   (s.rolling(ROLL_WINDOW).std() * ROLL_WINDOW ** 0.5)
        fig_roll = go.Figure()
        fig_roll.add_trace(go.Scatter(x=(rs := _rolling_sharpe(long_s).dropna()).index,
                                      y=rs.values, name=long_label,
                                      line=dict(color="#2563EB", width=2)))
        fig_roll.add_trace(go.Scatter(x=(rb := _rolling_sharpe(benchmark).dropna()).index,
                                      y=rb.values, name="EW universe",
                                      line=dict(color="#94A3B8", width=1.5, dash="dot")))
        fig_roll.add_hline(y=0, line_dash="dot", line_color="#64748B", line_width=1)
        fig_roll.update_layout(height=300, yaxis_title="Sharpe ratio (1Y rolling)",
                               hovermode="x unified", legend=dict(orientation="h", y=-0.2),
                               margin=dict(l=0, r=0, t=10, b=10))
        st.plotly_chart(fig_roll, use_container_width=True)

    elif roll_choice == "Rolling Information Ratio (1Y)":
        active  = long_s.subtract(benchmark.reindex(long_s.index, fill_value=0))
        roll_ir = (
            (active.rolling(ROLL_WINDOW).mean() * ROLL_WINDOW)
            / (active.rolling(ROLL_WINDOW).std() * ROLL_WINDOW ** 0.5)
        ).dropna()
        fig_roll = go.Figure()
        fig_roll.add_trace(go.Scatter(x=roll_ir.index, y=roll_ir.values, name="Rolling IR",
                                      line=dict(color="#2563EB", width=2),
                                      fill="tozeroy", fillcolor="rgba(37,99,235,0.07)"))
        for y_val, col in [(0, "#64748B"), (0.5, "#22C55E"), (-0.5, "#EF4444")]:
            fig_roll.add_hline(y=y_val,
                               line_dash="dot" if y_val == 0 else "dash",
                               line_color=col, line_width=1,
                               annotation_text=f"IR {y_val:.1f}" if y_val != 0 else "",
                               annotation_position="right")
        fig_roll.update_layout(height=300, yaxis_title="Information ratio (1Y rolling)",
                               hovermode="x unified", margin=dict(l=0, r=0, t=10, b=10))
        st.plotly_chart(fig_roll, use_container_width=True)
        st.caption(
            "IR > 0.5 (green) = consistently adding active return. "
            "Below zero = model underperformed EW universe on a risk-adjusted basis."
        )

    else:
        if not turnover_rows:
            st.info("Need at least 2 rebalance periods to compute turnover.")
        else:
            to_df  = pd.DataFrame(turnover_rows)
            fig_to = go.Figure(go.Bar(
                x=to_df["Period"], y=to_df["Turnover (%)"],
                marker_color="#2563EB",
                text=[f"{v:.0f}%" for v in to_df["Turnover (%)"]],
                textposition="outside",
            ))
            fig_to.update_layout(height=300, yaxis_title="One-way turnover (%)",
                                 yaxis_range=[0, 110],
                                 margin=dict(l=0, r=0, t=10, b=10))
            st.plotly_chart(fig_to, use_container_width=True)
            st.caption(
                "% of long portfolio replaced at each rebalance. First period = 100% (portfolio "
                f"built from scratch).  Average (ex-first): **{avg_turnover:.0f}%**."
            )

    # Quintile analysis
    st.divider()
    st.subheader("Quintile analysis")
    st.caption(
        "Universe split into 5 equal buckets by model score at each snapshot. "
        "Q1 = highest-scored stocks, Q5 = lowest. A working factor shows Q1 > Q2 > … > Q5."
    )
    with st.spinner("Computing quintiles…"):
        q_series = run_quintile_analysis(sel_model_id, sel_sectors)
    q_series = [s.loc[d_start:d_end] if not s.empty else s for s in q_series]

    Q_COLORS = ["#1D4ED8", "#60A5FA", "#94A3B8", "#F97316", "#DC2626"]
    Q_LABELS = [f"Q{i+1}" for i in range(N_QUINTILES)]

    q_tab1, q_tab2 = st.tabs(["Annualised returns by quintile", "Cumulative return"])
    with q_tab1:
        q_ann = []
        for i, s in enumerate(q_series):
            if s.empty:
                continue
            total   = (1 + s).prod() - 1
            n_years = len(s) / 252
            ann_r   = (1 + total) ** (1 / max(n_years, 1e-6)) - 1
            q_ann.append({"Quintile": Q_LABELS[i], "Ann. return": ann_r, "color": Q_COLORS[i]})
        if q_ann:
            fig_qa = go.Figure()
            for row in q_ann:
                fig_qa.add_trace(go.Bar(
                    x=[row["Quintile"]], y=[row["Ann. return"]], name=row["Quintile"],
                    marker_color=row["color"],
                    text=[f"{row['Ann. return']:+.1%}"], textposition="outside",
                    showlegend=False,
                ))
            bench_ann = (1 + benchmark).prod() ** (252 / max(len(benchmark), 1)) - 1
            fig_qa.add_hline(y=bench_ann, line_dash="dot", line_color="#94A3B8", line_width=1.5,
                             annotation_text=f"EW universe {bench_ann:+.1%}",
                             annotation_position="right")
            fig_qa.update_layout(height=340, yaxis_tickformat=".0%",
                                 yaxis_title="Annualised return",
                                 margin=dict(l=0, r=80, t=20, b=10))
            st.plotly_chart(fig_qa, use_container_width=True)

    with q_tab2:
        fig_qc = go.Figure()
        for i, s in enumerate(q_series):
            if not s.empty:
                cum = (1 + s).cumprod()
                fig_qc.add_trace(go.Scatter(x=cum.index, y=cum.values, name=Q_LABELS[i],
                                            line=dict(color=Q_COLORS[i], width=2)))
        fig_qc.add_trace(go.Scatter(x=cum_bench.index, y=cum_bench.values, name="EW universe",
                                    line=dict(color="#94A3B8", width=1.5, dash="dot")))
        fig_qc.update_layout(height=380, yaxis_title="Portfolio value (base = 1.0)",
                             hovermode="x unified", legend=dict(orientation="h", y=-0.15),
                             margin=dict(l=0, r=0, t=20, b=10))
        st.plotly_chart(fig_qc, use_container_width=True)

    # Holdings explorer
    st.divider()
    st.subheader("Portfolio holdings")
    period_labels    = [p["label"] for p in holdings_log]
    sel_period_label = st.segmented_control(
        "Period", period_labels, default=period_labels[-1], key="holdings_period"
    )
    sel_period = next((p for p in holdings_log if p["label"] == sel_period_label), holdings_log[-1])

    lcol, rcol = st.columns([3, 2])
    with lcol:
        long_df = sel_period["long"]
        st.markdown(f"**Long basket — {len(long_df)} stocks**")
        st.dataframe(long_df, hide_index=True, use_container_width=True, column_config={
            "Rank":       st.column_config.NumberColumn(width="small"),
            "Score":      st.column_config.NumberColumn(format="%.3f", width="small"),
            "Price data": st.column_config.TextColumn(width="small"),
        })
    with rcol:
        st.plotly_chart(_sector_chart(sel_period["long"], "Sector breakdown — long", "#2563EB"),
                        use_container_width=True)

    short_df = sel_period.get("short")
    if short_df is not None and not short_df.empty:
        st.markdown("---")
        slcol, srcol = st.columns([3, 2])
        with slcol:
            st.markdown(f"**Short basket — {len(short_df)} stocks**")
            st.dataframe(short_df, hide_index=True, use_container_width=True, column_config={
                "Rank":       st.column_config.NumberColumn(width="small"),
                "Score":      st.column_config.NumberColumn(format="%.3f", width="small"),
                "Price data": st.column_config.TextColumn(width="small"),
            })
        with srcol:
            st.plotly_chart(_sector_chart(short_df, "Sector breakdown — short", "#DC2626"),
                            use_container_width=True)


# ===========================================================================
# Tab 2 — Optimised Backtest
# ===========================================================================

with tab2:
    st.caption(
        "Walk-forward quarterly rebalancing using the CVXPY optimizer.  "
        "Universe: S&P 500 constituents at each snapshot.  "
        "Risk model: Barra (K=29 factors, LW fallback).  "
        f"Transaction cost: €{TC_EUR:.0f} per order; orders counted only when trade value ≥ €{TC_EUR/0.01:.0f} (~1% commission ratio).  "
        "Turnover: one-way quarterly limit.  "
        "Note: **realized TE** will exceed the strategy's ex-ante TE constraint — the optimizer "
        "targets ex-ante Barra TE at each rebalance; quarterly drift and index composition "
        "changes accumulate realized TE between rebalances."
    )

    # ── Controls ──────────────────────────────────────────────────────────────
    available_indices = db.get_available_benchmark_indices()

    if not PARAMS_FILE.exists():
        st.warning("strategy_params.xlsx not found. Run `create_strategy_params.py` first.")
        st.stop()

    xl = pd.ExcelFile(PARAMS_FILE)
    strats_df      = pd.read_excel(xl, sheet_name="Strategies", dtype=str)
    strats_df      = strats_df[strats_df["active"].str.strip().str.upper() == "TRUE"]
    strategy_opts  = {row["name"].strip(): row["strategy_id"].strip()
                      for _, row in strats_df.iterrows()}

    c1, c2, c3, c4, c5, c6 = st.columns([2, 1.5, 1, 1.5, 1.5, 1])
    with c1:
        sel_strat_name = st.selectbox("Strategy", list(strategy_opts.keys()), key="opt_bt_strat")
        sel_strat_id   = strategy_opts[sel_strat_name]
    with c2:
        portfolio_eur = st.number_input(
            "Portfolio size (€)", min_value=1_000, max_value=10_000_000,
            value=50_000, step=5_000, key="opt_bt_size",
        )
    with c3:
        max_to_pct = st.slider(
            "Max turnover (%)", min_value=5, max_value=50, value=10, step=5, key="opt_bt_to"
        )
    with c4:
        default_bench = "sp500" if "sp500" in available_indices else (available_indices[0] if available_indices else "")
        sel_bench = st.selectbox(
            "Benchmark", available_indices,
            index=available_indices.index(default_bench) if default_bench in available_indices else 0,
            key="opt_bt_bench",
        )
    with c5:
        rebal_freq = st.radio(
            "Rebalancing", ["Quarterly", "Monthly"],
            captions=["Alpha + risk", "Stale alpha, fresh Barra"],
            key="opt_bt_freq",
        ).lower()
    with c6:
        min_weight_pct = st.number_input(
            "Min weight (%)", min_value=0.0, max_value=1.0,
            value=0.05, step=0.01, format="%.2f", key="opt_bt_minw",
        )
    min_weight = min_weight_pct / 100.0

    run_clicked = st.button("▶ Run Optimised Backtest", type="primary", key="opt_bt_run")

    # ── Trigger computation ───────────────────────────────────────────────────
    result_key = f"opt_bt_{sel_strat_id}_{portfolio_eur}_{max_to_pct}_{sel_bench}_{rebal_freq}_{min_weight_pct}"

    if run_clicked:
        # Clear stale results from other parameter combinations
        for k in [k for k in st.session_state if k.startswith("opt_bt_") and k != result_key]:
            del st.session_state[k]
        est_time = "~30–60 s" if rebal_freq == "quarterly" else "~2–3 min"
        with st.spinner(f"Running {rebal_freq} walk-forward backtest for '{sel_strat_name}'…  ({est_time})"):
            st.session_state[result_key] = _run_optimised_backtest(
                strategy_id      = sel_strat_id,
                portfolio_eur    = float(portfolio_eur),
                max_turnover     = max_to_pct / 100.0,
                tc_per_trade_eur = TC_EUR,
                benchmark_name   = sel_bench,
                rebal_freq       = rebal_freq,
                min_weight       = min_weight,
            )

    # ── Display results ───────────────────────────────────────────────────────
    if result_key not in st.session_state:
        st.info("Configure the strategy above and click **▶ Run Optimised Backtest** to start.")
        st.stop()

    result = st.session_state[result_key]

    if "error" in result:
        st.error(result["error"])
        st.stop()

    port_series = result["port_series"]
    period_log  = result["period_log"]
    sector_map  = result["sector_map"]
    ticker_map  = result["ticker_map"]
    name_map    = result["name_map"]

    if result["warnings"]:
        with st.expander(f"{len(result['warnings'])} warning(s)"):
            for w in result["warnings"]:
                st.caption(w)

    bench_series = db.get_benchmark_returns(sel_bench)
    if bench_series.empty:
        st.warning(f"No benchmark returns found for '{sel_bench}'.")
        bench_series = pd.Series(0.0, index=port_series.index, name=sel_bench)
    bench_series = bench_series.reindex(port_series.index).fillna(0.0)
    bench_label  = sel_bench.replace("_", " ").title()

    # Summary metrics row
    avg_to_pct_actual  = np.mean([p["turnover"] for p in period_log[1:]]) * 100 if len(period_log) > 1 else 100.0
    total_tc           = sum(p["tc_pct"] for p in period_log) * portfolio_eur
    any_barra          = any(p["used_barra"] for p in period_log)
    any_relaxed        = any(p["relaxed_integer"] for p in period_log)
    freq_label         = result.get("rebal_freq", "quarterly").capitalize()
    m1, m2, m3, m4    = st.columns(4)
    m1.metric("Periods", f"{len(period_log)} ({freq_label})")
    m2.metric("Avg turnover", f"{avg_to_pct_actual:.0f}%")
    m3.metric("Total TC (est.)", f"€{total_tc:,.0f}")
    m4.metric("Risk model", "Barra" if any_barra else "Ledoit-Wolf")
    if any_relaxed:
        st.info(
            "**max_positions and min_position_if_held not applied** — these require MOSEK's "
            "mixed-integer solver and are infeasible when running strategies designed for the "
            "Russell 1000 universe against the S&P 500 (different benchmark composition makes "
            "the cardinality + active-weight combination geometrically infeasible). "
            "The backtest applies all continuous constraints: active risk, stock/sector/industry "
            "active weights, and turnover."
        )

    st.divider()

    # ── Result tabs ───────────────────────────────────────────────────────────
    pt1, pt2, pt3 = st.tabs(["Performance", "Analysis", "Holdings"])

    with pt1:
        st.plotly_chart(_cum_return_chart([
            {"series": port_series,  "name": result["strategy_name"], "color": "#2563EB"},
            {"series": bench_series, "name": bench_label, "color": "#94A3B8",
             "width": 1.5, "dash": "dot"},
        ]), use_container_width=True)

        st.plotly_chart(_drawdown_chart([
            {"series": port_series,  "name": result["strategy_name"], "color": "#2563EB",
             "fill_color": "rgba(37,99,235,0.08)"},
            {"series": bench_series, "name": bench_label, "color": "#94A3B8",
             "width": 1, "dash": "dot"},
        ]), use_container_width=True)

        st.divider()
        mc1, mc2 = st.columns(2)
        with mc1:
            st.subheader(result["strategy_name"])
            _metrics_table({
                **perf_metrics(port_series),
                **active_metrics(port_series, bench_series, avg_to_pct_actual),
            })
        with mc2:
            st.subheader(bench_label)
            _metrics_table(perf_metrics(bench_series))

    with pt2:
        st.subheader("Annual returns")
        st.plotly_chart(_annual_bar_chart([
            {"series": port_series,  "name": result["strategy_name"], "color": "#2563EB"},
            {"series": bench_series, "name": bench_label, "color": "#94A3B8"},
        ], height=340), use_container_width=True)

        st.divider()
        st.subheader("Turnover per period")
        to_rows = [
            {"Period": p["snap_date"][:10], "Turnover (%)": round(p["turnover"] * 100, 1),
             "# Trades": p["n_trades"], "TC (€)": round(p["tc_pct"] * portfolio_eur, 0)}
            for p in period_log
        ]
        fig_to = go.Figure(go.Bar(
            x=[r["Period"] for r in to_rows], y=[r["Turnover (%)"] for r in to_rows],
            marker_color="#2563EB",
            text=[f"{r['Turnover (%)']:.0f}%" for r in to_rows], textposition="outside",
        ))
        fig_to.add_hline(y=max_to_pct, line_dash="dash", line_color="#EF4444", line_width=1.5,
                         annotation_text=f"Limit {max_to_pct}%", annotation_position="right")
        max_to_shown = max(r["Turnover (%)"] for r in to_rows) * 1.2
        fig_to.update_layout(height=300, yaxis_title="One-way turnover (%)",
                             yaxis_range=[0, max(max_to_pct * 1.5, max_to_shown)],
                             margin=dict(l=0, r=60, t=10, b=10))
        st.plotly_chart(fig_to, use_container_width=True)
        st.caption(
            f"First period is 100% turnover (portfolio built from scratch). "
            f"Constraint: {max_to_pct}% one-way."
        )

        st.divider()
        st.subheader("Sector weights over time")
        sw_rows = [
            {"Period": p["snap_date"][:10], "Sector": sec, "Weight": w}
            for p in period_log
            for sec, w in p["sector_weights"].items()
        ]
        if sw_rows:
            sw_pivot = (
                pd.DataFrame(sw_rows)
                .pivot_table(index="Period", columns="Sector", values="Weight", aggfunc="sum")
                .fillna(0)
            )
            SECTOR_COLORS = [
                "#1D4ED8", "#2563EB", "#3B82F6", "#60A5FA", "#93C5FD",
                "#DC2626", "#EF4444", "#F87171", "#FCA5A5", "#FEE2E2",
                "#16A34A", "#22C55E",
            ]
            fig_sw = go.Figure()
            for j, sec in enumerate(sw_pivot.columns):
                fig_sw.add_trace(go.Bar(
                    name=sec, x=sw_pivot.index.tolist(), y=sw_pivot[sec].values,
                    marker_color=SECTOR_COLORS[j % len(SECTOR_COLORS)],
                ))
            fig_sw.update_layout(barmode="stack", height=380,
                                 yaxis_tickformat=".0%", yaxis_title="Portfolio weight",
                                 legend=dict(orientation="h", y=-0.3),
                                 margin=dict(l=0, r=0, t=10, b=10))
            st.plotly_chart(fig_sw, use_container_width=True)

    with pt3:
        st.subheader("Holdings snapshot")
        snap_labels = [p["snap_date"][:10] for p in period_log]
        sel_snap    = st.selectbox("Rebalance date", snap_labels,
                                   index=len(snap_labels) - 1, key="opt_bt_snap")
        sel_entry   = next((p for p in period_log if p["snap_date"][:10] == sel_snap), period_log[-1])

        h_df = pd.DataFrame([
            {"Ticker":  ticker_map.get(isin, isin),
             "Company": name_map.get(isin, ""),
             "Sector":  sector_map.get(isin, "Unknown"),
             "Weight":  w}
            for isin, w in sorted(sel_entry["weights"].items(), key=lambda x: -x[1])
        ])

        hc1, hc2 = st.columns([3, 2])
        with hc1:
            st.markdown(f"**{len(h_df)} positions** — {sel_snap}")
            max_w = h_df["Weight"].max() if not h_df.empty else 0.05
            st.dataframe(h_df, hide_index=True, use_container_width=True, column_config={
                "Weight": st.column_config.ProgressColumn(
                    "Weight", format="%.2%%", min_value=0, max_value=max_w
                ),
            })
        with hc2:
            sec_w = h_df.groupby("Sector")["Weight"].sum().reset_index().sort_values("Weight")
            fig_sec = go.Figure(go.Bar(
                x=sec_w["Weight"], y=sec_w["Sector"], orientation="h", marker_color="#2563EB",
                text=[f"{v:.1%}" for v in sec_w["Weight"]], textposition="outside",
            ))
            fig_sec.update_layout(title="Sector weights",
                                  height=max(200, len(sec_w) * 30 + 60),
                                  xaxis_tickformat=".0%",
                                  margin=dict(l=0, r=60, t=40, b=10))
            st.plotly_chart(fig_sec, use_container_width=True)

            m = sel_entry["metrics"]
            if m:
                st.markdown("**Optimiser metrics**")
                rows_m = []
                if "expected_alpha" in m:
                    rows_m.append(("Expected alpha", f"{m['expected_alpha']:+.4f}"))
                if "portfolio_vol" in m:
                    rows_m.append(("Portfolio vol", f"{m['portfolio_vol']:.2%}"))
                if "active_risk" in m and m.get("active_risk") != m.get("portfolio_vol"):
                    rows_m.append(("Active risk", f"{m['active_risk']:.2%}"))
                if "sharpe_ratio" in m:
                    rows_m.append(("Sharpe ratio", f"{m['sharpe_ratio']:.2f}"))
                if "info_ratio" in m:
                    rows_m.append(("Info ratio", f"{m['info_ratio']:.2f}"))
                rows_m += [
                    ("Positions",    str(sel_entry["n_positions"])),
                    ("Trades",       str(sel_entry["n_trades"])),
                    ("TC",           f"€{sel_entry['tc_pct'] * portfolio_eur:,.0f}"),
                    ("Risk model",   "Barra" if sel_entry["used_barra"] else "Ledoit-Wolf"),
                    ("Constraints",  "Relaxed (no max_positions)" if sel_entry["relaxed_integer"] else "Full"),
                    ("Alpha date",   sel_entry.get("alpha_date", sel_entry["snap_date"])),
                ]
                _metrics_table(dict(rows_m))
