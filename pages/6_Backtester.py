"""
6_Backtester.py — signal backtest, optimised backtest, and signal diagnostics.

Tab 1 — Signal Backtest   : rank stocks by model z-score, hold equal-weight top-N.
Tab 2 — Optimised Backtest : CVXPY walk-forward with quarterly rebalancing,
         two-way turnover constraint, Barra risk model, configurable universe,
         and per-trade EUR transaction costs.
Tab 3 — Signal Diagnostics : holistic predictive-power view across all signals —
         cumulative long-short curves, IC scorecard, IC decay, signal overlap.

All controls live in-page per tab (no sidebar).
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
import db
from config import (
    FACTORS_DB,
    FACTORS_REF,
    MODELS_DB,
    PARAMS_FILE,
    RETURNS_DB,
    RISK_DB,
    UNIVERSE_DB,
)
from scripts.backtest_engine import (
    find_nearest_before as _find_nearest_before,
    load_returns_matrix as _load_returns_matrix_impl,
    run_optimised_backtest as _run_optimised_backtest_impl,
)
from utils import get_db, inject_css

st.set_page_config(page_title="Backtester", layout="wide")
inject_css()
st.title("Backtester")

RISK_FREE   = 0.04  # annualised, used for Sharpe
N_QUINTILES = 5
TC_EUR      = 2.0   # €2 per trade (DeGiro US stocks)
SIGNAL_BACKTEST_START = "2021-05-31"


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
def load_all_factor_scores() -> pd.DataFrame:
    """Individual-factor z-scores, shaped exactly like model scores.

    factors.db stores unsigned ``factor_value_z``; direction (±1) lives only in
    factors_reference.csv. We apply it here so a higher score is always "better",
    then rename ``factor_id`` → ``model_id`` so the whole backtest/diagnostics
    machinery (which only reads ``model_value_z``) works unchanged.
    """
    ref     = pd.read_csv(FACTORS_REF)
    dir_map = dict(zip(ref["factor_id"], pd.to_numeric(ref["direction"], errors="coerce")))
    with get_db(FACTORS_DB) as conn:
        df = pd.read_sql(
            "SELECT data_date, factor_id, security_id, factor_value_z FROM factors", conn
        )
    z = pd.to_numeric(df["factor_value_z"], errors="coerce")
    df["model_value_z"] = z * df["factor_id"].map(dir_map)
    df = df.rename(columns={"factor_id": "model_id"})
    return df[["data_date", "model_id", "security_id", "model_value_z"]].dropna(
        subset=["model_value_z"]
    )


def _load_scores(source: str) -> pd.DataFrame:
    """Score frame for the selected signal source ('model' or 'factor')."""
    return load_all_factor_scores() if source == "factor" else load_all_model_scores()


@st.cache_data
def factor_label_map() -> dict[str, str]:
    """factor_id → 'Factor Name (CATEGORY)' for the factor selectbox/diagnostics."""
    ref = pd.read_csv(FACTORS_REF)
    return {r["factor_id"]: f"{r['factor_name']}" for _, r in ref.iterrows()}


@st.cache_data
def load_returns_matrix(min_isin_coverage: int = 200) -> pd.DataFrame:
    """Streamlit-cached wrapper over the shared backtest engine loader."""
    return _load_returns_matrix_impl(min_isin_coverage)


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def perf_metrics(ret: pd.Series, subtract_risk_free: bool = True) -> dict:
    ret = ret.dropna()
    if len(ret) < 10:
        return {}
    total   = (1 + ret).prod() - 1
    n_years = len(ret) / 252
    ann_ret = (1 + total) ** (1 / max(n_years, 1e-6)) - 1
    ann_vol = ret.std() * 252 ** 0.5
    sharpe  = (ann_ret - (RISK_FREE if subtract_risk_free else 0.0)) / ann_vol if ann_vol > 0 else np.nan
    cum     = (1 + ret).cumprod()
    max_dd  = (cum / cum.cummax() - 1).min()
    downside = ret[ret < 0].std() * 252 ** 0.5
    sortino = (ann_ret - (RISK_FREE if subtract_risk_free else 0.0)) / downside if downside > 0 else np.nan
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.nan
    monthly = ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    return {
        "Total return":    f"{total:+.1%}",
        "Ann. return":     f"{ann_ret:+.1%}",
        "Ann. volatility": f"{ann_vol:.1%}",
        "Sharpe ratio":    f"{sharpe:.2f}",
        "Sortino ratio":   f"{sortino:.2f}" if pd.notna(sortino) else "—",
        "Calmar ratio":    f"{calmar:.2f}" if pd.notna(calmar) else "—",
        "Max drawdown":    f"{max_dd:.1%}",
        "Daily win rate":  f"{(ret > 0).mean():.1%}",
        "Positive months": f"{(monthly > 0).mean():.1%}" if len(monthly) else "—",
        "Best month":      f"{monthly.max():+.1%}" if len(monthly) else "—",
        "Worst month":     f"{monthly.min():+.1%}" if len(monthly) else "—",
    }


def active_metrics(
    ret: pd.Series,
    bench: pd.Series,
    turnover_pct: float | None,
    turnover_label: str = "Avg rebal. turnover (2-way)",
) -> dict:
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
        out[turnover_label] = f"{turnover_pct:.0f}%"
    return out


def _cumulative_spread_series(long_ret: pd.Series, short_ret: pd.Series) -> pd.Series:
    common = long_ret.index.intersection(short_ret.index)
    long_cum = (1 + long_ret.loc[common]).cumprod() - 1
    short_cum = (1 + short_ret.loc[common]).cumprod() - 1
    return long_cum - short_cum


def spread_metrics(long_ret: pd.Series, short_ret: pd.Series, spread_daily: pd.Series) -> dict:
    spread_cum = _cumulative_spread_series(long_ret, short_ret).dropna()
    spread_daily = spread_daily.dropna()
    if len(spread_cum) < 10 or len(spread_daily) < 10:
        return {}
    total = spread_cum.iloc[-1]
    n_years = len(spread_daily) / 252
    ann_ret = (1 + total) ** (1 / max(n_years, 1e-6)) - 1
    ann_vol = spread_daily.std() * 252 ** 0.5
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    wealth = 1 + spread_cum
    max_dd = (wealth / wealth.cummax() - 1).min()
    downside = spread_daily[spread_daily < 0].std() * 252 ** 0.5
    sortino = ann_ret / downside if downside > 0 else np.nan
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.nan
    monthly = (
        long_ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        - short_ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    ).dropna()
    return {
        "Total return":    f"{total:+.1%}",
        "Ann. return":     f"{ann_ret:+.1%}",
        "Ann. volatility": f"{ann_vol:.1%}",
        "Sharpe ratio":    f"{sharpe:.2f}",
        "Sortino ratio":   f"{sortino:.2f}" if pd.notna(sortino) else "—",
        "Calmar ratio":    f"{calmar:.2f}" if pd.notna(calmar) else "—",
        "Max drawdown":    f"{max_dd:.1%}",
        "Daily win rate":  f"{(spread_daily > 0).mean():.1%}",
        "Positive months": f"{(monthly > 0).mean():.1%}" if len(monthly) else "—",
        "Best month":      f"{monthly.max():+.1%}" if len(monthly) else "—",
        "Worst month":     f"{monthly.min():+.1%}" if len(monthly) else "—",
    }


# ---------------------------------------------------------------------------
# Chart helpers  (shared by both tabs)
# ---------------------------------------------------------------------------

def _cum_return_chart(
    traces: list[dict],
    title: str = "Cumulative return (%)",
    height: int = 420,
    cumulative_excess_series: pd.Series | None = None,
) -> go.Figure:
    """
    Build a cumulative-return line chart. By default each line is a cumulative
    total return in %, starting at 0 — i.e. ((1 + r).cumprod() − 1) × 100.
    For precomputed cumulative-return series, set trace["cumulative"] = True.
    Each trace dict: {series, name, color, width=2, dash="solid"}.
    Optional cumulative_excess_series is cumulative portfolio return minus
    cumulative benchmark return (a fraction); it is the gap between those two
    lines and is plotted as % on the SAME axis.
    """
    fig = go.Figure()
    for t in traces:
        cum = t["series"] * 100 if t.get("cumulative") else ((1 + t["series"]).cumprod() - 1) * 100
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values, name=t["name"],
            line=dict(color=t["color"], width=t.get("width", 2), dash=t.get("dash", "solid")),
        ))
    if cumulative_excess_series is not None and not cumulative_excess_series.empty:
        fig.add_trace(go.Scatter(
            x=cumulative_excess_series.index, y=cumulative_excess_series.values * 100,
            name="Cumulative excess return",
            line=dict(color="#16A34A", width=1.5, dash="dot"),
        ))
    fig.add_hline(y=0, line_dash="dot", line_color="#94A3B8", line_width=1)
    fig.update_layout(
        title=title, height=height,
        yaxis=dict(title="Cumulative return (%)", ticksuffix="%"),
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
        cum  = 1 + t["series"] if t.get("cumulative") else (1 + t["series"]).cumprod()
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


def _rolling_risk_chart(
    port: pd.Series,
    bench: pd.Series,
    window: int = 63,
    height: int = 320,
) -> go.Figure | None:
    """
    Rolling annualised risk metrics for a portfolio vs its benchmark.

      • Tracking error  = std(port − bench) · √252   (left %, blue)
      • Absolute risk   = std(port)        · √252   (left %, slate)
      • Beta            = cov(port, bench) / var(bench)  (right axis, amber)

    Tracking error and absolute risk are both annualised % and share the left
    axis so their magnitudes are directly comparable. Beta is unitless (~1) and
    sits on a secondary right axis, so the level differences don't crush the
    % series — all three are readable on one chart.
    """
    common = port.index.intersection(bench.index)
    p = port.loc[common]
    b = bench.loc[common]
    if len(p) < window + 5:
        return None

    ann       = np.sqrt(252)
    active    = p - b
    roll_te   = (active.rolling(window).std() * ann * 100).dropna()
    roll_vol  = (p.rolling(window).std() * ann * 100).dropna()
    roll_var  = b.rolling(window).var()
    roll_beta = (p.rolling(window).cov(b) / roll_var).where(roll_var > 0).dropna()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=roll_vol.index, y=roll_vol.values, name="Absolute risk (vol)",
        line=dict(color="#94A3B8", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=roll_te.index, y=roll_te.values, name="Tracking error",
        line=dict(color="#2563EB", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=roll_beta.index, y=roll_beta.values, name="Beta (vs benchmark)",
        yaxis="y2", line=dict(color="#F59E0B", width=2, dash="dot"),
    ))
    fig.add_shape(
        type="line", xref="paper", x0=0, x1=1, yref="y2", y0=1, y1=1,
        line=dict(color="#F59E0B", width=1, dash="dot"),
    )
    fig.update_layout(
        title=f"Rolling risk ({window}-day, annualised)", height=height,
        hovermode="x unified",
        yaxis=dict(title="Risk (% annualised)", ticksuffix="%", rangemode="tozero"),
        yaxis2=dict(title="Beta", overlaying="y", side="right",
                    showgrid=False, zeroline=False),
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=0, r=0, t=40, b=10),
    )
    return fig


_FACTOR_PALETTE = [
    "#2563EB", "#DC2626", "#16A34A", "#F59E0B", "#7C3AED", "#0891B2",
    "#DB2777", "#65A30D", "#EA580C", "#0D9488", "#4F46E5", "#9333EA",
]


def _factor_exposure_chart(
    period_log: list[dict],
    active: bool,
    height: int = 400,
) -> go.Figure | None:
    """
    Portfolio Barra style/beta factor exposures across rebalance periods.

      active=True  → portfolio − benchmark exposure (centred on 0)
      active=False → absolute portfolio exposure

    Exposures are in standardised Barra units, so a single shared Y-axis keeps
    every factor directly comparable. Sector and market factors are excluded —
    sector tilts are covered by the "Sector weights over time" chart.

    Reuses db.load_barra_components / weighted_factor_exposures so the exposure
    maths matches the optimiser and Risk Explorer exactly.
    """
    records: dict[str, dict] = {}
    style_order: list[str] = []
    for p in period_log:
        bdate = p.get("barra_date")
        if not bdate:
            continue
        comps = db.load_barra_components(bdate)
        if comps is None:
            continue
        _F, fnames, X_df, _delta = comps
        styles = db.barra_style_factor_names(fnames)
        if not style_order:
            style_order = styles
        port = db.weighted_factor_exposures(X_df, p["weights"])
        if active:
            bm = p.get("bm_weights") or {}
            if not bm:
                continue
            exp = port - db.weighted_factor_exposures(X_df, bm)
        else:
            exp = port
        records[p["snap_date"][:10]] = {f: float(exp.get(f, 0.0)) for f in styles}

    if not records:
        return None
    df = pd.DataFrame.from_dict(records, orient="index").reindex(columns=style_order)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    fig = go.Figure()
    for j, f in enumerate(style_order):
        fig.add_trace(go.Scatter(
            x=df.index, y=df[f].values, name=db.barra_factor_label(f),
            mode="lines+markers",
            line=dict(color=_FACTOR_PALETTE[j % len(_FACTOR_PALETTE)], width=1.8),
            marker=dict(size=4),
        ))
    if active:
        fig.add_hline(y=0, line_dash="dot", line_color="#64748B", line_width=1)
    fig.update_layout(
        title=f"{'Active' if active else 'Absolute'} factor exposure over time",
        height=height, hovermode="x unified",
        yaxis_title="Exposure (Barra std. units)",
        legend=dict(orientation="h", y=-0.25),
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
        hide_index=True, width="stretch",
    )


def _metrics_comparison_table(metric_sets: dict[str, dict]) -> None:
    preferred_order = [
        "Total return",
        "Ann. return",
        "Ann. volatility",
        "Sharpe ratio",
        "Sortino ratio",
        "Calmar ratio",
        "Max drawdown",
        "Daily win rate",
        "Positive months",
        "Best month",
        "Worst month",
        "Active return (ann.)",
        "Tracking error",
        "Information ratio",
        "Beta (vs benchmark)",
        "Avg rebal. turnover (2-way)",
        "Avg turnover (2-way)",
        "Avg monthly turnover (2-way)",
    ]
    metrics = []
    seen = set()
    for metric in preferred_order:
        if any(metric in values for values in metric_sets.values()):
            metrics.append(metric)
            seen.add(metric)
    for values in metric_sets.values():
        for metric in values:
            if metric not in seen:
                metrics.append(metric)
                seen.add(metric)

    rows = []
    for metric in metrics:
        row = {"Metric": metric}
        for label, values in metric_sets.items():
            row[label] = values.get(metric, "")
        rows.append(row)

    df = pd.DataFrame(rows)
    value_cols = [c for c in df.columns if c != "Metric"]
    lower_is_better = {
        "Ann. volatility",
        "Max drawdown",
        "Tracking error",
        "Avg rebal. turnover (2-way)",
        "Avg turnover (2-way)",
        "Avg monthly turnover (2-way)",
        "Worst month",
    }
    higher_is_better = {
        "Total return",
        "Ann. return",
        "Sharpe ratio",
        "Sortino ratio",
        "Calmar ratio",
        "Daily win rate",
        "Positive months",
        "Best month",
        "Active return (ann.)",
        "Information ratio",
    }

    def _metric_value(v: object) -> float:
        if v is None:
            return np.nan
        text = str(v).strip()
        if not text or text == "—":
            return np.nan
        is_pct = text.endswith("%")
        text = text.replace("%", "").replace("+", "").replace(",", "")
        try:
            val = float(text)
        except ValueError:
            return np.nan
        return val / 100 if is_pct else val

    def _style_row(row: pd.Series) -> list[str]:
        metric = row["Metric"]
        styles = ["font-weight: 600; background-color: rgba(15,23,42,0.04)"]
        vals = pd.Series({_c: _metric_value(row[_c]) for _c in value_cols}).dropna()
        if metric in lower_is_better and len(vals) > 1:
            score = (vals.max() - vals) / (vals.max() - vals.min()) if vals.max() != vals.min() else vals * 0 + 0.5
        elif metric in higher_is_better and len(vals) > 1:
            score = (vals - vals.min()) / (vals.max() - vals.min()) if vals.max() != vals.min() else vals * 0 + 0.5
        else:
            return styles + ["" for _ in value_cols]
        for col in value_cols:
            if col not in score:
                styles.append("")
                continue
            s = float(score[col])
            if s >= 0.5:
                alpha = 0.06 + (s - 0.5) * 0.32
                styles.append(f"background-color: rgba(34,197,94,{alpha:.3f}); font-weight: {600 if s >= 0.85 else 400}")
            else:
                alpha = 0.06 + (0.5 - s) * 0.28
                styles.append(f"background-color: rgba(239,68,68,{alpha:.3f})")
        return styles

    st.dataframe(df.style.apply(_style_row, axis=1), hide_index=True, width="stretch")


def _align_benchmark_for_eval(
    port: pd.Series,
    bench: pd.Series,
) -> tuple[pd.Series, pd.Series, float]:
    if port.empty or bench.empty:
        return port.iloc[0:0], bench.iloc[0:0], 0.0
    bench_aligned = bench.reindex(port.index)
    valid_idx = bench_aligned.dropna().index
    coverage = len(valid_idx) / len(port) if len(port) else 0.0
    return port.loc[valid_idx], bench_aligned.loc[valid_idx], coverage


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
    model_id: str,
    portfolio_mode: str,
    bucket_pct: int,
    universe_name: str,
    sel_sectors: list,
    source: str = "model",
) -> tuple[
    pd.Series | None,
    pd.Series | None,
    pd.Series | None,
    pd.Series | None,
    list,
    str | None,
]:
    if not RETURNS_DB.exists():
        return None, None, None, None, [], "returns.db not found — run `create_returns.py --update`."
    if source == "model" and not MODELS_DB.exists():
        return None, None, None, None, [], "models.db not found — run `create_models.py`."
    if source == "factor" and not FACTORS_DB.exists():
        return None, None, None, None, [], "factors.db not found — run `create_factors.py`."

    scores     = _load_scores(source)
    ret_matrix = load_returns_matrix()
    ticker_map = db.get_ticker_map()

    uni = db.get_universe()[["security_id", "sector", "company_name"]].copy()
    uni["security_id"] = uni["security_id"].astype(str)
    sector_map = dict(zip(uni["security_id"], uni["sector"]))
    name_map   = dict(zip(uni["security_id"], uni["company_name"]))

    model_df = scores[scores["model_id"] == model_id].copy().dropna(subset=["model_value_z"])
    sector_isins = None
    if sel_sectors:
        sector_isins = set(uni[uni["sector"].isin(sel_sectors)]["security_id"])
        model_df = model_df[model_df["security_id"].isin(sector_isins)]

    with get_db(UNIVERSE_DB) as conn:
        universe_dates = sorted(r[0] for r in conn.execute(
            "SELECT DISTINCT snapshot_date FROM universe_snapshots WHERE index_name = ?",
            (universe_name,),
        ).fetchall())
    if not universe_dates:
        return None, None, None, None, [], f"No universe snapshots found for '{universe_name}'."

    snapshot_dates = [
        d for d in sorted(model_df["data_date"].unique())
        if d >= SIGNAL_BACKTEST_START
    ]
    if len(snapshot_dates) < 2:
        return None, None, None, None, [], (
            f"Need at least 2 snapshot dates on or after {SIGNAL_BACKTEST_START} "
            "for a backtest."
        )

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

    include_short = portfolio_mode == "Long-short"
    long_parts, bench_parts, short_parts, strategy_parts, holdings_log = [], [], [], [], []

    for i, snap in enumerate(snapshot_dates):
        next_snap = (
            snapshot_dates[i + 1] if i + 1 < len(snapshot_dates)
            else trading_index[-1].strftime("%Y-%m-%d")
        )
        t_start = next_td(snap)
        t_end   = next_td(next_snap)
        if t_start is None or t_end is None or t_start >= t_end:
            continue

        uni_snap = _find_nearest_before(snap, universe_dates)
        if uni_snap is None:
            continue
        universe_isins = set(db.get_universe_isins_at_date(universe_name, uni_snap))
        benchmark_isins = (
            universe_isins if sector_isins is None else universe_isins & sector_isins
        )
        snap_df = (
            model_df[model_df["data_date"] == snap]
            .dropna(subset=["model_value_z"])
        )
        snap_df = snap_df[snap_df["security_id"].isin(universe_isins)]
        if snap_df.empty:
            continue
        bucket_n = max(1, int(np.ceil(len(snap_df) * bucket_pct / 100.0)))
        if include_short:
            bucket_n = min(bucket_n, max(1, len(snap_df) // 2))
        score_lkp   = dict(zip(snap_df["security_id"], snap_df["model_value_z"]))
        long_isins  = snap_df.nlargest(bucket_n, "model_value_z")["security_id"].tolist()
        short_isins = (
            snap_df.nsmallest(bucket_n, "model_value_z")["security_id"].tolist()
            if include_short else []
        )
        period     = ret_matrix.loc[(ret_matrix.index >= t_start) & (ret_matrix.index < t_end)]
        price_cols = set(period.columns)

        def ew(isins):
            cols = [s for s in isins if s in price_cols]
            return period[cols].mean(axis=1) if cols else pd.Series(0.0, index=period.index)

        long_ret = ew(long_isins)
        long_parts.append(long_ret)
        bench_parts.append(ew(sorted(benchmark_isins)))
        if include_short:
            short_ret = ew(short_isins)
            short_parts.append(short_ret)
            strategy_parts.append(long_ret - short_ret)
        else:
            strategy_parts.append(long_ret)

        holdings_log.append({
            "label":       f"{snap[:10]}  →  {next_snap[:10]}",
            "snap_date":   snap[:10],
            "universe":    universe_name,
            "universe_snapshot": uni_snap,
            "long":        holdings_df(long_isins,  score_lkp, price_cols),
            "short":       holdings_df(short_isins, score_lkp, price_cols) if include_short else None,
            "long_isins":  long_isins,
            "short_isins": short_isins,
            "bench_isins": sorted(benchmark_isins),
            "bucket_pct":  bucket_pct,
        })

    if not long_parts:
        return None, None, None, None, [], "No overlapping price data found for this model and date range."

    return (
        pd.concat(long_parts).sort_index(),
        pd.concat(short_parts).sort_index() if include_short and short_parts else None,
        pd.concat(strategy_parts).sort_index(),
        pd.concat(bench_parts).sort_index(),
        holdings_log,
        None,
    )


# ---------------------------------------------------------------------------
# Quintile analysis
# ---------------------------------------------------------------------------

def run_quintile_analysis(
    model_id: str, universe_name: str, sel_sectors: list, source: str = "model"
) -> list[pd.Series]:
    scores     = _load_scores(source)
    ret_matrix = load_returns_matrix()
    model_df   = scores[scores["model_id"] == model_id].dropna(subset=["model_value_z"]).copy()

    if sel_sectors:
        uni = db.get_universe()[["security_id", "sector"]].copy()
        uni["security_id"] = uni["security_id"].astype(str)
        valid    = set(uni[uni["sector"].isin(sel_sectors)]["security_id"])
        model_df = model_df[model_df["security_id"].isin(valid)]

    with get_db(UNIVERSE_DB) as conn:
        universe_dates = sorted(r[0] for r in conn.execute(
            "SELECT DISTINCT snapshot_date FROM universe_snapshots WHERE index_name = ?",
            (universe_name,),
        ).fetchall())
    if not universe_dates:
        return [pd.Series(dtype=float) for _ in range(N_QUINTILES)]

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
        uni_snap = _find_nearest_before(snap, universe_dates)
        if uni_snap is None:
            continue
        universe_isins = set(db.get_universe_isins_at_date(universe_name, uni_snap))
        snap_df = snap_df[snap_df["security_id"].isin(universe_isins)].reset_index(drop=True)
        n      = len(snap_df)
        if n < N_QUINTILES:
            continue
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

def _filter_holdings_log(holdings_log: list, start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    rows = [
        p for p in holdings_log
        if start <= pd.Timestamp(p.get("snap_date", p["label"].split("→")[0].strip())) <= end
    ]
    return rows or holdings_log


def _equal_weight_dict(isins: list[str], gross: float = 1.0) -> dict[str, float]:
    if not isins:
        return {}
    w = gross / len(isins)
    return {isin: w for isin in isins}


def _two_way_turnover(curr: dict[str, float], prev: dict[str, float]) -> float:
    names = set(curr) | set(prev)
    return 0.5 * sum(abs(curr.get(isin, 0.0) - prev.get(isin, 0.0)) for isin in names)


def _compute_turnover(holdings_log: list, include_short: bool = False) -> list[dict]:
    rows = []
    prev_strategy: dict[str, float] | None = None
    prev_benchmark: dict[str, float] | None = None
    prev_date: pd.Timestamp | None = None
    for p in holdings_log:
        snap_date = pd.Timestamp(p.get("snap_date", p["label"].split("→")[0].strip()))
        strategy_gross = 2.0 if include_short else 1.0
        long_w = _equal_weight_dict(p.get("long_isins", []), 1.0 / strategy_gross)
        if include_short:
            short_w = _equal_weight_dict(p.get("short_isins", []), -1.0 / strategy_gross)
            curr_strategy = {**long_w, **short_w}
        else:
            curr_strategy = long_w
        curr_benchmark = _equal_weight_dict(p.get("bench_isins", []), 1.0)
        if not curr_strategy:
            continue
        if prev_strategy is not None and prev_date is not None:
            months = max((snap_date - prev_date).days / (365.25 / 12), 1e-6)
            strategy_period_turnover = _two_way_turnover(curr_strategy, prev_strategy) * 100
            benchmark_period_turnover = (
                _two_way_turnover(curr_benchmark, prev_benchmark) * 100
                if prev_benchmark is not None else np.nan
            )
            rows.append({
                "Period": p["label"].split("→")[0].strip(),
                "Months": round(months, 2),
                "Strategy monthly turnover (%)": round(strategy_period_turnover / months, 1),
                "Benchmark monthly turnover (%)": round(benchmark_period_turnover / months, 1)
                if pd.notna(benchmark_period_turnover) else np.nan,
                "Strategy period turnover (%)": round(strategy_period_turnover, 1),
                "Benchmark period turnover (%)": round(benchmark_period_turnover, 1)
                if pd.notna(benchmark_period_turnover) else np.nan,
            })
        prev_strategy = curr_strategy
        prev_benchmark = curr_benchmark
        prev_date = snap_date
    return rows


def _aggregate_monthly_turnover(turnover_rows: list[dict]) -> list[dict]:
    if not turnover_rows:
        return []
    df = pd.DataFrame(turnover_rows).copy()
    df["Month"] = pd.to_datetime(df["Period"]) + pd.offsets.MonthEnd(0)
    grouped = (
        df.groupby("Month", as_index=False)
        .agg({
            "Strategy period turnover (%)": "sum",
            "Benchmark period turnover (%)": "sum",
            "Period": "count",
        })
        .rename(columns={"Period": "Rebalances"})
    )
    grouped["Period"] = grouped["Month"].dt.strftime("%Y-%m-%d")
    grouped["Strategy monthly turnover (%)"] = grouped["Strategy period turnover (%)"].round(1)
    grouped["Benchmark monthly turnover (%)"] = grouped["Benchmark period turnover (%)"].round(1)
    return grouped[[
        "Period",
        "Rebalances",
        "Strategy monthly turnover (%)",
        "Benchmark monthly turnover (%)",
    ]].to_dict("records")


# ---------------------------------------------------------------------------
# Signal diagnostics helpers (Tab 3)
#
# Holistic, all-signals-at-once predictive-power analysis: how well does each
# model's z-score forecast forward returns, BEFORE any optimizer/constraints.
# Cross-sectional rank IC is the workhorse; everything reuses the model-score
# and returns matrices already loaded for the backtests.
# ---------------------------------------------------------------------------

DECAY_HORIZONS = (1, 3, 6, 12)   # forward horizons, in snapshots
_MIN_XS = 30                     # min cross-section size to score a date


def _periods_per_year(snaps: list[str]) -> float:
    """Annualisation factor from the median spacing of snapshot dates."""
    ts = pd.to_datetime(snaps)
    gaps = np.diff(ts.values).astype("timedelta64[D]").astype(float)
    med = np.median(gaps) if len(gaps) else 30.0
    return 365.25 / med if med > 0 else 12.0


def _forward_returns(ret: pd.DataFrame, snap_ts: list, snaps: list[str], k: int) -> dict:
    """{data_date: Series(isin -> compounded total return from snap[i] to snap[i+k])}."""
    out = {}
    for i in range(len(snaps) - k):
        sl = ret.loc[(ret.index > snap_ts[i]) & (ret.index <= snap_ts[i + k])]
        if not sl.empty:
            out[snaps[i]] = (1.0 + sl).prod(min_count=1) - 1.0
    return out


def _ic_series(score_wide: pd.DataFrame, fwd: dict, sec_map: dict | None) -> pd.Series:
    """Per-date cross-sectional rank IC of a signal vs forward return.

    score_wide: index=data_date, columns=isin, values=directional z-score.
    sec_map: if given, demean signal & forward return within GICS sector first
             (sector-neutral IC), isolating stock selection from sector tilt.
    """
    ics = {}
    for d, fr in fwd.items():
        if d not in score_wide.index:
            continue
        s = score_wide.loc[d]
        df = pd.concat([s.rename("sig"), fr.rename("fwd")], axis=1).dropna()
        if len(df) < _MIN_XS:
            continue
        if sec_map is not None:
            df["sec"] = df.index.map(sec_map)
            df = df.dropna(subset=["sec"])
            df["sig"] = df["sig"] - df.groupby("sec")["sig"].transform("mean")
            df["fwd"] = df["fwd"] - df.groupby("sec")["fwd"].transform("mean")
            if len(df) < _MIN_XS:
                continue
        ic = df["sig"].corr(df["fwd"], method="spearman")
        if pd.notna(ic):
            ics[d] = ic
    return pd.Series(ics).sort_index()


def _decile_panel(score_wide: pd.DataFrame, fwd: dict) -> tuple[pd.Series, pd.Series]:
    """Per-date (top-minus-bottom-decile forward return, monotonicity ρ).

    The spread series IS a long-short decile portfolio return per period: compound it
    for the cumulative signal curve, average it for the headline D10-D1 stat.
    """
    spreads, monos = {}, {}
    for d, fr in fwd.items():
        if d not in score_wide.index:
            continue
        df = pd.concat([score_wide.loc[d].rename("sig"), fr.rename("fwd")], axis=1).dropna()
        if len(df) < _MIN_XS:
            continue
        try:
            df["dec"] = pd.qcut(df["sig"].rank(method="first"), 10, labels=False)
        except ValueError:
            continue
        m = df.groupby("dec")["fwd"].mean()
        if len(m) < 10:
            continue
        spreads[d] = m.iloc[-1] - m.iloc[0]
        monos[d] = pd.Series(m.values).corr(pd.Series(range(10)), method="spearman")
    return pd.Series(spreads).sort_index(), pd.Series(monos)


def _persistence(score_wide: pd.DataFrame) -> float:
    """Mean rank autocorrelation of the signal across consecutive snapshots (1 = no turnover)."""
    idx = list(score_wide.index)
    acs = [score_wide.loc[a].corr(score_wide.loc[b], method="spearman")
           for a, b in zip(idx[:-1], idx[1:])]
    acs = [a for a in acs if pd.notna(a)]
    return float(np.mean(acs)) if acs else np.nan


def _window_control(snaps: list[str], prefix: str) -> tuple[str, str]:
    """A date-range control: quick presets + a slider snapped to real snapshot dates.

    Replaces the old sidebar date_input. `prefix` namespaces the widget keys so the
    same control can appear on more than one tab without colliding. Returns (lo, hi).
    """
    legacy_key = f"{prefix}_window"
    state_key = f"{prefix}_window_value"
    version_key = f"{prefix}_window_version"
    default_window = (snaps[0], snaps[-1])
    current_window = st.session_state.get(
        state_key,
        st.session_state.get(legacy_key, default_window),
    )
    if (
        not isinstance(current_window, (list, tuple))
        or len(current_window) != 2
        or current_window[0] not in snaps
        or current_window[1] not in snaps
    ):
        current_window = default_window
    st.session_state[state_key] = tuple(current_window)
    if version_key not in st.session_state:
        st.session_state[version_key] = 0

    def _on_or_after(cutoff: pd.Timestamp) -> str:
        for s in snaps:
            if pd.Timestamp(s) >= cutoff:
                return s
        return snaps[-1]

    last = pd.Timestamp(snaps[-1])
    presets = [
        ("All", snaps[0]),
        ("5Y",  _on_or_after(last - pd.DateOffset(years=5))),
        ("3Y",  _on_or_after(last - pd.DateOffset(years=3))),
        ("1Y",  _on_or_after(last - pd.DateOffset(years=1))),
        ("YTD", _on_or_after(last.replace(month=1, day=1))),
    ]
    cols = st.columns([1, 1, 1, 1, 1, 5])
    for col, (lbl, lo) in zip(cols, presets):
        if col.button(lbl, width="stretch", key=f"{prefix}_{lbl}"):
            st.session_state[state_key] = (lo, snaps[-1])
            st.session_state[version_key] += 1
    selected = st.select_slider(
        "Analysis window",
        options=snaps,
        value=st.session_state[state_key],
        key=f"{prefix}_window_slider_{st.session_state[version_key]}",
        format_func=lambda d: pd.Timestamp(d).strftime("%b %Y"),
    )
    if not isinstance(selected, (list, tuple)) or len(selected) != 2:
        st.session_state[state_key] = default_window
        return default_window
    st.session_state[state_key] = (selected[0], selected[1])
    return selected[0], selected[1]


@st.cache_data(show_spinner="Computing signal diagnostics …")
def compute_signal_diagnostics(
    date_lo: str, date_hi: str, sectors: tuple, source: str = "model"
) -> dict:
    """All-signal predictive-power diagnostics over the selected window/sectors.

    Returns dict of: summary (DataFrame), decay (DataFrame: signal × horizon),
    ic_series (DataFrame: date × signal, horizon-1 IC), ppy (float), n_snaps (int).
    Works for either composed models (source='model') or raw factors
    (source='factor'), since both expose a directional ``model_value_z``.
    """
    if source == "factor":
        labels  = factor_label_map()
        ordered = list(labels.keys())
    else:
        meta    = db.get_model_metadata()
        labels  = dict(zip(meta["ModelID"], meta["Model"]))
        ordered = (["ALP001"] +
                   [m for m in meta.sort_values("ModelID")["ModelID"] if m != "ALP001"])

    scores = _load_scores(source)
    scores = scores[(scores["data_date"] >= date_lo) & (scores["data_date"] <= date_hi)]
    ret    = load_returns_matrix()

    uni = db.get_universe()[["security_id", "gics_sector"]].dropna()
    sec_map = dict(zip(uni["security_id"], uni["gics_sector"]))
    if sectors:
        keep = set(uni[uni["gics_sector"].isin(sectors)]["security_id"])
        scores = scores[scores["security_id"].isin(keep)]

    snaps = sorted(scores["data_date"].unique())
    if len(snaps) < 6:
        return {"summary": pd.DataFrame(), "decay": pd.DataFrame(),
                "ic_series": pd.DataFrame(), "ls_curves": pd.DataFrame(),
                "ppy": np.nan, "n_snaps": len(snaps)}
    snap_ts = [pd.Timestamp(s) for s in snaps]
    ppy = _periods_per_year(snaps)

    fwd_by_h = {h: _forward_returns(ret, snap_ts, snaps, h) for h in DECAY_HORIZONS}

    # one wide z-score frame (date × isin) per model, computed once and reused
    wides = {mid: g.pivot_table(index="data_date", columns="security_id",
                                values="model_value_z")
             for mid, g in scores.groupby("model_id")}

    summary_rows, decay_rows, ic_h1, ls_curves = [], [], {}, {}
    for mid in ordered:
        sw = wides.get(mid)
        if sw is None or sw.empty:
            continue
        s_raw = _ic_series(sw, fwd_by_h[1], None)
        if s_raw.empty:
            continue
        label = f"{labels.get(mid, mid)} ({mid})"
        ic_h1[mid] = s_raw
        s_neu = _ic_series(sw, fwd_by_h[1], sec_map)
        sd = s_raw.std()
        spread_s, mono_s = _decile_panel(sw, fwd_by_h[1])
        if not spread_s.empty:
            # cumulative long-short decile P&L, indexed at the close of each period
            ls_curves[label] = (1.0 + spread_s).cumprod()
        summary_rows.append({
            "Signal": label,
            "n": len(s_raw),
            "IC": round(s_raw.mean(), 4),
            "IC-IR": round(s_raw.mean() / sd * np.sqrt(ppy), 2) if sd else np.nan,
            "t-stat": round(s_raw.mean() / sd * np.sqrt(len(s_raw)), 2) if sd else np.nan,
            "hit %": round(100 * (s_raw > 0).mean(), 0),
            "neutral IC": round(s_neu.mean(), 4) if not s_neu.empty else np.nan,
            "D10-D1": round(spread_s.mean(), 4) if not spread_s.empty else np.nan,
            "monotonic": round(float(mono_s.mean()), 2) if not mono_s.empty else np.nan,
            "persist": round(_persistence(sw), 2),
        })
        drow = {"Signal": label}
        for h in DECAY_HORIZONS:
            si = _ic_series(sw, fwd_by_h[h], None)
            drow[f"h={h}"] = round(si.mean(), 4) if not si.empty else np.nan
        decay_rows.append(drow)

    ls_df = pd.DataFrame(ls_curves)
    if not ls_df.empty:
        ls_df.index = pd.to_datetime(ls_df.index)
        ls_df = ls_df.sort_index()

    return {
        "summary":   pd.DataFrame(summary_rows),
        "decay":     pd.DataFrame(decay_rows),
        "ic_series": pd.DataFrame(ic_h1),
        "ls_curves": ls_df,
        "ppy":       ppy,
        "n_snaps":   len(snaps),
    }


# ---------------------------------------------------------------------------
# Optimised backtest helpers
# ---------------------------------------------------------------------------

def _run_optimised_backtest(
    strategy_id: str,
    portfolio_eur: float,
    max_turnover: float,
    tc_per_trade_eur: float,
    benchmark_name: str,
    universe_name: str = "sp500",
    rebal_freq: str = "quarterly",
    min_pos_if_held: float | None = None,
    max_positions_override: int | None = None,
    solver: str = "CLARABEL",
) -> dict:
    """Streamlit-cached wrapper over scripts.backtest_engine.run_optimised_backtest.

    The walk-forward simulation lives in the shared engine so the interactive page
    and the standalone HTML report (scripts/backtest_report.py) stay in lockstep.
    Cached on the full parameter set: an identical configuration returns instantly.
    """
    return _cached_optimised_backtest(
        strategy_id, portfolio_eur, max_turnover, tc_per_trade_eur, benchmark_name,
        universe_name, rebal_freq, min_pos_if_held, max_positions_override, solver,
    )


@st.cache_data(show_spinner=False, max_entries=8)
def _cached_optimised_backtest(*args) -> dict:
    return _run_optimised_backtest_impl(*args)


# ---------------------------------------------------------------------------
# Optimised backtest — export
# ---------------------------------------------------------------------------

def _build_backtest_workbook(
    result: dict,
    bench_series: pd.Series,
    bench_label: str,
    portfolio_eur: float,
) -> bytes:
    """
    Assemble an optimised-backtest result into a multi-sheet Excel workbook
    (in memory) for offline analysis. Sheets:

      summary           — run parameters + date range
      daily_returns     — portfolio vs benchmark daily & cumulative returns (%)
      periods           — per-rebalance turnover, costs and optimiser metrics
      holdings          — every position held at each rebalance
      factor_exposures  — Barra style/beta exposure (active + absolute) per rebalance

    The factor-exposure maths reuses db.* so the export matches the on-screen
    charts exactly.
    """
    port_series  = result["port_series"]
    period_log   = result["period_log"]
    ticker_map   = result["ticker_map"]
    name_map     = result["name_map"]
    sector_map   = result["sector_map"]
    industry_map = result["industry_map"]

    bench  = bench_series.reindex(port_series.index).fillna(0.0)
    active = port_series - bench
    cum_port  = ((1 + port_series).cumprod() - 1) * 100
    cum_bench = ((1 + bench).cumprod() - 1) * 100
    daily_df = pd.DataFrame({
        "date":              port_series.index,
        "portfolio_return":  port_series.values,
        "benchmark_return":  bench.values,
        "active_return":     active.values,
        "cum_portfolio_pct": cum_port.values,
        "cum_benchmark_pct": cum_bench.values,
        "cum_excess_pct":    (cum_port - cum_bench).values,
    })

    period_rows = []
    for p in period_log:
        m = p.get("metrics", {}) or {}
        period_rows.append({
            "snap_date":         p["snap_date"][:10],
            "next_snap":         p.get("next_snap", ""),
            "turnover_2way_pct": round(p["turnover"] * 100, 2),
            "n_positions":       p["n_positions"],
            "n_trades":          p["n_trades"],
            "tc_eur":            round(p["tc_pct"] * portfolio_eur, 2),
            "expected_alpha":    m.get("expected_alpha"),
            "portfolio_vol":     m.get("portfolio_vol"),
            "active_risk":       m.get("active_risk"),
            "sharpe_ratio":      m.get("sharpe_ratio"),
            "info_ratio":        m.get("info_ratio"),
            "used_barra":        p["used_barra"],
            "turnover_relaxed":  p.get("turnover_relaxed", False),
            "alpha_date":        p.get("alpha_date", ""),
            "barra_date":        p.get("barra_date", ""),
        })
    periods_df = pd.DataFrame(period_rows)

    holding_rows = []
    for p in period_log:
        snap = p["snap_date"][:10]
        for isin, w in sorted(p["weights"].items(), key=lambda x: -x[1]):
            holding_rows.append({
                "snap_date": snap,
                "isin":      isin,
                "ticker":    ticker_map.get(isin, isin),
                "company":   name_map.get(isin, ""),
                "sector":    sector_map.get(isin, "Unknown"),
                "industry":  industry_map.get(isin, ""),
                "weight":    round(w, 6),
                "value_eur": round(w * portfolio_eur, 2),
            })
    holdings_df = pd.DataFrame(holding_rows)

    exp_rows = []
    for p in period_log:
        bdate = p.get("barra_date")
        bm    = p.get("bm_weights") or {}
        if not bdate:
            continue
        comps = db.load_barra_components(bdate)
        if comps is None:
            continue
        _F, fnames, X_df, _delta = comps
        port_exp = db.weighted_factor_exposures(X_df, p["weights"])
        bench_exp = db.weighted_factor_exposures(X_df, bm) if bm else None
        for f in db.barra_style_factor_names(fnames):
            exp_rows.append({
                "snap_date":         p["snap_date"][:10],
                "factor":            db.barra_factor_label(f),
                "absolute_exposure": round(float(port_exp.get(f, 0.0)), 4),
                "active_exposure": (
                    round(float(port_exp.get(f, 0.0) - bench_exp.get(f, 0.0)), 4)
                    if bench_exp is not None else None
                ),
            })
    exposures_df = pd.DataFrame(exp_rows)

    summary_df = pd.DataFrame(
        [
            ("Strategy",             result.get("strategy_name", "")),
            ("Objective",            result.get("objective", "")),
            ("Universe",             result.get("universe_name", "")),
            ("Benchmark",            result.get("benchmark_name", bench_label)),
            ("Rebalancing",          result.get("rebal_freq", "")),
            ("Periods",              len(period_log)),
            ("Portfolio size (EUR)", portfolio_eur),
            ("Start",                str(port_series.index.min().date())),
            ("End",                  str(port_series.index.max().date())),
        ],
        columns=["metric", "value"],
    )

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        summary_df.to_excel(xl, sheet_name="summary", index=False)
        daily_df.to_excel(xl, sheet_name="daily_returns", index=False)
        periods_df.to_excel(xl, sheet_name="periods", index=False)
        holdings_df.to_excel(xl, sheet_name="holdings", index=False)
        if not exposures_df.empty:
            exposures_df.to_excel(xl, sheet_name="factor_exposures", index=False)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Shared metadata (controls now live in-page, per tab — no sidebar)
# ---------------------------------------------------------------------------

model_meta    = db.get_model_metadata()
model_options = {f"{r['Model']} ({r['ModelID']})": r["ModelID"]
                 for _, r in model_meta.iterrows()}
all_sectors   = sorted(db.get_universe()["sector"].dropna().unique())


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["Signal Backtest", "Optimised Backtest", "Signal Diagnostics"])


# ===========================================================================
# Tab 3 — Signal Diagnostics (holistic, all signals at once)
# ===========================================================================

with tab3:
    st.caption(
        "Predictive power of every signal's z-score vs realised forward returns — "
        "**before** the optimizer and its constraints. This isolates *does the signal "
        "forecast* from *does the optimized book make money*. Cross-sectional rank IC "
        "(Spearman), direction-adjusted, over the window and sectors selected below."
    )

    diag_source_label = st.segmented_control(
        "Signal type", ["Models", "Factors"], default="Models", key="diag_source",
        help="Models = composed alpha signals (models.db). Factors = the individual raw "
             "factors (factors.db) that feed them, direction-adjusted — use this to see "
             "which factors carry the IC and where models can improve.",
    )
    diag_source = "factor" if diag_source_label == "Factors" else "model"

    sel_sectors_diag = st.multiselect("Sector filter", all_sectors,
                                      placeholder="All sectors", key="diag_sectors")
    all_snaps = sorted(_load_scores(diag_source)["data_date"].unique())
    _lo, _hi = _window_control(all_snaps, "diag")

    diag = compute_signal_diagnostics(
        _lo, _hi, tuple(sorted(sel_sectors_diag)), source=diag_source
    )

    if diag["summary"].empty:
        st.warning("Not enough snapshots in the selected window/sectors to compute IC.")
    else:
        ppy = diag["ppy"]
        st.markdown(
            f"**{diag['n_snaps']} snapshots** · {_lo} → {_hi} · median spacing ≈ "
            f"{365.25 / ppy:.0f} days (annualisation ≈ ×{ppy:.0f})"
        )

        # ---- 0. Cumulative long-short performance (headline) -------------
        ls = diag["ls_curves"]
        if not ls.empty:
            st.subheader("Cumulative signal performance")
            st.caption(
                "Each line compounds a **long top-decile / short bottom-decile** portfolio "
                "for that signal, rebalanced every snapshot — the economic payoff behind the "
                "IC. Gross of costs and constraints (raw signal, not the optimized book), "
                "re-based to 1.0 at the window start."
            )
            fig_ls = go.Figure()
            for sig in ls.columns:
                s = ls[sig].dropna()
                if s.empty:
                    continue
                s = s / s.iloc[0]
                emph = sig.endswith("(ALP001)")
                fig_ls.add_trace(go.Scatter(
                    x=s.index, y=s.values, name=sig.split(" (")[0], mode="lines",
                    line=dict(width=3 if emph else 1.2,
                              color="#111827" if emph else None),
                ))
            fig_ls.add_hline(y=1.0, line_dash="dot", line_color="#9ca3af")
            fig_ls.update_layout(
                height=460, yaxis_title="growth of 1.0 (long-short)",
                hovermode="x unified", margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(font=dict(size=10)),
            )
            st.plotly_chart(fig_ls, width="stretch")

        # ---- 1. Cross-signal IC scorecard --------------------------------
        st.subheader("Signal scorecard")
        st.caption(
            "One row per signal at the 1-snapshot horizon. **IC** = mean rank correlation "
            "(0.02–0.05 is a strong equity factor; **t-stat > 2** ≈ reliable). "
            "**neutral IC** strips the GICS-sector tilt — a big drop means the signal is "
            "mostly a sector bet the optimizer neutralises away. **D10-D1** = top-minus-"
            "bottom decile forward return; **monotonic** (−1…1) checks the deciles line up. "
            "**persist** = rank autocorrelation (low = high turnover)."
        )
        summ = diag["summary"]

        def _hl_ic(v):
            try:
                v = float(v)
            except (TypeError, ValueError):
                return ""
            if v >= 0.03:
                return "background-color: rgba(34,197,94,0.18)"
            if v <= 0:
                return "background-color: rgba(239,68,68,0.18)"
            return ""

        def _hl_t(v):
            try:
                return "font-weight: 700" if abs(float(v)) >= 2 else ""
            except (TypeError, ValueError):
                return ""

        styled = (summ.style
                  .map(_hl_ic, subset=["IC", "neutral IC"])
                  .map(_hl_t, subset=["t-stat"])
                  .format({"IC": "{:.4f}", "neutral IC": "{:.4f}", "IC-IR": "{:.2f}",
                           "t-stat": "{:.2f}", "hit %": "{:.0f}", "D10-D1": "{:.4f}",
                           "monotonic": "{:.2f}", "persist": "{:.2f}"}))
        st.dataframe(styled, width="stretch", hide_index=True)

        c1, c2 = st.columns(2)

        # ---- 2. IC decay across horizons ---------------------------------
        with c1:
            st.subheader("IC decay")
            st.caption(
                "Mean IC as the forecast horizon lengthens. Lines that **rise** = slow, "
                "long-horizon signals (hold them; monthly churn wastes them); lines that "
                "**fade** = fast signals. Overlapping windows inflate the far-right points, "
                "so read the shape, not the absolute level."
            )
            decay = diag["decay"].set_index("Signal")
            fig_d = go.Figure()
            for sig in decay.index:
                emph = sig.endswith("(ALP001)")
                fig_d.add_trace(go.Scatter(
                    x=[h.replace("h=", "") for h in decay.columns],
                    y=decay.loc[sig].values, name=sig.split(" (")[0],
                    mode="lines+markers",
                    line=dict(width=3 if emph else 1.3,
                              color="#111827" if emph else None),
                ))
            fig_d.update_layout(
                height=420, xaxis_title="forward horizon (snapshots)", yaxis_title="mean IC",
                margin=dict(l=0, r=0, t=10, b=0), legend=dict(font=dict(size=10)),
            )
            fig_d.add_hline(y=0, line_dash="dot", line_color="#9ca3af")
            st.plotly_chart(fig_d, width="stretch")

        # ---- 3. Cross-signal IC correlation ------------------------------
        with c2:
            st.subheader("Signal overlap")
            st.caption(
                "Correlation of the per-date IC series across signals. **High positive** = "
                "two signals make the same forecast, so blending them adds no breadth "
                "(IR = IC·√breadth). Look for low/negative pairs — those are the real "
                "diversifiers worth weighting up."
            )
            ics = diag["ic_series"].drop(columns=["ALP001"], errors="ignore").dropna()
            if len(ics) >= 6:
                corr = ics.corr(method="spearman")
                if diag_source == "factor":
                    _fl   = factor_label_map()
                    short = {c: str(_fl.get(c, c))[:11] for c in corr.columns}
                else:
                    _mm   = db.get_model_metadata().set_index("ModelID")
                    short = {c: str(_mm.loc[c, "Model"])[:11] for c in corr.columns}
                corr = corr.rename(index=short, columns=short)
                fig_c = go.Figure(go.Heatmap(
                    z=corr.values, x=list(corr.columns), y=list(corr.index),
                    zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
                    text=corr.round(2).values, texttemplate="%{text}",
                    textfont=dict(size=9), colorbar=dict(title="ρ"),
                ))
                fig_c.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0),
                                    yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_c, width="stretch")
            else:
                st.info(f"Too few overlapping dates ({len(ics)}) for a stable correlation matrix.")



# ===========================================================================
# Tab 1 — Signal Backtest
# ===========================================================================

with tab1:
    st.caption(
        "Rank stocks by model score at each snapshot and hold equal-weight percentile buckets. "
        "Long-only holds the top bucket; long-short holds top bucket minus bottom bucket. "
        "Universe membership is carried forward from the latest available index snapshot. "
        "Pre-computed daily total returns; no transaction costs."
    )

    # ── In-page controls (replaces the old sidebar) ───────────────────────────
    signal_universes = [
        u for u in ("russell_1000", "sp500")
        if u in db.get_available_universe_indices()
    ]
    if not signal_universes:
        signal_universes = db.get_available_universe_indices()
    if not signal_universes:
        st.warning("No universe snapshots found. Run the universe snapshot pipeline first.")
        st.stop()

    bt_source_label = st.segmented_control(
        "Signal type", ["Models", "Factors"], default="Models", key="bt_source",
        help="Models = composed alpha signals (models.db). Factors = individual raw "
             "factors (factors.db), direction-adjusted. Factor coverage is sparser and "
             "some factors are sector-gated (banks/REITs), so buckets will be smaller.",
    )
    bt_source = "factor" if bt_source_label == "Factors" else "model"
    if bt_source == "factor":
        _flabels    = factor_label_map()
        bt_options  = {f"{name} ({fid})": fid for fid, name in _flabels.items()}
    else:
        bt_options  = model_options

    cc = st.columns([2.6, 1.5, 1.4, 1.4, 2.2])
    with cc[0]:
        sel_model_label = st.selectbox(
            "Factor" if bt_source == "factor" else "Model",
            list(bt_options.keys()), key=f"bt_signal_{bt_source}",
        )
        sel_model_id    = bt_options[sel_model_label]
    with cc[1]:
        default_signal_uni = "russell_1000" if "russell_1000" in signal_universes else signal_universes[0]
        sel_signal_universe = st.selectbox(
            "Universe",
            signal_universes,
            index=signal_universes.index(default_signal_uni),
            format_func=lambda u: u.replace("_", " ").title(),
            key="bt_universe",
        )
    with cc[2]:
        portfolio_mode = st.segmented_control(
            "Portfolio", ["Long-only", "Long-short"], default="Long-only", key="bt_mode"
        )
    with cc[3]:
        bucket_pct = st.slider("Bucket percentile", 5, 40, 20, step=5, key="bt_bucket_pct")
    with cc[4]:
        sel_sectors = st.multiselect("Sector filter", all_sectors,
                                     placeholder="All sectors", key="bt_sectors")

    signal_window_snaps = [
        d for d in sorted(_load_scores(bt_source)["data_date"].unique())
        if d >= SIGNAL_BACKTEST_START
    ]
    date_range = _window_control(signal_window_snaps, "bt")
    st.divider()

    with st.spinner("Running backtest…"):
        long_s, short_s, strategy_s, benchmark, holdings_log, err = run_backtest(
            sel_model_id, portfolio_mode, bucket_pct, sel_signal_universe, sel_sectors,
            source=bt_source,
        )

    if err:
        st.warning(err)
        st.stop()

    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        d_start   = pd.Timestamp(date_range[0])
        d_end     = pd.Timestamp(date_range[1])
        long_s    = long_s.loc[d_start:d_end]
        strategy_s = strategy_s.loc[d_start:d_end]
        benchmark = benchmark.loc[d_start:d_end]
        if short_s is not None:
            short_s = short_s.loc[d_start:d_end]
        holdings_log = _filter_holdings_log(holdings_log, d_start, d_end)
    else:
        d_start = long_s.index[0]
        d_end   = long_s.index[-1]

    universe_label = sel_signal_universe.replace("_", " ").title()
    model_short_name = sel_model_label.split(" (")[0]
    strategy_label = (
        f"Top {bucket_pct}% — {model_short_name} ({universe_label})"
        if portfolio_mode == "Long-only"
        else f"Top {bucket_pct}% - bottom {bucket_pct}% — {model_short_name} ({universe_label})"
    )
    long_label  = f"Top {bucket_pct}% basket"
    short_label = f"Bottom {bucket_pct}% basket (held long)"
    benchmark_label = f"EW {universe_label}"
    cum_bench   = (1 + benchmark).cumprod()   # reused in quintile chart

    include_short = portfolio_mode == "Long-short"
    strategy_cum_s = (
        _cumulative_spread_series(long_s, short_s)
        if include_short and short_s is not None else None
    )
    turnover_rows = _compute_turnover(holdings_log, include_short)
    turnover_rows_display = _aggregate_monthly_turnover(turnover_rows)[1:]
    avg_turnover  = (
        np.mean([r["Strategy monthly turnover (%)"] for r in turnover_rows_display])
        if turnover_rows_display else None
    )

    # Cumulative return chart
    cum_traces = [{
        "series": strategy_cum_s if strategy_cum_s is not None else strategy_s,
        "name": strategy_label,
        "color": "#2563EB",
        "cumulative": strategy_cum_s is not None,
    }]
    if short_s is not None:
        cum_traces.append({"series": long_s, "name": long_label,
                           "color": "#16A34A", "width": 1.3, "dash": "dot"})
        cum_traces.append({"series": short_s, "name": short_label,
                           "color": "#DC2626", "dash": "dash"})
    cum_traces.append({"series": benchmark, "name": benchmark_label,
                       "color": "#94A3B8", "width": 1.5, "dash": "dot"})
    st.plotly_chart(_cum_return_chart(cum_traces), width="stretch")

    # Drawdown chart
    dd_traces = [
        {"series": strategy_cum_s if strategy_cum_s is not None else strategy_s,
         "name": strategy_label, "color": "#2563EB",
         "cumulative": strategy_cum_s is not None,
         "fill_color": "rgba(37,99,235,0.08)"},
    ]
    if strategy_cum_s is None:
        dd_traces.append({"series": benchmark, "name": benchmark_label,
                          "color": "#94A3B8", "width": 1, "dash": "dot"})
    st.plotly_chart(_drawdown_chart(dd_traces), width="stretch")

    # Metrics
    st.divider()
    st.subheader("Performance summary")
    strategy_metrics = (
        spread_metrics(long_s, short_s, strategy_s)
        if strategy_cum_s is not None else perf_metrics(strategy_s)
    )
    if portfolio_mode == "Long-only":
        strategy_metrics = {
            **strategy_metrics,
            **active_metrics(
                strategy_s,
                benchmark,
                avg_turnover,
                turnover_label="Avg monthly turnover (2-way)",
            ),
        }
    elif avg_turnover is not None:
        strategy_metrics["Avg monthly turnover (2-way)"] = f"{avg_turnover:.0f}%"

    metric_sets = {"Strategy": strategy_metrics}
    if short_s is not None:
        metric_sets[f"Top {bucket_pct}%"] = perf_metrics(long_s)
        metric_sets[f"Bottom {bucket_pct}% (held long)"] = perf_metrics(short_s)
    metric_sets[benchmark_label] = perf_metrics(benchmark)
    _metrics_comparison_table(metric_sets)

    # Returns chart
    st.divider()
    st.subheader("Returns")
    ret_view = st.segmented_control(
        "View", ["Annual", "Monthly heatmap"], default="Annual", key="ret_view"
    )

    if ret_view == "Annual":
        if strategy_cum_s is not None:
            annual_spread = (
                long_s.resample("YE").apply(lambda x: (1 + x).prod() - 1)
                - short_s.resample("YE").apply(lambda x: (1 + x).prod() - 1)
            )
            fig_ann = go.Figure()
            fig_ann.add_trace(go.Bar(
                x=annual_spread.index.year.astype(str),
                y=annual_spread.values,
                name="Strategy",
                marker_color="#2563EB",
                text=[f"{v:+.1%}" if pd.notna(v) else "" for v in annual_spread.values],
                textposition="outside",
            ))
            for t in [
                {"series": long_s, "name": "Top bucket", "color": "#16A34A"},
                {"series": short_s, "name": "Bottom bucket held long", "color": "#DC2626"},
                {"series": benchmark, "name": benchmark_label, "color": "#94A3B8"},
            ]:
                ann = t["series"].resample("YE").apply(lambda x: (1 + x).prod() - 1)
                fig_ann.add_trace(go.Bar(
                    x=ann.index.year.astype(str), y=ann.values,
                    name=t["name"], marker_color=t["color"],
                    text=[f"{v:+.1%}" if pd.notna(v) else "" for v in ann.values],
                    textposition="outside",
                ))
            fig_ann.update_layout(
                barmode="group", height=320, yaxis_tickformat=".0%",
                margin=dict(l=0, r=0, t=20, b=20),
            )
            st.plotly_chart(fig_ann, width="stretch")
        else:
            annual_traces = [{"series": strategy_s, "name": "Strategy", "color": "#2563EB"}]
            annual_traces.append({"series": benchmark, "name": benchmark_label, "color": "#94A3B8"})
            st.plotly_chart(_annual_bar_chart(annual_traces), width="stretch")
    else:
        MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        if strategy_cum_s is not None:
            monthly = (
                long_s.resample("ME").apply(lambda x: (1 + x).prod() - 1)
                - short_s.resample("ME").apply(lambda x: (1 + x).prod() - 1)
            )
        else:
            monthly = strategy_s.resample("ME").apply(lambda x: (1 + x).prod() - 1)
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
        st.plotly_chart(fig_heat, width="stretch")
        st.caption("Selected strategy monthly returns. Green = positive, red = negative.")

    # Rolling metrics
    st.divider()
    st.subheader("Rolling metrics")
    roll_options = ["Rolling Sharpe (1Y)"]
    if portfolio_mode == "Long-only":
        roll_options.append("Rolling Information Ratio (1Y)")
    roll_choice = st.selectbox(
        "Metric",
        roll_options,
        key=f"roll_choice_{portfolio_mode}",
    )
    ROLL_WINDOW = 252

    if roll_choice == "Rolling Sharpe (1Y)":
        def _rolling_sharpe(s):
            hurdle = RISK_FREE if portfolio_mode == "Long-only" else 0.0
            return (s.rolling(ROLL_WINDOW).mean() * ROLL_WINDOW - hurdle) / \
                   (s.rolling(ROLL_WINDOW).std() * ROLL_WINDOW ** 0.5)
        fig_roll = go.Figure()
        fig_roll.add_trace(go.Scatter(x=(rs := _rolling_sharpe(strategy_s).dropna()).index,
                                      y=rs.values, name=strategy_label,
                                      line=dict(color="#2563EB", width=2)))
        if portfolio_mode == "Long-only":
            fig_roll.add_trace(go.Scatter(x=(rb := _rolling_sharpe(benchmark).dropna()).index,
                                          y=rb.values, name=benchmark_label,
                                          line=dict(color="#94A3B8", width=1.5, dash="dot")))
        fig_roll.add_hline(y=0, line_dash="dot", line_color="#64748B", line_width=1)
        fig_roll.update_layout(height=300, yaxis_title="Sharpe ratio (1Y rolling)",
                               hovermode="x unified", legend=dict(orientation="h", y=-0.2),
                               margin=dict(l=0, r=0, t=10, b=10))
        st.plotly_chart(fig_roll, width="stretch")

    elif roll_choice == "Rolling Information Ratio (1Y)":
        active  = strategy_s.subtract(benchmark.reindex(strategy_s.index, fill_value=0))
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
        st.plotly_chart(fig_roll, width="stretch")
        st.caption(
            "IR > 0.5 (green) = consistently adding active return. "
            f"Below zero = model underperformed {benchmark_label} on a risk-adjusted basis."
        )

    st.divider()
    st.subheader("Monthly two-way turnover")
    if not turnover_rows_display:
        st.info("Need at least 2 calendar months to show turnover after excluding the first month.")
    else:
        to_df  = pd.DataFrame(turnover_rows_display)
        fig_to = go.Figure()
        fig_to.add_trace(go.Bar(
            x=to_df["Period"], y=to_df["Strategy monthly turnover (%)"],
            name="Strategy", marker_color="#2563EB",
            text=[f"{v:.0f}%" for v in to_df["Strategy monthly turnover (%)"]],
            textposition="outside",
            customdata=to_df["Rebalances"],
            hovertemplate=(
                "%{x}<br>Monthly turnover: %{y:.1f}%"
                "<br>Rebalances in month: %{customdata}<extra></extra>"
            ),
        ))
        fig_to.add_trace(go.Bar(
            x=to_df["Period"], y=to_df["Benchmark monthly turnover (%)"],
            name=benchmark_label, marker_color="#94A3B8",
            text=[f"{v:.0f}%" if pd.notna(v) else "" for v in to_df["Benchmark monthly turnover (%)"]],
            textposition="outside",
            customdata=to_df["Rebalances"],
            hovertemplate=(
                "%{x}<br>Monthly turnover: %{y:.1f}%"
                "<br>Rebalances in month: %{customdata}<extra></extra>"
            ),
        ))
        y_max = max(
            to_df["Strategy monthly turnover (%)"].max(),
            to_df["Benchmark monthly turnover (%)"].max(skipna=True),
        )
        fig_to.update_layout(barmode="group", height=320, yaxis_title="Monthly two-way turnover (%)",
                             yaxis_range=[0, max(20, y_max * 1.2)],
                             margin=dict(l=0, r=0, t=10, b=10))
        st.plotly_chart(fig_to, width="stretch")
        avg_bench_turnover = to_df["Benchmark monthly turnover (%)"].mean()
        avg_text = f"Average strategy: **{avg_turnover:.0f}%**."
        bench_text = f" Average benchmark: **{avg_bench_turnover:.0f}%**." if pd.notna(avg_bench_turnover) else ""
        st.caption(
            "Monthly two-way turnover sums 0.5 × total absolute weight change across all rebalances in each calendar month. "
            "A fully replaced book is 100%, not 200%. "
            "The first calendar month is excluded. "
            f"{avg_text}{bench_text}"
        )

    # Quintile analysis
    st.divider()
    st.subheader("Quintile analysis")
    st.caption(
        "Universe split into 5 equal buckets by model score at each snapshot. "
        "Q1 = highest-scored stocks, Q5 = lowest. A working factor shows Q1 > Q2 > … > Q5."
    )
    with st.spinner("Computing quintiles…"):
        q_series = run_quintile_analysis(
            sel_model_id, sel_signal_universe, sel_sectors, source=bt_source
        )
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
                             annotation_text=f"{benchmark_label} {bench_ann:+.1%}",
                             annotation_position="right")
            fig_qa.update_layout(height=340, yaxis_tickformat=".0%",
                                 yaxis_title="Annualised return",
                                 margin=dict(l=0, r=80, t=20, b=10))
            st.plotly_chart(fig_qa, width="stretch")

    with q_tab2:
        fig_qc = go.Figure()
        for i, s in enumerate(q_series):
            if not s.empty:
                cum = (1 + s).cumprod()
                fig_qc.add_trace(go.Scatter(x=cum.index, y=cum.values, name=Q_LABELS[i],
                                            line=dict(color=Q_COLORS[i], width=2)))
        fig_qc.add_trace(go.Scatter(x=cum_bench.index, y=cum_bench.values, name=benchmark_label,
                                    line=dict(color="#94A3B8", width=1.5, dash="dot")))
        fig_qc.update_layout(height=380, yaxis_title="Portfolio value (base = 1.0)",
                             hovermode="x unified", legend=dict(orientation="h", y=-0.15),
                             margin=dict(l=0, r=0, t=20, b=10))
        st.plotly_chart(fig_qc, width="stretch")

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
        st.markdown(f"**Top {bucket_pct}% bucket — {len(long_df)} stocks**")
        st.dataframe(long_df, hide_index=True, width="stretch", column_config={
            "Rank":       st.column_config.NumberColumn(width="small"),
            "Score":      st.column_config.NumberColumn(format="%.3f", width="small"),
            "Price data": st.column_config.TextColumn(width="small"),
        })
    with rcol:
        st.plotly_chart(_sector_chart(sel_period["long"], "Sector breakdown — top bucket", "#2563EB"),
                        width="stretch")

    short_df = sel_period.get("short")
    if short_df is not None and not short_df.empty:
        st.markdown("---")
        slcol, srcol = st.columns([3, 2])
        with slcol:
            st.markdown(f"**Bottom {bucket_pct}% bucket — {len(short_df)} stocks**")
            st.dataframe(short_df, hide_index=True, width="stretch", column_config={
                "Rank":       st.column_config.NumberColumn(width="small"),
                "Score":      st.column_config.NumberColumn(format="%.3f", width="small"),
                "Price data": st.column_config.TextColumn(width="small"),
            })
        with srcol:
            st.plotly_chart(_sector_chart(short_df, "Sector breakdown — bottom bucket", "#DC2626"),
                            width="stretch")


# ===========================================================================
# Tab 2 — Optimised Backtest
# ===========================================================================

with tab2:
    st.caption(
        "Walk-forward optimiser using CVXPY.  "
        "Monthly mode trades on the first available trading day of each month and carries forward "
        "the latest scheduled monthly/weekly alpha and risk snapshot, excluding ad-hoc research dates.  "
        "Risk model: Barra (K=29 factors, LW fallback).  "
        f"Transaction cost: €{TC_EUR:.0f} per order; orders counted only when trade value ≥ €{TC_EUR/0.01:.0f} (~1% commission ratio).  "
        "Turnover limit is two-way (buys + sells): a fully replaced 35-stock portfolio ≈ 100%.  "
        "Note: **realized TE** will exceed the strategy's ex-ante TE constraint — the optimizer "
        "targets ex-ante Barra TE at each rebalance; drift and index composition changes accumulate between rebalances."
    )

    # ── Controls ──────────────────────────────────────────────────────────────
    available_indices  = db.get_available_benchmark_indices()
    available_universes = db.get_available_universe_indices()

    if not PARAMS_FILE.exists():
        st.warning("strategy_params.xlsx not found. Run `create_strategy_params.py` first.")
        st.stop()

    xl = pd.ExcelFile(PARAMS_FILE)
    strats_df      = pd.read_excel(xl, sheet_name="Strategies", dtype=str)
    strats_df      = strats_df[strats_df["active"].str.strip().str.upper() == "TRUE"]
    strategy_opts  = {row["name"].strip(): row["strategy_id"].strip()
                      for _, row in strats_df.iterrows()}

    # Row 1 — strategy / universe / benchmark selection
    r1c1, r1c2, r1c3, r1c4 = st.columns([2.5, 1.5, 1.5, 1])
    with r1c1:
        sel_strat_name = st.selectbox("Strategy", list(strategy_opts.keys()), key="opt_bt_strat")
        sel_strat_id   = strategy_opts[sel_strat_name]
    sel_strat_row = strats_df[strats_df["strategy_id"].str.strip() == sel_strat_id].iloc[0]
    default_solver = str(sel_strat_row.get("solver", "CLARABEL") or "CLARABEL").strip().upper()
    with r1c2:
        default_uni = "russell_1000" if "russell_1000" in available_universes else (available_universes[0] if available_universes else "sp500")
        sel_universe = st.selectbox(
            "Universe", available_universes,
            index=available_universes.index(default_uni) if default_uni in available_universes else 0,
            key="opt_bt_uni",
        )
    with r1c3:
        default_bench = "russell_1000" if "russell_1000" in available_indices else (available_indices[0] if available_indices else "")
        sel_bench = st.selectbox(
            "Benchmark", available_indices,
            index=available_indices.index(default_bench) if default_bench in available_indices else 0,
            key="opt_bt_bench",
        )
    with r1c4:
        rebal_freq = st.radio(
            "Rebalancing", ["Quarterly", "Monthly"],
            captions=["Every model snapshot", "Monthly trades"],
            key="opt_bt_freq",
        ).lower()

    # Row 2 — tuning parameters
    r2c1, r2c2, r2c3, r2c4, r2c5, r2c6 = st.columns([1.5, 1.5, 1, 1.2, 1.2, 1.2])
    with r2c1:
        portfolio_eur = st.number_input(
            "Portfolio size (€)", min_value=1_000, max_value=10_000_000,
            value=50_000, step=5_000, key="opt_bt_size",
        )
    with r2c2:
        max_to_pct = st.slider(
            "Max turnover (%)", min_value=5, max_value=100, value=30, step=5, key="opt_bt_to"
        )
        st.caption("Two-way: buys + sells")
    with r2c3:
        min_pos_opts = {"Strategy default": None, "0.25%": 0.0025, "0.5%": 0.005,
                        "1%": 0.01, "1.5%": 0.015, "2%": 0.02}
        sel_min_pos  = st.selectbox("Min position", list(min_pos_opts.keys()),
                                    index=0, key="opt_bt_minw")
        min_pos_if_held_override = min_pos_opts[sel_min_pos]
    with r2c4:
        max_pos_opts   = ["Strategy default", "50", "75", "100", "150", "200"]
        sel_max_pos    = st.selectbox("Max positions", max_pos_opts, index=0, key="opt_bt_maxpos")
        max_positions_override = None if sel_max_pos == "Strategy default" else int(sel_max_pos)
    with r2c5:
        solver_opts = ["CLARABEL", "MOSEK"]
        sel_solver = st.selectbox(
            "Solver",
            solver_opts,
            index=solver_opts.index(default_solver) if default_solver in solver_opts else 0,
            key="opt_bt_solver",
        )
    run_clicked = st.button("▶ Run Optimised Backtest", type="primary", key="opt_bt_run")

    # ── Trigger computation ───────────────────────────────────────────────────
    result_key = f"opt_bt_{sel_strat_id}_{sel_universe}_{portfolio_eur}_{max_to_pct}_{sel_bench}_{rebal_freq}_{sel_min_pos}_{sel_max_pos}_{sel_solver}"

    if run_clicked:
        # Clear stale results from other parameter combinations
        for k in [k for k in st.session_state if k.startswith("opt_bt_") and k != result_key]:
            del st.session_state[k]
        est_time = "~30–60 s" if rebal_freq == "quarterly" else "~2–3 min"
        with st.spinner(f"Running {rebal_freq} walk-forward backtest for '{sel_strat_name}'…  ({est_time})"):
            st.session_state[result_key] = _run_optimised_backtest(
                strategy_id            = sel_strat_id,
                portfolio_eur          = float(portfolio_eur),
                max_turnover           = max_to_pct / 100.0,
                tc_per_trade_eur       = TC_EUR,
                benchmark_name         = sel_bench,
                universe_name          = sel_universe,
                rebal_freq             = rebal_freq,
                min_pos_if_held        = min_pos_if_held_override,
                max_positions_override = max_positions_override,
                solver                 = sel_solver,
            )

    # ── Display results ───────────────────────────────────────────────────────
    if result_key not in st.session_state:
        st.info("Configure the strategy above and click **▶ Run Optimised Backtest** to start.")
        st.stop()

    result = st.session_state[result_key]

    if "error" in result:
        st.error(result["error"])
        st.stop()

    port_series  = result["port_series"]
    period_log   = result["period_log"]
    sector_map   = result["sector_map"]
    industry_map = result["industry_map"]
    ticker_map   = result["ticker_map"]
    name_map     = result["name_map"]

    if result["warnings"]:
        with st.expander(f"{len(result['warnings'])} warning(s)"):
            for w in result["warnings"]:
                st.caption(w)

    bench_series = db.get_benchmark_returns(sel_bench)
    if bench_series.empty:
        st.warning(f"No benchmark returns found for '{sel_bench}'.")
        bench_series = pd.Series(0.0, index=port_series.index, name=sel_bench)
    bench_label  = sel_bench.replace("_", " ").title()
    eval_port_series, eval_bench_series, bench_coverage = _align_benchmark_for_eval(
        port_series, bench_series
    )
    if eval_bench_series.empty:
        st.warning(f"No overlapping benchmark returns found for '{bench_label}' in this backtest window.")
        eval_port_series = port_series
        eval_bench_series = pd.Series(0.0, index=port_series.index, name=sel_bench)
    elif bench_coverage < 0.95:
        st.warning(
            f"Benchmark coverage for '{bench_label}' is {bench_coverage:.0%}; "
            "performance and active-risk metrics use overlapping dates only."
        )

    # Summary metrics row
    avg_to_pct_actual  = np.mean([p["turnover"] for p in period_log[1:]]) * 100 if len(period_log) > 1 else 100.0
    total_tc           = sum(p["tc_pct"] for p in period_log) * portfolio_eur
    any_barra          = any(p["used_barra"] for p in period_log)
    any_relaxed        = any(p["relaxed_integer"] for p in period_log)
    freq_label         = result.get("rebal_freq", "quarterly").capitalize()
    uni_label          = result.get("universe_name", "sp500").replace("_", " ").title()
    opt_bench_label    = result.get("benchmark_name", sel_bench).replace("_", " ").title()
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Universe", uni_label)
    m2.metric("Benchmark", opt_bench_label)
    m3.metric("Periods", f"{len(period_log)} ({freq_label})")
    m4.metric("Avg turnover (2-way)", f"{avg_to_pct_actual:.0f}%")
    m5.metric("Total TC (est.)", f"€{total_tc:,.0f}")
    m6.metric("Risk model", "Barra" if any_barra else "Ledoit-Wolf")
    if any_relaxed:
        st.info(
            "**max_positions / min_position_if_held not applied** — switch solver to MOSEK to enforce cardinality constraints."
        )

    # ── Download — full result as a multi-sheet workbook for offline analysis ──
    # Built once per result and cached in session_state (keyed on result_key) so
    # the workbook isn't re-assembled on every widget interaction / rerun.
    export_key = f"{result_key}__xlsx"
    if export_key not in st.session_state:
        export_result = {**result, "port_series": eval_port_series}
        st.session_state[export_key] = _build_backtest_workbook(
            export_result, eval_bench_series, bench_label, portfolio_eur
        )
    st.download_button(
        "⬇ Download backtest data (Excel)",
        data=st.session_state[export_key],
        file_name=f"backtest_{sel_strat_id}_{sel_universe}_{rebal_freq}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="opt_bt_download",
        help="Summary, daily returns, per-period metrics, holdings and factor exposures.",
    )

    st.divider()

    # ── Result tabs ───────────────────────────────────────────────────────────
    pt1, pt2, pt3 = st.tabs(["Performance", "Analysis", "Holdings"])

    with pt1:
        # Cumulative excess return = cumulative portfolio return − cumulative
        # benchmark return. Both start at 0, so the excess starts at 0.
        port_cum  = (1 + eval_port_series).cumprod()
        bench_cum = (1 + eval_bench_series).cumprod()
        cumulative_excess_series = (port_cum - 1) - (bench_cum - 1)

        # Always show both major US large-cap indices for context; add the
        # selected benchmark separately only when it is neither of them (so the
        # excess line's reference is always visible on the chart).
        perf_traces = [{"series": eval_port_series, "name": result["strategy_name"], "color": "#2563EB"}]
        ref_benches = [("sp500", "S&P 500", "#94A3B8"), ("russell_1000", "Russell 1000", "#F59E0B"),
                       ("sp500_equal_weight", "S&P 500 Equal Weight", "#10B981")]
        for ref_name, ref_label, ref_color in ref_benches:
            ref_s = db.get_benchmark_returns(ref_name)
            ref_s = ref_s.reindex(eval_port_series.index).dropna()
            if ref_s.empty:
                continue
            perf_traces.append({
                "series": ref_s,
                "name": ref_label, "color": ref_color, "width": 1.5, "dash": "dot",
            })
        if sel_bench not in {r[0] for r in ref_benches}:
            perf_traces.append({
                "series": eval_bench_series, "name": bench_label, "color": "#7C3AED",
                "width": 1.5, "dash": "dash",
            })

        st.plotly_chart(
            _cum_return_chart(perf_traces, cumulative_excess_series=cumulative_excess_series),
            width="stretch",
        )
        st.caption(f"Cumulative excess return is the strategy minus the selected benchmark (**{bench_label}**).")

        st.plotly_chart(_drawdown_chart([
            {"series": eval_port_series,  "name": result["strategy_name"], "color": "#2563EB",
             "fill_color": "rgba(37,99,235,0.08)"},
            {"series": eval_bench_series, "name": bench_label, "color": "#94A3B8",
             "width": 1, "dash": "dot"},
        ]), width="stretch")

        st.divider()
        st.subheader("Rolling risk")
        st.caption(
            "Tracking error and absolute risk are both annualised % on the left "
            "axis (directly comparable); beta (~1) is on the right axis. "
            "63-day rolling window."
        )
        fig_rr = _rolling_risk_chart(eval_port_series, eval_bench_series, window=63)
        if fig_rr is not None:
            st.plotly_chart(fig_rr, width="stretch")
        else:
            st.info("Not enough history for rolling risk metrics (need > 68 daily observations).")

        st.divider()
        mc1, mc2 = st.columns(2)
        with mc1:
            st.subheader(result["strategy_name"])
            _metrics_table({
                **perf_metrics(eval_port_series),
                **active_metrics(eval_port_series, eval_bench_series, avg_to_pct_actual),
            })
        with mc2:
            st.subheader(bench_label)
            _metrics_table(perf_metrics(eval_bench_series))

    with pt2:
        st.subheader("Annual returns")
        st.plotly_chart(_annual_bar_chart([
            {"series": eval_port_series,  "name": result["strategy_name"], "color": "#2563EB"},
            {"series": eval_bench_series, "name": bench_label, "color": "#94A3B8"},
        ], height=340), width="stretch")

        st.divider()
        st.subheader("Turnover per period")
        to_rows = [
            {"Period": p["snap_date"][:10], "Turnover (%)": round(p["turnover"] * 100, 1),
             "# Trades": p["n_trades"], "TC (€)": round(p["tc_pct"] * portfolio_eur, 0),
             "relaxed": p.get("turnover_relaxed", False)}
            for p in period_log
        ]
        bar_colors = ["#F59E0B" if r["relaxed"] else "#2563EB" for r in to_rows]
        fig_to = go.Figure(go.Bar(
            x=[r["Period"] for r in to_rows], y=[r["Turnover (%)"] for r in to_rows],
            marker_color=bar_colors,
            text=[f"{r['Turnover (%)']:.0f}%" for r in to_rows], textposition="outside",
        ))
        fig_to.add_hline(y=max_to_pct, line_dash="dash", line_color="#EF4444", line_width=1.5,
                         annotation_text=f"Requested limit {max_to_pct}%", annotation_position="right")
        max_to_shown = max(r["Turnover (%)"] for r in to_rows) * 1.2
        fig_to.update_layout(height=300, yaxis_title="Two-way turnover (%)",
                             yaxis_range=[0, max(max_to_pct * 1.5, max_to_shown)],
                             margin=dict(l=0, r=100, t=10, b=10))
        st.plotly_chart(fig_to, width="stretch")
        n_relaxed = sum(1 for r in to_rows if r["relaxed"])
        relaxed_note = f"  🟡 = {n_relaxed} period(s) where turnover limit was auto-relaxed to find a feasible solution." if n_relaxed else ""
        st.caption(
            f"Two-way turnover = buys + sells as % of portfolio. "
            f"First period is 100% (built from scratch). Requested constraint: {max_to_pct}% two-way.{relaxed_note}"
        )

        st.divider()
        pt2c1, pt2c2 = st.columns(2)
        with pt2c1:
            st.subheader("Position count")
            pos_rows = [{"Period": p["snap_date"][:10], "Positions": p["n_positions"]} for p in period_log]
            fig_pos = go.Figure(go.Scatter(
                x=[r["Period"] for r in pos_rows], y=[r["Positions"] for r in pos_rows],
                mode="lines+markers", line=dict(color="#2563EB", width=2), marker=dict(size=6),
            ))
            fig_pos.update_layout(height=240, yaxis_title="# positions",
                                  hovermode="x unified", margin=dict(l=0, r=20, t=10, b=10))
            st.plotly_chart(fig_pos, width="stretch")
        with pt2c2:
            st.subheader("Transaction costs")
            tc_rows = [{"Period": p["snap_date"][:10], "TC (€)": round(p["tc_pct"] * portfolio_eur, 0),
                        "Trades": p["n_trades"]} for p in period_log]
            fig_tc = go.Figure(go.Bar(
                x=[r["Period"] for r in tc_rows], y=[r["TC (€)"] for r in tc_rows],
                marker_color="#6366F1",
                text=[f"€{r['TC (€)']:.0f}" for r in tc_rows], textposition="outside",
                customdata=[r["Trades"] for r in tc_rows],
                hovertemplate="%{x}<br>TC: €%{y:,.0f}<br>Trades: %{customdata}<extra></extra>",
            ))
            fig_tc.update_layout(height=240, yaxis_title="TC (€)",
                                 margin=dict(l=0, r=20, t=10, b=10))
            st.plotly_chart(fig_tc, width="stretch")

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
            st.plotly_chart(fig_sw, width="stretch")

        st.divider()
        fe_h1, fe_h2 = st.columns([3, 1])
        with fe_h1:
            st.subheader("Factor exposure over time")
        with fe_h2:
            fe_view = st.segmented_control(
                "Exposure view", ["Active", "Absolute"], default="Active", key="opt_bt_fe_view"
            )
        st.caption(
            "Portfolio Barra style & beta exposures at each rebalance. "
            "**Active** = portfolio − benchmark (tilts vs the benchmark); "
            "**Absolute** = portfolio level. Standardised units → all factors comparable on one axis. "
            "Sector tilts are shown above."
        )
        fig_fe = _factor_exposure_chart(period_log, active=(fe_view != "Absolute"))
        if fig_fe is not None:
            st.plotly_chart(fig_fe, width="stretch")
        elif fe_view == "Active":
            st.info("Active factor exposure needs Barra snapshots and benchmark weights "
                    "(maximize_alpha strategies). Try the Absolute view.")
        else:
            st.info("Factor exposure requires Barra snapshots for the backtest periods.")

    with pt3:
        st.subheader("Holdings snapshot")
        h3c1, h3c2 = st.columns([2, 1])
        with h3c1:
            snap_labels = [p["snap_date"][:10] for p in period_log]
            sel_snap    = st.selectbox("Rebalance date", snap_labels,
                                       index=len(snap_labels) - 1, key="opt_bt_snap")
        with h3c2:
            sec_view = st.segmented_control(
                "Sector / industry view", ["Absolute", "Active"], default="Active", key="opt_bt_sec_view"
            )
        sel_entry = next((p for p in period_log if p["snap_date"][:10] == sel_snap), period_log[-1])
        has_bm_weights = bool(sel_entry.get("bm_sector_weights"))

        h_df = pd.DataFrame([
            {"Ticker":   ticker_map.get(isin, isin),
             "Company":  name_map.get(isin, ""),
             "Sector":   sector_map.get(isin, "Unknown"),
             "Industry": industry_map.get(isin, ""),
             "Weight %": round(w * 100, 3),
             "Value (€)": round(w * portfolio_eur, 0)}
            for isin, w in sorted(sel_entry["weights"].items(), key=lambda x: -x[1])
        ])

        hc1, hc2 = st.columns([3, 2])
        with hc1:
            st.markdown(f"**{len(h_df)} positions** — {sel_snap}")
            max_w = h_df["Weight %"].max() if not h_df.empty else 5.0
            st.dataframe(h_df, hide_index=True, width="stretch", column_config={
                "Weight %": st.column_config.ProgressColumn(
                    "Weight %", format="%.2f%%", min_value=0, max_value=max_w
                ),
                "Value (€)": st.column_config.NumberColumn("Value (€)", format="€%,.0f"),
            })
        with hc2:
            # Sector chart
            port_sec = sel_entry["sector_weights"]
            bm_sec   = sel_entry.get("bm_sector_weights", {})
            all_secs = sorted(set(port_sec) | set(bm_sec))
            if sec_view == "Active" and has_bm_weights:
                act_sec = {s: port_sec.get(s, 0.0) - bm_sec.get(s, 0.0) for s in all_secs}
                act_sec_s = sorted(act_sec.items(), key=lambda x: x[1])
                colors_s = ["#EF4444" if v < 0 else "#2563EB" for _, v in act_sec_s]
                fig_sec = go.Figure(go.Bar(
                    x=[v for _, v in act_sec_s], y=[s for s, _ in act_sec_s],
                    orientation="h", marker_color=colors_s,
                    text=[f"{v:+.1%}" for _, v in act_sec_s], textposition="outside",
                ))
                fig_sec.add_vline(x=0, line_color="#64748B", line_width=1)
                fig_sec.update_layout(title="Sector active weights",
                                      height=max(200, len(act_sec_s) * 30 + 60),
                                      xaxis_tickformat=".0%",
                                      margin=dict(l=0, r=70, t=40, b=10))
            else:
                sec_items = sorted(port_sec.items(), key=lambda x: x[1])
                fig_sec = go.Figure(go.Bar(
                    x=[v for _, v in sec_items], y=[s for s, _ in sec_items],
                    orientation="h", marker_color="#2563EB",
                    text=[f"{v:.1%}" for _, v in sec_items], textposition="outside",
                ))
                fig_sec.update_layout(title="Sector weights",
                                      height=max(200, len(sec_items) * 30 + 60),
                                      xaxis_tickformat=".0%",
                                      margin=dict(l=0, r=60, t=40, b=10))
            st.plotly_chart(fig_sec, width="stretch")

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
                    ("Turnover",     "⚠ Auto-relaxed" if sel_entry.get("turnover_relaxed") else "Within limit"),
                    ("Alpha date",   sel_entry.get("alpha_date", sel_entry["snap_date"])),
                ]
                _metrics_table(dict(rows_m))

        # Industry chart below
        st.divider()
        port_ind = sel_entry.get("industry_weights", {})
        bm_ind   = sel_entry.get("bm_industry_weights", {})
        all_inds = sorted(set(port_ind) | set(bm_ind))
        if sec_view == "Active" and has_bm_weights:
            act_ind = {i: port_ind.get(i, 0.0) - bm_ind.get(i, 0.0) for i in all_inds}
            # Show top 15 over + top 15 under by active weight
            sorted_ind = sorted(act_ind.items(), key=lambda x: x[1])
            bottom15 = sorted_ind[:15]
            top15    = sorted_ind[-15:][::-1]
            show_ind = list(dict.fromkeys(bottom15 + top15))  # preserve order, dedupe
            show_ind_s = sorted(show_ind, key=lambda x: x[1])
            colors_i = ["#EF4444" if v < 0 else "#2563EB" for _, v in show_ind_s]
            fig_ind = go.Figure(go.Bar(
                x=[v for _, v in show_ind_s], y=[i for i, _ in show_ind_s],
                orientation="h", marker_color=colors_i,
                text=[f"{v:+.1%}" for _, v in show_ind_s], textposition="outside",
            ))
            fig_ind.add_vline(x=0, line_color="#64748B", line_width=1)
            fig_ind.update_layout(title="Industry active weights (top 15 over/under)",
                                  height=max(300, len(show_ind_s) * 22 + 60),
                                  xaxis_tickformat=".0%",
                                  margin=dict(l=0, r=70, t=40, b=10))
        else:
            ind_items = sorted(port_ind.items(), key=lambda x: x[1])[-30:]
            fig_ind = go.Figure(go.Bar(
                x=[v for _, v in ind_items], y=[i for i, _ in ind_items],
                orientation="h", marker_color="#2563EB",
                text=[f"{v:.1%}" for _, v in ind_items], textposition="outside",
            ))
            fig_ind.update_layout(title="Industry weights (top 30)",
                                  height=max(300, len(ind_items) * 22 + 60),
                                  xaxis_tickformat=".0%",
                                  margin=dict(l=0, r=60, t=40, b=10))
        st.plotly_chart(fig_ind, width="stretch")
