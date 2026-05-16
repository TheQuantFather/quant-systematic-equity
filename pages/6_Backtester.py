"""
6_Backtester.py — Simple annual rebalancing factor backtest.

Strategy: rank stocks by model z-score at each snapshot date, hold an
equal-weight top-N portfolio until the next snapshot.  Optionally add a
short leg on the bottom-N stocks displayed separately for attribution.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
import db
from config import MODELS_DB, RETURNS_DB
from utils import get_db, inject_css

st.set_page_config(page_title="Backtester", layout="wide")
inject_css()
st.title("Factor Backtester")
st.caption(
    "Annual rebalancing: rank stocks by model score at each snapshot date, "
    "hold equal-weight top N until the next snapshot.  "
    "Returns are pre-computed daily total returns (split-invariant).  No transaction costs."
)

RISK_FREE  = 0.04   # annualised, used for Sharpe
N_QUINTILES = 5

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_all_model_scores() -> pd.DataFrame:
    with get_db(MODELS_DB) as conn:
        df = pd.read_sql(
            "SELECT data_date, model_id, security_id, model_value_z FROM models",
            conn,
        )
    df["model_value_z"] = pd.to_numeric(df["model_value_z"], errors="coerce")
    return df


@st.cache_data
def load_returns_matrix(min_isin_coverage: int = 200) -> pd.DataFrame:
    with get_db(RETURNS_DB) as conn:
        df = pd.read_sql(
            "SELECT isin, date, total_return FROM returns WHERE total_return IS NOT NULL",
            conn,
        )
    df["date"] = pd.to_datetime(df["date"])
    matrix = (
        df.pivot_table(index="date", columns="isin", values="total_return")
        .sort_index()
    )
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
    """Active return stats vs benchmark, plus turnover."""
    common = ret.index.intersection(bench.index)
    if len(common) < 63:
        return {}
    r = ret.loc[common].dropna()
    b = bench.loc[common].dropna()
    active  = r - b
    n_years = len(active) / 252
    ann_act = active.mean() * 252
    te      = active.std() * np.sqrt(252)
    ir      = ann_act / te if te > 0 else np.nan
    beta    = r.cov(b) / b.var() if b.var() > 0 else np.nan
    out = {
        "Active return (ann.)": f"{ann_act:+.1%}",
        "Tracking error":       f"{te:.1%}",
        "Information ratio":    f"{ir:.2f}" if pd.notna(ir) else "—",
        "Beta (vs EW uni)":     f"{beta:.2f}" if pd.notna(beta) else "—",
    }
    if turnover_pct is not None:
        out["Avg rebal. turnover"] = f"{turnover_pct:.0f}%"
    return out


# ---------------------------------------------------------------------------
# Backtest engine
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

    model_df = scores[scores["model_id"] == model_id].copy()
    model_df = model_df.dropna(subset=["model_value_z"])

    if sel_sectors:
        valid = set(uni[uni["sector"].isin(sel_sectors)]["security_id"])
        model_df = model_df[model_df["security_id"].isin(valid)]

    snapshot_dates = sorted(model_df["data_date"].unique())
    if len(snapshot_dates) < 2:
        return None, None, None, [], "Need at least 2 snapshot dates for a backtest."

    trading_index = ret_matrix.index

    def next_trading_day(d_str: str):
        ts  = pd.Timestamp(d_str)
        pos = trading_index.searchsorted(ts)
        return trading_index[pos] if pos < len(trading_index) else None

    def build_holdings_df(isins: list, score_lookup: dict, price_cols: set) -> pd.DataFrame:
        return pd.DataFrame([{
            "Rank":       rank,
            "Ticker":     ticker_map.get(isin, isin),
            "Company":    name_map.get(isin, ""),
            "Sector":     sector_map.get(isin, "Unknown"),
            "Score":      round(score_lookup.get(isin, np.nan), 3),
            "Price data": "✓" if isin in price_cols else "—",
        } for rank, isin in enumerate(isins, 1)])

    long_parts, bench_parts, short_parts, holdings_log = [], [], [], []

    for i, snap in enumerate(snapshot_dates):
        next_snap = (
            snapshot_dates[i + 1]
            if i + 1 < len(snapshot_dates)
            else trading_index[-1].strftime("%Y-%m-%d")
        )
        t_start = next_trading_day(snap)
        t_end   = next_trading_day(next_snap)
        if t_start is None or t_end is None or t_start >= t_end:
            continue

        snap_df    = model_df[model_df["data_date"] == snap].dropna(subset=["model_value_z"])
        score_lkp  = dict(zip(snap_df["security_id"], snap_df["model_value_z"]))
        long_isins = snap_df.nlargest(n_long,  "model_value_z")["security_id"].tolist()
        short_isins = snap_df.nsmallest(n_short, "model_value_z")["security_id"].tolist() if include_short else []

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
            "long":        build_holdings_df(long_isins,  score_lkp, price_cols),
            "short":       build_holdings_df(short_isins, score_lkp, price_cols) if include_short else None,
            "long_isins":  long_isins,
            "short_isins": short_isins,
        })

    if not long_parts:
        return None, None, None, [], "No overlapping price data found for this model and date range."

    long_s  = pd.concat(long_parts).sort_index()
    bench_s = pd.concat(bench_parts).sort_index()
    short_s = pd.concat(short_parts).sort_index() if include_short and short_parts else None

    return long_s, short_s, bench_s, holdings_log, None


# ---------------------------------------------------------------------------
# Quintile analysis
# ---------------------------------------------------------------------------

def run_quintile_analysis(model_id: str, sel_sectors: list) -> list[pd.Series]:
    """Return N_QUINTILES equal-weight return series, Q1 = best score, Q5 = worst."""
    scores     = load_all_model_scores()
    ret_matrix = load_returns_matrix()

    model_df = scores[scores["model_id"] == model_id].dropna(subset=["model_value_z"]).copy()

    if sel_sectors:
        uni = db.get_universe()[["security_id", "sector"]].copy()
        uni["security_id"] = uni["security_id"].astype(str)
        valid = set(uni[uni["sector"].isin(sel_sectors)]["security_id"])
        model_df = model_df[model_df["security_id"].isin(valid)]

    snapshot_dates = sorted(model_df["data_date"].unique())
    trading_index  = ret_matrix.index

    def next_td(d):
        pos = trading_index.searchsorted(pd.Timestamp(d))
        return trading_index[pos] if pos < len(trading_index) else None

    q_parts = [[] for _ in range(N_QUINTILES)]

    for i, snap in enumerate(snapshot_dates):
        next_snap = (
            snapshot_dates[i + 1]
            if i + 1 < len(snapshot_dates)
            else trading_index[-1].strftime("%Y-%m-%d")
        )
        t_start = next_td(snap)
        t_end   = next_td(next_snap)
        if t_start is None or t_end is None or t_start >= t_end:
            continue

        snap_df = model_df[model_df["data_date"] == snap].sort_values(
            "model_value_z", ascending=False
        ).reset_index(drop=True)
        n = len(snap_df)
        period = ret_matrix.loc[(ret_matrix.index >= t_start) & (ret_matrix.index < t_end)]

        for q in range(N_QUINTILES):
            s_idx  = int(q * n / N_QUINTILES)
            e_idx  = int((q + 1) * n / N_QUINTILES)
            isins  = snap_df.iloc[s_idx:e_idx]["security_id"].tolist()
            cols   = [s for s in isins if s in period.columns]
            q_parts[q].append(
                period[cols].mean(axis=1) if cols else pd.Series(0.0, index=period.index)
            )

    return [
        pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)
        for parts in q_parts
    ]


# ---------------------------------------------------------------------------
# Helpers
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


def _sector_chart(df: pd.DataFrame, title: str, color: str) -> go.Figure:
    counts = (
        df.groupby("Sector").size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=True)
    )
    fig = go.Figure(go.Bar(
        x=counts["Count"], y=counts["Sector"],
        orientation="h", marker_color=color,
        text=counts["Count"], textposition="outside",
    ))
    fig.update_layout(
        title=title,
        height=max(200, len(counts) * 30 + 60),
        margin=dict(l=0, r=40, t=40, b=10),
        xaxis_title="# stocks", yaxis_title="",
        xaxis=dict(showgrid=False),
    )
    return fig


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
    n_short = (
        st.slider("Short portfolio size (bottom N)", 10, 200, 50, step=10)
        if include_short else 0
    )

    st.divider()

    universe_meta = db.get_universe()
    all_sectors   = sorted(universe_meta["sector"].dropna().unique())
    sel_sectors   = st.multiselect("Sector filter", all_sectors, placeholder="All sectors")

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
# Run
# ---------------------------------------------------------------------------

with st.spinner("Running backtest…"):
    long_s, short_s, benchmark, holdings_log, err = run_backtest(
        sel_model_id, n_long, include_short, n_short, sel_sectors
    )

if err:
    st.warning(err)
    st.stop()

# Date filter
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

cum_long  = (1 + long_s).cumprod()
cum_bench = (1 + benchmark).cumprod()
cum_short = (1 + short_s).cumprod() if short_s is not None else None

turnover_rows = _compute_turnover(holdings_log)
avg_turnover  = np.mean([r["Turnover (%)"] for r in turnover_rows[1:]]) if len(turnover_rows) > 1 else None

# ---------------------------------------------------------------------------
# Cumulative return chart
# ---------------------------------------------------------------------------

fig_cum = go.Figure()
fig_cum.add_trace(go.Scatter(
    x=cum_long.index, y=cum_long.values, name=long_label,
    line=dict(color="#2563EB", width=2),
))
if cum_short is not None:
    fig_cum.add_trace(go.Scatter(
        x=cum_short.index, y=cum_short.values, name=short_label,
        line=dict(color="#DC2626", width=2, dash="dash"),
    ))
fig_cum.add_trace(go.Scatter(
    x=cum_bench.index, y=cum_bench.values, name="EW universe",
    line=dict(color="#94A3B8", width=1.5, dash="dot"),
))
fig_cum.update_layout(
    title="Cumulative return (base = 1.0)", height=420,
    yaxis_title="Portfolio value", hovermode="x unified",
    legend=dict(orientation="h", y=-0.15),
    margin=dict(l=0, r=0, t=40, b=10),
)
st.plotly_chart(fig_cum, use_container_width=True)

# ---------------------------------------------------------------------------
# Drawdown chart
# ---------------------------------------------------------------------------

dd_long  = cum_long  / cum_long.cummax()  - 1
dd_bench = cum_bench / cum_bench.cummax() - 1

fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(
    x=dd_long.index, y=dd_long.values, name=long_label,
    line=dict(color="#2563EB", width=1.5),
    fill="tozeroy", fillcolor="rgba(37,99,235,0.08)",
))
if cum_short is not None:
    dd_short = cum_short / cum_short.cummax() - 1
    fig_dd.add_trace(go.Scatter(
        x=dd_short.index, y=dd_short.values, name=short_label,
        line=dict(color="#DC2626", width=1.5, dash="dash"),
        fill="tozeroy", fillcolor="rgba(220,38,38,0.05)",
    ))
fig_dd.add_trace(go.Scatter(
    x=dd_bench.index, y=dd_bench.values, name="EW universe",
    line=dict(color="#94A3B8", width=1, dash="dot"),
))
fig_dd.update_layout(
    title="Drawdown", height=220,
    yaxis_tickformat=".0%", hovermode="x unified",
    legend=dict(orientation="h", y=-0.3),
    margin=dict(l=0, r=0, t=40, b=10),
)
st.plotly_chart(fig_dd, use_container_width=True)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

st.divider()
n_cols = 3 if short_s is not None else 2
cols   = st.columns(n_cols)

with cols[0]:
    st.subheader("Long basket")
    metrics = {**perf_metrics(long_s), **active_metrics(long_s, benchmark, avg_turnover)}
    st.dataframe(
        pd.DataFrame(metrics.items(), columns=["Metric", "Value"]),
        hide_index=True, use_container_width=True,
    )

if short_s is not None:
    with cols[1]:
        st.subheader("Short basket (held long)")
        st.caption("↑ outperformance here = headwind for short position")
        st.dataframe(
            pd.DataFrame(perf_metrics(short_s).items(), columns=["Metric", "Value"]),
            hide_index=True, use_container_width=True,
        )

with cols[-1]:
    st.subheader("EW benchmark")
    st.dataframe(
        pd.DataFrame(perf_metrics(benchmark).items(), columns=["Metric", "Value"]),
        hide_index=True, use_container_width=True,
    )

# ---------------------------------------------------------------------------
# Returns chart — annual or monthly heatmap
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Returns")

ret_view = st.segmented_control(
    "View", ["Annual", "Monthly heatmap"], default="Annual", key="ret_view"
)

if ret_view == "Annual":
    ann_long  = long_s.resample("YE").apply(lambda x: (1 + x).prod() - 1)
    ann_bench = benchmark.resample("YE").apply(lambda x: (1 + x).prod() - 1).reindex(ann_long.index)

    fig_ann = go.Figure()
    fig_ann.add_trace(go.Bar(
        x=ann_long.index.year.astype(str), y=ann_long.values,
        name="Long basket", marker_color="#2563EB",
        text=[f"{v:+.1%}" for v in ann_long.values], textposition="outside",
    ))
    if short_s is not None:
        ann_short = short_s.resample("YE").apply(lambda x: (1 + x).prod() - 1).reindex(ann_long.index)
        fig_ann.add_trace(go.Bar(
            x=ann_short.index.year.astype(str), y=ann_short.values,
            name="Short basket", marker_color="#DC2626",
            text=[f"{v:+.1%}" for v in ann_short.values], textposition="outside",
        ))
    fig_ann.add_trace(go.Bar(
        x=ann_bench.index.year.astype(str), y=ann_bench.values,
        name="EW universe", marker_color="#94A3B8",
        text=[f"{v:+.1%}" for v in ann_bench.values], textposition="outside",
    ))
    fig_ann.update_layout(
        barmode="group", height=320,
        yaxis_tickformat=".0%",
        margin=dict(l=0, r=0, t=20, b=20),
    )
    st.plotly_chart(fig_ann, use_container_width=True)

else:
    # Monthly heatmap — strategy only
    MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly = long_s.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    mdf = monthly.reset_index()
    mdf.columns = ["date", "ret"]
    mdf["Year"]  = mdf["date"].dt.year.astype(str)
    mdf["Month"] = mdf["date"].dt.strftime("%b")
    pivot = mdf.pivot(index="Year", columns="Month", values="ret").reindex(
        columns=MONTH_ORDER
    )
    text = pivot.map(lambda v: f"{v:+.1%}" if pd.notna(v) else "")

    fig_heat = go.Figure(go.Heatmap(
        z=pivot.values,
        x=MONTH_ORDER,
        y=pivot.index.tolist(),
        colorscale="RdYlGn",
        zmid=0,
        text=text.values,
        texttemplate="%{text}",
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

# ---------------------------------------------------------------------------
# Rolling metrics
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Rolling metrics")

roll_choice = st.selectbox(
    "Metric",
    ["Rolling Sharpe (1Y)", "Rolling Information Ratio (1Y)", "Turnover by period"],
    key="roll_choice",
)

ROLL_WINDOW = 252

if roll_choice == "Rolling Sharpe (1Y)":
    roll_sharpe = (
        (long_s.rolling(ROLL_WINDOW).mean() * ROLL_WINDOW - RISK_FREE)
        / (long_s.rolling(ROLL_WINDOW).std() * ROLL_WINDOW ** 0.5)
    ).dropna()

    bench_sharpe = (
        (benchmark.rolling(ROLL_WINDOW).mean() * ROLL_WINDOW - RISK_FREE)
        / (benchmark.rolling(ROLL_WINDOW).std() * ROLL_WINDOW ** 0.5)
    ).dropna()

    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(
        x=roll_sharpe.index, y=roll_sharpe.values, name=long_label,
        line=dict(color="#2563EB", width=2),
    ))
    fig_roll.add_trace(go.Scatter(
        x=bench_sharpe.index, y=bench_sharpe.values, name="EW universe",
        line=dict(color="#94A3B8", width=1.5, dash="dot"),
    ))
    fig_roll.add_hline(y=0, line_dash="dot", line_color="#64748B", line_width=1)
    fig_roll.update_layout(
        height=300, yaxis_title="Sharpe ratio (1Y rolling)",
        hovermode="x unified", legend=dict(orientation="h", y=-0.2),
        margin=dict(l=0, r=0, t=10, b=10),
    )
    st.plotly_chart(fig_roll, use_container_width=True)

elif roll_choice == "Rolling Information Ratio (1Y)":
    active      = long_s.subtract(benchmark.reindex(long_s.index, fill_value=0))
    roll_ir     = (
        (active.rolling(ROLL_WINDOW).mean() * ROLL_WINDOW)
        / (active.rolling(ROLL_WINDOW).std() * ROLL_WINDOW ** 0.5)
    ).dropna()

    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(
        x=roll_ir.index, y=roll_ir.values, name="Rolling IR",
        line=dict(color="#2563EB", width=2),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.07)",
    ))
    fig_roll.add_hline(y=0,    line_dash="dot", line_color="#64748B", line_width=1)
    fig_roll.add_hline(y=0.5,  line_dash="dash", line_color="#22C55E", line_width=1,
                       annotation_text="IR 0.5", annotation_position="right")
    fig_roll.add_hline(y=-0.5, line_dash="dash", line_color="#EF4444", line_width=1)
    fig_roll.update_layout(
        height=300, yaxis_title="Information ratio (1Y rolling)",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=10, b=10),
    )
    st.plotly_chart(fig_roll, use_container_width=True)
    st.caption(
        "IR > 0.5 (green) = the model is consistently adding active return relative to its risk. "
        "Periods below zero mean the model underperformed the EW universe on a risk-adjusted basis."
    )

else:  # Turnover by period
    if not turnover_rows:
        st.info("Need at least 2 rebalance periods to compute turnover.")
    else:
        to_df = pd.DataFrame(turnover_rows)
        fig_to = go.Figure(go.Bar(
            x=to_df["Period"], y=to_df["Turnover (%)"],
            marker_color="#2563EB",
            text=[f"{v:.0f}%" for v in to_df["Turnover (%)"]],
            textposition="outside",
        ))
        fig_to.update_layout(
            height=300, yaxis_title="One-way turnover (%)",
            yaxis_range=[0, 110],
            margin=dict(l=0, r=0, t=10, b=10),
        )
        st.plotly_chart(fig_to, use_container_width=True)
        st.caption(
            "% of long portfolio replaced at each annual rebalance. "
            "First period is always 100% (portfolio built from scratch). "
            f"Average (ex-first): **{avg_turnover:.0f}%**."
        )

# ---------------------------------------------------------------------------
# Quintile analysis
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Quintile analysis")
st.caption(
    "Universe split into 5 equal buckets by model score at each snapshot. "
    "Q1 = highest-scored stocks, Q5 = lowest. "
    "A working factor shows Q1 > Q2 > Q3 > Q4 > Q5."
)

with st.spinner("Computing quintiles…"):
    q_series = run_quintile_analysis(sel_model_id, sel_sectors)

# Apply same date filter
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
        q_ann.append({"Quintile": Q_LABELS[i], "Ann. return": ann_r,
                      "color": Q_COLORS[i]})

    if q_ann:
        fig_qa = go.Figure()
        for row in q_ann:
            fig_qa.add_trace(go.Bar(
                x=[row["Quintile"]], y=[row["Ann. return"]],
                name=row["Quintile"], marker_color=row["color"],
                text=[f"{row['Ann. return']:+.1%}"], textposition="outside",
                showlegend=False,
            ))
        # Benchmark reference line
        bench_ann = (1 + benchmark).prod() ** (252 / max(len(benchmark), 1)) - 1
        fig_qa.add_hline(
            y=bench_ann, line_dash="dot", line_color="#94A3B8", line_width=1.5,
            annotation_text=f"EW universe {bench_ann:+.1%}",
            annotation_position="right",
        )
        fig_qa.update_layout(
            height=340, yaxis_tickformat=".0%",
            yaxis_title="Annualised return",
            margin=dict(l=0, r=80, t=20, b=10),
        )
        st.plotly_chart(fig_qa, use_container_width=True)

with q_tab2:
    fig_qc = go.Figure()
    for i, s in enumerate(q_series):
        if s.empty:
            continue
        cum = (1 + s).cumprod()
        fig_qc.add_trace(go.Scatter(
            x=cum.index, y=cum.values, name=Q_LABELS[i],
            line=dict(color=Q_COLORS[i], width=2),
        ))
    fig_qc.add_trace(go.Scatter(
        x=cum_bench.index, y=cum_bench.values, name="EW universe",
        line=dict(color="#94A3B8", width=1.5, dash="dot"),
    ))
    fig_qc.update_layout(
        height=380, yaxis_title="Portfolio value (base = 1.0)",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=0, r=0, t=20, b=10),
    )
    st.plotly_chart(fig_qc, use_container_width=True)

# ---------------------------------------------------------------------------
# Holdings explorer
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Portfolio holdings")

period_labels    = [p["label"] for p in holdings_log]
sel_period_label = st.segmented_control(
    "Period", period_labels, default=period_labels[-1], key="holdings_period"
)
sel_period = next((p for p in holdings_log if p["label"] == sel_period_label), holdings_log[-1])

long_df = sel_period["long"]
lcol, rcol = st.columns([3, 2])
with lcol:
    st.markdown(f"**Long basket — {len(long_df)} stocks**")
    st.dataframe(
        long_df, hide_index=True, use_container_width=True,
        column_config={
            "Rank":       st.column_config.NumberColumn(width="small"),
            "Score":      st.column_config.NumberColumn(format="%.3f", width="small"),
            "Price data": st.column_config.TextColumn(width="small"),
        },
    )
with rcol:
    st.plotly_chart(_sector_chart(long_df, "Sector breakdown — long", "#2563EB"),
                    use_container_width=True)

short_df = sel_period.get("short")
if short_df is not None and not short_df.empty:
    st.markdown("---")
    slcol, srcol = st.columns([3, 2])
    with slcol:
        st.markdown(f"**Short basket — {len(short_df)} stocks**")
        st.dataframe(
            short_df, hide_index=True, use_container_width=True,
            column_config={
                "Rank":       st.column_config.NumberColumn(width="small"),
                "Score":      st.column_config.NumberColumn(format="%.3f", width="small"),
                "Price data": st.column_config.TextColumn(width="small"),
            },
        )
    with srcol:
        st.plotly_chart(_sector_chart(short_df, "Sector breakdown — short", "#DC2626"),
                        use_container_width=True)
