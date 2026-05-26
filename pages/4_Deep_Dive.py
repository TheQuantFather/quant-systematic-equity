"""
4_Deep_Dive.py — Single-stock deep dive: model score history, factor scores, financial history.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))
import db
from utils import inject_css

st.set_page_config(page_title="Stock Deep Dive", layout="wide")
inject_css()
st.title("Stock Deep Dive")

screener    = db.get_screener_df()
factor_meta = db.get_factor_metadata()
model_meta  = db.get_model_metadata()

MODEL_COL_NAMES  = [f"{m} Model" for m in model_meta["Model"]]
MODEL_COLS       = [c for c in screener.columns if c in MODEL_COL_NAMES]
MODEL_ID_TO_NAME = dict(zip(model_meta["ModelID"], MODEL_COL_NAMES[:len(model_meta)]))
MODEL_DISPLAY    = {f"{m} Model": m for m in model_meta["Model"]}

# ---------------------------------------------------------------------------
# Company selector
# ---------------------------------------------------------------------------

all_names = screener.sort_values("company_name")["display_name"].tolist()

selected_name = st.selectbox(
    "Search for a company (ticker or name)",
    all_names,
    index=None,
    placeholder="Type to search…",
)

if not selected_name:
    st.info("Select a company above to begin the analysis.")
    st.stop()

row = screener[screener["display_name"] == selected_name].iloc[0]
security_id  = str(row["security_id"])
company_name = row["company_name"]
ticker       = row["ticker"] or "N/A"
sector       = row["sector"] or "N/A"
industry     = row["industry"] or "N/A"

# ---------------------------------------------------------------------------
# Header card
# ---------------------------------------------------------------------------

st.subheader(f"{ticker} — {company_name}")
m1, m2, m3 = st.columns(3)
m1.metric("Sector", sector)
m2.metric("Industry", industry)
m3.metric("ISIN", security_id)

st.divider()

# ---------------------------------------------------------------------------
# Price returns
# ---------------------------------------------------------------------------

ret_df = db.get_returns_for_security(security_id)

if ret_df.empty:
    st.warning("No returns data found for this security.")
else:
    st.subheader("Price returns")

    # --- Horizon buttons ---
    HORIZONS = {"6M": 126, "1Y": 252, "3Y": 756, "5Y": 1260, "Max": None}
    sel_horizon = st.segmented_control(
        "Period", list(HORIZONS.keys()), default="1Y", key="ret_horizon"
    )
    n_days = HORIZONS[sel_horizon]

    plot_df = ret_df.dropna(subset=["total_return"]).copy()
    if n_days is not None:
        plot_df = plot_df.tail(n_days + 1)  # +1 for first anchor row

    # Cumulative return rebased to 0 at start of window
    plot_df = plot_df.copy()
    plot_df["cum_return"] = (np.cumprod(1 + plot_df["total_return"].values) - 1) * 100

    fig_ret = go.Figure()
    fig_ret.add_trace(go.Scatter(
        x=plot_df["date"],
        y=plot_df["cum_return"],
        mode="lines",
        line=dict(color="#2563EB", width=2),
        fill="tozeroy",
        fillcolor="rgba(37,99,235,0.08)",
        hovertemplate="%{x|%Y-%m-%d}<br><b>%{y:.2f}%</b><extra></extra>",
    ))
    fig_ret.add_hline(y=0, line_dash="dot", line_color="#94A3B8", line_width=1)
    fig_ret.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=10, b=10),
        xaxis_title="",
        yaxis_title="Cumulative return (%)",
        yaxis_ticksuffix="%",
        hovermode="x unified",
    )
    st.plotly_chart(fig_ret, use_container_width=True)

    # --- Horizon returns table ---
    all_rets = ret_df.dropna(subset=["total_return"])["total_return"].values
    all_dates = ret_df.dropna(subset=["total_return"])["date"]
    today = pd.Timestamp(date.today())

    def _period_return(n: int | None) -> str:
        arr = all_rets if n is None else all_rets[-n:]
        if len(arr) == 0:
            return "—"
        r = float(np.prod(1 + arr) - 1) * 100
        return f"{r:+.1f}%"

    def _ytd_return() -> str:
        jan1 = pd.Timestamp(date(today.year, 1, 1))
        mask = all_dates >= jan1
        arr = all_rets[mask.values]
        if len(arr) == 0:
            return "—"
        r = float(np.prod(1 + arr) - 1) * 100
        return f"{r:+.1f}%"

    horizon_rows = [
        {"Period": "1 Week",  "Return": _period_return(5)},
        {"Period": "1 Month", "Return": _period_return(21)},
        {"Period": "3 Month", "Return": _period_return(63)},
        {"Period": "6 Month", "Return": _period_return(126)},
        {"Period": "YTD",     "Return": _ytd_return()},
        {"Period": "1 Year",  "Return": _period_return(252)},
        {"Period": "3 Year",  "Return": _period_return(756)},
        {"Period": "5 Year",  "Return": _period_return(1260)},
        {"Period": "Max",     "Return": _period_return(None)},
    ]
    horizon_df = pd.DataFrame(horizon_rows)
    st.dataframe(horizon_df, use_container_width=False, hide_index=True, width=280)

st.divider()

# ---------------------------------------------------------------------------
# Current model scores
# ---------------------------------------------------------------------------

model_scores = {m: row[m] for m in MODEL_COLS if pd.notna(row.get(m))}

if model_scores:
    st.subheader("Model scores (latest snapshot)")
    st.caption("Direction-adjusted composite z-scores — higher is always better.")

    models_df = pd.DataFrame(
        {"Model": [MODEL_DISPLAY.get(m, m) for m in model_scores.keys()], "Score": list(model_scores.values())}
    ).sort_values("Score", ascending=True)

    fig_models = px.bar(
        models_df,
        x="Score",
        y="Model",
        orientation="h",
        color="Score",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        text=models_df["Score"].map(lambda x: f"{x:.3f}"),
    )
    fig_models.update_traces(textposition="outside")
    fig_models.update_layout(
        height=max(250, len(model_scores) * 42),
        coloraxis_showscale=False,
        margin=dict(l=0, r=60, t=10, b=10),
        xaxis_title="Z-score",
        yaxis_title="",
    )
    st.plotly_chart(fig_models, use_container_width=True)

# ---------------------------------------------------------------------------
# Score history across snapshot dates
# ---------------------------------------------------------------------------

models_ts = db.get_models_for_security(security_id)
factor_ts  = db.get_factors_for_security(security_id)

if not models_ts.empty and models_ts["data_date"].nunique() > 1:
    st.subheader("Score history across snapshots")

    left_ts, right_ts = st.columns(2)

    with left_ts:
        st.markdown("**Model scores over time**")
        mts = models_ts.copy()
        mts["model_name"] = mts["model_id"].map(MODEL_ID_TO_NAME).map(
            lambda x: MODEL_DISPLAY.get(x, x) if pd.notna(x) else x
        )
        mts = mts.dropna(subset=["model_value_z", "model_name"])

        fig_mts = px.line(
            mts.sort_values("data_date"),
            x="data_date",
            y="model_value_z",
            color="model_name",
            markers=True,
            labels={"data_date": "Snapshot", "model_value_z": "Z-score", "model_name": "Model"},
        )
        fig_mts.add_hline(y=0, line_dash="dot", line_color="#94A3B8")
        fig_mts.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=10, b=10),
            legend=dict(orientation="h", y=-0.3, font_size=11),
        )
        st.plotly_chart(fig_mts, use_container_width=True)

    with right_ts:
        st.markdown("**Factor z-scores over time**")
        available_ts = sorted(factor_ts["factor_name"].dropna().unique().tolist())
        default_ts   = [f for f in ["ROE", "Gross Margin", "Earnings Yield", "Revenue Growth"] if f in available_ts]
        sel_ts       = st.multiselect("Select factors", available_ts, default=default_ts[:3], key="dd_ts_factors")

        if sel_ts:
            fts = factor_ts[factor_ts["factor_name"].isin(sel_ts)][
                ["data_date", "factor_name", "factor_value_z"]
            ].dropna().sort_values("data_date")

            fig_fts = px.line(
                fts,
                x="data_date",
                y="factor_value_z",
                color="factor_name",
                markers=True,
                labels={"data_date": "Snapshot", "factor_value_z": "Z-score", "factor_name": "Factor"},
            )
            fig_fts.add_hline(y=0, line_dash="dot", line_color="#94A3B8")
            fig_fts.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=10, b=10),
                legend=dict(orientation="h", y=-0.3, font_size=11),
            )
            st.plotly_chart(fig_fts, use_container_width=True)
        else:
            st.info("Select factors to plot.")

    st.divider()

# ---------------------------------------------------------------------------
# Factor scores vs sector median (latest snapshot)
# ---------------------------------------------------------------------------

st.subheader("Factor scores vs sector median")

if factor_ts.empty:
    st.warning("No factor data found for this company.")
else:
    # Latest snapshot only for point-in-time comparison
    latest_date = factor_ts["data_date"].max()
    factor_latest = factor_ts[factor_ts["data_date"] == latest_date]

    sector_medians = db.get_sector_factor_medians()
    sector_row     = sector_medians[sector_medians["sector"] == sector]

    st.caption(f"Showing values from snapshot **{latest_date}**, compared to {sector} sector median.")

    tabs_cat = st.tabs(sorted(factor_meta["category"].unique().tolist()))

    for tab, category in zip(tabs_cat, sorted(factor_meta["category"].unique().tolist())):
        with tab:
            cat_factors = factor_meta[factor_meta["category"] == category]

            rows = []
            for _, frow in cat_factors.iterrows():
                fval = factor_latest[factor_latest["factor_name"] == frow["factor_name"]]["factor_value"]
                if fval.empty:
                    continue
                val = fval.values[0]
                median_val = None
                if not sector_row.empty and frow["factor_name"] in sector_row.columns:
                    median_val = sector_row.iloc[0][frow["factor_name"]]
                dir_label = "↑" if frow["direction"] == 1 else "↓"
                rows.append({
                    "Factor": f"{frow['factor_name']} {dir_label}",
                    "Stock": val,
                    "Sector median": median_val,
                })

            if not rows:
                st.info("No data for this category.")
                continue

            comp_df = pd.DataFrame(rows).dropna(subset=["Stock"])
            comp_df = comp_df.sort_values("Stock", ascending=True)

            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                x=comp_df["Stock"],
                y=comp_df["Factor"],
                orientation="h",
                name=ticker or company_name,
                marker_color="#2563EB",
            ))
            if "Sector median" in comp_df.columns:
                fig_comp.add_trace(go.Scatter(
                    x=comp_df["Sector median"],
                    y=comp_df["Factor"],
                    mode="markers",
                    name=f"{sector} median",
                    marker=dict(symbol="diamond", size=10, color="#F59E0B"),
                ))
            fig_comp.update_layout(
                height=max(300, len(comp_df) * 32),
                margin=dict(l=0, r=0, t=10, b=10),
                legend=dict(orientation="h", y=1.05),
                xaxis_title="Value",
                yaxis_title="",
            )
            st.plotly_chart(fig_comp, use_container_width=True)

# ---------------------------------------------------------------------------
# Financial history (from constituents)
# ---------------------------------------------------------------------------

st.subheader("Financial history")

constituents = db.get_constituents_for_security(security_id)

_Q_NUM = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "FY": 4}

def _sort_key(fy: int, fp: str) -> int:
    if pd.isna(fy):
        return 0
    return int(fy) * 10 + _Q_NUM.get(str(fp), 9)

def _period_label(fy: int, fp: str) -> str:
    return f"{fy}-{fp}"

def _smart_scale(series: pd.Series) -> tuple[pd.Series, str]:
    """Scale a numeric series to readable units; return (scaled, unit_label)."""
    mx = series.abs().max()
    if mx >= 1e9:
        return series / 1e9, "USD (B)"
    if mx >= 1e6:
        return series / 1e6, "USD (M)"
    return series, "USD"

def _derive_operating_income(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inject derived Operating Income rows for quarters where it is absent.
    Identity: Operating Income = Pretax Income (Loss) − Non-Operating Income (Loss).
    Mirrors the derivation in create_factors.py build_ltm() and validate_ticker.py.
    """
    OI_NAME    = "Operating Income (Loss)"
    PRETAX     = "Pretax Income (Loss)"
    NON_OP     = "Non-Operating Income (Loss)"
    STMT_TYPE  = "Income Statement"

    is_df = df[df["statement_type"] == STMT_TYPE].copy()
    if is_df.empty:
        return df

    periods_with_oi = set(
        zip(is_df.loc[is_df["constituent_name"] == OI_NAME, "fiscal_year"],
            is_df.loc[is_df["constituent_name"] == OI_NAME, "fiscal_period"])
    )
    # Use groupby().first() to get one scalar value per period, avoiding
    # MultiIndex .loc scalar/Series ambiguity when there are duplicate rows.
    pretax_by_period = (
        is_df[is_df["constituent_name"] == PRETAX]
        .groupby(["fiscal_year", "fiscal_period"])["constituent_value"].first()
    )
    nonop_by_period = (
        is_df[is_df["constituent_name"] == NON_OP]
        .groupby(["fiscal_year", "fiscal_period"])["constituent_value"].first()
    )
    # Template rows (one per period) for copying metadata
    template_rows = (
        is_df[is_df["constituent_name"] == PRETAX]
        .groupby(["fiscal_year", "fiscal_period"]).first()
    )

    new_rows = []
    for (fy, fp), pretax_val in pretax_by_period.items():
        if (fy, fp) in periods_with_oi:
            continue
        if pd.isna(pretax_val):
            continue
        non_op_val = nonop_by_period.get((fy, fp), 0.0)
        non_op_val = 0.0 if pd.isna(non_op_val) else float(non_op_val)
        template = template_rows.loc[(fy, fp)].to_dict()
        template["constituent_name"]  = OI_NAME
        template["constituent_value"] = float(pretax_val) - non_op_val
        template["constituent_id"]    = "80C2558A"
        new_rows.append(template)

    if not new_rows:
        return df
    return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)


constituents = _derive_operating_income(constituents)

if constituents.empty:
    st.warning("No financial constituent data found for this company.")
else:
    _STMT_ORDER = ["Income Statement", "Cash Flow Statement", "Balance Sheet"]
    stmt_types = [s for s in _STMT_ORDER if s in constituents["statement_type"].unique()]

    stmt_tabs = st.tabs(stmt_types)

    for stmt_tab, stmt_type in zip(stmt_tabs, stmt_types):
        with stmt_tab:
            stmt_df = constituents[constituents["statement_type"] == stmt_type].copy()
            is_balance = stmt_type == "Balance Sheet"

            metrics = sorted(stmt_df["constituent_name"].dropna().unique().tolist())
            if not metrics:
                st.info("No data.")
                continue

            # Sensible defaults per statement
            _defaults = {
                "Income Statement":    ["Revenue", "Gross Profit", "Operating Income (Loss)", "Net Income"],
                "Cash Flow Statement": ["Net Cash from Operating Activities", "Capital Expenditures",
                                        "Free Cash Flow", "Net Cash from Investing Activities"],
                "Balance Sheet":       ["Total Assets", "Total Equity", "Total Liabilities", "Cash & Cash Equivalents"],
            }
            defaults = [m for m in _defaults.get(stmt_type, []) if m in metrics] or metrics[:4]

            sel_metrics = st.multiselect(
                "Select metrics",
                metrics,
                default=defaults,
                key=f"metrics_{stmt_type}",
            )
            if not sel_metrics:
                continue

            filt = stmt_df[stmt_df["constituent_name"].isin(sel_metrics)].copy()
            filt["_sort"]  = filt.apply(lambda r: _sort_key(r["fiscal_year"], r["fiscal_period"]), axis=1)
            filt["_label"] = filt.apply(lambda r: _period_label(r["fiscal_year"], r["fiscal_period"]), axis=1)

            if is_balance:
                # Balance Sheet — one PIT value per Q4 (annual year-end)
                bs = filt[filt["fiscal_period"] == "Q4"].copy()
                if bs.empty:
                    bs = filt[filt["fiscal_period"] == "FY"].copy()
                if bs.empty:
                    bs = filt.copy()

                pivot = (
                    bs.pivot_table(
                        index=["_sort", "_label"],
                        columns="constituent_name",
                        values="constituent_value",
                        aggfunc="first",
                    )
                    .sort_index()
                    .reset_index()
                )
                x_vals   = pivot["_label"].tolist()
                all_bs = pd.concat([pivot[m].dropna() for m in sel_metrics if m in pivot.columns], ignore_index=True)
                _, unit_lbl = _smart_scale(all_bs) if not all_bs.empty else (None, "USD")
                divisor = 1e9 if "B" in unit_lbl else (1e6 if "M" in unit_lbl else 1)
                fig = go.Figure()
                for m in sel_metrics:
                    if m not in pivot.columns:
                        continue
                    vals = pivot[m].copy() / divisor
                    fig.add_trace(go.Scatter(
                        x=x_vals, y=vals,
                        name=m, mode="lines+markers",
                        hovertemplate=f"%{{x}}<br>{m}: %{{y:,.2f}} {unit_lbl}<extra></extra>",
                    ))
                st.caption("Point-in-time annual balance sheet (Q4 / year-end).")

            else:
                # Income Statement / Cash Flow — LTM via rolling 4-quarter sum
                q_data = filt[filt["fiscal_period"].isin(["Q1", "Q2", "Q3", "Q4"])].copy()
                if q_data.empty:
                    # Company only has annual data — show directly
                    q_data = filt[filt["fiscal_period"] == "FY"].copy()
                    use_ltm = False
                else:
                    use_ltm = True

                pivot = (
                    q_data.pivot_table(
                        index=["_sort", "_label"],
                        columns="constituent_name",
                        values="constituent_value",
                        aggfunc="first",
                    )
                    .sort_index()
                )
                x_vals = [lbl for _, lbl in pivot.index]

                metric_cols = [m for m in sel_metrics if m in pivot.columns]
                if use_ltm:
                    display = pivot[metric_cols].rolling(4, min_periods=4).sum()
                    enough  = display.notna().any(axis=1)
                    display = display[enough]
                    x_vals  = [lbl for (_, lbl), ok in zip(pivot.index, enough) if ok]
                    if display.empty:
                        st.caption("⚠ Fewer than 4 quarters of data — showing available periods.")
                        display = pivot[metric_cols]
                        x_vals  = [lbl for _, lbl in pivot.index]
                        use_ltm = False
                else:
                    display = pivot[metric_cols]

                all_is = pd.concat([display[m].dropna() for m in metric_cols if m in display.columns], ignore_index=True)
                _, unit_lbl = _smart_scale(all_is) if not all_is.empty else (None, "USD")
                divisor = 1e9 if "B" in unit_lbl else (1e6 if "M" in unit_lbl else 1)
                fig = go.Figure()
                for m in metric_cols:
                    if m not in display.columns:
                        continue
                    vals = display[m].copy() / divisor
                    fig.add_trace(go.Scatter(
                        x=x_vals, y=vals.values,
                        name=m, mode="lines+markers",
                        hovertemplate=f"%{{x}}<br>{m}: %{{y:,.2f}} {unit_lbl}<extra></extra>",
                    ))
                caption = "LTM (trailing twelve months) — rolling 4-quarter sum." if use_ltm \
                          else "Annual values."
                st.caption(caption)

            fig.update_layout(
                height=420,
                margin=dict(l=0, r=0, t=20, b=10),
                yaxis_title=unit_lbl,
                xaxis_title="Period",
                legend=dict(orientation="h", y=1.08),
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Raw data"):
                raw = filt[["_sort", "_label", "fiscal_period", "constituent_name", "constituent_value"]].rename(
                    columns={"_label": "Period"}
                ).sort_values(["constituent_name", "_sort"]).drop(columns="_sort")
                raw["constituent_value"] = raw["constituent_value"].map(
                    lambda x: f"{x:,.0f}" if pd.notna(x) else ""
                )
                st.dataframe(raw, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------
# Peer comparison
# ---------------------------------------------------------------------------

st.subheader("Peer comparison")
_composite_cols = [f"{m} Model" for m in model_meta[model_meta["IsComposite"] == 1]["Model"]]
MODEL_SORT = next((n for n in _composite_cols if n in screener.columns), MODEL_COLS[0] if MODEL_COLS else None)
_sort_label = MODEL_DISPLAY.get(MODEL_SORT, MODEL_SORT) if MODEL_SORT else "model"
st.caption(f"Top 10 peers in **{industry}** ranked by {_sort_label} score.")

if MODEL_SORT and industry != "N/A":
    peers = (
        screener[screener["industry"] == industry]
        .sort_values(MODEL_SORT, ascending=False)
        .head(10)
    )

    peer_display = (
        ["ticker", "company_name"] + MODEL_COLS +
        [c for c in ["Gross Margin", "ROE", "Earnings Yield", "Revenue Growth"] if c in screener.columns]
    )
    peer_display = [c for c in peer_display if c in peers.columns]

    styled_peers = peers[peer_display].copy()
    for c in [col for col in MODEL_COLS if col in styled_peers.columns]:
        styled_peers[c] = styled_peers[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    styled_peers = styled_peers.rename(columns=MODEL_DISPLAY)

    st.dataframe(styled_peers, use_container_width=True, hide_index=True)
else:
    st.info("Insufficient data for peer comparison.")
