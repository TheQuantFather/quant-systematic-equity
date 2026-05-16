"""
2_Factors.py — Factor distributions, spreads, cross-factor correlations, and time series.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import db
from utils import inject_css

st.set_page_config(page_title="Factor Analysis", layout="wide")
inject_css()
st.title("Factor Analysis")

factor_meta = db.get_factor_metadata()
screener    = db.get_screener_df()

CATEGORIES = sorted(factor_meta["category"].unique().tolist())

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------

tabs = st.tabs(CATEGORIES + ["Correlations", "Sector Spreads", "Time Series"])

# ---- Per-category tabs ----
for i, category in enumerate(CATEGORIES):
    with tabs[i]:
        st.subheader(f"{category} factors")

        cat_factors = factor_meta[factor_meta["category"] == category]["factor_name"].tolist()
        available   = [f for f in cat_factors if f in screener.columns]

        if not available:
            st.info("No data available for this category.")
            continue

        selected = st.selectbox("Factor", available, key=f"cat_{category}")
        col_data = screener[selected].dropna()

        desc = factor_meta[factor_meta["factor_name"] == selected]["description"].values
        direction = factor_meta[factor_meta["factor_name"] == selected]["direction"].values
        if len(desc):
            dir_label = "↑ higher is better" if (len(direction) and direction[0] == 1) else "↓ lower is better"
            st.caption(f"{desc[0]} — *{dir_label}*")

        left, right = st.columns([2, 1])

        with left:
            fig_hist = px.histogram(
                col_data,
                nbins=80,
                labels={"value": selected, "count": "Companies"},
                title=f"Distribution of {selected}",
                color_discrete_sequence=["#2563EB"],
            )
            fig_hist.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=20), showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)

        with right:
            st.markdown("**Summary statistics**")
            stats = col_data.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
            stats_df = stats.reset_index()
            stats_df.columns = ["Stat", "Value"]
            stats_df["Value"] = stats_df["Value"].map(lambda x: f"{x:,.4f}")
            st.dataframe(stats_df, hide_index=True, use_container_width=True)

        sector_data = screener[["sector", selected]].dropna()
        if not sector_data.empty:
            order = (
                sector_data.groupby("sector")[selected]
                .median()
                .sort_values(ascending=False)
                .index.tolist()
            )
            fig_box = px.box(
                sector_data,
                x="sector",
                y=selected,
                category_orders={"sector": order},
                labels={"sector": "", selected: "Value"},
                title=f"{selected} by sector",
                color="sector",
            )
            fig_box.update_layout(
                height=400,
                showlegend=False,
                xaxis_tickangle=-35,
                margin=dict(l=0, r=0, t=40, b=120),
            )
            st.plotly_chart(fig_box, use_container_width=True)

# ---- Correlations tab ----
with tabs[len(CATEGORIES)]:
    st.subheader("Cross-factor correlation matrix")

    all_factor_names  = factor_meta["factor_name"].tolist()
    available_factors = [f for f in all_factor_names if f in screener.columns]

    sel_cats = st.multiselect(
        "Filter by category",
        CATEGORIES,
        default=CATEGORIES,
        key="corr_cats",
    )
    filtered_meta  = factor_meta[factor_meta["category"].isin(sel_cats)]
    filtered_names = [f for f in filtered_meta["factor_name"].tolist() if f in screener.columns]

    if len(filtered_names) < 2:
        st.info("Select at least two categories.")
    else:
        corr_matrix = screener[filtered_names].corr(numeric_only=True)

        fig_heatmap = px.imshow(
            corr_matrix,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
            title="Pearson correlation between factors",
        )
        fig_heatmap.update_layout(
            height=max(400, len(filtered_names) * 22),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

        st.subheader("Most correlated pairs")
        pairs = (
            corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            .stack()
            .reset_index()
        )
        pairs.columns = ["Factor A", "Factor B", "Correlation"]
        pairs["abs_corr"] = pairs["Correlation"].abs()
        pairs = pairs.sort_values("abs_corr", ascending=False).head(15).drop(columns="abs_corr")
        pairs["Correlation"] = pairs["Correlation"].map(lambda x: f"{x:.3f}")
        st.dataframe(pairs, hide_index=True, use_container_width=True)

# ---- Sector Spreads tab ----
with tabs[len(CATEGORIES) + 1]:
    st.subheader("Factor spread by sector")
    st.caption(
        "Shows the interquartile range (Q75 − Q25) of each factor within each sector. "
        "Wider spread = more differentiation between stocks in that sector."
    )

    available_factors   = [f for f in factor_meta["factor_name"].tolist() if f in screener.columns]
    sel_factor_spread   = st.selectbox("Factor", available_factors, key="spread_factor")

    spread_df = (
        screener.groupby("sector")[sel_factor_spread]
        .agg(
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            count="count",
        )
        .reset_index()
        .dropna()
    )
    spread_df["iqr"] = spread_df["q75"] - spread_df["q25"]
    spread_df = spread_df.sort_values("median", ascending=True)

    fig_spread = go.Figure()
    fig_spread.add_trace(go.Bar(
        x=spread_df["median"],
        y=spread_df["sector"],
        orientation="h",
        name="Median",
        marker_color="#2563EB",
        error_x=dict(
            type="data",
            symmetric=False,
            array=spread_df["q75"] - spread_df["median"],
            arrayminus=spread_df["median"] - spread_df["q25"],
            color="#94A3B8",
        ),
    ))
    fig_spread.update_layout(
        height=500,
        xaxis_title=sel_factor_spread,
        yaxis_title="",
        margin=dict(l=0, r=40, t=20, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig_spread, use_container_width=True)

# ---- Time Series tab ----
with tabs[len(CATEGORIES) + 2]:
    st.subheader("Factor z-score trends over time")
    st.caption(
        "Median cross-sectional z-score per snapshot date. Shows how the typical level "
        "of each factor has shifted across annual snapshots."
    )

    long = db.get_factors_long()
    universe = db.get_universe()[["security_id", "sector"]].copy()
    universe["security_id"] = universe["security_id"].astype(str)
    long_with_sector = long.merge(universe, on="security_id", how="left")

    all_ts_factors = sorted(factor_meta["factor_name"].tolist())
    available_ts   = [f for f in all_ts_factors if f in long["factor_name"].values]

    col1, col2 = st.columns([2, 1])
    with col1:
        sel_ts_factors = st.multiselect(
            "Factors",
            available_ts,
            default=available_ts[:4],
            key="ts_factors",
        )
    with col2:
        sel_ts_sector = st.selectbox(
            "Sector (optional)",
            ["All sectors"] + sorted(universe["sector"].dropna().unique().tolist()),
            key="ts_sector",
        )

    if not sel_ts_factors:
        st.info("Select at least one factor.")
    else:
        ts_data = long_with_sector[long_with_sector["factor_name"].isin(sel_ts_factors)].copy()
        if sel_ts_sector != "All sectors":
            ts_data = ts_data[ts_data["sector"] == sel_ts_sector]

        ts_agg = (
            ts_data.groupby(["data_date", "factor_name"])["factor_value_z"]
            .median()
            .reset_index()
            .rename(columns={"factor_value_z": "Median z-score"})
        )

        if ts_agg.empty:
            st.info("No data for the selected combination.")
        else:
            fig_ts = px.line(
                ts_agg,
                x="data_date",
                y="Median z-score",
                color="factor_name",
                markers=True,
                labels={"data_date": "Snapshot date", "factor_name": "Factor"},
                title="Median factor z-score by snapshot" + (f" — {sel_ts_sector}" if sel_ts_sector != "All sectors" else ""),
            )
            fig_ts.add_hline(y=0, line_dash="dot", line_color="#94A3B8")
            fig_ts.update_layout(
                height=450,
                margin=dict(l=0, r=0, t=40, b=20),
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_ts, use_container_width=True)

            with st.expander("Raw values"):
                pivot_ts = ts_agg.pivot(index="data_date", columns="factor_name", values="Median z-score")
                st.dataframe(pivot_ts.round(4), use_container_width=True)
