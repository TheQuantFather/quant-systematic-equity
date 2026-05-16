"""
5_Themes.py — Sector/industry heatmaps, opportunity set, and thematic screens.
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

st.set_page_config(page_title="Themes & Opportunity Set", layout="wide")
inject_css()
st.title("Themes & Opportunity Set")

screener    = db.get_screener_df()
factor_meta = db.get_factor_metadata()
model_meta  = db.get_model_metadata()

MODEL_COL_NAMES = [f"{m} Model" for m in model_meta["Model"]]
MODEL_COLS      = [c for c in screener.columns if c in MODEL_COL_NAMES]
FACTOR_NAMES    = [f for f in factor_meta["factor_name"].tolist() if f in screener.columns]

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------

tab_heatmap, tab_opportunity, tab_bubbles, tab_top = st.tabs([
    "Sector Heatmap",
    "Opportunity Set",
    "Factor Bubbles",
    "Top Stocks by Theme",
])

# ---- Sector Heatmap ----
with tab_heatmap:
    st.subheader("Median model scores by sector")
    st.caption("Each cell shows the median z-score for that model within a sector. Green = above universe median.")

    if not MODEL_COLS:
        st.info("No model data available.")
    else:
        sector_model = db.get_sector_model_medians()
        heat_cols = [c for c in MODEL_COLS if c in sector_model.columns]

        if heat_cols and not sector_model.empty:
            heat_df = sector_model.set_index("sector")[heat_cols]
            heat_df = heat_df.dropna(how="all")

            # Sort sectors by mean model score
            heat_df["_mean"] = heat_df.mean(axis=1)
            heat_df = heat_df.sort_values("_mean", ascending=False).drop(columns="_mean")

            fig_heat = px.imshow(
                heat_df,
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                aspect="auto",
                text_auto=".2f",
                labels={"color": "Median score"},
            )
            fig_heat.update_layout(
                height=max(350, len(heat_df) * 28),
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis_title="",
                yaxis_title="",
            )
            st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()
    st.subheader("Median factor values by sector")

    available_cats = sorted(factor_meta["category"].unique().tolist())
    sel_cat_heat = st.selectbox("Factor category", available_cats, key="heat_cat")
    cat_factor_names = [
        f for f in factor_meta[factor_meta["category"] == sel_cat_heat]["factor_name"].tolist()
        if f in screener.columns
    ]

    if cat_factor_names:
        sector_factor_df = (
            screener.groupby("sector")[cat_factor_names]
            .median(numeric_only=True)
            .dropna(how="all")
        )
        sector_factor_df["_mean"] = sector_factor_df.rank(axis=0).mean(axis=1)
        sector_factor_df = sector_factor_df.sort_values("_mean", ascending=False).drop(columns="_mean")

        fig_fheat = px.imshow(
            sector_factor_df,
            color_continuous_scale="RdYlGn",
            aspect="auto",
            text_auto=".2f",
            labels={"color": "Median value"},
        )
        fig_fheat.update_layout(
            height=max(350, len(sector_factor_df) * 28),
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig_fheat, use_container_width=True)

# ---- Opportunity Set ----
with tab_opportunity:
    st.subheader("Opportunity set — stocks in the top quintile")
    st.caption(
        "Stocks scoring in the top 20% on the selected models simultaneously. "
        "This is a simple intersection screen — not a portfolio."
    )

    sel_opp_models = st.multiselect(
        "Models to screen on (top 20% each)",
        MODEL_COLS,
        default=MODEL_COLS[:2] if len(MODEL_COLS) >= 2 else MODEL_COLS,
        key="opp_models",
    )

    sel_opp_sector = st.multiselect("Sector filter", sorted(screener["sector"].dropna().unique()), key="opp_sector")

    if not sel_opp_models:
        st.info("Select at least one model.")
    else:
        opp_df = screener.copy()
        if sel_opp_sector:
            opp_df = opp_df[opp_df["sector"].isin(sel_opp_sector)]

        for m in sel_opp_models:
            threshold = opp_df[m].quantile(0.80)
            opp_df = opp_df[opp_df[m] >= threshold]

        opp_df = opp_df.sort_values(sel_opp_models[0], ascending=False)

        st.markdown(f"**{len(opp_df)}** stocks qualify")

        display_cols = (
            ["ticker", "company_name", "sector", "industry"]
            + sel_opp_models
            + [c for c in FACTOR_NAMES[:6] if c in opp_df.columns]
        )
        display_cols = [c for c in display_cols if c in opp_df.columns]

        styled = opp_df[display_cols].copy()
        for c in sel_opp_models:
            styled[c] = styled[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")

        st.dataframe(styled, use_container_width=True, hide_index=True, height=450)

        csv = opp_df[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button("Download opportunity set", csv, "opportunity_set.csv", "text/csv")

# ---- Factor Bubbles ----
with tab_bubbles:
    st.subheader("Factor scatter — explore relationships")
    st.caption("Plot any two factors against each other; size and color can encode a model score.")

    all_cols = FACTOR_NAMES + MODEL_COLS

    col1, col2, col3 = st.columns(3)
    x_axis = col1.selectbox("X axis", all_cols, index=0, key="bubble_x")
    y_axis = col2.selectbox("Y axis", all_cols, index=min(1, len(all_cols)-1), key="bubble_y")
    color_by = col3.selectbox("Color by", ["sector"] + MODEL_COLS, key="bubble_color")

    sel_bubble_sector = st.multiselect("Sector filter", sorted(screener["sector"].dropna().unique()), key="bubble_sector")

    plot_df = screener[["ticker", "company_name", "sector", x_axis, y_axis] +
                       ([color_by] if color_by not in ["sector", x_axis, y_axis] else [])].dropna(subset=[x_axis, y_axis])

    if sel_bubble_sector:
        plot_df = plot_df[plot_df["sector"].isin(sel_bubble_sector)]

    if len(plot_df) > 2000:
        plot_df = plot_df.sample(2000, random_state=42)

    if not plot_df.empty:
        fig_scatter = px.scatter(
            plot_df,
            x=x_axis,
            y=y_axis,
            color=color_by,
            hover_data=["ticker", "company_name", "sector"],
            color_continuous_scale="RdYlGn" if color_by in MODEL_COLS else None,
            color_continuous_midpoint=0 if color_by in MODEL_COLS else None,
            opacity=0.65,
            labels={x_axis: x_axis, y_axis: y_axis},
        )
        fig_scatter.update_layout(height=550, margin=dict(l=0, r=0, t=20, b=20))
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("No data to display.")

# ---- Top stocks by theme ----
with tab_top:
    st.subheader("Top & bottom stocks by model score")

    if not MODEL_COLS:
        st.info("No model data available.")
    else:
        sel_theme_model  = st.selectbox("Model", MODEL_COLS, key="theme_model")
        sel_theme_sector = st.selectbox(
            "Sector",
            ["All sectors"] + sorted(screener["sector"].dropna().unique().tolist()),
            key="theme_sector",
        )
        n_show = st.slider("Stocks to show (each side)", 5, 50, 20, key="theme_n")

        theme_df = screener.copy()
        if sel_theme_sector != "All sectors":
            theme_df = theme_df[theme_df["sector"] == sel_theme_sector]

        theme_df = theme_df[["ticker", "company_name", "sector", "industry", sel_theme_model]].dropna(
            subset=[sel_theme_model]
        )

        top_n    = theme_df.nlargest(n_show, sel_theme_model)
        bottom_n = theme_df.nsmallest(n_show, sel_theme_model)

        left_col, right_col = st.columns(2)

        with left_col:
            st.markdown(f"**Top {n_show} — {sel_theme_model}**")
            fig_top = px.bar(
                top_n.sort_values(sel_theme_model),
                x=sel_theme_model,
                y=top_n.sort_values(sel_theme_model).apply(
                    lambda r: r["ticker"] if r["ticker"] else r["company_name"][:25], axis=1
                ),
                orientation="h",
                color=sel_theme_model,
                color_continuous_scale="Greens",
                text=top_n.sort_values(sel_theme_model)[sel_theme_model].map(lambda x: f"{x:.2f}"),
            )
            fig_top.update_traces(textposition="outside")
            fig_top.update_layout(
                height=max(300, n_show * 22),
                coloraxis_showscale=False,
                margin=dict(l=0, r=40, t=10, b=10),
                xaxis_title="Score",
                yaxis_title="",
            )
            st.plotly_chart(fig_top, use_container_width=True)

        with right_col:
            st.markdown(f"**Bottom {n_show} — {sel_theme_model}**")
            fig_bot = px.bar(
                bottom_n.sort_values(sel_theme_model, ascending=False),
                x=sel_theme_model,
                y=bottom_n.sort_values(sel_theme_model, ascending=False).apply(
                    lambda r: r["ticker"] if r["ticker"] else r["company_name"][:25], axis=1
                ),
                orientation="h",
                color=sel_theme_model,
                color_continuous_scale="Reds_r",
                text=bottom_n.sort_values(sel_theme_model, ascending=False)[sel_theme_model].map(
                    lambda x: f"{x:.2f}"
                ),
            )
            fig_bot.update_traces(textposition="outside")
            fig_bot.update_layout(
                height=max(300, n_show * 22),
                coloraxis_showscale=False,
                margin=dict(l=0, r=40, t=10, b=10),
                xaxis_title="Score",
                yaxis_title="",
            )
            st.plotly_chart(fig_bot, use_container_width=True)
