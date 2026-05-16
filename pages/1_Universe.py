"""
1_Universe.py — Universe overview: sector/industry distribution and coverage.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import db
from utils import inject_css

st.set_page_config(page_title="Universe Overview", layout="wide")
inject_css()
st.title("Universe Overview")

universe    = db.get_universe()
screener    = db.get_screener_df()
model_meta  = db.get_model_metadata()
MODEL_COL_NAMES = [f"{m} Model" for m in model_meta["Model"]]

# ---------------------------------------------------------------------------
# Top-level KPIs
# ---------------------------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total companies", f"{len(universe):,}")
c2.metric("With factor data", f"{len(screener):,}")
c3.metric("Sectors", universe["sector"].nunique())
c4.metric("Industries", universe["industry"].nunique())

st.divider()

# ---------------------------------------------------------------------------
# Sector distribution
# ---------------------------------------------------------------------------

st.subheader("Companies by sector")

sector_df = (
    screener.groupby("sector")
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=True)
)

fig_sector = px.bar(
    sector_df,
    x="count",
    y="sector",
    orientation="h",
    labels={"count": "Companies", "sector": ""},
    color="count",
    color_continuous_scale="Blues",
    text="count",
)
fig_sector.update_traces(textposition="outside")
fig_sector.update_layout(
    height=500,
    showlegend=False,
    coloraxis_showscale=False,
    margin=dict(l=0, r=40, t=20, b=20),
)
st.plotly_chart(fig_sector, use_container_width=True)

# ---------------------------------------------------------------------------
# Sector → Industry treemap
# ---------------------------------------------------------------------------

st.subheader("Sector / industry breakdown")

industry_df = (
    screener.groupby(["sector", "industry"])
    .size()
    .reset_index(name="count")
    .dropna(subset=["sector", "industry"])
)

fig_tree = px.treemap(
    industry_df,
    path=["sector", "industry"],
    values="count",
    color="count",
    color_continuous_scale="Blues",
    hover_data={"count": True},
)
fig_tree.update_layout(height=600, margin=dict(l=0, r=0, t=20, b=0))
fig_tree.update_traces(textinfo="label+value")
st.plotly_chart(fig_tree, use_container_width=True)

# ---------------------------------------------------------------------------
# Model score distribution by sector (box plots)
# ---------------------------------------------------------------------------

model_cols = [c for c in screener.columns if c in MODEL_COL_NAMES]

if model_cols:
    st.subheader("Model score distribution by sector")
    selected_model = st.selectbox("Model", model_cols, key="uni_model")

    box_df = screener[["sector", selected_model]].dropna()
    # Sort sectors by median score descending
    order = (
        box_df.groupby("sector")[selected_model]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig_box = px.box(
        box_df,
        x="sector",
        y=selected_model,
        category_orders={"sector": order},
        labels={"sector": "", selected_model: "Score"},
        color="sector",
    )
    fig_box.update_layout(
        height=450,
        showlegend=False,
        xaxis_tickangle=-35,
        margin=dict(l=0, r=0, t=20, b=120),
    )
    st.plotly_chart(fig_box, use_container_width=True)

# ---------------------------------------------------------------------------
# Raw universe table (filterable)
# ---------------------------------------------------------------------------

with st.expander("Browse universe"):
    sectors = sorted(universe["sector"].dropna().unique().tolist())
    sel_sectors = st.multiselect("Filter by sector", sectors, key="uni_browse_sector")
    show_df = universe if not sel_sectors else universe[universe["sector"].isin(sel_sectors)]
    st.dataframe(
        show_df[["ticker", "company_name", "sector", "industry", "exchange"]]
        .sort_values("company_name"),
        use_container_width=True,
        hide_index=True,
    )
