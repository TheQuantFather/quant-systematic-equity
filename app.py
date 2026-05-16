"""
app.py — Home page for the Quant Factor Dashboard.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import db
from config import FACTORS_DB
from utils import get_db, inject_css

st.set_page_config(
    page_title="Quant Factor Dashboard",
    page_icon="📊",
    layout="wide",
)
inject_css()

st.title("Quant Factor Dashboard")
st.markdown("Systematic factor analysis across the US equity universe.")

# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

with st.spinner("Loading data…"):
    universe    = db.get_universe()
    factor_meta = db.get_factor_metadata()
    model_meta  = db.get_model_metadata()
    screener    = db.get_screener_df()

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Companies with factor data", f"{len(screener):,}")
col2.metric("Total universe", f"{len(universe):,}")
col3.metric("Factors", len(factor_meta))
col4.metric("Models", len(model_meta))
col5.metric("Sectors covered", universe["sector"].nunique())

st.divider()

# ---------------------------------------------------------------------------
# Quick stats panels
# ---------------------------------------------------------------------------

left, right = st.columns(2)

with left:
    st.subheader("Factor categories")
    cat_counts = factor_meta.groupby("category").size().reset_index(name="count")
    cat_counts = cat_counts.sort_values("count", ascending=False)
    for _, row in cat_counts.iterrows():
        st.markdown(f"- **{row['category']}** — {row['count']} factors")

    st.subheader("Models")
    for _, row in model_meta.iterrows():
        st.markdown(f"- **{row['Model']}** (`{row['ModelID']}`)")

with right:
    st.subheader("Top sectors by company count")
    sector_counts = (
        screener.groupby("sector")
        .size()
        .reset_index(name="companies")
        .sort_values("companies", ascending=False)
        .head(10)
    )
    st.dataframe(sector_counts, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Snapshot dates
# ---------------------------------------------------------------------------

with get_db(FACTORS_DB) as conn:
    snapshots = pd.read_sql("SELECT DISTINCT data_date FROM factors ORDER BY data_date", conn)["data_date"].tolist()

latest = snapshots[-1] if snapshots else "N/A"
st.markdown(f"**Available snapshots ({len(snapshots)}):** {' · '.join(snapshots)}")
st.caption(f"Latest snapshot: **{latest}**. Navigate using the sidebar to explore the universe, screen stocks, or deep-dive into individual names.")
