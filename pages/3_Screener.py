"""
3_Screener.py — Multi-factor stock screener with ranked output.
"""

import streamlit as st
import plotly.express as px
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import db
from utils import inject_css

st.set_page_config(page_title="Stock Screener", layout="wide")
inject_css()
st.title("Stock Screener")

screener    = db.get_screener_df()
factor_meta = db.get_factor_metadata()
model_meta  = db.get_model_metadata()

MODEL_COL_NAMES = [f"{m} Model" for m in model_meta["Model"]]
MODEL_COLS      = [c for c in screener.columns if c in MODEL_COL_NAMES]

FACTOR_NAMES = [f for f in factor_meta["factor_name"].tolist() if f in screener.columns]

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Filters")

    # Sector
    sectors = sorted(screener["sector"].dropna().unique().tolist())
    sel_sectors = st.multiselect("Sector", sectors, placeholder="All sectors")

    # Industry (dynamic based on sector selection)
    if sel_sectors:
        avail_industries = sorted(
            screener[screener["sector"].isin(sel_sectors)]["industry"].dropna().unique().tolist()
        )
    else:
        avail_industries = sorted(screener["industry"].dropna().unique().tolist())
    sel_industries = st.multiselect("Industry", avail_industries, placeholder="All industries")

    st.divider()

    # Model score range filters
    st.subheader("Model score filters")
    model_filters = {}
    for m in MODEL_COLS:
        col_data = screener[m].dropna()
        mn, mx = float(col_data.min()), float(col_data.max())
        lo, hi = st.slider(
            m,
            min_value=round(mn, 2),
            max_value=round(mx, 2),
            value=(round(mn, 2), round(mx, 2)),
            key=f"slider_{m}",
        )
        model_filters[m] = (lo, hi)

    st.divider()

    # Sort options
    st.subheader("Sort by")
    sort_options = MODEL_COLS + FACTOR_NAMES
    sort_col = st.selectbox("Column", sort_options, key="sort_col")
    sort_asc = st.toggle("Ascending", value=False)

    # Max rows
    max_rows = st.number_input("Max rows", min_value=10, max_value=5000, value=100, step=10)

# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------

df = screener.copy()

if sel_sectors:
    df = df[df["sector"].isin(sel_sectors)]
if sel_industries:
    df = df[df["industry"].isin(sel_industries)]

for m, (lo, hi) in model_filters.items():
    df = df[df[m].between(lo, hi) | df[m].isna()]

# Sort
if sort_col in df.columns:
    df = df.sort_values(sort_col, ascending=sort_asc)

df = df.head(max_rows)

# ---------------------------------------------------------------------------
# Results summary
# ---------------------------------------------------------------------------

st.markdown(f"**{len(df):,}** companies match your filters")

# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

display_cols = (
    ["ticker", "company_name", "sector", "industry"]
    + MODEL_COLS
    + [c for c in FACTOR_NAMES[:8] if c in df.columns]
)
display_cols = [c for c in display_cols if c in df.columns]

# Format numeric columns
styled = df[display_cols].copy()

numeric_cols = [c for c in MODEL_COLS + FACTOR_NAMES if c in styled.columns]
for c in numeric_cols:
    styled[c] = styled[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")

st.dataframe(
    styled,
    use_container_width=True,
    hide_index=True,
    height=500,
    column_config={
        "ticker": st.column_config.TextColumn("Ticker", width="small"),
        "company_name": st.column_config.TextColumn("Company"),
        "sector": st.column_config.TextColumn("Sector"),
        "industry": st.column_config.TextColumn("Industry"),
    },
)

# ---------------------------------------------------------------------------
# Score distribution for the filtered set
# ---------------------------------------------------------------------------

if MODEL_COLS and len(df) > 1:
    st.subheader("Score distribution (filtered set)")
    sel_model_chart = st.selectbox("Model", MODEL_COLS, key="screen_chart_model")
    plot_df = df[["company_name", sel_model_chart]].dropna()
    fig = px.histogram(
        plot_df,
        x=sel_model_chart,
        nbins=50,
        labels={sel_model_chart: "Score", "count": "Companies"},
        color_discrete_sequence=["#2563EB"],
    )
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=20), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

csv = df[display_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download results as CSV",
    data=csv,
    file_name="screener_results.csv",
    mime="text/csv",
)
