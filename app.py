"""
app.py — Home page for the Systematic Equity Investment Framework.
Run with: streamlit run app.py
"""

import datetime as _dt
from pathlib import Path

import streamlit as st
import pandas as pd
import db
from config import FACTORS_DB, RETURNS_DB
from utils import get_db, inject_css

st.set_page_config(
    page_title="Systematic Equity Investment Framework",
    page_icon="📊",
    layout="wide",
)
inject_css()

st.title("Systematic Equity Investment Framework")
st.markdown(
    "An end-to-end quantitative equity pipeline — point-in-time fundamentals, "
    "factor and model construction, Barra risk, and portfolio optimisation over "
    "the Russell 1000."
)

# ---------------------------------------------------------------------------
# Data-freshness banner — catches a silently-dead daily_ecosystem_update.
# `last_yahoo_update` is stamped by create_returns.py on every price refresh;
# if it lags by more than a few trading days the scheduled job has stopped.
# (Lightweight alarm only — full pipeline health lives on the Data Quality page.)
# ---------------------------------------------------------------------------

with get_db(RETURNS_DB) as conn:
    _last = conn.execute(
        "SELECT value FROM metadata WHERE key = 'last_yahoo_update'"
    ).fetchone()

if _last and _last[0]:
    _last_date = _dt.date.fromisoformat(_last[0])
    _stale_bd = len(pd.bdate_range(_last_date + _dt.timedelta(days=1), _dt.date.today()))
    if _stale_bd >= 3:
        st.error(
            f"⚠️ Prices last updated **{_last_date}** — {_stale_bd} trading days stale. "
            "The daily ecosystem update has likely stopped; run "
            "`python daily_ecosystem_update.py` and check the launchd job."
        )
    elif _stale_bd >= 1:
        st.warning(
            f"Prices last updated **{_last_date}** ({_stale_bd} trading day(s) ago)."
        )

# ---------------------------------------------------------------------------
# Scale of the framework
# ---------------------------------------------------------------------------

with st.spinner("Loading data…"):
    universe    = db.get_universe()
    factor_meta = db.get_factor_metadata()
    model_meta  = db.get_model_metadata()
    screener    = db.get_screener_df()

with get_db(FACTORS_DB) as conn:
    snapshots = pd.read_sql(
        "SELECT DISTINCT data_date FROM factors ORDER BY data_date", conn
    )["data_date"].tolist()
latest = snapshots[-1] if snapshots else "N/A"

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Universe", f"{len(universe):,}")
col2.metric("Companies scored", f"{len(screener):,}")
col3.metric("Factors", len(factor_meta))
col4.metric("Models", len(model_meta))
col5.metric("Latest snapshot", latest)

st.divider()

# ---------------------------------------------------------------------------
# Where to go — annotated launchpad (links resolve against this entrypoint)
# ---------------------------------------------------------------------------

st.subheader("Where to go")
nav_research, nav_portfolio, nav_data = st.columns(3)

with nav_research:
    st.markdown("**Research**")
    st.page_link("pages/5_Opportunities.py", label="Opportunities", icon="🔍")
    st.page_link("pages/4_Deep_Dive.py", label="Deep Dive", icon="🔬")
    st.page_link("pages/6_Backtester.py", label="Backtester", icon="📈")
    st.page_link("pages/13_Model_Architecture.py", label="Model Architecture", icon="🧬")
    st.caption("Screen the universe, dig into a name, test a signal, see how models are built.")

with nav_portfolio:
    st.markdown("**Portfolio & risk**")
    st.page_link("pages/8_Portfolio_Optimiser.py", label="Portfolio Optimiser", icon="🎯")
    st.page_link("pages/9_Risk_Explorer.py", label="Risk Explorer", icon="⚡")
    if Path("pages/12_Portfolio_Analytics.py").exists():
        st.page_link("pages/12_Portfolio_Analytics.py", label="Portfolio Analytics", icon="💼")
    st.caption("Build and rebalance the book, then inspect its Barra factor risk.")

with nav_data:
    st.markdown("**Data & monitoring**")
    st.page_link("pages/10_Data_Quality.py", label="Data Quality", icon="🩺")
    st.page_link("pages/11_Macro.py", label="Macro", icon="🌐")
    st.page_link("pages/7_Database.py", label="Database", icon="🗄️")
    if Path("pages/14_Intraday.py").exists():
        st.page_link("pages/14_Intraday.py", label="Intraday", icon="⏱️")
    st.caption("Pipeline health & coverage, macro signals, and the raw tables.")

st.divider()

# ---------------------------------------------------------------------------
# What's in the framework — factor categories + models
# ---------------------------------------------------------------------------

left, right = st.columns(2)

with left:
    st.subheader("Factors by category")
    cat_counts = (
        factor_meta.groupby("category").size()
        .reset_index(name="Factors").rename(columns={"category": "Category"})
        .sort_values("Factors", ascending=False)
    )
    st.dataframe(cat_counts, use_container_width=True, hide_index=True)

with right:
    st.subheader("Models")
    models_tbl = model_meta.assign(
        Type=model_meta["IsComposite"].map({1: "Composite", 0: "Base"})
        if "IsComposite" in model_meta.columns else "—"
    )[["ModelID", "Model", "Type"]].rename(columns={"ModelID": "ID"})
    st.dataframe(models_tbl, use_container_width=True, hide_index=True,
                 height=min(38 * (len(models_tbl) + 1), 460))

st.divider()

# ---------------------------------------------------------------------------
# Snapshot history — one-line summary, full list on demand
# ---------------------------------------------------------------------------

if snapshots:
    st.markdown(
        f"**{len(snapshots)} factor snapshots** · {snapshots[0]} → {latest}"
    )
    with st.expander("All snapshot dates"):
        st.text(" · ".join(snapshots))
st.caption(
    "Coverage, freshness and validation checks live on the **Data Quality** page. "
    "Pick a destination above or use the sidebar."
)
