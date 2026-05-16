"""
utils.py — Shared utilities for the factor pipeline.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import plotly.io as pio

# Dark plotly theme for all charts — applied once at import time
pio.templates.default = "plotly_dark"

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stDataFrame, button, input, select, textarea
    { font-family: 'Inter', sans-serif !important; }
#MainMenu, footer, [data-testid="stDeployButton"], [data-testid="stDecoration"]
    { display: none !important; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }
[data-testid="stMetric"] {
    background: rgba(59,130,246,0.07);
    border: 1px solid rgba(59,130,246,0.18);
    border-radius: 0.5rem;
    padding: 0.85rem 1rem;
}
[data-testid="stMetricValue"] { font-size: 1.35rem !important; }
hr { border-color: rgba(255,255,255,0.08) !important; }
</style>
"""


def inject_css() -> None:
    """Call once per page after st.set_page_config to apply global styles."""
    import streamlit as st
    st.markdown(_CSS, unsafe_allow_html=True)


@contextmanager
def get_db(path: str | Path) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for SQLite connections — always closes, even on exception."""
    conn = sqlite3.connect(str(path))
    try:
        yield conn
    finally:
        conn.close()


def classify_sector(sector: str | None, industry: str | None) -> str:
    """Map SimFin sector/industry to one of: 'reit', 'financial', 'general'."""
    s = (sector   or "").lower()
    i = (industry or "").lower()
    if s == "real estate" or "reit" in i:
        return "reit"
    if s == "financial services":
        return "financial"
    return "general"


def winsorized_zscore(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """
    Cross-sectional winsorize at [lower, upper] percentiles, then z-score.

    Returns a Series of the same shape.  Values in groups with fewer than 10
    observations or zero standard deviation are returned as NaN so they are
    excluded from downstream model scoring rather than distorting the cross-section.
    """
    if len(series) < 10:
        return pd.Series(np.nan, index=series.index)
    lo, hi = series.quantile(lower), series.quantile(upper)
    clipped = series.clip(lo, hi)
    mu, sigma = clipped.mean(), clipped.std()
    if sigma == 0:
        return pd.Series(np.nan, index=series.index)
    return (clipped - mu) / sigma
