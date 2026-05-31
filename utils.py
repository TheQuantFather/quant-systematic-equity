"""
utils.py — Shared utilities for the factor pipeline.
"""

import logging
import logging.handlers
import os
import sqlite3
import sys
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
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

/* ── Font: apply Inter to content elements (not icon spans) ───────────────── */
html, body,
p, h1, h2, h3, h4, h5, h6, li, td, th, label, a,
button, input, select, textarea,
[class*="css"],
.stMarkdown, .stDataFrame, .stText,
[data-testid="stMarkdownContainer"],
[data-testid="stWidgetLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"],
[data-testid="stSidebarNavLink"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* ── Typography refinements ──────────────────────────────────────────────── */
h1 { font-size: 1.75rem !important; font-weight: 600 !important; letter-spacing: -0.02em !important; line-height: 1.3 !important; }
h2 { font-size: 1.35rem !important; font-weight: 600 !important; letter-spacing: -0.015em !important; }
h3 { font-size: 1.1rem  !important; font-weight: 500 !important; letter-spacing: -0.01em !important; }
p, li, .stMarkdown p { font-size: 0.9rem !important; line-height: 1.65 !important; }

/* ── Hide Streamlit chrome ───────────────────────────────────────────────── */
#MainMenu, footer, [data-testid="stDeployButton"], [data-testid="stDecoration"]
    { display: none !important; }

/* ── Layout ──────────────────────────────────────────────────────────────── */
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }

/* ── Metric cards ────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: rgba(59,130,246,0.07);
    border: 1px solid rgba(59,130,246,0.18);
    border-radius: 0.5rem;
    padding: 0.85rem 1rem;
}
[data-testid="stMetricValue"] { font-size: 1.35rem !important; font-weight: 600 !important; }
[data-testid="stMetricLabel"] { font-size: 0.75rem !important; font-weight: 500 !important; letter-spacing: 0.03em !important; text-transform: uppercase !important; opacity: 0.7 !important; }

/* ── Dividers ────────────────────────────────────────────────────────────── */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* ── Sidebar nav: capitalize "app" → "App" (first entry from app.py) ──────── */
[data-testid="stSidebarNav"] li:first-child a span { text-transform: capitalize !important; }
</style>
"""


_LOG_DIR = Path(__file__).parent / "logs"

_FMT = logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger that writes to logs/<name>.log and stdout.

    - Level: LOG_LEVEL env var (default INFO). Set LOG_LEVEL=DEBUG for verbose output.
    - Log files rotate at 5 MB, keeping 3 backups.
    - Calling get_logger() twice with the same name returns the same logger (idempotent).
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured — don't add duplicate handlers

    level = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False  # don't bubble up to the root logger

    _LOG_DIR.mkdir(exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        _LOG_DIR / f"{name}.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    fh.setFormatter(_FMT)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(_FMT)
    logger.addHandler(sh)

    return logger


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


def get_snapshot_schedule(
    cadence: str | tuple[str, ...] | None = None,
    computed_only: bool = False,
) -> list[str]:
    """
    Read the canonical snapshot dates from universe.db `snapshot_schedule` — the
    single source of truth for the pipeline.  Returns dates sorted ascending.

    cadence       — restrict to one cadence ('monthly'/'weekly'/'legacy') or a tuple.
    computed_only — only dates whose factors have been computed (factors_computed_at
                    set).  Used by create_risk / create_barra (they need factors first).
    """
    from config import UNIVERSE_DB   # lazy import to avoid import-time coupling

    sql = "SELECT data_date FROM snapshot_schedule"
    conds, params = [], []
    if computed_only:
        conds.append("factors_computed_at IS NOT NULL")
    if cadence is not None:
        cads = (cadence,) if isinstance(cadence, str) else tuple(cadence)
        conds.append(f"cadence IN ({','.join('?' * len(cads))})")
        params.extend(cads)
    if conds:
        sql += " WHERE " + " AND ".join(conds)
    sql += " ORDER BY data_date"
    with get_db(UNIVERSE_DB) as conn:
        return [r[0] for r in conn.execute(sql, params).fetchall()]


def mark_snapshot_computed(data_date: str) -> None:
    """Stamp factors_computed_at for a snapshot date in the schedule (called by create_factors)."""
    from datetime import datetime
    from config import UNIVERSE_DB

    with get_db(UNIVERSE_DB) as conn:
        # Insert if the date isn't in the schedule yet (e.g. ad-hoc --date runs), else stamp.
        conn.execute(
            "INSERT INTO snapshot_schedule (data_date, cadence, factors_computed_at, created_at) "
            "VALUES (?, 'adhoc', ?, ?) "
            "ON CONFLICT(data_date) DO UPDATE SET factors_computed_at = excluded.factors_computed_at",
            (data_date, datetime.now().isoformat(timespec="seconds"),
             datetime.now().isoformat(timespec="seconds")),
        )
        conn.commit()


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
