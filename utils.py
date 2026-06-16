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


def get_barra_layout() -> dict:
    """Single source of truth for the Barra factor layout.

    Reads the Barra risk-factor tags from models_reference.csv
    (``barra_risk_factor`` / ``barra_order`` / ``barra_ortho_against``) and combines
    them with the structural blocks (market intercept, GICS sector dummies, beta) to
    produce the ordered factor vector shared by create_barra.py and the risk pages.

    Layout: ``[market | sectors | beta | models]``.

    Returns a dict with:
      sectors        list[str]                 GICS sector names (BARRA_SECTORS order)
      model_factors  list[(id, name, ortho)]   risk-factor models, ordered by barra_order;
                                                ortho = ModelID to residualise against, or None
      factor_names   list[str]                 full ordered factor-id vector (length K)
      groups         dict[str, int | slice]    display groups {"Market","Sector","Style"} ->
                                                index/slice. Style folds beta + model factors
                                                (a lone beta group adds no value).
      anchors        dict[str, int]            structural column anchors for the pipeline:
                                                market_idx / sector_start / sector_end /
                                                beta_idx / model_start / model_end
      factor_group   dict[str, str]            factor_id -> display group label
      pretty         dict[str, str]            factor_id -> human-readable display name
    """
    from config import MODELS_REF, BARRA_SECTORS

    df = pd.read_csv(MODELS_REF).drop_duplicates("ModelID")
    df = df[df["barra_risk_factor"].astype(str).str.strip().str.lower() == "true"]
    df = df.sort_values("barra_order")

    def _ortho(v) -> str | None:
        s = "" if pd.isna(v) else str(v).strip()
        return s or None

    model_factors = [(r.ModelID, r.Model, _ortho(r.barra_ortho_against))
                     for r in df.itertuples(index=False)]

    sectors    = list(BARRA_SECTORS)
    sector_ids = [f"sec_{s.replace(' ', '_').lower()}" for s in sectors]
    model_ids  = [mid for mid, _, _ in model_factors]

    factor_names = ["market"] + sector_ids + ["beta_60d"] + model_ids

    n_sec       = len(sector_ids)
    beta_idx    = 1 + n_sec
    model_start = beta_idx + 1
    model_end   = model_start + len(model_ids)

    # Structural anchors (fixed column positions) for the pipeline.
    anchors = {
        "market_idx":   0,
        "sector_start": 1,
        "sector_end":   1 + n_sec,
        "beta_idx":     beta_idx,
        "model_start":  model_start,
        "model_end":    model_end,
    }

    # Display groups: Market | Sector | Style. Style folds beta + the model
    # factors so there is no single-factor "Beta" group.
    groups = {
        "Market": 0,
        "Sector": slice(1, 1 + n_sec),
        "Style":  slice(beta_idx, model_end),
    }

    factor_group = {"market": "Market", "beta_60d": "Style"}
    pretty       = {"market": "Market", "beta_60d": "Beta (60d)"}
    for sid, sname in zip(sector_ids, sectors):
        factor_group[sid] = "Sector"
        pretty[sid]       = sname
    for mid, mname, _ in model_factors:
        factor_group[mid] = "Style"
        pretty[mid]       = mname

    return {
        "sectors":       sectors,
        "model_factors": model_factors,
        "factor_names":  factor_names,
        "groups":        groups,
        "anchors":       anchors,
        "factor_group":  factor_group,
        "pretty":        pretty,
    }


def classify_sector(sector: str | None, industry: str | None) -> str:
    """Map SimFin sector/industry to one of: 'reit', 'bank', 'financial', 'general'.

    'bank' = depository banks + consumer lenders (Credit Services: AmEx, Capital
    One, Discover, Sallie Mae…) — they have net interest income, loans and loan-loss
    provisions, so they get the bank-specific factors. Insurers, asset managers and
    brokers/exchanges stay 'financial' (their net interest income is incidental,
    not a lending model) and are scored on the generic 'all' factors.
    """
    s = (sector   or "").lower()
    i = (industry or "").lower()
    if s == "real estate" or "reit" in i:
        return "reit"
    if s == "financial services":
        # Exact SimFin industry strings — 'Banks' and 'Credit Services' are the
        # depository/consumer lenders. Investment banks sit under 'Brokers,
        # Exchanges & Other' and are NOT matched here (no naive "bank" substring).
        if i in ("banks", "credit services"):
            return "bank"
        return "financial"
    return "general"


# Which factor sector_types each company sector_type is allowed to receive.
# Single source of truth shared by the factor layer (create_factors gating) and
# the model layer (create_models coverage denominator) so both agree on which
# factors apply to a company.
ALLOWED_FACTOR_SECTORS: dict[str, set] = {
    'general':   {'all', 'general'},
    'financial': {'all'},              # insurers/asset-mgrs: revenue/margin factors don't fit
    'bank':      {'all', 'bank'},      # banks: 'all' factors + bank-specific; skip general revenue/margin
    'reit':      {'all', 'general', 'reit'},
}


def factor_applies_to_company(factor_sector_type: str | None,
                              company_sector_type: str | None) -> bool:
    """True if a factor with `factor_sector_type` is computed for a company of
    `company_sector_type` (mirrors create_factors' sector gating)."""
    allowed = ALLOWED_FACTOR_SECTORS.get(company_sector_type or 'general', {'all', 'general'})
    return (factor_sector_type or 'all') in allowed


def apply_weight_cap(weights: dict[str, float], cap: float = 0.03) -> dict[str, float]:
    """
    Redistribute weight above `cap` to uncapped names pro-rata until convergence.

    Standard approach for capped-index construction (e.g. S&P 500 3% Capped).
    Each pass fixes at least one additional name at the cap and rescales the
    remaining names to the residual budget, guaranteeing termination.
    Returns a normalised dict (weights sum to 1); zero-weight names are dropped.
    """
    if cap <= 0 or cap > 1:
        raise ValueError("cap must be in the interval (0, 1]")

    w = {k: v for k, v in weights.items() if v > 0}
    total = sum(w.values())
    if total <= 0:
        return {}
    w = {k: v / total for k, v in w.items()}

    if len(w) * cap < 1 - 1e-12:
        raise ValueError("cap is infeasible for the number of positive-weight names")

    capped: set[str] = set()
    for _ in range(len(w)):
        remaining = [k for k in w if k not in capped]
        if not remaining:
            break

        target = 1.0 - cap * len(capped)
        rem_total = sum(w[k] for k in remaining)
        if rem_total <= 0:
            break

        scale = target / rem_total
        for k in remaining:
            w[k] *= scale

        newly_capped = {k for k in remaining if w[k] > cap + 1e-12}
        if not newly_capped:
            break

        for k in newly_capped:
            w[k] = cap
        capped.update(newly_capped)

    # Remove harmless floating-point dust without creating a cap breach.
    residual = 1.0 - sum(w.values())
    if abs(residual) > 1e-12:
        room = {k: cap - v for k, v in w.items() if v < cap - 1e-12}
        room_total = sum(room.values())
        if residual > 0 and room_total > 0:
            for k, available in room.items():
                w[k] += residual * available / room_total
        else:
            scale = 1.0 / sum(w.values())
            w = {k: v * scale for k, v in w.items()}

    return w


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
