"""
config.py — Centralized configuration for the quantitative pipeline.

All database paths, reference file paths, portfolio paths, snapshot dates,
and model hyperparameters are defined here so they are changed in one place.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
LOG_DIR  = Path(__file__).parent / "logs"

# ---------------------------------------------------------------------------
# Database paths
# ---------------------------------------------------------------------------

UNIVERSE_DB      = DATA_DIR / "universe.db"
CONSTITUENTS_DB  = DATA_DIR / "constituents.db"
RETURNS_DB       = DATA_DIR / "returns.db"
FACTORS_DB       = DATA_DIR / "factors.db"
MODELS_DB        = DATA_DIR / "models.db"
RISK_DB          = DATA_DIR / "risk.db"   # Ledoit-Wolf covariance + Barra tables
MACRO_DB         = DATA_DIR / "macro.db"  # US macro signals: treasury yields, spreads, commodities, economic data

# ---------------------------------------------------------------------------
# Reference / mapping files
# ---------------------------------------------------------------------------

FACTORS_REF      = DATA_DIR / "factors_reference.csv"
MODELS_REF       = DATA_DIR / "models_reference.csv"
CONSTITUENTS_REF = DATA_DIR / "constituents_reference.csv"
CONCEPT_MAP_XLSX = DATA_DIR / "edgar_concept_map.xlsx"

# ---------------------------------------------------------------------------
# Portfolio files
# ---------------------------------------------------------------------------

OUTPUT_DIR    = DATA_DIR / "portfolio_output"
PARAMS_FILE   = DATA_DIR / "strategy_params.xlsx"
BENCHMARK_DIR = DATA_DIR / "universe_index"

# ---------------------------------------------------------------------------
# SimFin source data
# ---------------------------------------------------------------------------

SIMFIN_DIR   = DATA_DIR / "simfin"
PRICES_CSV   = SIMFIN_DIR / "us-shareprices-daily.csv"

# ---------------------------------------------------------------------------
# Snapshot / backfill dates
# ---------------------------------------------------------------------------

# Annual April-1 snapshots — one per fiscal year (used by factors, risk, barra)
BACKFILL_DATES = [
    "2021-04-01", "2022-04-01", "2023-04-01",
    "2024-04-01", "2025-04-01", "2026-04-01",
]

# Quarterly mid-period snapshots (~45 days after each quarter-end)
QUARTERLY_BACKFILL_DATES = [
    "2022-05-15", "2022-08-15", "2022-11-15",
    "2023-02-15", "2023-05-15", "2023-08-15", "2023-11-15",
    "2024-02-15", "2024-05-15", "2024-08-15", "2024-11-15",
    "2025-02-15", "2025-05-15", "2025-08-15", "2025-11-15",
    "2026-02-15",
]

# Ledoit-Wolf risk snapshot dates align with annual backfill
RISK_SNAPSHOT_DATES = BACKFILL_DATES

# ---------------------------------------------------------------------------
# Snapshot schedule — the SINGLE SOURCE OF TRUTH for snapshot dates is the
# `snapshot_schedule` table in universe.db (built by `create_universe.py
# --rebuild-schedule`, read everywhere via `utils.get_snapshot_schedule`).
# These two parameters define the generation RULE: a month-end monthly grid from
# SCHEDULE_MONTHLY_START up to SCHEDULE_WEEKLY_CUTOVER, beyond which the weekly
# cadence (added by daily_update.py) takes over.  BACKFILL_DATES /
# QUARTERLY_BACKFILL_DATES above are retained only as legacy cadence tags.
# ---------------------------------------------------------------------------
SCHEDULE_MONTHLY_START  = "2021-04-30"   # first month-end of the monthly grid
SCHEDULE_WEEKLY_CUTOVER = "2026-05-01"   # weekly cadence takes over on/after this date

# Macro signal backfill dates
MACRO_BACKFILL_START = "2015-01-01"  # 10 years of history

# ---------------------------------------------------------------------------
# Ledoit-Wolf covariance hyperparameters
# ---------------------------------------------------------------------------

LW_LOOKBACK_DAYS = 252   # 1 year of trading days
LW_MIN_HISTORY   = 126   # minimum valid days to include a stock (~6 months)
LW_WINSOR_CLIP   = 0.50  # clip daily returns at ±50% before LW estimation

# ---------------------------------------------------------------------------
# Barra factor risk model hyperparameters
# ---------------------------------------------------------------------------

HL_FACTOR_VAR  = 90      # EWMA half-life: factor variances (diag of F) — Newey-West applied here
HL_FACTOR_CORR = 240     # EWMA half-life: factor correlations (off-diag) — longer memory, no NW
HL_IDIO        = 60      # EWMA half-life: idiosyncratic variance
NW_LAGS        = 5       # Newey-West autocorrelation correction lags (variance only)
VRA_WINDOW     = 60      # VRA bias-statistic look-back (trading days)
SHRINK_IDIO    = 0.10    # Bayesian shrinkage weight toward cross-sectional mean
EIGENFLOOR     = 1e-6    # spectral floor for factor covariance (ensures PSD)
VRA_MIN        = 0.50    # lower clip for VRA bias statistic B² (factor & specific)
VRA_MAX        = 2.00    # upper clip for B²
MIN_STOCKS     = 50      # minimum stocks per day to run cross-sectional regression

# ---------------------------------------------------------------------------
# Barra sector definitions — order determines column indices 0-10 in X matrix.
# Style and fundamental factor IDs are loaded dynamically from FACTORS_REF
# (barra_factor_type / barra_factor_order columns) in create_barra.py and
# pages/9_Risk_Explorer.py.  Do not reorder BARRA_SECTORS without rebuilding
# risk.db (Barra tables).
# ---------------------------------------------------------------------------

BARRA_SECTORS = [
    "Communication Services", "Consumer Discretionary", "Consumer Staples",
    "Energy", "Financials", "Health Care", "Industrials",
    "Information Technology", "Materials", "Real Estate", "Utilities",
]
