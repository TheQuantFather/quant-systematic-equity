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

# ---------------------------------------------------------------------------
# Database paths
# ---------------------------------------------------------------------------

UNIVERSE_DB      = DATA_DIR / "universe.db"
CONSTITUENTS_DB  = DATA_DIR / "constituents.db"
RETURNS_DB       = DATA_DIR / "returns.db"
FACTORS_DB       = DATA_DIR / "factors.db"
MODELS_DB        = DATA_DIR / "models.db"
RISK_DB          = DATA_DIR / "risk.db"
BARRA_DB         = DATA_DIR / "barra.db"

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
# Ledoit-Wolf covariance hyperparameters
# ---------------------------------------------------------------------------

LW_LOOKBACK_DAYS = 252   # 1 year of trading days
LW_MIN_HISTORY   = 126   # minimum valid days to include a stock (~6 months)
LW_WINSOR_CLIP   = 0.50  # clip daily returns at ±50% before LW estimation

# ---------------------------------------------------------------------------
# Barra factor risk model hyperparameters
# ---------------------------------------------------------------------------

HL_FACTOR_COV = 90       # EWMA half-life: factor covariance (trading days)
HL_IDIO       = 60       # EWMA half-life: idiosyncratic variance
NW_LAGS       = 5        # Newey-West autocorrelation correction lags
VRA_WINDOW    = 60       # VRA bias-statistic look-back (trading days)
SHRINK_IDIO   = 0.10     # Bayesian shrinkage weight toward cross-sectional mean
EIGENFLOOR    = 1e-6     # spectral floor for factor covariance (ensures PSD)
VRA_MIN       = 0.25     # lower clip for VRA bias statistic B²
VRA_MAX       = 4.00     # upper clip for B²
MIN_STOCKS    = 50       # minimum stocks per day to run cross-sectional regression

BARRA_BACKFILL_START = "2020-01-01"

# ---------------------------------------------------------------------------
# Barra factor definitions — order determines column index in the X matrix;
# do not reorder without rebuilding barra.db.
# ---------------------------------------------------------------------------

BARRA_SECTORS = [
    "Communication Services", "Consumer Discretionary", "Consumer Staples",
    "Energy", "Financials", "Health Care", "Industrials",
    "Information Technology", "Materials", "Real Estate", "Utilities",
]

BARRA_STYLE_IDS = ["LMC11234", "ABC11234", "XYZ77890", "RVL11234", "W52H1234"]

BARRA_FUNDAMENTAL_IDS = [
    "TUV44567",  # Earnings Yield
    "WXY77890",  # Book-to-Price
    "JKL44556",  # ROE
    "ABC12345",  # Gross Margin
    "DEF67890",  # Operating Margin
    "BCD44567",  # Leverage
    "EFG77890",  # Debt-to-Assets
    "OPQ77890",  # Revenue Growth
    "LMN44567",  # Earnings Growth
    "KLM44567",  # Asset Turnover
    "YZA11234",  # Current Ratio
    "FCM11234",  # FCF Margin
]

# Derived slice indices into the K-length Barra factor vector
_N_SECTOR = len(BARRA_SECTORS)           # 11
_N_STYLE  = len(BARRA_STYLE_IDS)         # 5
_N_BETA   = 1
_N_FUND   = len(BARRA_FUNDAMENTAL_IDS)   # 12

BARRA_GROUPS = {
    "Sector":      slice(0, _N_SECTOR),
    "Style":       slice(_N_SECTOR, _N_SECTOR + _N_STYLE),
    "Beta":        _N_SECTOR + _N_STYLE,   # integer index — single factor
    "Fundamental": slice(_N_SECTOR + _N_STYLE + _N_BETA,
                         _N_SECTOR + _N_STYLE + _N_BETA + _N_FUND),
}
