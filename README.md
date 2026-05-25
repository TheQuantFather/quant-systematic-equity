# Systematic Equity Investing Framework

A full-stack quantitative investing system covering ~994 US equities from the **iShares Russell 1000 ETF** universe. Ingests EDGAR filings and price data, computes 28+ point-in-time factors across 9 models, estimates a Barra-style factor risk model (K=29), and runs a CVXPY portfolio optimiser with 9 configurable strategies.

## Architecture

```
iShares N-PORT-P (EDGAR)
  └─ create_universe.py      → universe.db           (company metadata, ISIN-based)

SimFin CSVs (initial load)
  └─ create_databases.py     → constituents.db        (financial time series, PIT)

EDGAR 10-K filings (incremental)
  └─ update_constituents.py  → constituents.db

SimFin CSVs
  └─ create_returns.py       → returns.db             (daily prices)

constituents.db + returns.db + universe.db
  └─ create_factors.py       → factors.db             (28+ factors × 28 snapshots)
  └─ create_models.py        → models.db              (9 models × 28 snapshots)
  └─ create_risk.py          → risk.db                (Ledoit-Wolf covariance, all snapshots)
  └─ create_barra.py         → risk.db                (Barra factor risk model, quarterly snapshots)

strategy_params.xlsx + models.db + risk.db
  └─ optimize_portfolio.py   → portfolio_output/      (weights + summary per strategy)

factors.db + models.db + universe.db + portfolio_output/ + risk.db
  └─ app.py + pages/         → Streamlit dashboard
```

## Running the pipeline

```bash
conda activate quant   # Python 3.13.5
```

### Full historical build
```bash
python create_universe.py
python create_databases.py
python create_returns.py
python create_svr.py --backfill
python create_factors.py --quarterly-backfill   # all quarterly snapshot dates
python create_models.py
python create_risk.py --backfill
python create_barra.py --backfill               # quarterly snapshots → risk.db
python create_strategy_params.py        # creates data/strategy_params.xlsx
python optimize_portfolio.py            # runs all active strategies
streamlit run app.py
```

### Rebuild universe snapshots only (leaves companies table intact)
```bash
python create_universe.py --rebuild-snapshots
```

### Weekly incremental update
```bash
python update_constituents.py [--limit N] [--ticker X] [--sector-type financial] [--force]
# --fill-gaps: pull missing annual 10-K years for a targeted subset of companies
python create_returns.py --update       # latest Yahoo Finance prices
python create_svr.py                    # latest FINRA short volume ratio (incremental)
python create_factors.py --date 2026-04-01
python create_models.py --date 2026-04-01
python create_risk.py --date 2026-04-01
python create_barra.py                  # defaults to most recent Friday → risk.db
# --date is repeatable for create_factors, create_models, create_barra: --date D1 --date D2
python optimize_portfolio.py
```

### Optimizer only
```bash
python optimize_portfolio.py --strategy core_active   # single strategy
python optimize_portfolio.py --list                   # list all strategies
```

## Factor model

### Snapshot dates

28 snapshots: 6 annual (April 1, ≥ 90-day lag for Dec FY-end filers) + 22 quarterly mid-period (15th of Feb/May/Aug/Nov). Dates are stored in `factors.db` `snapshot_dates` table and discovered automatically by `create_risk.py` and `create_barra.py` — no config list to maintain.

Annual snapshots:

| Snapshot | FY covered |
|----------|-----------|
| 2021-04-01 | FY2020 |
| 2022-04-01 | FY2021 |
| 2023-04-01 | FY2022 |
| 2024-04-01 | FY2023 |
| 2025-04-01 | FY2024 |
| 2026-04-01 | FY2025 |

### Point-in-time

Each snapshot uses the most recent annual report with `publish_date ≤ snapshot_date`. Prices as of `snapshot_date`. EDGAR rows use `acceptance_datetime`; SimFin rows use SimFin's publish_date.

### Factors (28+ total)

| Category | Count | Examples |
|----------|-------|---------|
| Quality | 15 | Gross Margin, ROE, ROA, Current Ratio, Leverage, Debt-to-Assets |
| Value | 5 | Earnings Yield, Book-to-Price, Sales-to-Price, Cash Yield, EV-to-EBIT |
| Growth | 5 | Revenue, Earnings, Cash Flow, Asset, Equity Growth |
| Momentum | 2 | 6M, 12M price momentum |
| Size | 1 | Log Market Cap |
| Low Volatility | 1 | Realized volatility (inverted) |
| Liquidity | 1 | Amihud illiquidity (inverted) |
| REIT-only | 3 | FFO Yield, FFO per Share, FFO Growth |

Direction is applied only at model score time (`z × weight × direction`); `factor_value_z` is always stored unsigned.

### Models (9 total)

| Model | ID | Components |
|-------|----|-----------|
| Quality | QUAL001 | 15 quality factors |
| Value | VAL001 | 5 value factors |
| Growth | GRO001 | 5 growth factors |
| Momentum | MOM001 | 2 momentum factors |
| Size | SIZ001 | Log Market Cap |
| Low Volatility | LVOL001 | Realized volatility |
| Liquidity | LIQ001 | Amihud illiquidity |
| Short Interest | SHI001 | FINRA SVR 20-day avg (70%) + 90-day percentile rank (30%) |
| Alpha (composite) | ALP001 | Equal-weight of Quality, Value, Growth, Momentum, Size |

## Barra factor risk model

### Overview

Barra-style factor covariance decomposition: **Σ = X F X' + Δ**

| Symbol | Description |
|--------|-------------|
| X (N×K) | Factor exposure matrix — sector dummies, style z-scores, beta, fundamentals |
| F (K×K) | Factor covariance — EWMA (hl=90d) + Newey-West (5 lags), annualised |
| Δ (N×N) | Diagonal idiosyncratic variance — EWMA (hl=60d), Bayesian-shrunk, annualised |

### Factor structure (K = 29)

| Group | Count | Factors |
|-------|-------|---------|
| Sector | 11 | All GICS sectors (no reference dropped; SVD handles rank deficiency) |
| Style | 5 | Log Market Cap, 12M momentum, 6M momentum, realized vol, 52-week high ratio |
| Beta | 1 | beta_60d vs equal-weight universe |
| Fundamental | 12 | Selected quality, value, growth, and leverage factors |

### Volatility Regime Adjustment (VRA)

Bias statistic B² = realized_var_ew / predicted_var_ew over 60-day window, clipped [0.25, 4.0]. Scales both F and Δ. Healthy values: ~0.8–0.9 (calm), ~1.2–1.3 (stress).

### Optimizer integration

Stacked-L drop-in: `L_barra = vstack([L_F.T @ X.T, diag(√δ)]).T` (shape N×K+N).  
`‖L_barra.T w‖² = w'(XFX'+Δ)w` — annual portfolio variance, consistent with `risk.db` convention.

### Per-strategy toggle

Set `use_barra_risk = FALSE` in the Strategies sheet of `strategy_params.xlsx` to force Ledoit-Wolf for a specific strategy. Default is Barra; falls back silently to Ledoit-Wolf on any load error.

## Portfolio optimiser

### Strategies (9 active)

| Strategy | Objective | Alpha signal | Universe |
|----------|-----------|-------------|---------|
| Core Active | maximize_alpha | Composite | Benchmark |
| Core Active (Strict) | maximize_alpha | Composite | Benchmark (2% TE) |
| Absolute Return | maximize_sharpe | Composite | Full (983 stocks) |
| Minimum Variance | minimize_variance | — | Full |
| Quality Compounder | maximize_sharpe | Quality only | Full (excl. Energy/Materials) |
| Defensive Income | maximize_sharpe | Quality + Low Vol | Full |
| Value Hunt | maximize_alpha | Value only | Benchmark (6% TE) |
| Momentum | maximize_sharpe | Momentum only | Full |
| All-Weather GARP | maximize_sharpe | Quality+Growth+Value | Full |

### Objectives

- **maximize_alpha** — active-weight SOCP vs benchmark. Requires `benchmark_file`.
- **maximize_sharpe** — Charnes-Cooper transform: solve for `y = w/σ_p`, recover `w = y/∑y`.
- **minimize_variance** — minimize `w'Σw`; ignores alpha signal.

### Solvers

- **CLARABEL** — default for continuous problems (no cardinality constraints).
- **MOSEK** — used automatically when `max_positions` or `min_position_if_held` integer constraints are active. License at `~/mosek/mosek.lic`.

### Configuration

Edit `data/strategy_params.xlsx` (4 sheets):
- **Strategies** — strategy_id, objective, benchmark_file, alpha_date, risk_date, solver, investable_universe, use_barra_risk
- **Constraints** — per-strategy constraint rows; toggle `enabled` TRUE/FALSE
- **Alpha_Weights** — model_id + weight rows; multiple rows per strategy for blending
- **Reference** — read-only guide to available models, objectives, constraints

Re-run `optimize_portfolio.py` (or click **▶ Run Optimisation** in the app) to apply changes.

## Dashboard pages

| Page | Description |
|------|-------------|
| Home | Universe summary, factor score distributions |
| Universe | Company search and metadata |
| Factors | Factor distributions, time series, peer comparisons |
| Screener | Multi-factor screener with export |
| Deep Dive | Single-stock factor attribution |
| Themes | Sector heatmaps, opportunity sets |
| Backtester | Historical factor backtest and strategy simulation |
| Database | Raw database explorer (all tables) |
| Portfolio | Strategy results: weights, sector/industry, factor tilts, risk attribution |
| Risk Explorer | Barra / Ledoit-Wolf deep-dive: correlations, factor vols, stock decomposition |
| Data Quality | Pipeline health: factor coverage, constituent fill rates, sync status across all DBs |

## Database schemas

### risk.db

Contains both Ledoit-Wolf covariance and Barra factor risk tables.

```sql
-- Ledoit-Wolf shrunk covariance (one row per snapshot date)
CREATE TABLE covariance_matrix (
    data_date        TEXT PRIMARY KEY,
    matrix_blob      BLOB,   -- zlib(numpy float32)
    isin_list        TEXT,   -- JSON array of ISINs
    n_stocks         INTEGER,
    shrinkage_coeff  REAL,
    lookback_days    INTEGER,
    computation_date TEXT
)

-- Barra: daily factor returns (used to estimate F)
CREATE TABLE factor_returns (
    trade_date TEXT, factor_id TEXT, factor_return REAL,
    PRIMARY KEY (trade_date, factor_id)
)
-- Barra: K×K factor covariance per snapshot
CREATE TABLE factor_covariance (
    snapshot_date TEXT PRIMARY KEY,
    factor_names  TEXT,   -- JSON array of K factor names
    cov_blob      BLOB    -- zlib(K×K float32), annualised
)
-- Barra: per-stock idiosyncratic variance per snapshot
CREATE TABLE idiosyncratic_vars (
    snapshot_date TEXT, security_id TEXT, idio_var REAL,
    PRIMARY KEY (snapshot_date, security_id)
)
-- Barra: factor exposures X per snapshot
CREATE TABLE factor_exposures (
    snapshot_date TEXT, security_id TEXT, factor_id TEXT, exposure REAL,
    PRIMARY KEY (snapshot_date, security_id, factor_id)
)
```

### factors.db
```sql
CREATE TABLE factors (
    data_date TEXT, factor_id TEXT, security_id TEXT,
    factor_value REAL, factor_value_z REAL,
    update_date TEXT, computation_date TEXT,
    PRIMARY KEY (data_date, factor_id, security_id)
)
```

### models.db
```sql
CREATE TABLE models (
    data_date TEXT, model_id TEXT, security_id TEXT,
    model_value REAL, model_value_z REAL, is_composite INTEGER DEFAULT 0,
    PRIMARY KEY (data_date, model_id, security_id)
)
```

## Project structure

```
├── app.py
├── pages/
│   ├── 1_Universe.py
│   ├── 2_Factors.py
│   ├── 3_Screener.py
│   ├── 4_Deep_Dive.py
│   ├── 5_Themes.py
│   ├── 6_Backtester.py
│   ├── 7_Database.py
│   ├── 8_Portfolio.py
│   ├── 9_Risk_Explorer.py
│   └── 10_Data_Quality.py
├── config.py                        # Single source of truth: all paths, dates, hyperparameters
├── db.py                           # Cached data access layer (Streamlit @st.cache_data wrappers)
├── utils.py                        # Shared utilities: get_db, classify_sector, winsorized_zscore, get_logger
├── create_universe.py
├── create_databases.py
├── update_constituents.py          # Incremental EDGAR 10-Q/10-K fetcher (logs to pull_log table)
├── create_returns.py
├── create_factors.py               # factors.db + snapshot_dates table
├── create_models.py
├── create_risk.py                  # Ledoit-Wolf covariance → risk.db
├── create_barra.py                 # Barra factor risk model → risk.db (same file)
├── create_strategy_params.py       # Reset strategy_params.xlsx template
├── optimize_portfolio.py           # CVXPY optimizer (3 objectives, 9 strategies)
├── exploratory/                    # Not committed — ad-hoc scripts
│   ├── degiro_orders.py
│   ├── explore_insider.py
│   ├── explore_short_interest.py
│   └── validate_constituents.py
├── data/
│   ├── universe.db
│   ├── constituents.db             # includes pull_log table
│   ├── returns.db
│   ├── factors.db                  # includes snapshot_dates table
│   ├── models.db
│   ├── risk.db                     # Ledoit-Wolf + Barra tables
│   ├── strategy_params.xlsx        # Strategy / constraint / alpha-weight config
│   ├── portfolio_output/           # {sid}_latest.csv + {sid}_latest_summary.json
│   ├── factors_reference.csv       # includes barra_factor_type + barra_factor_order columns
│   ├── models_reference.csv
│   ├── constituents_reference.csv
│   ├── edgar_concept_map.xlsx
│   └── universe_index/             # iShares Russell 1000 holdings CSVs
├── logs/                           # Rotating log files: <script_name>.log (5 MB, 3 backups)
├── CLAUDE.md
├── BACKLOG.md                      # Pending work and research ideas
└── .claude/
    └── commands/                   # Custom slash commands: /snapshot, /validate, /db-check
```

## Logging

All pipeline scripts use `get_logger(name)` from `utils.py` — no `print()` statements.

- **Log files**: `logs/<script_name>.log` — rotating at 5 MB, 3 backups retained.
- **Stdout**: every log line also mirrors to stdout with timestamp and level.
- **Format**: `YYYY-MM-DD HH:MM:SS  LEVEL     message`
- **Debug mode**: `LOG_LEVEL=DEBUG python create_factors.py --date 2026-04-01`

## Dependencies

```bash
pip install streamlit pandas numpy plotly openpyxl cvxpy scikit-learn clarabel
# MOSEK (optional, for integer/cardinality constraints): https://mosek.com/
```

## Disclaimer

For educational and research purposes only. Not investment advice.
