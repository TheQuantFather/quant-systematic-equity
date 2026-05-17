# Quant Factor Dashboard — Copilot Instructions

## Project overview

Systematic quantitative factor investing framework with a Streamlit dashboard. ~994 US equities from the iShares Russell 1000 ETF (IWB), 28+ factors, 6 models, 6 annual point-in-time snapshots. Primary identifier across all databases is **ISIN**.

## Python environment

Always use the `quant` conda environment (Python 3.13.5):
```bash
conda activate quant
# or directly: /Users/shivam/opt/anaconda3/envs/quant/bin/python3.13
```

## Architecture

```
iShares NPORT-P (EDGAR)  → create_universe.py    → universe.db  (company metadata + snapshots)
SimFin CSVs              → create_databases.py   → constituents.db  (historical financials)
EDGAR 10-K               → update_constituents.py → constituents.db (incremental updates)
SimFin CSVs              → create_returns.py     → returns.db   (daily prices)
constituents.db +
  returns.db + universe.db → create_factors.py  → factors.db   (28+ factors, z-scored per snapshot)
factors.db               → create_models.py     → models.db    (6 models, z-scored per snapshot)
factors.db + models.db +
  universe.db            → app.py + pages/       → Streamlit dashboard (reads via db.py)
```

## Key design decisions

### Security identifier
All databases use **ISIN** as `security_id` (e.g., `US0378331005`). Never use `str(simfin_id)` as a security identifier in new code.

### Point-in-time (no look-ahead bias)
- Per company, each snapshot uses the most recent annual report where `publish_date ≤ snapshot_date`
- Prices are referenced **as of the snapshot date**, not the report date
- Fallback when no publish date: `fiscal_year_end + 90 days`
- EDGAR rows: `acceptance_datetime` as `publish_date`. SimFin rows: SimFin's `publish_date`.

### Snapshot dates
Annual snapshots on April 1 following each fiscal year (defined in `BACKFILL_DATES` in create_factors.py and create_models.py): 2021-04-01 through 2026-04-01.

### Factor directionality
`direction` column in `factors_reference.csv` (1 or −1). **Never flip raw factor values or z-scores** — direction is applied only inside `compute_base_models()` in create_models.py as `z × weight × direction`.

Factors with `direction = -1`: Leverage, Debt-to-Assets, Capex Intensity, EV-to-EBIT, Working Capital Efficiency.
Factors with `direction = 1`: all others, including Log Market Cap (larger = better in Size model).

### Z-scores
Winsorized (1%–99%) then standardised cross-sectionally within each `(data_date, factor_id)` group. Function lives in `utils.py`. Groups with < 10 observations return NaN.

### Incremental pipeline
Both `create_factors.py` and `create_models.py` use `INSERT OR REPLACE` + `CREATE TABLE IF NOT EXISTS`, so running `--date` adds to existing databases without rebuilding.

### Sector-type classification
`create_factors.py` classifies each company as `general` / `financial` / `reit` from universe.db. REITs get 3 extra FFO-based factors. Financial companies skip Revenue/Gross Profit factors.

### Annual balance sheet period convention
Balance sheets: `fiscal_period = 'Q4'`. Income statements and cash flow: `fiscal_period = 'FY'`. Always filter `WHERE fiscal_period IN ('FY', 'Q4')` for annual data.

## Database schemas

### universe.db
```sql
-- companies: one row per ISIN
PRIMARY KEY (isin)
-- key cols: isin, ticker, company_name, gics_sector, simfin_sector, cik, simfin_id

-- universe_snapshots: Russell 1000 membership per date
PRIMARY KEY (snapshot_date, isin, index_name)
-- key cols: snapshot_date, isin, index_name, weight
```

### factors.db
```sql
PRIMARY KEY (data_date, factor_id, security_id)
-- factor_value: raw, unsigned
-- factor_value_z: cross-sectional winsorized z-score, unsigned
-- security_id: ISIN
```

### models.db
```sql
PRIMARY KEY (data_date, model_id, security_id)
-- model_value: direction-adjusted weighted sum of factor z-scores
-- model_value_z: cross-sectional winsorized z-score of model_value
-- is_composite: 0 = base model, 1 = Alpha composite
-- security_id: ISIN
```

## Data access layer (db.py)
All public functions return DataFrames and are decorated with `@st.cache_data`.

Key functions:
- `get_universe()` — companies from universe.db; adds `security_id = isin`, `sector`, `display_name`
- `get_factors_long()` — full time series, long format, with `factor_name`, `category`, `factor_value_z`
- `get_factors_wide()` — latest snapshot, one row per security, one column per factor
- `get_models_wide()` — latest snapshot, one row per security, one column per model ("Quality Model" etc.)
- `get_screener_df()` — universe + factors wide + models wide merged on `security_id` (ISIN)
- `get_factors_for_security(sid)` — all snapshots for one company (filter to `data_date == max` for point-in-time)
- `get_models_for_security(sid)` — all snapshots for one company

`MODEL_COLS` should always be derived dynamically — never hardcode a list of model names:
```python
model_meta      = db.get_model_metadata()
MODEL_COL_NAMES = [f"{m} Model" for m in model_meta["Model"]]
MODEL_COLS      = [c for c in screener.columns if c in MODEL_COL_NAMES]
```

## Dashboard pages

| Page | Key data sources |
|------|-----------------|
| 1_Universe.py | get_universe(), get_screener_df() |
| 2_Factors.py | get_screener_df(), get_factors_long() |
| 3_Screener.py | get_screener_df() |
| 4_Deep_Dive.py | get_screener_df(), get_factors_for_security(), get_models_for_security(), get_constituents_for_security() |
| 5_Themes.py | get_screener_df(), get_sector_model_medians() |

## Pipeline CLI

```bash
# Full historical build
python create_universe.py
python create_databases.py
python create_returns.py
python create_factors.py --backfill
python create_models.py

# Weekly incremental (new EDGAR filings)
python update_constituents.py
python create_factors.py --date 2026-04-01
python create_models.py --date 2026-04-01

# Single-date rebuild
python create_factors.py --date 2026-04-01 --clean
python create_models.py --date 2026-04-01
```

## Coding standards

- No comments unless the WHY is non-obvious
- Use pathlib for file paths, sqlite3 for DB operations, pandas for bulk reads
- Bulk inserts via `executemany()`, never row-by-row
- For historical price lookups: binary search with `np.searchsorted` on pre-loaded numpy arrays
- Type hints required; Python 3.13 style (`X | None` OK)
- `winsorized_zscore` lives in `utils.py` — import from there, don't redefine
