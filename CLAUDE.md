# CLAUDE.md — Project guidance for Claude Code

## Python environment
**Always use**: `/Users/shivam/opt/anaconda3/envs/quant/bin/python3.13` (Python 3.13.5, `conda activate quant`).

## Run order

### Full build (first time)
```bash
python create_databases.py          # constituents.db from SimFin CSVs
python create_returns.py --update   # returns.db (daily prices via Yahoo Finance)
python create_svr.py --backfill     # returns.db svr_daily table (~90 trading days of FINRA data)
python create_factors.py --quarterly-backfill  # factors.db (all quarterly snapshots)
python create_models.py             # models.db (9 models incl. Short Interest)
python create_risk.py --backfill    # risk.db (Ledoit-Wolf covariance, all snapshots)
python create_barra.py --backfill   # risk.db Barra tables (all quarterly snapshot dates)
python create_strategy_params.py    # data/strategy_params.xlsx (reset only)
python optimize_portfolio.py        # portfolio_output/ (all active strategies)
streamlit run app.py                # dashboard
```

### Rebuild universe snapshots only (keeps companies table intact)
```bash
python create_universe.py --rebuild-snapshots
```

### EDGAR quarterly backfill (fill SimFin/EDGAR data gaps)
Run when companies are missing recent quarterly data (visible in the DQ page → Constituents → "LTM window gaps").
Always use `caffeinate -s` — this takes 2–4 hours for the full universe.

```bash
caffeinate -s -i nohup /Users/shivam/opt/anaconda3/envs/quant/bin/python3.13 -u \
  update_constituents.py --fill-gaps --quarterly --timeout 90 \
  > /tmp/backfill_quarterly.out 2>&1 &
# Monitor: tail -f logs/update_constituents.log
# When done: check DQ page LTM gap count, then re-run factors/models/risk/barra
```

- `--fill-gaps` triggers company-by-company mode (index mode alone does NOT backfill)
- `--quarterly` fetches 10-Q filings; omit for annual-only (10-K) backfill
- `--timeout 90` skips any company that hangs for >90s (SIGALRM — may not interrupt httpx)
- edgar library HTTP timeouts are patched at startup: 45s read, 3 retries max (~2.25 min worst case)

### Long-running backfills — prevent Mac sleep
Any backfill that runs for more than ~30 minutes should be wrapped with `caffeinate -s` so
macOS doesn't suspend the process when the lid closes or the system idles.
`-s` holds a sleep assertion while on AC power; add `-i` to also block idle sleep on battery.

```bash
# Template — use for any backfill command
caffeinate -s -i nohup /Users/shivam/opt/anaconda3/envs/quant/bin/python3.13 -u \
  <script.py> [flags] > /tmp/<script>_out.log 2>&1 &
# Then monitor with: tail -f logs/<script>.log
```
Note: `-s` prevents system sleep on AC **only**; `-i` prevents idle sleep on battery too.
Lid-close sleep can still override both — for fully unattended overnight runs, use a remote server (EC2 t3.small).

### Incremental update (new EDGAR filings)
```bash
python update_constituents.py [--limit N] [--ticker X] [--sector-type financial] [--force]
python create_returns.py --update   # pull latest Yahoo Finance prices
python create_svr.py                # pull latest FINRA SVR data (incremental)
python create_factors.py --date 2026-04-01
python create_models.py --date 2026-04-01
python create_risk.py --date 2026-04-01
python create_barra.py --date 2026-04-01   # or just: python create_barra.py (latest Friday)
python optimize_portfolio.py        # re-run all strategies
```

### Data restatement — when a factor calculation changes
Any fix to factor logic (create_factors.py) or constituent data loading (load_constituent_data in
create_factors.py) changes raw factor values, which shifts z-scores cross-sectionally.
**Always re-run ALL historical snapshots together**, not just the affected date:

```bash
# 1. Re-run factors for all dates (snapshot_dates table drives the list automatically)
python3.13 -c "
import sqlite3
dates = [r[0] for r in sqlite3.connect('data/factors.db').execute(
    'SELECT data_date FROM snapshot_dates ORDER BY data_date').fetchall()]
print(' '.join(f'--date {d}' for d in dates))
" | xargs python create_factors.py

# 2. Re-run models for every date in factors.db (auto-discovers dates)
python create_models.py

# 3. Risk and Barra: --backfill reads snapshot_dates from factors.db automatically
python create_risk.py --backfill
python create_barra.py --backfill
```

**Why all dates?** Factor z-scores (`factor_value_z`) are cross-sectional — a value that changes for
one company shifts the entire distribution. Updating only one snapshot date leaves historical z-scores
inconsistent with the fixed calculation, which distorts any time-series or backtesting analysis.

## Architecture

| DB | PK | Content |
|----|-----|---------|
| `universe.db` | `isin` | Company metadata: ticker, cik, gics_sector, simfin_industry |
| `constituents.db` | `(constituent_id, security_id, publish_date)` | Financial statement data; PIT anchor = publish_date |
| `returns.db` | `(isin, date)` | Daily adjusted prices (`returns` table) + FINRA short volume ratio (`svr_daily` table) |
| `factors.db` | `(data_date, factor_id, security_id)` | 28+ factors × N snapshots (6 April-1 + quarterly 15th); values unsigned; N from `snapshot_dates` table |
| `models.db` | `(data_date, model_id, security_id)` | 9 model scores × N snapshots; direction applied |
| `risk.db` | `data_date` | Ledoit-Wolf covariance blobs (`covariance_matrix`) + Barra factor risk model: 4 tables (`factor_returns`, `factor_covariance`, `idiosyncratic_vars`, `factor_exposures`) |

**Portfolio layer**: `data/strategy_params.xlsx` (Excel) → `optimize_portfolio.py` (CVXPY) → `data/portfolio_output/{sid}_latest.csv` + `_summary.json` → `pages/8_Portfolio.py`

**Risk model**: Barra (default, in `risk.db`) or Ledoit-Wolf (also in `risk.db`, fallback). Toggle per strategy via `use_barra_risk` column in Strategies sheet (TRUE/default | FALSE for Ledoit-Wolf).

## Critical constraints

### Factor values are unsigned
`factor_value`/`factor_value_z` in factors.db = raw value, direction never applied. Direction (`±1` in `factors_reference.csv`) applied only in `compute_base_models()`: `z × weight × direction`.

### Balance sheet period convention
`fiscal_period = 'Q4'` for balance sheet rows; `'FY'` for income statement and cash flow. Always filter `WHERE fiscal_period IN ('FY', 'Q4')` for annual data.

### Point-in-time
- `publish_date ≤ snapshot_date` for constituents; fallback = `fiscal_year_end + 90d`
- `update_constituents.py` uses `acceptance_datetime` as publish_date; SimFin uses SimFin's publish_date
- Both coexist in constituents.db; `create_factors.py` picks latest qualifying per `(security_id, fiscal_year)`

### Q4 derivation from annual 10-K
EDGAR 10-Qs cover Q1–Q3 only; Q4 income/cashflow is filed annually as `fiscal_period='FY'`.
`load_constituent_data()` derives standalone Q4 for Flow items using the exact identity:
`Q4 = FY − Q1 − Q2 − Q3` — only when Q4 is absent and all three prior quarters are present.
Balance sheet items are not derived (10-K already stores them as `fiscal_period='Q4'`).

**Temporal guard**: Q1/Q2/Q3 must be published **before** the annual FY record.  If any were
published after, they belong to the next fiscal year and derivation is skipped.  This prevents
contamination for companies with early fiscal year-ends (January/February) where EDGAR labels
the next FY's quarterly filings under the same `fiscal_year` integer as the annual — e.g. NVDA
(FY ending Jan 2025): annual pub Feb-2025, but FY2026 Q1 pub May-2025 → same `fiscal_year=2025`.
Affected companies: typically retailers/tech with Jan/Feb year-ends (NVDA, WMT, HD, TGT, etc.).

**Flow-item filter in `select_ltm_data`**: quarters with only balance-sheet items (no Flow data)
are excluded from the LTM window.  This prevents orphaned EDGAR annual BS-only buckets — created
when the temporal guard blocks Q4 derivation — from overwriting the prior year's complete BS values.

### YTD decomposition (order-sensitive)
Some EDGAR filers tag only cumulative 6M/9M values in XBRL. `load_constituent_data()` calls
`_fix_ytd_quarters()` **before** Q4 derivation — this order is mandatory. Reversing it produces
wrong Q4 values because the Q4 math uses Q1+Q2+Q3 as inputs.
Heuristic: Flow item Q2/Q1 > 1.65 → treated as H1 cumulative; Q2_standalone = Q2 − Q1.
New data is handled cleanly upstream: `extract_filing_data(is_quarterly=True)` selects
standalone quarter columns over YTD columns in 10-Q XBRL where both are present.

### Incremental pipeline
`CREATE TABLE IF NOT EXISTS` + `INSERT OR REPLACE`. `--date` flag adds without dropping.

### `winsorized_zscore` lives in utils.py
Groups < 10 → NaN. Zero std → NaN. Never redefine elsewhere.

## Factor directionality

| Factor | direction |
|--------|-----------|
| All profitability / return / growth / momentum / liquidity | 1 |
| Leverage, Debt-to-Assets, Capex Intensity, EV-to-EBIT, Working Capital Efficiency | −1 |
| Log Market Cap | 1 (larger = better in Size model) |
| Realized Volatility | −1 (lower = better in Low Vol model) |
| 52-Week High Ratio | 1 |
| Amihud Illiquidity | −1 |

## Barra factor risk model

**Σ = X F X' + Δ** — K=29 factors (11 sector, 5 style, 1 beta, 12 fundamental). See README for full schema and estimation pipeline.

**Optimizer**: stacked-L drop-in — `L_barra = vstack([L_F.T @ X.T, diag(√δ)]).T` (shape N×K+N). No CVXPY changes needed.

**Fallback**: set `use_barra_risk = FALSE` in Strategies sheet to force Ledoit-Wolf. Falls back silently on any load error.

### Gotchas
- F is **annualised** (daily ε² × 252). `‖L_barra.T w‖²` = annualised portfolio variance — consistent with risk.db.
- VRA B² far from 1.0 (healthy: 0.8–1.3) indicates sparse factor coverage at that snapshot date.
- Fundamental z-scores are stale up to ~3 months between quarterly snapshots — acceptable; style/beta/sector update daily.
- Adding new factors: add a row to `factors_reference.csv` with `barra_factor_type` (`style`/`fundamental`) and `barra_factor_order` set, then re-run `--backfill`. `FACTOR_NAMES` order = column order — do not reorder without rebuilding risk.db Barra tables.
- `INSERT OR REPLACE` — safe to re-run for any date already in risk.db.
- No sector reference dropped — SVD handles rank deficiency naturally.

## Portfolio optimizer

**Objectives** (set in Strategies sheet, `objective` column):
- `maximize_alpha` — benchmark-aware; maximise active alpha; requires `benchmark_file`
- `maximize_sharpe` — absolute return; Charnes-Cooper transform; no benchmark
- `minimize_variance` — pure risk minimisation; ignores alpha

**Constraints** (Constraints sheet): `max_active_risk`, `max_stock_active_weight`, `max_sector_active_weight`, `max_industry_active_weight` (active-weight space, alpha only); `max_position`, `max_sector_weight`, `min_sector_weight`, `equal_sector_weight`, `sector_weight_tolerance`, `excluded_sectors` (pipe-separated), `max_industry_weight`, `max_portfolio_vol` (absolute space).

**Alpha blending**: `Alpha_Weights` sheet — multiple rows per strategy_id with model_id + weight; blended proportionally. `minimize_variance` ignores alpha entirely (placeholder row still required).

**Solver**: CLARABEL (default, installed). MOSEK needed for cardinality/integer constraints.

**Risk contributions**: `load_risk_contributions()` in `pages/8_Portfolio.py` computes per-stock `pct_of_risk = w_i(Σw)_i / (w'Σw)` and `vol_contribution = w_i(Σw)_i / σ_p` directly from risk.db blob.

## Dashboard MODEL_COLS

Always derive dynamically:
```python
model_meta      = db.get_model_metadata()
MODEL_COL_NAMES = [f"{m} Model" for m in model_meta["Model"]]
MODEL_COLS      = [c for c in screener.columns if c in MODEL_COL_NAMES]
```

## Universe DB reference tables

All security mappings and index configuration live in `universe.db` — never hardcode them in scripts:

| Table | Purpose | Seeded from |
|-------|---------|-------------|
| `isin_patch` | ticker → ISIN overrides for companies SimFin can't match | `_ISIN_PATCH_DEFAULTS` in `create_universe.py` |
| `ticker_alias` | iShares ticker → SimFin ticker aliases (e.g. BRKB → BRK-A) | `_SIMFIN_MATCH_DEFAULTS` in `create_universe.py` |
| `index_registry` | ETF metadata per tracked index (CIK, series_id, etc.) | `_INDEX_REGISTRY_DEFAULTS` in `create_universe.py` |
| `nport_accessions` | N-PORT-P EDGAR accession numbers per index × snapshot date | `_INDEX_REGISTRY_DEFAULTS` in `create_universe.py` |

All four tables use `CREATE TABLE IF NOT EXISTS` — the schema is created on first run but **no data is inserted by code**. The DB is the single source of truth; restore from Time Machine backup if needed (this is a single-laptop personal project, not a cold-clone setup).

To add a new mapping, insert directly into the DB:
```sql
INSERT INTO isin_patch (ticker, isin, note) VALUES ('XYZ', 'US1234567890', 'reason');
INSERT INTO ticker_alias (ticker, alias_ticker) VALUES ('NEWT', 'OLD');
INSERT INTO index_registry VALUES ('nasdaq_100', 'QQQ', 'Invesco QQQ', 'S000017754', '1100663');
INSERT INTO nport_accessions VALUES ('nasdaq_100', '2026-04-01', '0001752724-26-034803', '2025-12-31');
```

**Rule**: never add hardcoded ticker/ISIN/accession mappings inside pipeline scripts. Always ask before making significant pipeline changes or adding new mappings.

## Code quality standards

### Pipeline vs exploratory
**Pipeline scripts** (`create_*.py`, `update_*.py`, `optimize_portfolio.py`, `db.py`, all `pages/`) must meet the standards below. **Exploratory scripts** (anything in `exploratory/`) have lower requirements — readable and correct is enough. The `exploratory/` folder is git-ignored.

### DB connections
Always use `with get_db(X) as conn:` from `utils.py`. Never call `sqlite3.connect()` directly in pipeline code.

Two intentional exceptions (long-lived connections across a multi-phase run, closed with `try/finally`):
- `create_returns.py::connect()` — holds connection across backfill → migrate → update → checks
- `create_barra.py::_init_db()` — holds connection for the full Barra build session

### Paths
All DB paths, reference file paths, and directory constants come from `config.py`. Never define `Path("data/...")` or `DATA_DIR = Path(...)` in pipeline scripts.

### Shared utilities
`get_db`, `classify_sector`, `winsorized_zscore`, and `get_logger` live in `utils.py`. Never redefine them elsewhere.

### Transactions
Logically related writes (e.g. raw insert + z-score update) must land in a single atomic `conn.commit()`. No intermediate commits within a pipeline stage — partial writes corrupt incremental builds.

### SQL safety
- Table names: validate against a whitelist before interpolating into SQL (see `_assert_valid_table` in `pages/7_Database.py`)
- Values: always use `?` parameterized queries — never f-string or `.format()` user-supplied values into SQL

### Type hints
All function signatures in pipeline scripts must have type hints on parameters and return type.

### Logging
Every pipeline script uses `get_logger(name)` from `utils.py` — never `print()` for status/progress output.

```python
from utils import get_logger
log = get_logger("script_name")   # one call at module level
```

- Log files: `logs/<name>.log` — rotating, 5 MB per file, 3 backups.
- Also mirrors to stdout (same format, timestamps included).
- Level controlled by `LOG_LEVEL` env var (default `INFO`). Set `LOG_LEVEL=DEBUG` for verbose output.
- Severity guide: `log.debug()` for per-row detail, `log.info()` for stage milestones and summaries, `log.warning()` for recoverable issues, `log.error()` for failures before raising/exiting.
- One intentional `print()` allowed: `optimize_portfolio.py --list` CLI table (raw tabular stdout, no timestamps needed).

### Fail loudly
Pipeline scripts must call `log.error(...)` and either `raise` or `sys.exit(1)` on unrecoverable failure. Never silently swallow exceptions with a bare `except: pass` in a pipeline context.

## Data source strategy: EDGAR-first

**Direction**: Prefer EDGAR quarterly 10-Q data over SimFin wherever available.  SimFin is the
fallback for companies/periods not yet covered by EDGAR.

**Current state** (as of 2026-05):
- SimFin coverage: ~694 security_ids, deep historical data (2017+)
- EDGAR coverage: ~923 security_ids (ISIN-keyed), FY2024–2026 range
- ~682 companies have **both** — EDGAR is more current; `_map_sid` in `create_factors.py` unifies both under ISIN and `_dedup` picks the later publish_date
- ~240 companies EDGAR-only; ~11 SimFin-only

**Annual FY records are still required** — US 10-Qs only cover Q1–Q3; Q4 standalone flow data
is derived as `FY − Q1 − Q2 − Q3` from the annual 10-K (stored as `fiscal_period='FY'`).

**Migration path** (not yet done):
1. Run `update_constituents.py --backfill` for all companies to pull historical EDGAR 10-Q history
2. Verify EDGAR quarterly data quality and fiscal year labeling for non-December year-end companies
3. DELETE SimFin rows (`security_id = numeric simfin_id`) for companies/periods covered by EDGAR
4. Re-run full backfill (see "Data restatement" above) after any DB cleanup

**Do not delete SimFin data until** EDGAR backfill is complete and verified — SimFin is the only
quarterly source for ~532 companies and the only historical source for most overlap companies.

## Common gotchas

- `get_factors_for_security()` returns all snapshot dates — filter to `data_date == max` for current view.
- Screener/Themes display `model_value_z`, not raw `model_value`.
- Snapshot dates: 6 annual April-1 dates + quarterly mid-period (Feb/May/Aug/Nov 15th). All stored in `factors.db` `snapshot_dates` table.
- `edgar_concept_map.xlsx` row order = priority. `AllEquityBalance` must precede `CommonEquity` for banks (CommonEquity ≈ $4B for JPM vs correct $344B).
- `extract_filing_data(is_quarterly=False)` — pass `is_quarterly=True` for 10-Q calls. It selects standalone quarter columns (Q1/Q2/Q3) over cumulative YTD columns in XBRL. Also enforces accounting identity: Total Assets = Total L&E; corrects subsidiary-tagged consolidated rows in bank filings.
- `factors_reference.csv` `sector_type` column: `all` / `general` (excl. banks+REITs) / `reit`. Match when adding new factors.
- REIT companies get 3 FFO-based factors; financials skip Revenue/Gross Profit factors naturally.
- `excluded_sectors` constraint uses pipe separator: `"Energy|Materials"` (not comma — commas conflict with CSV parsing in some Excel locales).
- Re-running `create_strategy_params.py` overwrites any manual Excel edits. Only use to reset.
- `_fix_ytd_quarters` only corrects **positive** Q1 Flow items — companies reporting losses in Q1 are skipped (can't determine sign of cumulative from ratio alone).
- `_quarter_from_period()` allows a 1-month spillover for 52/53-week fiscal year companies whose quarter ends occasionally fall one day into the next calendar month (e.g. GD Q1 FY2026 ended Apr 5 instead of Mar 31). Both the quarter end month and the following month map to the same quarter.
- `_latest_expected_sk()` is the FYE-aware skip gate in `--fill-gaps --quarterly` mode. It returns `fiscal_year*10+q_num` for the most recently expected filing (2-month lag). Companies already at or above this key are skipped without a network call.
- `load_company_map()` includes companies with no `simfin_id` (EDGAR-only additions from N-PORT) using `-cik` as a synthetic key. Safe because `security_id = isin or str(simfin_id)` and all such companies have ISINs.

## Custom slash commands

Stored in `.claude/commands/` — available in Claude Code as `/command-name`:

| Command | Usage | Description |
|---------|-------|-------------|
| `/snapshot` | `/snapshot 2026-04-01` | Runs create_factors → models → risk → barra for a date |
| `/validate` | `/validate ABBV` | Shows LTM P&L, cash flow, balance sheet for a ticker |
| `/db-check` | `/db-check` | Health summary across all 6 databases, sync status |

See `BACKLOG.md` for pending work and research ideas.
