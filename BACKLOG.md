# Project Backlog

Items are grouped by theme, not priority. Each has a rough effort and value rating.
`[HIGH]` / `[MED]` / `[LOW]` = value add. `[easy]` / `[medium]` / `[hard]` = effort.

---

## Pipeline — Active

- [ ] **Re-run 6 recent snapshots** after fill-gaps completes (2025-11-15 → 2026-05-16)
  `create_factors/models/risk/barra --date` for each. *Activates Q4 derivation + YTD fix.*
  `[HIGH]` `[easy]`

- [ ] **Verify gross margins post-fix** — ABBV, COST, LIN, AMZN
  Run `/validate` on each, confirm no `n/a` on Gross Profit and no consecutive-quarter warnings.
  `[HIGH]` `[easy]`

- [ ] **Update risk.db** — one snapshot behind (2026-05-15 vs factors/models 2026-05-16)
  `create_risk.py --date 2026-05-16`
  `[MED]` `[easy]`

---

## Data Quality

- [ ] **YTD filers audit** — after running `_fix_ytd_quarters`, grep for `[YTD fix]` log lines
  to identify which companies had cumulative quarters corrected. Spot-check 2-3 against SEC filings.
  `[HIGH]` `[medium]`

- [ ] **Consecutive-quarter warnings audit** — after snapshot re-runs, grep logs for `[WARN]`
  non-consecutive quarter lines. Investigate any persistent gaps.
  `[HIGH]` `[medium]`

- [ ] **0-row filings investigation** — fill-gaps produced several `0 rows (filed ...)` entries
  (CG, BC, FHN, FNF etc). These are likely amended filings edgartools skips. Confirm they're
  covered by the non-zero filing for the same FY.
  `[MED]` `[easy]`

- [ ] **EDGAR quarterly backfill** — `update_constituents.py` only keeps 2 years of quarterly
  data. Some factors (momentum, growth) benefit from 3 years. Extend `min_sort_key` lookback
  and re-run quarterly pull.
  `[MED]` `[medium]`

---

## Factors & Models

- [ ] **Insider transaction factor** — rework `explore_insider.py` to fetch per-CIK
  (994 targeted lookups) instead of global Form 4 index. Build net-buy / net-sell signal,
  store in factors.db. Currently blocked by slow global fetch approach.
  `[HIGH]` `[medium]`

- [ ] **Short interest factor** — `explore_short_interest.py` already exists. Evaluate whether
  SVR (FINRA short volume ratio already in returns.db) is sufficient or if a dedicated SI factor
  adds incremental signal.
  `[MED]` `[easy]`

- [ ] **Fail loudly on < 4 LTM quarters** — `select_ltm_data` currently produces a partial LTM
  with a warning. Add a strict mode flag `require_4q=True` to return None instead. Useful for
  high-conviction factor runs.
  `[MED]` `[easy]`

- [ ] **Ticker override map** — handful of tickers that edgartools can't resolve by ticker
  (resolves by CIK instead). Maintain a small dict in `update_constituents.py` as a fallback.
  `[MED]` `[easy]`

- [ ] **Parallelise factor loop** — `create_factors.py` snapshot loop is sequential per date.
  Could run multiple snapshot dates in parallel with `concurrent.futures.ProcessPoolExecutor`.
  Mainly useful during backfills.
  `[LOW]` `[medium]`

---

## Risk Model

- [x] **Barra consolidated into risk.db** — Barra tables (`factor_returns`, `factor_covariance`,
  `idiosyncratic_vars`, `factor_exposures`) merged into `risk.db`; `barra.db` deleted.
  Snapshot dates now read dynamically from `factors.db` (`snapshot_dates` table) — no config list needed.

- [ ] **Barra factor coverage check** — after each snapshot, log how many stocks have
  non-zero style/fundamental exposures vs total universe. Sparse coverage inflates VRA.
  `[MED]` `[medium]`

---

## Portfolio Optimizer

- [ ] **Cardinality constraint** — MOSEK is installed; add optional max_stocks constraint
  to strategy_params.xlsx. Useful for concentrated strategies.
  `[MED]` `[medium]`

- [ ] **Transaction cost model** — current optimizer ignores turnover. Add a simple linear
  cost penalty (e.g. 10bps per unit of turnover) as an optional constraint column.
  `[MED]` `[hard]`

- [ ] **Benchmark-relative risk decomposition** — `load_risk_contributions()` gives absolute
  risk. Add active risk (tracking error) decomposition split by factor vs idiosyncratic.
  `[MED]` `[hard]`

---

## Dashboard

- [ ] **Insider signal page** — once insider factor is built, add a dashboard page showing
  recent insider purchases by sector/stock with forward return overlays.
  `[MED]` `[medium]`

- [ ] **Factor correlation heatmap** — page showing pairwise correlations between the 28
  factors at the latest snapshot. Useful for model weight decisions.
  `[LOW]` `[medium]`

- [ ] **Snapshot timeline view** — show factor/model values over time for a selected stock.
  Currently Deep Dive shows latest only.
  `[MED]` `[medium]`

---

## Skills / DevEx

- [ ] **`/gap-check` skill** — dry-run `update_constituents.py --fill-gaps` to show which
  tickers are missing FY data before committing to a full run.
  `[MED]` `[easy]`

- [ ] **`/rebalance` skill** — full workflow: update → snapshot → optimize → diff weights.
  `[MED]` `[medium]`

- [ ] **Automated daily update cron** — `daily_update.py` exists but unclear if scheduled.
  Wire to crontab or launchd for lights-out operation.
  `[MED]` `[medium]`

---

## Research / Exploratory

- [ ] **Insider transaction signal backtest** — once `explore_insider.py` is fixed, run
  the full 180-day signal → forward return analysis and decide if it merits a factor slot.
  `[HIGH]` `[medium]`

- [ ] **Akshare / alternative data fallback** — some international tickers have thin EDGAR
  coverage. Akshare covers HK/China-listed companies. Evaluate coverage gap first.
  `[LOW]` `[hard]`

- [ ] **Earnings surprise factor** — compare reported EPS vs consensus (would need a
  consensus data source; none currently in pipeline). Placeholder for future data vendor.
  `[HIGH]` `[hard]`

- [ ] **Macro regime overlay** — overlay a simple macro regime indicator (e.g. yield curve
  slope, credit spreads) on portfolio construction. Research question: does conditioning
  on regime improve alpha?
  `[LOW]` `[hard]`
