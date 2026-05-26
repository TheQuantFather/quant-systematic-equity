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

- [x] **Update risk.db** — caught up; 2026-05-22 risk + Barra ran 2026-05-25.

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

- [ ] **EDGAR historical quarterly backfill** — `MIN_QUARTERLY_FISCAL_YEAR` extended to 2021
  (was a rolling 2-year window). Run `--fill-gaps --quarterly` for all ~968 companies to pull
  Q1 FY2021 → present. Prerequisite for full historical factor restatement below. 2–4 hour job.
  After completion: re-run factors/models/risk/barra for all snapshot dates.
  `[HIGH]` `[medium]`

---

## Factors & Models

- [ ] **Full historical factor restatement** — log-transform (EV/EBITDA, EV-to-EBIT, Leverage),
  growth positive-base guards (Operating Income, EBITDA), momentum skip-month + vol-normalisation,
  and LT Reversal factor were all applied only to the 2026-05-22 snapshot. Rebuild all prior
  snapshot dates so cross-sectional z-scores are consistent across history.
  Run: `python create_factors.py --date <all dates>` → `create_models.py` → `create_risk.py --backfill` → `create_barra.py --backfill`
  `[HIGH]` `[easy]`

- [ ] **Cash Conversion Quality denominator guard** — AUR and ROIV sit at −199× and −90× due to
  near-zero revenue denominator. Fix requires a minimum revenue floor; design decision pending.
  `[MED]` `[easy]`

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

- [x] **Snapshot timeline view** — implemented in Deep Dive: model score history + factor z-score history across snapshot dates.

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

## Portfolio Strategies — Exploratory

- [ ] **ETF strategy** — systematic long-only strategy structured as a rules-based ETF.
  Define rebalance frequency, constituent selection rules (factor score thresholds), weighting
  scheme, and capacity constraints. Evaluate tradability vs current optimize_portfolio.py output.
  `[HIGH]` `[hard]`

- [ ] **Hedge fund strategy** — long-short market-neutral portfolio using Alpha model.
  Define gross/net exposure limits, short-side factor eligibility (low-score universe),
  leverage constraints, and fee/financing cost model. Analyse factor risk decomposition
  to ensure style neutrality on the short book.
  `[HIGH]` `[hard]`

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
