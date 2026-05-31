# Dynamic Alpha Weighting — Implementation Plan

Replace the fixed, hand-set weights in the Alpha model with **data-driven, walk-forward
weights** that maximise out-of-sample cross-sectional predictive power (rank-IC) of forward
returns — first at the model-blend level (`Alpha = Σ w·base_model`), later at the factor level
(`base_model = Σ w·factor`).

Status: **planning**. Nothing built yet beyond the baseline measurement below.

---

## 1. Current state

- [`create_models.py`](create_models.py) builds base models (Quality, Value, Growth, Momentum,
  Size) as fixed-weighted sums of direction-adjusted factor z-scores, then `Alpha (ALP001)` as a
  fixed blend: **0.30·Q + 0.20·V + 0.20·G + 0.20·M + 0.10·Size**.
- Standalone models (not in Alpha): Low Vol, Liquidity, Short Interest, LT Reversal.
- Weights live in [`data/models_reference.csv`](data/models_reference.csv) and are hand-chosen.

---

## 2. The binding constraint: ~28 cross-sections

- 30 factor snapshots → **28 usable snapshot→forward-return cross-sections** (~985 names each).
- This is *few in time*. Consequences that shape every decision below:
  - **Regularisation + shrinkage are mandatory** — a 5-parameter model is already at risk; 28
    factor weights would massively overfit without it.
  - **No flexible/deep models in v1** (gradient boosting, nets) — they need far more periods.
  - **Honest walk-forward (expanding window)** validation, never in-sample fit.
  - Expect **modest, robust** improvement — or a finding that the hand weights are already fine.
- **Lever to relax the constraint:** compute factor snapshots **monthly** rather than at the
  current 30 dates. Price factors (momentum, vol, 52w-high, Amihud) update daily; fundamental
  factors are point-in-time-valid but stale between quarters — acceptable. This roughly **doubles
  the cross-section count** and is the single highest-value enabler for everything below.

---

## 3. Baseline evidence (measured 2026-05-31, post full restatement)

Rank-IC of each model's `model_value_z` vs forward return to the next snapshot, across 28 cross-sections:

| Model | Alpha weight | mean IC | IC std | IR (mean/std) | hit rate |
|-------|-------------:|--------:|-------:|--------------:|---------:|
| **Alpha (ALP001)** | — | 0.052 | 0.127 | **0.41** | 68% |
| Quality | 0.30 | 0.004 | 0.082 | 0.05 | 50% |
| Value | 0.20 | 0.043 | 0.182 | 0.24 | 57% |
| Growth | 0.20 | −0.003 | 0.102 | −0.03 | 46% |
| Momentum | 0.20 | 0.032 | 0.138 | 0.23 | 54% |
| Size | 0.10 | 0.026 | 0.100 | 0.26 | 68% |
| Low Vol | — | 0.023 | 0.230 | 0.10 | 54% |
| Liquidity | — | 0.026 | 0.095 | 0.27 | 64% |
| Short Interest | — | 0.029 | 0.065 | 0.45 | (6 snaps only) |
| LT Reversal | — | −0.001 | 0.141 | −0.01 | 44% |

**Findings**
- The composite **Alpha IR ≈ 0.41 is the bar to beat** out-of-sample.
- **Hand weights are misallocated:** Quality carries the *largest* weight (0.30) but has ~zero IC;
  Growth (0.20) has negative IR; while Size (0.10), Value (0.20), Momentum (0.20), and the
  *unweighted* Liquidity all sit at IR 0.24–0.27. Short Interest looks strong (IR 0.45) but has
  only 6 snapshots (FINRA SVR data is recent).
- **Caveat:** with 28 observations the standard error on each IC is ≈ IC_std / 5.3 (e.g. Quality
  0.004 ± 0.015 — indistinguishable from zero). The *ranking/direction* is informative; the
  *magnitudes* are noisy. This is precisely why the method must **shrink**, not chase point estimates.

---

## 4. Design decisions (recommendations)

| Decision | Recommendation (v1) | Later |
|---|---|---|
| **Level to learn** | Model-blend (5 weights) — safest given data | Factor level (~28 weights) |
| **Prediction target** | Forward **total-return rank** | Barra-**residual** (pure-alpha) return |
| **Static vs dynamic** | **Dynamic** — re-estimated each snapshot on an expanding window | — |
| **Method** | **IC/IR-weighting with shrinkage** toward equal/current | Elastic-net Fama-MacBeth → LightGBM |
| **Integration** | New model **`ALP002` "Alpha (learned)"** alongside `ALP001` | Promote if it wins |

Rationale: start with the fewest parameters and the most interpretable, battle-tested method
(IC/IR weighting is standard practice), prove it out-of-sample, then add complexity only if it earns
its keep. Keep `ALP001` untouched so A/B comparison is clean.

---

## 5. Phased plan

**Phase 0 — Validation harness + baseline** *(baseline done; harness to formalise)*
Build the reusable `(snapshot factor/model z → forward return)` panel and lock the OOS metric suite:
mean rank-IC, IR, top-minus-bottom quintile spread, and a full backtest via
[`pages/6_Backtester.py`](pages/6_Backtester.py). Establishes ALP001 as the benchmark.

**Phase 1 — Dynamic model-blend weighting** `[HIGH] [medium]`
At each snapshot *t*, weight each base model ∝ a **shrunk trailing IR** computed only from snapshots
< *t* (expanding window, warm-up ≥ ~8–12 snapshots). Shrink toward equal-weight (James-Stein-style)
so noisy ICs don't dominate. Renormalise, drop negative-IR models (or floor at 0). Write `ALP002`.
Validate OOS vs ALP001.

**Phase 2 — Factor-level regularised combination** `[MED] [hard]`
Cross-sectional **elastic-net / ridge** of forward returns on factor z-scores, fit per period and
averaged (Fama-MacBeth), with weights shrunk toward the current factor priors. Compare to Phase 1.

**Phase 3 — Nonlinear (research only)** `[LOW] [hard]`
LightGBM on factor interactions with **purged + embargoed** CV and SHAP attribution. Only if
Phases 1–2 show real signal and (ideally) after the monthly-snapshot expansion.

**Foundational (parallel) — monthly factor snapshots** `[HIGH] [medium]`
Compute factors monthly to ~2× the cross-sections. Biggest single lever for statistical power.

---

## 6. Validation methodology (non-negotiable)

- **Expanding-window walk-forward** — weights at *t* use only data strictly before *t*. No in-sample fits.
- **Metrics:** OOS mean rank-IC, IR (mean/σ of IC), top-bottom quintile forward-return spread, and a
  full optimiser backtest **net of turnover/transaction costs** vs ALP001.
- **Guardrails:** purge/embargo around each snapshot; monitor weight **turnover** (smooth if jumpy);
  split by regime (must not only work in one value/growth cycle).
- **Acceptance bar:** `ALP002` must beat `ALP001` out-of-sample on **IR *and* backtest IR net of
  turnover**. If it doesn't, that is a legitimate finding — keep the fixed weights.

---

## 7. Integration (non-destructive)

- A `train_alpha_weights.py` (or a function inside `create_models.py`) writes **per-snapshot learned
  weights** to a new `model_weights (data_date, model_id, component_id, weight)` table.
- `create_models.py` computes `ALP002` from the stored per-snapshot weights (so the dynamic nature is
  preserved in history), alongside the existing fixed `ALP001`.
- Screener / Deep Dive / Backtester / optimiser can select `ALP001` vs `ALP002`. The optimiser's
  `Alpha_Weights` sheet already supports per-strategy model blending — a strategy can point at `ALP002`.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| Overfitting (28 obs) | Shrinkage, regularisation, strict OOS, fewest parameters first |
| Regime-chasing | Shrink toward priors, warm-up window, regime-split validation |
| Turnover from changing weights | Monitor; smooth/EWMA the weights; cap weight step |
| Look-ahead bias | Strict expanding window; purge/embargo |
| Multiple testing | Small pre-registered factor set + metrics; report honestly |

---

## 9. Locked v1 decisions (2026-05-31)

1. **Scope** — **model-blend (5 base-model weights)**. Directly targets the misallocation in §3.
2. **Target** — **forward total-return rank** (cross-sectional). Matches the rank-IC baseline.
3. **Method** — **IC/IR-weighting with shrinkage** toward equal-weight; floor negative-IR at 0.
4. **Data** — **proceed on the 28 snapshots now**; monthly-snapshot expansion deferred.

### Phase 1 concrete spec
For each snapshot *t* (after a warm-up of *W* snapshots):
- For each base model *m* ∈ {Q, V, G, M, Size}: compute trailing rank-IC series on snapshots `< t`,
  then `IR_m = mean(IC)/std(IC)`.
- **Shrink**: `IR_m_shrunk = IR_m · k/(k+λ)`-style toward the equal-weight prior (James-Stein flavour);
  **floor** negative shrunk IR at 0.
- `weight_m(t) = IR_m_shrunk / Σ IR_shrunk` (renormalised; equal-weight fallback if all ≤ 0).
- `ALP002_score(i,t) = Σ_m weight_m(t) · base_model_z(i,m,t)`, then cross-sectionally z-scored.
- **Validate** OOS: mean rank-IC, IR, quintile spread, and backtest vs `ALP001` — net of turnover.

Prototype lives in `exploratory/` first; promote to `create_models.py` + a `model_weights` table only
if it beats `ALP001` out-of-sample on IR *and* net-of-turnover backtest IR.

---

## 10. Phase 1 results (2026-05-31) — `exploratory/alpha_weights.py`

Walk-forward OOS rank-IC, blending the 5 base models (warmup 10, 18 OOS snapshots):

| Method | mean IC | IR | vs fixed |
|---|---|---|---|
| fixed (hand weights) | 0.0353 | 0.421 | — |
| **equal-weight (1/5)** | **0.0406** | **0.441** | **+0.020** |
| ic_ir (IR-weighted, shrunk) | 0.0379 | 0.389–0.414 | −0.03 … −0.01 |
| max_ir (inverse-cov) | 0.0268 | 0.299 | −0.122 |

**Verdict: optimisation loses; equal-weight wins — the classic "1/N" result.** With 28 cross-sections,
estimation error dominates: IR-weighting and especially the mean-variance (max-IR) combination
**overfit and underperform**. Equal-weight beats the hand weights **in every OOS window tested**
(warmup 6/10/14/18 → ΔIR +0.009/+0.020/+0.028/+0.037, always positive). The hand weights' flaw is
over-weighting **Quality**, which has ~zero IC over 2021–2026.

Supplementary (equal-weight over different model *sets*, same OOS window): adding **Liquidity** is
roughly neutral (IR 0.430 vs 0.451), adding **Low Vol** hurts badly (0.210); **dropping Quality**
(Value/Growth/Momentum/Size/Liquidity, equal) scores best (0.493) — but that's ~7 configs tested on 19
snapshots, so treat as suggestive, not conclusive (and Quality's weakness may be 2021–2026
regime-specific, not structural).

### Revised recommendation

1. **Adopt equal-weight as the Alpha default** — a small, *robust*, zero-estimation improvement over the
   hand weights. Promote as `ALP002` and A/B in the backtester/optimiser. **Do not deploy learned-weight
   optimisation** (IC/IR or max-IR) on the current data — it is premature and overfits.
2. **The real levers to beat equal-weight robustly are upstream of weighting:**
   - **More cross-sections** — monthly factor snapshots (~2×) to cut estimation error. *Highest value.*
   - **Cleaner target** — Barra-residual (pure-alpha) returns instead of raw forward return.
   - **Quality-model review** — a textbook-robust factor showing ~zero IC suggests its 16-factor
     construction is diluted/noisy; worth auditing before concluding "Quality doesn't work."
3. Re-run the optimisation experiments **after** the monthly-snapshot expansion — that's the regime where
   learned weights have a real chance.

### Monthly-resolution re-test (2026-05-31, after building the month-end grid)
Built the month-end monthly grid (snapshot_schedule single source; 61 month-ends, all 4 DBs). Re-ran the
4-method comparison on the consistent monthly grid (60 snapshots, 42 OOS — ~2.2× the original 28):

| Method | mean IC | IR | vs fixed |
|---|---|---|---|
| fixed | 0.0290 | 0.239 | — |
| **equal-weight** | 0.0324 | **0.243** | +0.004 |
| ic_ir | 0.0276 | 0.202 | −0.036 |
| max_ir | 0.0076 | 0.067 | −0.172 |

**Verdict unchanged and stronger: equal-weight still ≥ fixed; optimisation still overfits (max_ir collapses
to 0.067 with more data).** The 1/N result is robust to the data expansion — it was not a small-sample
artifact. (Absolute IRs are lower than the quarterly test because monthly forward returns are noisier, and
intra-quarter month-ends carry stale fundamentals.) Conclusion holds: **adopt equal-weight; the wins are
upstream (residual-return target, Quality-model redesign, new signals), not in optimising the 5 weights.**

---

## 11. Quality-model audit (2026-05-31) — `exploratory/factor_audit.py`

Per-sub-factor rank-IC of the 16 Quality factors (direction-adjusted, vs forward return):

- **Signal (profitability/cash-flow):** ROIC 0.31, OCF Ratio 0.27, Asset Turnover 0.26, FCF Margin 0.25,
  Interest Coverage 0.24, Operating Margin 0.23, Cash Conversion 0.19, ROE 0.17, Net Margin 0.15, ROA 0.12.
- **Near-zero:** Accruals 0.05 (weight 0.10!), Gross Profit-to-Assets 0.03.
- **Negative (dragging):** Debt-to-Assets −0.26, Working Capital Efficiency −0.15, Gross Margin −0.13
  (weight 0.10!), Leverage −0.13.

**Two compounding problems:** (1) **22% of weight on negative-IR factors**; (2) the **highest weights sit on
the weakest factors** (Gross Margin 0.10 / Accruals 0.10 / ROA 0.10) while the best factor (ROIC) gets 0.05.

**Validated fix (full-sample IR):** full current = 0.055 → full equal-weight = 0.100 → **profitability-core
(drop the 3 balance-sheet factors, equal-weight) = 0.198**. ~4× lift.

**Interpretation & recommendation:**
- The balance-sheet/leverage factors (Leverage, Debt-to-Assets) aren't junk — they're a **Safety/defensive
  theme distinct from profitability Quality**, and they fought the 2021–26 regime (levered/value names won).
  Don't delete them — **split them out** (their own theme, or fold into Low Vol/defensive).
- Redefine **Quality = profitability + cash-flow + efficiency, equal-weighted** (the 1/N principle from §10).
- Caveat: full-sample, partly regime-driven — validate walk-forward before promoting, and re-test once
  monthly snapshots expand the sample.
