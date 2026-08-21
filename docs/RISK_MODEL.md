# Risk Model

A plain-language guide to how risk is estimated in this system, why it's built
that way, and where it sits in the overall process.

---

## 1. What the risk model is for

Alpha tells you *which stocks you like*. The risk model tells you *how much they
can move, and how they move together*. The optimizer needs both: it maximises
expected return while keeping portfolio risk under control. Without a risk model
you'd happily load up on ten names that all crash together.

Concretely, the risk model produces a **covariance matrix** — a big table of how
every stock co-moves with every other stock. The optimizer uses it to:

- cap total portfolio volatility,
- limit how far the portfolio drifts from its benchmark ("active risk"),
- avoid stacking up correlated bets.

We build **two** risk models, but only one feeds portfolio construction:

| Model | What it is | When used |
|-------|-----------|-----------|
| **Barra factor model** | Risk explained through a small set of common factors | Production portfolio construction |
| **Ledoit-Wolf** | Shrunk sample covariance straight from returns | Analysis / sanity check only |

Both live in **`risk.db`**, but the optimizer requires Barra coverage. Ledoit-Wolf
is kept in the database for comparison, diagnostics, and risk-explorer views; it
is not a fallback when Barra is missing.

---

## 2. Ledoit-Wolf model (the simple one)

**Idea:** take the last **252 trading days** (1 year) of daily returns for every
stock, compute the plain sample covariance, then *shrink* it toward a simple
target.

**Why shrink?** With ~1000 stocks and only 252 days, the raw sample covariance is
unreliable — too many numbers estimated from too little data, full of noise.
Ledoit-Wolf automatically blends the noisy sample matrix with a stable, simple one
and picks the blend (the "shrinkage coefficient") that minimises error. It's a
well-known, robust, parameter-free method.

**Key choices (all in `config.py`):**

| Setting | Value | Why |
|---------|-------|-----|
| Lookback | 252 days | One year of daily history |
| Min history | 126 days | A stock needs ≥6 months of data to be included |
| Winsor clip | ±50% | Cap crazy daily returns so one bad print doesn't dominate |

Result is **annualised** (daily covariance × 252) and stored as a compressed blob
in `risk.db → covariance_matrix`.

**Limitation:** it's a black box of pairwise correlations. It can't tell you *why*
two stocks move together, and it struggles to give stable risk for stocks with
short history. That's what Barra fixes.

---

## 3. Barra factor model (the main one)

**Idea:** instead of estimating every pairwise correlation directly, explain each
stock's return as exposure to a handful of **common factors** plus a stock-specific
piece. Stocks move together because they share factor exposures (same sector, both
"value", both high-momentum, etc.).

This is the classic Barra structure:

```
   Σ  =  X · F · Xᵀ  +  Δ
   │      │   │   │      │
   │      │   │   │      └─ Δ: stock-specific ("idiosyncratic") risk, a diagonal
   │      │   │   └──────── Xᵀ: exposures, transposed
   │      │   └──────────── F:  factor covariance (how the factors co-move)
   │      └──────────────── X:  exposures (how much each stock loads on each factor)
   └─────────────────────── Σ:  the full stock covariance matrix
```

The win: instead of ~1000×1000 noisy correlations, you estimate a small `K×K`
factor covariance plus one number per stock. Far more stable, and **interpretable**
— you can decompose any portfolio's risk into "how much is sector bets vs style
bets vs stock-picking."

### The factors (K = 24)

The factor set is `[ market | sectors | beta | style/alpha models ]`:

| Block | Count | What |
|-------|-------|------|
| **Market** | 1 | A column of 1s — captures the whole-universe move so sectors become *deviations* from market |
| **Sectors** | 11 | The 11 GICS sectors as 0/1 dummies |
| **Beta** | 1 | Rolling **60-day** market beta of each stock |
| **Style/alpha models** | 11 | Profitability, Balance-Sheet Quality, Value, Growth, Momentum, Size, Low Vol, Liquidity, LT Reversal, ST Reversal, Short Interest |

The 11 style factors are exactly the base alpha models from `models.db` — their
cross-sectional z-scores *are* the Barra exposures. This is deliberate: the risk
model speaks the same language as the alpha model, so the optimizer can reason
about return and risk on the same axes.

The factor list, order, and orthogonalisation rules are **not hardcoded** — they're
read from `models_reference.csv` (the `barra_risk_factor` / `barra_order` columns)
via `get_barra_layout()`, the single source of truth.

### A few important design choices

- **Orthogonalisation:** some factors are residualised against others so they
  measure something *distinct*. Liquidity is made ⊥ Size (otherwise it's just size
  in disguise); Short Interest is made ⊥ Liquidity. Done per snapshot, then
  re-standardised.
- **Market + sectors are collinear** (the 11 sector dummies sum to the market
  column). We resolve this with a **cap-weighted sum-to-zero constraint** on the
  sectors: the big market move lands in the market factor, and each sector factor
  becomes a pure "this sector vs the market" tilt.
- **Cap weighting (√mktcap):** the daily regressions are weighted by the square
  root of market cap — large, liquid names anchor the factor estimates. This is the
  canonical Barra USE4 convention.

---

## 4. How Barra is estimated (the pipeline)

For every trading day we run one **cross-sectional regression**:

> *"Given each stock's factor exposures, what factor returns best explain today's
> actual stock returns?"*

`r = X·f + ε`, solved by **constrained weighted least squares** (weights = √mktcap,
constraint = sectors sum to zero). The outputs:

- **`f`** — that day's factor returns (stored permanently),
- **`ε`** — the leftover stock-specific return (its variance becomes Δ).

Daily stock returns are clipped at **±50%** before Barra estimation, matching the
Ledoit-Wolf input treatment. This is a risk-estimation guardrail: raw return data
can contain split, delisting, or stale-price artifacts, and one bad print should
not reset factor covariance, beta, VRA, or idiosyncratic variance.

We do this for the whole history, building a long daily time series of factor
returns. From that we build the two pieces of the model:

### Factor covariance `F` — two half-lives

Variances and correlations change at different speeds, so they get different memory:

| Piece | Half-life (lookback) | Extras | Why |
|-------|---------------------|--------|-----|
| **Variances** (diagonal) | **90 days** (EWMA) | + Newey-West, 5 lags | Vol regimes change fast; react quickly. NW corrects daily autocorrelation. |
| **Correlations** (off-diagonal) | **240 days** (EWMA) | none | Correlations are more stable; a short window would make them whip around in a vol spike. |

(EWMA = exponentially-weighted, so recent days matter more. "Half-life 90d" means a
day 90 days ago carries half the weight of today.) The two are reassembled into one
matrix and then **spectral-floored** (tiny/negative eigenvalues bumped up) to keep
it a valid, positive-definite covariance. Finally annualised (× 252).

### Idiosyncratic variance `Δ` — per stock

Each stock's own risk = EWMA of its squared residuals (**60-day** half-life),
annualised, then **Bayesian-shrunk 10%** toward the cross-sectional average (so a
stock with little data doesn't get a wild estimate).

Names with market data but no fundamental constituents are still carried through
the factor/model/Barra chain using price-only factors such as momentum, low-vol,
liquidity, reversals, short-interest and any other market-data factors available.
The model layer already shrinks sparse scores by valid/applicable factor coverage,
so these names do not receive full-conviction fundamental style exposure. If a
newly covered name has factor exposure but no residual history yet, Barra stores
the snapshot median `Delta` for that name until residuals accumulate.

At optimizer load time, Barra requires factor-exposure coverage for every
investable name after share-class aliases are applied. If any name would otherwise
enter with zero market/sector/style exposure, optimization is aborted instead of
substituting Ledoit-Wolf or treating the name as idiosyncratic-only. If a name has
factor exposures but is missing idiosyncratic variance, the loader uses the
snapshot median `Δ` rather than a hardcoded low-risk default.

---

## 5. VRA — Volatility Regime Adjustment

**The problem:** EWMA models react to vol *with a lag*. When markets suddenly get
violent (March 2020), the model under-predicts risk for a few weeks until the new
data filters in. When markets calm down, it over-predicts.

**The fix:** VRA is a fast correction that asks *"lately, have realised moves been
bigger or smaller than the model predicted?"* and scales the model up or down to
match. We compute **two** scalars over the last **60 days**:

- **B²_factor** — were the *factors* more volatile than `F` predicted? → scales `F`.
- **B²_specific** — were the *stock-specific* moves bigger than `Δ` predicted? → scales `Δ`.

Each is a "bias statistic": the average of squared standardised moves
(`(actual / predicted)²`). A value of **1.0 means the model was spot-on**; >1 means
it under-predicted risk (scale up), <1 means over-predicted (scale down).

Both are **clipped to [0.5, 2.0]** so the adjustment can never more-than-double or
more-than-halve the model. In normal conditions they sit around **0.8–1.2** — if
they're pinned at a clip, something's off and it's worth investigating.

---

## 6. Lookback summary (one place)

| Quantity | Lookback / half-life | Model |
|----------|---------------------|-------|
| Sample covariance | 252 days | Ledoit-Wolf |
| Min history to include a stock | 126 days | Ledoit-Wolf |
| Return clip | ±50% | Ledoit-Wolf + Barra risk estimation |
| Market beta | 60-day rolling | Barra |
| Factor variances | 90-day half-life (+ NW 5 lags) | Barra |
| Factor correlations | 240-day half-life | Barra |
| Idiosyncratic variance | 60-day half-life (+ 10% shrink) | Barra |
| VRA window | 60 days | Barra |

---

## 7. Point-in-time discipline (avoiding look-ahead)

A backtest is worthless if it secretly uses tomorrow's information. Two guards:

1. **Universe membership is point-in-time.** Each day's regression includes only
   the stocks that were *actually in the Russell 1000 / S&P 500 on that day*, read
   from `universe_snapshots` (N-PORT-backed holdings). Using today's index would
   bake in survivorship bias (losers already removed) and inclusion bias (winners
   added retroactively). The market-beta proxy uses this PIT universe too.

2. **Snapshots use only strictly-earlier data.** When building the risk model *as
   of* a date, every input must come from **before** that date (`< snap_ts`) — the
   same-day return is never used. This holds in historical backfills too, so every
   snapshot is built the same honest way.

---

## 8. Databases & tables

Everything lives in **`risk.db`**.

**Ledoit-Wolf:**

| Table | Keyed by | Holds |
|-------|----------|-------|
| `covariance_matrix` | `data_date` | n_stocks, shrinkage coeff, the matrix blob, ISIN list |

**Barra:**

| Table | Keyed by | Holds |
|-------|----------|-------|
| `factor_returns` | `(trade_date, factor_id)` | Daily factor returns — every trading day |
| `factor_covariance` | `snapshot_date` | The `K×K` factor covariance `F` (blob) |
| `idiosyncratic_vars` | `(snapshot_date, security_id)` | Per-stock specific variance `Δ` |
| `factor_exposures` | `(snapshot_date, security_id, factor_id)` | The exposure matrix `X` |

Inputs come from `returns.db` (daily returns), `models.db` (the style-factor
z-scores = exposures), `factors.db` (raw market cap for the √mktcap weights), and
`universe.db` (PIT membership + GICS sectors).

---

## 9. How it fits the overall process

```
constituents.db ─▶ factors.db ─▶ models.db ─┐
   (fundamentals)   (30+ factors) (11 models)│
                                             ├─▶  risk.db  ─▶  optimizer  ─▶  portfolio
returns.db ──────────────────────────────────┘   (Barra)       (CVXPY)
   (daily prices/returns)
```

The risk model is the **last build step before optimisation**. The daily/periodic
run order is:

```
create_factors  →  create_models  →  create_barra
```

`create_barra` builds the factor model and must run *after* factors and models,
because Barra exposures **are** the model z-scores. `create_risk` can still build
the Ledoit-Wolf matrix for analysis, but it is not part of the portfolio
construction dependency chain.

Coverage can be checked with:

```
python .scratch/barra_coverage_audit.py --date YYYY-MM-DD --index russell_1000
```

In the optimizer the Barra model plugs in as a stacked matrix built from the
Cholesky factor of `F`: `L_barra = [ chol(F)ᵀ·Xᵀ ; √Δ ]`. The optimizer uses
`‖Lᵀw‖²` for portfolio variance, active-risk limits, and vol targets.

---

## 10. Quick talking points

- **Barra is the optimizer risk model** — Ledoit-Wolf remains useful for
  diagnostics, but never silently replaces Barra in portfolio construction.
- **Barra = `XFXᵀ + Δ`** — common factor risk + stock-specific risk. Far more stable
  and interpretable than 1000×1000 raw correlations.
- **Risk and alpha share factors** — the 11 style factors in Barra *are* the alpha
  models, so risk and return live on the same axes.
- **Different half-lives for different things** — fast (90d) for variances, slow
  (240d) for correlations, because they move at different speeds.
- **VRA** keeps the model honest in fast-moving markets by scaling to recently
  realised volatility, clipped so it can't overreact.
- **Strict point-in-time** everywhere — PIT index membership and earlier-only data —
  so backtests don't cheat.
