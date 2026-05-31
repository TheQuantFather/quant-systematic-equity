#!/usr/bin/env python3
"""
create_barra.py — Barra-equivalent factor risk model.

Factor structure (K = 30):
  Market (1):        Intercept column of 1s. Captures the universe-wide premium
                     so sector factors become pure deviations from market.
  Sector (11):       All GICS sectors as dummies. Cap-weighted sum-to-zero
                     constraint resolves rank deficiency vs the market column.
  Style (6):         LMC11234, ABC11234, XYZ77890, RVL11234, W52H1234
                     (forward-filled from factors.db quarterly snapshots) +
                     beta_60d (computed daily from returns.db vs equal-weight
                     universe index).
  Fundamental (12):  TUV44567, WXY77890, JKL44556, ABC12345, DEF67890,
                     BCD44567, EFG77890, OPQ77890, LMN44567, KLM44567,
                     YZA11234, FCM11234  (forward-filled from factors.db).

Estimation pipeline:
  1. Daily constrained WLS cross-sectional regression: r_t = X_t f_t + ε_t
     subject to Σ_s w_s_cap · f_sector_s = 0 (cap-weighted sectors sum to 0).
     WLS weights = √mktcap from raw LMC factor value (canonical Barra USE4).
  2. Factor covariance F: two half-lives.
       Variances (diag):    EWMA hl=90d + Newey-West (5 lags)
       Correlations (off):  EWMA hl=240d (no NW — correlations are more stable)
       Reassemble:          F = D^½ R D^½, then spectral floor.
  3. Idio variance Δ:  EWMA (hl=60d) + Bayesian shrinkage (10% toward cross-mean).
  4. VRA (two scalars, each clipped to [0.5, 2.0]):
       B²_factor   = mean over k, last 60d of (f_t^k / σ̂_k)²  → scales F
       B²_specific = mean over i, last 60d of (ε_t^i / σ̂_i)² → scales Δ

Optimizer integration (optimize_portfolio.py):
  Stacked-L: L_barra = vstack([L_F.T @ X.T, diag(√δ)]).T (shape N×(K+N))
  Drop-in for Ledoit-Wolf L in  cp.norm(L_barra.T @ w, 2).

Output  data/risk.db (Barra tables, alongside Ledoit-Wolf covariance_matrix):
  factor_returns     trade_date  × factor_id × factor_return  (all trading days)
  factor_covariance  snapshot_date × K×K blob
  idiosyncratic_vars snapshot_date × security_id × idio_var
  factor_exposures   snapshot_date × security_id × factor_id × exposure

Update frequency:
  Estimation requires daily returns (stored permanently in factor_returns).
  Historical snapshots align with factors/models quarterly dates.
  Run weekly (no-arg or --date) for current-period portfolio construction.

Usage:
  python create_barra.py --backfill          # all 28 quarterly snapshot dates
  python create_barra.py --date 2026-05-01   # single snapshot for given date
  python create_barra.py                     # snapshot for most-recent Friday
"""

import argparse
import io
import json
import sqlite3
import sys
import zlib
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    RETURNS_DB, FACTORS_DB, UNIVERSE_DB, RISK_DB, FACTORS_REF,
    HL_FACTOR_VAR, HL_FACTOR_CORR, HL_IDIO, NW_LAGS, VRA_WINDOW,
    SHRINK_IDIO, EIGENFLOOR, VRA_MIN, VRA_MAX, MIN_STOCKS,
    BARRA_SECTORS as SECTORS,
)
from utils import get_db, get_logger, winsorized_zscore, get_snapshot_schedule

log = get_logger("create_barra")

# ---------------------------------------------------------------------------
# Load Barra factor IDs from reference CSV (order = barra_factor_order column)
# ---------------------------------------------------------------------------
_ref = pd.read_csv(str(FACTORS_REF))
_style_ref = (
    _ref[_ref["barra_factor_type"] == "style"]
    .sort_values("barra_factor_order")
)
_fund_ref = (
    _ref[_ref["barra_factor_type"] == "fundamental"]
    .sort_values("barra_factor_order")
)
STYLE_IDS       = _style_ref["factor_id"].tolist()
FUNDAMENTAL_IDS = _fund_ref["factor_id"].tolist()

# Ordered list used to index all K×K matrices and exposure vectors
FACTOR_NAMES = (
    ["market"]                                                  # index 0
    + [f"sec_{s.replace(' ', '_').lower()}" for s in SECTORS]   # indices 1-11
    + STYLE_IDS                                                 # indices 12-16
    + ["beta_60d"]                                              # index 17
    + FUNDAMENTAL_IDS                                           # indices 18-29
)
K = len(FACTOR_NAMES)  # 30

# Column-index anchors — keep in sync with FACTOR_NAMES layout above.
MARKET_IDX    = 0
SECTOR_START  = 1
SECTOR_END    = SECTOR_START + len(SECTORS)          # 12
STYLE_START   = SECTOR_END                           # 12
BETA_IDX      = STYLE_START + len(STYLE_IDS)         # 17
FUND_START    = BETA_IDX + 1                         # 18


def _get_snapshot_dates() -> list[str]:
    """
    Snapshot dates from the single source of truth — universe.db snapshot_schedule,
    restricted to dates whose factors have been computed (Barra needs factors first).
    Falls back to factors.db (snapshot_dates, then the factors table) for robustness.
    """
    try:
        dates = get_snapshot_schedule(computed_only=True)
        if dates:
            return dates
    except Exception:
        pass
    try:
        with get_db(FACTORS_DB) as conn:
            rows = conn.execute(
                "SELECT data_date FROM snapshot_dates ORDER BY data_date"
            ).fetchall()
            if rows:
                return [r[0] for r in rows]
            rows = conn.execute(
                "SELECT DISTINCT data_date FROM factors ORDER BY data_date"
            ).fetchall()
            if rows:
                return [r[0] for r in rows]
    except Exception:
        pass
    raise RuntimeError(
        "No snapshot dates found. Run create_universe.py --rebuild-schedule and create_factors.py first."
    )

_SECTOR_IDX = {s: i for i, s in enumerate(SECTORS)}


# ── DB schema ──────────────────────────────────────────────────────────────────

def _init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(RISK_DB))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS factor_returns (
            trade_date    TEXT NOT NULL,
            factor_id     TEXT NOT NULL,
            factor_return REAL NOT NULL,
            PRIMARY KEY (trade_date, factor_id)
        );
        CREATE TABLE IF NOT EXISTS factor_covariance (
            snapshot_date TEXT PRIMARY KEY,
            factor_names  TEXT NOT NULL,
            cov_blob      BLOB NOT NULL
        );
        CREATE TABLE IF NOT EXISTS idiosyncratic_vars (
            snapshot_date TEXT NOT NULL,
            security_id   TEXT NOT NULL,
            idio_var      REAL NOT NULL,
            PRIMARY KEY (snapshot_date, security_id)
        );
        CREATE TABLE IF NOT EXISTS factor_exposures (
            snapshot_date TEXT NOT NULL,
            security_id   TEXT NOT NULL,
            factor_id     TEXT NOT NULL,
            exposure      REAL NOT NULL,
            PRIMARY KEY (snapshot_date, security_id, factor_id)
        );
    """)
    conn.commit()
    return conn


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_returns_wide() -> pd.DataFrame:
    with get_db(RETURNS_DB) as conn:
        df = pd.read_sql_query(
            "SELECT date, isin, total_return FROM returns WHERE total_return IS NOT NULL",
            conn, parse_dates=["date"],
        )
    return df.pivot(index="date", columns="isin", values="total_return").sort_index()


def _load_factor_snapshots() -> dict:
    """
    Returns {snapshot_date_str: {isin: {factor_id: (raw_value, z_value)}}}.
    Raw LMC value is used for WLS weights; z-scores used for exposure matrix.
    """
    all_ids = STYLE_IDS + FUNDAMENTAL_IDS
    placeholders = ",".join("?" * len(all_ids))
    with get_db(FACTORS_DB) as conn:
        df = pd.read_sql_query(
            f"SELECT data_date, security_id, factor_id, factor_value, factor_value_z "
            f"FROM factors WHERE factor_id IN ({placeholders})",
            conn, params=all_ids,
        )

    snapshots: dict = {}
    for date_str, grp in df.groupby("data_date"):
        by_isin: dict = {}
        for _, row in grp.iterrows():
            isin = row["security_id"]
            if isin not in by_isin:
                by_isin[isin] = {}
            rv = float(row["factor_value"])   if pd.notna(row["factor_value"])   else None
            zv = float(row["factor_value_z"]) if pd.notna(row["factor_value_z"]) else None
            by_isin[isin][row["factor_id"]] = (rv, zv)
        snapshots[str(date_str)] = by_isin
    return snapshots


def _load_pit_membership() -> tuple[list[pd.Timestamp], dict[pd.Timestamp, set[str]]]:
    """
    Load Point-In-Time Russell 1000 membership from universe_snapshots.

    Returns:
      pit_snap_dates: sorted list of snapshot pd.Timestamps
      pit_membership: {snap_date_ts: set of ISINs in R1000 at that date}

    Why: the daily cross-sectional regression should include only stocks that
    were *actually in R1000* at trade date t, not today's R1000 members. Using
    today's `companies` table introduces survivorship bias (dropped names
    excluded everywhere) and inclusion bias (later-added names retroactively
    appearing in earlier regressions). The fix uses N-PORT-backed PIT holdings
    from `universe_snapshots`.
    """
    with get_db(UNIVERSE_DB) as conn:
        df = pd.read_sql_query(
            "SELECT snapshot_date, isin FROM universe_snapshots "
            "WHERE index_name = 'russell_1000' "
            "ORDER BY snapshot_date",
            conn,
        )
    if df.empty:
        raise RuntimeError(
            "No PIT R1000 membership in universe.db (universe_snapshots empty). "
            "Run create_universe.py --rebuild-snapshots first."
        )
    pit: dict[pd.Timestamp, set[str]] = {}
    for d, grp in df.groupby("snapshot_date"):
        pit[pd.Timestamp(d)] = set(grp["isin"])
    return sorted(pit.keys()), pit


def _pit_lookup(pit_snap_dates: list[pd.Timestamp]):
    """
    Build a fast `td → effective PIT snapshot date` lookup.
    Uses the latest snapshot ≤ td. Falls back to the earliest if td precedes all
    (mild look-ahead for pre-2021 dates, bounded and logged in main()).
    """
    arr = pd.DatetimeIndex(pit_snap_dates)
    def lookup(td: pd.Timestamp) -> pd.Timestamp:
        idx = arr.searchsorted(td, side="right") - 1
        return pit_snap_dates[max(int(idx), 0)]
    return lookup


def _load_universe() -> dict:
    """Returns {isin: gics_sector}."""
    with get_db(UNIVERSE_DB) as conn:
        df = pd.read_sql_query(
            "SELECT isin, gics_sector FROM companies WHERE gics_sector IS NOT NULL", conn
        )
    return dict(zip(df["isin"], df["gics_sector"]))


# ── Beta computation ───────────────────────────────────────────────────────────

def _compute_beta_60d(
    returns_wide: pd.DataFrame,
    pit_snap_dates: list[pd.Timestamp],
    pit_membership: dict[pd.Timestamp, set[str]],
) -> pd.DataFrame:
    """
    Rolling 60-day OLS beta for each stock vs the PIT R1000 market proxy.
    min_periods = 30. Result has same shape as returns_wide.

    Market proxy at trade date t = equal-weight mean of returns for stocks in
    R1000 as of t (latest PIT snapshot ≤ t). Without this, the proxy uses
    today's universe — survivorship + inclusion bias.
    """
    lookup = _pit_lookup(pit_snap_dates)
    cols = list(returns_wide.columns)
    snap_masks = {
        sd: np.fromiter((c in pit_membership[sd] for c in cols),
                        dtype=bool, count=len(cols))
        for sd in pit_snap_dates
    }
    R = returns_wide.values
    mkt_arr = np.full(len(returns_wide), np.nan)
    for i, td in enumerate(returns_wide.index):
        mask = snap_masks[lookup(td)]
        row  = R[i]
        valid = mask & np.isfinite(row)
        if valid.any():
            mkt_arr[i] = float(row[valid].mean())
    mkt = pd.Series(mkt_arr, index=returns_wide.index)

    betas = returns_wide.copy() * np.nan

    for col in returns_wide.columns:
        combined = pd.concat(
            [returns_wide[col].rename("s"), mkt.rename("m")], axis=1
        ).dropna()
        if len(combined) < 30:
            continue
        rc = combined["s"].rolling(60, min_periods=30).cov(combined["m"])
        rv = combined["m"].rolling(60, min_periods=30).var()
        b = (rc / rv).reindex(returns_wide.index)
        betas[col] = b

    return betas


# ── Exposure matrix ────────────────────────────────────────────────────────────

def _build_day_exposure(
    isins: list,
    isin_sector: dict,
    factor_snap: dict,   # {isin: {factor_id: (raw, z)}}
    beta_map: dict,      # {isin: beta_60d_value}
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build exposure matrix X (N×K), WLS weight vector (N,), and per-sector
    cap-weight vector (len(SECTORS),) for one trading day.

      • Market column (index 0): all 1s. Captures universe-wide return.
      • Sector dummies (indices SECTOR_START..SECTOR_END-1): binary 0/1.
      • Style/beta/fundamental (indices STYLE_START..end): z-scores; missing → 0.
      • WLS weights = √mktcap from raw LMC (canonical Barra USE4 — large caps
        anchor the factor return estimates).
      • Cap-per-sector accumulates mktcap by GICS sector → used to build the
        cap-weighted sum-to-zero constraint Σ_s w_s f_sec_s = 0.
    """
    N = len(isins)
    X = np.zeros((N, K))
    weights        = np.ones(N)
    cap_per_sector = np.zeros(len(SECTORS))

    # Market intercept (every stock loads 1.0)
    X[:, MARKET_IDX] = 1.0

    for i, isin in enumerate(isins):
        sector = isin_sector.get(isin)
        fvals  = factor_snap.get(isin, {})

        # Sector dummy
        if sector in _SECTOR_IDX:
            s_idx = _SECTOR_IDX[sector]
            X[i, SECTOR_START + s_idx] = 1.0

        # Style factors — z-scores
        for j, fid in enumerate(STYLE_IDS):
            pair = fvals.get(fid)
            if pair and pair[1] is not None and np.isfinite(pair[1]):
                X[i, STYLE_START + j] = pair[1]

        # Beta_60d
        b = beta_map.get(isin, np.nan)
        if b is not None and np.isfinite(float(b)):
            X[i, BETA_IDX] = float(b)

        # Fundamental factors — z-scores
        for j, fid in enumerate(FUNDAMENTAL_IDS):
            pair = fvals.get(fid)
            if pair and pair[1] is not None and np.isfinite(pair[1]):
                X[i, FUND_START + j] = pair[1]

        # WLS weight = √mktcap = exp(+lmc_raw/2). Cap is also used for the
        # sector sum-to-zero constraint (cap-weighted Barra convention).
        lmc_pair = fvals.get("LMC11234")
        if lmc_pair and lmc_pair[0] is not None and np.isfinite(lmc_pair[0]):
            cap = float(np.exp(lmc_pair[0]))
            weights[i] = np.sqrt(cap)
            if sector in _SECTOR_IDX:
                cap_per_sector[_SECTOR_IDX[sector]] += cap

    weights = np.clip(weights, 1e-8, None)
    weights /= weights.mean()  # normalise so mean weight = 1
    return X, weights, cap_per_sector


# ── Constrained WLS cross-sectional regression ───────────────────────────────

def _wls_constrained(
    r: np.ndarray, X: np.ndarray, w: np.ndarray, c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    WLS regression r = X f + ε with weights w, subject to linear constraint c'f = 0.

    We carry a Market intercept column AND 11 sector dummies in X — without the
    constraint these are perfectly collinear (sum of sectors = market column).
    With the cap-weighted sum-to-zero constraint on sector factors, the market
    column absorbs the universe-wide return and the sector returns become pure
    deviations from market.

    Implementation: pick the index `ref` with the largest |c[ref]| as the
    "dependent" factor. Reparametrise so f[ref] = -Σ_{i≠ref} c[i]/c[ref] · f[i]
    and run the reduced (K-1)-column WLS via lstsq (SVD; handles any residual
    rank deficiency). Back-compute f[ref] from the constraint.

    If c is all zero (degenerate — no constrained factors that day), falls back
    to plain WLS via lstsq.
    """
    sw = np.sqrt(w)
    sw_r = r * sw
    sw_X = X * sw[:, None]

    if not np.any(c):
        f, _, _, _ = np.linalg.lstsq(sw_X, sw_r, rcond=None)
        return f, r - X @ f

    ref = int(np.argmax(np.abs(c)))
    keep = np.delete(np.arange(len(c)), ref)

    # Reduced design: substituting f[ref] = -Σ_{i in keep} (c[i]/c[ref]) f[i]
    # into Xf gives Σ_{i in keep} (X[:,i] - (c[i]/c[ref]) X[:,ref]) · f[i].
    ratio = c[keep] / c[ref]                                      # (K-1,)
    sw_X_red = sw_X[:, keep] - np.outer(sw_X[:, ref], ratio)      # (N, K-1)

    f_red, _, _, _ = np.linalg.lstsq(sw_X_red, sw_r, rcond=None)

    f = np.empty(len(c))
    f[keep] = f_red
    f[ref]  = -float(ratio @ f_red)
    return f, r - X @ f


# ── Factor covariance: split-half-life EWMA + Newey-West ────────────────────

def _ewma_demean(F_hist: np.ndarray, hl: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute EWMA weights (normalised) and weighted-demeaned residuals U."""
    T = F_hist.shape[0]
    alpha = np.exp(-np.log(2.0) / hl)
    w = alpha ** np.arange(T - 1, -1, -1, dtype=float)
    w /= w.sum()
    mu = w @ F_hist
    U  = np.sqrt(w[:, None]) * (F_hist - mu)
    return w, U


def _ewma_split_cov(F_hist: np.ndarray, hl_var: int, hl_corr: int, nw_lags: int) -> np.ndarray:
    """
    Two-half-life EWMA factor covariance.

      Variances (diag of F): EWMA at hl_var + Newey-West (Bartlett, nw_lags).
                             Short HL: variances react quickly to vol regime
                             changes; NW corrects daily-return autocorrelation.
      Correlations (off-d):  EWMA at hl_corr, no NW. Long HL: correlations are
                             more stable than variances — using a short HL on
                             correlations would make them whip around during
                             vol spikes.
      Reassembly: F = D^½ · R · D^½, then symmetrise.

    Spectral floor is applied separately by the caller after VRA scaling.
    """
    # ── Variance block (short HL + NW) ──────────────────────────────────────
    _, U_v = _ewma_demean(F_hist, hl_var)
    V_var  = U_v.T @ U_v                           # Γ₀
    for k in range(1, nw_lags + 1):
        Gk = U_v[:-k].T @ U_v[k:]
        bf = 1.0 - k / (nw_lags + 1)               # Bartlett kernel weight
        V_var += bf * (Gk + Gk.T)
    V_var    = (V_var + V_var.T) / 2.0
    diag_var = np.maximum(np.diag(V_var), 1e-12)   # daily variances (NW-corrected)

    # ── Correlation block (long HL, no NW) ──────────────────────────────────
    _, U_c    = _ewma_demean(F_hist, hl_corr)
    V_cov_c   = U_c.T @ U_c
    V_cov_c   = (V_cov_c + V_cov_c.T) / 2.0
    std_c     = np.sqrt(np.maximum(np.diag(V_cov_c), 1e-12))
    R         = V_cov_c / np.outer(std_c, std_c)
    np.clip(R, -1.0, 1.0, out=R)
    np.fill_diagonal(R, 1.0)

    # ── Reassemble ──────────────────────────────────────────────────────────
    D_sqrt = np.sqrt(diag_var)
    F      = R * np.outer(D_sqrt, D_sqrt)
    return (F + F.T) / 2.0


def _spectral_floor(M: np.ndarray) -> np.ndarray:
    """Apply EIGENFLOOR to all eigenvalues of symmetric matrix M."""
    eigvals, eigvecs = np.linalg.eigh(M)
    return eigvecs @ np.diag(np.maximum(eigvals, EIGENFLOOR)) @ eigvecs.T


# ── Idiosyncratic variance: EWMA + Bayesian shrinkage ─────────────────────────

def _idio_variance(
    eps_sq_df: pd.DataFrame,   # (T, N) DataFrame of squared residuals, NaN where missing
    snap_ts: pd.Timestamp,
    hl: int,
    shrink: float,
) -> dict:
    """
    Per-stock EWMA of daily ε² up to snap_ts, annualised (×252), with
    Bayesian shrinkage (weight `shrink`) toward the cross-sectional mean.
    Returns {isin: annualised_idio_var}.
    """
    slice_df = eps_sq_df[eps_sq_df.index < snap_ts]
    if slice_df.empty:
        return {}

    T = len(slice_df)
    alpha = np.exp(-np.log(2.0) / hl)
    full_w = alpha ** np.arange(T - 1, -1, -1, dtype=float)  # (T,) unnormalised

    delta: dict = {}
    for col in slice_df.columns:
        col_data = slice_df[col].dropna()
        if len(col_data) < 5:
            continue
        # Weights at positions where this stock has data
        pos = slice_df.index.get_indexer(col_data.index)
        w_col = full_w[pos]
        w_col /= w_col.sum()
        delta[col] = float(w_col @ col_data.values) * 252.0

    if not delta:
        return delta

    mean_d = float(np.mean(list(delta.values())))
    for isin in delta:
        delta[isin] = (1.0 - shrink) * delta[isin] + shrink * mean_d

    return delta


# ── Volatility Regime Adjustment (split: factor B² + specific B²) ────────────

def _vra_split(
    f_df_pit: pd.DataFrame,       # (T_factor × K) factor returns (PIT, chronological)
    eps_sq_pit: pd.DataFrame,     # (T_eps × N_eps) squared residuals
    F_cov: np.ndarray,            # K×K annualised factor covariance (pre-VRA)
    delta: dict,                  # {isin: annualised idio variance}
    window: int,
) -> tuple[float, float]:
    """
    Two bias-statistic VRA scalars, each clipped to [VRA_MIN, VRA_MAX]:

      B²_factor   = mean over factors k and last `window` days of
                    (f_t^k / σ̂_k_daily)²
                    where σ̂_k_daily = √(F_cov[k,k] / 252).
                    > 1 → factors more volatile than F predicts → scale F up.

      B²_specific = mean over stocks i and last `window` days of
                    (ε_t^i / σ̂_i_daily)²
                    where σ̂_i_daily = √(δ_i / 252).
                    > 1 → idio risk under-predicted → scale Δ up.

    Bias-statistic form is cleaner than the realised/predicted-variance ratio
    because each z-score is a standardised observation; the mean of z² is a
    well-behaved estimator of misspecification rather than a noisy ratio.
    """
    # ── Factor bias ─────────────────────────────────────────────────────────
    if f_df_pit.empty:
        B2_factor = 1.0
    else:
        f_recent = f_df_pit.iloc[-window:]
        sigma_factor_daily = np.sqrt(np.maximum(np.diag(F_cov) / 252.0, 1e-12))   # (K,)
        z_factor = f_recent.values / sigma_factor_daily[None, :]                  # (T, K)
        B2_factor = float(np.nanmean(z_factor ** 2)) if np.isfinite(z_factor).any() else 1.0

    # ── Specific bias ───────────────────────────────────────────────────────
    if eps_sq_pit.empty or not delta:
        B2_specific = 1.0
    else:
        eps_sq_recent = eps_sq_pit.iloc[-window:]
        delta_daily   = np.array(
            [delta.get(c, np.nan) / 252.0 for c in eps_sq_recent.columns]
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            z2 = eps_sq_recent.values / delta_daily[None, :]
        B2_specific = float(np.nanmean(z2)) if np.isfinite(z2).any() else 1.0

    if not np.isfinite(B2_factor) or B2_factor <= 0:
        B2_factor = 1.0
    if not np.isfinite(B2_specific) or B2_specific <= 0:
        B2_specific = 1.0

    return (
        float(np.clip(B2_factor,   VRA_MIN, VRA_MAX)),
        float(np.clip(B2_specific, VRA_MIN, VRA_MAX)),
    )


# ── Core computation ───────────────────────────────────────────────────────────

def _compute_all_factor_returns(
    returns_wide: pd.DataFrame,
    betas_wide: pd.DataFrame,
    isin_sector: dict,
    factor_snapshots: dict,
    pit_snap_dates: list[pd.Timestamp],
    pit_membership: dict[pd.Timestamp, set[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run daily WLS cross-sectional regressions for all trading days.

    Returns:
        f_df:   (T_valid × K) DataFrame of factor returns indexed by trade date.
        eps_sq: (T_valid × N_all) DataFrame of squared residuals (NaN if stock
                not in regression on that day).
    """
    snap_keys = sorted(factor_snapshots.keys())
    trading_days = returns_wide.index

    f_data: dict   = {}   # {date_str: ndarray(K)}
    eps_sq_data: dict = {}  # {date_str: {isin: eps²}}

    pit_lookup_fn = _pit_lookup(pit_snap_dates)

    log.info("Running regressions (%d trading days × %d factors) — PIT R1000 universe ...",
             len(trading_days), K)
    for idx, td in enumerate(trading_days):
        td_str = td.strftime("%Y-%m-%d")

        latest_snap = next(
            (d for d in reversed(snap_keys) if d <= td_str), None
        )
        if latest_snap is None:
            continue

        fsnap     = factor_snapshots[latest_snap]
        day_rets  = returns_wide.loc[td].dropna()
        pit_isins = pit_membership[pit_lookup_fn(td)]
        isins_day = [i for i in day_rets.index
                     if i in isin_sector and i in fsnap and i in pit_isins]

        if len(isins_day) < MIN_STOCKS:
            continue

        r_day = day_rets[isins_day].values
        beta_row = betas_wide.loc[td] if td in betas_wide.index else pd.Series(dtype=float)
        beta_map = beta_row.dropna().to_dict()

        X_day, w_day, cap_sec = _build_day_exposure(isins_day, isin_sector, fsnap, beta_map)

        # Build cap-weighted sum-to-zero constraint on sector factors only:
        # c is K-vector with cap weights at sector indices, 0 elsewhere.
        c = np.zeros(K)
        c[SECTOR_START:SECTOR_END] = cap_sec
        f_t, eps_t = _wls_constrained(r_day, X_day, w_day, c)

        f_data[td_str]    = f_t
        eps_sq_data[td_str] = {isin: e ** 2 for isin, e in zip(isins_day, eps_t)}

        if (idx + 1) % 500 == 0:
            log.info("  ... %d/%d days", idx + 1, len(trading_days))

    # ── Build DataFrames ────────────────────────────────────────────────────
    if not f_data:
        raise RuntimeError("No valid regression days found. Check data sources.")

    f_df = pd.DataFrame.from_dict(f_data, orient="index", columns=FACTOR_NAMES)
    f_df.index = pd.to_datetime(f_df.index)
    f_df.sort_index(inplace=True)

    # Build eps_sq as wide DataFrame (trade_date × isin)
    all_isins = sorted({i for d in eps_sq_data.values() for i in d})
    eps_sq_df = pd.DataFrame(index=pd.to_datetime(sorted(eps_sq_data.keys())),
                              columns=all_isins, dtype=float)
    for date_str, row_dict in eps_sq_data.items():
        for isin, v in row_dict.items():
            eps_sq_df.loc[pd.Timestamp(date_str), isin] = v

    log.info("Done: %d days, %d stocks with residuals", len(f_df), len(all_isins))
    return f_df, eps_sq_df


def _build_and_save_snapshot(
    snap_date_str: str,
    returns_wide: pd.DataFrame,
    f_df: pd.DataFrame,
    eps_sq_df: pd.DataFrame,
    factor_snapshots: dict,
    isin_sector: dict,
    betas_wide: pd.DataFrame,
    conn: sqlite3.Connection,
    pit_snap_dates: list[pd.Timestamp],
    pit_membership: dict[pd.Timestamp, set[str]],
) -> None:
    """Compute full Barra model for one snapshot date and write to risk.db."""
    snap_ts = pd.Timestamp(snap_date_str)

    # Strict point-in-time: all time-series data must precede snap_ts so that
    # the same-day returns are never used in the covariance estimate (look-ahead
    # bias). Historical --backfill runs respect this too, so all snapshots are
    # built consistently from T-1 data.
    f_df_pit        = f_df[f_df.index             < snap_ts]
    returns_pit     = returns_wide[returns_wide.index < snap_ts]
    eps_sq_pit      = eps_sq_df[eps_sq_df.index   < snap_ts]
    betas_pit       = betas_wide[betas_wide.index  < snap_ts]

    if len(f_df_pit) < 60:
        log.warning("Insufficient history for %s — skipping.", snap_date_str)
        return

    F_hist = f_df_pit.values  # (T, K)

    # ── Factor covariance: two-half-life EWMA ────────────────────────────────
    # Variances (diag) at HL_FACTOR_VAR + NW; correlations at HL_FACTOR_CORR.
    # Returned in daily units; annualise so portfolio variance via
    # ||L_barra.T @ w||² matches risk.db's annual convention.
    F_cov = _ewma_split_cov(F_hist, HL_FACTOR_VAR, HL_FACTOR_CORR, NW_LAGS) * 252.0

    # ── Idiosyncratic variance ───────────────────────────────────────────────
    delta = _idio_variance(eps_sq_pit, snap_ts, HL_IDIO, SHRINK_IDIO)

    # ── Resolve factor snapshot and beta map for this date ───────────────────
    snap_keys = sorted(factor_snapshots.keys())
    latest_snap = next(
        (d for d in reversed(snap_keys) if d <= snap_date_str), snap_keys[0]
    )
    fsnap_now = factor_snapshots[latest_snap]

    # Use the last available trading day strictly before snap_date for betas
    beta_row = (
        betas_pit.iloc[-1]
        if not betas_pit.empty
        else betas_wide.iloc[-1]
    )
    beta_map_now = beta_row.dropna().to_dict()

    # ── Split VRA: factor-bias and specific-bias ─────────────────────────────
    # PIT-filter the snapshot universe: include only stocks in R1000 at snap_date.
    pit_now = pit_membership[_pit_lookup(pit_snap_dates)(snap_ts)]
    isins_snap = [i for i in fsnap_now if i in isin_sector and i in pit_now]
    B2_factor, B2_specific = _vra_split(f_df_pit, eps_sq_pit, F_cov, delta, VRA_WINDOW)
    if abs(B2_factor - 1.0) > 0.01 or abs(B2_specific - 1.0) > 0.01:
        log.info("VRA: B²_factor=%.3f  B²_specific=%.3f", B2_factor, B2_specific)
    F_cov = B2_factor * F_cov
    delta = {isin: B2_specific * v for isin, v in delta.items()}

    # ── Spectral floor ───────────────────────────────────────────────────────
    F_cov = _spectral_floor(F_cov)

    # ── Exposure matrix at snapshot date ────────────────────────────────────
    X_now, _, _ = _build_day_exposure(isins_snap, isin_sector, fsnap_now, beta_map_now)

    # ── Persist ──────────────────────────────────────────────────────────────
    cov_blob = zlib.compress(F_cov.astype(np.float32).tobytes())
    conn.execute(
        "INSERT OR REPLACE INTO factor_covariance "
        "(snapshot_date, factor_names, cov_blob) VALUES (?,?,?)",
        (snap_date_str, json.dumps(FACTOR_NAMES), cov_blob),
    )

    conn.execute("DELETE FROM idiosyncratic_vars WHERE snapshot_date=?", (snap_date_str,))
    if delta:
        conn.executemany(
            "INSERT INTO idiosyncratic_vars (snapshot_date, security_id, idio_var) VALUES (?,?,?)",
            [(snap_date_str, isin, float(v)) for isin, v in delta.items()],
        )

    conn.execute("DELETE FROM factor_exposures WHERE snapshot_date=?", (snap_date_str,))
    exp_rows = [
        (snap_date_str, isin, fn, float(X_now[i, j]))
        for i, isin in enumerate(isins_snap)
        for j, fn in enumerate(FACTOR_NAMES)
        if X_now[i, j] != 0.0
    ]
    if exp_rows:
        conn.executemany(
            "INSERT INTO factor_exposures "
            "(snapshot_date, security_id, factor_id, exposure) VALUES (?,?,?,?)",
            exp_rows,
        )

    conn.commit()

    log.info(
        "Saved snapshot %s: F(%dx%d), δ(%d stocks), X(%dx%d), "
        "VRA_factor=%.3f  VRA_specific=%.3f",
        snap_date_str, K, K, len(delta), len(isins_snap), K,
        B2_factor, B2_specific,
    )


# ── Snapshot date helpers ──────────────────────────────────────────────────────

def _most_recent_friday() -> str:
    d = date.today()
    while d.weekday() != 4:   # 4 = Friday
        d -= timedelta(days=1)
    return d.strftime("%Y-%m-%d")




# ── Main entry point ───────────────────────────────────────────────────────────

def main(snapshot_dates: list[str]) -> None:
    log.info("=== Barra Factor Risk Model ===")

    log.info("Loading universe...")
    isin_sector = _load_universe()
    log.info("  %d companies", len(isin_sector))

    log.info("Loading PIT R1000 membership ...")
    pit_snap_dates, pit_membership = _load_pit_membership()
    log.info("  %d snapshots: %s → %s, %d unique ISINs ever in R1000",
             len(pit_snap_dates), pit_snap_dates[0].date(), pit_snap_dates[-1].date(),
             len({i for s in pit_membership.values() for i in s}))

    log.info("Loading factor snapshots...")
    factor_snapshots = _load_factor_snapshots()
    log.info("  %d snapshots, %d stock-snapshot rows",
             len(factor_snapshots), sum(len(v) for v in factor_snapshots.values()))

    log.info("Loading returns...")
    returns_wide = _load_returns_wide()
    log.info("  %d trading days, %d stocks", len(returns_wide), returns_wide.shape[1])
    log.info("  Period: %s → %s", returns_wide.index[0].date(), returns_wide.index[-1].date())

    if returns_wide.index[0] < pit_snap_dates[0]:
        n_pre = int((returns_wide.index < pit_snap_dates[0]).sum())
        log.warning(
            "Returns extend %d trading days before earliest PIT R1000 snapshot (%s); "
            "those dates use the earliest snapshot as fallback (mild look-ahead, bounded).",
            n_pre, pit_snap_dates[0].date(),
        )

    log.info("Computing rolling 60-day betas (PIT R1000 market proxy)...")
    betas_wide = _compute_beta_60d(returns_wide, pit_snap_dates, pit_membership)

    log.info("Computing daily factor returns (PIT R1000 universe)...")
    f_df, eps_sq_df = _compute_all_factor_returns(
        returns_wide, betas_wide, isin_sector, factor_snapshots,
        pit_snap_dates, pit_membership,
    )

    conn = _init_db()

    # Persist all factor returns (INSERT OR REPLACE for idempotency)
    log.info("Saving factor returns to DB...")
    rows = [
        (ts.strftime("%Y-%m-%d"), fn, float(val))
        for ts, row in f_df.iterrows()
        for fn, val in zip(FACTOR_NAMES, row.values)
        if np.isfinite(val)
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO factor_returns (trade_date, factor_id, factor_return) VALUES (?,?,?)",
        rows,
    )
    conn.commit()
    log.info("  %s factor return rows saved", f"{len(rows):,}")

    # Build snapshots where we have at least some factor returns before the snap
    # date (strict PIT: snap-date returns are never used). Allow up to 7 calendar
    # days gap so a Friday snapshot builds when only Thursday returns are in DB.
    last_fr = f_df.index[-1].date()
    snap_dates_filtered = [
        d for d in snapshot_dates
        if date.fromisoformat(d) <= last_fr + timedelta(days=7)
    ]
    log.info("Building %d Barra snapshot(s)...", len(snap_dates_filtered))
    for snap_date_str in snap_dates_filtered:
        _build_and_save_snapshot(
            snap_date_str, returns_wide, f_df, eps_sq_df,
            factor_snapshots, isin_sector, betas_wide, conn,
            pit_snap_dates, pit_membership,
        )

    conn.close()
    log.info("Done.")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Barra-equivalent factor risk model.")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--backfill", action="store_true",
        help="Compute snapshots for all dates in factors.db snapshot_dates table.",
    )
    grp.add_argument(
        "--date", metavar="YYYY-MM-DD", action="append", dest="dates",
        help="Compute a snapshot for the given date (repeatable: --date D1 --date D2).",
    )
    args = parser.parse_args()

    if args.backfill:
        dates = _get_snapshot_dates()
        log.info("Backfill: %d snapshots from %s to %s", len(dates), dates[0], dates[-1])
    elif args.dates:
        dates = sorted(set(args.dates))
    else:
        dates = [_most_recent_friday()]
        log.info("Snapshot for most-recent Friday: %s", dates[0])

    main(dates)
