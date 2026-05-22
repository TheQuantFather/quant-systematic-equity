#!/usr/bin/env python3
"""
create_barra.py — Barra-equivalent factor risk model.

Factor structure (K = 29):
  Sector (11):       All GICS sectors as dummies — no reference dropped.
                     numpy lstsq (SVD) handles rank-deficiency; sector returns
                     represent cross-sectional deviations, not vs a reference.
  Style (6):         LMC11234, ABC11234, XYZ77890, RVL11234, W52H1234
                     (forward-filled from factors.db quarterly snapshots) +
                     beta_60d (computed daily from returns.db vs equal-weight
                     universe index).
  Fundamental (12):  TUV44567, WXY77890, JKL44556, ABC12345, DEF67890,
                     BCD44567, EFG77890, OPQ77890, LMN44567, KLM44567,
                     YZA11234, FCM11234  (forward-filled from factors.db).

Estimation pipeline:
  1. Daily WLS cross-sectional regression: r_t = X_t f_t + ε_t
     Weights = 1/√mktcap from raw LMC factor value.
  2. Factor covariance F: EWMA (hl=90d) + Newey-West (5 lags) + spectral floor.
  3. Idio variance Δ:  EWMA (hl=60d) + Bayesian shrinkage (10% toward cross-mean).
  4. VRA:              60-day bias statistic B² ∈ [0.25, 4.0]; scales F and Δ.

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
    HL_FACTOR_COV, HL_IDIO, NW_LAGS, VRA_WINDOW,
    SHRINK_IDIO, EIGENFLOOR, VRA_MIN, VRA_MAX, MIN_STOCKS,
    BARRA_SECTORS as SECTORS,
)
from utils import get_db, get_logger, winsorized_zscore

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
    [f"sec_{s.replace(' ', '_').lower()}" for s in SECTORS]   # indices 0-10
    + STYLE_IDS                                                 # indices 11-15
    + ["beta_60d"]                                              # index 16
    + FUNDAMENTAL_IDS                                           # indices 17-28
)
K = len(FACTOR_NAMES)  # 29


def _get_snapshot_dates() -> list[str]:
    """Discover snapshot dates from factors.db — snapshot_dates table, then factors table."""
    try:
        with get_db(FACTORS_DB) as conn:
            rows = conn.execute(
                "SELECT data_date FROM snapshot_dates ORDER BY data_date"
            ).fetchall()
            if rows:
                return [r[0] for r in rows]
            # Fallback: derive from factors table itself (handles pre-snapshot_dates era)
            rows = conn.execute(
                "SELECT DISTINCT data_date FROM factors ORDER BY data_date"
            ).fetchall()
            if rows:
                return [r[0] for r in rows]
    except Exception:
        pass
    raise RuntimeError(
        "No snapshot dates found in factors.db. Run create_factors.py first."
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


def _load_universe() -> dict:
    """Returns {isin: gics_sector}."""
    with get_db(UNIVERSE_DB) as conn:
        df = pd.read_sql_query(
            "SELECT isin, gics_sector FROM companies WHERE gics_sector IS NOT NULL", conn
        )
    return dict(zip(df["isin"], df["gics_sector"]))


# ── Beta computation ───────────────────────────────────────────────────────────

def _compute_beta_60d(returns_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling 60-day OLS beta for each stock vs equal-weight universe index.
    min_periods = 30.  Result has same shape as returns_wide.
    """
    mkt = returns_wide.mean(axis=1)  # equal-weight market proxy
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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build exposure matrix X (N×K) and WLS weight vector (N,) for one trading day.

    Sector dummies are binary (0/1).  All other columns use factor z-scores
    from the most recent quarterly snapshot.  Missing → 0 (neutral exposure).
    WLS weights = 1/√mktcap from raw LMC; falls back to equal weight if absent.
    """
    N = len(isins)
    X = np.zeros((N, K))
    weights = np.ones(N)

    for i, isin in enumerate(isins):
        sector = isin_sector.get(isin)
        fvals  = factor_snap.get(isin, {})

        # Sector dummy (indices 0..10)
        if sector in _SECTOR_IDX:
            X[i, _SECTOR_IDX[sector]] = 1.0

        # Style factors — z-scores (indices 11..15)
        for j, fid in enumerate(STYLE_IDS):
            pair = fvals.get(fid)
            if pair and pair[1] is not None and np.isfinite(pair[1]):
                X[i, 11 + j] = pair[1]

        # Beta_60d (index 16)
        b = beta_map.get(isin, np.nan)
        if b is not None and np.isfinite(float(b)):
            X[i, 16] = float(b)

        # Fundamental factors — z-scores (indices 17..28)
        for j, fid in enumerate(FUNDAMENTAL_IDS):
            pair = fvals.get(fid)
            if pair and pair[1] is not None and np.isfinite(pair[1]):
                X[i, 17 + j] = pair[1]

        # WLS weight: 1/√mktcap ≈ exp(-lmc_raw/2)
        lmc_pair = fvals.get("LMC11234")
        if lmc_pair and lmc_pair[0] is not None and np.isfinite(lmc_pair[0]):
            weights[i] = np.exp(-lmc_pair[0] / 2.0)

    weights = np.clip(weights, 1e-8, None)
    weights /= weights.mean()  # normalise so mean weight = 1
    return X, weights


# ── WLS cross-sectional regression ────────────────────────────────────────────

def _wls_regression(
    r: np.ndarray, X: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    WLS: r = X f + ε with weights w.
    Equivalent to OLS on sqrt(w)*r ~ sqrt(w)*X.
    numpy lstsq uses SVD — handles rank-deficient X (all 11 sector dummies + market).
    Returns (f: K,), (eps: N,).
    """
    sw = np.sqrt(w)
    f, _, _, _ = np.linalg.lstsq(X * sw[:, None], r * sw, rcond=None)
    return f, r - X @ f


# ── Factor covariance: EWMA + Newey-West ──────────────────────────────────────

def _ewma_nw_cov(F_hist: np.ndarray, hl: int, nw_lags: int) -> np.ndarray:
    """
    EWMA covariance with Newey-West autocorrelation correction.

    F_hist: (T, K) array of factor returns (chronological, most recent last).
    Returns (K, K) PSD covariance matrix.

    Approach:
      1. Compute exponential weights w_t = α^(T-1-t), normalised.
      2. EWMA-demean: μ = w · F_hist.
      3. Weighted residuals: U_t = √w_t · (F_t - μ).
      4. Γ₀ = U'U  (EWMA variance).
      5. Γ_k = U[:-k]'U[k:]  (lag-k cross-product of weighted residuals).
      6. V_NW = Γ₀ + Σ_k (1 - k/(L+1)) · (Γ_k + Γ_k').
    """
    T, _ = F_hist.shape
    alpha = np.exp(-np.log(2.0) / hl)
    w = alpha ** np.arange(T - 1, -1, -1, dtype=float)
    w /= w.sum()

    mu = w @ F_hist                               # (K,)
    U  = np.sqrt(w[:, None]) * (F_hist - mu)      # (T, K) weighted demeaned

    V = U.T @ U                                   # Γ₀
    for k in range(1, nw_lags + 1):
        Gk = U[:-k].T @ U[k:]                    # (K, K)
        bf = 1.0 - k / (nw_lags + 1)             # Bartlett kernel weight
        V += bf * (Gk + Gk.T)

    return (V + V.T) / 2.0                        # enforce symmetry


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
    slice_df = eps_sq_df[eps_sq_df.index <= snap_ts]
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


# ── Volatility Regime Adjustment ──────────────────────────────────────────────

def _vra(
    returns_wide: pd.DataFrame,
    F_cov: np.ndarray,       # K×K annualised factor covariance (pre-VRA)
    delta: dict,             # {isin: annualised idio var}
    snap_ts: pd.Timestamp,
    factor_snap: dict,       # {isin: {factor_id: (raw, z)}}
    isin_sector: dict,
    beta_map: dict,
) -> float:
    """
    B² = realised_var_ew / predicted_var_ew over last VRA_WINDOW trading days.
    Clipped to [VRA_MIN, VRA_MAX].

    predicted_var_ew (daily) = (X̄' F X̄ + mean(δ)/N) / 252
    where X̄ = mean of factor exposures across all stocks at snapshot date.
    """
    dates = returns_wide.index
    pos = dates.get_indexer([snap_ts], method="pad")[0]
    if pos < VRA_WINDOW:
        return 1.0

    window_slice = returns_wide.iloc[pos - VRA_WINDOW + 1 : pos + 1]
    r_ew = window_slice.mean(axis=1, skipna=True).values   # (VRA_WINDOW,)
    realised_var = float(np.nanmean(r_ew ** 2))

    isins = [i for i in factor_snap if i in isin_sector]
    if not isins:
        return 1.0

    X_snap, _ = _build_day_exposure(isins, isin_sector, factor_snap, beta_map)
    x_bar = X_snap.mean(axis=0)                             # (K,) mean exposure
    factor_daily_var = float(x_bar @ F_cov @ x_bar) / 252.0

    delta_vals = np.array([delta.get(i, 0.04) for i in isins])
    N = len(isins)
    idio_daily_var = float(np.mean(delta_vals)) / (N * 252.0)

    predicted_var = factor_daily_var + idio_daily_var
    if predicted_var < 1e-12:
        return 1.0

    B2 = realised_var / predicted_var
    return float(np.clip(B2, VRA_MIN, VRA_MAX))


# ── Core computation ───────────────────────────────────────────────────────────

def _compute_all_factor_returns(
    returns_wide: pd.DataFrame,
    betas_wide: pd.DataFrame,
    isin_sector: dict,
    factor_snapshots: dict,
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

    log.info("Running regressions (%d trading days × %d factors)...", len(trading_days), K)
    for idx, td in enumerate(trading_days):
        td_str = td.strftime("%Y-%m-%d")

        latest_snap = next(
            (d for d in reversed(snap_keys) if d <= td_str), None
        )
        if latest_snap is None:
            continue

        fsnap     = factor_snapshots[latest_snap]
        day_rets  = returns_wide.loc[td].dropna()
        isins_day = [i for i in day_rets.index if i in isin_sector and i in fsnap]

        if len(isins_day) < MIN_STOCKS:
            continue

        r_day = day_rets[isins_day].values
        beta_row = betas_wide.loc[td] if td in betas_wide.index else pd.Series(dtype=float)
        beta_map = beta_row.dropna().to_dict()

        X_day, w_day = _build_day_exposure(isins_day, isin_sector, fsnap, beta_map)
        f_t, eps_t   = _wls_regression(r_day, X_day, w_day)

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

    # ── Factor covariance ────────────────────────────────────────────────────
    # _ewma_nw_cov returns daily-unit covariance; annualise (×252) so that
    # ||L_barra.T @ w||² matches risk.db convention (annual portfolio variance)
    # and VRA comparisons use consistent daily ↔ annual scaling.
    F_cov = _ewma_nw_cov(F_hist, HL_FACTOR_COV, NW_LAGS) * 252.0

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

    # ── VRA ──────────────────────────────────────────────────────────────────
    isins_snap = [i for i in fsnap_now if i in isin_sector]
    B2 = _vra(returns_pit, F_cov, delta, returns_pit.index[-1] if not returns_pit.empty else snap_ts,
               fsnap_now, isin_sector, beta_map_now)
    if abs(B2 - 1.0) > 0.01:
        log.info("VRA B²=%.3f → scaling covariance", B2)
    F_cov = B2 * F_cov
    delta  = {isin: B2 * v for isin, v in delta.items()}

    # ── Spectral floor ───────────────────────────────────────────────────────
    F_cov = _spectral_floor(F_cov)

    # ── Exposure matrix at snapshot date ────────────────────────────────────
    X_now, _ = _build_day_exposure(isins_snap, isin_sector, fsnap_now, beta_map_now)

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
        "Saved snapshot %s: F(%dx%d), δ(%d stocks), X(%dx%d), VRA=%.3f",
        snap_date_str, K, K, len(delta), len(isins_snap), K, B2,
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

    log.info("Loading factor snapshots...")
    factor_snapshots = _load_factor_snapshots()
    log.info("  %d snapshots, %d stock-snapshot rows",
             len(factor_snapshots), sum(len(v) for v in factor_snapshots.values()))

    log.info("Loading returns...")
    returns_wide = _load_returns_wide()
    log.info("  %d trading days, %d stocks", len(returns_wide), returns_wide.shape[1])
    log.info("  Period: %s → %s", returns_wide.index[0].date(), returns_wide.index[-1].date())

    log.info("Computing rolling 60-day betas...")
    betas_wide = _compute_beta_60d(returns_wide)

    log.info("Computing daily factor returns...")
    f_df, eps_sq_df = _compute_all_factor_returns(
        returns_wide, betas_wide, isin_sector, factor_snapshots
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
