"""
optimize_portfolio.py — Mean-variance portfolio optimizer.

Reads strategy parameters from data/strategy_params.xlsx, runs CVXPY
optimization, and saves results to data/portfolio_output/.

Usage:
    python optimize_portfolio.py                         # all active strategies
    python optimize_portfolio.py --strategy core_active  # single strategy
    python optimize_portfolio.py --list                  # list available strategies

Objectives
----------
maximize_alpha      Benchmark-aware: maximize alpha subject to tracking-error and
                    active-weight constraints. Requires benchmark_file.

maximize_sharpe     Absolute return: maximize Sharpe via Charnes-Cooper transform.
                    No benchmark required.

minimize_variance   Pure risk minimisation: no alpha signal, just minimize w'Σw.
                    Good for capital preservation / low-vol mandates.

Risk model today:  Ledoit-Wolf sample covariance (risk.db)
Risk model future: swap load_covariance() for Barra BFB'+D — optimizer unchanged
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# MOSEK bootstrap — ensure Python 3.13 can load MOSEK 10.x (which ships .so
# files only for 3.7–3.12).  We symlink the 3.12 extension as 3.13 once.
# Harmless when Python ≤3.12 or when MOSEK 11+ (with native 3.13 wheel) is
# installed.
# ---------------------------------------------------------------------------
def _ensure_mosek_symlink() -> None:
    try:
        import mosek  # noqa: F401 — already works
        return
    except ImportError:
        pass
    ver = sys.version_info
    if ver < (3, 13):
        return
    try:
        import importlib.util
        spec = importlib.util.find_spec("mosek")
        if spec is None:
            return
        pkg_dir = Path(spec.submodule_search_locations[0])
        src = pkg_dir / f"_msk.cpython-312-darwin.so"
        dst = pkg_dir / f"_msk.cpython-{ver.major}{ver.minor}-darwin.so"
        if src.exists() and not dst.exists():
            dst.symlink_to(src)
    except Exception:
        pass

_ensure_mosek_symlink()

from config import (
    OUTPUT_DIR, PARAMS_FILE, UNIVERSE_DB, MODELS_DB, RISK_DB, BENCHMARK_DIR,
)
from utils import get_db, get_logger

log = get_logger("optimize_portfolio")


# ── Excel params ──────────────────────────────────────────────────────────────

def load_strategy_params(strategy_id: str | None = None) -> list[dict]:
    xl = pd.ExcelFile(PARAMS_FILE)

    strats = pd.read_excel(xl, sheet_name="Strategies", dtype=str)
    strats = strats[strats["active"].str.strip().str.upper() == "TRUE"]
    if strategy_id:
        strats = strats[strats["strategy_id"].str.strip() == strategy_id]
    if strats.empty:
        raise ValueError(f"No active strategy found: {strategy_id!r}")

    cons_df  = pd.read_excel(xl, sheet_name="Constraints",   dtype=str)
    alpha_df = pd.read_excel(xl, sheet_name="Alpha_Weights", dtype=str)

    result = []
    for _, row in strats.iterrows():
        sid = row["strategy_id"].strip()

        c_rows = cons_df[cons_df["strategy_id"].str.strip() == sid]
        constraints = {}
        for _, c in c_rows.iterrows():
            if str(c["enabled"]).strip().upper() != "TRUE":
                continue
            name = c["constraint"].strip()
            val  = str(c["value"]).strip()
            try:
                constraints[name] = float(val)
            except ValueError:
                upper = val.upper()
                if upper in ("TRUE", "FALSE"):
                    constraints[name] = upper == "TRUE"
                else:
                    constraints[name] = val   # keep as string (e.g. excluded_sectors)

        a_rows = alpha_df[alpha_df["strategy_id"].str.strip() == sid]
        alpha_weights = {}
        for _, a in a_rows.iterrows():
            alpha_weights[a["model_id"].strip()] = float(a["weight"])

        benchmark_file = str(row.get("benchmark_file", "") or "").strip()
        objective      = str(row.get("objective",       "") or "maximize_alpha").strip()
        use_barra_raw  = str(row.get("use_barra_risk",  "TRUE") or "TRUE").strip().upper()
        use_barra      = use_barra_raw != "FALSE"   # default True unless explicitly FALSE

        result.append({
            "strategy_id":         sid,
            "name":                row["name"].strip(),
            "benchmark_file":      benchmark_file,
            "alpha_date":          row["alpha_date"].strip(),
            "risk_date":           row["risk_date"].strip(),
            "solver":              row["solver"].strip(),
            "objective":           objective,
            "investable_universe": row["investable_universe"].strip(),
            "constraints":         constraints,
            "alpha_weights":       alpha_weights,
            "use_barra_risk":      use_barra,
        })
    return result


# ── Data loaders ─────────────────────────────────────────────────────────────

def load_benchmark(benchmark_file: str) -> pd.DataFrame:
    path = BENCHMARK_DIR / benchmark_file
    df = pd.read_csv(path, skiprows=2)
    df = df.dropna(subset=["Ticker"])
    df = df[df["Asset Class"].str.strip() == "Equity"]
    df["weight"] = pd.to_numeric(df["Weight (%)"], errors="coerce") / 100.0
    df = df.dropna(subset=["weight"])
    df = df.rename(columns={"Ticker": "ticker", "Sector": "benchmark_sector"})
    df["weight"] /= df["weight"].sum()
    return df[["ticker", "benchmark_sector", "weight"]].copy()


def load_universe_metadata() -> pd.DataFrame:
    with get_db(UNIVERSE_DB) as conn:
        df = pd.read_sql(
            "SELECT isin, ticker, company_name, gics_sector, simfin_industry "
            "FROM companies WHERE isin IS NOT NULL",
            conn,
        )
    return df


def map_tickers_to_isins(tickers: list[str]) -> dict[str, str]:
    placeholders = ",".join("?" * len(tickers))
    with get_db(UNIVERSE_DB) as conn:
        rows = conn.execute(
            f"SELECT ticker, isin FROM companies WHERE ticker IN ({placeholders})", tickers,
        ).fetchall()
    return {r[0]: r[1] for r in rows}


def load_alpha_scores(model_id: str, date: str) -> dict[str, float]:
    with get_db(MODELS_DB) as conn:
        rows = conn.execute(
            "SELECT security_id, model_value_z FROM models WHERE model_id=? AND data_date=?",
            (model_id, date),
        ).fetchall()
    return {r[0]: r[1] for r in rows if r[1] is not None}


def load_blended_alpha(alpha_weights: dict[str, float], date: str) -> dict[str, float]:
    """Load and blend alpha z-scores from one or more models."""
    if not alpha_weights:
        return {}
    total_w = sum(alpha_weights.values())
    blended: dict[str, float] = {}
    for model_id, w in alpha_weights.items():
        scores = load_alpha_scores(model_id, date)
        norm_w = w / total_w
        for isin, z in scores.items():
            blended[isin] = blended.get(isin, 0.0) + z * norm_w
    return blended


def load_covariance(risk_date: str) -> tuple[np.ndarray, list[str]]:
    import io, zlib
    with get_db(RISK_DB) as conn:
        row = conn.execute(
            "SELECT matrix_blob, isin_list FROM covariance_matrix WHERE data_date=?",
            (risk_date,),
        ).fetchone()
    if row is None:
        raise ValueError(f"No covariance matrix for {risk_date}. Run create_risk.py first.")
    cov   = np.load(io.BytesIO(zlib.decompress(row[0]))).astype(np.float64)
    isins = json.loads(row[1])
    return cov, isins


def _latest_barra_date() -> str | None:
    """Return the most recent snapshot_date in risk.db (Barra tables), or None if unavailable."""
    try:
        with get_db(RISK_DB) as conn:
            row = conn.execute(
                "SELECT MAX(snapshot_date) FROM factor_covariance"
            ).fetchone()
        return row[0] if row and row[0] else None
    except Exception:
        return None


def load_barra_L(barra_date: str, investable: list[str]) -> np.ndarray | None:
    """
    Build stacked-L matrix for the Barra risk model.

    Returns L_barra of shape (N, K+N) such that
        ||L_barra.T @ w||² = w' (X F X' + Δ) w = w' Σ_barra w.
    Drop-in replacement for Cholesky L in all cp.norm(L.T @ w, 2) expressions.
    Returns None if risk.db has no Barra snapshot for barra_date.
    """
    import zlib as _zlib
    try:
        with get_db(RISK_DB) as conn:
            row_fc = conn.execute(
                "SELECT factor_names, cov_blob FROM factor_covariance WHERE snapshot_date=?",
                (barra_date,),
            ).fetchone()
            if row_fc is None:
                return None

            factor_names = json.loads(row_fc[0])
            Kf = len(factor_names)
            F = np.frombuffer(_zlib.decompress(row_fc[1]), dtype=np.float32) \
                  .reshape(Kf, Kf).astype(np.float64)

            N          = len(investable)
            isin_idx   = {isin: i for i, isin in enumerate(investable)}
            factor_idx = {f: j for j, f in enumerate(factor_names)}

            ph = ",".join("?" * N)
            # Factor exposures
            X      = np.zeros((N, Kf))
            rows_x = conn.execute(
                f"SELECT security_id, factor_id, exposure FROM factor_exposures "
                f"WHERE snapshot_date=? AND security_id IN ({ph})",
                [barra_date] + list(investable),
            ).fetchall()
            for sec_id, fac_id, exp_val in rows_x:
                if sec_id in isin_idx and fac_id in factor_idx:
                    X[isin_idx[sec_id], factor_idx[fac_id]] = float(exp_val)

            # Idiosyncratic variances (annualised); default ≈ 20% annual vol if missing
            delta  = np.full(N, 0.04)
            rows_d = conn.execute(
                f"SELECT security_id, idio_var FROM idiosyncratic_vars "
                f"WHERE snapshot_date=? AND security_id IN ({ph})",
                [barra_date] + list(investable),
            ).fetchall()
            for sec_id, idio_var in rows_d:
                if sec_id in isin_idx:
                    delta[isin_idx[sec_id]] = float(idio_var)

        # Cholesky of factor covariance (apply spectral floor if needed)
        try:
            L_F = np.linalg.cholesky(F)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(F)
            F = eigvecs @ np.diag(np.maximum(eigvals, 1e-6)) @ eigvecs.T
            L_F = np.linalg.cholesky(F)

        A = L_F.T @ X.T                              # (K, N)
        B = np.diag(np.sqrt(np.maximum(delta, 0)))   # (N, N)
        return np.vstack([A, B]).T                   # (N, K+N)

    except Exception as exc:
        log.warning("Barra load failed (%s); falling back to Ledoit-Wolf.", exc)
        return None


def _variance(w: np.ndarray, Sigma, L: np.ndarray) -> float:
    """
    Portfolio variance w' Σ w.
    When Sigma is None (Barra mode) uses ||L.T @ w||² which equals w' Σ_barra w.
    """
    if Sigma is not None:
        return float(w @ Sigma @ w)
    return float(np.sum((L.T @ w) ** 2))


# ── Shared setup helpers ──────────────────────────────────────────────────────

def _build_investable(strategy, bm_df, risk_isins, risk_isin_idx):
    if strategy["investable_universe"] == "benchmark_only":
        return sorted({i for i in bm_df["isin"] if i in risk_isin_idx})
    return sorted(set(risk_isins))


def _sector_industry_matrices(investable, gics_df):
    def _sector(isin):
        try:
            v = gics_df.loc[isin, "gics_sector"]
            return v if v else "Unknown"
        except KeyError:
            return "Unknown"

    def _industry(isin):
        try:
            v = gics_df.loc[isin, "simfin_industry"]
            return v if v else "Unknown"
        except KeyError:
            return "Unknown"

    sectors    = sorted({_sector(i)   for i in investable})
    industries = sorted({_industry(i) for i in investable})

    B_sector = np.zeros((len(sectors),    len(investable)))
    B_ind    = np.zeros((len(industries), len(investable)))
    for j, isin in enumerate(investable):
        B_sector[sectors.index(_sector(isin)), j]       = 1.0
        B_ind[industries.index(_industry(isin)), j]     = 1.0

    return sectors, industries, B_sector, B_ind, _sector, _industry


def _lp_prescreen(
    strategy: dict,
    investable: list[str],
    alpha: np.ndarray,
    b: np.ndarray,
    Sigma: np.ndarray | None,
    L: np.ndarray,
    sectors: list,
    industries: list,
    B_sector: np.ndarray,
    B_ind: np.ndarray,
    n_keep: int,
) -> list[int] | None:
    """
    Solve the LP relaxation (no integer constraints) and return sorted indices of
    the top n_keep stocks by weight. Used to build a small, constraint-aware
    candidate set before solving the full MIP.
    """
    c_relax = {k: v for k, v in strategy["constraints"].items()
               if k not in ("max_positions", "min_position_if_held")}
    s_relax = {**strategy, "constraints": c_relax, "solver": "CLARABEL"}
    try:
        lp_weights, _ = _optimize_alpha(
            s_relax, investable, alpha, b, Sigma, L,
            sectors, industries, B_sector, B_ind)
        keep_idx = np.argsort(-lp_weights)[:n_keep]
        return sorted(keep_idx.tolist())
    except Exception:
        return None


def _covariance_submatrix(risk_cov, risk_isin_idx, investable):
    idx   = [risk_isin_idx[isin] for isin in investable]
    Sigma = risk_cov[np.ix_(idx, idx)]
    Sigma = (Sigma + Sigma.T) / 2.0
    eigmin = np.linalg.eigvalsh(Sigma).min()
    if eigmin < 1e-8:
        Sigma += (abs(eigmin) + 1e-8) * np.eye(len(investable))
    return Sigma, np.linalg.cholesky(Sigma)


# ── Sector constraint helper (w-space and y-space) ───────────────────────────

def _add_sector_constraints(cvx, var, scale, sectors, B_sector, c):
    """
    Apply sector constraints to CVXPY variable `var`.

    w-space (maximize_alpha / minimize_variance): var=w, scale=1.0
    y-space (maximize_sharpe):                   var=y, scale=cp.sum(y)

    Reads from constraints dict:
        excluded_sectors      pipe-separated sector names to zero out, e.g. "Energy|Materials"
        equal_sector_weight   bool — each sector gets 1/n_active ± sector_weight_tolerance
        sector_weight_tolerance  float tolerance around equal weight target
        min_sector_weight     float floor on each non-excluded sector
        max_sector_weight     float cap on each non-excluded sector
    """
    excluded_raw = c.get("excluded_sectors", "") or ""
    excluded = {s.strip() for s in str(excluded_raw).split("|") if s.strip()}
    active   = [s for s in sectors if s not in excluded]
    n_active = max(len(active), 1)

    equal_sw = c.get("equal_sector_weight", False)
    eq_tol   = c.get("sector_weight_tolerance", 0.0)
    min_sw   = c.get("min_sector_weight", None)
    max_sw   = c.get("max_sector_weight", None)

    for s_idx, s_name in enumerate(sectors):
        sw = B_sector[s_idx] @ var
        if s_name in excluded:
            cvx.append(sw == 0)
        elif equal_sw:
            eq_w = 1.0 / n_active
            cvx.append(sw <= (eq_w + eq_tol) * scale)
            cvx.append(sw >= max(0.0, eq_w - eq_tol) * scale)
        else:
            if min_sw is not None:
                cvx.append(sw >= min_sw * scale)
            if max_sw is not None:
                cvx.append(sw <= max_sw * scale)


# ── Integer constraint helpers ────────────────────────────────────────────────

# Minimum assumed portfolio vol; used as a denominator for big-M bounds in the
# Charnes-Cooper (maximize_sharpe) formulation where the decision variable is
# y = w/σ_p rather than w directly.
_SIGMA_FLOOR = 0.05


def _has_integer_constraints(c: dict) -> bool:
    return c.get("max_positions") is not None or c.get("min_position_if_held") is not None


def _ensure_mosek(strategy: dict) -> None:
    """Auto-promote solver to MOSEK when integer constraints are present."""
    if _has_integer_constraints(strategy["constraints"]):
        if strategy.get("solver", "CLARABEL").upper() != "MOSEK":
            log.info("Integer constraints detected — switching solver to MOSEK (was %s)", strategy["solver"])
            strategy["solver"] = "MOSEK"


# ── Objective 1: maximize_alpha (benchmark-aware) ─────────────────────────────

def _optimize_alpha(strategy, investable, alpha, b, Sigma, L,
                    sectors, industries, B_sector, B_ind,
                    prev_weights_arr: np.ndarray | None = None,
                    max_turnover: float | None = None):
    c = strategy["constraints"]
    N = len(investable)

    w  = cp.Variable(N, name="weights")
    aw = w - b

    cvx = [cp.sum(w) == 1.0, w >= 0]

    # ── Integer: cardinality + minimum position ────────────────────────────
    max_pos_n = c.get("max_positions")
    min_held  = c.get("min_position_if_held")
    z = None
    if max_pos_n is not None or min_held is not None:
        z = cp.Variable(N, boolean=True)
        cvx.append(cp.sum(z) <= int(max_pos_n or N))
        cvx.append(w <= z)                      # w_i > 0 requires z_i = 1
        if min_held is not None:
            cvx.append(w >= min_held * z)       # if held, at least min_held

    # Active-weight constraints
    max_saw = c.get("max_stock_active_weight", 0.02)
    cvx += [aw <= max_saw, aw >= -max_saw]

    max_ar = c.get("max_active_risk", 0.04)
    cvx.append(cp.norm(L.T @ aw, 2) <= max_ar)

    max_secaw = c.get("max_sector_active_weight", None)
    if max_secaw is not None:
        for s in range(len(sectors)):
            s_aw = B_sector[s] @ aw
            cvx += [s_aw <= max_secaw, s_aw >= -max_secaw]

    max_indaw = c.get("max_industry_active_weight", None)
    if max_indaw is not None:
        for g in range(len(industries)):
            g_aw = B_ind[g] @ aw
            cvx += [g_aw <= max_indaw, g_aw >= -max_indaw]

    # Absolute sector constraints (excluded_sectors, min/max_sector_weight)
    _add_sector_constraints(cvx, w, 1.0, sectors, B_sector, c)

    # One-way turnover constraint: sum(|w - w_prev|) / 2 ≤ max_turnover
    if prev_weights_arr is not None and max_turnover is not None:
        cvx.append(cp.sum(cp.abs(w - prev_weights_arr)) / 2 <= max_turnover)

    prob = cp.Problem(cp.Maximize(alpha @ w), cvx)
    _solve(prob, strategy["solver"])

    weights = np.array(w.value).clip(0)
    if z is not None:
        weights *= (np.array(z.value) > 0.5).astype(float)
    weights /= weights.sum()

    act         = weights - b
    active_risk = float(np.sqrt(_variance(act, Sigma, L)))
    exp_alpha   = float(alpha @ weights)
    n_pos       = int((weights > 1e-4).sum())
    info_ratio  = exp_alpha / active_risk if active_risk > 0 else 0.0

    log.info("Expected alpha: %+.4f | Active risk: %.2f%% | Info ratio: %.2f | Positions: %d",
             exp_alpha, active_risk * 100, info_ratio, n_pos)

    return weights, {
        "expected_alpha": round(exp_alpha, 4),
        "active_risk":    round(active_risk, 4),
        "info_ratio":     round(info_ratio, 4),
    }


# ── Objective 2: maximize_sharpe (Charnes-Cooper) ────────────────────────────

def _optimize_sharpe(strategy, investable, alpha, Sigma, L,
                     sectors, industries, B_sector, B_ind,
                     prev_weights_arr: np.ndarray | None = None,
                     max_turnover: float | None = None):
    """
    Charnes-Cooper: let y = w/σ_p.
    Maximize alpha @ y  s.t.  ||L.T @ y||₂ ≤ 1, y ≥ 0, per-stock/sector bounds.
    Recover weights:  w = y / sum(y),  σ_p = ||L.T @ w||₂.

    For MIP (cardinality / min position): binary z_i added with big-M bounds.
    The zero-out big-M is max_position / _SIGMA_FLOOR (tight bound on y_i).
    Minimum position uses a linear big-M relaxation of the bilinear term y_i ≥ p*t*z_i.
    """
    c = strategy["constraints"]
    N = len(investable)

    y = cp.Variable(N, name="scaled_weights", nonneg=True)
    t = cp.sum(y)   # proportional to 1/σ_p

    cvx = [cp.norm(L.T @ y, 2) <= 1]

    max_pos = c.get("max_position", 0.05)
    cvx.append(y <= max_pos * t)

    # ── Integer: cardinality + minimum position ────────────────────────────
    max_pos_n = c.get("max_positions")
    min_held  = c.get("min_position_if_held")
    z = None
    if max_pos_n is not None or min_held is not None:
        z = cp.Variable(N, boolean=True)
        big_M = max_pos / _SIGMA_FLOOR          # upper bound on y_i when held
        cvx.append(cp.sum(z) <= int(max_pos_n or N))
        cvx.append(y <= big_M * z)              # y_i > 0 requires z_i = 1
        if min_held is not None:
            # y_i >= min_held * t * z_i  linearised via big-M on (1-z_i):
            #   y_i >= min_held * t - (min_held/σ_floor) * (1 - z_i)
            M_lo = min_held / _SIGMA_FLOOR
            cvx.append(y >= min_held * t - M_lo * (1 - z))

    # Sector constraints (y-space: scale by t)
    _add_sector_constraints(cvx, y, t, sectors, B_sector, c)

    # Industry cap
    max_ind = c.get("max_industry_weight", None)
    if max_ind is not None:
        for g in range(len(industries)):
            cvx.append(B_ind[g] @ y <= max_ind * t)

    # Max portfolio vol: σ_p ≤ v  →  ||L.T @ y||₂ ≤ v * t
    max_vol = c.get("max_portfolio_vol", None)
    if max_vol is not None:
        cvx.append(cp.norm(L.T @ y, 2) <= max_vol * t)

    # One-way turnover in Charnes-Cooper space:
    # sum(|w - w_prev|)/2 ≤ T  ⟺  sum(|y - w_prev·t|) ≤ 2·T·t
    if prev_weights_arr is not None and max_turnover is not None:
        cvx.append(cp.sum(cp.abs(y - prev_weights_arr * t)) <= 2 * max_turnover * t)

    prob = cp.Problem(cp.Maximize(alpha @ y), cvx)
    _solve(prob, strategy["solver"])

    y_val = np.array(y.value).clip(0)
    if z is not None:
        y_val *= (np.array(z.value) > 0.5).astype(float)
    weights = y_val / y_val.sum()

    port_vol   = float(np.sqrt(_variance(weights, Sigma, L)))
    exp_return = float(alpha @ weights)
    sharpe     = exp_return / port_vol if port_vol > 0 else 0.0
    n_pos      = int((weights > 1e-4).sum())

    log.info("Expected return: %+.4f | Portfolio vol: %.2f%% | Sharpe: %.2f | Positions: %d",
             exp_return, port_vol * 100, sharpe, n_pos)

    return weights, {
        "expected_alpha": round(exp_return, 4),
        "portfolio_vol":  round(port_vol, 4),
        "active_risk":    round(port_vol, 4),
        "sharpe_ratio":   round(sharpe, 4),
        "info_ratio":     round(sharpe, 4),
    }


# ── Objective 3: minimize_variance ───────────────────────────────────────────

def _optimize_min_variance(strategy, investable, b, Sigma, L,
                           sectors, industries, B_sector, B_ind,
                           prev_weights_arr: np.ndarray | None = None,
                           max_turnover: float | None = None):
    c = strategy["constraints"]
    N = len(investable)

    w = cp.Variable(N, name="weights")

    cvx = [cp.sum(w) == 1.0, w >= 0]

    max_pos = c.get("max_position", 0.05)

    # ── Integer: cardinality + minimum position ────────────────────────────
    max_pos_n = c.get("max_positions")
    min_held  = c.get("min_position_if_held")
    z = None
    if max_pos_n is not None or min_held is not None:
        z = cp.Variable(N, boolean=True)
        cvx.append(cp.sum(z) <= int(max_pos_n or N))
        cvx.append(w <= max_pos * z)            # zero-out + upper bound combined
        if min_held is not None:
            cvx.append(w >= min_held * z)
    else:
        cvx.append(w <= max_pos)

    # Sector constraints (w-space)
    _add_sector_constraints(cvx, w, 1.0, sectors, B_sector, c)

    # Industry cap
    max_ind = c.get("max_industry_weight", None)
    if max_ind is not None:
        for g in range(len(industries)):
            cvx.append(B_ind[g] @ w <= max_ind)

    # One-way turnover constraint: sum(|w - w_prev|) / 2 ≤ max_turnover
    if prev_weights_arr is not None and max_turnover is not None:
        cvx.append(cp.sum(cp.abs(w - prev_weights_arr)) / 2 <= max_turnover)

    prob = cp.Problem(cp.Minimize(cp.sum_squares(L.T @ w)), cvx)
    _solve(prob, strategy["solver"])

    weights = np.array(w.value).clip(0)
    if z is not None:
        weights *= (np.array(z.value) > 0.5).astype(float)
    weights /= weights.sum()

    port_vol = float(np.sqrt(_variance(weights, Sigma, L)))
    n_pos    = int((weights > 1e-4).sum())

    log.info("Portfolio vol: %.2f%% | Positions: %d", port_vol * 100, n_pos)

    return weights, {
        "expected_alpha": 0.0,
        "portfolio_vol":  round(port_vol, 4),
        "active_risk":    round(port_vol, 4),
        "sharpe_ratio":   0.0,
        "info_ratio":     0.0,
    }


# ── Solver ────────────────────────────────────────────────────────────────────

_MOSEK_PARAMS = {
    "MSK_IPAR_NUM_THREADS":             4,
    "MSK_DPAR_INTPNT_CO_TOL_PFEAS":     1e-8,
    "MSK_DPAR_INTPNT_CO_TOL_DFEAS":     1e-8,
    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP":   1e-8,
}

# Extra params applied only when the problem contains integer variables.
# Allow MOSEK up to 300 s of branch-and-bound; it will return the best
# incumbent if time expires (status = optimal_inaccurate → still accepted).
_MOSEK_PARAMS_MIP = {
    **_MOSEK_PARAMS,
    "MSK_DPAR_OPTIMIZER_MAX_TIME":      300.0,
    "MSK_DPAR_MIO_MAX_TIME":            300.0,
    "MSK_DPAR_MIO_TOL_REL_GAP":         1e-3,   # 0.1% optimality gap is fine in practice
}


def _solve(prob, solver_name: str):
    solver_map = {
        "ECOS":    cp.ECOS,
        "SCS":     cp.SCS,
        "CLARABEL": cp.CLARABEL,
        "MOSEK":   cp.MOSEK,
    }
    key = solver_name.upper()
    solver = solver_map.get(key, cp.CLARABEL)
    n_vars = sum(v.size for v in prob.variables())
    log.info("Solving (%s, vars=%d) ...", solver_name, n_vars)

    is_mip = any(v.attributes.get("boolean") for v in prob.variables())
    kwargs: dict = {"verbose": False}
    if solver is cp.MOSEK:
        kwargs["mosek_params"] = _MOSEK_PARAMS_MIP if is_mip else _MOSEK_PARAMS

    prob.solve(solver=solver, **kwargs)
    val = f"  obj={prob.value:.4f}" if prob.value is not None else ""
    log.info("status=%s%s", prob.status, val)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Optimization failed: {prob.status}")


# ── Backtest entry point ──────────────────────────────────────────────────────

def optimize_for_backtest(
    alpha_weights: dict[str, float],
    objective: str,
    constraints: dict,
    alpha_date: str,
    barra_date: str | None,
    risk_date: str,
    sp500_isins: list[str],
    bm_weights: dict[str, float],
    prev_weights: dict[str, float] | None,
    max_turnover: float,
    solver: str = "CLARABEL",
    min_weight: float = 0.0,
) -> tuple[dict[str, float], dict] | None:
    """
    Single-period optimizer for walk-forward backtest. No file I/O or console output.

    Parameters
    ----------
    alpha_weights : model_id → blend weight (from Alpha_Weights sheet)
    objective     : "maximize_alpha" | "maximize_sharpe" | "minimize_variance"
    constraints   : dict of constraint name → value (from Constraints sheet)
    alpha_date    : model snapshot date used to load alpha scores
    barra_date    : Barra snapshot to use (None → Ledoit-Wolf only)
    risk_date     : Ledoit-Wolf snapshot date (fallback / universe alignment)
    sp500_isins   : candidate universe for this period
    bm_weights    : {isin: weight} for benchmark vector (maximize_alpha only);
                    pass {} for equal-weight fallback
    prev_weights  : {isin: weight} from previous period (None → first period,
                    no turnover constraint applied)
    max_turnover  : one-way turnover fraction, e.g. 0.10 for 10%
    solver        : CVXPY solver name
    min_weight    : minimum position weight; positions below this are zeroed and
                    the portfolio is renormalised (post-processing, no MIP required)

    Returns
    -------
    ({isin: weight}, metrics_dict) or None on failure/infeasibility.
    """
    import contextlib
    import io as _io

    try:
        # Ledoit-Wolf covariance used for dimension alignment and fallback risk
        risk_cov, risk_isins = load_covariance(risk_date)
        risk_isin_idx = {isin: i for i, isin in enumerate(risk_isins)}

        # Investable = S&P 500 ∩ LW risk model; require at least 50 stocks
        investable = sorted(set(sp500_isins) & set(risk_isins))
        if len(investable) < 50:
            return None

        # Alpha scores
        alpha_lookup = load_blended_alpha(alpha_weights, alpha_date)
        alpha_arr    = np.array([alpha_lookup.get(isin, 0.0) for isin in investable])

        # Sector metadata for sector constraints
        meta_df = load_universe_metadata()
        gics_df = meta_df.set_index("isin")

        # Risk model: Barra (preferred) → Ledoit-Wolf
        Sigma, L = _covariance_submatrix(risk_cov, risk_isin_idx, investable)
        used_barra = False
        if barra_date is not None:
            buf = _io.StringIO()
            with contextlib.redirect_stdout(buf):
                L_barra = load_barra_L(barra_date, investable)
            if L_barra is not None:
                L          = L_barra
                Sigma      = None
                used_barra = True

        N = len(investable)

        # Benchmark vector (only used in maximize_alpha objective)
        if bm_weights:
            b_raw = np.array([bm_weights.get(isin, 0.0) for isin in investable])
            b_sum = b_raw.sum()
            b = b_raw / b_sum if b_sum > 1e-10 else np.full(N, 1.0 / N)
        else:
            b = np.full(N, 1.0 / N)  # equal-weight fallback

        # Previous-period weights aligned to current investable universe
        prev_w_arr: np.ndarray | None = None
        if prev_weights is not None:
            raw   = np.array([prev_weights.get(isin, 0.0) for isin in investable])
            total = raw.sum()
            if total > 1e-10:
                prev_w_arr = raw / total

        sectors, industries, B_sector, B_ind, _sector_fn, _industry_fn = \
            _sector_industry_matrices(investable, gics_df)

        # Integer constraints (max_positions, min_position_if_held) require MOSEK and are
        # operational construction constraints, not alpha/risk constraints.  In the backtest
        # context the strategy is often evaluated against a different universe than it was
        # designed for (e.g. Core Active vs S&P 500 instead of Russell 1000), which makes the
        # MIP geometrically infeasible.  Strip them here; all continuous constraints apply.
        _INTEGER_CONSTRAINT_KEYS = ("max_positions", "min_position_if_held")
        relaxed_integer = any(constraints.get(k) is not None for k in _INTEGER_CONSTRAINT_KEYS)
        effective_constraints = {k: v for k, v in constraints.items()
                                 if k not in _INTEGER_CONSTRAINT_KEYS}

        strategy = {
            "constraints": effective_constraints,
            "solver":      solver,
            "objective":   objective,
        }

        # Run optimizer with stdout suppressed (internal prints not relevant in backtest)
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            if objective == "maximize_sharpe":
                weights, extra = _optimize_sharpe(
                    strategy, investable, alpha_arr, Sigma, L,
                    sectors, industries, B_sector, B_ind,
                    prev_weights_arr=prev_w_arr, max_turnover=max_turnover,
                )
            elif objective == "minimize_variance":
                weights, extra = _optimize_min_variance(
                    strategy, investable, b, Sigma, L,
                    sectors, industries, B_sector, B_ind,
                    prev_weights_arr=prev_w_arr, max_turnover=max_turnover,
                )
            else:  # maximize_alpha
                weights, extra = _optimize_alpha(
                    strategy, investable, alpha_arr, b, Sigma, L,
                    sectors, industries, B_sector, B_ind,
                    prev_weights_arr=prev_w_arr, max_turnover=max_turnover,
                )

        result_weights = {
            isin: float(w) for isin, w in zip(investable, weights) if w > 1e-5
        }

        # Drop positions below operational minimum and renormalise
        if min_weight > 0.0:
            result_weights = {isin: w for isin, w in result_weights.items() if w >= min_weight}
            total = sum(result_weights.values())
            if total > 1e-10:
                result_weights = {isin: w / total for isin, w in result_weights.items()}

        extra.update({
            "used_barra":        used_barra,
            "relaxed_integer":   relaxed_integer,
            "n_positions":       len(result_weights),
            "alpha_date":        alpha_date,
            "barra_date":        barra_date,
            "risk_date":         risk_date,
        })
        return result_weights, extra

    except Exception as exc:
        # Caller decides how to handle (e.g. carry forward previous weights)
        log.warning("optimize_for_backtest failed (%s): %s", alpha_date, exc)
        return None


# ── Main orchestration ────────────────────────────────────────────────────────

def run_optimization(strategy: dict) -> tuple[pd.DataFrame, dict]:
    log.info("=== Strategy: %s  (%s)  Objective: %s ===",
             strategy["name"], strategy["strategy_id"], strategy["objective"])

    objective = strategy["objective"]

    log.info("Loading covariance ...")
    risk_cov, risk_isins = load_covariance(strategy["risk_date"])
    risk_isin_idx = {isin: i for i, isin in enumerate(risk_isins)}
    log.info("  %d stocks in risk model", len(risk_isins))

    # Benchmark — always load for display if benchmark_file is set;
    # only fed into the optimiser's b vector for maximize_alpha.
    if strategy["benchmark_file"]:
        log.info("Loading benchmark ...")
        bm_df = load_benchmark(strategy["benchmark_file"])
        ticker_to_isin = map_tickers_to_isins(list(bm_df["ticker"]))
        bm_df["isin"] = bm_df["ticker"].map(ticker_to_isin)
        bm_df = bm_df.dropna(subset=["isin"])
        bm_df["weight"] /= bm_df["weight"].sum()
        log.info("  %d benchmark stocks", len(bm_df))
    else:
        bm_df = pd.DataFrame(columns=["isin", "weight"])

    # Alpha scores (blended from Alpha_Weights sheet)
    log.info("Loading alpha scores ...")
    alpha_lookup = load_blended_alpha(strategy["alpha_weights"], strategy["alpha_date"])
    models_str   = ", ".join(f"{m}×{w:.2g}" for m, w in strategy["alpha_weights"].items())
    log.info("  %d scores  [%s]", len(alpha_lookup), models_str)

    meta_df = load_universe_metadata()
    gics_df = meta_df.set_index("isin")

    investable  = _build_investable(strategy, bm_df, risk_isins, risk_isin_idx)
    alpha_arr   = np.array([alpha_lookup.get(isin, 0.0) for isin in investable])

    c_pre       = strategy["constraints"]

    N           = len(investable)
    isin_to_pos = {isin: i for i, isin in enumerate(investable)}
    log.info("Investable universe: %d stocks", N)

    # ── Risk model: Barra (default) or Ledoit-Wolf fallback ──────────────────
    Sigma, L = _covariance_submatrix(risk_cov, risk_isin_idx, investable)

    use_barra  = strategy.get("use_barra_risk", True)
    barra_date = _latest_barra_date()
    used_barra_date: str | None = None
    if use_barra and barra_date:
        L_barra = load_barra_L(barra_date, investable)
        if L_barra is not None:
            L     = L_barra
            Sigma = None   # not needed — _variance() uses ||L.T @ w||² instead
            used_barra_date = barra_date
            log.info("Risk model: Barra (%s)  [%d factors+stocks]", barra_date, L.shape[1])
        else:
            log.info("Risk model: Ledoit-Wolf (Barra unavailable)")
    else:
        log.info("Risk model: Ledoit-Wolf")

    alpha    = alpha_arr

    # bm_display: benchmark weights for results CSV (all strategies)
    bm_lookup   = dict(zip(bm_df["isin"], bm_df["weight"])) if not bm_df.empty else {}
    bm_display  = np.array([bm_lookup.get(isin, 0.0) for isin in investable])
    if bm_display.sum() > 0:
        bm_display /= bm_display.sum()

    # b: benchmark vector fed into the optimiser (only for maximize_alpha)
    b = bm_display.copy() if objective == "maximize_alpha" else np.zeros(N)

    sectors, industries, B_sector, B_ind, _sector_fn, _industry_fn = \
        _sector_industry_matrices(investable, gics_df)

    # LP-guided pre-screen for MIP strategies.
    # Solve the continuous relaxation first (CLARABEL, fast), keep the top
    # max_positions stocks by LP weight as the MIP candidate set.
    # Using 1× (not 2×) because the LP-positive stocks (those the LP actually needs
    # to satisfy constraints) are all ranked above the zero-weight stocks, so a
    # 1× window includes all of them while keeping the MIP size small.
    max_pos_n = c_pre.get("max_positions")
    if (max_pos_n is not None and len(investable) > int(max_pos_n)
            and objective == "maximize_alpha"):
        n_cand = int(max_pos_n)
        log.info("LP pre-screen: relaxation → top %d candidates ...", n_cand)
        lp_idx = _lp_prescreen(
            strategy, investable, alpha, b, Sigma, L,
            sectors, industries, B_sector, B_ind, n_cand)
        if lp_idx is not None:
            lp_arr = np.array(lp_idx)
            # Save full-universe sector matrices before subsetting — needed to
            # correct sector benchmark weights after the pre-screen drops stocks.
            b_full_pre        = b.copy()
            B_sector_full_pre = B_sector.copy()
            sectors_full_pre  = list(sectors)

            investable  = [investable[i] for i in lp_idx]
            alpha       = alpha[lp_arr]
            b           = b[lp_arr]
            bm_display  = bm_display[lp_arr]
            Sigma, L    = _covariance_submatrix(risk_cov, risk_isin_idx, investable)
            if used_barra_date:
                L_barra = load_barra_L(used_barra_date, investable)
                if L_barra is not None:
                    L = L_barra
                    Sigma = None
            sectors, industries, B_sector, B_ind, _sector_fn, _industry_fn = \
                _sector_industry_matrices(investable, gics_df)

            # Dropped stocks still have benchmark weight, so the subsetted b
            # underestimates sector benchmark weights — the MIP sector constraint
            # would target the wrong level (e.g. Industrials at 4.6% instead of
            # 9.2%).  Rescale b within each sector so B_sector @ b equals the
            # full-benchmark sector weight.
            for s_idx, sector in enumerate(sectors):
                if sector not in sectors_full_pre:
                    continue
                s_full     = sectors_full_pre.index(sector)
                full_sec_w = float(B_sector_full_pre[s_full] @ b_full_pre)
                incl_sec_w = float(B_sector[s_idx] @ b)
                if incl_sec_w > 1e-10 and full_sec_w > 0:
                    b[B_sector[s_idx].astype(bool)] *= full_sec_w / incl_sec_w

            N = len(investable)
            log.info("LP pre-screen done (%d candidates)", N)
        else:
            log.info("LP pre-screen failed — using full universe")

    # Alpha-score pre-screen for maximize_sharpe / minimize_variance MIPs.
    # The LP pre-screen above is only for maximize_alpha (needs a benchmark LP).
    # For other objectives with cardinality constraints, the full-universe MIP
    # can be intractable (e.g. 998 binary vars). Pre-screen to top 5× max_positions
    # by alpha score — fast heuristic that keeps the MIP size manageable.
    if max_pos_n is not None and objective in ("maximize_sharpe", "minimize_variance"):
        n_alpha_cand = min(len(investable), int(max_pos_n) * 5)
        if n_alpha_cand < len(investable):
            top_idx = np.argsort(-alpha)[:n_alpha_cand]
            top_arr = np.array(sorted(top_idx.tolist()))
            investable  = [investable[i] for i in top_arr]
            alpha       = alpha[top_arr]
            b           = b[top_arr]
            bm_display  = bm_display[top_arr]
            Sigma, L    = _covariance_submatrix(risk_cov, risk_isin_idx, investable)
            if used_barra_date:
                L_barra = load_barra_L(used_barra_date, investable)
                if L_barra is not None:
                    L = L_barra
                    Sigma = None
            sectors, industries, B_sector, B_ind, _sector_fn, _industry_fn = \
                _sector_industry_matrices(investable, gics_df)
            N = len(investable)
            log.info("MIP pre-screen (alpha): top %d candidates (max_positions=%s)", N, max_pos_n)

    _ensure_mosek(strategy)

    # Dispatch to objective
    if objective == "maximize_sharpe":
        weights, extra = _optimize_sharpe(
            strategy, investable, alpha, Sigma, L,
            sectors, industries, B_sector, B_ind)
    elif objective == "minimize_variance":
        weights, extra = _optimize_min_variance(
            strategy, investable, b, Sigma, L,
            sectors, industries, B_sector, B_ind)
    else:
        weights, extra = _optimize_alpha(
            strategy, investable, alpha, b, Sigma, L,
            sectors, industries, B_sector, B_ind)

    # Build results DataFrame
    isin_to_ticker = dict(zip(meta_df["isin"], meta_df["ticker"]))
    isin_to_name   = dict(zip(meta_df["isin"], meta_df["company_name"]))

    rows = []
    for j, isin in enumerate(investable):
        rows.append({
            "isin":             isin,
            "ticker":           isin_to_ticker.get(isin, ""),
            "company_name":     isin_to_name.get(isin, ""),
            "gics_sector":      _sector_fn(isin),
            "industry":         _industry_fn(isin),
            "benchmark_weight": round(float(bm_display[j]), 6),
            "portfolio_weight": round(float(weights[j]), 6),
            "active_weight":    round(float(weights[j]) - float(bm_display[j]), 6),
            "alpha_score":      round(float(alpha[j]), 4),
        })
    results_df = pd.DataFrame(rows).sort_values("active_weight", ascending=False)

    summary = {
        "strategy_id": strategy["strategy_id"],
        "name":        strategy["name"],
        "objective":   objective,
        "status":      "optimal",
        "alpha_date":  strategy["alpha_date"],
        "risk_date":   strategy["risk_date"],
        "n_positions": int((weights > 1e-4).sum()),
        "n_benchmark": int((bm_display > 0).sum()),
        "run_date":    datetime.now().strftime("%Y-%m-%d %H:%M"),
        **extra,
    }
    if used_barra_date:
        summary["barra_date"] = used_barra_date
    return results_df, summary


# ── Output ────────────────────────────────────────────────────────────────────

def save_results(results_df: pd.DataFrame, summary: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sid = summary["strategy_id"]

    latest = OUTPUT_DIR / f"{sid}_latest.csv"
    results_df.to_csv(latest, index=False)

    summary_path = OUTPUT_DIR / f"{sid}_latest_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("Saved → %s", latest)
    return latest


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Portfolio optimizer")
    parser.add_argument("--strategy", help="Strategy ID (default: all active)")
    parser.add_argument("--list",     action="store_true", help="List strategies")
    parser.add_argument("--solver",   help="Override solver for all strategies (e.g. MOSEK, CLARABEL, SCS)")
    args = parser.parse_args()

    if not PARAMS_FILE.exists():
        log.error("ERROR: %s not found. Run create_strategy_params.py first.", PARAMS_FILE)
        return

    if args.list:
        xl = pd.ExcelFile(PARAMS_FILE)
        df = pd.read_excel(xl, sheet_name="Strategies")
        # print() intentional here — tabular display for CLI listing only
        print(df[["strategy_id", "name", "active", "objective"]].to_string(index=False))
        return

    strategies = load_strategy_params(args.strategy)
    if args.solver:
        for s in strategies:
            s["solver"] = args.solver.upper()
    log.info("Running %d strategy/strategies ...", len(strategies))

    for s in strategies:
        try:
            results_df, summary = run_optimization(s)
            save_results(results_df, summary)
        except Exception as exc:
            log.error("ERROR in %s: %s", s["strategy_id"], exc)
            raise

    log.info("Done.")


if __name__ == "__main__":
    main()
