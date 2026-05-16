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
    OUTPUT_DIR, PARAMS_FILE, UNIVERSE_DB, MODELS_DB, RISK_DB, BARRA_DB, BENCHMARK_DIR,
)
from utils import get_db


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
    """Return the most recent snapshot_date in barra.db, or None if unavailable."""
    if not BARRA_DB.exists():
        return None
    try:
        with get_db(BARRA_DB) as conn:
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
    Returns None if barra.db has no snapshot for barra_date.
    """
    import zlib as _zlib
    try:
        with get_db(BARRA_DB) as conn:
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
        print(f"  [WARN] Barra load failed ({exc}); falling back to Ledoit-Wolf.")
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
            print(f"  [INFO] Integer constraints detected — switching solver to MOSEK "
                  f"(was {strategy['solver']})")
            strategy["solver"] = "MOSEK"


# ── Objective 1: maximize_alpha (benchmark-aware) ─────────────────────────────

def _optimize_alpha(strategy, investable, alpha, b, Sigma, L,
                    sectors, industries, B_sector, B_ind):
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

    print(f"\n  Expected alpha: {exp_alpha:+.4f}")
    print(f"  Active risk:    {active_risk:.2%}")
    print(f"  Info ratio:     {info_ratio:.2f}")
    print(f"  Positions:      {n_pos}")

    return weights, {
        "expected_alpha": round(exp_alpha, 4),
        "active_risk":    round(active_risk, 4),
        "info_ratio":     round(info_ratio, 4),
    }


# ── Objective 2: maximize_sharpe (Charnes-Cooper) ────────────────────────────

def _optimize_sharpe(strategy, investable, alpha, Sigma, L,
                     sectors, industries, B_sector, B_ind):
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

    print(f"\n  Expected return (alpha proxy): {exp_return:+.4f}")
    print(f"  Portfolio vol:  {port_vol:.2%}")
    print(f"  Sharpe ratio:   {sharpe:.2f}")
    print(f"  Positions:      {n_pos}")

    return weights, {
        "expected_alpha": round(exp_return, 4),
        "portfolio_vol":  round(port_vol, 4),
        "active_risk":    round(port_vol, 4),
        "sharpe_ratio":   round(sharpe, 4),
        "info_ratio":     round(sharpe, 4),
    }


# ── Objective 3: minimize_variance ───────────────────────────────────────────

def _optimize_min_variance(strategy, investable, b, Sigma, L,
                           sectors, industries, B_sector, B_ind):
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

    prob = cp.Problem(cp.Minimize(cp.sum_squares(L.T @ w)), cvx)
    _solve(prob, strategy["solver"])

    weights = np.array(w.value).clip(0)
    if z is not None:
        weights *= (np.array(z.value) > 0.5).astype(float)
    weights /= weights.sum()

    port_vol = float(np.sqrt(_variance(weights, Sigma, L)))
    n_pos    = int((weights > 1e-4).sum())

    print(f"\n  Portfolio vol:  {port_vol:.2%}")
    print(f"  Positions:      {n_pos}")

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
# Allow MOSEK up to 120 s of branch-and-bound; it will return the best
# incumbent if time expires (status = optimal_inaccurate → still accepted).
_MOSEK_PARAMS_MIP = {
    **_MOSEK_PARAMS,
    "MSK_DPAR_OPTIMIZER_MAX_TIME":      120.0,
    "MSK_DPAR_MIO_MAX_TIME":            120.0,
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
    print(f"  Solving ({solver_name}, vars={n_vars}) ...", end=" ", flush=True)

    is_mip = any(v.attributes.get("boolean") for v in prob.variables())
    kwargs: dict = {"verbose": False}
    if solver is cp.MOSEK:
        kwargs["mosek_params"] = _MOSEK_PARAMS_MIP if is_mip else _MOSEK_PARAMS

    prob.solve(solver=solver, **kwargs)
    val = f"  obj={prob.value:.4f}" if prob.value is not None else ""
    print(f"status={prob.status}{val}")
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Optimization failed: {prob.status}")


# ── Main orchestration ────────────────────────────────────────────────────────

def run_optimization(strategy: dict) -> tuple[pd.DataFrame, dict]:
    print(f"\n{'='*60}")
    print(f"  Strategy:  {strategy['name']}  ({strategy['strategy_id']})")
    print(f"  Objective: {strategy['objective']}")
    print(f"{'='*60}")

    objective = strategy["objective"]

    print("  Loading covariance ...", end=" ", flush=True)
    risk_cov, risk_isins = load_covariance(strategy["risk_date"])
    risk_isin_idx = {isin: i for i, isin in enumerate(risk_isins)}
    print(f"{len(risk_isins)} stocks in risk model")

    # Benchmark — always load for display if benchmark_file is set;
    # only fed into the optimiser's b vector for maximize_alpha.
    if strategy["benchmark_file"]:
        print("  Loading benchmark ...", end=" ", flush=True)
        bm_df = load_benchmark(strategy["benchmark_file"])
        ticker_to_isin = map_tickers_to_isins(list(bm_df["ticker"]))
        bm_df["isin"] = bm_df["ticker"].map(ticker_to_isin)
        bm_df = bm_df.dropna(subset=["isin"])
        bm_df["weight"] /= bm_df["weight"].sum()
        print(f"{len(bm_df)} benchmark stocks")
    else:
        bm_df = pd.DataFrame(columns=["isin", "weight"])

    # Alpha scores (blended from Alpha_Weights sheet)
    print("  Loading alpha scores ...", end=" ", flush=True)
    alpha_lookup = load_blended_alpha(strategy["alpha_weights"], strategy["alpha_date"])
    models_str   = ", ".join(f"{m}×{w:.2g}" for m, w in strategy["alpha_weights"].items())
    print(f"{len(alpha_lookup)} scores  [{models_str}]")

    meta_df = load_universe_metadata()
    gics_df = meta_df.set_index("isin")

    investable  = _build_investable(strategy, bm_df, risk_isins, risk_isin_idx)
    alpha_arr   = np.array([alpha_lookup.get(isin, 0.0) for isin in investable])

    # Pre-screen universe when cardinality is set — reduces MIP problem size.
    # Keep the top candidates by alpha (sharpe/alpha objectives) or by low
    # individual variance (min_variance), capped at max_positions * 5 or 200.
    c_pre       = strategy["constraints"]
    max_pos_n   = c_pre.get("max_positions")
    if max_pos_n is not None:
        prescreen_n = max(int(max_pos_n) * 5, 200)
        if len(investable) > prescreen_n:
            if strategy["objective"] == "minimize_variance":
                # Pre-fetch diagonal of covariance to rank by lowest individual vol
                diag_full = np.array([risk_cov[risk_isin_idx[i], risk_isin_idx[i]]
                                      for i in investable])
                keep_idx = np.argsort(diag_full)[:prescreen_n]
            else:
                keep_idx = np.argsort(-alpha_arr)[:prescreen_n]
            keep_idx  = np.sort(keep_idx)
            investable = [investable[i] for i in keep_idx]
            alpha_arr  = alpha_arr[keep_idx]
            print(f"  MIP pre-screen: {len(investable)} candidates "
                  f"(max_positions={max_pos_n})")

    N           = len(investable)
    isin_to_pos = {isin: i for i, isin in enumerate(investable)}
    print(f"  Investable universe: {N} stocks")

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
            print(f"  Risk model:  Barra ({barra_date})  [{L.shape[1]} factors+stocks]")
        else:
            print(f"  Risk model:  Ledoit-Wolf (Barra unavailable)")
    else:
        print(f"  Risk model:  Ledoit-Wolf")

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

    print(f"\n  Saved → {latest}")
    return latest


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Portfolio optimizer")
    parser.add_argument("--strategy", help="Strategy ID (default: all active)")
    parser.add_argument("--list",     action="store_true", help="List strategies")
    parser.add_argument("--solver",   help="Override solver for all strategies (e.g. MOSEK, CLARABEL, SCS)")
    args = parser.parse_args()

    if not PARAMS_FILE.exists():
        print(f"ERROR: {PARAMS_FILE} not found. Run create_strategy_params.py first.")
        return

    if args.list:
        xl = pd.ExcelFile(PARAMS_FILE)
        df = pd.read_excel(xl, sheet_name="Strategies")
        print(df[["strategy_id", "name", "active", "objective"]].to_string(index=False))
        return

    strategies = load_strategy_params(args.strategy)
    if args.solver:
        for s in strategies:
            s["solver"] = args.solver.upper()
    print(f"Running {len(strategies)} strategy/strategies ...")

    for s in strategies:
        try:
            results_df, summary = run_optimization(s)
            save_results(results_df, summary)
        except Exception as exc:
            print(f"\n  ERROR in {s['strategy_id']}: {exc}")
            raise

    print("\nDone.")


if __name__ == "__main__":
    main()
