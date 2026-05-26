"""Tests for the portfolio optimizer in optimize_portfolio.py.

Uses a tiny synthetic 5-stock universe to verify that each objective
respects the standard constraint set and produces a valid simplex of weights.

CLARABEL is the default solver — installed with cvxpy.  Tests deliberately
avoid integer constraints (max_positions / min_position_if_held) so they
don't depend on MOSEK being licensed.
"""

import numpy as np
import pytest

# Importing optimize_portfolio runs the MOSEK symlink shim — harmless.
from optimize_portfolio import (
    _optimize_alpha,
    _optimize_min_variance,
    _optimize_sharpe,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

def _make_universe(n=5):
    """Diagonal-Sigma universe — equal idiosyncratic risk, no correlations."""
    investable = [f"ISIN-{i}" for i in range(n)]
    Sigma = np.eye(n) * 0.04            # 20% annual vol each
    L     = np.linalg.cholesky(Sigma)
    alpha = np.arange(1.0, n + 1.0)     # increasing alpha by stock
    b     = np.full(n, 1.0 / n)         # equal-weight benchmark
    return investable, alpha, b, Sigma, L


def _single_sector_industry(n=5):
    """All stocks in one sector / one industry for the basic happy paths."""
    sectors    = ["Tech"]
    industries = ["Software"]
    B_sector   = np.ones((1, n))
    B_ind      = np.ones((1, n))
    return sectors, industries, B_sector, B_ind


def _basic_strategy(objective, **constraint_overrides):
    constraints = {
        "max_position":             0.5,
        "max_stock_active_weight":  0.3,
        "max_active_risk":          1.0,    # loose — should not bind
    }
    constraints.update(constraint_overrides)
    return {
        "constraints": constraints,
        "solver":      "CLARABEL",
        "objective":   objective,
    }


# ── Sanity properties (apply to every objective) ────────────────────────────

def _assert_valid_simplex(weights, max_position):
    assert weights.shape[0] > 0
    assert np.all(weights >= -1e-7), f"negative weight: {weights}"
    assert weights.sum() == pytest.approx(1.0, abs=1e-5)
    assert weights.max() <= max_position + 1e-5


# ── maximize_alpha ──────────────────────────────────────────────────────────

def test_maximize_alpha_returns_valid_portfolio():
    investable, alpha, b, Sigma, L = _make_universe()
    sectors, industries, B_sector, B_ind = _single_sector_industry()
    strat = _basic_strategy("maximize_alpha")

    weights, info = _optimize_alpha(
        strat, investable, alpha, b, Sigma, L,
        sectors, industries, B_sector, B_ind,
    )
    _assert_valid_simplex(weights, max_position=0.5)
    assert info["expected_alpha"] > 0           # positive alpha extraction
    assert info["info_ratio"] > 0


def test_maximize_alpha_tilts_toward_higher_alpha():
    # Stock 5 has the highest alpha; in expectation the optimizer should
    # overweight it relative to the benchmark.
    investable, alpha, b, Sigma, L = _make_universe()
    sectors, industries, B_sector, B_ind = _single_sector_industry()
    strat = _basic_strategy("maximize_alpha")

    weights, _ = _optimize_alpha(
        strat, investable, alpha, b, Sigma, L,
        sectors, industries, B_sector, B_ind,
    )
    # Highest-alpha name overweighted; lowest-alpha name underweighted.
    assert weights[-1] > b[-1]
    assert weights[0]  < b[0]


def test_maximize_alpha_respects_max_stock_active_weight():
    investable, alpha, b, Sigma, L = _make_universe()
    sectors, industries, B_sector, B_ind = _single_sector_industry()
    strat = _basic_strategy("maximize_alpha", max_stock_active_weight=0.05)

    weights, _ = _optimize_alpha(
        strat, investable, alpha, b, Sigma, L,
        sectors, industries, B_sector, B_ind,
    )
    active = weights - b
    assert np.all(np.abs(active) <= 0.05 + 1e-5)


def test_maximize_alpha_respects_tracking_error():
    investable, alpha, b, Sigma, L = _make_universe()
    sectors, industries, B_sector, B_ind = _single_sector_industry()
    # Tight TE budget — the optimizer should hug the benchmark closely.
    strat = _basic_strategy("maximize_alpha", max_active_risk=0.01)

    weights, _ = _optimize_alpha(
        strat, investable, alpha, b, Sigma, L,
        sectors, industries, B_sector, B_ind,
    )
    active     = weights - b
    active_te  = np.sqrt(active @ Sigma @ active)
    assert active_te <= 0.01 + 1e-5


# ── minimize_variance ───────────────────────────────────────────────────────

def test_minimize_variance_equal_weight_with_equal_vols():
    # With diagonal Σ and equal variances, the unconstrained min-var portfolio
    # is exactly equal-weighted.  max_position=0.5 leaves room for that.
    investable, _, b, Sigma, L = _make_universe()
    sectors, industries, B_sector, B_ind = _single_sector_industry()
    strat = _basic_strategy("minimize_variance")

    weights, info = _optimize_min_variance(
        strat, investable, b, Sigma, L,
        sectors, industries, B_sector, B_ind,
    )
    _assert_valid_simplex(weights, max_position=0.5)
    np.testing.assert_allclose(weights, np.full(5, 0.2), atol=1e-4)
    assert info["portfolio_vol"] > 0


def test_minimize_variance_binding_max_position():
    # Concentrate volatility in stock 0 — without the position cap, min-var
    # would pile into the low-vol names.  With max_position=0.3, no name can
    # exceed 30%.
    n = 5
    investable = [f"ISIN-{i}" for i in range(n)]
    variances  = np.array([0.01, 0.04, 0.04, 0.04, 0.04])
    Sigma      = np.diag(variances)
    L          = np.linalg.cholesky(Sigma)
    b          = np.full(n, 1.0 / n)
    sectors, industries, B_sector, B_ind = _single_sector_industry(n)

    strat = _basic_strategy("minimize_variance", max_position=0.3)
    weights, _ = _optimize_min_variance(
        strat, investable, b, Sigma, L,
        sectors, industries, B_sector, B_ind,
    )
    _assert_valid_simplex(weights, max_position=0.3)
    assert weights[0] == pytest.approx(0.3, abs=1e-4)    # cap binds on low-vol


# ── maximize_sharpe ─────────────────────────────────────────────────────────

def test_maximize_sharpe_returns_valid_portfolio():
    investable, alpha, _b, Sigma, L = _make_universe()
    sectors, industries, B_sector, B_ind = _single_sector_industry()
    strat = _basic_strategy("maximize_sharpe")

    weights, info = _optimize_sharpe(
        strat, investable, alpha, Sigma, L,
        sectors, industries, B_sector, B_ind,
    )
    _assert_valid_simplex(weights, max_position=0.5)
    assert info["sharpe_ratio"] > 0
    assert info["portfolio_vol"] > 0


def test_maximize_sharpe_orders_by_alpha_with_equal_vols():
    # Diagonal Σ and equal variances: the unconstrained Sharpe portfolio is
    # proportional to Σ⁻¹α — weights should rank-order the alphas.
    investable, alpha, _b, Sigma, L = _make_universe()
    sectors, industries, B_sector, B_ind = _single_sector_industry()
    strat = _basic_strategy("maximize_sharpe", max_position=0.5)   # loose cap

    weights, _ = _optimize_sharpe(
        strat, investable, alpha, Sigma, L,
        sectors, industries, B_sector, B_ind,
    )
    # Strictly monotonic in alpha (stock i+1 has higher alpha → higher weight)
    assert np.all(np.diff(weights) > 0)


def test_maximize_sharpe_max_position_binds_when_tight():
    # Tighten the cap below the unconstrained allocation (~33% on top name)
    # — now the cap should bind.
    investable, alpha, _b, Sigma, L = _make_universe()
    sectors, industries, B_sector, B_ind = _single_sector_industry()
    strat = _basic_strategy("maximize_sharpe", max_position=0.25)

    weights, _ = _optimize_sharpe(
        strat, investable, alpha, Sigma, L,
        sectors, industries, B_sector, B_ind,
    )
    assert weights[-1] == pytest.approx(0.25, abs=1e-3)   # cap binds on highest-alpha
    assert np.all(weights <= 0.25 + 1e-5)


# ── Sector constraints ──────────────────────────────────────────────────────

def test_excluded_sector_gets_zero_weight():
    # 4 stocks in Tech, 1 in Energy.  excluded_sectors="Energy" must zero it.
    investable = [f"ISIN-{i}" for i in range(5)]
    Sigma      = np.eye(5) * 0.04
    L          = np.linalg.cholesky(Sigma)
    alpha      = np.array([3.0, 3.0, 3.0, 3.0, 10.0])     # last stock = highest alpha
    b          = np.full(5, 0.2)

    sectors    = ["Tech", "Energy"]
    industries = ["X"]
    # Stock 4 is in Energy; rest in Tech.
    B_sector   = np.array([
        [1, 1, 1, 1, 0],     # Tech row
        [0, 0, 0, 0, 1],     # Energy row
    ], dtype=float)
    B_ind      = np.ones((1, 5))

    strat = _basic_strategy("maximize_alpha",
                            excluded_sectors="Energy",
                            max_stock_active_weight=1.0)

    weights, _ = _optimize_alpha(
        strat, investable, alpha, b, Sigma, L,
        sectors, industries, B_sector, B_ind,
    )
    # Even though stock 4 has the highest alpha, its sector is excluded → weight 0.
    assert weights[4] == pytest.approx(0.0, abs=1e-6)
    # Remaining weight is split among the Tech names.
    assert weights[:4].sum() == pytest.approx(1.0, abs=1e-5)


# ── Turnover constraint ─────────────────────────────────────────────────────

def test_max_turnover_limits_one_way_change():
    # Start from an equal-weight prior portfolio; force turnover budget = 10%.
    # The post-optimisation portfolio should not move further than that.
    investable, alpha, b, Sigma, L = _make_universe()
    sectors, industries, B_sector, B_ind = _single_sector_industry()
    strat = _basic_strategy("maximize_alpha")

    prev = np.full(5, 0.2)
    weights, _ = _optimize_alpha(
        strat, investable, alpha, b, Sigma, L,
        sectors, industries, B_sector, B_ind,
        prev_weights_arr=prev, max_turnover=0.10,
    )
    one_way = np.abs(weights - prev).sum() / 2
    assert one_way <= 0.10 + 1e-5
