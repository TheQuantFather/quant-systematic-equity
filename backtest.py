"""backtest — headless walk-forward optimised backtest.

Single source of truth for the CVXPY walk-forward backtest. Both the Streamlit
Backtester page (pages/6_Backtester.py) and the standalone HTML report generator
(scripts/backtest_report.py) call ``run_optimised_backtest`` here, so the
simulation logic never drifts between the interactive and the published views.

The page wraps these with ``st.cache_data``; this module itself has no Streamlit
dependency and queries the databases directly (matching optimize_portfolio.py
and report_utils.py).
"""
from __future__ import annotations

import sqlite3
from typing import Callable

import numpy as np
import pandas as pd

from config import MODELS_DB, PARAMS_FILE, RETURNS_DB, RISK_DB, UNIVERSE_DB
from utils import get_db, get_snapshot_schedule

ProgressCB = Callable[[int, int, str], None]


# ---------------------------------------------------------------------------
# Direct database helpers (headless equivalents of db.py's cached wrappers)
# ---------------------------------------------------------------------------

def load_returns_matrix(min_isin_coverage: int = 200) -> pd.DataFrame:
    """Wide daily total-return matrix (date × isin), truncated to the last day
    with at least ``min_isin_coverage`` priced names."""
    with get_db(RETURNS_DB) as conn:
        df = pd.read_sql(
            "SELECT isin, date, total_return FROM returns WHERE total_return IS NOT NULL", conn
        )
    df["date"] = pd.to_datetime(df["date"])
    matrix = df.pivot_table(index="date", columns="isin", values="total_return").sort_index()
    matrix.columns.name = None
    coverage   = matrix.notna().sum(axis=1)
    last_valid = coverage[coverage >= min_isin_coverage].index.max()
    if pd.notna(last_valid):
        matrix = matrix.loc[:last_valid]
    return matrix


def find_nearest_before(target: str, dates: list[str]) -> str | None:
    candidates = [d for d in dates if d <= target]
    return max(candidates) if candidates else None


def universe_meta() -> pd.DataFrame:
    """Per-security metadata keyed by security_id (= isin): sector, industry,
    ticker, company_name. Mirrors db.get_universe() column derivation."""
    with get_db(UNIVERSE_DB) as conn:
        df = pd.read_sql(
            "SELECT isin, ticker, company_name, gics_sector, simfin_sector, simfin_industry "
            "FROM companies",
            conn,
        )
    df["ticker"]      = df["ticker"].fillna("")
    df["sector"]      = df["gics_sector"].fillna(df["simfin_sector"]).fillna("Unknown")
    df["industry"]    = df["simfin_industry"].fillna("Unknown")
    df["security_id"] = df["isin"]
    return df[["security_id", "sector", "industry", "ticker", "company_name"]]


def isins_at_date(index_name: str, snapshot_date: str) -> list[str]:
    with get_db(UNIVERSE_DB) as conn:
        rows = conn.execute(
            "SELECT isin FROM universe_snapshots WHERE snapshot_date = ? AND index_name = ?",
            (snapshot_date, index_name),
        ).fetchall()
    return [r[0] for r in rows]


def weights_at_date(index_name: str, snapshot_date: str) -> dict[str, float]:
    """{isin: weight} for an index at snapshot_date, normalised to sum=1."""
    with get_db(UNIVERSE_DB) as conn:
        rows = conn.execute(
            "SELECT isin, weight FROM universe_snapshots "
            "WHERE snapshot_date = ? AND index_name = ? AND weight IS NOT NULL",
            (snapshot_date, index_name),
        ).fetchall()
    if not rows:
        return {}
    raw   = {r[0]: float(r[1]) for r in rows}
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {isin: w / total for isin, w in raw.items()}


# ---------------------------------------------------------------------------
# Walk-forward optimised backtest
# ---------------------------------------------------------------------------

def run_optimised_backtest(
    strategy_id: str,
    portfolio_eur: float,
    max_turnover: float,
    tc_per_trade_eur: float,
    benchmark_name: str,
    universe_name: str = "sp500",
    rebal_freq: str = "quarterly",
    min_pos_if_held: float | None = None,
    max_positions_override: int | None = None,
    solver: str = "CLARABEL",
    rebalance_cadences: set[str] | None = None,
    progress_cb: ProgressCB | None = None,
) -> dict:
    """Walk-forward optimised backtest. Starts from the first date where a Barra
    risk-model snapshot is available, so all periods use a consistent risk model.

    ``progress_cb(i, n, snap_date)`` is invoked once per rebalance date (the
    Streamlit page passes a progress-bar updater; the report passes a logger).

    Returns a results dict or ``{"error": str}`` on failure.
    """
    from optimize_portfolio import load_strategy_params, optimize_for_backtest

    if not PARAMS_FILE.exists():
        return {"error": "strategy_params.xlsx not found. Run create_strategy_params.py first."}
    try:
        strategies = load_strategy_params(strategy_id)
    except Exception as exc:
        return {"error": str(exc)}
    if not strategies:
        return {"error": f"Strategy '{strategy_id}' not found or inactive."}

    sp            = strategies[0]
    alpha_weights = sp["alpha_weights"]
    objective     = sp["objective"]
    constraints   = dict(sp["constraints"])

    # All available model, universe, and benchmark snapshot dates (carry-forward for gaps).
    with get_db(MODELS_DB) as conn:
        model_dates = sorted(r[0] for r in conn.execute("SELECT DISTINCT data_date FROM models").fetchall())
    with get_db(UNIVERSE_DB) as conn:
        universe_dates = sorted(r[0] for r in conn.execute(
            "SELECT DISTINCT snapshot_date FROM universe_snapshots WHERE index_name = ?",
            (universe_name,),
        ).fetchall())
        benchmark_dates = sorted(r[0] for r in conn.execute(
            "SELECT DISTINCT snapshot_date FROM universe_snapshots WHERE index_name = ?",
            (benchmark_name,),
        ).fetchall())
    if not model_dates or not universe_dates:
        return {"error": f"Need at least 1 model date and 1 '{universe_name}' universe snapshot."}

    # Available Barra risk dates. Ledoit-Wolf remains in risk.db for analysis
    # pages, but it is not an optimizer fallback.
    barra_dates: list[str] = []
    try:
        with get_db(RISK_DB) as conn:
            barra_dates = sorted(r[0] for r in conn.execute(
                "SELECT DISTINCT snapshot_date FROM factor_covariance"
            ).fetchall())
    except Exception:
        barra_dates = []

    pre_warnings: list[str] = []

    # Signal/risk carry-forward pools = every computed snapshot. Monthly mode
    # rebalances on the last snapshot of each calendar month, landing directly on
    # snapshot dates (including recent weekly/ad-hoc ones), so there is no staleness
    # and no dependence on cadence tags.
    alpha_lookup_dates = model_dates
    barra_lookup_dates = barra_dates

    ret_matrix    = load_returns_matrix()
    trading_index = ret_matrix.index

    def next_td(d_str: str):
        pos = trading_index.searchsorted(pd.Timestamp(d_str))
        return trading_index[pos] if pos < len(trading_index) else None

    # ── Build rebalancing schedule ────────────────────────────────────────────
    # Start from the first date with Barra coverage so all periods use consistent risk model.
    first_barra = barra_lookup_dates[0] if barra_lookup_dates else None
    if first_barra is None:
        return {"error": "No Barra snapshots found. Run create_barra.py --backfill first."}

    # Snapshot dates that carry BOTH an alpha model and a Barra risk snapshot,
    # from the first Barra date onward — the only dates we can actually rebalance on.
    usable_snaps = [d for d in sorted(set(model_dates) & set(barra_dates)) if d >= first_barra]

    if rebal_freq == "monthly":
        # One rebalance per calendar month: the last usable snapshot in each month.
        # Follows whatever cadence the pipeline actually produced (month-ends
        # historically, weekly recently) without reading cadence tags, so the book is
        # rebalanced every month through the latest month and never held stale.
        by_month: dict[str, str] = {}
        for d in usable_snaps:
            by_month[d[:7]] = d
        rebal_dates = sorted(by_month.values())
    else:
        # Quarterly: every usable snapshot date, optionally restricted to specific
        # snapshot cadences (canonical schedule) so a weekly run never forces a rebalance.
        rebal_dates = list(usable_snaps)
        if rebalance_cadences:
            allowed = set(get_snapshot_schedule(cadence=tuple(rebalance_cadences), computed_only=True))
            rebal_dates = [d for d in rebal_dates if d in allowed]

    if len(rebal_dates) < 2:
        return {"error": "Not enough rebalancing dates in the backtest window."}

    # Apply overrides (take precedence over strategy defaults)
    if max_positions_override is not None:
        constraints["max_positions"] = max_positions_override
    if min_pos_if_held is not None:
        constraints["min_position_if_held"] = min_pos_if_held

    meta_df      = universe_meta()
    sector_map   = dict(zip(meta_df["security_id"], meta_df["sector"]))
    industry_map = dict(zip(meta_df["security_id"], meta_df["industry"]))
    ticker_map   = dict(zip(meta_df["security_id"], meta_df["ticker"]))
    name_map     = dict(zip(meta_df["security_id"], meta_df["company_name"]))

    prev_weights: dict[str, float] | None = None
    period_log:   list[dict]              = []
    return_parts: list[pd.Series]         = []
    warnings:     list[str]               = pre_warnings.copy()
    benchmark_weights_fallback_warned     = False

    for i, snap_date in enumerate(rebal_dates):
        if progress_cb is not None:
            progress_cb(i, len(rebal_dates), snap_date)
        next_snap = (
            rebal_dates[i + 1] if i + 1 < len(rebal_dates)
            else trading_index[-1].strftime("%Y-%m-%d")
        )
        t_start = next_td(snap_date)
        t_end   = next_td(next_snap)
        if t_start is None or t_end is None or t_start >= t_end:
            continue

        # Alpha, universe, and benchmark weights are carried forward from the
        # most recent available snapshot.
        alpha_date   = find_nearest_before(snap_date, alpha_lookup_dates)
        uni_snap     = find_nearest_before(snap_date, universe_dates)
        bm_snap      = find_nearest_before(snap_date, benchmark_dates)
        barra_date   = find_nearest_before(snap_date, barra_lookup_dates)
        risk_date    = barra_date

        if alpha_date is None or uni_snap is None:
            warnings.append(f"{snap_date}: no alpha or universe snapshot available — skipped.")
            continue
        if barra_date is None:
            warnings.append(f"{snap_date}: no Barra risk date available — skipped.")
            continue

        uni_isins = isins_at_date(universe_name, uni_snap)
        if objective == "maximize_alpha":
            if bm_snap is not None:
                bm_weights = weights_at_date(benchmark_name, bm_snap)
            else:
                bm_weights = weights_at_date(universe_name, uni_snap)
                if not benchmark_weights_fallback_warned:
                    warnings.append(
                        f"No constituent weights found for benchmark '{benchmark_name}' — "
                        f"optimizer constraints use universe '{universe_name}' weights instead."
                    )
                    benchmark_weights_fallback_warned = True
        else:
            bm_weights = {}
        if len(uni_isins) < 50:
            warnings.append(f"{snap_date}: only {len(uni_isins)} stocks in '{universe_name}' — skipped.")
            continue

        # Progressive turnover relaxation: when the combination of tight turnover
        # and shifting risk-model factor loadings makes the period infeasible, step
        # up 1.5× and 2.25× before removing the constraint entirely. The actual
        # turnover used is recorded in opt_metrics["turnover_relaxed"].
        _to_attempts = [max_turnover, max_turnover * 1.5, max_turnover * 1.5 ** 2, None]
        opt_result = None
        to_used: float | None = None
        for _to in _to_attempts:
            opt_result = optimize_for_backtest(
                alpha_weights=alpha_weights,
                objective=objective,
                constraints=constraints,
                alpha_date=alpha_date,
                barra_date=barra_date,
                risk_date=risk_date,
                sp500_isins=uni_isins,
                bm_weights=bm_weights,
                prev_weights=prev_weights,
                max_turnover=_to if _to is not None else 9999.0,
                solver=solver,
                risk_aversion=sp.get("risk_aversion", 0.0),
            )
            if opt_result is not None:
                to_used = _to
                break

        if opt_result is None:
            if prev_weights is None:
                warnings.append(f"{snap_date}: optimization failed before any portfolio was built — skipped.")
                continue
            warnings.append(f"{snap_date}: optimization failed — carrying forward previous weights.")
            new_weights: dict[str, float] = prev_weights
            opt_metrics: dict = {}
        else:
            new_weights, opt_metrics = opt_result
            if to_used != max_turnover:
                label = f"{to_used * 100:.0f}%" if to_used is not None else "unconstrained"
                warnings.append(
                    f"{snap_date}: turnover relaxed to {label} (requested {max_turnover * 100:.0f}% was infeasible)."
                )
                opt_metrics["turnover_relaxed"] = True

        # Transaction costs: count a trade only when the EUR value of the order meets a
        # minimum order size. Monthly rebalancing produces many small weight tweaks that
        # would not be executed in practice. Threshold scales with portfolio so TC% is
        # independent of portfolio size.
        min_order_eur   = max(tc_per_trade_eur / 0.01, 200.0)  # cap TC at ~1% of order value
        trade_threshold = min_order_eur / portfolio_eur
        if prev_weights is None:
            n_trades = sum(1 for w in new_weights.values() if w >= trade_threshold)
        else:
            n_trades = sum(
                1 for isin in set(new_weights) | set(prev_weights)
                if abs(new_weights.get(isin, 0.0) - prev_weights.get(isin, 0.0)) >= trade_threshold
            )
        tc_pct = n_trades * tc_per_trade_eur / portfolio_eur

        # Hold-period return simulation
        period = ret_matrix.loc[(ret_matrix.index >= t_start) & (ret_matrix.index < t_end)]
        avail  = [isin for isin in new_weights if isin in period.columns]
        if avail:
            w_arr = np.array([new_weights[isin] for isin in avail])
            # Scale to the invested fraction: names without return data are
            # redistributed pro-rata, but a deliberate cash buffer (weights summing
            # below 1 under max_cash_weight) is preserved and earns 0% for the period.
            invested = min(1.0, sum(new_weights.values()))
            w_arr   *= invested / w_arr.sum()
            port_returns = pd.Series(
                period[avail].fillna(0.0).values @ w_arr, index=period.index
            )
        else:
            port_returns = pd.Series(0.0, index=period.index)

        if len(port_returns) > 0:
            port_returns.iloc[0] -= tc_pct

        # Two-way turnover: sum of absolute weight changes (buys + sells)
        if prev_weights is not None and new_weights:
            actual_to = sum(
                abs(new_weights.get(isin, 0.0) - prev_weights.get(isin, 0.0))
                for isin in set(new_weights) | set(prev_weights)
            )
        else:
            actual_to = 1.0

        sector_weights:    dict[str, float] = {}
        bm_sector_weights: dict[str, float] = {}
        industry_weights:    dict[str, float] = {}
        bm_industry_weights: dict[str, float] = {}
        for isin, w in new_weights.items():
            sec = sector_map.get(isin, "Unknown")
            ind = industry_map.get(isin, "Unknown")
            sector_weights[sec] = sector_weights.get(sec, 0.0) + w
            industry_weights[ind] = industry_weights.get(ind, 0.0) + w
        for isin, w in bm_weights.items():
            sec = sector_map.get(isin, "Unknown")
            ind = industry_map.get(isin, "Unknown")
            bm_sector_weights[sec] = bm_sector_weights.get(sec, 0.0) + w
            bm_industry_weights[ind] = bm_industry_weights.get(ind, 0.0) + w

        period_log.append({
            "snap_date":            snap_date,
            "next_snap":            next_snap[:10],
            "alpha_date":           alpha_date,
            "universe_snapshot":     uni_snap,
            "benchmark_name":        benchmark_name,
            "benchmark_snapshot":    bm_snap if bm_snap is not None else uni_snap,
            "weights":              new_weights,
            "bm_weights":           dict(bm_weights),
            "barra_date":           barra_date,
            "n_trades":             n_trades,
            "tc_pct":               tc_pct,
            "turnover":             actual_to,
            "sector_weights":       sector_weights,
            "bm_sector_weights":    bm_sector_weights,
            "industry_weights":     industry_weights,
            "bm_industry_weights":  bm_industry_weights,
            "metrics":              opt_metrics,
            "used_barra":           opt_metrics.get("used_barra", False),
            "relaxed_integer":      opt_metrics.get("relaxed_integer", False),
            "turnover_relaxed":     opt_metrics.get("turnover_relaxed", False),
            "n_positions":          opt_metrics.get("n_positions", len(new_weights)),
        })
        return_parts.append(port_returns)
        prev_weights = new_weights

    if not return_parts:
        return {"error": "No valid backtest periods found."}

    return {
        "port_series":   pd.concat(return_parts).sort_index(),
        "period_log":    period_log,
        "warnings":      warnings,
        "strategy_name": sp["name"],
        "objective":     objective,
        "rebal_freq":    rebal_freq,
        "universe_name": universe_name,
        "benchmark_name": benchmark_name,
        "sector_map":    sector_map,
        "industry_map":  industry_map,
        "ticker_map":    ticker_map,
        "name_map":      name_map,
    }
