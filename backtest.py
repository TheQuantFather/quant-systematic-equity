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

from typing import Callable

import numpy as np
import pandas as pd

from classification.register import VIEW_FALLBACK, view_labels
from config import MODELS_DB, PARAMS_FILE, RETURNS_DB, RISK_DB, UNIVERSE_DB
from securities.corporate_actions import convert_weights, holding_return_matrix, load_conversions
from securities.data import (
    metadata_by_sid,
    returns_matrix_by_sid,
    universe_members_by_sid,
    validate_universe_identity,
)
from utils import get_db, get_snapshot_schedule, holding_period

ProgressCB = Callable[[int, int, str], None]


# ---------------------------------------------------------------------------
# Direct database helpers (headless equivalents of db.py's cached wrappers)
# ---------------------------------------------------------------------------


def load_returns_matrix(min_sid_coverage: int = 200) -> pd.DataFrame:
    """Continuous daily total returns keyed by stable integer SID."""
    with get_db(RETURNS_DB) as conn, get_db(UNIVERSE_DB) as master:
        matrix = returns_matrix_by_sid(conn, master, min_sid_coverage)
        return holding_return_matrix(matrix, load_conversions(master, require_valuation=True))


def find_nearest_before(target: str, dates: list[str]) -> str | None:
    candidates = [d for d in dates if d <= target]
    return max(candidates) if candidates else None


def universe_meta(metadata: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-security metadata keyed by security_id (= sid): sector, industry,
    ticker, company_name. Mirrors db.get_universe() column derivation."""
    if metadata is None:
        with get_db(UNIVERSE_DB) as conn:
            metadata = metadata_by_sid(conn)
    df = metadata.copy()
    df["ticker"] = df["ticker"].fillna("")
    if "business_sector" in df.columns:  # in-house labels (CLASSIFICATION_SOURCE)
        df["sector"], df["industry"] = df["business_sector"], df["business_industry"]
    else:
        df["sector"] = df["gics_sector"].fillna(df["simfin_sector"]).fillna("Unknown")
        df["industry"] = df["simfin_industry"].fillna("Unknown")
    df["security_id"] = df["sid"]
    return df[["security_id", "sector", "industry", "ticker", "company_name"]]


def group_weights(weights: dict[int, float], views: pd.DataFrame) -> dict[str, dict[str, dict]]:
    """{scheme: {level: {group: weight}}} for each classification view in ``views``.

    ``views`` is view_labels() indexed by sid; SIDs without metadata use the
    scheme's fallback group.
    """
    out: dict[str, dict[str, dict]] = {}
    w = pd.Series(weights, dtype=float)
    for (scheme, level), labels in views.items():
        groups = labels.reindex(w.index).fillna(VIEW_FALLBACK[scheme])
        out.setdefault(scheme, {})[level] = w.groupby(groups).sum().to_dict()
    return out


def sids_at_date(index_name: str, snapshot_date: str) -> list[int]:
    with get_db(UNIVERSE_DB) as conn:
        members = universe_members_by_sid(conn, index_name, snapshot_date)
    return members["sid"].tolist()


def weights_at_date(index_name: str, snapshot_date: str) -> dict[int, float]:
    """{sid: weight} for an index at snapshot_date, normalised to sum=1."""
    with get_db(UNIVERSE_DB) as conn:
        members = universe_members_by_sid(conn, index_name, snapshot_date).dropna(subset="weight")
    if members.empty:
        return {}
    raw = dict(zip(members["sid"], members["weight"].astype(float)))
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {sid: w / total for sid, w in raw.items()}


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
    short_borrow_bps: float | None = None,
    progress_cb: ProgressCB | None = None,
    allow_assumed_attributes: bool = True,
) -> dict:
    """Walk-forward optimised backtest. Starts from the first date where a Barra
    risk-model snapshot is available, so all periods use a consistent risk model.

    ``progress_cb(i, n, snap_date)`` is invoked once per rebalance date (the
    Streamlit page passes a progress-bar updater; the report passes a logger).

    Returns a results dict or ``{"error": str}`` on failure.
    """
    from optimize_portfolio import (
        _with_business_labels,
        load_latest_market_caps,
        load_strategy_params,
        optimize_for_backtest,
    )

    if not PARAMS_FILE.exists():
        return {"error": "strategy_params.xlsx not found. Run create_strategy_params.py first."}
    try:
        strategies = load_strategy_params(strategy_id)
    except Exception as exc:
        return {"error": str(exc)}
    if not strategies:
        return {"error": f"Strategy '{strategy_id}' not found or inactive."}

    sp = strategies[0]
    alpha_weights = sp["alpha_weights"]
    objective = sp["objective"]
    constraints = dict(sp["constraints"])

    # Long/short (market-neutral) book: signed weights, no invested-fraction
    # rescaling, and a borrow drag on the short leg. Gated behind long_only=False
    # so long-only strategies are simulated exactly as before.
    is_long_short = not bool(constraints.get("long_only", True))
    if short_borrow_bps is None:
        short_borrow_bps = float(constraints.get("short_borrow_bps", 0.0) or 0.0)

    # All available model, universe, and benchmark snapshot dates (carry-forward for gaps).
    with get_db(MODELS_DB) as conn:
        model_dates = sorted(
            r[0] for r in conn.execute("SELECT DISTINCT data_date FROM models").fetchall()
        )
    with get_db(UNIVERSE_DB) as conn:
        universe_dates = sorted(
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT snapshot_date FROM universe_snapshots WHERE index_name = ?",
                (universe_name,),
            ).fetchall()
        )
        benchmark_dates = sorted(
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT snapshot_date FROM universe_snapshots WHERE index_name = ?",
                (benchmark_name,),
            ).fetchall()
        )
        try:
            validate_universe_identity(conn, list({universe_name, benchmark_name}))
        except ValueError as exc:
            return {"error": str(exc)}
    if not model_dates or not universe_dates:
        return {"error": f"Need at least 1 model date and 1 '{universe_name}' universe snapshot."}

    # Available Barra risk dates.
    barra_dates: list[str] = []
    try:
        with get_db(RISK_DB) as conn:
            barra_dates = sorted(
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT snapshot_date FROM factor_covariance"
                ).fetchall()
            )
    except Exception:
        barra_dates = []

    pre_warnings: list[str] = []

    # Signal/risk carry-forward pools = every computed snapshot. Monthly mode
    # rebalances on the last snapshot of each calendar month, landing directly on
    # snapshot dates (including recent weekly/ad-hoc ones), so there is no staleness
    # and no dependence on cadence tags.
    alpha_lookup_dates = model_dates
    barra_lookup_dates = barra_dates

    ret_matrix = load_returns_matrix()
    if ret_matrix.empty:
        return {"error": "No SID return observations available for the backtest."}
    trading_index = ret_matrix.index

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
            allowed = set(
                get_snapshot_schedule(cadence=tuple(rebalance_cadences), computed_only=True)
            )
            rebal_dates = [d for d in rebal_dates if d in allowed]

    if len(rebal_dates) < 2:
        return {"error": "Not enough rebalancing dates in the backtest window."}

    # Apply overrides (take precedence over strategy defaults)
    if max_positions_override is not None:
        constraints["max_positions"] = max_positions_override
    if min_pos_if_held is not None:
        constraints["min_position_if_held"] = min_pos_if_held

    with get_db(UNIVERSE_DB) as conn:
        optimizer_metadata = metadata_by_sid(conn)
    # These helpers already use latest metadata/prices in the existing backtest.
    # Load them once per run; signal and risk snapshots still change each period.
    market_caps_lookup = load_latest_market_caps(
        optimizer_metadata, optimizer_metadata["sid"].tolist(), by_sid=True
    )
    sector_map, industry_map, ticker_map, name_map = {}, {}, {}, {}
    label_map: dict[str, dict[str, dict]] = {}  # scheme -> level -> {sid: group}, latest wins

    prev_weights: dict[int, float] | None = None
    period_log: list[dict] = []
    return_parts: list[pd.Series] = []
    warnings: list[str] = pre_warnings.copy()
    benchmark_weights_fallback_warned = False
    with get_db(UNIVERSE_DB) as conn:
        conversions = load_conversions(conn, require_valuation=True)

    for i, snap_date in enumerate(rebal_dates):
        if progress_cb is not None:
            progress_cb(i, len(rebal_dates), snap_date)
        next_snap = (
            rebal_dates[i + 1]
            if i + 1 < len(rebal_dates)
            else trading_index[-1].strftime("%Y-%m-%d")
        )
        # Hold-period rows: window starts the session AFTER the snapshot, so the
        # snapshot-day return (already embedded in the signal's close) is never
        # credited to the book (no 1-day look-ahead). None → unusable period.
        period = holding_period(ret_matrix, snap_date, next_snap)
        if period is None:
            warnings.append(
                f"{snap_date}: no trading observations in the holding period — skipped."
            )
            continue
        if prev_weights is not None:
            prev_weights = convert_weights(prev_weights, conversions, snap_date)

        # Alpha, universe, and benchmark weights are carried forward from the
        # most recent available snapshot.
        alpha_date = find_nearest_before(snap_date, alpha_lookup_dates)
        uni_snap = find_nearest_before(snap_date, universe_dates)
        bm_snap = find_nearest_before(snap_date, benchmark_dates)
        barra_date = find_nearest_before(snap_date, barra_lookup_dates)
        risk_date = barra_date

        if alpha_date is None or uni_snap is None:
            warnings.append(f"{snap_date}: no alpha or universe snapshot available — skipped.")
            continue
        if barra_date is None:
            warnings.append(f"{snap_date}: no Barra risk date available — skipped.")
            continue

        uni_sids = sids_at_date(universe_name, uni_snap)
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
        if len(uni_sids) < 50:
            warnings.append(
                f"{snap_date}: only {len(uni_sids)} stocks in '{universe_name}' — skipped."
            )
            continue

        with get_db(UNIVERSE_DB) as conn:
            period_metadata = metadata_by_sid(
                conn, as_of=snap_date, allow_assumed=allow_assumed_attributes
            )
        period_metadata = _with_business_labels(period_metadata, snap_date)
        relevant_metadata = period_metadata[
            period_metadata.sid.isin(set(uni_sids) | set(bm_weights))
        ]
        if relevant_metadata.metadata_status.eq("assumed_history").any() and not any(
            "assumed company attribute history" in warning for warning in warnings
        ):
            warnings.append(
                "This backtest uses assumed company attribute history for dates without "
                "dated source coverage; historical classifications are reconstructed from "
                "the available snapshots. Set allow_assumed_attributes=False to exclude "
                "these assumptions; unavailable classifications then use Unknown groups. "
                "This does not certify precomputed signals or market data."
            )
        # Missing classifications do not remove securities or stop the run.
        # The optimizer groups unavailable sector/industry labels as Unknown;
        # in-house unclassified names are held at zero weight instead.
        required_labels = {"gics_sector": "GICS sector"}
        if constraints.get("max_industry_active_weight") is not None:
            required_labels["simfin_industry"] = "SimFin industry"
        labels = period_metadata.set_index("sid")
        inhouse = "business_status" in period_metadata.columns
        if inhouse:
            required_labels = {"business_status": "in-house classification"}
            missing = [
                sid
                for sid in uni_sids
                if sid not in labels.index or labels.at[sid, "business_status"] != "accepted"
            ]
        else:
            missing = [
                sid
                for sid in uni_sids
                if sid not in labels.index
                or any(
                    pd.isna(labels.at[sid, field]) or not str(labels.at[sid, field]).strip()
                    for field in required_labels
                )
            ]
        if missing and not any("missing dated company classification" in w for w in warnings):
            warnings.append(
                f"{snap_date}: missing dated company classification for {len(missing)} "
                f"universe securities ({', '.join(required_labels.values())}); "
                + (
                    "unclassified names are held at zero weight."
                    if inhouse
                    else "continuing with Unknown groups for unavailable classifications."
                )
            )
        period_meta = universe_meta(period_metadata)
        period_sectors = dict(zip(period_meta["security_id"], period_meta["sector"]))
        period_industries = dict(zip(period_meta["security_id"], period_meta["industry"]))
        sector_map.update(period_sectors)
        industry_map.update(period_industries)
        # Both classification schemes, as of this date, for display.
        views = view_labels(period_metadata.set_index("sid"))
        for (scheme, level), view in views.items():
            label_map.setdefault(scheme, {}).setdefault(level, {}).update(view.to_dict())
        ticker_map.update(zip(period_meta["security_id"], period_meta["ticker"]))
        name_map.update(zip(period_meta["security_id"], period_meta["company_name"]))

        # Relax turnover only after the solver explicitly proves infeasibility.
        # Timeouts, uncertain infeasibility and data/solver errors do not justify
        # changing the requested portfolio rules. There is no turnover constraint
        # to relax before the first portfolio has been built.
        _to_attempts = [max_turnover, max_turnover * 1.5, max_turnover * 1.5**2, None]
        opt_result = None
        failure_details: list[str] = []
        to_used: float | None = None
        for _to in _to_attempts:
            failure_statuses: list[str] = []
            opt_result = optimize_for_backtest(
                alpha_weights=alpha_weights,
                objective=objective,
                constraints=constraints,
                alpha_date=alpha_date,
                barra_date=barra_date,
                risk_date=risk_date,
                candidate_sids=uni_sids,
                bm_weights=bm_weights,
                prev_weights=prev_weights,
                max_turnover=_to if _to is not None else 9999.0,
                solver=solver,
                risk_aversion=sp.get("risk_aversion", 0.0),
                metadata=period_metadata,
                market_caps_lookup=market_caps_lookup,
                failure_details=failure_details,
                failure_statuses=failure_statuses,
            )
            if opt_result is not None:
                to_used = _to
                break
            if prev_weights is None or failure_statuses != ["infeasible"]:
                break

        if opt_result is None:
            reason = failure_details[-1] if failure_details else "see optimizer logs"
            if prev_weights is None:
                warnings.append(
                    f"{snap_date}: optimization failed before any portfolio was built — skipped: {reason}."
                )
                continue
            warnings.append(
                f"{snap_date}: optimization failed — carrying forward previous weights: {reason}."
            )
            new_weights: dict[int, float] = prev_weights
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
        min_order_eur = max(tc_per_trade_eur / 0.01, 200.0)  # cap TC at ~1% of order value
        trade_threshold = min_order_eur / portfolio_eur
        if prev_weights is None:
            # |w| so short opens (negative weights) are counted as trades too.
            n_trades = sum(1 for w in new_weights.values() if abs(w) >= trade_threshold)
        else:
            n_trades = sum(
                1
                for isin in set(new_weights) | set(prev_weights)
                if abs(new_weights.get(isin, 0.0) - prev_weights.get(isin, 0.0)) >= trade_threshold
            )
        tc_pct = n_trades * tc_per_trade_eur / portfolio_eur

        # Hold-period return simulation (period computed above, look-ahead-safe).
        # Universe, optimizer weights, and continuous return columns all use SID.
        # Re-ISINs no longer need a snapshot/current-ISIN translation here.
        avail = [isin for isin in new_weights if isin in period.columns]
        if avail:
            w_arr = np.array([new_weights[isin] for isin in avail])
            if is_long_short:
                # Signed weights: no invested-fraction rescaling (a dollar-neutral
                # book sums to ~0). Names without return data in a period simply do
                # not contribute. port_return_t = Σ w_i · r_{i,t}.
                port_returns = pd.Series(
                    period[avail].fillna(0.0).values @ w_arr, index=period.index
                )
            else:
                # Scale to the invested fraction: names without return data are
                # redistributed pro-rata, but a deliberate cash buffer (weights summing
                # below 1 under max_cash_weight) is preserved and earns 0% for the period.
                invested = min(1.0, sum(new_weights.values()))
                w_arr *= invested / w_arr.sum()
                port_returns = pd.Series(
                    period[avail].fillna(0.0).values @ w_arr, index=period.index
                )
        else:
            port_returns = pd.Series(0.0, index=period.index)

        # Borrow fee on the short leg: Σ|w_short| × (bps/1e4) spread annually,
        # spread across the hold period's trading days (one drag per period on day 0).
        borrow_pct = 0.0
        if is_long_short and short_borrow_bps > 0.0 and len(port_returns) > 0:
            short_notional = sum(-w for w in new_weights.values() if w < 0.0)
            borrow_pct = short_notional * (short_borrow_bps / 1e4) * (len(port_returns) / 252.0)

        if len(port_returns) > 0:
            port_returns.iloc[0] -= tc_pct + borrow_pct

        # Two-way turnover: sum of absolute weight changes (buys + sells)
        if prev_weights is not None and new_weights:
            actual_to = sum(
                abs(new_weights.get(isin, 0.0) - prev_weights.get(isin, 0.0))
                for isin in set(new_weights) | set(prev_weights)
            )
        else:
            actual_to = 1.0

        sector_weights: dict[str, float] = {}
        bm_sector_weights: dict[str, float] = {}
        industry_weights: dict[str, float] = {}
        bm_industry_weights: dict[str, float] = {}
        for isin, w in new_weights.items():
            sec = period_sectors.get(isin, "Unknown")
            ind = period_industries.get(isin, "Unknown")
            sector_weights[sec] = sector_weights.get(sec, 0.0) + w
            industry_weights[ind] = industry_weights.get(ind, 0.0) + w
        for isin, w in bm_weights.items():
            sec = period_sectors.get(isin, "Unknown")
            ind = period_industries.get(isin, "Unknown")
            bm_sector_weights[sec] = bm_sector_weights.get(sec, 0.0) + w
            bm_industry_weights[ind] = bm_industry_weights.get(ind, 0.0) + w

        period_log.append(
            {
                "snap_date": snap_date,
                "next_snap": next_snap[:10],
                "alpha_date": alpha_date,
                "universe_snapshot": uni_snap,
                "benchmark_name": benchmark_name,
                "benchmark_snapshot": bm_snap if bm_snap is not None else uni_snap,
                "weights": new_weights,
                "bm_weights": dict(bm_weights),
                "barra_date": barra_date,
                "n_trades": n_trades,
                "tc_pct": tc_pct,
                "borrow_pct": borrow_pct,
                "gross": opt_metrics.get("gross", sum(abs(w) for w in new_weights.values())),
                "net": opt_metrics.get("net", sum(new_weights.values())),
                "turnover": actual_to,
                "sector_weights": sector_weights,
                "bm_sector_weights": bm_sector_weights,
                "industry_weights": industry_weights,
                "bm_industry_weights": bm_industry_weights,
                "group_weights": group_weights(new_weights, views),
                "bm_group_weights": group_weights(bm_weights, views),
                "metrics": opt_metrics,
                "used_barra": opt_metrics.get("used_barra", False),
                "relaxed_integer": opt_metrics.get("relaxed_integer", False),
                "turnover_relaxed": opt_metrics.get("turnover_relaxed", False),
                "n_positions": opt_metrics.get("n_positions", len(new_weights)),
            }
        )
        return_parts.append(port_returns)
        prev_weights = convert_weights(
            new_weights, conversions, period.index[-1].strftime("%Y-%m-%d")
        )

    if not return_parts:
        reason = warnings[0] if warnings else "No usable holding periods."
        return {"error": f"No valid backtest periods found. {reason}", "warnings": warnings}

    return {
        "port_series": pd.concat(return_parts).sort_index(),
        "identity_key": "sid",
        "period_log": period_log,
        "warnings": warnings,
        "strategy_name": sp["name"],
        "objective": objective,
        "is_long_short": is_long_short,
        "short_borrow_bps": short_borrow_bps,
        "rebal_freq": rebal_freq,
        "universe_name": universe_name,
        "benchmark_name": benchmark_name,
        "sector_map": sector_map,
        "industry_map": industry_map,
        "label_map": label_map,
        "ticker_map": ticker_map,
        "name_map": name_map,
    }
