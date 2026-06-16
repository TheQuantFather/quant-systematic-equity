"""backtest_report.py — standalone HTML showcase of a strategy's walk-forward backtest.

Runs the shared CVXPY walk-forward engine (scripts.backtest_engine) for one active
strategy and renders a self-contained HTML report: cumulative growth vs benchmarks,
drawdown, rolling tracking error, calendar-year returns, active sector tilts, turnover,
a headline metrics table, latest holdings, and a methodology narrative.

Parameterized by strategy_id, so the same script produces the report for any active
strategy in strategy_params.xlsx (Core Active, Core Active Strict, Absolute Return).

Usage:
  python scripts/backtest_report.py core_active_strict
  python scripts/backtest_report.py core_active_strict --rebal-freq monthly --output reports/x.html

The simulation itself is NOT reimplemented here — it is the exact same engine the
Streamlit Backtester page calls, so the published numbers match the interactive tool.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))             # repo root: config, optimize_portfolio
sys.path.insert(0, str(ROOT / "scripts")) # scripts: report_utils, backtest_engine

from config import RETURNS_DB, RISK_DB                     # noqa: E402
from backtest_engine import run_optimised_backtest         # noqa: E402
from optimize_portfolio import load_strategy_params        # noqa: E402
from report_utils import CSS, fmt_pct, kpi                 # noqa: E402

RISK_FREE      = 0.04   # annualised, matches the Backtester page
PORTFOLIO_EUR  = 50_000
MAX_TURNOVER   = 0.10
TC_PER_TRADE   = 2.0    # €2 per trade (DeGiro US stocks)

# Style/model factors plotted on the active-exposure-over-time chart. All are
# cap-weighted cross-sectional z-scores (comparable units); beta and sectors are
# shown elsewhere (KPI / sector-tilt chart).
STYLE_FACTORS = [
    ("PROF001", "Profitability"), ("VAL001", "Value"), ("GRO001", "Growth"),
    ("MOM001", "Momentum"), ("SIZ001", "Size"),
]
FACTOR_PALETTE = ["#C44E52", "#4C72B0", "#2C9F4E", "#C7913A", "#8172B3"]

# Benchmark colours
COL_PORT = "#C44E52"   # strategy (accent red)
COL_BENCH = "#2C3E50"  # primary benchmark (dark slate)
COL_REF1 = "#888888"   # S&P 500 cap-weight reference
COL_REF2 = "#bbbbbb"   # S&P 500 equal-weight reference
COL_POS  = "#2C9F4E"
COL_NEG  = "#C44E52"

REPORTS_DIR = ROOT / "reports"


# ---------------------------------------------------------------------------
# Metrics (standard return statistics — formatting layer for the report)
# ---------------------------------------------------------------------------

def ann_stats(ret: pd.Series) -> dict:
    ret = ret.dropna()
    if len(ret) < 10:
        return {}
    total   = (1 + ret).prod() - 1
    n_years = len(ret) / 252
    ann_ret = (1 + total) ** (1 / max(n_years, 1e-6)) - 1
    ann_vol = ret.std() * np.sqrt(252)
    cum     = (1 + ret).cumprod()
    max_dd  = (cum / cum.cummax() - 1).min()
    sharpe  = (ann_ret - RISK_FREE) / ann_vol if ann_vol > 0 else np.nan
    downside = ret[ret < 0].std() * np.sqrt(252)
    sortino = (ann_ret - RISK_FREE) / downside if downside > 0 else np.nan
    calmar  = ann_ret / abs(max_dd) if max_dd < 0 else np.nan
    monthly = ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    return {
        "total": total, "ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe,
        "sortino": sortino, "calmar": calmar, "max_dd": max_dd,
        "win": float((ret > 0).mean()),
        "pos_months": float((monthly > 0).mean()) if len(monthly) else np.nan,
        "n_years": n_years,
    }


def active_stats(r: pd.Series, b: pd.Series) -> dict:
    common = r.index.intersection(b.index)
    r = r.loc[common].dropna()
    b = b.loc[common].dropna()
    common = r.index.intersection(b.index)
    r, b = r.loc[common], b.loc[common]
    if len(common) < 63:
        return {}
    active  = r - b
    ann_act = active.mean() * 252
    te      = active.std() * np.sqrt(252)
    ir      = ann_act / te if te > 0 else np.nan
    beta    = r.cov(b) / b.var() if b.var() > 0 else np.nan
    return {"ann_act": ann_act, "te": te, "ir": ir, "beta": beta}


def load_benchmark_series(index_name: str) -> pd.Series:
    with sqlite3.connect(RETURNS_DB) as c:
        df = pd.read_sql_query(
            "SELECT date, total_return FROM benchmark_returns "
            "WHERE index_name = ? AND total_return IS NOT NULL ORDER BY date",
            c, params=(index_name,),
        )
    if df.empty:
        return pd.Series(dtype=float)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["total_return"]


# ---------------------------------------------------------------------------
# Charts (plotly → embedded HTML fragments)
# ---------------------------------------------------------------------------

def _html(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})


def chart_growth_drawdown(port: pd.Series, benches: dict[str, tuple[pd.Series, str]],
                          bench_label: str) -> str:
    start = port.index.min()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.05,
                        subplot_titles=("Growth of 100 (net of modelled trading costs)",
                                        "Drawdown vs running peak"))
    pidx = (1 + port).cumprod() * 100
    fig.add_trace(go.Scatter(x=pidx.index, y=pidx.values, name="Strategy",
                             line=dict(color=COL_PORT, width=2.4)), row=1, col=1)
    for series, color in benches.values():
        b = series[series.index >= start]
        if b.empty:
            continue
        bidx = (1 + b).cumprod() * 100
        lbl = [k for k, v in benches.items() if v[0] is series][0]
        fig.add_trace(go.Scatter(x=bidx.index, y=bidx.values, name=lbl,
                                 line=dict(color=color, width=1.5, dash="dot")), row=1, col=1)
    dd = pidx / pidx.cummax() - 1
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values * 100, name="Drawdown",
                             line=dict(color=COL_NEG, width=1.4), fill="tozeroy",
                             fillcolor="rgba(196,78,82,0.15)", showlegend=False), row=2, col=1)
    fig.update_yaxes(title="Index", row=1, col=1)
    fig.update_yaxes(title="DD (%)", ticksuffix="%", row=2, col=1)
    fig.update_layout(height=540, template="plotly_white",
                      legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
                      margin=dict(l=50, r=20, t=60, b=30))
    return _html(fig)


def chart_rolling_te(port: pd.Series, bench: pd.Series, bench_label: str, window: int = 63) -> str:
    common = port.index.intersection(bench.index)
    active = (port.loc[common] - bench.loc[common]).dropna()
    te = active.rolling(window).std() * np.sqrt(252) * 100
    fig = go.Figure(go.Scatter(x=te.index, y=te.values, name="Rolling TE",
                               line=dict(color=COL_BENCH, width=1.8)))
    fig.update_layout(height=320, template="plotly_white",
                      title=f"Rolling {window}-day annualised tracking error vs {bench_label}",
                      yaxis=dict(title="TE (%)", ticksuffix="%"),
                      margin=dict(l=50, r=20, t=55, b=30))
    return _html(fig)


def chart_annual_returns(port: pd.Series, bench: pd.Series, bench_label: str) -> str:
    def by_year(s: pd.Series) -> pd.Series:
        return s.groupby(s.index.year).apply(lambda x: (1 + x).prod() - 1)
    p = by_year(port)
    b = by_year(bench.loc[bench.index.intersection(port.index)])
    years = sorted(set(p.index) | set(b.index))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=years, y=[p.get(y, np.nan) * 100 for y in years], name="Strategy",
                         marker_color=COL_PORT))
    fig.add_trace(go.Bar(x=years, y=[b.get(y, np.nan) * 100 for y in years], name=bench_label,
                         marker_color=COL_BENCH))
    fig.add_hline(y=0, line=dict(color="#444", width=1))
    fig.update_layout(height=360, template="plotly_white", barmode="group",
                      title="Calendar-year total return",
                      yaxis=dict(title="Return (%)", ticksuffix="%"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                      margin=dict(l=50, r=20, t=60, b=30))
    return _html(fig)


def chart_sector_tilts(period: dict) -> str:
    sw, bw = period["sector_weights"], period["bm_sector_weights"]
    sectors = sorted(set(sw) | set(bw), key=lambda s: (sw.get(s, 0) - bw.get(s, 0)))
    active = [(sw.get(s, 0) - bw.get(s, 0)) * 100 for s in sectors]
    colors = [COL_POS if a >= 0 else COL_NEG for a in active]
    fig = go.Figure(go.Bar(x=active, y=sectors, orientation="h", marker_color=colors,
                           text=[f"{a:+.1f}" for a in active], textposition="auto"))
    fig.add_vline(x=0, line=dict(color="#444", width=1))
    fig.update_layout(height=380, template="plotly_white",
                      title="Active sector tilts — latest rebalance (portfolio − benchmark)",
                      xaxis=dict(title="Active weight (pp)", ticksuffix="pp"),
                      margin=dict(l=20, r=20, t=60, b=30))
    return _html(fig)


def chart_turnover(period_log: list[dict]) -> str:
    rebs  = period_log[1:]   # first period is the initial build (turnover = 100%)
    dates = pd.to_datetime([p["snap_date"] for p in rebs])
    to    = [p["turnover"] * 100 for p in rebs]
    avg   = float(np.mean(to)) if to else 0.0
    fig = go.Figure(go.Bar(
        x=dates, y=to, marker_color=COL_BENCH,
        hovertemplate="%{x|%Y-%m-%d}<br>two-way turnover %{y:.1f}%<extra></extra>"))
    fig.add_hline(y=avg, line=dict(color=COL_PORT, width=1.4, dash="dash"),
                  annotation_text=f"avg {avg:.0f}%", annotation_position="top left")
    fig.update_layout(height=320, template="plotly_white",
                      title="Two-way turnover per rebalance (buys + sells, excludes initial build)",
                      yaxis=dict(title="Turnover (%)", ticksuffix="%", rangemode="tozero"),
                      xaxis=dict(title=None),
                      margin=dict(l=55, r=20, t=55, b=40))
    return _html(fig)


def active_factor_exposures(period_log: list[dict]) -> tuple[list, dict[str, list]]:
    """Per rebalance, cap-weighted active exposure (portfolio − benchmark) on each
    style/model factor, read from the Barra factor_exposures table at that period's
    barra_date. Returns (dates, {factor_label: [values]})."""
    fids   = [f for f, _ in STYLE_FACTORS]
    ph     = ",".join("?" * len(fids))
    cache: dict[str, dict[str, dict[str, float]]] = {}
    dates: list = []
    series: dict[str, list[float]] = {lbl: [] for _, lbl in STYLE_FACTORS}
    for pl in period_log:
        bd = pl["barra_date"]
        if bd is None:
            continue
        if bd not in cache:
            with sqlite3.connect(RISK_DB) as c:
                rows = c.execute(
                    f"SELECT factor_id, security_id, exposure FROM factor_exposures "
                    f"WHERE snapshot_date = ? AND factor_id IN ({ph})", (bd, *fids)
                ).fetchall()
            d: dict[str, dict[str, float]] = {f: {} for f in fids}
            for fid, sid, e in rows:
                d[fid][sid] = e
            cache[bd] = d
        exp = cache[bd]
        w, bw = pl["weights"], pl["bm_weights"]
        all_ids = set(w) | set(bw)
        dates.append(pd.Timestamp(pl["snap_date"]))
        for fid, lbl in STYLE_FACTORS:
            e = exp[fid]
            series[lbl].append(sum((w.get(i, 0.0) - bw.get(i, 0.0)) * e.get(i, 0.0) for i in all_ids))
    return dates, series


def chart_factor_exposures(dates: list, series: dict[str, list]) -> str:
    fig = go.Figure()
    for (lbl, vals), col in zip(series.items(), FACTOR_PALETTE):
        fig.add_trace(go.Scatter(x=dates, y=vals, name=lbl, mode="lines",
                                 line=dict(width=1.9, color=col)))
    fig.add_hline(y=0, line=dict(color="#444", width=1))
    fig.update_layout(height=400, template="plotly_white",
                      title="Active factor exposures vs benchmark over time (cap-weighted model z-scores)",
                      yaxis=dict(title="Active exposure (z)"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
                      margin=dict(l=55, r=20, t=60, b=30))
    return _html(fig)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def metrics_table(cols: dict[str, dict]) -> str:
    """cols: {column_label: ann_stats dict}. First column is the strategy."""
    rows = [
        ("Total return",     lambda m: fmt_pct(m.get("total"), 1, True)),
        ("Annualised return", lambda m: fmt_pct(m.get("ann_ret"), 1, True)),
        ("Annualised vol",   lambda m: fmt_pct(m.get("ann_vol"), 1)),
        ("Sharpe ratio",     lambda m: f"{m.get('sharpe', float('nan')):.2f}"),
        ("Sortino ratio",    lambda m: f"{m.get('sortino', float('nan')):.2f}"),
        ("Calmar ratio",     lambda m: f"{m.get('calmar', float('nan')):.2f}"),
        ("Max drawdown",     lambda m: fmt_pct(m.get("max_dd"), 1)),
        ("Daily win rate",   lambda m: fmt_pct(m.get("win"), 1)),
        ("Positive months",  lambda m: fmt_pct(m.get("pos_months"), 0)),
    ]
    heads = "".join(f"<th>{c}</th>" for c in cols)
    body = ""
    for label, fn in rows:
        cells = "".join(f"<td>{fn(m)}</td>" for m in cols.values())
        body += f"<tr><td>{label}</td>{cells}</tr>"
    return (f'<div class="table-wrap"><table><thead><tr><th>Metric</th>{heads}</tr></thead>'
            f"<tbody>{body}</tbody></table></div>")


def holdings_table(period: dict, result: dict, top_n: int = 15) -> str:
    weights = period["weights"]
    bm      = period["bm_weights"]
    tmap, nmap, smap = result["ticker_map"], result["name_map"], result["sector_map"]
    rows = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    body = ""
    for isin, w in rows:
        tkr = tmap.get(isin, "") or isin[:6]
        nm  = (nmap.get(isin) or "")[:38]
        sec = smap.get(isin, "—")
        act = (w - bm.get(isin, 0.0)) * 100
        acls = "pos" if act >= 0 else "neg"
        body += (f"<tr><td>{tkr}</td><td>{nm}</td><td>{sec}</td>"
                 f"<td>{w*100:.2f}%</td><td class='{acls}'>{act:+.2f}pp</td></tr>")
    return (f'<div class="table-wrap"><table><thead><tr>'
            f"<th>Ticker</th><th>Company</th><th>Sector</th><th>Weight</th><th>Active</th>"
            f"</tr></thead><tbody>{body}</tbody></table></div>")


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

def _constraint_summary(constraints: dict) -> str:
    keys = [
        ("max_active_risk", "Active-risk cap", lambda v: f"{float(v)*100:.0f}%"),
        ("max_stock_active_weight", "Max stock active wt", lambda v: f"±{float(v)*100:.1f}pp"),
        ("max_sector_active_weight", "Max sector active wt", lambda v: f"±{float(v)*100:.1f}pp"),
        ("max_position", "Max issuer weight", lambda v: f"{float(v)*100:.0f}%"),
        ("max_positions", "Max positions", lambda v: f"{int(float(v))}"),
    ]
    items = []
    for key, label, fmt in keys:
        if key in constraints:
            try:
                items.append(f"<li><strong>{label}:</strong> {fmt(constraints[key])}</li>")
            except (ValueError, TypeError):
                pass
    return "<ul>" + "".join(items) + "</ul>" if items else ""


def _cadence_label(period_log: list[dict]) -> str:
    snaps = pd.to_datetime([pl["snap_date"] for pl in period_log])
    if len(snaps) < 2:
        return f"{len(period_log)} rebalance"
    gap = int(np.median(np.diff(snaps).astype("timedelta64[D]").astype(int)))
    if   gap <= 12:  word = "weekly"
    elif gap <= 45:  word = "monthly"
    elif gap <= 100: word = "quarterly"
    else:            word = f"~{gap}-day"
    return f"{word} ({len(period_log)} rebalances)"


def build_html(strategy_id: str, result: dict, sp: dict, solver_used: str,
               port: pd.Series, bench_primary: pd.Series, bench_label: str,
               extra_benches: dict, charts: dict) -> str:
    p_stats = ann_stats(port)
    b_stats = ann_stats(bench_primary.loc[bench_primary.index.intersection(port.index)])
    a_stats = active_stats(port, bench_primary)

    # Extra reference columns for the metrics table
    metric_cols = {sp["name"]: p_stats, bench_label: b_stats}
    for lbl, series in extra_benches.items():
        metric_cols[lbl] = ann_stats(series.loc[series.index.intersection(port.index)])

    period_log = result["period_log"]
    n_rebal    = len(period_log)
    avg_pos    = np.mean([pl["n_positions"] for pl in period_log])
    avg_to     = np.mean([pl["turnover"] for pl in period_log[1:]]) if n_rebal > 1 else float("nan")
    start, end = port.index.min(), port.index.max()
    span_yrs   = p_stats.get("n_years", 0.0)

    kpi_html = "".join([
        kpi("Ann. return", fmt_pct(p_stats.get("ann_ret"), 1, True),
            f"vs {fmt_pct(b_stats.get('ann_ret'), 1, True)} {bench_label}",
            cls="pos" if (p_stats.get("ann_ret") or 0) > (b_stats.get("ann_ret") or 0) else "neg"),
        kpi("Ann. volatility", fmt_pct(p_stats.get("ann_vol"), 1),
            f"{fmt_pct(b_stats.get('ann_vol'), 1)} benchmark"),
        kpi("Sharpe", f"{p_stats.get('sharpe', float('nan')):.2f}",
            f"rf {RISK_FREE*100:.0f}%",
            cls="pos" if (p_stats.get("sharpe") or 0) > 0.5 else ""),
        kpi("Max drawdown", fmt_pct(p_stats.get("max_dd"), 1),
            f"{fmt_pct(b_stats.get('max_dd'), 1)} benchmark"),
        kpi("Information ratio", f"{a_stats.get('ir', float('nan')):.2f}",
            f"TE {fmt_pct(a_stats.get('te'), 1)} · active {fmt_pct(a_stats.get('ann_act'), 1, True)}",
            cls="pos" if (a_stats.get("ir") or 0) > 0 else "neg"),
        kpi("Avg turnover", fmt_pct(avg_to, 0),
            f"{avg_pos:.0f} avg positions · {n_rebal} rebalances"),
    ])

    objective_h = {"maximize_alpha": "Maximise alpha (benchmark-relative)",
                   "maximize_sharpe": "Maximise Sharpe (Charnes-Cooper)",
                   "minimize_variance": "Minimise variance"}.get(sp["objective"], sp["objective"])

    cadence = _cadence_label(period_log)
    warn_n = len([w for w in result["warnings"] if "relaxed" in w or "failed" in w])
    warn_line = (f" {warn_n} period(s) required turnover relaxation or carry-forward."
                 if warn_n else " No period required turnover relaxation or fallback.")
    relaxed_int = any(pl.get("relaxed_integer") for pl in period_log)
    relax_note = ("" if not relaxed_int else
                  f" Under {solver_used} (open-source) the integer position-count constraint is "
                  f"relaxed to its continuous form.")

    return f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><title>{sp['name']} — backtest showcase</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
{CSS}
<script charset="utf-8" src="https://cdn.plot.ly/plotly-3.5.0.min.js" integrity="sha256-fHbNLP+GlIXN+efbQec78UkemUz3NJp7UmfGxC1tNxs=" crossorigin="anonymous"></script>
</head><body>
<div class="wrap">
<header>
  <h1>{sp['name']} — walk-forward backtest</h1>
  <div class="sub">{objective_h} · benchmark {bench_label} ·
       {start:%b %Y}–{end:%b %Y} ({span_yrs:.1f}Y) · {cadence} ·
       {solver_used} solver · generated {datetime.now():%Y-%m-%d %H:%M}</div>
  <div class="biz">Point-in-time walk-forward: at each rebalance the optimiser sees only data
       published by that date (alpha z-scores, Barra risk model, prices, and index membership),
       then holds the resulting weights until the next rebalance. Net of €{TC_PER_TRADE:.0f}/trade
       modelled costs.{warn_line}</div>
</header>

<div class="kpi-grid">{kpi_html}</div>

<section>
  <h2>Growth &amp; drawdown</h2>
  <div class="lead">Strategy vs {bench_label}{(' and ' + ', '.join(extra_benches)) if extra_benches else ''}.
       Indexed to 100 at the first rebalance; drawdown is measured against the strategy's own running peak.</div>
  {charts['growth']}
</section>

<section>
  <h2>Headline metrics</h2>
  <div class="lead">Risk-free rate {RISK_FREE*100:.0f}% for Sharpe/Sortino. Active statistics are
       measured against {bench_label}.</div>
  {metrics_table(metric_cols)}
  <p class="note"><strong>Active vs {bench_label}:</strong>
     active return {fmt_pct(a_stats.get('ann_act'), 1, True)} ·
     tracking error {fmt_pct(a_stats.get('te'), 1)} ·
     information ratio {a_stats.get('ir', float('nan')):.2f} ·
     beta {a_stats.get('beta', float('nan')):.2f}.</p>
</section>

<section>
  <h2>Calendar-year returns</h2>
  <div class="lead">Year-by-year total return, strategy vs {bench_label}. Partial first/last years
       reflect the backtest window, not full calendar years.</div>
  {charts['annual']}
</section>

<section>
  <h2>Active factor exposures over time</h2>
  <div class="lead">Cap-weighted active tilt (portfolio − benchmark) on each alpha factor at every
       rebalance, read from the Barra exposure matrix. Positive = the portfolio leans into that
       factor versus {bench_label}; values are in cross-sectional z-score units.</div>
  {charts['factors']}
</section>

<section>
  <h2>Tracking error &amp; turnover</h2>
  <div class="lead">How tightly the strategy hugs {bench_label} over time, and how much it trades to
       stay there. Turnover is two-way (buys + sells) and excludes the initial portfolio build.</div>
  {charts['te']}
  <div style="height:14px;"></div>
  {charts['turnover']}
</section>

<section>
  <h2>Latest positioning</h2>
  <div class="lead">Active sector tilts and the largest holdings as of the final rebalance
       ({period_log[-1]['snap_date']}). Active = portfolio weight − benchmark weight.</div>
  {charts['sectors']}
  <h3 style="margin-top:14px; font-size:14px;">Top {min(15, len(period_log[-1]['weights']))} holdings</h3>
  {holdings_table(period_log[-1], result)}
</section>

<section>
  <h2>Methodology</h2>
  <div class="lead">How the backtest is constructed.</div>
  <p><strong>Signal &amp; objective.</strong> Alpha is the {sp['name']} blend
     ({', '.join(sp['alpha_weights'])}). The optimiser solves
     <em>{objective_h}</em> with the {solver_used} solver subject to the active-space constraints
     below, re-solving at every model snapshot ({cadence}).{relax_note}</p>
  <p><strong>Risk model.</strong> A from-scratch Barra-style factor model (market + GICS sectors +
     beta + base-model factors) supplies the covariance used for the active-risk and volatility
     constraints; Ledoit-Wolf is the fallback. Both are point-in-time.</p>
  <p><strong>Constraints.</strong></p>
  {_constraint_summary(sp['constraints'])}
  <p><strong>Trading frictions.</strong> A trade is only counted when its order value clears a
     minimum size (so trivial weight tweaks aren't charged), at €{TC_PER_TRADE:.0f}/trade. When a
     tight turnover cap collides with shifting risk loadings, the cap is relaxed 1.5×/2.25× before
     being dropped — recorded per period.</p>
  <p class="note"><strong>Caveats.</strong> Results are gross of management fees, financing and
     slippage beyond the modelled per-trade cost; deliberate cash buffers earn 0%; ad-hoc research
     snapshots are excluded so rebalances fall on the scheduled grid. Past performance does not
     indicate future results.</p>
</section>

<footer>
  Generated by <code>scripts/backtest_report.py {strategy_id}</code> on
  {datetime.now():%Y-%m-%d}. Self-contained walk-forward backtest from a private systematic-equity
  framework. For research and illustration only — not investment advice.
</footer>
</div>
</body></html>"""


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _progress(i: int, n: int, snap: str) -> None:
    print(f"  [{i+1:>2}/{n}] {snap}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a backtest showcase HTML for a strategy.")
    ap.add_argument("strategy_id", nargs="?", default="core_active_strict",
                    help="Active strategy_id from strategy_params.xlsx")
    ap.add_argument("--rebal-freq", choices=["quarterly", "monthly"], default="quarterly")
    ap.add_argument("--cadences", default="monthly",
                    help="Comma-separated snapshot cadences to rebalance on (canonical "
                         "snapshot_schedule). Default 'monthly' rebalances only on month-end "
                         "snapshots, ignoring recent weekly/adhoc snapshots. Use 'monthly,weekly' "
                         "to include weekly, or 'all' for every snapshot.")
    ap.add_argument("--portfolio", type=float, default=PORTFOLIO_EUR)
    ap.add_argument("--max-turnover", type=float, default=MAX_TURNOVER)
    ap.add_argument("--solver", default="CLARABEL",
                    help="Override the strategy's solver. CLARABEL (default) is open-source and "
                         "relaxes any integer position-count constraints to continuous.")
    ap.add_argument("--output", default=None, help="Output HTML path")
    args = ap.parse_args()

    strategies = load_strategy_params(args.strategy_id)
    if not strategies:
        sys.exit(f"No active strategy '{args.strategy_id}' found in strategy_params.xlsx")
    sp = strategies[0]

    benchmark_name = sp["benchmark_index"] or "sp500"
    if sp["universe_index"]:
        universe_name = sp["universe_index"]
    elif sp["investable_universe"] == "benchmark_only":
        universe_name = benchmark_name
    else:
        universe_name = "sp500"
    solver = args.solver or "CLARABEL"
    cadences = (None if args.cadences.strip().lower() == "all"
                else {c.strip() for c in args.cadences.split(",") if c.strip()})

    print(f"Running {args.rebal_freq} backtest for '{sp['name']}' "
          f"(universe={universe_name}, benchmark={benchmark_name}, solver={solver}, "
          f"cadences={args.cadences})…")
    result = run_optimised_backtest(
        strategy_id=args.strategy_id,
        portfolio_eur=args.portfolio,
        max_turnover=args.max_turnover,
        tc_per_trade_eur=TC_PER_TRADE,
        benchmark_name=benchmark_name,
        universe_name=universe_name,
        rebal_freq=args.rebal_freq,
        solver=solver,
        rebalance_cadences=cadences,
        progress_cb=_progress,
    )
    if "error" in result:
        sys.exit(f"Backtest failed: {result['error']}")

    port = result["port_series"].dropna()

    # Benchmark series: primary (the strategy's benchmark) + cap- and equal-weight references
    bench_labels = {
        "sp500_3pct_capped": "S&P 500 3% Capped",
        "sp500": "S&P 500",
        "sp500_equal_weight": "S&P 500 Equal-Weight",
        "russell_1000": "Russell 1000",
    }
    bench_primary = load_benchmark_series(benchmark_name)
    bench_label   = bench_labels.get(benchmark_name, benchmark_name)
    if bench_primary.empty:
        sys.exit(f"No benchmark_returns series for '{benchmark_name}'.")

    extra_benches: dict[str, pd.Series] = {}
    for ref in ("sp500", "sp500_equal_weight"):
        if ref == benchmark_name:
            continue
        s = load_benchmark_series(ref)
        if not s.empty:
            extra_benches[bench_labels.get(ref, ref)] = s

    # Build chart fragments
    growth_benches = {bench_label: (bench_primary, COL_BENCH)}
    for i, (lbl, s) in enumerate(extra_benches.items()):
        growth_benches[lbl] = (s, COL_REF1 if i == 0 else COL_REF2)
    fx_dates, fx_series = active_factor_exposures(result["period_log"])
    charts = {
        "growth":  chart_growth_drawdown(port, growth_benches, bench_label),
        "annual":  chart_annual_returns(port, bench_primary, bench_label),
        "te":      chart_rolling_te(port, bench_primary, bench_label),
        "turnover": chart_turnover(result["period_log"]),
        "sectors": chart_sector_tilts(result["period_log"][-1]),
        "factors": chart_factor_exposures(fx_dates, fx_series),
    }

    html = build_html(args.strategy_id, result, sp, solver, port, bench_primary, bench_label,
                      extra_benches, charts)
    out = Path(args.output) if args.output else REPORTS_DIR / f"{args.strategy_id}_backtest.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"\n✓ Wrote {out}  ({out.stat().st_size/1024:.0f} KB, "
          f"{len(result['period_log'])} rebalances, "
          f"{port.index.min():%Y-%m-%d}→{port.index.max():%Y-%m-%d})")


if __name__ == "__main__":
    main()
