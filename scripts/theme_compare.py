#!/usr/bin/env python3
"""
theme_compare.py — Cross-theme momentum vs valuation comparison report.

Sibling to theme_report.py. Where theme_report builds ONE basket's deep
dive, this loads several thematic baskets at once and renders a single
comparison HTML focused on the two questions a tactical allocator asks:
where is each theme on momentum, and where is it on valuation — i.e. which
sub-themes are heated/extended and which look like a buying opportunity.

Reuses theme_report's data-assembly functions (load_basket_members,
compute_weights, aggregate_models, aggregate_ltm, basket_return_series,
aggregate_barra) and report_utils plumbing so signals never drift from the
single-name / single-theme reports.

Run from project root:
    python scripts/theme_compare.py
(baskets are defined in BASKETS below; edit there to change membership.)
"""
from __future__ import annotations

import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))
from report_utils import (  # noqa: E402
    REPORTS_DIR, RISK_DB, UNIV_DB, RETURNS_DB, CSS, fmt_pct, fmt_money, kpi, placeholder,
    perf_block, load_benchmarks, load_returns,
)
from theme_report import (  # noqa: E402
    load_basket_members, compute_weights, aggregate_ltm,
    basket_return_series, aggregate_models, aggregate_barra,
)


# ---------------------------------------------------------------------------
# Basket definitions — AI sub-themes (R1000-covered names only)
# ---------------------------------------------------------------------------

BASKETS: list[dict] = [
    {"slug": "semis", "name": "Semiconductors", "color": "#4C72B0",
     "tickers": ["NVDA", "AVGO", "AMD", "MU", "MRVL", "LRCX", "AMAT", "KLAC", "QCOM", "MPWR"]},
    {"slug": "ai_industrials", "name": "AI Industrials (power + cooling)", "color": "#2C9F4E",
     "tickers": ["VRT", "ETN", "GEV", "PWR", "FIX", "EME", "NVT", "HUBB", "GNRC", "CMI"]},
    {"slug": "ai_power", "name": "AI Power / Electrification", "color": "#C7913A",
     "tickers": ["CEG", "VST", "NRG", "TLN", "GEV", "PWR", "ETN", "SO"]},
    {"slug": "cyber", "name": "Cybersecurity", "color": "#C44E52",
     "tickers": ["CRWD", "PANW", "ZS", "FTNT", "OKTA", "GEN", "S", "RBRK", "NET"]},
]

WEIGHT = "cap"
EW_START = "2025-01-01"  # cumulative return comparison starts here (equal-weighted)


def equal_weight_cum(members: list[dict], start: str) -> pd.DataFrame | None:
    """Equal-weighted basket total-return index (base 100) from `start`.

    Independent of the cap-weighted metrics everywhere else in the report — this
    answers "if I'd held each name equally since `start`, how did the basket
    compound?" rather than "where does the cap-weighted basket sit today?".
    """
    ew = {m["ticker"]: 1.0 / len(members) for m in members}
    ret = basket_return_series(members, ew)
    if ret.empty:
        return None
    ret = ret[ret["date"] >= start].copy()
    if ret.empty:
        return None
    idx = (1 + ret["total_return"].fillna(0)).cumprod()
    ret["cum"] = idx / idx.iloc[0] * 100.0  # base 100 on first trading day >= start
    return ret[["date", "cum"]].reset_index(drop=True)


def market_ex_ai_cum(exclude_isins: set[str], start: str,
                     snapshot: str = "2024-12-31") -> pd.DataFrame | None:
    """Equal-weighted S&P 500 total-return index EXCLUDING the AI-basket names,
    base 100 from `start`. Daily-rebalanced equal weight (mean of each day's
    constituent returns) so it is robust to any mid-period delistings.

    Membership is point-in-time: S&P 500 constituents as of `snapshot`.
    """
    import sqlite3
    with sqlite3.connect(UNIV_DB) as u:
        spx = [r[0] for r in u.execute(
            "SELECT DISTINCT isin FROM universe_snapshots "
            "WHERE index_name='sp500' AND snapshot_date=?", (snapshot,)).fetchall()]
    keep = [i for i in spx if i not in exclude_isins]
    if not keep:
        return None
    with sqlite3.connect(RETURNS_DB) as c:
        df = pd.read_sql_query(
            f"SELECT isin, date, total_return FROM returns "
            f"WHERE date >= ? AND isin IN ({','.join(['?']*len(keep))})",
            c, params=[start] + keep, parse_dates=["date"])
    if df.empty:
        return None
    wide = df.pivot_table(index="date", columns="isin", values="total_return")
    ew = wide.mean(axis=1, skipna=True)        # equal weight, daily rebalanced
    idx = (1 + ew.fillna(0)).cumprod()
    out = pd.DataFrame({"date": idx.index, "cum": idx.values / idx.values[0] * 100.0})
    out.attrs["n_keep"] = len(keep)
    out.attrs["n_excl"] = len([i for i in spx if i in exclude_isins])
    return out


# ---------------------------------------------------------------------------
# Per-basket metric assembly
# ---------------------------------------------------------------------------

def build_theme(spec: dict) -> dict:
    """Load one basket and reduce it to the comparison metric set."""
    print(f"[{spec['slug']}] loading {len(spec['tickers'])} names ...")
    members = load_basket_members(spec["tickers"])
    if not members:
        raise SystemExit(f"No members for {spec['slug']}")
    weights = compute_weights(members, WEIGHT)
    members.sort(key=lambda m: weights[m["ticker"]], reverse=True)

    agg = aggregate_ltm(members, weights)
    model_summary, model_pivot = aggregate_models(members, weights)
    bz = model_summary["basket_z"]

    ret = basket_return_series(members, weights)
    perf = perf_block(ret) if not ret.empty else {}
    ew_cum = equal_weight_cum(members, EW_START)

    barra_df, idio_vol, _ = aggregate_barra(members, weights)
    beta_row = barra_df[barra_df["factor_id"] == "beta_60d"]
    beta = float(beta_row["exposure"].iloc[0]) if not beta_row.empty else float("nan")

    ni = agg["cur"].get("NetIncome", 0.0)
    pe = agg["mcap_total"] / ni if ni and ni > 0 else None

    eff_n = 1.0 / sum(w ** 2 for w in weights.values())
    mdf = agg["members_df"]
    top3 = mdf.nlargest(3, "Weight %")["Weight %"].sum() if len(mdf) >= 3 else 100.0

    # per-name value/momentum for dispersion scatter
    pn = model_pivot.reset_index()[["ticker"]].copy()
    pn["Value"] = model_pivot["Value"].values if "Value" in model_pivot else np.nan
    pn["Momentum"] = model_pivot["Momentum"].values if "Momentum" in model_pivot else np.nan
    pn = pn.merge(mdf.reset_index()[["Ticker", "Company", "Mkt Cap ($B)"]],
                  left_on="ticker", right_on="Ticker", how="left")

    return {
        "slug": spec["slug"], "name": spec["name"], "color": spec["color"],
        "members": members, "weights": weights,
        "resolved": [m["ticker"] for m in members],
        "n": len(members), "mcap_total": agg["mcap_total"],
        "val_z": bz.get("Value", float("nan")),
        "mom_z": bz.get("Momentum", float("nan")),
        "alp_z": bz.get("Alpha", float("nan")),
        "gro_z": bz.get("Growth", float("nan")),
        "prof_z": bz.get("Profitability", float("nan")),
        "lvol_z": bz.get("Low Volatility", float("nan")),
        "shi_z": bz.get("Short Interest", float("nan")),
        "size_z": bz.get("Size", float("nan")),
        "ps": agg["aggregate_ps"], "pe": pe,
        "rev_yoy": agg["weighted_rev_yoy"],
        "gross_margin": agg["weighted_gross_margin"],
        "op_margin": agg["weighted_op_margin"],
        "ret_3m": perf.get("3m"), "ret_6m": perf.get("6m"),
        "ret_ytd": perf.get("ytd"), "ret_1y": perf.get("1y"),
        "vol_1y": perf.get("vol_1y"), "max_dd": perf.get("max_dd_1y"),
        "off_52wh": perf.get("off_52wh"), "beta": beta, "idio_vol": idio_vol,
        "eff_n": eff_n, "top3": top3,
        "snap": model_summary["snap"], "per_name": pn,
        "ew_cum": ew_cum,
    }


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _quadrant_bg(fig: go.Figure, xr: tuple, yr: tuple, labels: dict) -> None:
    """Shade the four quadrants around (0,0) and label them."""
    x0, x1 = xr
    y0, y1 = yr
    fig.add_shape(type="rect", x0=0, x1=x1, y0=0, y1=y1, line_width=0,
                  fillcolor="rgba(44,159,78,0.06)", layer="below")   # mom+ val+  sweet
    fig.add_shape(type="rect", x0=0, x1=x1, y0=y0, y1=0, line_width=0,
                  fillcolor="rgba(196,78,82,0.06)", layer="below")   # mom+ val-  heated
    fig.add_shape(type="rect", x0=x0, x1=0, y0=0, y1=y1, line_width=0,
                  fillcolor="rgba(76,114,176,0.05)", layer="below")  # mom- val+  contrarian
    fig.add_shape(type="rect", x0=x0, x1=0, y0=y0, y1=0, line_width=0,
                  fillcolor="rgba(120,120,120,0.05)", layer="below")  # mom- val- avoid
    fig.add_hline(y=0, line=dict(color="#999", width=1, dash="dot"))
    fig.add_vline(x=0, line=dict(color="#999", width=1, dash="dot"))
    for (qx, qy), (txt, anc) in labels.items():
        fig.add_annotation(x=qx, y=qy, text=txt, showarrow=False,
                           font=dict(size=11, color="#888"),
                           xanchor=anc[0], yanchor=anc[1])


def chart_quadrant(themes: list[dict]) -> str:
    xs = [t["mom_z"] for t in themes]
    ys = [t["val_z"] for t in themes]
    xpad = max(0.25, max(abs(v) for v in xs) * 1.35)
    ypad = max(0.25, max(abs(v) for v in ys) * 1.35)
    xr, yr = (-xpad, xpad), (-ypad, ypad)
    fig = go.Figure()
    _quadrant_bg(fig, xr, yr, {
        (xr[1], yr[1]): ("trend + value", ("right", "top")),
        (xr[1], yr[0]): ("heated / expensive", ("right", "bottom")),
        (xr[0], yr[1]): ("cheap, out of favour", ("left", "top")),
        (xr[0], yr[0]): ("weak + expensive", ("left", "bottom")),
    })
    sizes = [max(26, min(70, math.sqrt(t["mcap_total"] / 1e9) * 2.2)) for t in themes]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        text=[t["name"].split(" (")[0] for t in themes],
        textposition="top center", textfont=dict(size=12, color="#333"),
        marker=dict(size=sizes, color=[t["color"] for t in themes],
                    line=dict(color="#fff", width=2), opacity=0.9),
        hovertext=[f"{t['name']}<br>Momentum z {t['mom_z']:+.2f}<br>Value z {t['val_z']:+.2f}"
                   for t in themes],
        hoverinfo="text", showlegend=False,
    ))
    fig.update_layout(
        height=520, template="plotly_white",
        title="Where each AI sub-theme sits — Momentum vs Value (cap-weighted basket z-scores)",
        xaxis=dict(title="◄ weaker   MOMENTUM z   stronger ►", range=list(xr), zeroline=False),
        yaxis=dict(title="◄ expensive   VALUE z   cheaper ►", range=list(yr), zeroline=False),
        margin=dict(l=60, r=30, t=60, b=50),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})


def chart_cumulative(themes: list[dict], bm: pd.DataFrame, start: str,
                     exai: pd.DataFrame | None = None) -> str:
    """Equal-weighted cumulative total-return lines, base 100 from `start`,
    with an S&P 500 reference and an equal-weight S&P 500 ex-AI reference."""
    fig = go.Figure()
    for t in themes:
        c = t.get("ew_cum")
        if c is None or c.empty:
            continue
        final = float(c["cum"].iloc[-1])
        fig.add_trace(go.Scatter(
            x=c["date"].tolist(), y=c["cum"].tolist(), mode="lines",
            name=t["name"].split(" (")[0], line=dict(color=t["color"], width=2.4),
            hovertemplate="%{x|%b %Y}<br>%{y:.0f} (base 100)<extra></extra>",
        ))
        fig.add_annotation(x=c["date"].iloc[-1], y=final, text=f"{final-100:+.0f}%",
                           showarrow=False, xanchor="left", xshift=5,
                           font=dict(color=t["color"], size=11))
    b = bm[bm["index_name"] == "sp500"].copy()
    b = b[b["date"] >= start].sort_values("date")
    if not b.empty:
        idx = (1 + b["total_return"].fillna(0)).cumprod()
        b["cum"] = idx / idx.iloc[0] * 100.0
        fig.add_trace(go.Scatter(
            x=b["date"].tolist(), y=b["cum"].tolist(), mode="lines", name="S&P 500 (cap-wt)",
            line=dict(color="#999", width=1.5, dash="dot"),
            hovertemplate="%{x|%b %Y}<br>%{y:.0f}<extra></extra>",
        ))
        fig.add_annotation(x=b["date"].iloc[-1], y=float(b["cum"].iloc[-1]),
                           text=f"{float(b['cum'].iloc[-1])-100:+.0f}%",
                           showarrow=False, xanchor="left", xshift=5,
                           font=dict(color="#999", size=11))
    if exai is not None and not exai.empty:
        fig.add_trace(go.Scatter(
            x=exai["date"].tolist(), y=exai["cum"].tolist(), mode="lines",
            name="S&P 500 equal-wt, ex-AI", line=dict(color="#111", width=1.8, dash="dash"),
            hovertemplate="%{x|%b %Y}<br>%{y:.0f}<extra></extra>",
        ))
        fig.add_annotation(x=exai["date"].iloc[-1], y=float(exai["cum"].iloc[-1]),
                           text=f"{float(exai['cum'].iloc[-1])-100:+.0f}%",
                           showarrow=False, xanchor="left", xshift=5,
                           font=dict(color="#111", size=11))
    fig.add_hline(y=100, line=dict(color="#ccc", width=1))
    fig.update_layout(
        height=480, template="plotly_white",
        title=f"Equal-weighted cumulative total return since {start} (base 100)",
        yaxis=dict(title="Index (base 100)"),
        xaxis=dict(title=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0),
        margin=dict(l=55, r=70, t=70, b=40),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})


def chart_momentum_bars(themes: list[dict]) -> str:
    horizons = [("ret_3m", "3M"), ("ret_6m", "6M"), ("ret_ytd", "YTD"), ("ret_1y", "1Y")]
    fig = go.Figure()
    for key, lbl in horizons:
        fig.add_trace(go.Bar(
            name=lbl,
            x=[t["name"].split(" (")[0] for t in themes],
            y=[(t[key] or 0) * 100 for t in themes],
            text=[fmt_pct(t[key], 0, True) if t[key] is not None else "—" for t in themes],
            textposition="outside",
        ))
    fig.add_hline(y=0, line=dict(color="#444", width=1))
    fig.update_layout(
        height=420, template="plotly_white", barmode="group",
        title="Price momentum by horizon (cap-weighted basket total return)",
        yaxis=dict(title="Return (%)", ticksuffix="%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
        margin=dict(l=50, r=20, t=70, b=40),
        colorway=["#bcd", "#89a", "#4C72B0", "#C44E52"],
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})


def chart_valuation_bars(themes: list[dict]) -> str:
    names = [t["name"].split(" (")[0] for t in themes]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="P/S (LTM)", x=names, y=[t["ps"] for t in themes],
        marker_color="#4C72B0", yaxis="y1",
        text=[f"{t['ps']:.1f}x" if t["ps"] else "—" for t in themes], textposition="outside",
    ))
    fig.add_trace(go.Scatter(
        name="Rev growth YoY", x=names, y=[(t["rev_yoy"] or 0) * 100 for t in themes],
        mode="markers+text", marker=dict(color="#2C9F4E", size=14, symbol="diamond"),
        text=[fmt_pct(t["rev_yoy"], 0, True) if t["rev_yoy"] is not None else "—" for t in themes],
        textposition="top center", yaxis="y2",
    ))
    fig.update_layout(
        height=420, template="plotly_white",
        title="Valuation vs growth — basket P/S (bars) and revenue growth (diamonds)",
        yaxis=dict(title="P/S (LTM)", ticksuffix="x"),
        yaxis2=dict(title="Rev YoY (%)", overlaying="y", side="right", ticksuffix="%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
        margin=dict(l=50, r=60, t=70, b=40),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})


def chart_pername(themes: list[dict]) -> str:
    fig = go.Figure()
    for t in themes:
        pn = t["per_name"].dropna(subset=["Value", "Momentum"])
        if pn.empty:
            continue
        fig.add_trace(go.Scatter(
            x=pn["Momentum"].tolist(), y=pn["Value"].tolist(),
            mode="markers+text", name=t["name"].split(" (")[0],
            text=pn["ticker"].tolist(), textposition="top center",
            textfont=dict(size=9, color="#555"),
            marker=dict(size=11, color=t["color"], line=dict(color="#fff", width=1)),
        ))
    fig.add_hline(y=0, line=dict(color="#999", width=1, dash="dot"))
    fig.add_vline(x=0, line=dict(color="#999", width=1, dash="dot"))
    fig.update_layout(
        height=520, template="plotly_white",
        title="Every name — Momentum vs Value z (spot the cheap-with-momentum outliers)",
        xaxis=dict(title="Momentum z", zeroline=False),
        yaxis=dict(title="Value z", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0),
        margin=dict(l=50, r=20, t=70, b=40),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _z(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "<td>—</td>"
    klass = "pos" if v > 0 else "neg"
    return f'<td class="{klass}">{v:+.2f}</td>'


def _pct(v, signed=True, warn_hi=None, warn_lo=None) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "<td>—</td>"
    klass = ""
    if signed:
        klass = "pos" if v > 0 else "neg"
    return f'<td class="{klass}">{fmt_pct(v, 0, signed)}</td>'


def master_table(themes: list[dict]) -> str:
    cols = [
        ("Theme", lambda t: f'<td style="color:{t["color"]};font-weight:600">{t["name"]}</td>'),
        ("# / Mkt cap", lambda t: f'<td>{t["n"]} · {fmt_money(t["mcap_total"], "B")}</td>'),
        ("Value z", lambda t: _z(t["val_z"])),
        ("Momentum z", lambda t: _z(t["mom_z"])),
        ("Alpha z", lambda t: _z(t["alp_z"])),
        ("P/S", lambda t: f'<td>{t["ps"]:.1f}x</td>' if t["ps"] else "<td>—</td>"),
        ("P/E", lambda t: f'<td>{t["pe"]:.0f}x</td>' if t["pe"] else "<td>n/m</td>"),
        ("Rev YoY", lambda t: _pct(t["rev_yoy"])),
        ("Op margin", lambda t: _pct(t["op_margin"], signed=False)),
        ("YTD", lambda t: _pct(t["ret_ytd"])),
        ("1Y", lambda t: _pct(t["ret_1y"])),
        ("1Y vol", lambda t: _pct(t["vol_1y"], signed=False)),
        ("Max DD", lambda t: _pct(t["max_dd"])),
        ("Off 52wk hi", lambda t: _pct(t["off_52wh"])),
        ("Beta", lambda t: f'<td>{t["beta"]:.2f}</td>' if not math.isnan(t["beta"]) else "<td>—</td>"),
    ]
    head = "<tr>" + "".join(f"<th>{c[0]}</th>" for c in cols) + "</tr>"
    rows = "".join("<tr>" + "".join(fn(t) for _, fn in cols) + "</tr>" for t in themes)
    return f"<div class='table-wrap'><table>{head}{rows}</table></div>"


def crowding_table(themes: list[dict]) -> str:
    cols = [
        ("Theme", lambda t: f'<td style="color:{t["color"]};font-weight:600">{t["name"]}</td>'),
        ("Top-3 wt", lambda t: f'<td>{t["top3"]:.0f}%</td>'),
        ("Effective N", lambda t: f'<td>{t["eff_n"]:.1f}</td>'),
        ("1Y vol", lambda t: _pct(t["vol_1y"], signed=False)),
        ("Max DD (1Y)", lambda t: _pct(t["max_dd"])),
        ("Off 52wk hi", lambda t: _pct(t["off_52wh"])),
        ("Beta", lambda t: f'<td>{t["beta"]:.2f}</td>' if not math.isnan(t["beta"]) else "<td>—</td>"),
        ("Short-Int z", lambda t: _z(t["shi_z"])),
        ("Low-Vol z", lambda t: _z(t["lvol_z"])),
        ("Profit z", lambda t: _z(t["prof_z"])),
        ("Growth z", lambda t: _z(t["gro_z"])),
    ]
    head = "<tr>" + "".join(f"<th>{c[0]}</th>" for c in cols) + "</tr>"
    rows = "".join("<tr>" + "".join(fn(t) for _, fn in cols) + "</tr>" for t in themes)
    return f"<div class='table-wrap'><table>{head}{rows}</table></div>"


def overlap_note(themes: list[dict]) -> str:
    from collections import Counter
    member_in = {}
    for t in themes:
        for tk in t["resolved"]:
            member_in.setdefault(tk, []).append(t["name"].split(" (")[0])
    shared = {tk: ths for tk, ths in member_in.items() if len(ths) > 1}
    if not shared:
        return "<p class='note'>No names appear in more than one basket — the themes are disjoint bets.</p>"
    items = "".join(
        f"<li><strong>{tk}</strong> — {', '.join(ths)}</li>" for tk, ths in sorted(shared.items())
    )
    return (f"<p>These names sit in more than one basket, so the themes are <strong>not independent "
            f"bets</strong> — a position in both double-counts the overlap:</p><ul>{items}</ul>")


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

def build_html(themes: list[dict], bm: pd.DataFrame,
               exai: pd.DataFrame | None = None) -> str:
    snap = themes[0]["snap"]
    exai_note = ""
    if exai is not None and not exai.empty:
        exai_note = (f" The dashed line is the <strong>equal-weighted S&amp;P 500 excluding the "
                     f"{exai.attrs.get('n_excl', 0)} AI-basket names</strong> "
                     f"({exai.attrs.get('n_keep', 0)} remaining constituents, daily-rebalanced) — "
                     f"the market's return once the AI trade is stripped out.")
    total_names = sum(t["n"] for t in themes)

    hottest = max(themes, key=lambda t: (t["mom_z"] if not math.isnan(t["mom_z"]) else -9))
    cheapest = max(themes, key=lambda t: (t["val_z"] if not math.isnan(t["val_z"]) else -9))
    best_alpha = max(themes, key=lambda t: (t["alp_z"] if not math.isnan(t["alp_z"]) else -9))

    kpi_html = "".join([
        kpi("Themes", f"{len(themes)}", f"{total_names} names · {WEIGHT}-weighted"),
        kpi("Snapshot", snap, "quant cross-section"),
        kpi("Strongest momentum", hottest["name"].split(" (")[0], f"Mom z {hottest['mom_z']:+.2f}"),
        kpi("Cheapest on Value", cheapest["name"].split(" (")[0], f"Value z {cheapest['val_z']:+.2f}"),
        kpi("Best composite Alpha", best_alpha["name"].split(" (")[0], f"Alpha z {best_alpha['alp_z']:+.2f}",
            cls="pos" if best_alpha["alp_z"] > 0 else ""),
        kpi("Most extended", min(themes, key=lambda t: abs(t["off_52wh"] or -9))["name"].split(" (")[0],
            "nearest its 52-wk high"),
    ])

    member_lists = "".join(
        f"<li><strong style='color:{t['color']}'>{t['name']}</strong> "
        f"<span class='note'>({t['n']}): {', '.join(t['resolved'])}</span></li>"
        for t in themes
    )

    return f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><title>AI sub-themes — momentum vs valuation</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
{CSS}
<script charset="utf-8" src="https://cdn.plot.ly/plotly-3.5.0.min.js" integrity="sha256-fHbNLP+GlIXN+efbQec78UkemUz3NJp7UmfGxC1tNxs=" crossorigin="anonymous"></script>
</head><body>
<div class="wrap">
<header>
  <h1>AI sub-themes — momentum vs valuation</h1>
  <div class="sub">Where the AI trade's sub-baskets sit on the quant cross-section ·
       {len(themes)} themes · {total_names} names · {WEIGHT}-weighted · snapshot {snap} ·
       generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
  <div class="biz">A tactical map for a short-hold basket bet: are these sub-themes heated and
       extended, or is there a value entry? Momentum and Value are cap-weighted basket z-scores vs
       the Russell-1000 cross-section; returns, vol and drawdown are the static-weight basket series.</div>
</header>

<div class="kpi-grid">{kpi_html}</div>

<section>
  <h2>Executive summary</h2>
  <div class="lead">The cross-theme read in plain terms.</div>
  <div class="placeholder">{placeholder('EXEC_SUMMARY')}</div>
</section>

<section>
  <h2>The map — momentum vs value</h2>
  <div class="lead">Each bubble is a whole basket (size ∝ aggregate market cap). Up = cheap vs the
       universe; right = strong momentum. Bottom-right is the classic heated/crowded corner; top-left
       is cheap-but-out-of-favour. This is the single picture that answers "heated or opportunity?".</div>
  {chart_quadrant(themes)}
</section>

<section>
  <h2>Cumulative return since 2025</h2>
  <div class="lead">Equal-weighted baskets (each name held at 1/N, not cap-weighted), indexed to 100 at the
       first trading day of {EW_START[:4]}. This is the "what did it actually do" line, and because it is
       equal-weighted it is <strong>not</strong> dominated by the largest mega-caps — unlike the cap-weighted
       z-scores and fundamentals elsewhere in this report. S&amp;P 500 cap-weight (total return) shown dotted
       for reference.{exai_note}</div>
  {chart_cumulative(themes, bm, EW_START, exai)}
</section>

<section>
  <h2>Full comparison</h2>
  <div class="lead">Every metric side by side. Value / Momentum / Alpha are direction-adjusted basket
       z-scores (positive = favourable vs universe). P/E shown as n/m where the basket is not GAAP-profitable.</div>
  {master_table(themes)}
</section>

<section>
  <h2>Momentum detail</h2>
  <div class="lead">Price momentum across horizons. A theme that is strong at 1Y but fading at 3M is
       losing steam; one accelerating into the recent window is where the crowd is now.</div>
  {chart_momentum_bars(themes)}
</section>

<section>
  <h2>Valuation detail</h2>
  <div class="lead">Basket price-to-sales against revenue growth — the cheap-looking theme is only cheap
       if its growth holds. A high P/S paired with decelerating growth is the expensive-and-vulnerable case.</div>
  {chart_valuation_bars(themes)}
</section>

<section>
  <h2>Heat, crowding &amp; risk</h2>
  <div class="lead">How extended and how concentrated each basket is. Low effective-N and a small
       distance off the 52-week high mean a crowded, fully-priced trade with less margin for error.</div>
  {crowding_table(themes)}
</section>

<section>
  <h2>Name-level dispersion</h2>
  <div class="lead">Within each theme, which individual names are the cheap-with-momentum outliers
       (upper-right) versus the expensive leaders carrying the basket (lower-right).</div>
  {chart_pername(themes)}
</section>

<section>
  <h2>Theme overlap</h2>
  <div class="lead">Shared constituents mean the baskets are correlated bets, not independent ones.</div>
  {overlap_note(themes)}
</section>

<section>
  <h2>Per-theme tactical read</h2>
  <div class="lead">One call per basket, grounded in the numbers above.</div>
  <div class="placeholder">{placeholder('THEME_READS')}</div>
</section>

<div class="verdict">
  <h3><span class="pill">{placeholder('VERDICT_TAG')}</span>Verdict — where to put the tactical bet</h3>
  <div class="placeholder">{placeholder('VERDICT_BODY')}</div>
</div>

<section class="sources">
  <h2>Sources &amp; references</h2>
  <ul class="placeholder">{placeholder('SOURCES')}</ul>
  <h3 style="font-size:14px;margin-top:14px;">Basket membership</h3>
  <ul>{member_lists}</ul>
  <p class="note"><strong>Quant-infra sources:</strong> <code>data/models.db</code> ({snap}),
     <code>data/factors.db</code>, <code>data/risk.db</code> (Barra), <code>data/constituents.db</code>
     (EDGAR 10-Q/10-K), <code>data/returns.db</code> (Yahoo close + total return).
     Value/Momentum/Alpha are cap-weighted direction-adjusted model z-scores.</p>
</section>

<footer>
  Generated by <code>scripts/theme_compare.py</code> · personal research, not investment advice.
  Basket weights are static (today's cap mix applied to historical returns) — an exposure view, not a
  rebalanced backtest. Baskets overlap (see Theme overlap). P/S and P/E are aggregate
  (sum market cap / sum revenue or net income), not cap-weighted per-name medians.
</footer>

</div></body></html>"""


# ---------------------------------------------------------------------------
# Per-name deep-dive (single basket)
# ---------------------------------------------------------------------------

def load_name_betas(isins: list[str]) -> dict[str, float]:
    import sqlite3
    with sqlite3.connect(RISK_DB) as c:
        snap = c.execute("SELECT MAX(snapshot_date) FROM factor_exposures").fetchone()[0]
        rows = c.execute(
            f"SELECT security_id, exposure FROM factor_exposures "
            f"WHERE snapshot_date=? AND factor_id='beta_60d' "
            f"AND security_id IN ({','.join(['?']*len(isins))})",
            [snap] + isins,
        ).fetchall()
    return {r[0]: float(r[1]) for r in rows}


def build_name_rows(spec: dict) -> tuple[list[dict], str]:
    """Per-name quant + fundamentals + momentum for one basket."""
    print(f"[{spec['slug']}] per-name — loading {len(spec['tickers'])} names ...")
    members = load_basket_members(spec["tickers"])
    weights = compute_weights(members, WEIGHT)
    members.sort(key=lambda m: weights[m["ticker"]], reverse=True)
    agg = aggregate_ltm(members, weights)
    model_summary, pivot = aggregate_models(members, weights)
    mdf = agg["members_df"]
    betas = load_name_betas([m["isin"] for m in members])

    rows = []
    for m in members:
        tk, isin = m["ticker"], m["isin"]
        perf = perf_block(load_returns(isin))
        pz = pivot.loc[tk].to_dict() if tk in pivot.index else {}
        fr = mdf.loc[tk].to_dict() if tk in mdf.index else {}
        ltm = (m.get("ltm_block") or {}).get("ltm", {})
        ni = ltm.get("NetIncome")
        mcap = m["mcap"]
        rows.append({
            "ticker": tk, "company": m.get("short_name") or tk,
            "mcap": mcap, "weight": weights.get(tk, 0.0) * 100,
            "alpha_z": pz.get("Alpha"), "val_z": pz.get("Value"),
            "mom_z": pz.get("Momentum"), "gro_z": pz.get("Growth"),
            "prof_z": pz.get("Profitability"), "bsq_z": pz.get("Balance Sheet Quality"),
            "lvol_z": pz.get("Low Volatility"), "shi_z": pz.get("Short Interest"),
            "ps": fr.get("P/S (LTM)"), "pe": (mcap / ni) if (ni and ni > 0) else None,
            "rev_yoy": fr.get("Rev YoY %"), "op_margin": fr.get("Op Margin %"),
            "ret_3m": perf.get("3m"), "ret_1y": perf.get("1y"),
            "ret_ytd": perf.get("ytd"), "off_52wh": perf.get("off_52wh"),
            "vol_1y": perf.get("vol_1y"), "beta": betas.get(isin, float("nan")),
        })
    rows.sort(key=lambda r: (r["alpha_z"] if r["alpha_z"] is not None else -9), reverse=True)
    return rows, model_summary["snap"]


def _fz(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "<td>—</td>"
    return f'<td class="{"pos" if v > 0 else "neg"}">{v:+.2f}</td>'


def _fp(v, signed=True, pct=True) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "<td>—</td>"
    v2 = v / 100 if pct else v
    klass = ("pos" if v > 0 else "neg") if signed else ""
    return f'<td class="{klass}">{fmt_pct(v2, 0, signed)}</td>'


def name_table(rows: list[dict]) -> str:
    cols = [
        ("Name", lambda r: f'<td>{r["company"]} <span class="note">{r["ticker"]}</span></td>'),
        ("Mkt cap", lambda r: f'<td>{fmt_money(r["mcap"], "B")}</td>'),
        ("Alpha z", lambda r: _fz(r["alpha_z"])),
        ("Value z", lambda r: _fz(r["val_z"])),
        ("Mom z", lambda r: _fz(r["mom_z"])),
        ("Growth z", lambda r: _fz(r["gro_z"])),
        ("Quality z", lambda r: _fz(r["prof_z"])),
        ("BalSheet z", lambda r: _fz(r["bsq_z"])),
        ("P/S", lambda r: f'<td>{r["ps"]:.1f}x</td>' if r["ps"] else "<td>—</td>"),
        ("P/E", lambda r: f'<td>{r["pe"]:.0f}x</td>' if r["pe"] else "<td>n/m</td>"),
        ("Rev YoY", lambda r: _fp(r["rev_yoy"])),
        ("Op mgn", lambda r: _fp(r["op_margin"], signed=False)),
        ("3M", lambda r: _fp(r["ret_3m"] * 100 if r["ret_3m"] is not None else None)),
        ("1Y", lambda r: _fp(r["ret_1y"] * 100 if r["ret_1y"] is not None else None)),
        ("Off 52wh", lambda r: _fp(r["off_52wh"] * 100 if r["off_52wh"] is not None else None)),
        ("Beta", lambda r: f'<td>{r["beta"]:.2f}</td>' if not math.isnan(r["beta"]) else "<td>—</td>"),
    ]
    head = "<tr>" + "".join(f"<th>{c[0]}</th>" for c in cols) + "</tr>"
    body = "".join("<tr>" + "".join(fn(r) for _, fn in cols) + "</tr>" for r in rows)
    return f"<div class='table-wrap'><table>{head}{body}</table></div>"


def name_scatter(rows: list[dict], color: str) -> str:
    pts = [r for r in rows if r["mom_z"] is not None and r["val_z"] is not None]
    xs = [r["mom_z"] for r in pts]
    ys = [r["val_z"] for r in pts]
    sizes = [max(14, min(46, math.sqrt(r["mcap"] / 1e9) * 3.2)) for r in pts]
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color="#999", width=1, dash="dot"))
    fig.add_vline(x=0, line=dict(color="#999", width=1, dash="dot"))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text", text=[r["ticker"] for r in pts],
        textposition="top center", textfont=dict(size=11, color="#333"),
        marker=dict(size=sizes, color=color, opacity=0.85, line=dict(color="#fff", width=1.5)),
        hovertext=[f"{r['company']}<br>Alpha {r['alpha_z']:+.2f} · Mom {r['mom_z']:+.2f} · Val {r['val_z']:+.2f}"
                   for r in pts], hoverinfo="text", showlegend=False,
    ))
    fig.update_layout(
        height=480, template="plotly_white",
        title="Names on Momentum vs Value (size ∝ market cap) — upper-right = cheap-with-momentum",
        xaxis=dict(title="Momentum z", zeroline=False),
        yaxis=dict(title="Value z", zeroline=False),
        margin=dict(l=55, r=30, t=60, b=45),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})


def build_names_html(spec: dict, rows: list[dict], snap: str) -> str:
    best_alpha = rows[0]
    cheapest = max(rows, key=lambda r: (r["val_z"] if r["val_z"] is not None else -9))
    hottest = max(rows, key=lambda r: (r["mom_z"] if r["mom_z"] is not None else -9))
    kpi_html = "".join([
        kpi("Names", f"{len(rows)}", spec["name"].split(" (")[0]),
        kpi("Snapshot", snap, "quant cross-section"),
        kpi("Best Alpha", f"{best_alpha['ticker']}", f"Alpha z {best_alpha['alpha_z']:+.2f}",
            cls="pos" if best_alpha["alpha_z"] and best_alpha["alpha_z"] > 0 else ""),
        kpi("Cheapest (Value z)", f"{cheapest['ticker']}", f"Value z {cheapest['val_z']:+.2f}"),
        kpi("Strongest momentum", f"{hottest['ticker']}", f"Mom z {hottest['mom_z']:+.2f}"),
        kpi("Median P/S", f"{np.nanmedian([r['ps'] for r in rows if r['ps']]):.1f}x", "basket"),
    ])
    return f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><title>{spec['name']} — name-by-name</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
{CSS}
<script charset="utf-8" src="https://cdn.plot.ly/plotly-3.5.0.min.js" integrity="sha256-fHbNLP+GlIXN+efbQec78UkemUz3NJp7UmfGxC1tNxs=" crossorigin="anonymous"></script>
</head><body>
<div class="wrap">
<header>
  <h1>{spec['name']} — name by name</h1>
  <div class="sub">Per-name thesis, risks and quant signals to pick 2–3 for an opportunistic book ·
       {len(rows)} names · snapshot {snap} · generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
  <div class="biz">Ranked by composite Alpha z. Alpha / Value / Momentum / Growth / Quality are
       direction-adjusted model z-scores vs the Russell-1000 cross-section (positive = favourable).
       Returns, vol and beta are per-name.</div>
</header>

<div class="kpi-grid">{kpi_html}</div>

<section>
  <h2>The screen</h2>
  <div class="lead">Every name, ranked by Alpha. Read Value+Momentum together: a name high on both (upper-right
       in the chart) is cheap with the trend behind it; high Quality/Balance-Sheet z means the fundamentals can
       carry it through a capex wobble.</div>
  {name_table(rows)}
</section>

<section>
  <h2>Positioning map</h2>
  <div class="lead">Where each name sits on momentum vs value. The names to shortlist for a dip-buy are
       up-and-right; the crowded leaders are lower-right (strong momentum, expensive).</div>
  {name_scatter(rows, spec['color'])}
</section>

<section>
  <h2>Name-by-name — thesis &amp; risks</h2>
  <div class="lead">The structural bet, the main risks, and the quant tell for each. Full company names used.</div>
  <div class="placeholder">{placeholder('NAME_CARDS')}</div>
</section>

<div class="verdict">
  <h3><span class="pill">{placeholder('VERDICT_TAG')}</span>The 2–3 to buy</h3>
  <div class="placeholder">{placeholder('VERDICT_BODY')}</div>
</div>

<section class="sources">
  <h2>Sources &amp; references</h2>
  <ul class="placeholder">{placeholder('SOURCES')}</ul>
  <p class="note"><strong>Quant-infra sources:</strong> <code>data/models.db</code> ({snap}),
     <code>data/factors.db</code>, <code>data/risk.db</code> (Barra beta), <code>data/constituents.db</code>
     (EDGAR 10-Q/10-K), <code>data/returns.db</code> (Yahoo close + total return).</p>
</section>

<footer>
  Generated by <code>scripts/theme_compare.py --names {spec['slug']}</code> · personal research, not investment advice.
</footer>
</div></body></html>"""


def run_names(slug: str) -> None:
    spec = next((b for b in BASKETS if b["slug"] == slug), None)
    if spec is None:
        raise SystemExit(f"Unknown basket slug '{slug}'. Options: {', '.join(b['slug'] for b in BASKETS)}")
    rows, snap = build_name_rows(spec)
    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / f"{slug}_names.html"
    out.write_text(build_names_html(spec, rows, snap), encoding="utf-8")
    print(f"\nDone → {out}\n")
    print(f"NAME METRICS ({spec['name']}) ".ljust(70, "="))
    for r in rows:
        print(f"\n{r['company']} ({r['ticker']})  mcap={fmt_money(r['mcap'],'B')}")
        print(f"  Alpha z={r['alpha_z']:+.2f}  Value z={r['val_z']:+.2f}  Mom z={r['mom_z']:+.2f}  "
              f"Growth z={r['gro_z']:+.2f}  Quality z={r['prof_z']:+.2f}  BalSheet z={r['bsq_z']:+.2f}  "
              f"ShortInt z={r['shi_z']:+.2f}")
        pe = f"{r['pe']:.0f}x" if r['pe'] else "n/m"
        print(f"  P/S={r['ps']:.1f}x  P/E={pe}  RevYoY={r['rev_yoy']:+.0f}%  OpMgn={r['op_margin']:.0f}%  "
              f"3M={fmt_pct(r['ret_3m'],1,True)}  1Y={fmt_pct(r['ret_1y'],1,True)}  "
              f"off52wh={fmt_pct(r['off_52wh'],1)}  beta={r['beta']:.2f}")


def run_compare() -> None:
    themes = [build_theme(spec) for spec in BASKETS]
    bm = load_benchmarks(start="2024-12-01")
    exclude_isins = {m["isin"] for t in themes for m in t["members"]}
    exai = market_ex_ai_cum(exclude_isins, EW_START)
    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / "ai_themes_compare.html"
    out.write_text(build_html(themes, bm, exai), encoding="utf-8")
    print(f"\nDone → {out}\n")
    if exai is not None:
        print(f"ex-AI EW line: {exai.attrs.get('n_keep')} names, "
              f"final {exai['cum'].iloc[-1]:.0f} ({exai['cum'].iloc[-1]-100:+.0f}%)")
    # dump metrics so the narrative can be grounded in real numbers
    print("THEME METRICS ".ljust(60, "="))
    for t in themes:
        print(f"\n{t['name']}  [{', '.join(t['resolved'])}]")
        print(f"  mcap={fmt_money(t['mcap_total'],'B')}  Value z={t['val_z']:+.2f}  "
              f"Mom z={t['mom_z']:+.2f}  Alpha z={t['alp_z']:+.2f}  Growth z={t['gro_z']:+.2f}  "
              f"Prof z={t['prof_z']:+.2f}  LowVol z={t['lvol_z']:+.2f}  ShortInt z={t['shi_z']:+.2f}")
        print(f"  P/S={t['ps']:.1f}x  P/E={'%.0fx'%t['pe'] if t['pe'] else 'n/m'}  "
              f"RevYoY={fmt_pct(t['rev_yoy'],1,True)}  OpMargin={fmt_pct(t['op_margin'],1)}")
        print(f"  3M={fmt_pct(t['ret_3m'],1,True)}  6M={fmt_pct(t['ret_6m'],1,True)}  "
              f"YTD={fmt_pct(t['ret_ytd'],1,True)}  1Y={fmt_pct(t['ret_1y'],1,True)}  "
              f"vol={fmt_pct(t['vol_1y'],0)}  maxDD={fmt_pct(t['max_dd'],1)}  "
              f"off52wh={fmt_pct(t['off_52wh'],1)}  beta={t['beta']:.2f}")
        print(f"  top3={t['top3']:.0f}%  effN={t['eff_n']:.1f}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--names", help="Per-name deep-dive for one or more basket slugs "
                                    "(comma-separated), e.g. --names ai_industrials,cyber")
    args = ap.parse_args()
    if args.names:
        for slug in [s.strip() for s in args.names.split(",") if s.strip()]:
            run_names(slug)
    else:
        run_compare()


if __name__ == "__main__":
    main()
