#!/usr/bin/env python3
"""
theme_report.py — Quant-infra thematic basket research report.

Companion to single_name_report.py. Takes a list of tickers, computes
cap-weighted (or equal-weighted) basket aggregates across the quant
infra (factors, models, Barra exposures, LTM fundamentals, returns)
and emits reports/<slug>_theme.html with narrative placeholders for
the /theme slash command to fill.

Reuses single_name_report utilities heavily — DB plumbing, LTM
derivation, perf_block, short_name, CSS template — to avoid drift.

Run from project root:
    python scripts/theme_report.py \\
        --slug ai_industrials --name "AI Industrials" \\
        --tickers VRT,ETN,GEV,PWR,FIX,EME,NVT,HUBB,GNRC,CMI \\
        [--weight cap|equal]
"""
from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))
from report_utils import (  # noqa: E402
    RETURNS_DB, CONST_DB, MODELS_DB, FACTORS_DB, RISK_DB, UNIV_DB,
    REPORTS_DIR, MODEL_NAMES, COLOR_ZS, COLOR_PEER, COLOR_POS, COLOR_NEG,
    NARRATIVE_PLACEHOLDERS,
    CSS, fmt_pct, fmt_money, kpi, placeholder,
    lookup_company, short_name, latest_market_cap,
    compute_ltm_block, load_returns, load_benchmarks, perf_block,
    load_factor_reference,
)


# ---------------------------------------------------------------------------
# Basket data assembly
# ---------------------------------------------------------------------------

def load_basket_members(tickers: list[str]) -> list[dict]:
    """For each ticker: company metadata + LTM block + latest mcap + returns."""
    members = []
    for t in tickers:
        try:
            comp = lookup_company(t)
        except SystemExit:
            print(f"  warning: {t} not in universe.db, skipping")
            continue
        mcap = latest_market_cap(comp["isin"], comp.get("simfin_id"))
        if mcap is None:
            print(f"  warning: {t} has no market cap available, skipping")
            continue
        try:
            ltm = compute_ltm_block(comp["isin"], comp.get("simfin_id"))
        except Exception as e:
            print(f"  warning: {t} LTM failed ({e}), continuing without fundamentals")
            ltm = {}
        comp["mcap"] = mcap
        comp["ltm_block"] = ltm
        members.append(comp)
    return members


def compute_weights(members: list[dict], scheme: str) -> dict[str, float]:
    """Returns ticker → weight."""
    if scheme == "equal":
        n = len(members)
        return {m["ticker"]: 1.0 / n for m in members}
    if scheme == "cap":
        total = sum(m["mcap"] for m in members)
        return {m["ticker"]: m["mcap"] / total for m in members}
    raise ValueError(f"Unknown weighting scheme: {scheme}")


def basket_return_series(members: list[dict], weights: dict[str, float]) -> pd.DataFrame:
    """Cap-weighted daily total return series with static (today's) weights.

    Truncates to the date range where ALL members have return data — otherwise
    a member that IPO'd partway through (e.g. GEV in Apr 2024) gets its current
    weight applied to zero pre-IPO returns + full post-IPO run-up, which
    materially distorts the basket's apparent performance.
    """
    frames = {}
    first_valid = {}
    for m in members:
        df = load_returns(m["isin"])
        if df.empty:
            continue
        s = df.set_index("date")["total_return"]
        frames[m["ticker"]] = s
        first_valid[m["ticker"]] = s.dropna().index.min()
    if not frames:
        return pd.DataFrame()
    rets = pd.DataFrame(frames)
    # Truncate to the latest first-valid date so every member contributes from day 1.
    start = max(first_valid.values())
    rets = rets.loc[rets.index >= start]
    rets = rets.fillna(0.0)
    w_series = pd.Series(weights).reindex(rets.columns).fillna(0)
    basket = (rets * w_series).sum(axis=1)
    out = basket.to_frame("total_return")
    out["close"] = (1 + out["total_return"]).cumprod() * 100  # synthetic basket index
    out.index.name = "date"
    out = out.reset_index()
    out.attrs["common_start"] = start.strftime("%Y-%m-%d")
    out.attrs["binding_member"] = max(first_valid, key=first_valid.get)
    return out


def aggregate_ltm(members: list[dict], weights: dict[str, float]) -> dict:
    """Sum-style aggregates of LTM income/cash (treats basket like a holding co)."""
    keys = ["Revenue", "GrossProfit", "OperatingIncomeLoss", "NetIncome",
            "ResearchAndDevelopmentExpenses"]
    cur_sum = {k: 0.0 for k in keys}
    prev_sum = {k: 0.0 for k in keys}
    mcap_sum = 0.0
    weighted_yoy_num = 0.0
    weighted_yoy_den = 0.0
    member_rows = []
    for m in members:
        ltm = m["ltm_block"].get("ltm", {}) if m.get("ltm_block") else {}
        prev = m["ltm_block"].get("prev_ltm", {}) if m.get("ltm_block") else {}
        rev = ltm.get("Revenue")
        rev_prev = prev.get("Revenue")
        w = weights.get(m["ticker"], 0.0)
        # Member row for the per-name table
        member_rows.append({
            "Ticker": m["ticker"],
            "Company": m.get("short_name") or m["ticker"],
            "Weight %": w * 100,
            "Mkt Cap ($B)": m["mcap"] / 1e9,
            "Rev LTM ($M)": rev / 1e6 if rev else None,
            "Rev YoY %": (rev / rev_prev - 1) * 100 if (rev and rev_prev) else None,
            "Gross Margin %": (ltm.get("GrossProfit") / rev * 100) if (rev and ltm.get("GrossProfit")) else None,
            "Op Margin %": (ltm.get("OperatingIncomeLoss") / rev * 100) if (rev and ltm.get("OperatingIncomeLoss") is not None) else None,
            "P/S (LTM)": m["mcap"] / rev if rev else None,
        })
        mcap_sum += m["mcap"]
        for k in keys:
            v = ltm.get(k)
            vp = prev.get(k)
            if v is not None:
                cur_sum[k] += v
            if vp is not None:
                prev_sum[k] += vp
        # Weighted YoY rev growth
        if rev and rev_prev:
            weighted_yoy_num += w * (rev / rev_prev - 1)
            weighted_yoy_den += w

    rev_basket = cur_sum["Revenue"]
    weighted_rev_yoy = weighted_yoy_num / weighted_yoy_den if weighted_yoy_den else None
    sum_rev_yoy = (cur_sum["Revenue"] / prev_sum["Revenue"] - 1) if prev_sum["Revenue"] else None

    return {
        "cur": cur_sum, "prev": prev_sum,
        "mcap_total": mcap_sum,
        "weighted_rev_yoy": weighted_rev_yoy,
        "sum_rev_yoy": sum_rev_yoy,
        "aggregate_ps": mcap_sum / rev_basket if rev_basket else None,
        "weighted_gross_margin": cur_sum["GrossProfit"] / rev_basket if rev_basket else None,
        "weighted_op_margin": cur_sum["OperatingIncomeLoss"] / rev_basket if rev_basket else None,
        "weighted_rd_pct": cur_sum["ResearchAndDevelopmentExpenses"] / rev_basket if rev_basket else None,
        "members_df": pd.DataFrame(member_rows).set_index("Ticker"),
    }


def aggregate_models(members: list[dict], weights: dict[str, float]) -> tuple[dict, pd.DataFrame]:
    """Cap-weighted average model z-scores + per-name table (member x model)."""
    isins = [m["isin"] for m in members]
    with sqlite3.connect(MODELS_DB) as c:
        snap = c.execute("SELECT MAX(data_date) FROM models").fetchone()[0]
        df = pd.read_sql_query(
            f"SELECT security_id, model_id, model_value_z FROM models "
            f"WHERE data_date = ? AND security_id IN ({','.join(['?']*len(isins))})",
            c, params=[snap] + isins,
        )
    isin_to_ticker = {m["isin"]: m["ticker"] for m in members}
    df["ticker"] = df["security_id"].map(isin_to_ticker)
    df["model"] = df["model_id"].map(MODEL_NAMES).fillna(df["model_id"])

    pivot = df.pivot(index="ticker", columns="model", values="model_value_z")
    order = ["Alpha (Composite)", "Quality", "Value", "Growth", "Momentum",
             "Size", "Low Volatility", "Liquidity", "Short Interest", "LT Reversal"]
    cols = [c for c in order if c in pivot.columns]
    pivot = pivot[cols]
    # cap-weighted basket z per model
    w = pd.Series(weights).reindex(pivot.index).fillna(0)
    basket_z = {col: float((pivot[col].fillna(0) * w).sum() / w.sum()) for col in cols}
    return {"snap": snap, "basket_z": basket_z, "models_order": cols}, pivot


def aggregate_factors(members: list[dict], weights: dict[str, float]) -> pd.DataFrame:
    """Direction-adjusted cap-weighted factor z-scores."""
    isins = [m["isin"] for m in members]
    with sqlite3.connect(FACTORS_DB) as c:
        snap = c.execute("SELECT MAX(data_date) FROM factors").fetchone()[0]
        df = pd.read_sql_query(
            f"SELECT security_id, factor_id, factor_value_z FROM factors "
            f"WHERE data_date = ? AND security_id IN ({','.join(['?']*len(isins))})",
            c, params=[snap] + isins,
        )
    fref = load_factor_reference()
    df = df.merge(fref[["factor_id", "factor_name", "category", "direction"]],
                  on="factor_id", how="left")
    df["signed_z"] = df["factor_value_z"] * df["direction"]
    isin_to_ticker = {m["isin"]: m["ticker"] for m in members}
    df["ticker"] = df["security_id"].map(isin_to_ticker)
    w = pd.Series(weights)
    df["weight"] = df["ticker"].map(w).fillna(0)
    # Weighted average direction-adjusted z per factor
    grouped = df.dropna(subset=["signed_z"]).groupby("factor_name").apply(
        lambda g: pd.Series({
            "weighted_z": (g["signed_z"] * g["weight"]).sum() / max(g["weight"].sum(), 1e-12),
            "category": g["category"].iloc[0],
        })
    ).reset_index()
    grouped.attrs["snap"] = snap
    return grouped


def aggregate_barra(members: list[dict], weights: dict[str, float]) -> tuple[pd.DataFrame, float, str]:
    """Cap-weighted Barra exposures + cap-weighted basket idio vol (sqrt of weighted idio var)."""
    isins = [m["isin"] for m in members]
    with sqlite3.connect(RISK_DB) as c:
        snap = c.execute("SELECT MAX(snapshot_date) FROM factor_exposures").fetchone()[0]
        expos = pd.read_sql_query(
            f"SELECT security_id, factor_id, exposure FROM factor_exposures "
            f"WHERE snapshot_date = ? AND security_id IN ({','.join(['?']*len(isins))})",
            c, params=[snap] + isins,
        )
        idio = pd.read_sql_query(
            f"SELECT security_id, idio_var FROM idiosyncratic_vars "
            f"WHERE snapshot_date = ? AND security_id IN ({','.join(['?']*len(isins))})",
            c, params=[snap] + isins,
        )
    isin_to_ticker = {m["isin"]: m["ticker"] for m in members}
    expos["ticker"] = expos["security_id"].map(isin_to_ticker)
    w = pd.Series(weights)
    expos["weight"] = expos["ticker"].map(w).fillna(0)
    weighted = (expos.groupby("factor_id")
                     .apply(lambda g: (g["exposure"] * g["weight"]).sum())
                     .reset_index(name="exposure"))
    fref = load_factor_reference()
    name_map = dict(zip(fref["factor_id"], fref["factor_name"]))
    name_map.update({"beta_60d": "Beta (60d)"})
    weighted["name"] = weighted["factor_id"].map(name_map).fillna(weighted["factor_id"])
    weighted = weighted[~weighted["factor_id"].str.startswith("sec_")]

    # Idio vol — assume independence across members; basket idio var = sum(w_i^2 * idio_var_i)
    idio["ticker"] = idio["security_id"].map(isin_to_ticker)
    idio["weight"] = idio["ticker"].map(w).fillna(0)
    basket_idio_var = float((idio["weight"] ** 2 * idio["idio_var"]).sum())
    basket_idio_vol = math.sqrt(basket_idio_var) if basket_idio_var > 0 else float("nan")
    return weighted, basket_idio_vol, snap


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def chart_composition(members_df: pd.DataFrame) -> str:
    df = members_df.sort_values("Weight %", ascending=True)
    # Convert to plain Python lists — pandas Series get serialised by Plotly as
    # binary {"dtype":"f8","bdata":"..."} which fails to render in some browsers.
    weights = df["Weight %"].tolist()
    names = df["Company"].tolist()
    fig = go.Figure(go.Bar(
        x=weights, y=names, orientation="h",
        marker_color=COLOR_PEER,
        text=[f"{w:.1f}%" for w in weights], textposition="auto",
    ))
    fig.update_layout(
        height=max(280, 30 * len(df) + 80),
        template="plotly_white",
        title="Basket composition (weight)",
        xaxis=dict(title="Weight (%)", ticksuffix="%"),
        margin=dict(l=140, r=40, t=50, b=40),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})


def chart_basket_perf(basket_ret: pd.DataFrame, bm: pd.DataFrame, basket_name: str) -> str:
    # Start chart from basket's earliest available date — every member must have
    # data from that point (enforced in basket_return_series).
    cutoff = basket_ret["date"].min()
    end = basket_ret["date"].max()
    years = (end - cutoff).days / 365.25
    z = basket_ret[basket_ret["date"] >= cutoff].copy()
    z["idx"] = (1 + z["total_return"].fillna(0)).cumprod() * 100
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32],
                        vertical_spacing=0.05,
                        subplot_titles=(f"Indexed total return ({years:.1f}Y, base 100)",
                                        "Drawdown vs running peak"))
    fig.add_trace(go.Scatter(x=z["date"], y=z["idx"], name=basket_name,
                             line=dict(color=COLOR_ZS, width=2.4)), row=1, col=1)
    bm_styles = {"sp500": ("S&P 500", "#444"),
                 "ai_tech": ("AI / Tech", "#888"),
                 "russell_1000": ("Russell 1000", "#bbb")}
    for idx_name, (lbl, color) in bm_styles.items():
        b = bm[bm["index_name"] == idx_name].copy()
        b = b[b["date"] >= cutoff].sort_values("date")
        if b.empty:
            continue
        b["idx"] = (1 + b["total_return"].fillna(0)).cumprod() * 100
        fig.add_trace(go.Scatter(x=b["date"], y=b["idx"], name=lbl,
                                 line=dict(color=color, width=1.4, dash="dot")), row=1, col=1)
    z["dd"] = z["idx"] / z["idx"].cummax() - 1
    fig.add_trace(go.Scatter(x=z["date"], y=z["dd"]*100, name="DD",
                             line=dict(color=COLOR_NEG, width=1.6),
                             fill="tozeroy", fillcolor="rgba(196,78,82,0.15)",
                             showlegend=False), row=2, col=1)
    fig.update_yaxes(title="Index", row=1, col=1)
    fig.update_yaxes(title="DD (%)", ticksuffix="%", row=2, col=1)
    fig.update_layout(height=520, template="plotly_white",
                      legend=dict(orientation="h", yanchor="bottom", y=1.05,
                                  xanchor="left", x=0),
                      margin=dict(l=40, r=20, t=60, b=30))
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})


def chart_model_bars(basket_z: dict, models_order: list[str]) -> str:
    vals = [basket_z[c] for c in models_order]
    colors = [COLOR_POS if v > 0 else COLOR_NEG for v in vals]
    fig = go.Figure(go.Bar(
        x=models_order, y=vals, marker_color=colors,
        text=[f"{v:+.2f}" for v in vals], textposition="auto",
    ))
    fig.add_hline(y=0, line=dict(color="#444", width=1))
    fig.update_layout(
        height=380, template="plotly_white",
        title="Weighted model z-scores (basket vs universe)",
        yaxis=dict(title="Weighted z-score"),
        xaxis=dict(tickangle=-30),
        margin=dict(l=50, r=20, t=60, b=80),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})


def chart_dispersion(members_df: pd.DataFrame) -> str:
    """Per-name P/S vs Rev YoY, sized by weight."""
    df = members_df.dropna(subset=["P/S (LTM)", "Rev YoY %"]).copy()
    if df.empty:
        return "<p class='note'>No fundamentals available to plot.</p>"
    sizes = ((df["Weight %"].clip(2, 30)) ** 0.7 * 4).tolist()
    fig = go.Figure(go.Scatter(
        x=df["Rev YoY %"].tolist(), y=df["P/S (LTM)"].tolist(),
        mode="markers+text", text=df["Company"].tolist(), textposition="top center",
        marker=dict(color=COLOR_PEER, size=sizes, line=dict(color="#333", width=1)),
        textfont=dict(size=11),
    ))
    fig.update_layout(
        height=440, template="plotly_white",
        title="Member dispersion — P/S vs Revenue YoY (size = basket weight)",
        xaxis=dict(title="Rev YoY (%)", ticksuffix="%"),
        yaxis=dict(title="P/S (LTM)"),
        margin=dict(l=50, r=20, t=60, b=40),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})


def chart_factor_bars(grouped: pd.DataFrame) -> str:
    g = grouped.dropna(subset=["weighted_z"]).sort_values("weighted_z").copy()
    vals = g["weighted_z"].tolist()
    names = g["factor_name"].tolist()
    colors = [COLOR_POS if v > 0 else COLOR_NEG for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}" for v in vals], textposition="auto",
    ))
    fig.add_vline(x=0, line=dict(color="#444", width=1))
    fig.update_layout(
        height=max(420, 22 * len(g) + 80),
        template="plotly_white",
        title="Weighted factor z-scores (direction-adjusted: positive = favourable)",
        xaxis=dict(title="Weighted z-score"),
        margin=dict(l=180, r=40, t=60, b=40),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True})


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def render_member_table(members_df: pd.DataFrame) -> str:
    cols = ["Weight %", "Mkt Cap ($B)", "Rev LTM ($M)", "Rev YoY %",
            "Gross Margin %", "Op Margin %", "P/S (LTM)"]
    rows = []
    for tkr, row in members_df.iterrows():
        cells = [f"<td>{row['Company']}</td>"]
        for c in cols:
            v = row[c]
            if pd.isna(v):
                cells.append("<td>—</td>")
            elif c in ("Weight %",):
                cells.append(f"<td>{v:.1f}%</td>")
            elif c in ("Mkt Cap ($B)",):
                cells.append(f"<td>{v:,.1f}</td>")
            elif c in ("Rev LTM ($M)",):
                cells.append(f"<td>{v:,.0f}</td>")
            elif c == "P/S (LTM)":
                cells.append(f"<td>{v:.2f}x</td>")
            else:
                klass = "pos" if v > 0 else "neg"
                cells.append(f'<td class="{klass}">{v:+.1f}%</td>')
        rows.append(f"<tr>{''.join(cells)}</tr>")
    head = "<tr><th>Company</th>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
    return f"<div class='table-wrap'><table>{head}{''.join(rows)}</table></div>"


def render_model_heatmap_table(pivot: pd.DataFrame, ticker_to_name: dict[str, str]) -> str:
    """Per-name model z heatmap as a tinted HTML table."""
    cols = list(pivot.columns)
    rows = []
    for tkr, row in pivot.iterrows():
        name = ticker_to_name.get(tkr, tkr)
        cells = [f"<td>{name}</td>"]
        for c in cols:
            v = row[c]
            if pd.isna(v):
                cells.append("<td>—</td>")
            else:
                # tint by z: green if positive, red if negative, intensity ∝ |z|
                a = min(abs(v) / 2.0, 1.0)
                color = f"rgba(44,159,78,{a:.2f})" if v > 0 else f"rgba(196,78,82,{a:.2f})"
                cells.append(f'<td style="background:{color}">{v:+.2f}</td>')
        rows.append(f"<tr>{''.join(cells)}</tr>")
    head = "<tr><th>Company</th>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
    return f"<div class='table-wrap'><table>{head}{''.join(rows)}</table></div>"


def render_barra_table(df: pd.DataFrame) -> str:
    df = df.sort_values("exposure", key=lambda s: s.abs(), ascending=False).head(12)
    rows = []
    for _, r in df.iterrows():
        e = float(r["exposure"])
        klass = "pos" if e > 0 else "neg"
        rows.append(f"<tr><td>{r['name']}</td><td class='{klass}'>{e:+.2f}</td></tr>")
    head = "<tr><th>Factor</th><th>Exposure (σ)</th></tr>"
    return f"<div class='table-wrap'><table>{head}{''.join(rows)}</table></div>"


def build_html(payload: dict) -> str:
    p = payload
    slug = p["slug"]
    basket_name = p["basket_name"]
    members = p["members"]
    weights = p["weights"]
    agg = p["agg"]
    perf = p["basket_perf"]
    snap = p["model_snap"]

    # KPI cards
    rev = agg["cur"]["Revenue"]
    op = agg["cur"]["OperatingIncomeLoss"]
    kpi_html = "".join([
        kpi("# Names", f"{len(members)}", f"{p['weight_scheme']}-weighted"),
        kpi("Basket mkt cap", fmt_money(agg["mcap_total"], "B"), f"largest: {p['largest_name']}"),
        kpi("Aggregate P/S", f"{agg['aggregate_ps']:.2f}x" if agg["aggregate_ps"] else "—",
            f"sum mcap / sum revenue"),
        kpi("Rev YoY (weighted)", fmt_pct(agg["weighted_rev_yoy"], 1) if agg["weighted_rev_yoy"] is not None else "—",
            f"{fmt_pct(agg['weighted_gross_margin'], 0)} gross margin" if agg["weighted_gross_margin"] else "",
            cls="pos" if (agg["weighted_rev_yoy"] or 0) > 0.05 else ""),
        kpi("Basket 1Y return", fmt_pct(perf.get("1y"), 1, True) if perf.get("1y") is not None else "—",
            f"YTD {fmt_pct(perf.get('ytd'), 1, True)}" if perf.get("ytd") is not None else "",
            cls="pos" if (perf.get("1y") or 0) > 0 else "neg"),
        kpi("Alpha z (weighted)", f"{p['basket_alpha_z']:+.2f}",
            "vs universe cross-section",
            cls="pos" if p["basket_alpha_z"] > 0.3 else ("neg" if p["basket_alpha_z"] < -0.3 else "")),
    ])

    ticker_to_name = {m["ticker"]: m.get("short_name") or m["ticker"] for m in members}

    # Quant facts block — populated from data, no narrative needed
    bz = p["basket_z"]
    quant_facts_html = f"""
    <p><strong>Basket quant snapshot ({snap}):</strong>
       cap-weighted Alpha z = <strong>{bz.get('Alpha (Composite)', float('nan')):+.2f}</strong>,
       Quality {bz.get('Quality', float('nan')):+.2f}, Value {bz.get('Value', float('nan')):+.2f},
       Growth {bz.get('Growth', float('nan')):+.2f}, Momentum {bz.get('Momentum', float('nan')):+.2f},
       Size {bz.get('Size', float('nan')):+.2f}.
       Cap-weighted Barra beta {p['basket_beta']:+.2f}; basket idio vol ~{p['basket_idio_vol']*100:.0f}%
       (assuming idio independence across names).</p>
    <p><strong>Aggregate fundamentals (LTM):</strong>
       basket revenue {fmt_money(rev, 'B')} growing
       {fmt_pct(agg['weighted_rev_yoy'], 1, True) if agg['weighted_rev_yoy'] is not None else '—'} (cap-weighted),
       gross margin {fmt_pct(agg['weighted_gross_margin'], 1) if agg['weighted_gross_margin'] else '—'},
       GAAP op margin {fmt_pct(agg['weighted_op_margin'], 1) if agg['weighted_op_margin'] else '—'}.</p>
    <p><strong>Performance:</strong>
       basket 1Y total return {fmt_pct(perf.get('1y'), 1, True)} ·
       YTD {fmt_pct(perf.get('ytd'), 1, True)} · 1Y realised vol {fmt_pct(perf.get('vol_1y'), 0)} ·
       1Y max DD {fmt_pct(perf.get('max_dd_1y'), 1)}.</p>
    """

    # Largest / smallest / dispersion stats
    largest = members[0]["ticker"]
    if len(members) >= 3:
        mdf = agg["members_df"]
        top3_concentration = mdf.nlargest(3, "Weight %")["Weight %"].sum()
    else:
        top3_concentration = 100.0

    sources_html = ""
    for s in p["sources"]:
        sources_html += f"<li><a href='{s['url']}' target='_blank' rel='noopener'>{s['title']}</a></li>"

    return f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><title>{basket_name} — thematic basket research</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
{CSS}
<script charset="utf-8" src="https://cdn.plot.ly/plotly-3.5.0.min.js" integrity="sha256-fHbNLP+GlIXN+efbQec78UkemUz3NJp7UmfGxC1tNxs=" crossorigin="anonymous"></script>
</head><body>
<div class="wrap">
<header>
  <h1>{basket_name}</h1>
  <div class="sub">Thematic basket of {len(members)} names ·
       {p['weight_scheme']}-weighted · snapshot {snap} ·
       generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
</header>

<div class="kpi-grid">{kpi_html}</div>

<section>
  <h2>Executive summary</h2>
  <div class="lead">What the basket-level numbers say about this theme.</div>
  {quant_facts_html}
  <div class="placeholder">{placeholder('EXEC_SUMMARY')}</div>
</section>

<section>
  <h2>Composition</h2>
  <div class="lead">Top-{min(3,len(members))} concentration: <strong>{top3_concentration:.0f}%</strong> of basket weight.
       Effective N (1/sum w²) = <strong>{1/sum(w**2 for w in weights.values()):.1f}</strong>.</div>
  {p['chart_composition']}
</section>

<section>
  <h2>Performance &amp; drawdown</h2>
  <div class="lead">Static-weight basket return — today's cap weights applied to historical daily returns,
       indexed to 100 from the earliest date all members had price history
       (truncated by the most-recently-listed member). Benchmarks: S&amp;P 500, AI / Tech composite, Russell 1000.</div>
  {p['chart_basket_perf']}
</section>

<section>
  <h2>Quant signals — basket vs universe</h2>
  <div class="lead">Cap-weighted average model z-scores. Positive = the basket is over-represented on
       that factor relative to the universe cross-section.</div>
  {p['chart_model_bars']}
  <h3 style="margin-top: 14px; font-size: 14px;">Per-name model z-scores (heatmap)</h3>
  {render_model_heatmap_table(p['model_pivot'], ticker_to_name)}
</section>

<section>
  <h2>Fundamentals — per-name table</h2>
  <div class="lead">LTM revenue, growth, margins and P/S for each constituent. Last quarter end varies by
       fiscal calendar.</div>
  {render_member_table(agg['members_df'])}
  {p['chart_dispersion']}
</section>

<section>
  <h2>Factor exposures &amp; Barra</h2>
  <div class="lead">Cap-weighted Barra exposures (basket sees these factor tilts vs the universe).
       Basket idio vol ~{p['basket_idio_vol']*100:.0f}% annualised.</div>
  <div class="split">
    <div>
      <h3 style="margin:0 0 10px; font-size:14px;">Top-magnitude Barra exposures</h3>
      {render_barra_table(p['barra_df'])}
    </div>
    <div>
      <p style="font-size: 13px;">Reading: the largest absolute Barra exposures tell you what the basket
      <em>is</em> in factor language — which style, growth, profitability or sector tilts dominate. Use this
      to plan hedges (e.g. hedge the dominant factor) or to confirm the theme expression is what you intended.</p>
    </div>
  </div>
  <div style="margin-top:18px;">{p['chart_factor_bars']}</div>
</section>

<section>
  <h2>Bull / bear thesis</h2>
  <div class="thesis">
    <div class="col bull">
      <h3><span class="tag bull">BULL</span> Why the theme works</h3>
      <ul class="placeholder">{placeholder('BULL_CASE')}</ul>
    </div>
    <div class="col bear">
      <h3><span class="tag bear">BEAR</span> What could derail it</h3>
      <ul class="placeholder">{placeholder('BEAR_CASE')}</ul>
    </div>
  </div>
</section>

<div class="verdict">
  <h3><span class="pill">{placeholder('VERDICT_TAG')}</span>Verdict</h3>
  <div class="placeholder">{placeholder('VERDICT_BODY')}</div>
</div>

<section class="sources">
  <h2>Sources &amp; references</h2>
  <ul class="placeholder">{placeholder('SOURCES')}</ul>
  <p class="note"><strong>Quant-infra sources:</strong> <code>data/models.db</code> ({snap}),
     <code>data/factors.db</code>, <code>data/risk.db</code> (Barra K=29),
     <code>data/constituents.db</code> (EDGAR 10-Q/10-K + SimFin),
     <code>data/returns.db</code> (Yahoo Finance close + total return).</p>
</section>

<footer>
  Generated by <code>scripts/theme_report.py</code> · personal research, not investment advice.
  Basket weights are static (today's cap mix applied to historical returns) — not a rebalanced backtest.
</footer>

</div></body></html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slug", required=True, help="URL-safe basket slug, e.g. ai_industrials")
    ap.add_argument("--name", required=True, help="Display name, e.g. 'AI Industrials'")
    ap.add_argument("--tickers", required=True, help="Comma-separated tickers")
    ap.add_argument("--weight", choices=["cap", "equal"], default="cap",
                    help="Weighting scheme (default: cap)")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    print(f"[{args.slug}] Loading {len(tickers)} basket members ...")
    members = load_basket_members(tickers)
    if not members:
        raise SystemExit("No members loaded — check tickers.")
    print(f"  resolved: {', '.join(m['ticker'] for m in members)}")

    weights = compute_weights(members, args.weight)
    # Sort members by weight desc — used in composition chart & largest-name kpi
    members.sort(key=lambda m: weights[m["ticker"]], reverse=True)
    largest_name = members[0].get("short_name") or members[0]["ticker"]

    print(f"[{args.slug}] Aggregating LTM fundamentals ...")
    agg = aggregate_ltm(members, weights)

    print(f"[{args.slug}] Building basket return series ...")
    basket_ret = basket_return_series(members, weights)
    if basket_ret.empty:
        raise SystemExit("No return data for any member.")
    basket_perf = perf_block(basket_ret)
    bm = load_benchmarks(start=(basket_ret["date"].max() - pd.Timedelta(days=3*365+90)).strftime("%Y-%m-%d"))

    print(f"[{args.slug}] Aggregating models / factors / Barra ...")
    model_summary, model_pivot = aggregate_models(members, weights)
    factor_grouped = aggregate_factors(members, weights)
    barra_df, basket_idio_vol, barra_snap = aggregate_barra(members, weights)

    # Pull Beta separately
    beta_row = barra_df[barra_df["factor_id"] == "beta_60d"]
    basket_beta = float(beta_row["exposure"].iloc[0]) if not beta_row.empty else float("nan")

    print(f"[{args.slug}] Rendering charts ...")
    chart_comp = chart_composition(agg["members_df"])
    chart_perf = chart_basket_perf(basket_ret, bm, args.name)
    chart_models = chart_model_bars(model_summary["basket_z"], model_summary["models_order"])
    chart_disp = chart_dispersion(agg["members_df"])
    chart_facs = chart_factor_bars(factor_grouped)

    sources = [
        {"title": "S&P Global — Sector ETF flows & weights (XLI)",
         "url": "https://www.spglobal.com/spdji/en/indices/equity/sp-500-industrials-sector/"},
    ]

    payload = {
        "slug": args.slug, "basket_name": args.name,
        "members": members, "weights": weights,
        "weight_scheme": args.weight,
        "largest_name": largest_name,
        "agg": agg, "basket_perf": basket_perf,
        "model_snap": model_summary["snap"],
        "basket_z": model_summary["basket_z"],
        "basket_alpha_z": model_summary["basket_z"].get("Alpha (Composite)", float("nan")),
        "model_pivot": model_pivot,
        "barra_df": barra_df, "basket_idio_vol": basket_idio_vol,
        "basket_beta": basket_beta,
        "chart_composition": chart_comp, "chart_basket_perf": chart_perf,
        "chart_model_bars": chart_models, "chart_dispersion": chart_disp,
        "chart_factor_bars": chart_facs,
        "sources": sources,
    }

    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / f"{args.slug}_theme.html"
    out.write_text(build_html(payload), encoding="utf-8")
    print(f"[{args.slug}] Done → {out}")
    print("\nNarrative placeholders for /theme to fill:")
    for ph in NARRATIVE_PLACEHOLDERS:
        print(f"  {{{{{ph}}}}}")


if __name__ == "__main__":
    main()
