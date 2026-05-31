#!/usr/bin/env python3
"""
single_name_report.py — Quant-infra single-security research report.

Builds a self-contained HTML report at reports/<TICKER>_report.html
combining factor/model/Barra signals, fundamentals, peer comparison
and price-action charts for any ticker present in universe.db.

The HTML is rendered with narrative placeholders ({{EXEC_SUMMARY}},
{{BULL_CASE}}, {{BEAR_CASE}}, {{VERDICT_TAG}}, {{VERDICT_BODY}},
{{SOURCES}}) that the /single-name slash command fills in after the
script writes the file. Run the script alone and you get a complete
quant report with the narrative slots showing 'TBD'.

Run from project root:
    python scripts/single_name_report.py --ticker ZS
    python scripts/single_name_report.py --ticker NVDA --peers AMD,INTC,QCOM,AVGO
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
    RETURNS_DB, CONST_DB, MODELS_DB, FACTORS_DB, RISK_DB, UNIV_DB, REPORTS_DIR,
    MODEL_NAMES, MODEL_RADAR_ORDER, MODEL_TABLE_ORDER, ALPHA_BLEND_NOTE,
    COLOR_ZS, COLOR_PEER, COLOR_POS, COLOR_NEG,
    NARRATIVE_PLACEHOLDERS,
    short_name, lookup_company, latest_market_cap,
    load_concept_map, load_factor_reference, load_constituent_df,
    quarter_value, derive_q4, get_latest_quarter, trailing_4q, sum_concept_over,
    compute_ltm_block,
    load_returns, load_benchmarks, perf_block,
    CSS, fmt_pct, fmt_money, kpi, placeholder,
)

DEFAULT_PEER_COUNT = 7


# ---------------------------------------------------------------------------
# Single-name-specific helpers (peer selection)
# Shared helpers (lookup_company, latest_market_cap, short_name) live in report_utils.
# ---------------------------------------------------------------------------

def pick_peers(target: dict, n: int = DEFAULT_PEER_COUNT) -> list[dict]:
    """Pick n peers by simfin_industry, ranked by market cap proximity."""
    industry = target.get("simfin_industry")
    if not industry:
        return []
    with sqlite3.connect(UNIV_DB) as c:
        rows = c.execute(
            "SELECT isin, ticker, company_name, simfin_id FROM companies "
            "WHERE simfin_industry = ? AND ticker != ?",
            (industry, target["ticker"]),
        ).fetchall()
    target_mcap = latest_market_cap(target["isin"], target.get("simfin_id"))
    if target_mcap is None:
        target_mcap = 1e10  # fallback
    candidates = []
    for isin, t, name, sf in rows:
        mc = latest_market_cap(isin, sf)
        if mc is None:
            continue
        candidates.append({
            "isin": isin, "ticker": t, "company_name": name,
            "short_name": short_name(name, t), "simfin_id": sf,
            "mcap": mc, "ratio": mc / target_mcap,
        })
    # Rank by closeness in log-mcap space
    candidates.sort(key=lambda r: abs(math.log(r["mcap"] / target_mcap)))
    return candidates[:n]


def resolve_peers(target: dict, peer_arg: str | None) -> list[dict]:
    if peer_arg:
        tickers = [t.strip().upper() for t in peer_arg.split(",") if t.strip()]
        out = []
        for t in tickers:
            try:
                out.append(lookup_company(t))
            except SystemExit:
                print(f"  warning: peer {t} not in universe, skipping")
        return out
    return pick_peers(target)


# ---------------------------------------------------------------------------
# Single-name model / factor / Barra loaders
# (Constituents / LTM helpers and Returns / perf live in report_utils.)
# ---------------------------------------------------------------------------

def load_model_zs(target_isin: str, peer_isins: list[str]) -> pd.DataFrame:
    all_isins = [target_isin] + peer_isins
    with sqlite3.connect(MODELS_DB) as c:
        latest = c.execute("SELECT MAX(data_date) FROM models").fetchone()[0]
        df = pd.read_sql_query(
            f"SELECT security_id, model_id, model_value, model_value_z FROM models "
            f"WHERE data_date = ? AND security_id IN ({','.join(['?']*len(all_isins))})",
            c, params=[latest] + all_isins,
        )
    df["model"] = df["model_id"].map(MODEL_NAMES).fillna(df["model_id"])
    df.attrs["snap"] = latest
    return df


def load_factor_zs(isin: str) -> pd.DataFrame:
    with sqlite3.connect(FACTORS_DB) as c:
        latest = c.execute("SELECT MAX(data_date) FROM factors").fetchone()[0]
        df = pd.read_sql_query(
            "SELECT factor_id, factor_value, factor_value_z FROM factors "
            "WHERE data_date = ? AND security_id = ?",
            c, params=(latest, isin),
        )
    fref = load_factor_reference()
    df = df.merge(fref[["factor_id", "factor_name", "category", "direction"]],
                  on="factor_id", how="left")
    df.attrs["snap"] = latest
    return df


def load_barra_exposures(isin: str) -> tuple[pd.DataFrame, float, str]:
    with sqlite3.connect(RISK_DB) as c:
        latest = c.execute("SELECT MAX(snapshot_date) FROM factor_exposures").fetchone()[0]
        df = pd.read_sql_query(
            "SELECT factor_id, exposure FROM factor_exposures "
            "WHERE security_id=? AND snapshot_date=?",
            c, params=(isin, latest),
        )
        idio_row = c.execute(
            "SELECT idio_var FROM idiosyncratic_vars WHERE security_id=? AND snapshot_date=?",
            (isin, latest),
        ).fetchone()
    idio_vol = math.sqrt(idio_row[0]) if idio_row else float("nan")
    fref = load_factor_reference()
    name_map = dict(zip(fref["factor_id"], fref["factor_name"]))
    name_map.update({"beta_60d": "Beta (60d)"})
    df["name"] = df["factor_id"].map(name_map).fillna(df["factor_id"])
    df = df[~df["factor_id"].str.startswith("sec_")].copy()
    return df, idio_vol, latest


# ---------------------------------------------------------------------------
# Peer tables
# ---------------------------------------------------------------------------

def compute_peer_fundamentals(target: dict, peers: list[dict]) -> pd.DataFrame:
    rows = []
    with sqlite3.connect(RETURNS_DB) as c:
        for company in [target] + peers:
            t = company["ticker"]
            isin = company["isin"]
            sf = company.get("simfin_id")
            try:
                bl = compute_ltm_block(isin, sf)
            except Exception:
                bl = {}
            if not bl:
                continue
            ltm = bl["ltm"]; prev = bl["prev_ltm"]
            rev = ltm.get("Revenue"); rev_p = prev.get("Revenue")
            gp = ltm.get("GrossProfit"); op = ltm.get("OperatingIncomeLoss")
            ni = ltm.get("NetIncome"); rd = ltm.get("ResearchAndDevelopmentExpenses")
            shares = bl["shares"]
            last = c.execute("SELECT close FROM returns WHERE isin=? ORDER BY date DESC LIMIT 1",
                             (isin,)).fetchone()
            price = last[0] if last else None
            if not (rev and shares and price):
                continue
            mcap = price * shares
            rows.append({
                "Ticker": t,
                "Company": company.get("short_name") or short_name(company.get("company_name"), t),
                "Last Q": f"FY{bl['fy_latest']} {bl['fp_latest']}",
                "Rev LTM ($M)": rev / 1e6,
                "Rev YoY %": (rev / rev_p - 1) * 100 if rev_p else None,
                "Gross Margin %": gp / rev * 100 if gp else None,
                "Op Margin %": op / rev * 100 if op else None,
                "R&D / Rev %": rd / rev * 100 if rd else None,
                "Mkt Cap ($B)": mcap / 1e9,
                "P/S (LTM)": mcap / rev,
            })
    return pd.DataFrame(rows).set_index("Ticker")


def compute_peer_returns(target: dict, peers: list[dict]) -> pd.DataFrame:
    rows = []
    with sqlite3.connect(RETURNS_DB) as c:
        for company in [target] + peers:
            isin = company["isin"]
            df = pd.read_sql_query(
                "SELECT date, close, total_return FROM returns WHERE isin=? ORDER BY date",
                c, params=(isin,),
            )
            df["date"] = pd.to_datetime(df["date"])
            if len(df) < 60:
                continue
            p = perf_block(df)
            rows.append({
                "Ticker": company["ticker"],
                "Company": company.get("short_name") or short_name(company.get("company_name"), company["ticker"]),
                **p,
            })
    return pd.DataFrame(rows).set_index("Ticker")


# ---------------------------------------------------------------------------
# Charts (unchanged from zs_report; parameterised on target ticker)
# ---------------------------------------------------------------------------

def chart_price_drawdown(target_label: str, target_ret: pd.DataFrame,
                         bm: pd.DataFrame) -> str:
    cutoff = target_ret["date"].max() - pd.Timedelta(days=3 * 365)
    z = target_ret[target_ret["date"] >= cutoff].copy()
    z["idx"] = (1 + z["total_return"].fillna(0)).cumprod() * 100
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32],
                        vertical_spacing=0.05,
                        subplot_titles=("Indexed total return (3Y, base 100)",
                                        "Drawdown vs running peak"))
    fig.add_trace(go.Scatter(x=z["date"], y=z["idx"], name=target_label,
                             line=dict(color=COLOR_ZS, width=2.4)), row=1, col=1)
    bm_styles = {"sp500": ("S&P 500", "#444444"),
                 "ai_tech": ("AI / Tech", "#888888"),
                 "russell_1000_growth": ("R1000 Growth", "#bbbbbb")}
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
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"responsive": True})


def chart_model_radar(target_label: str, target_isin: str, models_df: pd.DataFrame) -> str:
    zs = models_df[models_df["security_id"] == target_isin].copy()
    order = MODEL_RADAR_ORDER
    zs = zs[zs["model"].isin(order)].set_index("model").reindex(order)
    z = zs["model_value_z"].fillna(0).tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=z + [z[0]], theta=order + [order[0]],
                                  fill="toself",
                                  line=dict(color=COLOR_ZS, width=2),
                                  fillcolor="rgba(196,78,82,0.18)",
                                  name=f"{target_label} z-score"))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-2.2, 2.2],
                                    tickfont=dict(size=10), gridcolor="#dddddd")),
        showlegend=False, height=420, template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"responsive": True})


def chart_revenue_trend(ltm_block: dict) -> str:
    q = pd.DataFrame(ltm_block.get("q_rev", []))
    if q.empty:
        return "<p class='note'>No revenue history available.</p>"
    q["label"] = q["fy"].astype(str) + " " + q["fp"]
    q["rev_m"] = q["rev"] / 1e6
    q = q.sort_values(["fy", "fp"]).reset_index(drop=True)
    q["yoy"] = q.groupby("fp")["rev"].pct_change()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=q["label"], y=q["rev_m"], name="Revenue ($M)",
                        marker_color="#4C72B0"), secondary_y=False)
    fig.add_trace(go.Scatter(x=q["label"], y=q["yoy"]*100, name="YoY growth",
                            mode="lines+markers",
                            line=dict(color=COLOR_NEG, width=2.5),
                            marker=dict(size=8)), secondary_y=True)
    fig.update_yaxes(title="Revenue ($M)", secondary_y=False)
    fig.update_yaxes(title="YoY growth (%)", ticksuffix="%", secondary_y=True)
    fig.update_xaxes(tickangle=-45)
    fig.update_layout(height=420, template="plotly_white",
                      legend=dict(orientation="h", yanchor="bottom", y=1.05,
                                  xanchor="left", x=0),
                      margin=dict(l=40, r=40, t=40, b=80))
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"responsive": True})


def chart_peer_valuation(target_ticker: str, peer_df: pd.DataFrame,
                         peer_ret: pd.DataFrame) -> str:
    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45],
                        subplot_titles=("P/S (LTM) vs Revenue YoY %",
                                        "1Y total return (cohort)"))
    df = peer_df.dropna(subset=["P/S (LTM)", "Rev YoY %"]).copy()
    if not df.empty:
        colors = [COLOR_ZS if t == target_ticker else COLOR_PEER for t in df.index]
        sizes  = (df["Mkt Cap ($B)"].clip(2, 200)) ** 0.5 * 5
        labels = [df.loc[t].get("Company") or t for t in df.index]
        fig.add_trace(go.Scatter(
            x=df["Rev YoY %"], y=df["P/S (LTM)"],
            mode="markers+text", text=labels, textposition="top center",
            marker=dict(color=colors, size=sizes, line=dict(color="#333", width=1)),
            textfont=dict(size=11), showlegend=False,
        ), row=1, col=1)
    fig.update_xaxes(title="Rev YoY (%)", ticksuffix="%", row=1, col=1)
    fig.update_yaxes(title="P/S (LTM)", row=1, col=1)

    if not peer_ret.empty and "1y" in peer_ret.columns:
        rt = peer_ret.dropna(subset=["1y"]).sort_values("1y").copy()
        bar_colors = [COLOR_ZS if t == target_ticker else COLOR_PEER for t in rt.index]
        y_labels = [rt.loc[t].get("Company") or t for t in rt.index]
        fig.add_trace(go.Bar(x=rt["1y"]*100, y=y_labels, orientation="h",
                            marker_color=bar_colors,
                            text=[f"{v*100:.1f}%" for v in rt["1y"]],
                            textposition="auto", showlegend=False), row=1, col=2)
    fig.update_xaxes(title="1Y total return (%)", ticksuffix="%", row=1, col=2)
    fig.update_layout(height=460, template="plotly_white",
                      margin=dict(l=40, r=40, t=60, b=40))
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"responsive": True})


def chart_factor_bars(fac_df: pd.DataFrame) -> str:
    fac_df = fac_df.dropna(subset=["factor_value_z"]).copy()
    fac_df["signed_z"] = fac_df["factor_value_z"] * fac_df["direction"]
    fac_df = fac_df.sort_values("signed_z")
    colors = [COLOR_POS if v > 0 else COLOR_NEG for v in fac_df["signed_z"]]
    fig = go.Figure(go.Bar(x=fac_df["signed_z"], y=fac_df["factor_name"],
                           orientation="h", marker_color=colors,
                           text=[f"{v:+.2f}" for v in fac_df["signed_z"]],
                           textposition="auto"))
    fig.add_vline(x=0, line=dict(color="#444", width=1))
    fig.update_layout(height=720, template="plotly_white",
                      title="Factor z-scores (direction-adjusted: positive = favourable)",
                      xaxis=dict(title="Direction-adjusted z-score"),
                      margin=dict(l=180, r=40, t=60, b=40))
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"responsive": True})


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

# CSS, fmt_pct, fmt_money, kpi, placeholder all live in report_utils now.


def render_peer_table(target_ticker: str, df: pd.DataFrame) -> str:
    cols = ["Last Q", "Rev LTM ($M)", "Rev YoY %", "Gross Margin %",
            "Op Margin %", "R&D / Rev %", "Mkt Cap ($B)", "P/S (LTM)"]
    rows = []
    for tkr, row in df.iterrows():
        focal = ' class="focal"' if tkr == target_ticker else ""
        name = row.get("Company") or tkr
        cells = [f"<td>{name}</td>"]
        for c in cols:
            v = row[c]
            if pd.isna(v):
                cells.append("<td>—</td>")
            elif c == "Last Q":
                cells.append(f"<td>{v}</td>")
            elif c in ("Rev LTM ($M)", "Mkt Cap ($B)"):
                cells.append(f"<td>{v:,.1f}</td>")
            elif c == "P/S (LTM)":
                cells.append(f"<td>{v:.2f}x</td>")
            else:
                klass = "pos" if v > 0 else "neg"
                cells.append(f'<td class="{klass}">{v:+.1f}%</td>')
        rows.append(f"<tr{focal}>{''.join(cells)}</tr>")
    head = "<tr><th>Company</th>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
    return f"<div class='table-wrap'><table>{head}{''.join(rows)}</table></div>"


def render_return_table(target_ticker: str, df: pd.DataFrame) -> str:
    cols_lbl = [("1m", "1M"), ("3m", "3M"), ("6m", "6M"), ("ytd", "YTD"),
                ("1y", "1Y"), ("3y", "3Y"), ("vol_1y", "1Y vol"),
                ("max_dd_1y", "1Y MaxDD"), ("off_52wh", "Off 52wH")]
    rows = []
    for tkr, row in df.iterrows():
        focal = ' class="focal"' if tkr == target_ticker else ""
        name = row.get("Company") or tkr
        cells = [f"<td>{name}</td>"]
        for k, _ in cols_lbl:
            v = row.get(k)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                cells.append("<td>—</td>")
            else:
                if k == "vol_1y":
                    cells.append(f"<td>{v*100:.1f}%</td>")
                else:
                    klass = "pos" if v > 0 else "neg"
                    cells.append(f'<td class="{klass}">{v*100:+.1f}%</td>')
        rows.append(f"<tr{focal}>{''.join(cells)}</tr>")
    head = "<tr><th>Company</th>" + "".join(f"<th>{l}</th>" for _, l in cols_lbl) + "</tr>"
    return f"<div class='table-wrap'><table>{head}{''.join(rows)}</table></div>"


def render_model_table(target_isin: str, models_df: pd.DataFrame) -> str:
    zs = models_df[models_df["security_id"] == target_isin].set_index("model")
    order = MODEL_TABLE_ORDER
    rows = []
    for m in order:
        if m not in zs.index:
            continue
        z = float(zs.loc[m, "model_value_z"])
        klass = "pos" if z > 0.3 else ("neg" if z < -0.3 else "")
        scale_pos = int(round((z + 2.5) / 5 * 20))
        bar = "■" * max(0, min(20, scale_pos))
        rows.append(f"<tr><td>{m}</td><td class='{klass}'>{z:+.2f}</td>"
                    f"<td style='font-family: monospace; color:#777'>{bar}</td></tr>")
    head = "<tr><th>Model</th><th>z-score</th><th>Distribution</th></tr>"
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
    company = p["company"]
    target_ticker = company["ticker"]
    target_isin = company["isin"]
    short = company.get("short_name") or target_ticker
    ltm = p["ltm_block"].get("ltm", {})
    rev = ltm.get("Revenue"); gp = ltm.get("GrossProfit")
    op = ltm.get("OperatingIncomeLoss"); rd = ltm.get("ResearchAndDevelopmentExpenses")
    perf = p["target_perf"]
    shares = p["ltm_block"].get("shares")
    mcap_b = (perf["last_close"] * shares / 1e9) if shares else None
    ps = (perf["last_close"] * shares / rev) if (shares and rev) else None

    rev_yoy = p["ltm_block"].get("rev_yoy")
    one_y = perf.get("1y")
    off_52wh = perf.get("off_52wh")

    kpi_html = "".join([
        kpi("Last close", f"${perf['last_close']:.2f}", f"as of {perf['dt']}"),
        kpi("Market cap", fmt_money(mcap_b * 1e9, "B") if mcap_b else "—",
            f"{shares/1e6:.1f}M shrs" if shares else ""),
        kpi("P / S (LTM)", f"{ps:.1f}x" if ps else "—",
            f"{company.get('simfin_industry','')}".strip() or company.get("gics_industry", "")),
        kpi("Rev YoY", fmt_pct(rev_yoy, 1) if rev_yoy is not None else "—",
            f"FY{p['ltm_block'].get('fy_latest','?')} {p['ltm_block'].get('fp_latest','')} LTM" if rev_yoy is not None else "",
            cls="pos" if (rev_yoy or 0) > 0.10 else ("neg" if (rev_yoy or 0) < 0 else "")),
        kpi("1Y return", fmt_pct(one_y, 1, True) if one_y is not None else "—",
            f"YTD {fmt_pct(perf.get('ytd'), 1, True)}" if perf.get("ytd") is not None else "",
            cls="neg" if (one_y or 0) < 0 else "pos"),
        kpi("Off 52w high", fmt_pct(off_52wh, 1, True),
            f"max DD {fmt_pct(perf.get('max_dd_1y'), 1)}",
            cls="neg" if (off_52wh or 0) < -0.2 else ("warn" if (off_52wh or 0) < -0.1 else "")),
    ])

    gm_pct = gp/rev*100 if gp and rev else float("nan")
    op_pct = op/rev*100 if op and rev else float("nan")
    rd_pct = rd/rev*100 if rd and rev else float("nan")

    biz = company.get("business_summary") or ""
    if biz and len(biz) > 600:
        biz = biz[:600].rsplit(" ", 1)[0] + "…"

    _base_zs_str = ", ".join(
        f"{name} {z:+.2f}" for name, z in p["base_zs"].items() if not math.isnan(z)
    )
    quant_facts_html = f"""
    <p><strong>Quant snapshot ({p['model_snap']}):</strong>
       Composite Alpha z = <strong>{p['alpha_z']:+.2f}</strong>,
       {_base_zs_str}.
       Barra idiosyncratic vol ~{p['idio_vol']*100:.0f}% annualised · 60d beta {p['beta_60d']:+.2f}.</p>
    <p><strong>Fundamentals (LTM):</strong> revenue {fmt_money(rev, 'M')} ({fmt_pct(rev_yoy, 1, True) if rev_yoy is not None else 'n/a YoY'}),
       gross margin {gm_pct:.1f}%, GAAP op margin {op_pct:.1f}%, R&amp;D intensity {rd_pct:.1f}%
       (last reported quarter: FY{p['ltm_block'].get('fy_latest','?')} {p['ltm_block'].get('fp_latest','?')}).</p>
    <p><strong>Price:</strong> ${perf['last_close']:.2f} as of {perf['dt']} · 1Y total return {fmt_pct(one_y, 1, True)} ·
       YTD {fmt_pct(perf.get('ytd'), 1, True)} · {fmt_pct(off_52wh, 0, True)} from 52w high ·
       1Y realised vol {fmt_pct(perf.get('vol_1y'), 0)} · 1Y max DD {fmt_pct(perf.get('max_dd_1y'), 1)}.</p>
    """

    html = f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><title>{target_ticker} — single-name research</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
{CSS}
<script charset="utf-8" src="https://cdn.plot.ly/plotly-3.5.0.min.js" integrity="sha256-fHbNLP+GlIXN+efbQec78UkemUz3NJp7UmfGxC1tNxs=" crossorigin="anonymous"></script>
</head><body>
<div class="wrap">
<header>
  <h1>{company['company_name'] or target_ticker} ({target_ticker})</h1>
  <div class="sub">{company.get('gics_sector','—')} · {company.get('gics_industry','—')} ·
       price as of {perf['dt']} · model snapshot {p['model_snap']} ·
       generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
  <div class="biz">{biz}</div>
</header>

<div class="kpi-grid">{kpi_html}</div>

<section>
  <h2>Executive summary</h2>
  <div class="lead">What the numbers say, in three minutes.</div>
  {quant_facts_html}
  <div class="placeholder">{placeholder('EXEC_SUMMARY')}</div>
</section>

<section>
  <h2>Price &amp; drawdown</h2>
  <div class="lead">3-year indexed total return vs S&amp;P 500, AI/Tech, R1000 Growth — and the drawdown
       from running peak.</div>
  {p['chart_price']}
</section>

<section>
  <h2>Quant signals</h2>
  <div class="lead">Where {short} sits on the cross-section today, across the 9 models in
       <code>models.db</code>. Direction-adjusted: positive z = favourable.</div>
  <div class="split">
    <div>
      {render_model_table(target_isin, p['models_df'])}
      <p class="note">{ALPHA_BLEND_NOTE}</p>
    </div>
    <div>{p['chart_radar']}</div>
  </div>
</section>

<section>
  <h2>Fundamentals — LTM and trend</h2>
  <div class="lead">Quarterly revenue and YoY growth from EDGAR 10-Q/10-K extracts.
       Q4 derived as FY − Q1 − Q2 − Q3 per pipeline convention.</div>
  {p['chart_revenue']}
  <p class="note">LTM revenue {fmt_money(rev, 'M')} · LTM gross profit {fmt_money(gp, 'M')} ({gm_pct:.1f}%) ·
     LTM GAAP op income {fmt_money(op, 'M')} ({op_pct:.1f}%) · R&amp;D intensity {rd_pct:.1f}%.</p>
</section>

<section>
  <h2>Peer comparison</h2>
  <div class="lead">Industry peers ({company.get('simfin_industry','—')}) ranked by market-cap proximity to
       the target. Focal row highlighted.</div>
  {render_peer_table(target_ticker, p['peer_df'])}
  {p['chart_peer']}
  <h3 style="margin-top:18px; font-size:15px;">Price performance</h3>
  {render_return_table(target_ticker, p['peer_ret'])}
</section>

<section>
  <h2>Factor exposures (Barra) &amp; cross-section</h2>
  <div class="lead">Barra factor model. Idiosyncratic vol ~{p['idio_vol']*100:.0f}% annualised.
       Beta (60d) {p['beta_60d']:+.2f}.</div>
  <div class="split">
    <div>
      <h3 style="margin:0 0 10px; font-size:14px;">Top-magnitude Barra exposures</h3>
      {render_barra_table(p['barra_df'])}
    </div>
    <div>
      <p style="font-size:13px;">Direction-adjusted factor z-scores below — positive means the factor reading
      is favourable for the name relative to the cross-section. Read the largest positive and largest negative
      bars as the key fundamental drivers.</p>
    </div>
  </div>
  <div style="margin-top:18px;">{p['chart_factors']}</div>
</section>

<section>
  <h2>Bull / bear thesis</h2>
  <div class="thesis">
    <div class="col bull">
      <h3><span class="tag bull">BULL</span> Why it could work</h3>
      <ul class="placeholder">{placeholder('BULL_CASE')}</ul>
    </div>
    <div class="col bear">
      <h3><span class="tag bear">BEAR</span> Why it could fail</h3>
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
  <p class="note"><strong>Quant-infra sources:</strong> <code>data/models.db</code> ({p['model_snap']}),
     <code>data/factors.db</code>, <code>data/risk.db</code> (Barra factor model), <code>data/constituents.db</code>
     (EDGAR 10-Q/10-K + SimFin), <code>data/returns.db</code> (Yahoo Finance close + total return).</p>
</section>

<footer>
  Generated by <code>scripts/single_name_report.py</code> · personal research, not investment advice.
  Models are direction-adjusted (positive = favourable). Z-scores are cross-sectional within the
  universe at the snapshot date.
</footer>

</div></body></html>"""
    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticker", required=True, help="Ticker symbol (e.g. ZS)")
    ap.add_argument("--peers", default=None,
                    help="Comma-separated peer tickers. Default: auto-pick from same simfin_industry.")
    ap.add_argument("--peer-count", type=int, default=DEFAULT_PEER_COUNT,
                    help=f"Number of peers to auto-pick (default {DEFAULT_PEER_COUNT})")
    args = ap.parse_args()

    ticker = args.ticker.strip().upper()
    print(f"[{ticker}] Looking up universe.db ...")
    company = lookup_company(ticker)
    print(f"  {company['company_name']} · {company.get('gics_sector','?')} / {company.get('simfin_industry','?')}")

    print(f"[{ticker}] Resolving peer cohort ...")
    peers = resolve_peers(company, args.peers)
    if not peers:
        print(f"  warning: no peers resolved")
    else:
        print(f"  peers ({len(peers)}): " + ", ".join(p["ticker"] for p in peers))

    print(f"[{ticker}] LTM block ...")
    ltm_block = compute_ltm_block(company["isin"], company.get("simfin_id"))

    print(f"[{ticker}] Returns + benchmarks ...")
    target_ret = load_returns(company["isin"])
    bm_start = (target_ret["date"].max() - pd.Timedelta(days=3 * 365 + 90)).strftime("%Y-%m-%d") \
        if not target_ret.empty else "2023-01-01"
    bm = load_benchmarks(start=bm_start)
    target_perf = perf_block(target_ret) if not target_ret.empty else {}

    print(f"[{ticker}] Models / factors / Barra ...")
    peer_isins = [p["isin"] for p in peers]
    models_df = load_model_zs(company["isin"], peer_isins)
    factor_df = load_factor_zs(company["isin"])
    barra_df, idio_vol, barra_snap = load_barra_exposures(company["isin"])

    print(f"[{ticker}] Peer tables ...")
    peer_fund = compute_peer_fundamentals(company, peers)
    peer_ret = compute_peer_returns(company, peers)

    print(f"[{ticker}] Charts ...")
    target_label = company.get("short_name") or ticker
    chart_price = chart_price_drawdown(target_label, target_ret, bm)
    chart_radar = chart_model_radar(target_label, company["isin"], models_df)
    chart_revenue = chart_revenue_trend(ltm_block)
    chart_peer = chart_peer_valuation(ticker, peer_fund, peer_ret)
    chart_factors = chart_factor_bars(factor_df)

    # Pull headline z-scores for the exec-summary block
    zs_models = models_df[models_df["security_id"] == company["isin"]].set_index("model_id")
    def mz(mid: str) -> float:
        return float(zs_models.loc[mid, "model_value_z"]) if mid in zs_models.index else float("nan")
    alpha_z = mz("ALP001")
    _radar_names = set(MODEL_RADAR_ORDER)
    base_zs: dict[str, float] = {
        name: mz(mid) for mid, name in MODEL_NAMES.items() if name in _radar_names
    }

    beta_row = barra_df[barra_df["factor_id"] == "beta_60d"]
    beta_60d = float(beta_row["exposure"].iloc[0]) if not beta_row.empty else float("nan")

    payload = {
        "company": company,
        "ltm_block": ltm_block,
        "target_perf": target_perf,
        "models_df": models_df, "model_snap": models_df.attrs.get("snap", "—"),
        "factor_df": factor_df, "barra_df": barra_df,
        "idio_vol": idio_vol, "barra_snap": barra_snap,
        "peer_df": peer_fund, "peer_ret": peer_ret,
        "chart_price": chart_price, "chart_radar": chart_radar,
        "chart_revenue": chart_revenue, "chart_peer": chart_peer,
        "chart_factors": chart_factors,
        "alpha_z": alpha_z, "base_zs": base_zs, "beta_60d": beta_60d,
    }

    REPORTS_DIR.mkdir(exist_ok=True)
    out_html = REPORTS_DIR / f"{ticker}_report.html"
    html = build_html(payload)
    out_html.write_text(html, encoding="utf-8")
    print(f"[{ticker}] Done → {out_html}")
    print(f"\nNarrative placeholders remaining for /single-name to fill:")
    for ph in NARRATIVE_PLACEHOLDERS:
        print(f"  {{{{{ph}}}}}")


if __name__ == "__main__":
    main()
