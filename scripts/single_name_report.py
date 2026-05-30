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

ROOT       = Path(__file__).parent.parent
RETURNS_DB = ROOT / "data" / "returns.db"
CONST_DB   = ROOT / "data" / "constituents.db"
MODELS_DB  = ROOT / "data" / "models.db"
FACTORS_DB = ROOT / "data" / "factors.db"
RISK_DB    = ROOT / "data" / "risk.db"
UNIV_DB    = ROOT / "data" / "universe.db"
CMAP_XLSX  = ROOT / "data" / "edgar_concept_map.xlsx"
FREF_CSV   = ROOT / "data" / "factors_reference.csv"
REPORTS_DIR = ROOT / "reports"

MODEL_NAMES = {
    "ALP001": "Alpha (Composite)",
    "QUAL001": "Quality",
    "VAL001": "Value",
    "GRO001": "Growth",
    "MOM001": "Momentum",
    "SIZ001": "Size",
    "LVOL001": "Low Volatility",
    "LIQ001": "Liquidity",
    "SHI001": "Short Interest",
    "LTR001": "LT Reversal",
}

COLOR_ZS   = "#C44E52"
COLOR_PEER = "#4C72B0"
COLOR_POS  = "#2C9F4E"
COLOR_NEG  = "#C44E52"

DEFAULT_PEER_COUNT = 7
NARRATIVE_PLACEHOLDERS = ["EXEC_SUMMARY", "BULL_CASE", "BEAR_CASE",
                          "VERDICT_TAG", "VERDICT_BODY", "SOURCES"]


# ---------------------------------------------------------------------------
# Universe lookups
# ---------------------------------------------------------------------------

_NAME_SUFFIX_RE = None
def short_name(company_name: str | None, ticker: str | None = None) -> str:
    """Drop legal suffixes from company_name. Falls back to ticker if empty."""
    if not company_name:
        return ticker or "?"
    import re
    global _NAME_SUFFIX_RE
    if _NAME_SUFFIX_RE is None:
        _NAME_SUFFIX_RE = re.compile(
            r"[,\s]+(Inc\.?|Corporation|Corp\.?|Company|Co\.?|Holdings?|"
            r"Group|Plc|Ltd\.?|Limited|N\.V\.?|S\.A\.?|AG|SE|AB|"
            r"Class [ABC]|The)$",
            flags=re.IGNORECASE,
        )
    name = company_name.strip()
    # Strip up to 2 trailing suffixes (e.g. "X Holdings, Inc.")
    for _ in range(2):
        new = _NAME_SUFFIX_RE.sub("", name).strip().rstrip(",").strip()
        if new == name:
            break
        name = new
    # Title-case if SHOUTING — preserves mixed-case like "CrowdStrike" or "PayPal"
    if name and name.isupper() and len(name) > 3:
        name = name.title()
    return name or ticker or "?"


def lookup_company(ticker: str) -> dict:
    with sqlite3.connect(UNIV_DB) as c:
        row = c.execute(
            "SELECT isin, ticker, company_name, gics_sector, gics_industry, "
            "       simfin_id, simfin_industry, business_summary "
            "FROM companies WHERE ticker = ?",
            (ticker,),
        ).fetchone()
    if not row:
        raise SystemExit(f"Ticker {ticker} not found in universe.db")
    keys = ["isin", "ticker", "company_name", "gics_sector",
            "gics_industry", "simfin_id", "simfin_industry", "business_summary"]
    out = dict(zip(keys, row))
    out["short_name"] = short_name(out["company_name"], out["ticker"])
    return out


def latest_market_cap(isin: str, simfin_id) -> float | None:
    with sqlite3.connect(RETURNS_DB) as c:
        last = c.execute("SELECT close FROM returns WHERE isin=? ORDER BY date DESC LIMIT 1",
                         (isin,)).fetchone()
    if not last:
        return None
    price = last[0]
    # Try to find shares from latest constituent SharesAverage
    with sqlite3.connect(CONST_DB) as c:
        sid_list = [isin] + ([str(simfin_id)] if simfin_id else [])
        ph = ",".join("?" * len(sid_list))
        row = c.execute(
            f"SELECT constituent_value FROM constituents "
            f"WHERE security_id IN ({ph}) AND constituent_id = ? "
            f"AND constituent_value IS NOT NULL "
            f"ORDER BY publish_date DESC LIMIT 1",
            sid_list + ["IIG88888"],  # IIG88888 = SharesAverage in concept map
        ).fetchone()
    shares = row[0] if row else None
    return price * shares if shares else None


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
# Constituents / LTM
# ---------------------------------------------------------------------------

def load_concept_map() -> dict[str, str]:
    df = pd.read_excel(CMAP_XLSX)
    return df.drop_duplicates("constituent_id").set_index("constituent_id")["standard_concept"].to_dict()


def load_factor_reference() -> pd.DataFrame:
    return pd.read_csv(FREF_CSV)


def load_constituent_df(isin: str, simfin_id) -> pd.DataFrame:
    """Pull constituents under both isin and str(simfin_id), then dedup by
    (concept, fy, fp) keeping the latest publish_date — mirrors create_factors._dedup."""
    sid_list = [isin] + ([str(simfin_id)] if simfin_id else [])
    ph = ",".join("?" * len(sid_list))
    with sqlite3.connect(CONST_DB) as c:
        df = pd.read_sql_query(
            f"SELECT security_id, constituent_id, constituent_value, fiscal_year, "
            f"       fiscal_period, publish_date "
            f"FROM constituents WHERE security_id IN ({ph})",
            c, params=sid_list,
        )
    if df.empty:
        return df
    cmap = load_concept_map()
    df["concept"] = df["constituent_id"].map(cmap)
    df = (df.sort_values("publish_date")
            .drop_duplicates(["concept", "fiscal_year", "fiscal_period"], keep="last"))
    return df


def quarter_value(df: pd.DataFrame, concept: str, fy: int, fp: str) -> float | None:
    rows = df[(df["concept"] == concept) & (df["fiscal_year"] == fy) & (df["fiscal_period"] == fp)]
    return float(rows["constituent_value"].iloc[0]) if not rows.empty else None


def derive_q4(df: pd.DataFrame, concept: str, fy: int) -> float | None:
    fy_val = quarter_value(df, concept, fy, "FY")
    q1 = quarter_value(df, concept, fy, "Q1")
    q2 = quarter_value(df, concept, fy, "Q2")
    q3 = quarter_value(df, concept, fy, "Q3")
    if all(x is not None for x in [fy_val, q1, q2, q3]):
        return fy_val - q1 - q2 - q3
    return None


def get_latest_quarter(df: pd.DataFrame) -> tuple[int, str] | None:
    q_rows = df[df["fiscal_period"].isin(["Q1", "Q2", "Q3", "Q4"])].copy()
    if q_rows.empty:
        return None
    q_rows["qkey"] = q_rows["fiscal_year"] * 10 + q_rows["fiscal_period"].str[1].astype(int)
    last = q_rows.sort_values("qkey").iloc[-1]
    return int(last["fiscal_year"]), str(last["fiscal_period"])


def trailing_4q(fy_latest: int, fp_latest: str) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    q_idx = int(fp_latest[1])
    fy = fy_latest
    for _ in range(4):
        out.append((fy, f"Q{q_idx}"))
        q_idx -= 1
        if q_idx == 0:
            q_idx = 4
            fy -= 1
    return out


def sum_concept_over(df: pd.DataFrame, concept: str, quarters: list[tuple[int, str]]) -> float | None:
    total = 0.0
    for fy, fp in quarters:
        v = quarter_value(df, concept, fy, fp)
        if v is None and fp == "Q4":
            v = derive_q4(df, concept, fy)
        if v is None:
            return None
        total += v
    return total


def compute_ltm_block(isin: str, simfin_id) -> dict:
    df = load_constituent_df(isin, simfin_id)
    if df.empty:
        return {}
    last = get_latest_quarter(df)
    if last is None:
        return {}
    fy_l, fp_l = last
    q4 = trailing_4q(fy_l, fp_l)
    q4_prev = [(fy - 1, fp) for fy, fp in q4]

    metrics = ["Revenue", "GrossProfit", "OperatingIncomeLoss", "NetIncome",
               "ResearchAndDevelopmentExpenses"]
    cur = {m: sum_concept_over(df, m, q4) for m in metrics}
    prev = {m: sum_concept_over(df, m, q4_prev) for m in metrics}

    shares = quarter_value(df, "SharesAverage", fy_l, fp_l)
    if shares is None:
        # fallback to latest available shares
        sh = df[df["concept"] == "SharesAverage"].sort_values(["fiscal_year", "fiscal_period"])
        shares = float(sh["constituent_value"].iloc[-1]) if not sh.empty else None

    rev_yoy = (cur["Revenue"] / prev["Revenue"] - 1) if cur["Revenue"] and prev["Revenue"] else None

    q_rev = []
    for fy in range(fy_l - 4, fy_l + 1):
        for fp in ["Q1", "Q2", "Q3", "Q4"]:
            v = quarter_value(df, "Revenue", fy, fp)
            if v is None and fp == "Q4":
                v = derive_q4(df, "Revenue", fy)
            if v is not None:
                q_rev.append({"fy": fy, "fp": fp, "rev": v})

    return {
        "fy_latest": fy_l, "fp_latest": fp_l,
        "ltm": cur, "prev_ltm": prev,
        "shares": shares, "rev_yoy": rev_yoy, "q_rev": q_rev,
    }


# ---------------------------------------------------------------------------
# Returns / risk / quant snapshots
# ---------------------------------------------------------------------------

def load_returns(isin: str) -> pd.DataFrame:
    with sqlite3.connect(RETURNS_DB) as c:
        df = pd.read_sql_query(
            "SELECT date, close, total_return, volume FROM returns WHERE isin = ? ORDER BY date",
            c, params=(isin,),
        )
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_benchmarks(start: str) -> pd.DataFrame:
    with sqlite3.connect(RETURNS_DB) as c:
        df = pd.read_sql_query(
            "SELECT index_name, date, close, total_return FROM benchmark_returns "
            "WHERE index_name IN ('sp500','msci_usa','ai_tech','russell_1000_growth') "
            "  AND date >= ?",
            c, params=(start,),
        )
    df["date"] = pd.to_datetime(df["date"])
    return df


def perf_block(ret_df: pd.DataFrame) -> dict:
    out = {}
    def cum(n: int) -> float:
        return float((1 + ret_df.tail(n)["total_return"]).prod() - 1)
    out["1m"] = cum(21) if len(ret_df) >= 21 else None
    out["3m"] = cum(63) if len(ret_df) >= 63 else None
    out["6m"] = cum(126) if len(ret_df) >= 126 else None
    out["1y"] = cum(252) if len(ret_df) >= 252 else None
    out["3y"] = cum(252 * 3) if len(ret_df) >= 252 * 3 else None
    ytd_mask = ret_df["date"] >= "2026-01-01"
    out["ytd"] = float((1 + ret_df[ytd_mask]["total_return"]).prod() - 1) if ytd_mask.any() else None
    out["vol_1y"] = float(ret_df.tail(252)["total_return"].std() * np.sqrt(252)) if len(ret_df) >= 60 else None
    cum_ret = (1 + ret_df.tail(252)["total_return"]).cumprod()
    out["max_dd_1y"] = float((cum_ret / cum_ret.cummax() - 1).min()) if len(ret_df) >= 60 else None
    out["last_close"] = float(ret_df["close"].iloc[-1])
    out["52w_high"] = float(ret_df.tail(252)["close"].max())
    out["52w_low"] = float(ret_df.tail(252)["close"].min())
    out["off_52wh"] = out["last_close"] / out["52w_high"] - 1
    out["dt"] = ret_df["date"].iloc[-1].strftime("%Y-%m-%d")
    return out


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
    return fig.to_html(full_html=False, include_plotlyjs="cdn",
                       config={"responsive": True})


def chart_model_radar(target_label: str, target_isin: str, models_df: pd.DataFrame) -> str:
    zs = models_df[models_df["security_id"] == target_isin].copy()
    order = ["Quality", "Value", "Growth", "Momentum", "Size", "Low Volatility",
             "Liquidity", "Short Interest", "LT Reversal"]
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

CSS = """
<style>
:root {
  --bg: #f7f7f8; --card: #fff; --ink: #1a1a1a; --muted: #666;
  --line: #e5e5e7; --accent: #C44E52; --pos: #2C9F4E; --neg: #C44E52;
  --warn: #C7913A;
}
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Arial, sans-serif;
       background: var(--bg); color: var(--ink); margin: 0; padding: 0; line-height: 1.55; }
.wrap { max-width: 1180px; margin: 0 auto; padding: 32px 28px 64px; }
header { border-bottom: 1px solid var(--line); padding-bottom: 18px; margin-bottom: 26px; }
header h1 { margin: 0 0 4px; font-size: 30px; letter-spacing: -0.01em; }
header .sub { color: var(--muted); font-size: 14px; }
header .biz { color: var(--muted); font-size: 13px; margin-top: 6px; max-width: 900px; }
.kpi-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; margin: 22px 0 8px; }
.kpi { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; }
.kpi .lbl { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }
.kpi .val { font-size: 22px; font-weight: 600; margin-top: 4px; font-feature-settings: "tnum"; }
.kpi .sub { color: var(--muted); font-size: 11px; margin-top: 2px; }
.kpi.pos .val { color: var(--pos); } .kpi.neg .val { color: var(--neg); }
.kpi.warn .val { color: var(--warn); }
section { background: var(--card); border: 1px solid var(--line); border-radius: 12px;
          padding: 22px 24px; margin-top: 22px; }
section h2 { margin: 0 0 6px; font-size: 19px; }
section .lead { color: var(--muted); font-size: 13px; margin-bottom: 14px; }
.split { display: grid; grid-template-columns: 1fr 1fr; gap: 22px; }
@media (max-width: 880px) {
  .split, .thesis { grid-template-columns: 1fr; }
  .kpi-grid { grid-template-columns: repeat(3, 1fr); }
}
@media (max-width: 560px) {
  .wrap { padding: 18px 14px 40px; }
  header h1 { font-size: 22px; }
  header .sub { font-size: 12px; }
  .kpi-grid { grid-template-columns: repeat(2, 1fr); gap: 8px; }
  .kpi { padding: 10px 12px; }
  .kpi .val { font-size: 18px; }
  section { padding: 16px 14px; }
  section h2 { font-size: 17px; }
  .verdict { padding: 14px 16px; }
  th, td { padding: 6px 8px; font-size: 12px; }
}
.table-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; }
@media (max-width: 560px) { .table-wrap table { min-width: 540px; } }
table { border-collapse: collapse; width: 100%; font-size: 13px; font-feature-settings: "tnum"; }
th, td { padding: 8px 10px; text-align: right; border-bottom: 1px solid var(--line); }
th:first-child, td:first-child { text-align: left; }
th { background: #f0f0f1; font-weight: 600; color: #333; }
tr.focal { background: #fff2f2; }
tr.focal td:first-child { color: var(--accent); font-weight: 600; }
.pos { color: var(--pos); } .neg { color: var(--neg); }
.tag { display: inline-block; font-size: 11px; padding: 2px 8px; border-radius: 999px;
       border: 1px solid var(--line); color: var(--muted); margin-right: 6px; }
.tag.bull { background: #eaf6ec; color: #1f6f3c; border-color: #c4e2cc; }
.tag.bear { background: #fbe9ea; color: #8c2b2f; border-color: #f0c4c6; }
.thesis { display: grid; grid-template-columns: 1fr 1fr; gap: 22px; }
.thesis .col { padding: 16px; border-radius: 10px; }
.thesis .bull { background: #f3faf5; border: 1px solid #c8e6cf; }
.thesis .bear { background: #fdf3f4; border: 1px solid #ecc7c9; }
.thesis h3 { margin: 0 0 8px; font-size: 15px; }
.thesis ul { margin: 0; padding-left: 18px; }
.thesis li { margin: 6px 0; font-size: 14px; }
.verdict { background: #fffbea; border: 1px solid #f0deaa; padding: 18px 22px;
           border-radius: 12px; margin-top: 22px; }
.verdict h3 { margin: 0 0 6px; }
.verdict .pill { display: inline-block; padding: 3px 10px; border-radius: 999px;
                 font-weight: 600; font-size: 12px; background: var(--accent);
                 color: #fff; margin-right: 8px; }
footer { font-size: 11px; color: var(--muted); margin-top: 26px;
         border-top: 1px solid var(--line); padding-top: 12px; }
.sources a { color: #336; text-decoration: none; }
.sources a:hover { text-decoration: underline; }
.sources li { margin: 4px 0; font-size: 12px; }
.note { font-size: 12px; color: var(--muted); margin-top: 8px; }
.placeholder { color: #999; font-style: italic; }
</style>
"""


def fmt_pct(x: float | None, dec: int = 1, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x*100:+.{dec}f}%" if signed else f"{x*100:.{dec}f}%"


def fmt_money(x: float | None, scale: str = "M") -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"${x/{'M': 1e6, 'B': 1e9}[scale]:,.1f}{scale}"


def kpi(label: str, val: str, sub: str = "", cls: str = "") -> str:
    return (f'<div class="kpi {cls}"><div class="lbl">{label}</div>'
            f'<div class="val">{val}</div>'
            f'{f"<div class=\"sub\">{sub}</div>" if sub else ""}</div>')


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
    order = ["Alpha (Composite)", "Quality", "Value", "Growth", "Momentum",
             "Size", "Low Volatility", "Liquidity", "Short Interest", "LT Reversal"]
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


def placeholder(name: str) -> str:
    """Markdown-style placeholder the slash command fills in."""
    return f"{{{{{name}}}}}"


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

    quant_facts_html = f"""
    <p><strong>Quant snapshot ({p['model_snap']}):</strong>
       Composite Alpha z = <strong>{p['alpha_z']:+.2f}</strong>,
       Quality {p['qual_z']:+.2f}, Value {p['val_z']:+.2f},
       Growth {p['gro_z']:+.2f}, Momentum {p['mom_z']:+.2f}.
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
      <p class="note">Alpha is the composite blend (30% Quality, 20% Value, 20% Growth, 20% Momentum, 10% Size).</p>
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
    def mz(mid):
        return float(zs_models.loc[mid, "model_value_z"]) if mid in zs_models.index else float("nan")
    alpha_z = mz("ALP001"); qual_z = mz("QUAL001"); val_z = mz("VAL001")
    gro_z = mz("GRO001"); mom_z = mz("MOM001")

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
        "alpha_z": alpha_z, "qual_z": qual_z, "val_z": val_z,
        "gro_z": gro_z, "mom_z": mom_z, "beta_60d": beta_60d,
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
