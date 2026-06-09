#!/usr/bin/env python3
"""
report_utils.py — Shared utilities for single-name and thematic-basket
HTML report scripts (single_name_report.py, theme_report.py).

Holds:
- DB path constants and reference-data loaders
- Universe lookups (lookup_company, latest_market_cap, short_name)
- Constituents/LTM logic (load_constituent_df, compute_ltm_block, Q4 derivation)
- Returns + perf block (load_returns, load_benchmarks, perf_block)
- Shared CSS string, KPI/format helpers, narrative placeholder convention

These exist outside the project's root `utils.py` because they are
report-specific (HTML rendering, LTM derivation tuned to the EDGAR/SimFin
constituents schema). The root utils.py owns the lower-level shared
primitives (get_db, classify_sector, winsorized_zscore, get_logger).
"""
from __future__ import annotations

import math
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).parent.parent
RETURNS_DB  = ROOT / "data" / "returns.db"
CONST_DB    = ROOT / "data" / "constituents.db"
MODELS_DB   = ROOT / "data" / "models.db"
FACTORS_DB  = ROOT / "data" / "factors.db"
RISK_DB     = ROOT / "data" / "risk.db"
UNIV_DB     = ROOT / "data" / "universe.db"
CMAP_XLSX   = ROOT / "data" / "edgar_concept_map.xlsx"
FREF_CSV    = ROOT / "data" / "factors_reference.csv"
REPORTS_DIR = ROOT / "reports"
MREF_CSV    = ROOT / "data" / "models_reference.csv"

# ---------------------------------------------------------------------------
# Constants — derived dynamically from models_reference.csv so adding/
# renaming models never requires editing this file.
# ---------------------------------------------------------------------------
_mref = pd.read_csv(MREF_CSV)[["ModelID", "Model", "IsComposite"]].drop_duplicates()

# model_id → display name  (e.g. "PROF001" → "Profitability")
MODEL_NAMES: dict[str, str] = dict(zip(_mref["ModelID"], _mref["Model"]))

# Base model display names in CSV order — used for radar charts and filtering
_base_ids = _mref.loc[_mref["IsComposite"] == 0, "ModelID"].tolist()
_comp_ids = _mref.loc[_mref["IsComposite"] == 1, "ModelID"].tolist()
MODEL_RADAR_ORDER: list[str] = [MODEL_NAMES[m] for m in _base_ids]

# Composite-first full order — used for model tables (Alpha at top, then base models)
MODEL_TABLE_ORDER: list[str] = [MODEL_NAMES[m] for m in _comp_ids + _base_ids]

# Human-readable Alpha composition note — derived from Alpha rows in the CSV
def _build_alpha_note() -> str:
    _alpha = pd.read_csv(MREF_CSV)
    rows = _alpha[_alpha["IsComposite"] == 1][["Factors", "Weights"]].drop_duplicates()
    total = rows["Weights"].sum()
    parts = [
        f"{int(round(r['Weights'] / total * 100))}% {MODEL_NAMES.get(r['Factors'], r['Factors'])}"
        for _, r in rows.iterrows()
    ]
    return f"Alpha is the composite blend ({', '.join(parts)})."

ALPHA_BLEND_NOTE: str = _build_alpha_note()

COLOR_ZS   = "#C44E52"
COLOR_PEER = "#4C72B0"
COLOR_POS  = "#2C9F4E"
COLOR_NEG  = "#C44E52"

NARRATIVE_PLACEHOLDERS = ["EXEC_SUMMARY", "BULL_CASE", "BEAR_CASE",
                          "VERDICT_TAG", "VERDICT_BODY", "SOURCES"]

# Share-outstanding constituent IDs to try (EDGAR universal first, SimFin legacy second).
SHARES_CIDS = ("B3C4D5E6",)  # IIG88888 excluded: collides with InterestAndDividendIncome in EDGAR concept map


# ---------------------------------------------------------------------------
# Universe lookups
# ---------------------------------------------------------------------------

_NAME_SUFFIX_RE = re.compile(
    r"[,\s]+(Inc\.?|Corporation|Corp\.?|Company|Co\.?|Holdings?|"
    r"Group|Plc|Ltd\.?|Limited|N\.V\.?|S\.A\.?|AG|SE|AB|"
    r"Class [ABC]|The)$",
    flags=re.IGNORECASE,
)


def short_name(company_name: str | None, ticker: str | None = None) -> str:
    """Drop legal suffixes from company_name. Falls back to ticker if empty."""
    if not company_name:
        return ticker or "?"
    name = company_name.strip()
    for _ in range(2):
        new = _NAME_SUFFIX_RE.sub("", name).strip().rstrip(",").strip()
        if new == name:
            break
        name = new
    if name and name.isupper() and len(name) > 3:
        name = name.title()
    return name or ticker or "?"


def lookup_company(ticker: str) -> dict:
    """Return a dict of company metadata for `ticker`. Raises SystemExit if absent."""
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
    """Most-recent close × most-recent reported shares outstanding."""
    with sqlite3.connect(RETURNS_DB) as c:
        last = c.execute(
            "SELECT close FROM returns WHERE isin=? ORDER BY date DESC LIMIT 1",
            (isin,)).fetchone()
    if not last:
        return None
    price = last[0]
    with sqlite3.connect(CONST_DB) as c:
        sid_list = [isin] + ([str(simfin_id)] if simfin_id else [])
        ph = ",".join("?" * len(sid_list))
        cid_ph = ",".join("?" * len(SHARES_CIDS))
        row = c.execute(
            f"SELECT constituent_value FROM constituents "
            f"WHERE security_id IN ({ph}) AND constituent_id IN ({cid_ph}) "
            f"AND constituent_value IS NOT NULL "
            f"ORDER BY publish_date DESC LIMIT 1",
            sid_list + list(SHARES_CIDS),
        ).fetchone()
    shares = row[0] if row else None
    return price * shares if shares else None


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

def load_concept_map() -> dict[str, str]:
    """constituent_id → EDGAR standard_concept name."""
    df = pd.read_excel(CMAP_XLSX)
    return df.drop_duplicates("constituent_id").set_index("constituent_id")["standard_concept"].to_dict()


def load_factor_reference() -> pd.DataFrame:
    return pd.read_csv(FREF_CSV)


# ---------------------------------------------------------------------------
# Constituents / LTM
# ---------------------------------------------------------------------------

def load_constituent_df(isin: str, simfin_id) -> pd.DataFrame:
    """Pull constituents under both isin and str(simfin_id); dedup by
    (concept, fy, fp) keeping the latest publish_date.
    Mirrors create_factors._dedup."""
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
    """Q4 = FY − Q1 − Q2 − Q3, only if all four are present."""
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
    """Compute LTM revenue/gross-profit/op-income/net-income/R&D plus prior-year-LTM
    plus a quarterly Revenue history for charting."""
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
# Returns / perf
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
            "WHERE index_name IN "
            "  ('sp500','msci_usa','ai_tech','russell_1000','russell_1000_growth') "
            "  AND date >= ?",
            c, params=(start,),
        )
    df["date"] = pd.to_datetime(df["date"])
    return df


def perf_block(ret_df: pd.DataFrame) -> dict:
    """Horizon returns + risk metrics. Requires 'date', 'close', 'total_return' columns."""
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


# ---------------------------------------------------------------------------
# HTML rendering — shared CSS and small helpers
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


def placeholder(name: str) -> str:
    """Markdown-style placeholder for narrative slots filled by the slash command."""
    return f"{{{{{name}}}}}"
