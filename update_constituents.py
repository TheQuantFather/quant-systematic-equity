"""
update_constituents.py — Incremental EDGAR updater for constituents.db.

Normal mode (default): downloads the EDGAR filing index for the last N days,
filters to universe companies by CIK, and processes only companies that actually
filed.  No more rotating through all 936 companies.

Backfill / targeted modes (--fill-gaps, --force, --ticker, --cik): iterate
company-by-company using the existing approach (slower but complete).

Usage:
    python update_constituents.py                    # index mode: last 8 days
    python update_constituents.py --days 30          # index mode: last 30 days
    python update_constituents.py --ticker AAPL      # single company (annual)
    python update_constituents.py --ticker AAPL --quarterly  # single company (quarterly)
    python update_constituents.py --cik 320193       # by SEC CIK
    python update_constituents.py --dry-run          # report without writing
    python update_constituents.py --fill-gaps        # backfill missing fiscal years (FY2019+)
    python update_constituents.py --force            # re-fetch all years, overwriting existing data
    python update_constituents.py --sector-type financial  # filter by sector type
"""

import argparse
import re
import signal
import socket
import sqlite3
import subprocess
import time
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from typing import Generator, Optional

import pandas as pd
from edgar import Company, get_filings, set_identity

from config import CONSTITUENTS_DB, UNIVERSE_DB, CONCEPT_MAP_XLSX
from utils import classify_sector, get_db, get_logger

log = get_logger("update_constituents")

# Earliest fiscal year to fetch. Set to 2017 to support 2019-04-01 factor
# snapshots (needs FY2018 annual data) with one extra year of LTM buffer.
MIN_FISCAL_YEAR = 2017

# Earliest quarterly 10-Q to fetch. EDGAR XBRL quality pre-2021 is inconsistent;
# 2021 gives 5 full years of quarterly history for growth/momentum factors.
MIN_QUARTERLY_FISCAL_YEAR = 2021

set_identity("personal-research shivam3125@gmail.com")

# Cap edgar's bulk HTTP timeout so stuck companies can't hang for 40+ minutes.
# Default: 8 retries × 300s read timeout = 40 min max. New: 3 retries × 45s = 2.25 min max.
import edgar.httprequests as _ehr
import httpx as _httpx
_ehr.BULK_TIMEOUT = _httpx.Timeout(45.0, connect=10.0)
_ehr.BULK_RETRY_ATTEMPTS = 3


@contextmanager
def _time_limit(seconds: int) -> Generator[None, None, None]:
    """Raise TimeoutError if the block takes longer than `seconds`."""
    def _handler(signum: int, frame: object) -> None:
        raise TimeoutError(f"exceeded {seconds}s per-company limit")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ---------------------------------------------------------------------------
# Concept maps — loaded from data/edgar_concept_map.xlsx at startup.
#
# The Excel file has three sheets: "Income Statement", "Balance Sheet",
# "Cash Flow Statement".  Each sheet has columns:
#   standard_concept  — edgartools standard_concept name
#   constituent_id    — target constituent_id in constituents.db
#   description       — human-readable note (ignored at runtime)
#
# Row order within each sheet determines priority: when multiple concepts map
# to the same constituent_id, the row closer to the top wins.  To raise the
# priority of a concept, move its row up in the sheet.
# ---------------------------------------------------------------------------

def load_concept_maps() -> tuple[dict, dict, dict]:
    """
    Load EDGAR concept maps from data/edgar_concept_map.xlsx.
    Returns (income_map, balance_map, cashflow_map) as ordered dicts.
    """
    sheet_keys = {
        "Income Statement":    "income",
        "Balance Sheet":       "balance",
        "Cash Flow Statement": "cashflow",
    }
    maps: dict[str, dict] = {k: {} for k in sheet_keys.values()}
    xl = pd.ExcelFile(CONCEPT_MAP_XLSX)
    for sheet_name, key in sheet_keys.items():
        if sheet_name not in xl.sheet_names:
            raise ValueError(f"Missing sheet '{sheet_name}' in {CONCEPT_MAP_XLSX}")
        df = xl.parse(sheet_name)
        for _, row in df.iterrows():
            sc  = str(row.get("standard_concept", "") or "").strip()
            cid = str(row.get("constituent_id",   "") or "").strip()
            if sc and cid:
                maps[key][sc] = cid   # first occurrence = highest priority
    return maps["income"], maps["balance"], maps["cashflow"]

# Load once at module import so the rest of the code uses _INCOME / _BALANCE / _CASHFLOW
_INCOME, _BALANCE, _CASHFLOW = load_concept_maps()

# Statement type label used in constituents.db
_STMT_LABEL = {
    "income":   "Income Statement",
    "balance":  "Balance Sheet",
    "cashflow": "Cash Flow Statement",
}

# fiscal_period convention matching SimFin / create_factors.py
_FISCAL_PERIOD = {
    "income":   "FY",
    "balance":  "Q4",   # SimFin stores annual balance sheets as Q4
    "cashflow": "FY",
}


# ---------------------------------------------------------------------------
# Universe loading
# ---------------------------------------------------------------------------

def load_company_map(sector_type_filter: Optional[str] = None) -> dict[int, dict]:
    """
    Returns {simfin_id: {isin, ticker, cik, company_name, sector_type, fye_month}} from universe.db companies table.
    Companies without a simfin_id but with a CIK are included using -cik as a synthetic key
    (safe because security_id = isin or str(simfin_id), and all such companies have ISINs).
    sector_type_filter — if provided, only includes companies of that sector type.
    """
    with get_db(UNIVERSE_DB) as conn:
        rows = conn.execute(
            "SELECT simfin_id, isin, ticker, cik, company_name, simfin_sector, simfin_industry, fiscal_year_end "
            "FROM companies WHERE simfin_id IS NOT NULL OR cik IS NOT NULL"
        ).fetchall()
    out: dict[int, dict] = {}
    for simfin_id, isin, ticker, cik, company_name, sector, industry, fye in rows:
        st = classify_sector(sector, industry)
        if sector_type_filter and st != sector_type_filter:
            continue
        key = int(simfin_id) if simfin_id is not None else -int(cik)
        out[key] = {
            "isin":         isin,
            "ticker":       ticker,
            "cik":          int(cik) if cik is not None else None,
            "company_name": company_name,
            "sector_type":  st,
            "fye_month":    int(fye) if fye is not None else 12,
        }
    return out


def build_cik_universe_map(sector_type_filter: Optional[str] = None) -> dict[int, tuple[str, dict]]:
    """
    Returns {cik: (security_id, info)} for all universe companies with a known CIK.
    Used for O(1) lookup when filtering the EDGAR filing index.
    """
    with get_db(UNIVERSE_DB) as conn:
        rows = conn.execute(
            "SELECT simfin_id, isin, ticker, cik, company_name, simfin_sector, simfin_industry, fiscal_year_end "
            "FROM companies WHERE cik IS NOT NULL"
        ).fetchall()
    result: dict[int, tuple[str, dict]] = {}
    for simfin_id, isin, ticker, cik, company_name, sector, industry, fye in rows:
        st = classify_sector(sector, industry)
        if sector_type_filter and st != sector_type_filter:
            continue
        security_id = isin or str(simfin_id)
        info = {
            "isin":         isin,
            "ticker":       ticker,
            "cik":          int(cik),
            "company_name": company_name,
            "sector_type":  st,
            "fye_month":    int(fye) if fye is not None else 12,
        }
        result[int(cik)] = (security_id, info)
    return result


def _quarter_from_period(period: str, fye_month: int = 12) -> tuple[str, int] | None:
    """
    Map period_of_report string to (fiscal_period, fiscal_year) for 10-Q filings.

    fye_month — fiscal year end calendar month (1-12); defaults to 12 (December).
    Uses the company's fiscal year end to correctly identify which fiscal quarter
    each period_of_report falls in, regardless of calendar alignment.

    Allows a 1-month spillover for 52-53 week fiscal year companies whose quarter
    ends sometimes fall one day into the next calendar month (e.g. GD Q1 FY2026
    ended April 5 instead of March 31; STX Q1 FY2026 ended October 3 vs September).

    Returns None for Q4 (period ending in fye_month) — covered by the 10-K fetcher.
    Returns None for unrecognised periods.
    """
    try:
        m = int(period[5:7])
        y = int(period[:4])
    except (ValueError, IndexError):
        return None

    # Calendar months when each fiscal quarter ends:
    #   Q1 ends fye_month-9, Q2 ends fye_month-6, Q3 ends fye_month-3, Q4=fye_month (skip)
    def _qend(offset: int) -> int:
        return ((fye_month - offset - 1) % 12) + 1

    def _next(mo: int) -> int:
        return (mo % 12) + 1

    q1_end = _qend(9)
    q2_end = _qend(6)
    q3_end = _qend(3)

    # Skip Q4 and its 1-month spillover — both covered by 10-K
    if m == fye_month or m == _next(fye_month):
        return None

    # Fiscal year: if the period month falls after the FYE in the calendar year,
    # the period belongs to the fiscal year that ends the following calendar year.
    # Example: AAPL (FYE=Sep), Q1 ends Dec 2024 → fiscal year 2025 (ends Sep 2025).
    fy = y + (1 if m > fye_month else 0)

    if m == q1_end or m == _next(q1_end):
        return ("Q1", fy)
    if m == q2_end or m == _next(q2_end):
        return ("Q2", fy)
    if m == q3_end or m == _next(q3_end):
        return ("Q3", fy)
    return None  # period doesn't align with expected quarter ends — skip


def _latest_expected_sk(today_d: date, fye_month: int) -> int:
    """
    Return the sort_key (fiscal_year*10+q_num) of the most recent quarterly
    filing that should be available by today_d for a company with the given
    fiscal year end month.  Uses a 2-month filing lag (covers the 40-45 day
    SEC deadline for large/accelerated filers).
    """
    def _qend(offset: int) -> int:
        return ((fye_month - offset - 1) % 12) + 1

    q_defs = [(_qend(9), 1), (_qend(6), 2), (_qend(3), 3)]  # (end_month, q_num)
    today_ym = today_d.year * 12 + today_d.month

    best_sk = 0
    for q_month, q_num in q_defs:
        for y in [today_d.year, today_d.year - 1]:
            qend_ym = y * 12 + q_month
            if qend_ym + 2 > today_ym:
                continue  # filing deadline not yet reached
            fy = y + (1 if q_month > fye_month else 0)
            best_sk = max(best_sk, fy * 10 + q_num)

    return best_sk


def get_latest_quarter_per_company(conn: sqlite3.Connection) -> dict[str, int]:
    """
    Returns {security_id: max_sort_key} where sort_key = fiscal_year*10 + quarter_num.
    Used to detect whether new 10-Q data exists beyond what is already stored.
    """
    rows = conn.execute(
        """SELECT security_id,
                  MAX(fiscal_year * 10 + CASE fiscal_period
                      WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 ELSE 0 END)
           FROM constituents
           WHERE fiscal_period IN ('Q1','Q2','Q3')
           GROUP BY security_id"""
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def get_stored_quarters_per_company(conn: sqlite3.Connection) -> dict[str, set[int]]:
    """
    Returns {security_id: {sort_key, ...}} for all stored quarterly rows.
    sort_key = fiscal_year*10 + quarter_num (Q1=1, Q2=2, Q3=3).
    Used for gap-aware quarterly backfill — lets the loop skip stored quarters
    without breaking early, so missing quarters below the max are still fetched.
    """
    rows = conn.execute(
        """SELECT security_id,
                  fiscal_year * 10 + CASE fiscal_period
                      WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 ELSE 0 END
           FROM constituents
           WHERE fiscal_period IN ('Q1','Q2','Q3')"""
    ).fetchall()
    result: dict[str, set[int]] = {}
    for sid, sk in rows:
        result.setdefault(str(sid), set()).add(sk)
    return result


def get_latest_fy_per_company(conn: sqlite3.Connection) -> dict[str, int]:
    """Returns {security_id (str): max fiscal_year (int)} from constituents.db."""
    rows = conn.execute(
        "SELECT security_id, MAX(fiscal_year) FROM constituents "
        "WHERE fiscal_year IS NOT NULL GROUP BY security_id"
    ).fetchall()
    return {str(r[0]): int(r[1]) for r in rows if r[1] is not None}


def get_fy_set_per_company(conn: sqlite3.Connection) -> dict[str, set]:
    """Returns {security_id (str): set of fiscal_years with annual income data in DB}.

    A fiscal year is considered 'covered' when either:
    - A full-year annual filing row exists (fiscal_period='FY', from EDGAR 10-K), or
    - A Q4 standalone income row exists (fiscal_period='Q4', from SimFin quarterly data).

    This prevents fill-gaps from re-fetching years already covered by SimFin Q4 data.
    """
    rows = conn.execute(
        "SELECT security_id, fiscal_year FROM constituents "
        "WHERE fiscal_year IS NOT NULL "
        "  AND fiscal_period IN ('Q4', 'FY') "
        "  AND statement_type = 'Income Statement' "
        "GROUP BY security_id, fiscal_year"
    ).fetchall()
    result: dict[str, set] = {}
    for sid, fy in rows:
        result.setdefault(str(sid), set()).add(int(fy))
    return result


# ---------------------------------------------------------------------------
# XBRL parsing helpers
# ---------------------------------------------------------------------------

# edgartools column names for period data look like "2025-06-30 (Q2)" or "2025-06-30 (H1)".
# We match on the leading date to distinguish them from attribute columns (balance, weight …).
_PERIOD_COL_RE = re.compile(r'^\d{4}-\d{2}-\d{2}')
_STANDALONE_LABELS = frozenset({"Q1", "Q2", "Q3", "Q4"})
# YTD/cumulative period labels edgartools uses (case-insensitive match via upper())
_YTD_LABELS       = frozenset({"H1", "H2", "6M", "9M", "YTD", "TTM"})


def _period_label(col: str) -> str:
    """Extract the period label from '2025-06-30 (Q2)' → 'Q2'. Returns '' if absent."""
    m = re.search(r'\(([^)]+)\)', col)
    return m.group(1).upper() if m else ''


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Remove segment/dimension breakdown rows, keep consolidated totals."""
    if "dimension" in df.columns:
        mask = df["dimension"].isna() | (df["dimension"] == False)  # noqa: E712
        df = df[mask]
    return df


def _select_col(df: pd.DataFrame, prefer_standalone: bool = False) -> Optional[str]:
    """
    Return the best period column from an edgartools statement DataFrame.

    edgartools names period columns as 'YYYY-MM-DD (label)' where label is one of
    Q1–Q4 (standalone quarter), H1/H2/6M/9M/YTD (cumulative), or FY (full year).

    When prefer_standalone=True (quarterly filings), standalone quarter columns are
    preferred over YTD/cumulative columns for the same period-end date.  If only a
    YTD column exists, it is returned with a warning so the caller can handle it.
    """
    period_cols = [c for c in df.columns if _PERIOD_COL_RE.match(c)]
    if not period_cols:
        return None
    if not prefer_standalone:
        return period_cols[0]

    standalone = [c for c in period_cols if _period_label(c) in _STANDALONE_LABELS]
    if standalone:
        return standalone[0]

    # Only YTD/cumulative columns available — return first but caller will log warning
    return period_cols[0]


def _parse_statement(
    stmt,
    concept_map: dict[str, str],
    prefer_standalone: bool = False,
) -> dict[str, float]:
    """
    Extract constituent_id → value from one edgartools statement object.

    Priority is determined by position in concept_map (Excel row order: earlier = higher).
    When multiple rows share the same standard_concept (e.g. a subsidiary row and a
    consolidated row both tagged 'Assets'), the largest absolute value is used — this
    selects the consolidated total for banks where subsidiary rows appear first.

    prefer_standalone=True: for quarterly 10-Q filings, selects a standalone quarter
    column (Q1–Q4) over a YTD/cumulative column (H1, 9M, YTD …) when both exist for
    the same period-end date.  Logs a warning when only YTD is available.
    """
    if stmt is None:
        return {}
    df = _clean_df(stmt.to_dataframe())
    col = _select_col(df, prefer_standalone=prefer_standalone)
    if col is None:
        return {}

    # Warn when a quarterly filing exposes only YTD data
    if prefer_standalone and _period_label(col) in _YTD_LABELS:
        log.warning("[WARN period] only cumulative (%s) column available "
                    "for quarterly context — YTD decomposition will be needed", _period_label(col))

    concept_priority = {sc: i for i, sc in enumerate(concept_map.keys())}

    # Collect candidates per constituent_id: (concept_priority, abs_val, raw_val)
    candidates: dict[str, list] = {}
    for _, row in df.iterrows():
        sc = row.get("standard_concept")
        if not sc or sc not in concept_map:
            continue
        cid  = concept_map[sc]
        prio = concept_priority[sc]
        val  = row.get(col)
        if pd.notna(val) and val != 0:
            candidates.setdefault(cid, []).append((prio, abs(float(val)), float(val)))

    # Lowest priority index wins; ties broken by largest absolute value
    result: dict[str, float] = {}
    for cid, items in candidates.items():
        items.sort(key=lambda x: (x[0], -x[1]))
        result[cid] = items[0][2]
    return result


def _derive_working_capital_change(cf_data: dict) -> Optional[float]:
    """
    Approximate Change in Working Capital by summing available CF components.

    SimFin pre-aggregates this; edgartools exposes components.
    Uses: ChangeInReceivables (6E42C12C) + ChangeInOtherWorkingCapital (BF654FC5)
    """
    recv = cf_data.get("6E42C12C")
    other = cf_data.get("BF654FC5")
    components = [v for v in (recv, other) if v is not None]
    return sum(components) if components else None


def extract_filing_data(
    filing,
    is_quarterly: bool = False,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """
    Parse all three statements from a filing.
    Returns (income_data, balance_data, cashflow_data) — each maps constituent_id → value.
    is_quarterly=True selects standalone quarter columns over YTD cumulative for income/CF.
    """
    xbrl  = filing.xbrl()
    stmts = xbrl.statements

    income_data   = _parse_statement(stmts.income_statement(),    _INCOME,   prefer_standalone=is_quarterly)
    balance_data  = _parse_statement(stmts.balance_sheet(),       _BALANCE)
    cashflow_data = _parse_statement(stmts.cash_flow_statement(), _CASHFLOW, prefer_standalone=is_quarterly)

    # ── Accounting identity corrections ─────────────────────────────────────
    # For banks and other complex filers, XBRL may present a subsidiary/segment
    # row before the consolidated total.  After the priority/abs-value heuristic
    # in _parse_statement the consolidated row should win, but we enforce the
    # two core identities as a safety net:
    #   Total Assets = Total Liabilities & Equity
    #   Total Equity = Total Liabilities & Equity − Total Liabilities
    _ID_ASSETS       = "3BD29B6F"   # Total Assets
    _ID_LAE          = "28CC275C"   # Total Liabilities & Equity
    _ID_LIAB         = "3B25F87A"   # Total Liabilities
    _ID_EQUITY       = "06EF64B2"   # Total Equity
    _ID_CUR_LIAB     = "2B0918F0"   # Total Current Liabilities
    _ID_NONCUR_LIAB  = "D5A1CF3F"   # Total Non-current Liabilities
    _ID_LTD          = "D7815EBF"   # Long-term Debt

    lae    = balance_data.get(_ID_LAE)
    liab   = balance_data.get(_ID_LIAB)
    assets = balance_data.get(_ID_ASSETS)
    equity = balance_data.get(_ID_EQUITY)

    if lae is not None and lae != 0:
        # Assets must equal L&E; if missing or >10% off, override with L&E
        if assets is None or abs(assets / lae - 1) > 0.10:
            balance_data[_ID_ASSETS] = lae
            assets = lae

        # Par-value fingerprint: some filers (utilities especially) expose no
        # stockholders'-equity total and tag only their common-stock PAR value under
        # the CommonEquity concept, leaving `equity` at a tiny figure (e.g. WEC:
        # $3.3M vs ~$14B).  Detect as |equity| < 5% of assets, then rebuild equity
        # from the identity using a COMPLETE liabilities total.
        equity_is_par = equity is not None and assets and abs(equity) < 0.05 * abs(assets)

        if equity is None or equity_is_par:
            recon_liab = liab
            if recon_liab is None:
                cur    = balance_data.get(_ID_CUR_LIAB)
                noncur = balance_data.get(_ID_NONCUR_LIAB)
                ltd    = balance_data.get(_ID_LTD)
                if cur is not None and noncur is not None:
                    candidate = cur + noncur
                    # Reject when the subtotals exclude a separate long-term-debt /
                    # "capitalization" section (candidate < LTD) — that would
                    # understate liabilities and overstate equity (e.g. NI, PPL,
                    # which tag equity correctly and never reach this branch anyway).
                    if ltd is None or candidate >= ltd:
                        recon_liab = candidate
            if recon_liab is not None:
                computed_equity = lae - recon_liab
                if computed_equity > 0.05 * abs(assets):   # accept only a plausible result
                    balance_data[_ID_EQUITY] = computed_equity
                    if liab is None:
                        balance_data[_ID_LIAB] = recon_liab

        # Existing safety net for subsidiary/segment mis-tags (banks): equity present
        # and plausible, but materially off the identity → override from L&E − liab.
        elif liab is not None:
            computed_equity = lae - liab
            if computed_equity != 0 and abs(equity / computed_equity - 1) > 0.50:
                balance_data[_ID_EQUITY] = computed_equity

        # Reconstruct Total Liabilities from the identity when no single
        # "Total liabilities" line is tagged — common for many industrials/tech
        # that report only current + non-current subtotals (e.g. ACN, ETN, JCI,
        # GRMN, NXPI).  Uses the now-finalised equity so Debt-to-Assets and other
        # liability ratios are populated rather than null.
        if balance_data.get(_ID_LIAB) is None:
            eq_final = balance_data.get(_ID_EQUITY)
            if eq_final is not None and (lae - eq_final) > 0:
                balance_data[_ID_LIAB] = lae - eq_final

    # ── Derived / fallback fields ────────────────────────────────────────────
    # Net Income (CDD1D338): filers without minority interest or preferred
    # dividends often tag only "net income attributable to common shareholders"
    # (82FA34CC) and expose no standalone NetIncome concept, leaving Net Income
    # — and every ratio built on it (ROE, ROA, Net Margin, Earnings Yield …) —
    # missing (e.g. WAT, BKNG, PAYX, MNST).  Fall back to NI-to-common, but ONLY
    # when the primary tag is absent so genuine minority-interest filers keep
    # their full net income.
    if "CDD1D338" not in income_data and "82FA34CC" in income_data:
        income_data["CDD1D338"] = income_data["82FA34CC"]

    # Gross Profit (7A1B2BB6): EDGAR XBRL rarely includes a standalone GP tag.
    # Derive from Revenue - Cost of Revenue when both are present.
    # abs(cor): some filers report CoR as a negative offset; abs normalises both conventions.
    if "7A1B2BB6" not in income_data:
        rev = income_data.get("9801FC7E")
        cor = income_data.get("112032A1")
        if rev is not None and cor is not None:
            income_data["7A1B2BB6"] = rev - abs(cor)

    # Working capital change (5B2FCB8E) from CF components
    wc = _derive_working_capital_change(cashflow_data)
    if wc is not None:
        cashflow_data["5B2FCB8E"] = wc

    # Shares fallback: if balance sheet didn't have SharesIssued/SharesYearEnd,
    # use income statement SharesAverage
    if "B3C4D5E6" not in balance_data and "B3C4D5E6" in income_data:
        balance_data["B3C4D5E6"] = income_data["B3C4D5E6"]

    # ── Shares unit correction ───────────────────────────────────────────────
    # Some XBRL filers use decimals=-6 (shares in millions), so edgartools
    # returns e.g. 718 instead of 718,000,000.  All companies in our universe
    # have >= 1,000,000 actual shares, so any parsed value below that threshold
    # is definitively in millions and must be scaled up.
    _ID_SHARES = "B3C4D5E6"
    shares = balance_data.get(_ID_SHARES)
    if shares is not None and 0 < shares < 1_000_000:
        balance_data[_ID_SHARES] = shares * 1_000_000
        if income_data.get(_ID_SHARES) is not None:
            income_data[_ID_SHARES] = income_data[_ID_SHARES] * 1_000_000

    return income_data, balance_data, cashflow_data


# ---------------------------------------------------------------------------
# DB insertion
# ---------------------------------------------------------------------------

def build_rows(
    security_id: str,
    fiscal_year: int,
    publish_date: str,
    report_date: str,
    income_data: dict,
    balance_data: dict,
    cashflow_data: dict,
    period_override: str | None = None,
) -> list[tuple]:
    """
    Format extracted data into rows matching constituents.db schema.
    period_override: if set (e.g. 'Q2'), overrides _FISCAL_PERIOD for all statements.
    Used for 10-Q quarterly inserts where all three statements share the same quarter.
    """
    today = date.today().isoformat()
    rows: list[tuple] = []

    stmt_data = [
        ("income",   income_data),
        ("balance",  balance_data),
        ("cashflow", cashflow_data),
    ]
    for stmt_key, data in stmt_data:
        stmt_label    = _STMT_LABEL[stmt_key]
        fiscal_period = period_override if period_override else _FISCAL_PERIOD[stmt_key]
        for constituent_id, value in data.items():
            rows.append((
                today,           # data_date
                constituent_id,
                security_id,
                value,           # constituent_value
                stmt_label,      # statement_type
                report_date,     # report_date  = period_of_report
                publish_date,    # publish_date = acceptance_datetime (PIT anchor)
                publish_date,    # available_date
                today,           # update_date
                fiscal_year,
                fiscal_period,
                "USD",
            ))
    return rows


def insert_rows(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    conn.executemany(
        """INSERT OR REPLACE INTO constituents
           (data_date, constituent_id, security_id, constituent_value,
            statement_type, report_date, publish_date, available_date,
            update_date, fiscal_year, fiscal_period, currency)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Pull logging — audit trail in constituents.db
# ---------------------------------------------------------------------------

def _init_pull_log(conn: sqlite3.Connection) -> None:
    """Create the pull_log table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pull_log (
            run_id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_timestamp       TEXT NOT NULL,
            completed_at        TEXT,
            duration_seconds    REAL,
            mode                TEXT,
            args_days           INTEGER,
            args_ticker         TEXT,
            args_cik            INTEGER,
            args_sector_type    TEXT,
            args_quarterly      INTEGER,
            args_fill_gaps      INTEGER,
            args_force          INTEGER,
            args_limit          INTEGER,
            args_dry_run        INTEGER,
            universe_size       INTEGER,
            filings_in_index    INTEGER,
            filings_matched     INTEGER,
            companies_checked   INTEGER,
            companies_inserted  INTEGER,
            companies_no_new    INTEGER,
            companies_failed    INTEGER,
            companies_skipped   INTEGER,
            rows_inserted       INTEGER,
            git_hash            TEXT,
            host                TEXT
        )
    """)
    conn.commit()


def _get_git_hash() -> str:
    """Return first 8 chars of current git commit hash, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            capture_output=True, text=True, timeout=3,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _write_pull_log(
    conn: sqlite3.Connection,
    args,
    mode: str,
    universe_size: int,
    start_time: float,
    rows_inserted: int,
    companies_checked: int,
    companies_inserted: int,
    companies_no_new: int,
    companies_failed: int,
    companies_skipped: int,
    filings_in_index: int = 0,
    filings_matched: int = 0,
) -> int:
    """Initialise log table if needed, write one run-level row, return run_id."""
    _init_pull_log(conn)
    cur = conn.execute(
        """INSERT INTO pull_log (
               run_timestamp, completed_at, duration_seconds, mode,
               args_days, args_ticker, args_cik, args_sector_type,
               args_quarterly, args_fill_gaps, args_force, args_limit, args_dry_run,
               universe_size, filings_in_index, filings_matched,
               companies_checked, companies_inserted, companies_no_new,
               companies_failed, companies_skipped, rows_inserted,
               git_hash, host
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.fromtimestamp(start_time).isoformat(timespec="seconds"),
            datetime.now().isoformat(timespec="seconds"),
            round(time.time() - start_time, 1),
            mode,
            getattr(args, "days", None),
            getattr(args, "ticker", None),
            getattr(args, "cik", None),
            getattr(args, "sector_type", None),
            1 if getattr(args, "quarterly", False) else 0,
            1 if getattr(args, "fill_gaps", False) else 0,
            1 if getattr(args, "force", False) else 0,
            getattr(args, "limit", None),
            1 if getattr(args, "dry_run", False) else 0,
            universe_size,
            filings_in_index,
            filings_matched,
            companies_checked,
            companies_inserted,
            companies_no_new,
            companies_failed,
            companies_skipped,
            rows_inserted,
            _get_git_hash(),
            socket.gethostname(),
        ),
    )
    conn.commit()
    return cur.lastrowid


# ---------------------------------------------------------------------------
# EDGAR index-mode helpers
# ---------------------------------------------------------------------------

def _publish_date(filing) -> str:
    """
    Return the PIT publish date for a Filing object — always the SEC acceptance
    datetime so the same filing maps to one publish_date regardless of how it was
    fetched.  This matters for after-hours filings, where acceptance_datetime is
    the prior calendar day vs filing_date: mixing the two creates duplicate rows
    (different publish_date = different PK) that distort point-in-time selection.

    Company.get_filings() objects expose acceptance_datetime directly; index-mode
    Filing objects from get_filings() do not, so we read it from the filing header
    (one request).  filing_date is only a last-resort fallback.
    """
    dt = getattr(filing, "acceptance_datetime", None)
    if dt:
        return str(dt)[:10]
    try:
        hdr_dt = filing.header.acceptance_datetime
        if hdr_dt:
            return str(hdr_dt)[:10]
    except Exception:
        pass
    return str(filing.filing_date)[:10]


def fetch_recent_filings(form_type: str, days_back: int) -> list:
    """
    Fetch EDGAR filing index for form_type filed in the last days_back calendar days.
    Returns a list of Filing objects from the edgar library.
    """
    since     = (date.today() - timedelta(days=days_back)).isoformat()
    today_str = date.today().isoformat()
    filings   = get_filings(form=form_type, filing_date=f"{since}:{today_str}")
    return list(filings) if filings else []


def process_filing_annual(
    filing,
    security_id: str,
    ticker: str,
    latest_fy: int | None,
    conn: sqlite3.Connection,
    fye_month: int = 12,
    dry_run: bool = False,
) -> int:
    """
    Insert annual data from a single pre-fetched 10-K Filing object.
    Returns number of rows inserted (0 on error or if already in DB).
    """
    period = filing.period_of_report
    if not period:
        return 0
    try:
        fy = int(period[:4])
        period_month = int(period[5:7])
    except ValueError:
        return 0
    # 52/53-week fiscal year spillover: a Dec-FYE company whose year-end lands in
    # early January (e.g. Jan 1–3) should still be labelled as the prior year.
    # Same 1-month window used by _quarter_from_period for quarterly filings.
    if fye_month == 12 and period_month == 1:
        fy -= 1
    if fy < MIN_FISCAL_YEAR:
        return 0
    if latest_fy is not None and fy <= latest_fy:
        return 0

    publish_date = _publish_date(filing)
    report_date  = period

    if dry_run:
        log.info("[%s] would fetch FY%s  (filed %s)", ticker, fy, publish_date)
        return 0

    try:
        income_d, balance_d, cashflow_d = extract_filing_data(filing)
    except Exception as exc:
        log.warning("[%s] FY%s XBRL parse error: %s", ticker, fy, exc)
        return 0

    rows = build_rows(security_id, fy, publish_date, report_date, income_d, balance_d, cashflow_d)
    if rows:
        insert_rows(conn, rows)
        log.info("[%s] FY%s  %d rows  (filed %s)", ticker, fy, len(rows), publish_date)
    return len(rows)


def process_filing_quarterly(
    filing,
    security_id: str,
    ticker: str,
    stored_sks: set[int] | None,
    conn: sqlite3.Connection,
    fye_month: int = 12,
    dry_run: bool = False,
) -> int:
    """
    Insert quarterly data from a single pre-fetched 10-Q Filing object.
    Returns number of rows inserted (0 on error or if already in DB).
    """
    period = filing.period_of_report
    if not period:
        return 0
    qinfo = _quarter_from_period(period, fye_month)
    if qinfo is None:
        return 0  # Q4 — skip, covered by 10-K

    q_label, q_fy = qinfo
    period_num = {"Q1": 1, "Q2": 2, "Q3": 3}[q_label]
    sort_key   = q_fy * 10 + period_num

    min_sort_key = MIN_QUARTERLY_FISCAL_YEAR * 10 + 1
    if sort_key < min_sort_key:
        return 0
    if stored_sks is not None and sort_key in stored_sks:
        return 0

    publish_date = _publish_date(filing)
    report_date  = period

    if dry_run:
        log.info("[%s] would fetch %s FY%s  (filed %s)", ticker, q_label, q_fy, publish_date)
        return 0

    try:
        income_d, balance_d, cashflow_d = extract_filing_data(filing, is_quarterly=True)
    except Exception as exc:
        log.warning("[%s] %s FY%s XBRL parse error: %s", ticker, q_label, q_fy, exc)
        return 0

    rows = build_rows(security_id, q_fy, publish_date, report_date, income_d, balance_d, cashflow_d, period_override=q_label)
    if rows:
        insert_rows(conn, rows)
        log.info("[%s] %s FY%s  %d rows  (filed %s)", ticker, q_label, q_fy, len(rows), publish_date)
    return len(rows)


# ---------------------------------------------------------------------------
# Company-by-company helpers (backfill / targeted modes)
# ---------------------------------------------------------------------------

def resolve_company(info: dict) -> Optional[Company]:
    """Look up a company on EDGAR, preferring CIK over ticker."""
    cik    = info.get("cik")
    ticker = info.get("ticker")
    name   = info.get("company_name", "")

    if cik:
        try:
            return Company(int(cik))
        except Exception:
            pass
    if ticker:
        try:
            return Company(ticker)
        except Exception:
            pass
    # Last resort: name search
    try:
        return Company(name)
    except Exception:
        return None


def process_company(
    simfin_id: int,
    info: dict,
    latest_fy: Optional[int],
    conn: sqlite3.Connection,
    dry_run: bool = False,
    fill_gaps: bool = False,
    existing_fys: Optional[set] = None,
) -> int:
    """
    Fetch and insert annual data for one company.

    Normal mode: only fetches fiscal years newer than latest_fy.
    fill_gaps mode: fetches any year in MIN_FISCAL_YEAR..current_year that is
      not already in existing_fys, so historical gaps are filled without
      re-fetching years that already have data.

    Returns the number of rows inserted (0 if nothing new or on error).
    """
    security_id = info.get("isin") or str(simfin_id)   # prefer ISIN, fall back to simfin_id
    ticker      = info.get("ticker", "?")

    co = resolve_company(info)
    if co is None:
        log.warning("[%s] EDGAR lookup failed — skipping", ticker)
        return 0

    filings = co.get_filings(form="10-K")
    if not filings or len(filings) == 0:
        return 0

    fye_month = info.get("fye_month", 12)
    inserted = 0
    for filing in list(filings):
        period   = filing.period_of_report        # e.g. '2024-09-28'
        if not period:
            continue
        try:
            fy           = int(period[:4])
            period_month = int(period[5:7])
        except ValueError:
            continue
        # 52/53-week spillover: Dec-FYE company whose year-end lands in January
        if fye_month == 12 and period_month == 1:
            fy -= 1

        if fy < MIN_FISCAL_YEAR:
            break  # don't go further back than the earliest snapshot needs

        if fill_gaps:
            # Skip years we already have; keep iterating to find older gaps
            if existing_fys and fy in existing_fys:
                continue
        else:
            if latest_fy is not None and fy <= latest_fy:
                break  # incremental mode: stop once we hit existing data

        publish_date = _publish_date(filing)
        report_date  = period

        if dry_run:
            log.info("[%s] would fetch FY%s  (filed %s)", ticker, fy, publish_date)
            continue

        try:
            income_d, balance_d, cashflow_d = extract_filing_data(filing)
        except Exception as exc:
            log.warning("[%s] FY%s XBRL parse error: %s", ticker, fy, exc)
            continue

        rows = build_rows(
            security_id, fy, publish_date, report_date,
            income_d, balance_d, cashflow_d,
        )
        # Persist each filing as it is fetched, not once at the end: a filing is a
        # self-contained unit, so a mid-loop timeout or network error keeps the
        # years already retrieved instead of discarding the whole company's batch.
        insert_rows(conn, rows)
        inserted += len(rows)
        log.info("[%s] FY%s  %d rows  (filed %s)", ticker, fy, len(rows), publish_date)

    return inserted


def process_company_quarterly(
    simfin_id: int,
    info: dict,
    stored_sort_keys: set[int],
    conn: sqlite3.Connection,
    dry_run: bool = False,
) -> int:
    """
    Fetch and insert quarterly 10-Q data for one company.

    Stores balance sheet, income statement, and cash flow with
    fiscal_period = 'Q1'/'Q2'/'Q3' derived from the period_of_report date,
    adjusted for the company's fiscal year end month (info['fye_month']).

    Note: edgartools parses the primary period context from XBRL, which for
    income statement / cash flow is typically the 3-month standalone quarter.
    If a filer reports only YTD cumulative values in XBRL, the stored values
    will be YTD; create_factors.py's select_ltm_data() sums 4 standalone
    quarters — inspect with validate_constituents.py if factors look off.

    Returns the number of rows inserted (0 if nothing new or on error).
    """
    security_id = info.get("isin") or str(simfin_id)
    ticker      = info.get("ticker", "?")
    fye_month   = info.get("fye_month", 12)

    co = resolve_company(info)
    if co is None:
        return 0

    filings = co.get_filings(form="10-Q")
    if not filings or len(filings) == 0:
        return 0

    min_sort_key = MIN_QUARTERLY_FISCAL_YEAR * 10 + 1

    inserted = 0
    for filing in list(filings):
        period = filing.period_of_report
        if not period:
            continue

        qinfo = _quarter_from_period(period, fye_month)
        if qinfo is None:
            continue  # FYE quarter — skip, covered by 10-K

        q_label, q_fy = qinfo
        period_num = {"Q1": 1, "Q2": 2, "Q3": 3}[q_label]
        sort_key = q_fy * 10 + period_num

        if sort_key < min_sort_key:
            break  # too old — stop iterating (filings are sorted newest-first)

        if sort_key in stored_sort_keys:
            continue  # already stored — skip without breaking, to fill any gaps below

        publish_date = _publish_date(filing)
        report_date  = period

        if dry_run:
            log.info("[%s] would fetch %s FY%s  (filed %s)", ticker, q_label, q_fy, publish_date)
            continue

        try:
            income_d, balance_d, cashflow_d = extract_filing_data(filing, is_quarterly=True)
        except Exception as exc:
            log.warning("[%s] %s FY%s XBRL parse error: %s", ticker, q_label, q_fy, exc)
            continue

        rows = build_rows(
            security_id, q_fy, publish_date, report_date,
            income_d, balance_d, cashflow_d,
            period_override=q_label,
        )
        # Persist each filing as it is fetched (see process_company): keeps the
        # quarters already retrieved if a later one times out.
        insert_rows(conn, rows)
        inserted += len(rows)
        log.info("[%s] %s FY%s  %d rows  (filed %s)", ticker, q_label, q_fy, len(rows), publish_date)

    return inserted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Incremental EDGAR updater for constituents.db"
    )
    parser.add_argument("--ticker",  metavar="TICKER",
                        help="Update a single company by ticker (backfill mode)")
    parser.add_argument("--cik",     metavar="CIK", type=int,
                        help="Update a single company by SEC CIK (backfill mode)")
    parser.add_argument("--days",    metavar="N",   type=int, default=8,
                        help="Index mode: look back N calendar days for new filings (default 8)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be fetched without writing to DB")
    parser.add_argument("--force",   action="store_true",
                        help="Re-fetch all years >= MIN_FISCAL_YEAR, overwriting existing data "
                             "(use after fixing the concept map or parsing bugs)")
    parser.add_argument("--fill-gaps", action="store_true", dest="fill_gaps",
                        help="Fetch only the specific fiscal years missing from the DB "
                             "(leaves existing rows untouched; efficient for FY2019 backfill)")
    parser.add_argument("--sector-type", metavar="TYPE", dest="sector_type",
                        choices=["general", "financial", "reit"],
                        help="Only process companies of this sector type")
    parser.add_argument("--quarterly", action="store_true",
                        help="Backfill mode only: fetch 10-Q quarterly filings instead of annual 10-K")
    parser.add_argument("--limit",   metavar="N",   type=int,
                        help="Backfill mode only: cap at N companies")
    parser.add_argument("--timeout", metavar="N",   type=int, default=90,
                        help="Backfill mode: per-company timeout in seconds before skipping (default 90)")
    args = parser.parse_args()

    use_index_mode = not (args.ticker or args.cik or args.fill_gaps or args.force)

    start_time: float = time.time()

    # Determine log mode label
    if use_index_mode:
        log_mode = "index"
    elif args.ticker or args.cik:
        log_mode = "targeted_quarterly" if args.quarterly else "targeted_annual"
    elif args.fill_gaps:
        log_mode = "fill_gaps_quarterly" if args.quarterly else "fill_gaps"
    elif args.force:
        log_mode = "force_quarterly" if args.quarterly else "force"
    else:
        log_mode = "bulk_quarterly" if args.quarterly else "bulk_annual"

    if use_index_mode:
        # ── EDGAR filing index mode ─────────────────────────────────────────
        # Download the EDGAR index, filter to universe, process only actual filers.
        cik_map = build_cik_universe_map(sector_type_filter=args.sector_type)
        log.info("Universe: %s companies with known CIK", f"{len(cik_map):,}")
        log.info("Mode: index (last %d days)", args.days)

        with get_db(CONSTITUENTS_DB) as conn:
            latest_fy_map  = get_latest_fy_per_company(conn)
            stored_q_map_i = get_stored_quarters_per_company(conn)

        total_rows = 0
        total_checked = 0
        co_inserted = 0
        co_no_new   = 0
        co_failed   = 0
        filings_in_index = 0
        filings_matched  = 0

        # ── Annual 10-K ──────────────────────────────────────────────────────
        log.info("Fetching 10-K index (last %d days) ...", args.days)
        annual_filings = fetch_recent_filings("10-K", args.days)
        annual_matched = [
            (f, cik_map[int(f.cik)][0], cik_map[int(f.cik)][1])
            for f in annual_filings
            if f.cik and int(f.cik) in cik_map
        ]
        filings_in_index += len(annual_filings)
        filings_matched  += len(annual_matched)
        log.info("%d 10-K filings in index, %d match universe", len(annual_filings), len(annual_matched))

        with get_db(CONSTITUENTS_DB) as conn:
            for filing, security_id, info in annual_matched:
                ticker    = info.get("ticker", "?")
                isin      = info.get("isin")
                latest_fy = latest_fy_map.get(isin) or latest_fy_map.get(security_id)
                total_checked += 1
                fye_month = info.get("fye_month", 12)
                try:
                    n = process_filing_annual(
                        filing, security_id, ticker, latest_fy, conn,
                        fye_month=fye_month,
                        dry_run=args.dry_run,
                    )
                    if n > 0:
                        co_inserted += 1
                    else:
                        co_no_new += 1
                except Exception as exc:
                    log.warning("[%s] skipped — %s: %s", ticker, type(exc).__name__, exc)
                    co_failed += 1
                    n = 0
                total_rows += n

        # ── Quarterly 10-Q ───────────────────────────────────────────────────
        log.info("Fetching 10-Q index (last %d days) ...", args.days)
        quarterly_filings = fetch_recent_filings("10-Q", args.days)
        quarterly_matched = [
            (f, cik_map[int(f.cik)][0], cik_map[int(f.cik)][1])
            for f in quarterly_filings
            if f.cik and int(f.cik) in cik_map
        ]
        filings_in_index += len(quarterly_filings)
        filings_matched  += len(quarterly_matched)
        log.info("%d 10-Q filings in index, %d match universe", len(quarterly_filings), len(quarterly_matched))

        with get_db(CONSTITUENTS_DB) as conn:
            for filing, security_id, info in quarterly_matched:
                ticker    = info.get("ticker", "?")
                isin      = info.get("isin")
                stored_sks = (
                    stored_q_map_i.get(isin, set()) | stored_q_map_i.get(security_id, set())
                )
                fye_month = info.get("fye_month", 12)
                total_checked += 1
                try:
                    n = process_filing_quarterly(
                        filing, security_id, ticker, stored_sks or None, conn,
                        fye_month=fye_month,
                        dry_run=args.dry_run,
                    )
                    if n > 0:
                        co_inserted += 1
                    else:
                        co_no_new += 1
                except Exception as exc:
                    log.warning("[%s] skipped — %s: %s", ticker, type(exc).__name__, exc)
                    co_failed += 1
                    n = 0
                total_rows += n

        action = "would insert" if args.dry_run else "inserted"
        log.info("Done — %d universe filings checked, %s rows %s.",
                 total_checked, f"{total_rows:,}", action)

        with get_db(CONSTITUENTS_DB) as log_conn:
            run_id = _write_pull_log(
                log_conn, args, log_mode, len(cik_map), start_time,
                rows_inserted=total_rows,
                companies_checked=total_checked,
                companies_inserted=co_inserted,
                companies_no_new=co_no_new,
                companies_failed=co_failed,
                companies_skipped=0,
                filings_in_index=filings_in_index,
                filings_matched=filings_matched,
            )
        log.info("Run logged — run_id=%s", run_id)

    else:
        # ── Company-by-company mode (backfill / targeted) ───────────────────
        company_map = load_company_map(sector_type_filter=args.sector_type)

        with get_db(CONSTITUENTS_DB) as conn:
            latest_fy_map     = get_latest_fy_per_company(conn)
            fy_set_map        = get_fy_set_per_company(conn) if args.fill_gaps else {}
            latest_q_map      = get_latest_quarter_per_company(conn) if args.quarterly else {}
            stored_q_map      = get_stored_quarters_per_company(conn) if args.quarterly else {}

        mode_label = "quarterly (10-Q)" if args.quarterly else "annual (10-K)"
        log.info("Universe: %s companies  |  mode: %s", f"{len(company_map):,}", mode_label)
        log.info("Companies with existing data: %s", f"{len(latest_fy_map):,}")

        # Build candidate list
        if args.ticker:
            candidates = [
                (sid, info)
                for sid, info in company_map.items()
                if (info.get("ticker") or "").upper() == args.ticker.upper()
            ]
            if not candidates:
                log.error("Ticker %r not found in universe.", args.ticker)
                return
        elif args.cik:
            candidates = [
                (sid, info)
                for sid, info in company_map.items()
                if info.get("cik") == args.cik
            ]
            if not candidates:
                log.error("CIK %s not found in universe.", args.cik)
                return
        else:
            # Prioritise companies already in constituents.db (known good), then zero-data companies.
            isins_in_db = set(latest_fy_map.keys())
            in_db     = [(sid, info) for sid, info in company_map.items()
                         if info.get("isin") in isins_in_db]
            not_in_db = [(sid, info) for sid, info in company_map.items()
                         if info.get("isin") not in isins_in_db]
            candidates = in_db + not_in_db

        if args.limit:
            candidates = candidates[: args.limit]

        if args.dry_run:
            log.info("[DRY RUN] Checking %s companies ...", f"{len(candidates):,}")
        else:
            log.info("Processing %s companies ...", f"{len(candidates):,}")

        total_rows = 0
        stale_count = 0
        companies_skipped = 0
        co_inserted = 0
        co_no_new   = 0
        co_failed   = 0

        with get_db(CONSTITUENTS_DB) as conn:
            for i, (simfin_id, info) in enumerate(candidates):
                # Look up latest_fy by ISIN (new) then fall back to simfin_id str (legacy rows)
                isin = info.get("isin")
                latest_fy = (
                    latest_fy_map.get(isin) if isin else None
                ) or latest_fy_map.get(str(simfin_id))
                ticker = info.get("ticker", "?")

                if args.force:
                    latest_fy = None   # treat as if no existing data → re-fetch all years

                # Determine which fiscal years this company is missing (fill-gaps mode)
                existing_fys: Optional[set] = None
                if args.fill_gaps and not args.quarterly:
                    # Annual fill-gaps only: skip companies with complete FY coverage.
                    # When --quarterly is also set, the quarterly path has its own skip
                    # logic (EDGAR latest_sk check) so we must not skip here.
                    existing_fys = (
                        fy_set_map.get(isin or "", set()) | fy_set_map.get(str(simfin_id), set())
                    )
                    current_year = date.today().year
                    needed = set(range(MIN_FISCAL_YEAR, current_year + 1))
                    missing = needed - existing_fys
                    if not missing:
                        companies_skipped += 1
                        continue  # nothing to fill for this company

                elif not args.fill_gaps and not args.quarterly:
                    # Quick check: is this company likely stale?
                    current_year = date.today().year
                    if latest_fy is not None and latest_fy >= current_year - 1 and not args.ticker and not args.cik:
                        companies_skipped += 1
                        continue  # already up to date

                if args.quarterly:
                    # EDGAR-first: check only ISIN-keyed stored data (not SimFin-keyed).
                    # SimFin coverage must not block EDGAR from fetching the same quarters.
                    latest_sk = latest_q_map.get(isin) if isin else None
                    # Skip if EDGAR already has the most recent expected quarter AND
                    # no internal gaps exist in the backfill window. Without the gap
                    # check, companies whose latest_sk was advanced by a recent fetch
                    # (e.g. Q1 FY2026) silently skip over missing older quarters
                    # (e.g. Q3 FY2025) that never got fetched.
                    today_d     = date.today()
                    fye_month   = info.get("fye_month", 12)
                    expected_sk = _latest_expected_sk(today_d, fye_month)
                    if not args.ticker and not args.cik and latest_sk is not None and latest_sk >= expected_sk:
                        min_gap_sk = MIN_QUARTERLY_FISCAL_YEAR * 10 + 1  # mirrors process_company_quarterly
                        stored_sk_set = stored_q_map.get(isin, set()) if isin else set()
                        all_expected = {
                            y * 10 + q
                            for y in range(min_gap_sk // 10, latest_sk // 10 + 1)
                            for q in (1, 2, 3)
                            if min_gap_sk <= y * 10 + q <= latest_sk
                        }
                        if all_expected <= stored_sk_set:
                            companies_skipped += 1
                            continue  # EDGAR up to date, no internal gaps

                stale_count += 1
                if i > 0 and i % 10 == 0:
                    time.sleep(0.5)   # light throttle every 10 companies

                try:
                    with _time_limit(args.timeout):
                        if args.quarterly:
                            # Use ISIN-keyed stored sort_keys only — SimFin-keyed quarters must not
                            # block EDGAR from fetching (EDGAR-first strategy).
                            # --force clears even EDGAR-stored keys to allow a full re-fetch.
                            stored_sk = set() if args.force else stored_q_map.get(isin, set())
                            n = process_company_quarterly(
                                simfin_id, info, stored_sk, conn,
                                dry_run=args.dry_run,
                            )
                        else:
                            n = process_company(
                                simfin_id, info, latest_fy, conn,
                                dry_run=args.dry_run,
                                fill_gaps=args.fill_gaps,
                                existing_fys=existing_fys,
                            )
                    if n > 0:
                        co_inserted += 1
                    else:
                        co_no_new += 1
                except TimeoutError:
                    ticker = info.get("ticker", str(simfin_id))
                    log.warning("[%s] timed out after %ds — skipping", ticker, args.timeout)
                    co_failed += 1
                    n = 0
                except Exception as exc:
                    ticker = info.get("ticker", str(simfin_id))
                    log.warning("[%s] skipped — %s: %s", ticker, type(exc).__name__, exc)
                    co_failed += 1
                    n = 0
                total_rows += n

        if args.dry_run:
            log.info("[DRY RUN] %d stale companies identified.", stale_count)
        else:
            log.info("Done — inserted %s rows across %d companies.", f"{total_rows:,}", stale_count)

        with get_db(CONSTITUENTS_DB) as log_conn:
            run_id = _write_pull_log(
                log_conn, args, log_mode, len(company_map), start_time,
                rows_inserted=total_rows,
                companies_checked=stale_count,
                companies_inserted=co_inserted,
                companies_no_new=co_no_new,
                companies_failed=co_failed,
                companies_skipped=companies_skipped,
            )
        log.info("Run logged — run_id=%s", run_id)


if __name__ == "__main__":
    main()
