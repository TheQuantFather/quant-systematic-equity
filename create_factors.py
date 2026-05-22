#!/usr/bin/env python3
"""
create_factors.py — Point-in-time factor snapshots, written to factors.db.

Each snapshot date produces one cross-section:
  - Only financial data with publish_date <= snapshot_date is used (no look-ahead).
  - Per company, the most recent annual report available by that date is chosen.
  - Prices are referenced as of snapshot_date.
  - Z-scores are computed cross-sectionally within (data_date, factor_id).

Usage:
  python create_factors.py                     # defaults to today
  python create_factors.py --date 2025-04-01   # single snapshot
  python create_factors.py --backfill          # all predefined historical dates

BACKFILL_DATES cover April 1 following each fiscal year end, giving all companies
at least 90 days to file their annual reports before the snapshot is taken.

Run order:
  create_databases.py → create_returns.py → create_factors.py → create_models.py
"""

import argparse
import sqlite3
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from typing import Optional

from config import (
    UNIVERSE_DB, CONSTITUENTS_DB, RETURNS_DB, FACTORS_DB,
    FACTORS_REF, CONSTITUENTS_REF,
    BACKFILL_DATES, QUARTERLY_BACKFILL_DATES,
)
from utils import classify_sector, get_db, get_logger, winsorized_zscore

log = get_logger("create_factors")

_PERIOD_ORDER = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_universe() -> dict:
    """Returns {isin (str): {company_name, simfin_sector, simfin_industry, sector_type, ticker}} for all companies."""
    with get_db(UNIVERSE_DB) as conn:
        rows = conn.execute(
            "SELECT isin, company_name, simfin_sector, simfin_industry, ticker "
            "FROM companies"
        ).fetchall()
    log.info("Universe: %s companies", f"{len(rows):,}")
    universe = {}
    for isin, company_name, sector, industry, ticker in rows:
        universe[isin] = {
            'company_name': company_name,
            'industry':     industry,
            'sector':       sector,
            'sector_type':  classify_sector(sector, industry),
            'ticker':       ticker or '',
        }
    return universe


def load_ticker_map() -> dict:
    """Returns {isin (str): ticker (str)} from universe.db."""
    with get_db(UNIVERSE_DB) as conn:
        rows = conn.execute("SELECT isin, ticker FROM companies WHERE ticker IS NOT NULL").fetchall()
    return {isin: ticker for isin, ticker in rows if ticker}


def load_snapshot_isins(snapshot: date) -> set | None:
    """
    Returns the set of ISINs in the Russell 1000 at the closest available
    universe_snapshots date to `snapshot`.  Returns None if the table is empty
    (fallback: use all companies in the companies table).
    """
    date_str = snapshot.strftime('%Y-%m-%d')
    with get_db(UNIVERSE_DB) as conn:
        matched = conn.execute(
            "SELECT snapshot_date FROM universe_snapshots WHERE index_name = 'russell_1000' "
            "ORDER BY ABS(julianday(snapshot_date) - julianday(?)) LIMIT 1",
            (date_str,)
        ).fetchone()
        if not matched:
            return None
        matched_date = matched[0]
        rows = conn.execute(
            "SELECT isin FROM universe_snapshots WHERE snapshot_date = ? AND index_name = 'russell_1000'",
            (matched_date,)
        ).fetchall()
    if matched_date != date_str:
        log.info("No universe_snapshots for %s — using closest (%s, %s companies)",
                 date_str, matched_date, f"{len(rows):,}")
    return {r[0] for r in rows}


def _fix_ytd_quarters(data: dict, name_to_kind: dict[str, str]) -> int:
    """
    Detect and decompose YTD-cumulative Flow values stored as quarterly rows.

    Heuristic: for a Flow item with positive Q1, if Q2_stored / Q1 > 1.65 the
    Q2 row is likely 6M cumulative (H1 YTD). Decompose in-place:
        Q2_standalone = Q2_stored − Q1
        Q3_standalone = Q3_stored − Q2_stored  (Q2_stored is the YTD base)

    Threshold 1.65 allows genuine seasonal Q2 uplifts (up to 65% above Q1)
    before treating the value as cumulative.  Q3 uses the same Q1 denominator
    (9M/Q1 ≈ 3 for uniform distributions) — not Q2_stored/H1 ≈ 1.5 which would
    never clear the 1.65 bar.

    Returns count of values corrected.
    """
    fixed = 0
    for sid_data in data.values():
        years = {fy for (fy, _) in sid_data}
        for fy in years:
            q1_dict = sid_data.get((fy, 'Q1'), {})
            q2_dict = sid_data.get((fy, 'Q2'))
            q3_dict = sid_data.get((fy, 'Q3'))
            if not q2_dict:
                continue
            for name in list(q2_dict):
                if name.startswith('_') or name_to_kind.get(name) != 'Flow':
                    continue
                v1 = q1_dict.get(name)
                v2 = q2_dict.get(name)
                if v1 is None or v2 is None or v1 <= 0:
                    continue
                if v2 / v1 > 1.65:
                    q2_stored = v2
                    q2_dict[name] = v2 - v1
                    fixed += 1
                    if q3_dict is not None:
                        v3 = q3_dict.get(name)
                        if v3 is not None and v3 / v1 > 1.65:
                            q3_dict[name] = v3 - q2_stored
                            fixed += 1
    return fixed


def load_constituent_data() -> dict:
    """
    Returns {security_id: {(fiscal_year, fiscal_period): quarter_dict}} where
    quarter_dict = {constituent_name: value, '_publish_date': pd.Timestamp,
                    '_sort_key': int}.

    sort_key = fiscal_year * 10 + period_num (Q1=1 … Q4=4) — used for ordering.
    For restated data, the row with the latest publish_date wins per
    (security_id, constituent_id, fiscal_year, fiscal_period).

    YTD decomposition: some EDGAR filers report only 6M/9M cumulative values in
    XBRL. _fix_ytd_quarters() detects these via Q2/Q1 ratio > 1.65 and converts
    them to standalone quarters in-place (must run before Q4 derivation).

    Q4 standalone derivation: EDGAR 10-Q filings cover Q1–Q3; Q4 income/cashflow
    is only available from the annual 10-K (stored as fiscal_period='FY').  When
    Q4 is absent but FY and all three of Q1+Q2+Q3 are present, Q4 is derived for
    each Flow constituent using the exact accounting identity:
        Q4 = FY − Q1 − Q2 − Q3
    Balance sheet items (Stock kind) are not derived — the 10-K already stores
    them directly as fiscal_period='Q4'.

    Annual-only filers: SimFin annual-only coverage (e.g. LAZ, ELV, LH, COR)
    stores income/CF as fiscal_period='FY' with no Q1/Q2/Q3 breakdown.  Their
    Q4 balance-sheet rows ARE loaded.  After Q4 derivation, FY Flow values are
    assigned directly to the Q4 bucket.  select_ltm_data() then detects the
    annual-only pattern (consecutive Q4 sort-key gaps ≥10) and uses the single
    most-recent annual period as the full LTM rather than summing multiple years.
    """
    ref = pd.read_csv(CONSTITUENTS_REF)
    id_to_name   = dict(zip(ref['constituent_id'], ref['constituent_name']))
    id_to_kind   = dict(zip(ref['constituent_id'], ref['data_kind']))
    name_to_kind = dict(zip(ref['constituent_name'], ref['data_kind']))

    # Build simfin_id → isin mapping and universe ISIN set from universe.db
    with get_db(UNIVERSE_DB) as conn_u:
        id_map = pd.read_sql_query(
            "SELECT CAST(simfin_id AS TEXT) AS simfin_id, isin FROM companies "
            "WHERE simfin_id IS NOT NULL",
            conn_u,
        )
        all_isins = set(
            r[0] for r in conn_u.execute("SELECT isin FROM companies WHERE isin IS NOT NULL").fetchall()
        )
    simfin_to_isin = dict(zip(id_map['simfin_id'], id_map['isin']))

    def _map_sid(sid: str) -> Optional[str]:
        return simfin_to_isin.get(sid) or (sid if sid in all_isins else None)

    def _load_and_clean(query: str, conn) -> pd.DataFrame:
        df = pd.read_sql_query(query, conn)
        df['constituent_name'] = df['constituent_id'].map(id_to_name)
        df['data_kind']        = df['constituent_id'].map(id_to_kind)
        df = df.dropna(subset=['constituent_name'])
        df['security_id'] = df['security_id'].astype(str).apply(_map_sid)
        df = df.dropna(subset=['security_id'])
        df['fiscal_year']  = df['fiscal_year'].astype(int)
        df['publish_date'] = pd.to_datetime(df['publish_date'])
        return df

    with get_db(CONSTITUENTS_DB) as conn:
        # Primary: quarterly rows (Q1–Q4) — covers SimFin + EDGAR 10-Q
        df_q = _load_and_clean(
            "SELECT security_id, constituent_id, constituent_value, "
            "       fiscal_year, fiscal_period, publish_date "
            "FROM constituents "
            "WHERE fiscal_period IN ('Q1','Q2','Q3','Q4') "
            "  AND publish_date IS NOT NULL",
            conn,
        )
        # Annual: income + cashflow stored as FY from EDGAR 10-K
        # Balance sheet rows from 10-K are already stored as Q4 — not needed here.
        df_fy = _load_and_clean(
            "SELECT security_id, constituent_id, constituent_value, "
            "       fiscal_year, fiscal_period, publish_date "
            "FROM constituents "
            "WHERE fiscal_period = 'FY' "
            "  AND statement_type IN ('Income Statement', 'Cash Flow Statement') "
            "  AND publish_date IS NOT NULL",
            conn,
        )

    # For restated data, keep the latest-published value per
    # (security_id, constituent_id, fiscal_year, fiscal_period)
    def _dedup(df: pd.DataFrame) -> pd.DataFrame:
        return (df.sort_values('publish_date')
                  .groupby(['security_id', 'constituent_id', 'fiscal_year', 'fiscal_period'],
                           as_index=False)
                  .last())

    df_q  = _dedup(df_q)
    df_fy = _dedup(df_fy)

    df_q['sort_key'] = df_q['fiscal_year'] * 10 + df_q['fiscal_period'].map(_PERIOD_ORDER)

    data: dict = {}
    for row in df_q.itertuples(index=False):
        sid = row.security_id
        key = (row.fiscal_year, row.fiscal_period)
        bucket = data.setdefault(sid, {}).setdefault(key, {
            '_publish_date': row.publish_date,
            '_sort_key':     row.sort_key,
        })
        if row.publish_date > bucket['_publish_date']:
            bucket['_publish_date'] = row.publish_date
        bucket[row.constituent_name] = row.constituent_value

    # Fix YTD-cumulative quarterly values before Q4 derivation (order matters).
    ytd_fixed = _fix_ytd_quarters(data, name_to_kind)
    if ytd_fixed:
        log.info("[YTD fix] decomposed %s Q2/Q3 Flow values from cumulative to standalone", f"{ytd_fixed:,}")

    # Derive Q4 standalone for companies where Q4 income/cashflow is absent.
    # Q4 = FY − Q1 − Q2 − Q3 (exact accounting identity for Flow items).
    # Only applied when ALL of Q1, Q2, Q3 carry the constituent for that year.
    derived_q4 = 0
    for row in df_fy.itertuples(index=False):
        sid = row.security_id
        fy  = row.fiscal_year
        if sid not in data:
            continue
        sid_data = data[sid]
        name = row.constituent_name
        if id_to_kind.get(row.constituent_id) != 'Flow':
            continue  # Stock items not applicable
        q4_key = (fy, 'Q4')
        if name in sid_data.get(q4_key, {}):
            continue  # this specific Flow constituent already in Q4 bucket
        q1 = sid_data.get((fy, 'Q1'), {})
        q2 = sid_data.get((fy, 'Q2'), {})
        q3 = sid_data.get((fy, 'Q3'), {})
        if not (q1 and q2 and q3):
            continue  # can't derive Q4 without all three prior quarters
        # Temporal guard: Q1/Q2/Q3 must be published before the annual filing.
        # If any were published after the annual, they belong to the next fiscal year
        # (common for January/February fiscal-year-end companies like NVDA, WMT, HD
        # where EDGAR labels the next FY's quarters under the same fiscal_year as the
        # annual — e.g. NVDA FY2025 annual pub=Feb-2025 vs FY2026 Q1 pub=May-2025).
        fy_pub = row.publish_date
        if (q1.get('_publish_date', pd.Timestamp.max) > fy_pub or
                q2.get('_publish_date', pd.Timestamp.max) > fy_pub or
                q3.get('_publish_date', pd.Timestamp.max) > fy_pub):
            continue
        v1 = q1.get(name)
        v2 = q2.get(name)
        v3 = q3.get(name)
        if v1 is None or v2 is None or v3 is None:
            continue  # constituent missing in at least one prior quarter
        q4_val = row.constituent_value - v1 - v2 - v3
        q4_sort = fy * 10 + _PERIOD_ORDER['Q4']
        bucket = sid_data.setdefault(q4_key, {
            '_publish_date': row.publish_date,
            '_sort_key':     q4_sort,
        })
        if row.publish_date > bucket['_publish_date']:
            bucket['_publish_date'] = row.publish_date
        bucket[name] = q4_val
        derived_q4 += 1

    if derived_q4:
        log.info("[Q4 derivation] derived %s Q4 Flow values from FY − (Q1+Q2+Q3)", f"{derived_q4:,}")

    # Annual-only filers: companies whose income/cash-flow is stored only as
    # fiscal_period='FY' (no Q1/Q2/Q3 quarterly breakdown — e.g. SimFin
    # annual-only coverage for LAZ, ELV, LH, COR, etc.).  Their Q4 balance
    # sheet rows ARE present in the quarterly data (data[sid][(fy, 'Q4')]),
    # but Flow items are absent.  Assign FY Flow values directly to the Q4
    # bucket so that select_ltm_data() can surface them.
    #
    # Restricted to companies with NO Q1/Q2/Q3 data in ANY year — this
    # excludes companies that have quarterly coverage for recent years but
    # only annual coverage for older years (mixing would overstate LTM by
    # summing an annual Q4 bucket together with quarterly periods).
    annual_only_sids = {
        sid for sid, qdata in data.items()
        if not any(fp in ('Q1', 'Q2', 'Q3') for (_, fp) in qdata)
    }
    annual_direct = 0
    for row in df_fy.itertuples(index=False):
        sid  = row.security_id
        fy   = row.fiscal_year
        name = row.constituent_name
        if id_to_kind.get(row.constituent_id) != 'Flow':
            continue
        if sid not in annual_only_sids:
            continue
        sid_data = data[sid]
        # Skip if this specific Flow constituent already present in Q4
        if sid_data.get((fy, 'Q4'), {}).get(name) is not None:
            continue
        q4_key = (fy, 'Q4')
        if q4_key not in sid_data:
            continue  # no balance-sheet anchor year — skip
        bucket = sid_data[q4_key]
        bucket[name] = row.constituent_value
        if row.publish_date > bucket['_publish_date']:
            bucket['_publish_date'] = row.publish_date
        annual_direct += 1

    if annual_direct:
        log.info("[annual-only] assigned %s FY Flow values to Q4 for annual-only filers", f"{annual_direct:,}")

    _fix_shares_units(data)
    log.info("Loaded quarterly constituent data for %s companies", f"{len(data):,}")
    return data


def _fix_shares_units(data: dict) -> None:
    """
    Correct unit errors in 'Shares (Basic)' in place using log-median correction.

    SimFin quarterly balance sheet files occasionally store shares in millions
    (e.g. 718 instead of 718,000,000) or thousands (e.g. 106,831 instead of
    106,831,000) for certain companies in certain quarters, and conversely
    sometimes spike by 1000x.  Correction is applied when a value deviates
    from the per-company log-median by ≥1.5 orders of magnitude; the power of
    10 needed to bring it back to the median is rounded to the nearest integer.
    Companies with fewer than 3 quarterly observations are skipped.
    """
    shares_key = 'Shares (Basic)'
    fixed = 0
    for sid, quarters in data.items():
        entries = [
            (qdict['_sort_key'], qkey, qdict[shares_key])
            for qkey, qdict in quarters.items()
            if shares_key in qdict and qdict[shares_key] > 0
        ]
        if len(entries) < 3:
            continue
        entries.sort()
        values = np.array([e[2] for e in entries])
        median_log = float(np.median(np.log10(values)))
        for sk, qkey, val in entries:
            diff = np.log10(val) - median_log
            if abs(diff) >= 1.5:
                power = -round(float(diff))
                data[sid][qkey][shares_key] = val * (10 ** power)
                fixed += 1
    if fixed:
        log.info("[shares fix] corrected %d unit-mismatched Shares (Basic) values", fixed)


def load_kind_map() -> dict:
    """Returns {constituent_name: data_kind} — 'Stock' (balance sheet) or 'Flow' (income/CF)."""
    ref = pd.read_csv(CONSTITUENTS_REF)
    return dict(zip(ref['constituent_name'], ref['data_kind']))


def load_svr_data() -> dict:
    """
    Returns {isin: (dates_np, svr_np)} from svr_daily in returns.db.
    dates_np is datetime64[ns], svr_np is float64. Both sorted ascending by date.
    Returns empty dict if table doesn't exist or returns.db is missing.
    """
    if not RETURNS_DB.exists():
        return {}
    with get_db(RETURNS_DB) as conn:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='svr_daily'"
        ).fetchall()]
        if not tables:
            return {}
        df = pd.read_sql_query(
            "SELECT isin, date, svr FROM svr_daily WHERE svr IS NOT NULL ORDER BY isin, date",
            conn,
        )
    if df.empty:
        return {}
    df["date"] = pd.to_datetime(df["date"])
    result: dict = {}
    for isin, grp in df.groupby("isin", sort=False):
        grp = grp.sort_values("date")
        result[isin] = (grp["date"].values, grp["svr"].values.astype(np.float64))
    log.info("Loaded SVR data for %s ISINs", f"{len(result):,}")
    return result


def load_price_data() -> dict:
    """
    Returns {isin: (dates_np, total_returns_np, closes_np, volumes_np)} from returns table.
    """
    if not RETURNS_DB.exists():
        log.warning("returns.db not found — price-based factors will be empty.")
        return {}
    log.info("Loading full price history from returns.db ...")
    with get_db(RETURNS_DB) as conn:
        df = pd.read_sql_query(
            "SELECT isin, date, total_return, close, volume FROM returns "
            "WHERE close IS NOT NULL AND close > 0 ORDER BY isin, date",
            conn,
        )
    df["date"] = pd.to_datetime(df["date"])
    df["total_return"] = pd.to_numeric(df["total_return"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    prices: dict = {}
    for isin, grp in df.groupby("isin", sort=False):
        grp = grp.sort_values("date")
        prices[isin] = (
            grp["date"].values,
            grp["total_return"].values,
            grp["close"].values,
            grp["volume"].values.astype(np.float64),
        )
    log.info("Loaded price data for %s ISINs", f"{len(prices):,}")
    return prices


def load_factor_name_to_id() -> dict:
    """Returns {factor_name: factor_id} from factors_reference.csv."""
    df = pd.read_csv(FACTORS_REF)
    return dict(zip(df['factor_name'], df['factor_id']))


def load_factor_sector_types() -> dict:
    """
    Returns {factor_name: sector_type} from factors_reference.csv.

    sector_type values:
      'all'       — computed for every company
      'general'   — only for general + REIT companies (not financial services)
      'reit'      — only for REIT companies
      'financial' — only for financial services companies (reserved for future)
    """
    df = pd.read_csv(FACTORS_REF)
    return dict(zip(df['factor_name'], df['sector_type']))


# Allowed factor sector_types per company sector_type
_ALLOWED_FACTOR_SECTORS: dict[str, set] = {
    'general':   {'all', 'general'},
    'financial': {'all'},              # skip revenue/WC/liquidity factors for banks
    'reit':      {'all', 'general', 'reit'},
}


# ---------------------------------------------------------------------------
# Point-in-time selection
# ---------------------------------------------------------------------------

def select_ltm_data(
    sid_data: dict,
    kind_map: dict,
    snapshot: date,
) -> tuple[dict, dict]:
    """
    Return (ltm_cdata, prior_ltm_cdata) for a company at a given snapshot date.

    ltm_cdata       — LTM built from the 4 most recent quarters with
                      publish_date <= snapshot.
    prior_ltm_cdata — LTM built from the 4 quarters immediately before those
                      (quarters 5–8), used for YoY growth factors.

    Flow items (income / cash flow): summed across the 4 quarters.
    Stock items (balance sheet):     value from the most recent quarter.

    Returns ({}, {}) when fewer than 1 quarter is available.

    Consecutive-quarter validation: warns to stdout when the 4 quarters are not
    strictly adjacent (e.g. a missing filing leaves a gap). Non-consecutive quarters
    still produce a result, but Flow totals will be understated — the warning flags
    cases worth investigating in validate_constituents.py.

    Known limitation: if a filer reports only YTD cumulative income statement facts
    in XBRL (no standalone 3-month tag), the stored quarterly values will be YTD
    and summing them here will overstate LTM. This is rare for Russell 1000 filers.
    The fix is to derive standalone quarters algebraically in update_constituents.py
    before storing.
    """
    snap_ts = pd.Timestamp(snapshot)

    def _has_flow(bucket: dict) -> bool:
        """True if the bucket contains at least one Flow item (not BS-only)."""
        return any(
            not k.startswith('_') and v is not None and kind_map.get(k, 'Flow') == 'Flow'
            for k, v in bucket.items()
        )

    # Keep only quarters published on or before snapshot that have at least one
    # Flow item.  BS-only buckets (e.g. orphaned EDGAR annual balance-sheet rows
    # whose Q4 derivation was skipped) are excluded so they cannot overwrite
    # balance-sheet values from a complete prior-year quarter in build_ltm.
    available = [
        q for q in sid_data.values()
        if q['_publish_date'] <= snap_ts and _has_flow(q)
    ]
    if not available:
        return {}, {}

    # Sort ascending by sort_key so tail(4) = most recent
    available.sort(key=lambda q: q['_sort_key'])

    # Annual-only filers (e.g. SimFin annual coverage): all consecutive periods
    # are spaced ≥10 sort-key units apart (one Q4 per fiscal year, gap = 10).
    # Normal quarterly gaps are 1 (within year) or 7 (Q4→Q1 cross-year).
    # For these, the single most-recent period already represents a full LTM;
    # summing multiple annual Q4 periods would overstate by N×.
    is_annual_only = (
        len(available) >= 2 and
        all(
            available[i + 1]['_sort_key'] - available[i]['_sort_key'] >= 10
            for i in range(len(available) - 1)
        )
    )

    if is_annual_only:
        recent_4 = available[-1:]
        prior_4  = available[-2:-1] if len(available) >= 2 else []
    else:
        recent_4 = available[-4:]
        prior_4  = available[-8:-4] if len(available) >= 8 else []

    def _check_consecutive(quarters: list, label: str) -> None:
        """Warn if the 4 quarters are not consecutive.

        sort_key = fiscal_year * 10 + period_num (Q1=1 … Q4=4).
        Consecutive steps: +1 within year, or +7 for Q4→Q1 cross-year (e.g. 20244→20251).
        A gap means a missing filing ended up in the LTM window, which silently
        understates annualised Flow totals.
        """
        if len(quarters) < 2:
            return
        keys = [q['_sort_key'] for q in quarters]
        for i in range(len(keys) - 1):
            gap = keys[i + 1] - keys[i]
            if gap not in (1, 7):
                log.warning(
                    "[%s] non-consecutive quarters in LTM (sort_keys %s) — gap %d at position %d; snapshot=%s",
                    label, keys, gap, i, snapshot,
                )
                return

    _check_consecutive(recent_4, "LTM")
    _check_consecutive(prior_4,  "prior-LTM")

    def build_ltm(quarters: list) -> dict:
        ltm: dict = {}
        for q in quarters:
            # Derive Gross Profit per-quarter when absent but Revenue and Cost of Revenue
            # are both available. Sign convention is normalised (both positive) so
            # GP = Revenue - CoR holds for SimFin and EDGAR data alike.
            if (q.get('Gross Profit') is None
                    and q.get('Revenue') is not None
                    and q.get('Cost of Revenue') is not None):
                q = {**q, 'Gross Profit': q['Revenue'] - q['Cost of Revenue']}

            # Derive Operating Income when absent.
            # Identity: Operating Income = Pretax Income − Non-Operating Income.
            # Some XBRL filers (e.g. LLY, banks) omit the OperatingIncomeLoss tag
            # but always report Pretax.  When Non-Operating Income is also absent we
            # treat it as zero — a minor approximation but avoids NaN propagation.
            if (q.get('Operating Income (Loss)') is None
                    and q.get('Pretax Income (Loss)') is not None):
                non_op = q.get('Non-Operating Income (Loss)') or 0.0
                q = {**q, 'Operating Income (Loss)': q['Pretax Income (Loss)'] - non_op}

            for name, val in q.items():
                if name.startswith('_') or val is None:
                    continue
                kind = kind_map.get(name, 'Flow')
                if kind == 'Stock':
                    ltm[name] = val          # later quarter overwrites — sorted asc, so last = most recent
                else:
                    ltm[name] = ltm.get(name, 0.0) + val
        return ltm

    return build_ltm(recent_4), build_ltm(prior_4)


# ---------------------------------------------------------------------------
# Price lookup
# ---------------------------------------------------------------------------

def get_close(
    prices: dict,
    isin: str,
    ref_date: date = None,
) -> Optional[float]:
    """Raw close price at ref_date for market cap calculations."""
    entry = prices.get(isin)
    if entry is None:
        return None
    dates, _, closes, _ = entry
    if len(dates) == 0:
        return None
    if ref_date is None:
        idx = len(dates) - 1
    else:
        ref_np = np.datetime64(ref_date, "D").astype("datetime64[ns]")
        idx = int(np.searchsorted(dates, ref_np, side="right")) - 1
    if idx < 0:
        return None
    val = closes[idx]
    return float(val) if np.isfinite(val) else None


# ---------------------------------------------------------------------------
# Factor computation
# ---------------------------------------------------------------------------

def compute_quality_factors(cdata: dict) -> dict:
    """19 quality factors from income, balance, and cash flow data."""
    f = {}
    revenue      = cdata.get('Revenue')
    cost_rev     = cdata.get('Cost of Revenue')
    gross_p      = cdata.get('Gross Profit') or (
        (revenue - cost_rev) if (revenue is not None and cost_rev is not None) else None
    )
    op_income    = cdata.get('Operating Income (Loss)')
    net_income   = cdata.get('Net Income')
    equity       = cdata.get('Total Equity')
    assets       = cdata.get('Total Assets')
    op_cf        = cdata.get('Net Cash from Operating Activities')
    capex        = cdata.get('Change in Fixed Assets & Intangibles')
    cur_assets   = cdata.get('Total Current Assets')
    cur_liab     = cdata.get('Total Current Liabilities')
    short_debt   = cdata.get('Short Term Debt') or 0  # many firms report no ST debt; treat as zero
    long_debt    = cdata.get('Long Term Debt')
    total_liab   = cdata.get('Total Liabilities')
    wc_change    = cdata.get('Change in Working Capital')
    interest_exp = cdata.get('Interest Expense, Net')
    invest_inc   = cdata.get('Investment Income, Interest')
    pretax       = cdata.get('Pretax Income (Loss)')
    tax_exp      = cdata.get('Income Tax (Expense) Benefit, Net')
    cash         = cdata.get('Cash, Cash Equivalents & Short Term Investments') or 0

    if revenue is not None and gross_p is not None and revenue != 0:
        f['Gross Margin'] = gross_p / revenue
    if revenue is not None and op_income is not None and revenue != 0:
        f['Operating Margin'] = op_income / revenue
    if revenue is not None and net_income is not None and revenue != 0:
        nm = net_income / revenue
        if abs(nm) <= 2:  # >200% signals bad SimFin revenue data (e.g. REITs with near-zero reported revenue)
            f['Net Margin'] = nm
    if net_income is not None and equity is not None and equity > 0:
        f['ROE'] = net_income / equity
    if net_income is not None and assets is not None and assets != 0:
        f['ROA'] = net_income / assets
    if op_cf is not None and net_income is not None and net_income != 0:
        f['Operating Cash Flow Ratio'] = op_cf / net_income
    if op_cf is not None and capex is not None and revenue is not None and revenue != 0:
        # capex (Change in Fixed Assets & Intangibles) is stored as negative (cash outflow);
        # op_cf + capex correctly subtracts it (op_cf - |capex|).
        f['FCF Margin'] = (op_cf + capex) / revenue
    if op_cf is not None and revenue is not None and revenue != 0:
        f['Cash Conversion Quality'] = op_cf / revenue
    if cur_assets is not None and cur_liab is not None and cur_liab != 0:
        f['Current Ratio'] = cur_assets / cur_liab
    if long_debt is not None and equity is not None and equity > 0:
        f['Leverage'] = (short_debt + long_debt) / equity
    if total_liab is not None and assets is not None and assets != 0:
        f['Debt-to-Assets'] = total_liab / assets
    if equity is not None and assets is not None and assets != 0:
        f['Equity Ratio'] = equity / assets
    if revenue is not None and assets is not None and assets != 0:
        f['Asset Turnover'] = revenue / assets
    if wc_change is not None and revenue is not None and revenue != 0:
        f['Working Capital Efficiency'] = wc_change / revenue
    if capex is not None and revenue is not None and revenue > 0 and abs(capex) > 0:
        f['Capex Intensity'] = float(np.log(abs(capex) / revenue))  # log-transform; lower = capital-light

    # Interest Coverage: edgartools stores InterestExpense as negative (expense sign, weight=-1).
    # SimFin stores it as a positive net expense (already nets against interest income).
    # We distinguish by sign: negative → EDGAR gross expense; positive → SimFin net expense.
    # For EDGAR we compute net = abs(expense) − investment_income; skip if net ≤ 0 (net earner).
    if interest_exp is not None and op_income is not None:
        if interest_exp < 0:
            # EDGAR path: gross expense stored negative; invest_inc is positive (or 0 if absent)
            net_interest = abs(interest_exp) - (invest_inc or 0.0)
        else:
            # SimFin path: already net, positive = expense
            net_interest = interest_exp
        if net_interest > 0:
            f['Interest Coverage'] = op_income / net_interest

    # ROIC = NOPAT / Invested Capital; Invested Capital = equity + debt - cash
    if (op_income is not None and equity is not None and long_debt is not None):
        if pretax is not None and pretax > 0 and tax_exp is not None:
            # tax_exp is negative when it's an expense; clamp effective rate to [0, 0.5]
            t_rate = max(0.0, min(0.5, -tax_exp / pretax))
        else:
            t_rate = 0.21  # US statutory fallback for loss-makers or missing tax data
        nopat = op_income * (1.0 - t_rate)
        invested_capital = equity + short_debt + long_debt - cash
        if invested_capital > 0:
            f['ROIC'] = nopat / invested_capital

    # Accruals Ratio (Sloan 1996): high accruals = earnings not backed by cash = lower quality
    if net_income is not None and op_cf is not None and assets is not None and assets > 0:
        f['Accruals Ratio'] = (net_income - op_cf) / assets

    # Gross Profit to Assets (Novy-Marx 2013): gross profitability
    if gross_p is not None and assets is not None and assets > 0:
        f['Gross Profit to Assets'] = gross_p / assets

    return f


def compute_value_factors(
    cdata: dict, isin: str, prices: dict, ref_date: date = None
) -> dict:
    """7 value factors expressed as ratios to market cap."""
    f = {}
    shares = cdata.get('Shares (Basic)')
    price  = get_close(prices, isin, ref_date=ref_date)
    if not (price and shares and price > 0 and shares > 0):
        return f
    market_cap = price * shares
    net_income = cdata.get('Net Income')
    equity     = cdata.get('Total Equity')
    revenue    = cdata.get('Revenue')
    op_cf      = cdata.get('Net Cash from Operating Activities')
    op_income  = cdata.get('Operating Income (Loss)')
    short_debt = cdata.get('Short Term Debt')
    long_debt  = cdata.get('Long Term Debt')
    cash       = cdata.get('Cash, Cash Equivalents & Short Term Investments')
    # D&A may come from CF statement (+) or IS (-) depending on which was stored last;
    # abs() makes EV/EBITDA robust to either sign convention.
    da         = cdata.get('Depreciation & Amortization')
    dividends  = cdata.get('Dividends Paid')

    if net_income is not None:
        f['Earnings Yield'] = net_income / market_cap
    if equity is not None:
        f['Book-to-Price'] = equity / market_cap
    if revenue is not None and revenue != 0:
        f['Sales-to-Price'] = revenue / market_cap
    if op_cf is not None:
        f['Cash Yield'] = op_cf / market_cap
    if cash is not None and op_income is not None and op_income > 0:
        ev = market_cap + (short_debt or 0) + (long_debt or 0) - cash
        if ev > 0:
            f['EV-to-EBIT'] = ev / op_income

    # EV/EBITDA: EBITDA = op_income + D&A; require both positive and EV > 0
    if (cash is not None and op_income is not None and da is not None):
        ebitda = op_income + abs(da)
        ev     = market_cap + (short_debt or 0) + (long_debt or 0) - cash
        if ev > 0 and ebitda > 0:
            f['EV/EBITDA'] = ev / ebitda

    # Dividend Yield: only for companies that actually paid cash dividends (Dividends Paid < 0)
    if dividends is not None and dividends < 0:
        f['Dividend Yield'] = abs(dividends) / market_cap

    return f


def compute_growth_factors(cdata: dict, cdata_prior: dict) -> dict:
    """7 YoY growth factors. Earnings/CF growth skipped when prior year is negative."""
    f = {}

    def yoy(name: str, require_positive_base: bool = False):
        cur = cdata.get(name)
        pri = cdata_prior.get(name)
        if cur is None or pri is None or pri == 0:
            return None
        if require_positive_base and pri < 0:
            return None
        return (cur - pri) / abs(pri)

    for name, kw in [
        ('Revenue Growth',            {'name': 'Revenue'}),
        ('Earnings Growth',           {'name': 'Net Income', 'require_positive_base': True}),
        ('Cash Flow Growth',          {'name': 'Net Cash from Operating Activities', 'require_positive_base': True}),
        ('Asset Growth',              {'name': 'Total Assets'}),
        ('Equity Growth',             {'name': 'Total Equity'}),
        ('Operating Income Growth',   {'name': 'Operating Income (Loss)'}),
    ]:
        v = yoy(**kw)
        if v is not None:
            f[name] = v

    # EBITDA Growth: derive EBITDA = op_income + abs(D&A) for current and prior period
    def _ebitda(cd: dict) -> Optional[float]:
        op = cd.get('Operating Income (Loss)')
        da = cd.get('Depreciation & Amortization')
        if op is not None and da is not None:
            return op + abs(da)
        return None

    cur_ebitda = _ebitda(cdata)
    pri_ebitda = _ebitda(cdata_prior)
    if cur_ebitda is not None and pri_ebitda is not None and pri_ebitda != 0:
        f['EBITDA Growth'] = (cur_ebitda - pri_ebitda) / abs(pri_ebitda)

    return f


def compute_momentum_factors(
    isin: str, prices: dict, ref_date: date = None
) -> dict:
    """6M and 12M cumulative total return — split-invariant."""
    f = {}
    entry = prices.get(isin)
    if entry is None:
        return f
    dates, total_rets, _, _ = entry

    if ref_date is None:
        ref_idx = len(dates) - 1
        ref_dt  = pd.Timestamp(dates[-1]).date()
    else:
        ref_np  = np.datetime64(ref_date, "D").astype("datetime64[ns]")
        ref_idx = int(np.searchsorted(dates, ref_np, side="right")) - 1
        ref_dt  = ref_date
    if ref_idx < 1:
        return f

    for months, name in [(6, "6M Momentum"), (12, "12M Momentum")]:
        target = ref_dt - relativedelta(months=months)
        tgt_np = np.datetime64(target, "D").astype("datetime64[ns]")
        start_idx = int(np.searchsorted(dates, tgt_np, side="right"))
        if start_idx > ref_idx:
            continue
        window = total_rets[start_idx : ref_idx + 1]
        valid  = window[np.isfinite(window)]
        if len(valid) < 20:
            continue
        f[name] = float(np.prod(1.0 + valid) - 1.0)
    return f


def compute_size_factor(
    cdata: dict, isin: str, prices: dict, ref_date: date = None
) -> dict:
    """Log market cap — natural log of price × basic shares."""
    shares = cdata.get('Shares (Basic)')
    price  = get_close(prices, isin, ref_date=ref_date)
    if price and shares and price > 0 and shares > 0:
        return {'Log Market Cap': np.log(price * shares)}
    return {}


def compute_low_vol_factors(
    isin: str, prices: dict, ref_date: date = None, lookback: int = 252
) -> dict:
    """Realized volatility (annualized) and 52-week high ratio from return history."""
    f = {}
    entry = prices.get(isin)
    if entry is None or len(entry[0]) < 2:
        return f
    dates, total_rets, closes, _ = entry

    if ref_date is None:
        ref_idx = len(dates) - 1
    else:
        ref_np  = np.datetime64(ref_date, "D").astype("datetime64[ns]")
        ref_idx = int(np.searchsorted(dates, ref_np, side="right")) - 1
    if ref_idx < 1:
        return f

    start_idx  = max(0, ref_idx - lookback)
    win_rets   = total_rets[start_idx : ref_idx + 1]

    if len(win_rets) < 64:
        return f

    # Realized volatility from log(1 + total_return), stored as log(vol) for normality
    log_rets = np.log(1.0 + win_rets)
    valid    = log_rets[np.isfinite(log_rets)]
    if len(valid) >= 20:
        vol = float(np.std(valid, ddof=1) * np.sqrt(252))
        if vol > 0:
            f["Realized Volatility"] = float(np.log(vol))  # log-transform compresses extreme outliers

    # 52-week high ratio via cumulative returns (split-invariant)
    # ratio = current_cum_level / max_cum_level over the window
    finite_rets = win_rets[np.isfinite(win_rets)]
    if len(finite_rets) >= 2:
        cum_levels = np.cumprod(1.0 + finite_rets)
        if cum_levels.max() > 0:
            f["52-Week High Ratio"] = float(cum_levels[-1] / cum_levels.max())

    return f


def compute_liquidity_factors(
    isin: str, prices: dict, ref_date: date = None, lookback: int = 252
) -> dict:
    """Amihud illiquidity ratio (×1e6) from daily return and volume history."""
    f = {}
    entry = prices.get(isin)
    if entry is None or len(entry) < 4:
        return f
    dates, total_rets, closes, volumes = entry

    if ref_date is None:
        ref_idx = len(dates) - 1
    else:
        ref_np  = np.datetime64(ref_date, "D").astype("datetime64[ns]")
        ref_idx = int(np.searchsorted(dates, ref_np, side="right")) - 1
    if ref_idx < 1:
        return f

    start_idx   = max(0, ref_idx - lookback)
    win_rets    = total_rets[start_idx : ref_idx + 1]
    win_closes  = closes[start_idx : ref_idx + 1]
    win_volumes = volumes[start_idx : ref_idx + 1]

    if len(win_rets) < 64:
        return f

    abs_rets    = np.abs(win_rets)
    dollar_vols = win_closes * win_volumes
    mask = np.isfinite(abs_rets) & (dollar_vols > 0)
    if mask.sum() < 20:
        return f

    amihud = float(np.mean(abs_rets[mask] / dollar_vols[mask]))
    if np.isfinite(amihud) and amihud > 0:
        f["Amihud Illiquidity"] = float(np.log(amihud * 1e6))  # log-transform for normality

    return f


def compute_svr_factors(
    isin: str, svr_data: dict, ref_date: date = None, window_20: int = 20, window_90: int = 90
) -> dict:
    """
    SVR 20-Day Avg  — trailing 20-day mean SVR (requires ≥10 observations).
    SVR 90-Day Rank — percentile rank of that 20-day avg within the trailing 90 days
                      of daily SVR values (requires ≥45 observations).
    Both have direction=-1: lower SVR = less short pressure = better signal.
    """
    entry = svr_data.get(isin)
    if entry is None:
        return {}
    dates, svrs = entry

    if ref_date is None:
        ref_idx = len(dates) - 1
    else:
        ref_np  = np.datetime64(ref_date, "D").astype("datetime64[ns]")
        ref_idx = int(np.searchsorted(dates, ref_np, side="right")) - 1
    if ref_idx < 1:
        return {}

    # 20-day trailing window
    start_20  = max(0, ref_idx - window_20 + 1)
    win_20    = svrs[start_20 : ref_idx + 1]
    valid_20  = win_20[np.isfinite(win_20)]
    if len(valid_20) < 10:
        return {}
    svr_20 = float(np.mean(valid_20))

    f = {"SVR 20-Day Avg": svr_20}

    # 90-day trailing window for rank
    start_90 = max(0, ref_idx - window_90 + 1)
    win_90   = svrs[start_90 : ref_idx + 1]
    valid_90 = win_90[np.isfinite(win_90)]
    if len(valid_90) >= 45:
        # percentile of current 20-day SVR within the 90-day history
        rank = float(np.mean(valid_90 <= svr_20))  # fraction of days with SVR ≤ today's
        f["SVR 90-Day Rank"] = rank

    return f


def compute_reit_factors(
    cdata: dict, cdata_prior: dict, isin: str, prices: dict, ref_date: date = None
) -> dict:
    """
    REIT-specific factors based on Funds From Operations (FFO).
    FFO = Net Income + Depreciation & Amortization (D&A from CF statement).
    Only called for companies with sector_type == 'reit'.
    """
    f = {}
    net_income = cdata.get('Net Income')
    da         = cdata.get('Depreciation & Amortization')
    revenue    = cdata.get('Revenue')

    if net_income is None or da is None:
        return f

    ffo = net_income + abs(da)   # abs() handles any sign convention

    shares = cdata.get('Shares (Basic)')
    price  = get_close(prices, isin, ref_date=ref_date)
    if price and shares and price > 0 and shares > 0:
        market_cap = price * shares
        f['FFO Yield'] = ffo / market_cap

    if revenue is not None and revenue > 0:
        ratio = ffo / revenue
        if abs(ratio) <= 10:  # cap at 1000%; higher means SimFin revenue data is unreliable
            f['FFO Margin'] = ratio

    # FFO Growth — skip if prior FFO was zero or negative
    net_income_prior = cdata_prior.get('Net Income')
    da_prior         = cdata_prior.get('Depreciation & Amortization')
    if net_income_prior is not None and da_prior is not None:
        ffo_prior = net_income_prior + abs(da_prior)
        if ffo_prior > 0:
            f['FFO Growth'] = (ffo - ffo_prior) / ffo_prior

    return f


# ---------------------------------------------------------------------------
# Database setup and writing
# ---------------------------------------------------------------------------

def setup_factors_db(conn: sqlite3.Connection, clean: bool = False) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    existing = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='factors'"
    ).fetchone()
    if existing:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(factors)").fetchall()]
        if 'fiscal_year' in cols or clean:
            reason = "old schema (had fiscal_year column)" if 'fiscal_year' in cols else "--clean flag"
            log.info("Dropping factors table (%s) — will rebuild from scratch", reason)
            conn.execute("DROP TABLE factors")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS factors (
            data_date        TEXT    NOT NULL,
            factor_id        TEXT    NOT NULL,
            security_id      TEXT    NOT NULL,
            factor_value     REAL    NOT NULL,
            factor_value_z   REAL,
            update_date      TEXT    NOT NULL,
            computation_date TEXT    NOT NULL,
            PRIMARY KEY (data_date, factor_id, security_id)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_f_date_fid ON factors (data_date, factor_id)"
    )
    conn.execute("""
        CREATE TABLE IF NOT EXISTS snapshot_dates (
            data_date  TEXT PRIMARY KEY,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Single-date snapshot
# ---------------------------------------------------------------------------

def run_for_date(
    snapshot: date,
    universe: dict,
    ticker_map: dict,
    constituent_data: dict,
    kind_map: dict,
    prices: dict,
    svr_data: dict,
    factor_name_to_id: dict,
    factor_sector_types: dict,
    conn: sqlite3.Connection,
) -> int:
    """Compute and write all factor rows for one snapshot date. Returns row count."""
    date_str      = snapshot.strftime('%Y-%m-%d')
    today         = datetime.now().strftime('%Y-%m-%d')
    rows          = []
    unknown_names: set = set()

    snapshot_isins = load_snapshot_isins(snapshot)
    if snapshot_isins is not None:
        active_universe = {isin: meta for isin, meta in universe.items()
                           if isin in snapshot_isins}
    else:
        active_universe = universe

    for isin, meta in active_universe.items():
        ticker = meta.get('ticker') or ticker_map.get(isin)
        if not ticker:
            continue
        sid_data = constituent_data.get(isin)
        if not sid_data:
            continue

        cdata, cdata_prior = select_ltm_data(sid_data, kind_map, snapshot)
        if not cdata:
            continue

        sector_type = meta.get('sector_type', 'general')

        factors: dict = {}
        factors.update(compute_quality_factors(cdata))
        factors.update(compute_value_factors(cdata, isin, prices, ref_date=snapshot))
        factors.update(compute_growth_factors(cdata, cdata_prior))
        factors.update(compute_momentum_factors(isin, prices, ref_date=snapshot))
        factors.update(compute_size_factor(cdata, isin, prices, ref_date=snapshot))
        factors.update(compute_low_vol_factors(isin, prices, ref_date=snapshot))
        factors.update(compute_liquidity_factors(isin, prices, ref_date=snapshot))
        factors.update(compute_svr_factors(isin, svr_data, ref_date=snapshot))
        if sector_type == 'reit':
            factors.update(compute_reit_factors(cdata, cdata_prior, isin, prices, ref_date=snapshot))

        allowed = _ALLOWED_FACTOR_SECTORS.get(sector_type, {'all', 'general'})

        for name, value in factors.items():
            if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                continue
            factor_id = factor_name_to_id.get(name)
            if factor_id is None:
                unknown_names.add(name)
                continue
            # Skip factors not applicable to this company's sector type
            if factor_sector_types.get(name, 'all') not in allowed:
                continue
            rows.append((date_str, factor_id, isin, float(value), today, today))

    if unknown_names:
        log.warning("factor names not in factors_reference.csv (skipped): %s", sorted(unknown_names))

    conn.executemany(
        "INSERT OR REPLACE INTO factors "
        "(data_date, factor_id, security_id, factor_value, update_date, computation_date) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )

    # Cross-sectional z-scores — computed before committing so insert + z-score
    # update land in one atomic transaction (same connection sees uncommitted rows).
    df = pd.read_sql_query(
        "SELECT rowid, factor_id, factor_value FROM factors WHERE data_date = ?",
        conn, params=(date_str,)
    )
    df['factor_value_z'] = (
        df.groupby('factor_id')['factor_value']
        .transform(winsorized_zscore)
    )
    conn.executemany(
        "UPDATE factors SET factor_value_z = ? WHERE rowid = ?",
        df[['factor_value_z', 'rowid']].itertuples(index=False, name=None),
    )
    conn.commit()

    n_companies = len({r[2] for r in rows})  # r[2] is isin
    log.info("%s: %s companies, %s factor rows", date_str, f"{n_companies:,}", f"{len(rows):,}")
    return len(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute point-in-time factor snapshots and write to factors.db"
    )
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument('--date',              metavar='YYYY-MM-DD', action='append', dest='dates',
                     help='Snapshot date (repeatable: --date D1 --date D2)')
    grp.add_argument('--backfill',          action='store_true',
                     help='Run all annual April-1 backfill dates')
    grp.add_argument('--quarterly-backfill', action='store_true',
                     help='Run all quarterly snapshot dates (May/Aug/Nov/Feb)')
    parser.add_argument('--clean', action='store_true',
                        help='Drop and rebuild the factors table before running')
    args = parser.parse_args()

    if args.backfill:
        dates_to_run = [datetime.strptime(d, '%Y-%m-%d').date() for d in BACKFILL_DATES]
    elif args.quarterly_backfill:
        all_dates = sorted(set(BACKFILL_DATES + QUARTERLY_BACKFILL_DATES))
        dates_to_run = [datetime.strptime(d, '%Y-%m-%d').date() for d in all_dates]
    elif args.dates:
        dates_to_run = sorted({datetime.strptime(d, '%Y-%m-%d').date() for d in args.dates})
    else:
        dates_to_run = [date.today()]

    log.info("Loading reference data ...")
    universe            = load_universe()
    ticker_map          = load_ticker_map()
    factor_name_to_id   = load_factor_name_to_id()
    factor_sector_types = load_factor_sector_types()
    kind_map            = load_kind_map()

    log.info("Loading constituent data from constituents.db ...")
    constituent_data = load_constituent_data()

    prices   = load_price_data()
    svr_data = load_svr_data()

    with get_db(FACTORS_DB) as conn:
        setup_factors_db(conn, clean=args.clean)

        total_rows = 0
        log.info("Processing %d date(s): %s", len(dates_to_run), [str(d) for d in dates_to_run])
        for snapshot in dates_to_run:
            total_rows += run_for_date(
                snapshot, universe, ticker_map,
                constituent_data, kind_map, prices, svr_data,
                factor_name_to_id, factor_sector_types, conn,
            )
            conn.execute(
                "INSERT OR REPLACE INTO snapshot_dates (data_date, created_at) "
                "VALUES (?, datetime('now'))",
                (str(snapshot),),
            )
            conn.commit()

    log.info("Done — %s total factor rows across %d date(s)", f"{total_rows:,}", len(dates_to_run))


if __name__ == "__main__":
    main()
