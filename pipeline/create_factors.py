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
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from typing import Optional


from config import (
    UNIVERSE_DB, CONSTITUENTS_DB, RETURNS_DB, FACTORS_DB,
    FACTORS_REF, CONSTITUENTS_REF,
)
from utils import (
    classify_sector, get_db, get_logger, winsorized_zscore,
    get_snapshot_schedule, mark_snapshot_computed,
    ALLOWED_FACTOR_SECTORS as _ALLOWED_FACTOR_SECTORS,
)

log = get_logger("create_factors")

_PERIOD_ORDER = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
FACTOR_UNIVERSE_INDEXES = ("russell_1000", "sp500")


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
    Returns the union of ISINs in the factor coverage indexes at the latest
    universe_snapshots date on or before `snapshot`. Returns None if the table
    is empty (fallback: use all companies in the companies table).
    """
    date_str = snapshot.strftime('%Y-%m-%d')
    out: set[str] = set()
    sources: dict[str, str] = {}
    with get_db(UNIVERSE_DB) as conn:
        for index_name in FACTOR_UNIVERSE_INDEXES:
            matched = conn.execute(
                """
                SELECT snapshot_date
                FROM universe_snapshots
                WHERE index_name = ? AND snapshot_date <= ?
                ORDER BY snapshot_date DESC
                LIMIT 1
                """,
                (index_name, date_str),
            ).fetchone()
            if not matched:
                continue
            matched_date = matched[0]
            sources[index_name] = matched_date
            rows = conn.execute(
                """
                SELECT isin
                FROM universe_snapshots
                WHERE snapshot_date = ? AND index_name = ?
                """,
                (matched_date, index_name),
            ).fetchall()
            out.update(r[0] for r in rows)

        if not out:
            return None

    if any(src != date_str for src in sources.values()):
        src_txt = ", ".join(f"{idx}={src}" for idx, src in sorted(sources.items()))
        log.info("No exact factor-universe snapshot for %s — using %s (%s companies)",
                 date_str, src_txt, f"{len(out):,}")
    return out


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


def _dedup_constituent_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep one row per mapped PIT constituent key.

    Native ISIN rows are preferred over legacy rows mapped from SimFin IDs; within
    the same source priority, the EARLIEST publish_date wins — the original filing
    date when a figure first became public. A 10-Q/10-K reprints prior-period
    figures as comparative columns (e.g. Q2-2020 reappears in the Q2-2021 10-Q,
    stamped a year later); keeping the latest date would wrongly defer that figure's
    PIT availability by a year. We do not track restatements, so original-as-reported
    is exactly the point-in-time value we want.
    """
    return (df.sort_values(['source_priority', 'publish_date'], ascending=[True, False])
              .groupby(['security_id', 'constituent_id', 'fiscal_year', 'fiscal_period'],
                       as_index=False)
              .last())


def _q4_inputs_available_before_annual(
    q1: dict,
    q2: dict,
    q3: dict,
    name: str,
    annual_publish_date: pd.Timestamp,
) -> bool:
    """Return True when this constituent's Q1-Q3 values predate the annual."""
    q1_pub = q1.get('_publish_by_name', {}).get(name, pd.Timestamp.max)
    q2_pub = q2.get('_publish_by_name', {}).get(name, pd.Timestamp.max)
    q3_pub = q3.get('_publish_by_name', {}).get(name, pd.Timestamp.max)
    return q1_pub <= annual_publish_date and q2_pub <= annual_publish_date and q3_pub <= annual_publish_date


def _load_security_data_starts(conn_u: sqlite3.Connection) -> dict[str, pd.Timestamp]:
    rows = conn_u.execute(
        """
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='security_data_start'
        """
    ).fetchone()
    if not rows:
        return {}
    starts = conn_u.execute("SELECT isin, min_report_date FROM security_data_start").fetchall()
    out: dict[str, pd.Timestamp] = {}
    for isin, min_report_date in starts:
        ts = pd.to_datetime(min_report_date, errors="coerce")
        if pd.notna(ts):
            out[str(isin)] = ts
    return out


def _apply_security_data_starts(
    df: pd.DataFrame,
    data_starts: dict[str, pd.Timestamp],
) -> pd.DataFrame:
    """Drop rows before a security's current-issuer data start date."""
    if not data_starts or df.empty:
        return df
    report_dates = pd.to_datetime(df['report_date'], errors='coerce')
    min_dates = df['security_id'].map(data_starts)
    keep = min_dates.isna() | (report_dates >= min_dates)
    return df[keep].copy()


def load_constituent_data() -> dict:
    """
    Returns {security_id: {(fiscal_year, fiscal_period): quarter_dict}} where
    quarter_dict = {constituent_name: value, '_publish_date': pd.Timestamp,
                    '_sort_key': int, '_publish_by_name': dict}.

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
        data_starts = _load_security_data_starts(conn_u)
    simfin_to_isin = dict(zip(id_map['simfin_id'], id_map['isin']))

    def _map_sid(sid: str) -> Optional[str]:
        return simfin_to_isin.get(sid) or (sid if sid in all_isins else None)

    def _load_and_clean(query: str, conn) -> pd.DataFrame:
        df = pd.read_sql_query(query, conn)
        df['original_security_id'] = df['security_id'].astype(str)
        # Native ISIN rows are EDGAR/current-source rows. Legacy SimFin rows are
        # mapped from simfin_id to ISIN below and can use different fiscal-year
        # labels for Jan/Feb FYE companies. When both sources collide on the same
        # mapped key, prefer native ISIN data; within a source, latest publish wins.
        df['source_priority'] = df['original_security_id'].isin(all_isins).astype(int)
        df['constituent_name'] = df['constituent_id'].map(id_to_name)
        df['data_kind']        = df['constituent_id'].map(id_to_kind)
        df = df.dropna(subset=['constituent_name'])
        df['security_id'] = df['original_security_id'].apply(_map_sid)
        df = df.dropna(subset=['security_id'])
        df['fiscal_year']  = df['fiscal_year'].astype(int)
        df['publish_date'] = pd.to_datetime(df['publish_date'])
        df = _apply_security_data_starts(df, data_starts)
        return df

    with get_db(CONSTITUENTS_DB) as conn:
        # Primary: quarterly rows (Q1–Q4) — covers SimFin + EDGAR 10-Q
        df_q = _load_and_clean(
            "SELECT security_id, constituent_id, constituent_value, "
            "       fiscal_year, fiscal_period, publish_date, report_date "
            "FROM constituents "
            "WHERE fiscal_period IN ('Q1','Q2','Q3','Q4') "
            "  AND publish_date IS NOT NULL",
            conn,
        )
        # Annual: income + cashflow stored as FY from EDGAR 10-K
        # Balance sheet rows from 10-K are already stored as Q4 — not needed here.
        df_fy = _load_and_clean(
            "SELECT security_id, constituent_id, constituent_value, "
            "       fiscal_year, fiscal_period, publish_date, report_date "
            "FROM constituents "
            "WHERE fiscal_period = 'FY' "
            "  AND statement_type IN ('Income Statement', 'Cash Flow Statement') "
            "  AND publish_date IS NOT NULL",
            conn,
        )

    df_q  = _dedup_constituent_rows(df_q)
    df_fy = _dedup_constituent_rows(df_fy)

    df_q['sort_key'] = df_q['fiscal_year'] * 10 + df_q['fiscal_period'].map(_PERIOD_ORDER)

    data: dict = {}
    for row in df_q.itertuples(index=False):
        sid = row.security_id
        key = (row.fiscal_year, row.fiscal_period)
        bucket = data.setdefault(sid, {}).setdefault(key, {
            '_publish_date': row.publish_date,
            '_sort_key':     row.sort_key,
            '_publish_by_name': {},
        })
        if row.publish_date > bucket['_publish_date']:
            bucket['_publish_date'] = row.publish_date
        bucket[row.constituent_name] = row.constituent_value
        bucket['_publish_by_name'][row.constituent_name] = row.publish_date

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
        v1 = q1.get(name)
        v2 = q2.get(name)
        v3 = q3.get(name)
        if v1 is None or v2 is None or v3 is None:
            continue  # constituent missing in at least one prior quarter

        # Temporal guard: this constituent's Q1/Q2/Q3 values must be published
        # before the annual filing.
        # If any were published after the annual, they belong to the next fiscal year
        # (common for January/February fiscal-year-end companies like NVDA, WMT, HD
        # where EDGAR labels the next FY's quarters under the same fiscal_year as the
        # annual — e.g. NVDA FY2025 annual pub=Feb-2025 vs FY2026 Q1 pub=May-2025).
        # Use per-constituent publish dates rather than the bucket-level max: mixed
        # SimFin/EDGAR buckets can contain unrelated later-published items, and those
        # must not block Q4 derivation for values that were actually available.
        fy_pub = row.publish_date
        if not _q4_inputs_available_before_annual(q1, q2, q3, name, fy_pub):
            continue

        q4_val = row.constituent_value - v1 - v2 - v3
        q4_sort = fy * 10 + _PERIOD_ORDER['Q4']
        bucket = sid_data.setdefault(q4_key, {
            '_publish_date': row.publish_date,
            '_sort_key':     q4_sort,
            '_publish_by_name': {},
        })
        if row.publish_date > bucket['_publish_date']:
            bucket['_publish_date'] = row.publish_date
        bucket[name] = q4_val
        bucket['_publish_by_name'][name] = row.publish_date
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
        bucket.setdefault('_publish_by_name', {})[name] = row.publish_date
        if row.publish_date > bucket['_publish_date']:
            bucket['_publish_date'] = row.publish_date
        annual_direct += 1

    if annual_direct:
        log.info("[annual-only] assigned %s FY Flow values to Q4 for annual-only filers", f"{annual_direct:,}")

    # Price-anchor the share-unit check: split history keeps real splits (e.g.
    # CMG's 50:1) from looking like unit errors; latest closes resolve which
    # share scale yields a plausible market cap (handles reverse mergers like QXO).
    _fix_shares_units(data, load_splits(), _load_latest_closes())
    log.info("Loaded quarterly constituent data for %s companies", f"{len(data):,}")
    return data


# Plausibility bounds for the price-anchored share-unit check. A unit error is
# always a clean factor of 1e3/1e6, so it pushes BOTH implied market cap and the
# share count far outside these bounds. We test both because a ×1000 error of a
# $1–5B name lands in the same market-cap zone as a real mega-cap (NVDA ~$5T) —
# the share-count ceiling disambiguates (real names never carry tens of billions
# of shares the way an inflated count does).
_MCAP_BAND_LO = 1e7       # $10M   — below any name in this universe
_MCAP_BAND_HI = 1e13      # $10T   — above the largest mega-cap, with headroom
_MAX_PLAUSIBLE_SHARES = 5e10   # 50B shares — NVDA ~24B is the realistic ceiling
_SHARE_UNIT_POWERS = (-6, -3, 3, 6)


def _load_latest_closes() -> dict:
    """{isin: latest close} from returns.db — used to price-anchor share-unit checks."""
    if not RETURNS_DB.exists():
        return {}
    with get_db(RETURNS_DB) as conn:
        rows = conn.execute(
            "SELECT isin, close FROM ("
            "  SELECT isin, close, ROW_NUMBER() OVER "
            "    (PARTITION BY isin ORDER BY date DESC) AS rn "
            "  FROM returns WHERE close > 0"
            ") WHERE rn = 1"
        ).fetchall()
    return {isin: close for isin, close in rows}


def _fix_shares_units(
    data: dict, splits: dict | None = None, latest_closes: dict | None = None
) -> None:
    """
    Correct order-of-magnitude unit errors in 'Shares (Basic)' in place, anchored
    on price rather than a per-company median.

    Source balance-sheet files occasionally store shares in thousands or millions
    (e.g. 957,800 stored as 957,800,000,000) for some companies in some quarters.
    A unit error is always a clean factor of 1e3 or 1e6, so it moves the implied
    market cap (split-normalised shares × latest price) and the share count far
    outside their plausible bounds. We keep the stored value when its market cap is
    in [$10M, $10T] AND its share count is ≤ 50B; otherwise we rescale by the single
    clean factor (10^±3 / 10^±6) that lands both back in-bounds, choosing the one
    whose market cap is closest to the band centre. Values with no clean factor that
    fits, and companies with no price, are left untouched — the rule never guesses a
    scale. The share-count ceiling is what separates a real mega-cap (NVDA ~$5T,
    ~24B shares) from a ×1000-inflated mid-cap that lands at a similar market cap.

    This replaces an earlier log-median heuristic that failed on two cases:
      • identity changes / reverse mergers (e.g. QXO: ~5M shell shares → ~700M
        post-merger) where the median spans two real regimes and flags the real
        current count as an error; and
      • persistently corrupt series where the median picks the wrong scale.

    Split-awareness: shares are normalised to the current split basis (× product
    of split ratios dated after each quarter's publish date) before the price
    check, so a real split (e.g. CMG 50:1) is never seen as a unit error. The
    correction is applied to the raw stored value (the unit error lives there).
    """
    shares_key = 'Shares (Basic)'
    splits = splits or {}
    latest_closes = latest_closes or {}
    center_log = float(np.log10(np.sqrt(_MCAP_BAND_LO * _MCAP_BAND_HI)))
    fixed = 0
    for sid, quarters in data.items():
        price = latest_closes.get(sid)
        if not (price and price > 0):
            continue
        sid_splits = splits.get(sid, ())
        for qdict in quarters.values():
            raw = qdict.get(shares_key)
            if not (raw and raw > 0):
                continue
            asof = qdict.get('_publish_date')
            norm_factor = 1.0
            for split_date, ratio in sid_splits:
                if asof is not None and split_date > asof:
                    norm_factor *= ratio
            norm = raw * norm_factor
            if (_MCAP_BAND_LO <= norm * price <= _MCAP_BAND_HI
                    and norm <= _MAX_PLAUSIBLE_SHARES):
                continue  # already plausible — keep stored value
            inbounds = [
                (p, norm * (10 ** p) * price)
                for p in _SHARE_UNIT_POWERS
                if _MCAP_BAND_LO <= norm * (10 ** p) * price <= _MCAP_BAND_HI
                and norm * (10 ** p) <= _MAX_PLAUSIBLE_SHARES
            ]
            if not inbounds:
                continue  # no clean factor fits — don't guess
            power = min(inbounds, key=lambda pm: abs(np.log10(pm[1]) - center_log))[0]
            qdict[shares_key] = raw * (10 ** power)
            fixed += 1
    if fixed:
        log.info("[shares fix] price-anchored correction of %d Shares (Basic) values", fixed)


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


def load_short_interest_data() -> dict:
    """
    Returns {isin: (settlement_dates_np, days_to_cover_np)} from the semi-monthly
    short_interest table in returns.db. Both arrays sorted ascending by date.
    Distinct from load_svr_data: this is settlement-date short *interest* (shares
    short / ADV), not daily short *volume*. Empty dict if table/DB absent.
    """
    if not RETURNS_DB.exists():
        return {}
    with get_db(RETURNS_DB) as conn:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='short_interest'"
        ).fetchall()]
        if not tables:
            return {}
        df = pd.read_sql_query(
            "SELECT isin, settlement_date, days_to_cover FROM short_interest "
            "WHERE days_to_cover IS NOT NULL ORDER BY isin, settlement_date",
            conn,
        )
    if df.empty:
        return {}
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    result: dict = {}
    for isin, grp in df.groupby("isin", sort=False):
        grp = grp.sort_values("settlement_date")
        result[isin] = (grp["settlement_date"].values,
                        grp["days_to_cover"].values.astype(np.float64))
    log.info("Loaded short-interest data for %s ISINs", f"{len(result):,}")
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


def load_splits() -> dict:
    """
    Returns {isin: [(split_date_ts, ratio), ...]} from returns.db `splits`, sorted by date.

    returns.db prices are split-adjusted to the current basis, but constituents.db
    shares are actual point-in-time counts.  These ratios let the market-cap path
    convert a historical share count to the current basis so price × shares is
    consistent across splits.  Empty dict when the table is absent (pre-backfill) —
    callers then leave shares unadjusted, preserving prior behaviour.
    """
    if not RETURNS_DB.exists():
        return {}
    with get_db(RETURNS_DB) as conn:
        has_tbl = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='splits'"
        ).fetchone()
        if not has_tbl:
            return {}
        df = pd.read_sql_query(
            "SELECT isin, date, ratio FROM splits WHERE ratio > 0 ORDER BY isin, date",
            conn,
        )
    splits: dict = {}
    for isin, grp in df.groupby("isin", sort=False):
        splits[isin] = [
            (pd.Timestamp(d), float(r)) for d, r in zip(grp["date"], grp["ratio"])
        ]
    if splits:
        log.info("Loaded split history for %s ISINs", f"{len(splits):,}")
    return splits


def _apply_split_adjustment(cdata: dict, isin: str, splits: dict) -> None:
    """
    Express the LTM share count in the current split-adjusted basis, in place.

    returns.db `close` is split-adjusted to today's basis; the share count is as of
    its source quarter (`_shares_asof`).  Multiplying by the product of every split
    ratio dated after that quarter restates shares onto the same basis as the price,
    so market cap is continuous through splits.  No-op when shares, the as-of date, or
    split history are missing.
    """
    shares = cdata.get('Shares (Basic)')
    asof   = cdata.get('_shares_asof')
    sid_splits = splits.get(isin)
    if shares is None or asof is None or not sid_splits:
        return
    asof_ts = pd.Timestamp(asof)
    factor = 1.0
    for split_date, ratio in sid_splits:
        if split_date > asof_ts:
            factor *= ratio
    if factor != 1.0:
        cdata['Shares (Basic)'] = shares * factor


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


def load_log_transform_factor_ids() -> set[str]:
    """Returns the set of factor_ids whose values are log-transformed before z-scoring.

    Used for always-positive ratio factors with heavy right-tail skew (e.g. EV/EBITDA,
    Leverage) where log(x) brings the distribution to near-normal before winsorization.
    Controlled via the log_transform column in factors_reference.csv.
    """
    df = pd.read_csv(FACTORS_REF)
    return set(df.loc[df['log_transform'] == True, 'factor_id'])


# _ALLOWED_FACTOR_SECTORS is imported from utils (shared with create_models, the
# model-layer coverage denominator) so the factor and model layers agree on which
# factors apply to a company.


# ---------------------------------------------------------------------------
# Point-in-time selection
# ---------------------------------------------------------------------------

def _build_ltm(quarters: list, kind_map: dict, min_flow_quarters: int) -> dict:
    """Sum Flow items / take latest Stock item across a window of quarters.

    Flow items require a complete window (>= min_flow_quarters) — partial public
    histories otherwise produce understated LTM that looks precise but is not
    economically comparable. Annual-only filers pass min_flow_quarters=1.
    """
    ltm: dict = {}
    flow_counts: dict = {}
    for q in quarters:
        # Derive Gross Profit per-quarter when absent but Revenue and Cost of
        # Revenue are both available. Sign convention is normalised (both
        # positive) so GP = Revenue - CoR holds for SimFin and EDGAR alike.
        if (q.get('Gross Profit') is None
                and q.get('Revenue') is not None
                and q.get('Cost of Revenue') is not None):
            q = {**q, 'Gross Profit': q['Revenue'] - q['Cost of Revenue']}

        # Derive Operating Income when absent.
        # Identity: Operating Income = Pretax Income - Non-Operating Income.
        # Some XBRL filers (e.g. LLY, banks) omit the OperatingIncomeLoss tag
        # but always report Pretax. When Non-Operating Income is also absent we
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
                if name == 'Shares (Basic)':
                    # Record the publish date of the quarter supplying the share
                    # count so the market-cap path can split-adjust it to the
                    # current basis (matching split-adjusted returns.db prices).
                    ltm['_shares_asof'] = q.get('_publish_date')
            else:
                ltm[name] = ltm.get(name, 0.0) + val
                flow_counts[name] = flow_counts.get(name, 0) + 1

    # Null out Flow items that lacked a complete window.
    for name, cnt in flow_counts.items():
        if cnt < min_flow_quarters:
            ltm[name] = None
    return ltm


def _available_quarters(
    sid_data: dict, kind_map: dict, snapshot: date,
) -> tuple[list, bool]:
    """Sorted (ascending) PIT quarters that contain a Flow item, plus the
    annual-only flag. Shared by select_ltm_data and select_growth_series so the
    LTM windowing rules stay identical."""
    snap_ts = pd.Timestamp(snapshot)

    def _has_flow(bucket: dict) -> bool:
        return any(
            not k.startswith('_') and v is not None and kind_map.get(k, 'Flow') == 'Flow'
            for k, v in bucket.items()
        )

    available = [
        q for q in sid_data.values()
        if q['_publish_date'] <= snap_ts and _has_flow(q)
    ]
    available.sort(key=lambda q: q['_sort_key'])

    is_annual_only = (
        len(available) >= 2 and
        all(
            available[i + 1]['_sort_key'] - available[i]['_sort_key'] >= 10
            for i in range(len(available) - 1)
        )
    )
    return available, is_annual_only


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

    Flow items require a full four-quarter window unless the filer is annual-only.
    Short-history companies still get Stock and price-based factors, but accounting
    ratios that depend on LTM flows stay null rather than using partial-year data.

    Known limitation: if a filer reports only YTD cumulative income statement facts
    in XBRL (no standalone 3-month tag), the stored quarterly values will be YTD
    and summing them here will overstate LTM. This is rare for Russell 1000 filers.
    The fix is to derive standalone quarters algebraically in update_constituents.py
    before storing.
    """
    available, is_annual_only = _available_quarters(sid_data, kind_map, snapshot)
    if not available:
        return {}, {}

    # Annual-only filers (e.g. SimFin annual coverage): all consecutive periods
    # are spaced ≥10 sort-key units apart (one Q4 per fiscal year, gap = 10).
    # Normal quarterly gaps are 1 (within year) or 7 (Q4→Q1 cross-year).
    # For these, the single most-recent period already represents a full LTM;
    # summing multiple annual Q4 periods would overstate by N×.
    if is_annual_only:
        recent_4 = available[-1:]
        prior_4  = available[-2:-1] if len(available) >= 2 else []
        min_flow = 1
    else:
        recent_4 = available[-4:]
        prior_4  = available[-8:-4] if len(available) >= 8 else []
        min_flow = 4

    return (_build_ltm(recent_4, kind_map, min_flow),
            _build_ltm(prior_4,  kind_map, min_flow))


# Flow metrics whose multi-year trend feeds the growth factors. EBITDA is
# derived per-window from Operating Income + |D&A|.
_GROWTH_TREND_FIELDS = (
    'Revenue', 'Net Income', 'Net Cash from Operating Activities',
    'Operating Income (Loss)',
)


def select_growth_series(
    sid_data: dict,
    kind_map: dict,
    snapshot: date,
    n_years: int = 5,
) -> dict:
    """Annual LTM series (oldest → newest) per growth metric for trend factors.

    Builds up to n_years complete trailing-12-month windows stepping back one
    fiscal year at a time (4 quarters per step; annual-only filers use one FY
    period per step). Each window is built with the same completeness rules as
    select_ltm_data, so a partial year is dropped rather than understated.

    Returns {metric_name: [v_oldest, …, v_newest]} including a derived
    'EBITDA' series. Series shorter than 2 points are omitted; the trend
    computation downstream requires ≥3.
    """
    available, is_annual_only = _available_quarters(sid_data, kind_map, snapshot)
    if not available:
        return {}

    step, min_flow = (1, 1) if is_annual_only else (4, 4)

    # Carve consecutive non-overlapping windows from the most recent backwards.
    windows: list[dict] = []
    end = len(available)
    while end - step >= 0 and len(windows) < n_years:
        windows.append(_build_ltm(available[end - step:end], kind_map, min_flow))
        end -= step
    windows.reverse()  # oldest → newest

    series: dict[str, list] = {}
    for field in _GROWTH_TREND_FIELDS:
        vals = [w.get(field) for w in windows]
        if all(v is not None for v in vals) and len(vals) >= 2:
            series[field] = vals

    ebitda = [
        _ebitda(w.get('Operating Income (Loss)'), w.get('Depreciation & Amortization'))
        for w in windows
    ]
    if all(v is not None for v in ebitda) and len(ebitda) >= 2:
        series['EBITDA'] = ebitda

    # FFO = Net Income + |D&A| per window (REIT growth, sector_type='reit').
    ffo = [
        (w['Net Income'] + abs(w['Depreciation & Amortization']))
        if w.get('Net Income') is not None and w.get('Depreciation & Amortization') is not None
        else None
        for w in windows
    ]
    if all(v is not None for v in ffo) and len(ffo) >= 2:
        series['FFO'] = ffo

    # Stock-item annual series (latest-in-window balance-sheet value) — feeds the
    # earnings-stability factor (cross-time ROA volatility), not the growth trends.
    for stock_field in ('Total Assets',):
        vals = [w.get(stock_field) for w in windows]
        if all(v is not None for v in vals) and len(vals) >= 2:
            series[stock_field] = vals

    return series


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
# Factor computation helpers
# ---------------------------------------------------------------------------

def _ebitda(op_income: float | None, da: float | None) -> float | None:
    """EBITDA = Operating Income + |D&A|. Returns None if either input is missing."""
    if op_income is None or da is None:
        return None
    return op_income + abs(da)


def _enterprise_value(
    market_cap: float,
    short_debt: float | None,
    long_debt: float | None,
    cash: float | None,
) -> float | None:
    """EV = market_cap + debt - cash. Returns None when EV ≤ 0."""
    ev = market_cap + (short_debt or 0) + (long_debt or 0) - (cash or 0)
    return ev if ev > 0 else None


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
    retained     = cdata.get('Retained Earnings')

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

    # Altman Z''-score (1995 non-manufacturer / emerging-market variant): a
    # distress/safety composite that drops the Sales/Assets term of the original
    # Z-score (which is heavily industry-sensitive). Higher = further from
    # bankruptcy = safer. The classification constant (+3.25) is omitted because a
    # cross-sectional z-score is invariant to it. Not meaningful for banks/REITs —
    # gated to general companies via the DEF001 model override.
    #   Z'' = 6.56·(WC/TA) + 3.26·(RE/TA) + 6.72·(EBIT/TA) + 1.05·(BookEquity/TL)
    if (assets is not None and assets > 0 and total_liab is not None and total_liab > 0
            and retained is not None and op_income is not None and equity is not None
            and cur_assets is not None and cur_liab is not None):
        working_capital = cur_assets - cur_liab
        f['Altman Z-Score'] = (
            6.56 * (working_capital / assets)
            + 3.26 * (retained / assets)
            + 6.72 * (op_income / assets)
            + 1.05 * (equity / total_liab)
        )

    return f


def compute_stability_factors(growth_series: dict) -> dict:
    """Earnings-stability safety factor (QMJ 'Safety' pillar).

    Cross-time volatility of annual ROA over the trailing window — lower
    variability of profitability = more stable earnings = safer (direction −1).
    ROA (Net Income / Total Assets) is used rather than ROE because many
    buyback-heavy large caps carry negative book equity, which makes ROE
    undefined; assets are always positive, preserving coverage. Requires ≥3
    aligned annual points (else NULL — matches the growth-trend minimum).
    """
    f = {}
    ni = growth_series.get('Net Income')
    ta = growth_series.get('Total Assets')
    if (ni and ta and len(ni) == len(ta) and len(ni) >= 3
            and all(a is not None and a > 0 for a in ta)):
        roa = [n / a for n, a in zip(ni, ta)]
        f['Earnings Stability'] = float(np.std(roa, ddof=1))
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
    capex      = cdata.get('Change in Fixed Assets & Intangibles')  # negative cash outflow
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
    # Book-to-Price only when equity is positive: negative book equity (buyback-heavy
    # names like ABBV/MCD/SBUX) is not "expensive" — B/M is undefined there (the
    # Fama-French convention excludes negative-BE firms rather than ranking them).
    if equity is not None and equity > 0:
        f['Book-to-Price'] = equity / market_cap
    if revenue is not None and revenue != 0:
        f['Sales-to-Price'] = revenue / market_cap
    if op_cf is not None:
        f['Cash Yield'] = op_cf / market_cap
    # FCF Yield: free cash flow (OCF − capex) over market cap. capex is stored as a
    # negative cash outflow, so op_cf + capex = OCF − |capex| (matches FCF Margin).
    if op_cf is not None and capex is not None:
        f['FCF Yield'] = (op_cf + capex) / market_cap
    if cash is not None and op_income is not None and op_income > 0:
        ev = _enterprise_value(market_cap, short_debt, long_debt, cash)
        if ev is not None:
            f['EV-to-EBIT'] = ev / op_income

    if cash is not None:
        ebitda = _ebitda(op_income, da)
        if ebitda is not None and ebitda > 0:
            ev = _enterprise_value(market_cap, short_debt, long_debt, cash)
            if ev is not None:
                f['EV/EBITDA'] = ev / ebitda

    # Dividend Yield: only for companies that actually paid cash dividends (Dividends Paid < 0)
    if dividends is not None and dividends < 0:
        f['Dividend Yield'] = abs(dividends) / market_cap

    return f


_GROWTH_MIN_YEARS = 3   # least-squares trend needs ≥3 annual points

# Map each growth factor to the series key produced by select_growth_series.
_GROWTH_FACTOR_FIELDS = (
    ('Revenue Growth',          'Revenue'),
    ('Earnings Growth',         'Net Income'),
    ('Cash Flow Growth',        'Net Cash from Operating Activities'),
    ('Operating Income Growth', 'Operating Income (Loss)'),
    ('EBITDA Growth',           'EBITDA'),
)


def _trend_growth(series: list) -> float | None:
    """MSCI-style growth: least-squares slope of the annual series divided by
    the mean absolute level. Robust to a single depressed/negative base year —
    a trough recovery no longer explodes the ratio (the KEY/asset-growth fix).

    Requires ≥3 points. Returns None for a missing series or a degenerate
    (~flat-zero) series whose mean level is not meaningfully positive.
    """
    if not series:
        return None
    n = len(series)
    if n < _GROWTH_MIN_YEARS:
        return None
    y = np.asarray(series, dtype=float)
    scale = float(np.mean(np.abs(y)))
    if scale <= 0:
        return None
    x = np.arange(n, dtype=float)
    # Closed-form OLS slope; var(x) > 0 for n ≥ 2.
    slope = float(np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2))
    return slope / scale


def compute_growth_factors(growth_series: dict) -> dict:
    """5 trend growth factors: annual-series slope ÷ mean(|level|) over the
    trailing 3–5 fiscal years (whatever history is available, min 3).

    Replaces the old 1-year YoY ratio, which was distorted by small/volatile
    base years (e.g. a bank recovering from a one-off loss showed +7,000% EPS
    growth). The trend measures *sustained* growth and sees through base effects.
    """
    f = {}
    for factor_name, field in _GROWTH_FACTOR_FIELDS:
        series = growth_series.get(field)
        if not series:
            continue
        v = _trend_growth(series)
        if v is not None:
            f[factor_name] = v
    return f


def compute_momentum_factors(
    isin: str, prices: dict, ref_date: date = None
) -> dict:
    """Risk-adjusted momentum and reversal factors.

    All factors share the same computation: compounded total return over the
    window divided by annualised realised volatility (std of log returns × √252).

      ST Reversal   : T−1m  → T       (the skipped recent month; direction=−1)
      6M Momentum   : T−6m  → T−1m    (skip-month)
      12M Momentum  : T−12m → T−1m    (skip-month)
      LT Reversal   : T−36m → T−13m   (window before momentum; direction=−1)
    """
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

    for start_months, end_months, name, min_obs in [
        (1,  0,  "ST Reversal",  15),
        (6,  1,  "6M Momentum",  20),
        (12, 1,  "12M Momentum", 20),
        (36, 13, "LT Reversal",  20),
    ]:
        end_dt    = ref_dt - relativedelta(months=end_months)
        end_np    = np.datetime64(end_dt, "D").astype("datetime64[ns]")
        end_idx   = int(np.searchsorted(dates, end_np, side="right")) - 1

        start_dt  = ref_dt - relativedelta(months=start_months)
        start_np  = np.datetime64(start_dt, "D").astype("datetime64[ns]")
        start_idx = int(np.searchsorted(dates, start_np, side="right"))

        if end_idx < 1 or start_idx > end_idx:
            continue
        window   = total_rets[start_idx : end_idx + 1]
        valid    = window[np.isfinite(window)]
        if len(valid) < min_obs:
            continue
        ret      = float(np.prod(1.0 + valid) - 1.0)
        log_rets = np.log(1.0 + valid)
        log_rets = log_rets[np.isfinite(log_rets)]
        vol      = float(np.std(log_rets) * np.sqrt(252))
        if vol <= 0:
            continue
        f[name]  = ret / vol
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


def compute_short_interest_factor(
    isin: str, si_data: dict, ref_date: date = None, pub_lag_days: int = 14
) -> dict:
    """
    Days to Cover — FINRA consolidated short interest ratio (shares short / ADV)
    as of the latest settlement date disseminated on/before the snapshot.

    Direction=-1: higher days-to-cover = more short pressure = worse signal.
    `pub_lag_days` (14 calendar days) guards against look-ahead — FINRA publishes
    ~8 business days after each settlement date, so a snapshot can only "know" a
    settlement date that is at least ~2 weeks old. Value is left unsigned (the
    factor is log-transformed and z-scored downstream).
    """
    entry = si_data.get(isin)
    if entry is None:
        return {}
    sdates, dtc = entry

    cutoff = (ref_date - timedelta(days=pub_lag_days)) if ref_date is not None else None
    if cutoff is None:
        idx = len(sdates) - 1
    else:
        cutoff_np = np.datetime64(cutoff, "D").astype("datetime64[ns]")
        idx = int(np.searchsorted(sdates, cutoff_np, side="right")) - 1
    if idx < 0:
        return {}

    val = float(dtc[idx])
    if not np.isfinite(val) or val < 0:
        return {}
    return {"Days to Cover": val}


def compute_reit_factors(
    cdata: dict, growth_series: dict, isin: str, prices: dict, ref_date: date = None
) -> dict:
    """
    REIT-specific factors based on Funds From Operations (FFO).
    FFO = Net Income + Depreciation & Amortization (D&A from CF statement).
    Only called for companies with sector_type == 'reit'.

    FFO Growth uses the same multi-year trend method as the other growth factors
    (slope of the annual FFO series ÷ mean(|FFO|)) — see _trend_growth.
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

    # FFO Growth — multi-year trend of the annual FFO series.
    ffo_growth = _trend_growth(growth_series.get('FFO'))
    if ffo_growth is not None:
        f['FFO Growth'] = ffo_growth

    # FFO Payout = distributions paid / FFO. Lower payout = bigger coverage cushion
    # = safer (direction −1). Dividends Paid is stored negative (cash outflow).
    dividends = cdata.get('Dividends Paid')
    if dividends is not None and dividends < 0 and ffo > 0:
        payout = abs(dividends) / ffo
        if payout <= 5:   # cap at 500%; higher = unreliable FFO/dividend data
            f['FFO Payout'] = payout

    return f


def compute_bank_factors(
    cdata: dict, isin: str, prices: dict, ref_date: date = None
) -> dict:
    """
    Bank-specific factors for sector_type == 'bank' (depository banks + consumer
    lenders). Built from EDGAR bank line items ingested 2026-06 (Net Interest
    Income, Noninterest Income/Expense, Provision/NII-after-provision, Loans,
    Goodwill, Intangibles). Every factor is NULL-safe — a bank missing a concept
    (e.g. JPM, which combines goodwill+intangibles) simply omits that one factor
    and is renormalised over what it has.

    Generic margin/revenue factors don't fit banks, so banks are gated out of
    those (sector_type) and scored here on bank economics instead:
      Net Interest Margin    NII / total assets            (+1; earning-asset proxy)
      Efficiency Ratio       NonIntExp / (NII + NonIntInc) (−1; lower = leaner)
      PPOP Return on Assets  (NII + NonIntInc − NonIntExp) / assets (+1)
      Credit Cost            provision / loans             (−1; higher = riskier)
      PPOP Yield             PPOP / market cap             (+1; value)
      Tangible Book-to-Price (equity − goodwill − intang) / market cap (+1; value)
    """
    f = {}
    nii    = cdata.get('Net Interest Income')
    nonii  = cdata.get('Noninterest Income')
    nonie  = cdata.get('Noninterest Expense')
    niiap  = cdata.get('Net Interest Income After Provision')
    prov   = cdata.get('Provision for Credit Losses')
    loans  = cdata.get('Loans Receivable')
    gw     = cdata.get('Goodwill')
    intang = cdata.get('Intangible Assets')
    assets = cdata.get('Total Assets')
    equity = cdata.get('Total Equity')

    # Pre-provision operating profit (pre-provision, pre-tax).
    ppop = (nii + nonii - nonie) if (nii is not None and nonii is not None
                                     and nonie is not None) else None

    if nii is not None and assets is not None and assets > 0:
        f['Net Interest Margin'] = nii / assets

    if nii is not None and nonii is not None and nonie is not None:
        total_rev = nii + nonii
        if total_rev > 0:
            f['Efficiency Ratio'] = nonie / total_rev

    if ppop is not None and assets is not None and assets > 0:
        f['PPOP Return on Assets'] = ppop / assets

    # Credit Cost: prefer the direct provision; else derive NII − NII-after-provision.
    provision = prov if prov is not None else (
        (nii - niiap) if (nii is not None and niiap is not None) else None)
    if provision is not None and loans is not None and loans > 0:
        f['Credit Cost'] = provision / loans

    shares = cdata.get('Shares (Basic)')
    price  = get_close(prices, isin, ref_date=ref_date)
    if price and shares and price > 0 and shares > 0:
        market_cap = price * shares
        if ppop is not None:
            f['PPOP Yield'] = ppop / market_cap
        # Tangible book only when positive (mirrors the Book-to-Price equity gate):
        # goodwill/intangibles default to 0 when a bank doesn't break them out.
        if equity is not None and equity > 0:
            tangible = equity - (gw or 0) - (intang or 0)
            if tangible > 0:
                f['Tangible Book-to-Price'] = tangible / market_cap

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
    splits: dict,
    svr_data: dict,
    si_data: dict,
    factor_name_to_id: dict,
    factor_sector_types: dict,
    log_transform_ids: set[str],
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

        # Growth factors use the multi-year trend series, not the prior-year LTM.
        cdata, _ = select_ltm_data(sid_data, kind_map, snapshot)
        if not cdata:
            continue
        growth_series = select_growth_series(sid_data, kind_map, snapshot)

        # Restate point-in-time shares onto the current split-adjusted basis so
        # market cap (split-adjusted price × shares) is continuous through splits.
        _apply_split_adjustment(cdata, isin, splits)

        sector_type = meta.get('sector_type', 'general')

        factors: dict = {}
        factors.update(compute_quality_factors(cdata))
        factors.update(compute_value_factors(cdata, isin, prices, ref_date=snapshot))
        factors.update(compute_growth_factors(growth_series))
        factors.update(compute_stability_factors(growth_series))
        factors.update(compute_momentum_factors(isin, prices, ref_date=snapshot))
        factors.update(compute_size_factor(cdata, isin, prices, ref_date=snapshot))
        factors.update(compute_low_vol_factors(isin, prices, ref_date=snapshot))
        factors.update(compute_liquidity_factors(isin, prices, ref_date=snapshot))
        factors.update(compute_svr_factors(isin, svr_data, ref_date=snapshot))
        factors.update(compute_short_interest_factor(isin, si_data, ref_date=snapshot))
        if sector_type == 'reit':
            factors.update(compute_reit_factors(cdata, growth_series, isin, prices, ref_date=snapshot))
        if sector_type == 'bank':
            factors.update(compute_bank_factors(cdata, isin, prices, ref_date=snapshot))

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

    # A snapshot rebuild is authoritative for that date. Delete existing rows
    # first so factors that no longer pass data-quality gates do not linger from
    # an older, more permissive run.
    conn.execute("DELETE FROM factors WHERE data_date = ?", (date_str,))
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
    # For heavily right-skewed always-positive factors (EV/EBITDA, EV-to-EBIT, Leverage),
    # z-score on log(value) so the distribution is near-normal before winsorization.
    # Raw factor_value is preserved unchanged in the DB.
    z_input = df['factor_value'].copy()
    z_input[df['factor_id'].isin(log_transform_ids)] = np.log(
        df.loc[df['factor_id'].isin(log_transform_ids), 'factor_value']
    )
    df['factor_value_z'] = (
        df.assign(_z=z_input)
        .groupby('factor_id')['_z']
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
    grp.add_argument('--schedule',          action='store_true',
                     help='Compute all scheduled dates not yet computed (the snapshot_schedule '
                          'table in universe.db is the single source of truth)')
    grp.add_argument('--schedule-all',      action='store_true',
                     help='Recompute EVERY date in the schedule (full restatement)')
    grp.add_argument('--backfill',          action='store_true',
                     help='[deprecated alias for --schedule]')
    grp.add_argument('--quarterly-backfill', action='store_true',
                     help='[deprecated alias for --schedule]')
    parser.add_argument('--clean', action='store_true',
                        help='Drop and rebuild the factors table before running')
    args = parser.parse_args()

    if args.schedule_all:
        scheduled = get_snapshot_schedule()
        dates_to_run = [datetime.strptime(d, '%Y-%m-%d').date() for d in scheduled]
    elif args.schedule or args.backfill or args.quarterly_backfill:
        # Pending = scheduled but factors not yet computed (single source of truth).
        computed = set(get_snapshot_schedule(computed_only=True))
        pending  = [d for d in get_snapshot_schedule() if d not in computed]
        dates_to_run = [datetime.strptime(d, '%Y-%m-%d').date() for d in pending]
        log.info("Schedule: %d pending date(s) of %d total", len(pending), len(get_snapshot_schedule()))
    elif args.dates:
        dates_to_run = sorted({datetime.strptime(d, '%Y-%m-%d').date() for d in args.dates})
    else:
        dates_to_run = [date.today()]

    log.info("Loading reference data ...")
    universe            = load_universe()
    ticker_map          = load_ticker_map()
    factor_name_to_id   = load_factor_name_to_id()
    factor_sector_types = load_factor_sector_types()
    log_transform_ids   = load_log_transform_factor_ids()
    kind_map            = load_kind_map()

    log.info("Loading constituent data from constituents.db ...")
    constituent_data = load_constituent_data()

    prices   = load_price_data()
    splits   = load_splits()
    svr_data = load_svr_data()
    si_data  = load_short_interest_data()

    with get_db(FACTORS_DB) as conn:
        setup_factors_db(conn, clean=args.clean)

        total_rows = 0
        log.info("Processing %d date(s): %s", len(dates_to_run), [str(d) for d in dates_to_run])
        for snapshot in dates_to_run:
            total_rows += run_for_date(
                snapshot, universe, ticker_map,
                constituent_data, kind_map, prices, splits, svr_data, si_data,
                factor_name_to_id, factor_sector_types, log_transform_ids, conn,
            )
            conn.execute(
                "INSERT OR REPLACE INTO snapshot_dates (data_date, created_at) "
                "VALUES (?, datetime('now'))",
                (str(snapshot),),
            )
            conn.commit()
            # Stamp the single source of truth (universe.db snapshot_schedule).
            mark_snapshot_computed(str(snapshot))

    log.info("Done — %s total factor rows across %d date(s)", f"{total_rows:,}", len(dates_to_run))


if __name__ == "__main__":
    main()
