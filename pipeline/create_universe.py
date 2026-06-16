"""
create_universe.py

Maintains universe.db. The production universe membership source is EDGAR
N-PORT-P filings for registered ETF proxies such as IVV (S&P 500) and IWB
(Russell 1000). Local iShares holdings CSVs and SimFin metadata are retained
only as explicit legacy/bootstrap paths.

Core tables:

  companies                — security master keyed by ISIN. Still contains some
                             legacy SimFin fields while the EDGAR-first security
                             master is being completed.

  universe_snapshots       — raw point-in-time index membership, one row per
                             (snapshot_date, isin, index_name), normally rebuilt
                             from N-PORT accessions.

  clean_universe_snapshots — optimizer-facing cleaned membership with identity
                             mapping/tradability audit fields.

Common production commands:
  python pipeline/create_universe.py --ensure-snapshot YYYY-MM-DD
  python pipeline/create_universe.py --rebuild-snapshots
  python pipeline/create_universe.py --materialize-clean-snapshots --clean-mode live --clean-latest-only

Legacy/bootstrap only:
  python pipeline/create_universe.py --legacy-rebuild-companies
  python pipeline/create_universe.py --rebuild-snapshots --include-legacy-csv
"""

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import sys

import pandas as pd
import sqlite3
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    DATA_DIR, SIMFIN_DIR, UNIVERSE_DB as DB_PATH, FACTORS_DB, CONSTITUENTS_DB,
    SCHEDULE_MONTHLY_START, SCHEDULE_WEEKLY_CUTOVER,
)
from utils import get_db, get_logger

log = get_logger("create_universe")

INDEX_DIR  = DATA_DIR / "universe_index"

# iShares sector label → official GICS sector name
GICS_SECTOR_NORM: dict[str, str] = {
    "Communication":          "Communication Services",
    "Consumer Discretionary": "Consumer Discretionary",
    "Consumer Staples":       "Consumer Staples",
    "Energy":                 "Energy",
    "Financials":             "Financials",
    "Health Care":            "Health Care",
    "Industrials":            "Industrials",
    "Information Technology": "Information Technology",
    "Materials":              "Materials",
    "Real Estate":            "Real Estate",
    "Utilities":              "Utilities",
}


# ---------------------------------------------------------------------------
# Reference table helpers
#
# These four tables in universe.db are the single source of truth for all
# security/index mappings. They are never dropped on rebuild — edits made
# directly in the DB survive full reruns.
#
# To populate on a fresh DB, restore universe.db from backup (Time Machine).
# The tables are small and stable; they don't need to be reproduced from code.
# ---------------------------------------------------------------------------

def seed_isin_patch_table(conn: "sqlite3.Connection") -> None:
    """Create isin_patch table if it doesn't exist yet."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS isin_patch (
            ticker TEXT PRIMARY KEY,
            isin   TEXT NOT NULL,
            note   TEXT
        )
    """)
    conn.commit()


def load_isin_patch() -> dict[str, str]:
    """Return ticker→ISIN overrides from universe.db."""
    with get_db(DB_PATH) as conn:
        rows = conn.execute("SELECT ticker, isin FROM isin_patch").fetchall()
    return {r[0]: r[1] for r in rows}


def seed_ticker_alias_table(conn: "sqlite3.Connection") -> None:
    """Create ticker_alias table if it doesn't exist yet."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ticker_alias (
            ticker       TEXT PRIMARY KEY,
            alias_ticker TEXT NOT NULL,
            note         TEXT
        )
    """)
    conn.commit()


def load_ticker_alias() -> dict[str, str]:
    """Return iShares ticker→SimFin ticker aliases from universe.db."""
    with get_db(DB_PATH) as conn:
        rows = conn.execute("SELECT ticker, alias_ticker FROM ticker_alias").fetchall()
    return {r[0]: r[1] for r in rows}


def seed_simfin_exclude_table(conn: "sqlite3.Connection") -> None:
    """Create simfin_exclude table if it doesn't exist yet."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS simfin_exclude (
            ticker TEXT PRIMARY KEY,
            note   TEXT
        )
    """)
    conn.commit()


def load_simfin_exclude() -> set[str]:
    """Return iShares tickers that must not be enriched from SimFin by ticker."""
    with get_db(DB_PATH) as conn:
        rows = conn.execute("SELECT ticker FROM simfin_exclude").fetchall()
    return {str(r[0]).upper().strip() for r in rows}


def seed_security_data_start_table(conn: "sqlite3.Connection") -> None:
    """Create security_data_start table if it doesn't exist yet."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS security_data_start (
            isin            TEXT PRIMARY KEY,
            min_report_date TEXT NOT NULL,
            note            TEXT
        )
    """)
    conn.commit()


def seed_registry_tables(conn: "sqlite3.Connection") -> None:
    """Create index_registry and nport_accessions tables if they don't exist yet."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS index_registry (
            index_name    TEXT PRIMARY KEY,
            etf_ticker    TEXT NOT NULL,
            etf_name      TEXT NOT NULL,
            series_id     TEXT,
            cik           TEXT,
            is_investable INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nport_accessions (
            index_name    TEXT NOT NULL,
            snapshot_date TEXT NOT NULL,
            accession     TEXT NOT NULL,
            period_ending TEXT,
            PRIMARY KEY (index_name, snapshot_date),
            FOREIGN KEY (index_name) REFERENCES index_registry(index_name)
        )
    """)
    conn.commit()


def seed_nport_security_metadata_table(conn: "sqlite3.Connection") -> None:
    """Create N-PORT security metadata staging table if it does not exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nport_security_metadata (
            accession          TEXT NOT NULL,
            filing_cik         TEXT NOT NULL,
            isin               TEXT NOT NULL,
            security_name      TEXT,
            security_title     TEXT,
            cusip              TEXT,
            lei                TEXT,
            balance            REAL,
            units              TEXT,
            currency           TEXT,
            market_value       REAL,
            weight             REAL,
            payoff_profile     TEXT,
            asset_category     TEXT,
            issuer_category    TEXT,
            investment_country TEXT,
            is_restricted      TEXT,
            fair_value_level   TEXT,
            fetched_at         TEXT NOT NULL,
            PRIMARY KEY (accession, isin)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_nport_security_metadata_isin
        ON nport_security_metadata(isin)
    """)
    conn.commit()


def seed_nport_company_candidates_table(conn: "sqlite3.Connection") -> None:
    """Create derived N-PORT company candidate staging table if it does not exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nport_company_candidates (
            isin                       TEXT PRIMARY KEY,
            security_name              TEXT,
            security_title             TEXT,
            cusip                      TEXT,
            lei                        TEXT,
            currency                   TEXT,
            investment_country         TEXT,
            first_snapshot_date        TEXT,
            last_snapshot_date         TEXT,
            max_weight                 REAL,
            seen_indexes               TEXT,
            accessions                 TEXT,
            company_status             TEXT NOT NULL,
            existing_ticker            TEXT,
            existing_company_name      TEXT,
            existing_gics_sector       TEXT,
            existing_gics_industry     TEXT,
            resolved_ticker            TEXT,
            resolved_cik               TEXT,
            resolved_company_name      TEXT,
            resolved_exchange          TEXT,
            resolution_status          TEXT,
            resolution_confidence      REAL,
            resolver_sources           TEXT,
            resolution_evidence        TEXT,
            resolved_at                TEXT,
            candidate_source           TEXT NOT NULL,
            staged_at                  TEXT NOT NULL
        )
    """)
    existing_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(nport_company_candidates)").fetchall()
    }
    for col_name, col_type in {
        "resolved_ticker": "TEXT",
        "resolved_cik": "TEXT",
        "resolved_company_name": "TEXT",
        "resolved_exchange": "TEXT",
        "resolution_status": "TEXT",
        "resolution_confidence": "REAL",
        "resolver_sources": "TEXT",
        "resolution_evidence": "TEXT",
        "resolved_at": "TEXT",
    }.items():
        if col_name not in existing_cols:
            conn.execute(f"ALTER TABLE nport_company_candidates ADD COLUMN {col_name} {col_type}")
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_nport_company_candidates_status
        ON nport_company_candidates(company_status)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_nport_company_candidates_last_snapshot
        ON nport_company_candidates(last_snapshot_date)
    """)
    conn.commit()


def seed_snapshot_schedule_table(conn: "sqlite3.Connection") -> None:
    """Create the snapshot_schedule table — the single source of truth for snapshot dates."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS snapshot_schedule (
            data_date           TEXT PRIMARY KEY,
            cadence             TEXT NOT NULL,        -- 'monthly' | 'weekly' | 'legacy'
            factors_computed_at TEXT,                 -- stamped by create_factors when computed
            created_at          TEXT NOT NULL
        )
    """)
    conn.commit()


def rebuild_snapshot_schedule(
    monthly_start: str = SCHEDULE_MONTHLY_START,
    weekly_cutover: str = SCHEDULE_WEEKLY_CUTOVER,
) -> None:
    """
    (Re)build the canonical snapshot calendar in universe.db — the single source of
    truth that create_factors / create_risk / create_barra all read.

    Rule:
      - 'monthly' : month-end of every month from monthly_start up to weekly_cutover.
      - 'weekly'  : existing computed snapshot dates on/after weekly_cutover (the
                    weekly cadence added by daily_ecosystem_update.py — left as-is, not generated).
      - 'legacy'  : existing computed snapshot dates before weekly_cutover that are not
                    month-ends (the old 15th-quarterly / April-1-annual grid) — kept,
                    tagged, never recomputed.

    Idempotent: re-running updates cadence tags but preserves factors_computed_at
    (which create_factors stamps as it computes each date).
    """
    month_ends = [d.strftime("%Y-%m-%d")
                  for d in pd.date_range(monthly_start, weekly_cutover, freq="ME")]

    # Discover already-computed dates from factors.db to tag weekly/legacy and to
    # bootstrap factors_computed_at for dates that already exist.
    computed: set[str] = set()
    try:
        with get_db(FACTORS_DB) as fconn:
            computed = {r[0] for r in fconn.execute("SELECT data_date FROM snapshot_dates").fetchall()}
    except Exception as exc:                       # factors.db / table may not exist yet
        log.warning("Could not read factors.db snapshot_dates (%s) — schedule built from rule only", exc)

    cadence: dict[str, str] = {d: "monthly" for d in month_ends}
    for d in computed:
        if d >= weekly_cutover:
            cadence[d] = "weekly"
        elif d not in cadence:
            cadence[d] = "legacy"

    now = datetime.now().isoformat(timespec="seconds")
    with get_db(DB_PATH) as conn:
        seed_snapshot_schedule_table(conn)
        for d, cad in sorted(cadence.items()):
            conn.execute(
                "INSERT INTO snapshot_schedule (data_date, cadence, factors_computed_at, created_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(data_date) DO UPDATE SET cadence = excluded.cadence",
                (d, cad, now if d in computed else None, now),
            )
        conn.commit()
        counts = dict(conn.execute(
            "SELECT cadence, COUNT(*) FROM snapshot_schedule GROUP BY cadence").fetchall())
        n_pending = conn.execute(
            "SELECT COUNT(*) FROM snapshot_schedule WHERE factors_computed_at IS NULL").fetchone()[0]
    log.info("snapshot_schedule rebuilt: %s | %d date(s) pending factor computation",
             counts, n_pending)


def load_index_registry() -> dict[str, dict]:
    """Load index registry from universe.db index_registry + nport_accessions tables."""
    with get_db(DB_PATH) as conn:
        reg_rows = conn.execute(
            "SELECT index_name, etf_ticker, etf_name, series_id, cik FROM index_registry"
        ).fetchall()
        acc_rows = conn.execute(
            "SELECT index_name, snapshot_date, accession, period_ending FROM nport_accessions"
        ).fetchall()
    result: dict[str, dict] = {}
    for r in reg_rows:
        result[r[0]] = {
            "etf_ticker": r[1], "etf_name": r[2],
            "series_id": r[3], "cik": r[4], "filings": {},
        }
    for r in acc_rows:
        if r[0] in result:
            result[r[0]]["filings"][r[1]] = (r[2], r[3])
    return result


def seed_all_reference_tables(conn: "sqlite3.Connection") -> None:
    """Ensure all persistent reference tables exist (schema only, no data insertion)."""
    seed_isin_patch_table(conn)
    seed_ticker_alias_table(conn)
    seed_simfin_exclude_table(conn)
    seed_security_data_start_table(conn)
    seed_registry_tables(conn)
    seed_nport_security_metadata_table(conn)
    seed_nport_company_candidates_table(conn)
    seed_snapshot_schedule_table(conn)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_simfin() -> pd.DataFrame:
    """Merge SimFin companies + industries into one normalised DataFrame."""
    companies  = pd.read_csv(SIMFIN_DIR / "companies.csv")
    industries = pd.read_csv(SIMFIN_DIR / "industries.csv", sep=";")

    companies["IndustryId"]  = companies["IndustryId"].fillna(0).astype(int)
    industries["IndustryId"] = industries["IndustryId"].astype(int)

    df = companies.merge(
        industries[["IndustryId", "Industry", "Sector"]],
        on="IndustryId", how="left",
    )
    df = df.rename(columns={
        "SimFinId":                      "simfin_id",
        "Ticker":                        "ticker",
        "Company Name":                  "company_name",
        "ISIN":                          "isin",
        "CIK":                           "cik",
        "End of financial year (month)": "fiscal_year_end",
        "Number Employees":              "num_employees",
        "Business Summary":              "business_summary",
        "Main Currency":                 "currency",
        "Industry":                      "simfin_industry",
        "Sector":                        "simfin_sector",
    })
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["cik"]    = pd.to_numeric(df["cik"],    errors="coerce").astype("Int64")
    df["simfin_id"] = pd.to_numeric(df["simfin_id"], errors="coerce").astype("Int64")
    return df


# Canonical index name for known iShares ETF products.
# Key = ETF name as it appears in the first line of the CSV (lower-cased).
_ISHARES_ETF_TO_INDEX: dict[str, str] = {
    "ishares russell 1000 etf": "russell_1000",
    "ishares russell 2000 etf": "russell_2000",
    "ishares russell 3000 etf": "russell_3000",
    "ishares msci usa etf":     "msci_usa",
    "ishares core s&p 500 etf": "sp_500",
}


def _parse_ishares_date(path: Path) -> str:
    """Extract snapshot date string (YYYY-MM-DD) from an iShares holdings CSV header.

    Handles multiple date formats: 'May 04, 2026'  |  '07/May/2026'
    """
    if path.suffix.lower() != ".csv":
        raise ValueError(f"iShares holdings file must be CSV, got: {path.name}")

    _FMTS = ["%B %d, %Y", "%B %d %Y", "%d/%b/%Y"]
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            if "Fund Holdings as of" not in line:
                continue
            parts = line.strip().split(",")
            date_str = " ".join(p.strip().strip('"') for p in parts[1:]).strip()
            for fmt in _FMTS:
                try:
                    return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
                except ValueError:
                    continue
            raise ValueError(f"Could not parse date '{date_str}' in {path}")
    raise ValueError(f"'Fund Holdings as of' not found in {path}")


def _infer_index_name(path: Path) -> str:
    """Infer the canonical index name from an iShares holdings CSV.

    Reads the ETF name from the first line of the file and maps it via
    _ISHARES_ETF_TO_INDEX. Falls back to stripping trailing date parts
    from the filename stem (e.g. russell_1000_2026_05_04 → russell_1000).
    """
    if path.suffix.lower() == ".csv":
        with open(path, encoding="utf-8-sig") as f:
            first = f.readline().strip().strip('"')
        idx = _ISHARES_ETF_TO_INDEX.get(first.lower())
        if idx:
            return idx

    name_parts: list[str] = []
    for p in path.stem.lower().split("_"):
        if len(p) == 4 and p.isdigit() and int(p) >= 1900:
            break
        name_parts.append(p)
    return "_".join(name_parts) if name_parts else "unknown_index"


def load_ishares(path: Path) -> tuple[pd.DataFrame, str, str]:
    """Parse an iShares holdings CSV.  Returns (equity_df, snapshot_date_str, index_name)."""
    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"iShares holdings files must be CSV (got {path.name}). "
            "Download the holdings CSV from iShares — not the Excel/XLS version."
        )
    snapshot_date = _parse_ishares_date(path)
    index_name    = _infer_index_name(path)

    # Find the row index of the column header (first row starting with "Ticker,").
    # Different iShares products have different numbers of preamble rows.
    header_row: int | None = None
    with open(path, encoding="utf-8-sig") as fh:
        for i, line in enumerate(fh):
            if line.strip().startswith("Ticker,") or line.strip() == "Ticker":
                header_row = i
                break
    if header_row is None:
        raise ValueError(f"Could not find 'Ticker' column header in {path}")
    df = pd.read_csv(path, skiprows=header_row, encoding="utf-8-sig")

    eq = df[df["Asset Class"] == "Equity"].copy()
    for col in ["Market Value", "Weight (%)", "Price"]:
        if col in eq.columns:
            eq[col] = (
                eq[col].astype(str).str.replace(",", "", regex=False)
                       .apply(pd.to_numeric, errors="coerce")
            )
    eq["Ticker"] = eq["Ticker"].astype(str).str.strip().str.upper()
    return eq.reset_index(drop=True), snapshot_date, index_name


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def build_companies(
    ishares_frames: list[tuple[pd.DataFrame, str, str]],
    simfin: pd.DataFrame,
    patch: dict[str, str] | None = None,
    alias: dict[str, str] | None = None,
    simfin_exclude: set[str] | None = None,
) -> pd.DataFrame:
    """
    Build the companies table from all iShares snapshots merged with SimFin.
    One row per unique iShares ticker (deduped by ISIN, latest snapshot wins).

    patch: ISIN overrides (ticker → ISIN). If None, load_isin_patch() is used.
    alias: SimFin ticker aliases (iShares ticker → SimFin ticker). If None, load_ticker_alias() is used.
    simfin_exclude: iShares tickers to keep off SimFin ticker matching. Use for
                    ticker reuse where SimFin still describes an old issuer.
    """
    _patch = patch if patch is not None else load_isin_patch()
    _alias = alias if alias is not None else load_ticker_alias()
    _simfin_exclude = simfin_exclude if simfin_exclude is not None else load_simfin_exclude()
    today = datetime.now().date().isoformat()

    # SimFin lookup by ticker
    sf_by_ticker: dict[str, pd.Series] = {
        str(r["ticker"]): r
        for _, r in simfin.iterrows()
        if str(r["ticker"]) not in ("NAN", "")
    }

    seen_isins: dict[str, dict] = {}   # isin → row dict (latest snapshot wins)

    for eq, snapshot_date, index_name in ishares_frames:
        for _, ih in eq.iterrows():
            ticker = str(ih["Ticker"]).upper().strip()
            if not ticker or ticker in ("NAN", "-"):
                continue

            # Match to SimFin: direct ticker, then alias from DB. Some tickers
            # are reused by unrelated issuers; those are excluded by reference
            # table so iShares/EDGAR metadata remains authoritative.
            sf = None
            if ticker not in _simfin_exclude:
                sf = sf_by_ticker.get(ticker)
                if sf is None:
                    sf = sf_by_ticker.get(_alias.get(ticker, ""))

            # Resolve ISIN: patch always wins over SimFin (SimFin has wrong ISINs for ~39 companies)
            isin: str | None = _patch.get(ticker)
            if not isin and sf is not None and pd.notna(sf.get("isin")) and str(sf["isin"]).strip():
                isin = str(sf["isin"]).strip()
            if not isin:
                isin = f"NOISN_{ticker}"   # synthetic placeholder

            row: dict = {
                "isin":               isin,
                "ticker":             ticker,
                "company_name":       (
                    str(sf["company_name"])
                    if sf is not None and pd.notna(sf.get("company_name"))
                    else str(ih["Name"])
                ),
                "gics_sector":        GICS_SECTOR_NORM.get(str(ih.get("Sector", "")), str(ih.get("Sector", ""))),
                "gics_industry_group": None,
                "gics_industry":      None,
                "gics_sub_industry":  None,
                "country":            str(ih.get("Location", "")),
                "exchange":           str(ih.get("Exchange", "")),
                "currency":           str(ih.get("Currency", "USD")),
                "fiscal_year_end":    (
                    int(sf["fiscal_year_end"])
                    if sf is not None and pd.notna(sf.get("fiscal_year_end"))
                    else None
                ),
                "num_employees":      (
                    int(sf["num_employees"])
                    if sf is not None and pd.notna(sf.get("num_employees"))
                    else None
                ),
                "business_summary":   (
                    str(sf["business_summary"])
                    if sf is not None and pd.notna(sf.get("business_summary"))
                    else None
                ),
                "cik":                (
                    str(int(sf["cik"]))
                    if sf is not None and pd.notna(sf.get("cik"))
                    else None
                ),
                "cusip":              None,
                "simfin_id":          (
                    int(sf["simfin_id"])
                    if sf is not None and pd.notna(sf.get("simfin_id"))
                    else None
                ),
                "simfin_sector":      (
                    str(sf["simfin_sector"])
                    if sf is not None and pd.notna(sf.get("simfin_sector"))
                    else None
                ),
                "simfin_industry":    (
                    str(sf["simfin_industry"])
                    if sf is not None and pd.notna(sf.get("simfin_industry"))
                    else None
                ),
                "data_date":          snapshot_date,
                "update_date":        today,
            }

            # Later snapshots overwrite earlier ones for the same ISIN
            seen_isins[isin] = row

    df = pd.DataFrame(list(seen_isins.values()))
    return df.reset_index(drop=True)


def build_snapshots(
    ishares_frames: list[tuple[pd.DataFrame, str, str]],
    companies: pd.DataFrame,
    alias: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build universe_snapshots from all iShares frames."""
    _alias = alias if alias is not None else load_ticker_alias()
    isin_by_ticker: dict[str, str] = dict(zip(companies["ticker"], companies["isin"]))

    rows = []
    for eq, snapshot_date, index_name in ishares_frames:
        seen: set[tuple] = set()
        for _, ih in eq.iterrows():
            ticker = str(ih["Ticker"]).upper().strip()
            isin   = isin_by_ticker.get(ticker)
            if isin is None:
                alias_ticker = _alias.get(ticker)
                if alias_ticker:
                    isin = isin_by_ticker.get(alias_ticker)
            if isin is None:
                continue
            key = (snapshot_date, isin, index_name)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "snapshot_date": snapshot_date,
                "isin":          isin,
                "index_name":    index_name,
                "weight":        float(ih["Weight (%)"]) if pd.notna(ih.get("Weight (%)")) else None,
                "market_value":  float(ih["Market Value"]) if pd.notna(ih.get("Market Value")) else None,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Historical universe snapshots from EDGAR N-PORT-P
# ---------------------------------------------------------------------------

def _nport_text(inv: ET.Element, ns: dict[str, str], tag: str) -> str | None:
    el = inv.find(f"n:{tag}", ns)
    if el is None or el.text is None:
        return None
    text = el.text.strip()
    return text or None


def _nport_float(inv: ET.Element, ns: dict[str, str], tag: str) -> float | None:
    text = _nport_text(inv, ns, tag)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_nport_ec_holdings(xml_data: bytes) -> list[dict]:
    root = ET.fromstring(xml_data)
    ns = {"n": "http://www.sec.gov/edgar/nport"}
    holdings = []
    for inv in root.findall(".//n:invstOrSec", ns):
        asset_category = _nport_text(inv, ns, "assetCat")
        if asset_category != "EC":
            continue
        isin_el = inv.find("n:identifiers/n:isin", ns)
        isin = isin_el.get("value", "").strip() if isin_el is not None else ""
        if not isin or isin == "N/A":
            continue
        holdings.append({
            "isin": isin,
            "security_name": _nport_text(inv, ns, "name"),
            "security_title": _nport_text(inv, ns, "title"),
            "cusip": _nport_text(inv, ns, "cusip"),
            "lei": _nport_text(inv, ns, "lei"),
            "balance": _nport_float(inv, ns, "balance"),
            "units": _nport_text(inv, ns, "units"),
            "currency": _nport_text(inv, ns, "curCd"),
            "market_value": _nport_float(inv, ns, "valUSD"),
            "weight": _nport_float(inv, ns, "pctVal"),
            "payoff_profile": _nport_text(inv, ns, "payoffProfile"),
            "asset_category": asset_category,
            "issuer_category": _nport_text(inv, ns, "issuerCat"),
            "investment_country": _nport_text(inv, ns, "invCountry"),
            "is_restricted": _nport_text(inv, ns, "isRestrictedSec"),
            "fair_value_level": _nport_text(inv, ns, "fairValLevel"),
        })
    return holdings


def _fetch_nport_holdings(acc: str, cik: str) -> list[dict]:
    acc_clean = acc.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/primary_doc.xml"
    xml_data = _edgar_fetch_bytes(url, timeout=60)
    return _parse_nport_ec_holdings(xml_data)


def _fetch_nport_isins(acc: str, cik: str, known_isins: set[str] | None = None) -> list[dict]:
    """
    Fetch one N-PORT-P filing and return ALL equity holdings as dicts.

    Universe-snapshots stores the true historical R1000 membership — we want
    every ISIN listed in the filing, regardless of whether it's currently in
    the `companies` table. Downstream consumers (Barra PIT filter, etc.)
    intersect with their own data at usage time.

    The `known_isins` parameter is accepted for backwards compatibility but
    is no longer used as a filter. Filtering it out at parse time would leak
    *scrape-time* universe state into the historical record and produced the
    under-counted snapshots (~749 vs ~1000) seen in pre-2024 dates.
    """
    return [
        {
            "isin": h["isin"],
            "weight": h["weight"],
            "market_value": h["market_value"],
        }
        for h in _fetch_nport_holdings(acc, cik)
    ]


def build_historical_snapshots(
    known_isins: set[str],
    registry: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """
    Fetch N-PORT-P filings from EDGAR for all indexes in the registry and
    return a DataFrame of universe_snapshots rows (one per company per snapshot date).

    registry: loaded from universe.db index_registry/nport_accessions tables.
              If None, load_index_registry() is called automatically.
    Deduplicates fetches: if multiple snapshot dates share the same accession for
    an index, the XML is fetched once and applied to all matching dates.
    All EC holdings reported in the N-PORT filing are included. Downstream
    consumers decide how to handle missing metadata/returns coverage.
    """
    _registry = registry if registry is not None else load_index_registry()
    rows = []
    for index_name, idx in _registry.items():
        cik      = idx["cik"]
        filings  = idx["filings"]  # {snapshot_date: (acc, period)}

        # Deduplicate: group snapshot dates by accession
        acc_to_dates: dict[str, list[str]] = {}
        for snap_date, (acc, _period) in sorted(filings.items()):
            acc_to_dates.setdefault(acc, []).append(snap_date)

        log.info("[%s]", index_name)
        for acc, snap_dates in sorted(acc_to_dates.items()):
            try:
                holdings = _fetch_nport_isins(acc, cik, known_isins)
            except Exception as e:
                log.warning("  %s: fetch error — %s", snap_dates[0], e)
                continue
            for snap_date in snap_dates:
                for h in holdings:
                    rows.append({
                        "snapshot_date": snap_date,
                        "isin":          h["isin"],
                        "index_name":    index_name,
                        "weight":        h["weight"],
                        "market_value":  h["market_value"],
                    })
                log.info("  %s: %d companies  (acc %s)", snap_date, len(holdings), acc)
            time.sleep(0.3)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# EDGAR metadata enrichment
# ---------------------------------------------------------------------------

_EDGAR_HEADERS = {"User-Agent": "universe-builder shivam3125@gmail.com"}

def _edgar_fetch(url: str, timeout: int = 10) -> dict:
    req = urllib.request.Request(url, headers=_EDGAR_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _edgar_fetch_bytes(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=_EDGAR_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def enrich_edgar_metadata(companies: pd.DataFrame) -> pd.DataFrame:
    """
    For companies missing a CIK, look up via EDGAR's company_tickers.json,
    then fetch submissions metadata (fiscalYearEnd, exchanges) for each one.
    Mutates and returns the companies DataFrame.
    """
    # Tickers where iShares uses a different format than EDGAR (no hyphen vs hyphen)
    _EDGAR_TICKER_ALIAS: dict[str, str] = {
        "BFA":  "BF-A",   # Brown-Forman Class A
        "BFB":  "BF-B",   # Brown-Forman Class B
        "BRKA": "BRK-A",  # Berkshire A
        "BRKB": "BRK-B",  # Berkshire B
    }

    # Download EDGAR's full ticker→CIK map (~13 k US public companies)
    try:
        tickers_data = _edgar_fetch("https://www.sec.gov/files/company_tickers.json", timeout=15)
    except Exception as e:
        log.warning("[EDGAR] Could not fetch company_tickers.json: %s", e)
        return companies

    # Build ticker → CIK dict (CIK as zero-padded 10-digit string)
    edgar_cik: dict[str, str] = {}
    for entry in tickers_data.values():
        tk  = str(entry.get("ticker", "")).upper().strip()
        cik = str(entry.get("cik_str", "")).zfill(10)
        if tk:
            edgar_cik[tk] = cik

    # Identify rows that are missing CIK
    missing_mask = companies["cik"].isna()
    missing = companies[missing_mask].copy()
    if missing.empty:
        return companies

    log.info("[EDGAR] Looking up CIKs for %d companies ...", len(missing))
    filled = 0

    for idx, row in missing.iterrows():
        ticker     = str(row["ticker"]).upper().strip()
        edgar_tick = _EDGAR_TICKER_ALIAS.get(ticker, ticker)
        cik        = edgar_cik.get(edgar_tick)
        if cik is None:
            continue

        companies.at[idx, "cik"] = cik
        filled += 1

        # Fetch richer metadata from submissions API
        try:
            sub = _edgar_fetch(
                f"https://data.sec.gov/submissions/CIK{cik}.json", timeout=8
            )
            # fiscal_year_end: submissions gives "1231" (MMDD) → extract month int
            fye_str = sub.get("fiscalYearEnd", "")
            if fye_str and len(fye_str) == 4 and fye_str.isdigit():
                companies.at[idx, "fiscal_year_end"] = int(fye_str[:2])
            # exchange (first one listed)
            exchanges = sub.get("exchanges", [])
            if exchanges and not companies.at[idx, "exchange"]:
                companies.at[idx, "exchange"] = exchanges[0]
            time.sleep(0.07)   # respect EDGAR rate limit
        except Exception:
            pass

    log.info("[EDGAR] Filled CIK for %d / %d missing companies", filled, len(missing))
    return companies


# ---------------------------------------------------------------------------
# N-PORT helpers
# ---------------------------------------------------------------------------

def _fetch_nport_all_ec_isins(acc: str, cik: str) -> set[str]:
    """Fetch one N-PORT-P filing and return the set of ISINs for all EC holdings."""
    return {h["isin"] for h in _fetch_nport_holdings(acc, cik)}


def _selected_nport_accessions(
    *,
    indexes: list[str] | None = None,
    only_latest: bool = False,
) -> list[tuple[str, str, str, str | None, str]]:
    """Return unique (index_name, accession, cik, period_ending, snapshot_date) rows."""
    registry = load_index_registry()
    rows: list[tuple[str, str, str, str | None, str]] = []
    selected = set(indexes or [])
    for index_name, idx in registry.items():
        if selected and index_name not in selected:
            continue
        cik = idx.get("cik")
        if not cik or not idx["filings"]:
            continue
        filings = idx["filings"]
        dates = [max(filings)] if only_latest else sorted(filings)
        for snapshot_date in dates:
            accession, period_ending = filings[snapshot_date]
            rows.append((index_name, accession, cik, period_ending, snapshot_date))

    # Same accession can back multiple snapshot dates. Fetch/store once.
    dedup: dict[tuple[str, str], tuple[str, str, str, str | None, str]] = {}
    for row in rows:
        _, accession, cik, _, _ = row
        dedup.setdefault((accession, cik), row)
    return sorted(dedup.values(), key=lambda r: (r[0], r[4], r[1]))


def refresh_nport_security_metadata(
    *,
    indexes: list[str] | None = None,
    only_latest: bool = False,
    force: bool = False,
) -> None:
    """Fetch N-PORT EC holding metadata and stage it in nport_security_metadata."""
    accessions = _selected_nport_accessions(indexes=indexes, only_latest=only_latest)
    if not accessions:
        raise RuntimeError("No N-PORT accessions selected. Check index_registry/nport_accessions.")

    now = datetime.now().isoformat(timespec="seconds")
    insert_sql = """
        INSERT OR REPLACE INTO nport_security_metadata (
            accession, filing_cik, isin, security_name, security_title, cusip,
            lei, balance, units, currency, market_value, weight, payoff_profile,
            asset_category, issuer_category, investment_country, is_restricted,
            fair_value_level, fetched_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with get_db(DB_PATH) as conn:
        seed_nport_security_metadata_table(conn)
        existing = {
            r[0] for r in conn.execute(
                "SELECT DISTINCT accession FROM nport_security_metadata"
            ).fetchall()
        }

    written = skipped = failures = 0
    log.info(
        "Refreshing N-PORT security metadata: accessions=%d indexes=%s latest_only=%s force=%s",
        len(accessions), ",".join(indexes or ["all"]), only_latest, force,
    )
    for index_name, accession, cik, period_ending, snapshot_date in accessions:
        if accession in existing and not force:
            skipped += 1
            continue
        try:
            holdings = _fetch_nport_holdings(accession, cik)
        except Exception as exc:
            failures += 1
            log.warning("[%s %s %s] N-PORT metadata fetch failed: %s",
                        index_name, snapshot_date, accession, exc)
            continue
        rows = [
            (
                accession, cik, h["isin"], h["security_name"], h["security_title"],
                h["cusip"], h["lei"], h["balance"], h["units"], h["currency"],
                h["market_value"], h["weight"], h["payoff_profile"],
                h["asset_category"], h["issuer_category"], h["investment_country"],
                h["is_restricted"], h["fair_value_level"], now,
            )
            for h in holdings
        ]
        with get_db(DB_PATH) as conn:
            seed_nport_security_metadata_table(conn)
            conn.executemany(insert_sql, rows)
            conn.commit()
        written += len(rows)
        log.info("[%s %s] %s: staged %d EC holdings", index_name, snapshot_date, accession, len(rows))
        time.sleep(0.3)

    log.info(
        "N-PORT security metadata refresh done: rows_written=%s skipped_accessions=%d failures=%d",
        f"{written:,}", skipped, failures,
    )


def _first_clean(values: pd.Series) -> str | None:
    for value in values:
        if pd.isna(value):
            continue
        cleaned = str(value).strip()
        if cleaned and cleaned.upper() != "N/A":
            return cleaned
    return None


def stage_nport_company_candidates(
    *,
    indexes: list[str] | None = None,
    only_latest: bool = False,
) -> None:
    """Derive reviewable security-master candidates from staged N-PORT metadata.

    This deliberately does not mutate companies. It gives us a durable audit
    layer for missing or weak security-master coverage before we resolve
    tickers/CIKs or apply any production backfill.
    """
    idx_filter = ""
    params: list[object] = []
    if indexes:
        idx_filter = f"AND us.index_name IN ({','.join('?' * len(indexes))})"
        params.extend(indexes)

    latest_filter = ""
    if only_latest:
        latest_filter = """
          AND us.snapshot_date = (
              SELECT MAX(us2.snapshot_date)
              FROM universe_snapshots us2
              WHERE us2.index_name = us.index_name
          )
        """

    sql = f"""
        SELECT us.index_name, us.snapshot_date, us.isin, us.weight, us.market_value,
               na.accession,
               nm.security_name, nm.security_title, nm.cusip, nm.lei,
               nm.currency, nm.investment_country,
               c.ticker AS existing_ticker,
               c.company_name AS existing_company_name,
               c.gics_sector AS existing_gics_sector,
               c.gics_industry AS existing_gics_industry
        FROM universe_snapshots us
        JOIN nport_accessions na
          ON na.index_name = us.index_name
         AND na.snapshot_date = us.snapshot_date
        JOIN nport_security_metadata nm
          ON nm.accession = na.accession
         AND nm.isin = us.isin
        LEFT JOIN companies c ON c.isin = us.isin
        WHERE 1=1 {idx_filter} {latest_filter}
    """

    with get_db(DB_PATH) as conn:
        seed_nport_security_metadata_table(conn)
        seed_nport_company_candidates_table(conn)
        existing_resolutions = pd.read_sql_query(
            """
            SELECT isin, resolved_ticker, resolved_cik, resolved_company_name,
                   resolved_exchange, resolution_status, resolution_confidence,
                   resolver_sources, resolution_evidence, resolved_at
            FROM nport_company_candidates
            """,
            conn,
        )
        raw = pd.read_sql_query(sql, conn, params=tuple(params))

    if raw.empty:
        log.warning(
            "No staged N-PORT company candidates found. Run --refresh-nport-metadata first."
        )
        return

    raw["weight"] = pd.to_numeric(raw["weight"], errors="coerce").fillna(0.0)
    raw = raw.sort_values(
        ["isin", "snapshot_date", "weight"],
        ascending=[True, False, False],
    )

    preferred = (
        raw.groupby("isin", as_index=False)
        .agg(
            security_name=("security_name", _first_clean),
            security_title=("security_title", _first_clean),
            cusip=("cusip", _first_clean),
            lei=("lei", _first_clean),
            currency=("currency", _first_clean),
            investment_country=("investment_country", _first_clean),
            existing_ticker=("existing_ticker", _first_clean),
            existing_company_name=("existing_company_name", _first_clean),
            existing_gics_sector=("existing_gics_sector", _first_clean),
            existing_gics_industry=("existing_gics_industry", _first_clean),
        )
    )
    rollup = (
        raw.groupby("isin", as_index=False)
        .agg(
            first_snapshot_date=("snapshot_date", "min"),
            last_snapshot_date=("snapshot_date", "max"),
            max_weight=("weight", "max"),
            seen_indexes=("index_name", lambda s: "|".join(sorted({str(v) for v in s if pd.notna(v)}))),
            accessions=("accession", lambda s: "|".join(sorted({str(v) for v in s if pd.notna(v)}))),
        )
    )
    candidates = preferred.merge(rollup, on="isin", how="left")
    candidates["company_status"] = candidates["existing_company_name"].notna().map(
        {True: "exists_in_companies", False: "missing_from_companies"}
    )
    candidates["candidate_source"] = "nport_security_metadata"
    candidates["staged_at"] = datetime.now().isoformat(timespec="seconds")
    if not existing_resolutions.empty:
        candidates = candidates.merge(existing_resolutions, on="isin", how="left")
    else:
        for col in [
            "resolved_ticker", "resolved_cik", "resolved_company_name",
            "resolved_exchange", "resolution_status", "resolution_confidence",
            "resolver_sources", "resolution_evidence", "resolved_at",
        ]:
            candidates[col] = None
    candidates = candidates[
        [
            "isin", "security_name", "security_title", "cusip", "lei",
            "currency", "investment_country", "first_snapshot_date",
            "last_snapshot_date", "max_weight", "seen_indexes", "accessions",
            "company_status", "existing_ticker", "existing_company_name",
            "existing_gics_sector", "existing_gics_industry",
            "resolved_ticker", "resolved_cik", "resolved_company_name",
            "resolved_exchange", "resolution_status", "resolution_confidence",
            "resolver_sources", "resolution_evidence", "resolved_at",
            "candidate_source", "staged_at",
        ]
    ]

    with get_db(DB_PATH) as conn:
        seed_nport_company_candidates_table(conn)
        conn.execute("DELETE FROM nport_company_candidates")
        candidates.to_sql("nport_company_candidates", conn, if_exists="append", index=False)
        conn.commit()

    missing = int(candidates["company_status"].eq("missing_from_companies").sum())
    log.info(
        "Staged N-PORT company candidates: rows=%s missing_from_companies=%d latest_only=%s indexes=%s",
        f"{len(candidates):,}",
        missing,
        only_latest,
        ",".join(indexes or ["all"]),
    )


def _resolution_name_tokens(value: object) -> set[str]:
    raw = str(value or "").upper()
    replacements = {
        "&": " AND ",
        ".": " ",
        ",": " ",
        "'": " ",
        "\"": " ",
        "/": " ",
        "-": " ",
        "(": " ",
        ")": " ",
    }
    for old, new in replacements.items():
        raw = raw.replace(old, new)
    stop = {
        "THE", "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY",
        "PLC", "LTD", "LIMITED", "NV", "N", "V", "SA", "AG", "LP", "LLC",
        "CLASS", "COM", "COMMON", "STOCK", "NEW", "GROUP", "HOLDINGS",
    }
    return {t for t in raw.split() if t and t not in stop and len(t) > 1}


def _resolution_name_similarity(left: object, right: object) -> float:
    l_tokens = _resolution_name_tokens(left)
    r_tokens = _resolution_name_tokens(right)
    if not l_tokens or not r_tokens:
        return 0.0
    return len(l_tokens & r_tokens) / len(l_tokens | r_tokens)


def _load_sec_company_tickers() -> dict[str, dict[str, str]]:
    """Load SEC ticker->CIK/title reference from company_tickers.json."""
    data = _edgar_fetch("https://www.sec.gov/files/company_tickers.json", timeout=20)
    out: dict[str, dict[str, str]] = {}
    for entry in data.values():
        ticker = str(entry.get("ticker", "")).upper().strip()
        if not ticker:
            continue
        cik_raw = entry.get("cik_str")
        cik = str(cik_raw).zfill(10) if cik_raw is not None else ""
        out[ticker] = {
            "cik": cik,
            "title": str(entry.get("title", "")).strip(),
        }
    return out


def _sec_ticker_variants(ticker: str | None) -> list[str]:
    if not ticker:
        return []
    raw = str(ticker).upper().strip()
    variants = [raw]
    compact = "".join(ch for ch in raw if ch.isalnum())
    if compact and compact not in variants:
        variants.append(compact)
    trimmed = raw.rstrip("/")
    if trimmed and trimmed not in variants:
        variants.append(trimmed)
    no_trailing_digits = compact.rstrip("0123456789")
    if no_trailing_digits and no_trailing_digits not in variants:
        variants.append(no_trailing_digits)
    return variants


def _match_sec_ticker(
    *,
    openfigi_ticker: str | None,
    security_name: object,
    security_title: object,
    sec_by_ticker: dict[str, dict[str, str]],
) -> tuple[str | None, dict[str, str], float, str]:
    names = [security_name, security_title]
    for variant in _sec_ticker_variants(openfigi_ticker):
        sec = sec_by_ticker.get(variant)
        if sec:
            score = max(_resolution_name_similarity(name, sec.get("title")) for name in names)
            return variant, sec, score, "sec_ticker_variant"

    best_ticker = None
    best_sec: dict[str, str] = {}
    best_score = 0.0
    for ticker, sec in sec_by_ticker.items():
        score = max(_resolution_name_similarity(name, sec.get("title")) for name in names)
        if score > best_score:
            best_ticker = ticker
            best_sec = sec
            best_score = score

    if best_ticker and best_score >= 0.75:
        return best_ticker, best_sec, best_score, "sec_name_match"
    return None, {}, 0.0, "no_sec_match"


def _fetch_sec_submission_summary(cik: str) -> dict[str, object]:
    if not cik:
        return {}
    try:
        sub = _edgar_fetch(f"https://data.sec.gov/submissions/CIK{cik}.json", timeout=15)
    except Exception:
        return {}
    return {
        "name": sub.get("name") or "",
        "tickers": sub.get("tickers") or [],
        "exchanges": sub.get("exchanges") or [],
    }


def resolve_nport_company_candidates(limit: int | None = None) -> None:
    """Resolve missing N-PORT company candidates to ticker/CIK for review.

    Updates only resolution columns on nport_company_candidates. It does not
    insert into companies and does not create manual mappings.
    """
    figi_key = _load_openfigi_api_key()
    with get_db(DB_PATH) as conn:
        seed_nport_company_candidates_table(conn)
        sql = """
            SELECT isin, security_name, security_title, cusip, lei
            FROM nport_company_candidates
            WHERE company_status = 'missing_from_companies'
            ORDER BY max_weight DESC, isin
        """
        if limit is not None:
            sql += " LIMIT ?"
            candidates = pd.read_sql_query(sql, conn, params=(limit,))
        else:
            candidates = pd.read_sql_query(sql, conn)

    if candidates.empty:
        log.info("No missing N-PORT company candidates to resolve.")
        return

    isins = candidates["isin"].astype(str).tolist()
    log.info("Resolving %d N-PORT candidate ISINs via OpenFIGI ...", len(isins))
    isin_to_ticker = _openfigi_map_isins(isins, figi_key=figi_key)
    log.info("  OpenFIGI resolved %d / %d tickers", len(isin_to_ticker), len(isins))

    try:
        sec_by_ticker = _load_sec_company_tickers()
    except Exception as exc:
        log.warning("SEC company_tickers.json fetch failed: %s", exc)
        sec_by_ticker = {}

    rows: list[tuple] = []
    now = datetime.now().isoformat(timespec="seconds")
    for candidate in candidates.to_dict("records"):
        isin = str(candidate["isin"])
        openfigi_ticker = isin_to_ticker.get(isin)
        sec_ticker, sec, sec_name_score, sec_match_rule = _match_sec_ticker(
            openfigi_ticker=openfigi_ticker,
            security_name=candidate.get("security_name"),
            security_title=candidate.get("security_title"),
            sec_by_ticker=sec_by_ticker,
        )
        ticker = sec_ticker or openfigi_ticker
        cik = sec.get("cik", "")
        sec_title = sec.get("title", "")
        submission = _fetch_sec_submission_summary(cik) if cik else {}
        submission_name = str(submission.get("name") or "")
        exchanges = submission.get("exchanges") or []
        exchange = str(exchanges[0]) if exchanges else ""
        resolved_name = submission_name or sec_title
        name_similarity = max(
            _resolution_name_similarity(candidate.get("security_name"), resolved_name),
            _resolution_name_similarity(candidate.get("security_title"), resolved_name),
        )

        sources: list[str] = []
        evidence_parts = [
            f"nport_name={candidate.get('security_name') or ''}",
            f"nport_title={candidate.get('security_title') or ''}",
            f"nport_cusip={candidate.get('cusip') or ''}",
            f"nport_lei={candidate.get('lei') or ''}",
        ]
        if openfigi_ticker:
            sources.append("openfigi_isin")
            evidence_parts.append(f"openfigi_ticker={openfigi_ticker}")
        if cik:
            sources.append("sec_company_tickers")
            evidence_parts.append(f"sec_match_rule={sec_match_rule}")
            evidence_parts.append(f"sec_name_score={sec_name_score:.3f}")
            evidence_parts.append(f"sec_ticker={sec_ticker or ''}")
            evidence_parts.append(f"sec_cik={cik}")
            evidence_parts.append(f"sec_title={sec_title}")
        if submission:
            sources.append("sec_submissions")
            evidence_parts.append(f"sec_submission_name={submission_name}")
            evidence_parts.append(f"sec_exchanges={'|'.join(str(x) for x in exchanges)}")
        evidence_parts.append(f"name_similarity={name_similarity:.3f}")

        if not ticker:
            status = "unresolved_no_ticker"
            confidence = 0.0
        elif not cik:
            status = "ticker_resolved_no_sec_cik"
            confidence = 0.55
        elif sec_match_rule == "sec_name_match" and name_similarity >= 0.75:
            status = "resolved_sec_name_match"
            confidence = 0.85
        elif name_similarity >= 0.5:
            status = "resolved"
            confidence = 0.9
        elif name_similarity > 0:
            status = "needs_review_name_weak"
            confidence = 0.7
        else:
            status = "needs_review_name_mismatch"
            confidence = 0.45

        rows.append((
            ticker,
            cik or None,
            resolved_name or None,
            exchange or None,
            status,
            confidence,
            "|".join(sources) if sources else "none",
            "; ".join(evidence_parts),
            now,
            isin,
        ))

    update_sql = """
        UPDATE nport_company_candidates
        SET resolved_ticker = ?,
            resolved_cik = ?,
            resolved_company_name = ?,
            resolved_exchange = ?,
            resolution_status = ?,
            resolution_confidence = ?,
            resolver_sources = ?,
            resolution_evidence = ?,
            resolved_at = ?
        WHERE isin = ?
    """
    with get_db(DB_PATH) as conn:
        seed_nport_company_candidates_table(conn)
        conn.executemany(update_sql, rows)
        conn.commit()

    status_counts = pd.Series([r[4] for r in rows]).value_counts().to_dict()
    log.info("N-PORT candidate resolution staged: %s", status_counts)


def promote_nport_company_candidates(min_confidence: float = 0.85) -> None:
    """Insert reviewed N-PORT candidate resolutions into companies.

    Promotion is intentionally conservative:
      - only missing candidates with a resolved ticker and CIK are eligible;
      - the resolved ticker must already exist in companies, so sector/classification
        metadata is copied from a known row instead of guessed;
      - unresolved or ticker-only rows remain in nport_company_candidates.
    """
    eligible_statuses = ("resolved", "resolved_sec_name_match")
    with get_db(DB_PATH) as conn:
        seed_nport_company_candidates_table(conn)
        candidates = pd.read_sql_query(
            f"""
            SELECT isin, security_name, security_title, cusip, currency,
                   investment_country, last_snapshot_date, resolved_ticker,
                   resolved_cik, resolved_company_name, resolved_exchange,
                   resolution_status, resolution_confidence
            FROM nport_company_candidates
            WHERE company_status = 'missing_from_companies'
              AND resolution_status IN ({','.join('?' * len(eligible_statuses))})
              AND resolution_confidence >= ?
              AND resolved_ticker IS NOT NULL AND resolved_ticker != ''
              AND resolved_cik IS NOT NULL AND resolved_cik != ''
            ORDER BY max_weight DESC, isin
            """,
            conn,
            params=(*eligible_statuses, min_confidence),
        )

        if candidates.empty:
            log.info("No reviewed N-PORT candidates eligible for promotion.")
            return

        tickers = candidates["resolved_ticker"].dropna().astype(str).unique().tolist()
        ph = ",".join("?" * len(tickers))
        existing = pd.read_sql_query(
            f"""
            SELECT *
            FROM companies
            WHERE ticker IN ({ph})
            ORDER BY ticker, COALESCE(update_date, '') DESC, COALESCE(data_date, '') DESC
            """,
            conn,
            params=tuple(tickers),
        )

    if existing.empty:
        log.warning("No existing same-ticker companies rows found; nothing promoted.")
        return

    existing_by_ticker = (
        existing.drop_duplicates(subset=["ticker"], keep="first")
        .set_index("ticker")
        .to_dict("index")
    )
    company_cols = [
        "isin", "ticker", "company_name", "gics_sector", "gics_industry_group",
        "gics_industry", "gics_sub_industry", "country", "exchange", "currency",
        "fiscal_year_end", "num_employees", "business_summary", "cik", "cusip",
        "simfin_id", "simfin_sector", "simfin_industry", "data_date", "update_date",
        "delisted_date",
    ]
    today = datetime.now().strftime("%Y-%m-%d")
    rows: list[tuple] = []
    skipped: list[tuple[str, str, str]] = []
    for candidate in candidates.to_dict("records"):
        ticker = str(candidate["resolved_ticker"])
        base = existing_by_ticker.get(ticker)
        if not base:
            skipped.append((candidate["isin"], ticker, "no_existing_ticker_metadata"))
            continue
        row = {col: base.get(col) for col in company_cols}
        row["isin"] = candidate["isin"]
        row["ticker"] = ticker
        row["company_name"] = (
            candidate.get("resolved_company_name")
            or candidate.get("security_name")
            or base.get("company_name")
        )
        row["cik"] = candidate.get("resolved_cik") or base.get("cik")
        row["cusip"] = candidate.get("cusip") or base.get("cusip")
        row["exchange"] = base.get("exchange") or candidate.get("resolved_exchange")
        row["currency"] = base.get("currency") or candidate.get("currency")
        row["data_date"] = candidate.get("last_snapshot_date") or today
        row["update_date"] = today
        row["delisted_date"] = None
        rows.append(tuple(row.get(col) for col in company_cols))

    if not rows:
        log.warning("No N-PORT candidates promoted. Skipped=%s", skipped)
        return

    insert_sql = (
        f"INSERT OR REPLACE INTO companies ({','.join(company_cols)}) "
        f"VALUES ({','.join('?' * len(company_cols))})"
    )
    with get_db(DB_PATH) as conn:
        conn.executemany(insert_sql, rows)
        conn.commit()

    log.info("Promoted %d N-PORT candidate(s) into companies.", len(rows))
    for isin, ticker, reason in skipped:
        log.warning("Skipped %s/%s: %s", isin, ticker, reason)


def _normalise_company_name(value: object) -> str:
    text = str(value or "").upper()
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def repair_company_identifier_continuity() -> None:
    """Carry known same-issuer metadata across current ISIN changes.

    The rule is intentionally narrow: same ticker + same normalised company name,
    one unambiguous source row with a legacy identifier bridge, and a current row
    missing that bridge. This keeps redomiciles/share-class ISIN changes from
    losing factor, risk, and market-cap coverage without guessing across issuers.
    """
    continuity_cols = [
        "gics_sector", "gics_industry_group", "gics_industry", "gics_sub_industry",
        "country", "exchange", "currency", "fiscal_year_end", "num_employees",
        "business_summary", "cik", "simfin_id", "simfin_sector", "simfin_industry",
    ]

    with get_db(DB_PATH) as conn:
        companies = pd.read_sql_query(
            """
            SELECT isin, ticker, company_name, delisted_date, update_date, data_date,
                   gics_sector, gics_industry_group, gics_industry, gics_sub_industry,
                   country, exchange, currency, fiscal_year_end, num_employees,
                   business_summary, cik, simfin_id, simfin_sector, simfin_industry
            FROM companies
            WHERE ticker IS NOT NULL AND ticker != ''
              AND company_name IS NOT NULL AND company_name != ''
            """,
            conn,
        )

        if companies.empty:
            log.info("No companies available for identifier-continuity repair.")
            return

        companies["_name_key"] = companies["company_name"].map(_normalise_company_name)
        updates: list[tuple] = []
        repaired: list[tuple[str, str, str]] = []

        for (_ticker, _name_key), group in companies.groupby(["ticker", "_name_key"], dropna=False):
            source = group[group["simfin_id"].notna()].copy()
            if source["simfin_id"].dropna().nunique() != 1:
                continue
            source = source.sort_values(
                by=["update_date", "data_date", "isin"],
                ascending=[False, False, True],
                na_position="last",
            ).iloc[0]

            targets = group[
                group["simfin_id"].isna()
                & (group["delisted_date"].isna() | (group["delisted_date"].astype(str) == ""))
            ]
            for target in targets.to_dict("records"):
                values = []
                changed = False
                for col in continuity_cols:
                    current = target.get(col)
                    replacement = source.get(col)
                    is_missing = pd.isna(current) or current == ""
                    if is_missing and not pd.isna(replacement) and replacement != "":
                        values.append(replacement)
                        changed = True
                    else:
                        values.append(current)
                if changed:
                    updates.append((*values, target["isin"]))
                    repaired.append((target["ticker"], target["isin"], source["isin"]))

        if not updates:
            log.info("No identifier-continuity repairs needed.")
            return

        set_clause = ", ".join(f"{col}=?" for col in continuity_cols)
        conn.executemany(
            f"UPDATE companies SET {set_clause} WHERE isin=?",
            updates,
        )
        conn.commit()

    log.info("Repaired identifier continuity for %d company row(s).", len(repaired))
    for ticker, target_isin, source_isin in repaired[:25]:
        log.info("  %-8s %s <- %s", ticker, target_isin, source_isin)
    if len(repaired) > 25:
        log.info("  ... %d more", len(repaired) - 25)


def consolidate_same_issuer_fundamentals() -> None:
    """Migrate stranded fundamentals onto a same-issuer current ISIN.

    Targets exactly the breakage where a redomicile / ISIN-change leaves the
    current snapshot ISIN with no ISIN-keyed fundamentals while a legacy ISIN
    for the *same issuer* still carries them (e.g. Amcor, Pinnacle Financial,
    both promoted from N-PORT under a new ISIN). The constituents fundamentals
    (incl. shares outstanding) are re-keyed onto the current ISIN so factors and
    market cap resolve.

    The rule is deliberately narrow to avoid perturbing working names:
      * same ticker + same normalised company name,
      * canonical = the single current-snapshot member of the group,
      * canonical and legacy share one non-null simfin_id (same-issuer guard),
      * canonical has ZERO ISIN-keyed constituents rows (so re-keying cannot
        create a primary-key conflict and cannot overwrite good data),
      * legacy ISIN(s) are not current-snapshot members and do carry rows.

    True dual-class pairs (distinct tickers like BFA/BFB, or two ISINs both
    present in the current snapshot like Clearway A/C) never match and are left
    untouched.
    """
    with get_db(DB_PATH) as conn:
        companies = pd.read_sql_query(
            """
            SELECT isin, ticker, company_name, cik, simfin_id, delisted_date
            FROM companies
            WHERE ticker IS NOT NULL AND ticker != ''
              AND company_name IS NOT NULL AND company_name != ''
            """,
            conn,
        )
        snapshot_isins = {
            r[0] for r in conn.execute(
                """
                SELECT us.isin FROM universe_snapshots us
                JOIN (SELECT index_name, MAX(snapshot_date) AS d
                      FROM universe_snapshots GROUP BY index_name) latest
                  ON latest.index_name = us.index_name AND latest.d = us.snapshot_date
                """
            ).fetchall()
        }

    if companies.empty:
        log.info("No companies available for fundamentals consolidation.")
        return

    companies["_name_key"] = companies["company_name"].map(_normalise_company_name)

    def _isin_keyed_count(cur, isin: str) -> int:
        return cur.execute(
            "SELECT COUNT(*) FROM constituents WHERE security_id = ?", (isin,)
        ).fetchone()[0]

    migrations: list[tuple[str, str, str]] = []  # (ticker, legacy_isin, canon_isin)
    simfin_clear: list[str] = []                 # legacy isins to null simfin_id on

    with get_db(CONSTITUENTS_DB) as cconn:
        cur = cconn.cursor()
        for (ticker, _name_key), group in companies.groupby(["ticker", "_name_key"], dropna=False):
            isins = list(group["isin"])
            if len(isins) < 2:
                continue
            simfins = group["simfin_id"].dropna().unique()
            if len(simfins) != 1:
                continue  # need a single shared issuer id
            canon = [i for i in isins if i in snapshot_isins]
            if len(canon) != 1:
                continue  # not a clean one-current-member group
            canon_isin = canon[0]
            if _isin_keyed_count(cur, canon_isin) != 0:
                continue  # canonical already has fundamentals — leave it alone
            # canonical must itself carry the shared simfin_id (same issuer)
            canon_row = group[group["isin"] == canon_isin].iloc[0]
            if pd.isna(canon_row["simfin_id"]):
                continue
            for legacy_isin in [i for i in isins if i != canon_isin and i not in snapshot_isins]:
                legacy_row = group[group["isin"] == legacy_isin].iloc[0]
                if legacy_row["simfin_id"] != canon_row["simfin_id"]:
                    continue
                if _isin_keyed_count(cur, legacy_isin) == 0:
                    continue
                cur.execute(
                    "UPDATE OR IGNORE constituents SET security_id = ? WHERE security_id = ?",
                    (canon_isin, legacy_isin),
                )
                migrations.append((ticker, legacy_isin, canon_isin))
                simfin_clear.append(legacy_isin)
        cconn.commit()

    if not migrations:
        log.info("No same-issuer fundamentals to consolidate.")
        return

    # Resolve the shared-simfin_id collision so create_factors maps the SimFin
    # numeric id deterministically to the canonical ISIN.
    with get_db(DB_PATH) as conn:
        conn.executemany(
            "UPDATE companies SET simfin_id = NULL WHERE isin = ?",
            [(i,) for i in simfin_clear],
        )
        conn.commit()

    log.info("Consolidated same-issuer fundamentals for %d ISIN(s):", len(migrations))
    for ticker, legacy_isin, canon_isin in migrations:
        log.info("  %-8s %s -> %s", ticker, legacy_isin, canon_isin)
    log.warning(
        "Fundamentals moved to the canonical ISIN. Restate the snapshot date(s) where "
        "the canonical ISIN is a member (typically only the latest), e.g. "
        "create_factors --date <D> -> create_models --date <D> -> create_risk --date <D> "
        "-> create_barra --date <D>. Do NOT restate all historical dates: those snapshots "
        "still reference the legacy ISIN, whose factors were already computed and would be "
        "dropped. (If full historical consolidation is wanted, the legacy snapshot ISINs "
        "must first be normalised to the canonical ISIN.)"
    )


# ---------------------------------------------------------------------------
# FMP ISIN refresh  (--refresh-isins)
# ---------------------------------------------------------------------------

_FMP_BASE = "https://financialmodelingprep.com/stable"

# Tickers whose iShares format differs from FMP's expected symbol format.
_FMP_TICKER_ALIAS: dict[str, str] = {
    "BFA":  "BF-A",
    "BFB":  "BF-B",
    "BRKA": "BRK-A",
    "BRKB": "BRK-B",
}


def _load_fmp_api_key() -> str | None:
    """Read FMP_API_KEY from .env file in the current working directory."""
    env = Path(".env")
    if not env.exists():
        return None
    for line in env.read_text().splitlines():
        line = line.strip()
        if line.startswith("FMP_API_KEY="):
            return line.split("=", 1)[1].strip()
    return None


def _fmp_fetch_isin(ticker: str, api_key: str) -> str | None:
    """Fetch ISIN for one ticker from FMP /stable/profile. Returns None if not found.

    Raises urllib.error.HTTPError with code 429 if the account is rate-limited
    so callers can stop early and report the issue.
    """
    fmp_ticker = _FMP_TICKER_ALIAS.get(ticker, ticker)
    url = f"{_FMP_BASE}/profile?symbol={fmp_ticker}&apikey={api_key}"
    try:
        data = _edgar_fetch(url, timeout=10)
        if isinstance(data, list) and data:
            isin = data[0].get("isin", "")
            return isin if isin else None
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise   # propagate rate-limit so callers can stop early
    except Exception:
        pass
    return None


def refresh_isins() -> None:
    """
    Fetch ISINs from FMP for all tickers found in universe_index CSV files,
    and write them into the isin_patch table in universe.db.

    isin_patch overrides SimFin ISINs in build_companies() — this is the
    authoritative source for correct ISINs. Run once after adding new index
    files or when ISINs need refreshing.

    Requires FMP_API_KEY in .env (project root).
    """
    api_key = _load_fmp_api_key()
    if not api_key:
        raise RuntimeError(
            "FMP_API_KEY not found in .env. "
            "Add FMP_API_KEY=<your_key> to the .env file in the project root."
        )

    index_files = sorted(INDEX_DIR.glob("*.csv")) if INDEX_DIR.exists() else []
    if not index_files:
        raise RuntimeError(f"No CSV files found in {INDEX_DIR}")

    tickers: set[str] = set()
    for path in index_files:
        eq, _, _ = load_ishares(path)
        tickers.update(eq["Ticker"].dropna().unique())

    ticker_list = sorted(tickers)

    # Skip tickers already resolved in isin_patch to preserve the daily request quota.
    with get_db(DB_PATH) as conn:
        seed_isin_patch_table(conn)
        existing = {r[0] for r in conn.execute("SELECT ticker FROM isin_patch").fetchall()}

    pending = [t for t in ticker_list if t not in existing]
    log.info("Tickers: %d total, %d already in isin_patch, %d to fetch from FMP ...",
             len(ticker_list), len(existing), len(pending))

    if not pending:
        log.info("Nothing to fetch — all tickers already resolved.")
        return

    with get_db(DB_PATH) as conn:
        hits, misses = 0, []
        rate_limited = False
        for i, ticker in enumerate(pending):
            if (i + 1) % 100 == 0:
                log.info("  %d/%d ...", i + 1, len(pending))
            try:
                isin = _fmp_fetch_isin(ticker, api_key)
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    log.error("[ERROR] FMP rate limit hit after %d requests (HTTP 429). "
                              "Create a new free FMP account, update FMP_API_KEY in .env, and re-run.", i)
                    rate_limited = True
                    break
                raise
            if isin:
                conn.execute(
                    "INSERT OR REPLACE INTO isin_patch (ticker, isin, note) VALUES (?, ?, ?)",
                    (ticker, isin, "FMP /stable/profile"),
                )
                hits += 1
            else:
                misses.append(ticker)
            time.sleep(0.05)   # ~20 req/s — stay well under FMP rate limit
        conn.commit()

    label = "rate-limited — re-run with new FMP key" if rate_limited else "not found by FMP"
    log.info("%d new ISINs written to isin_patch, %d %s", hits, len(misses), label)
    if misses and not rate_limited:
        log.info("Not found: %s%s", ", ".join(misses[:20]), " ..." if len(misses) > 20 else "")


# ---------------------------------------------------------------------------
# Fix ISINs via N-PORT + FMP validation  (--fix-isins)
# ---------------------------------------------------------------------------

def fix_isins() -> None:
    """
    Find companies with stale/wrong ISINs by comparing the companies table against
    the latest N-PORT filing, then query FMP to get the authoritative ISIN and
    write it to isin_patch.

    Phase 1 (no FMP needed): identify suspect tickers whose effective ISIN is absent
    from N-PORT's EC holdings list.

    Phase 2 (FMP needed): query FMP /stable/profile for each suspect ticker;
    accept the result only if FMP's ISIN appears in N-PORT (cross-validated).

    Requires FMP_API_KEY in .env for Phase 2. Without it, Phase 1 prints the suspect
    tickers so you can act on them manually or add a key and re-run.
    """
    # ---------- Phase 1: identify suspects via N-PORT comparison ----------
    log.info("Loading companies table and isin_patch ...")
    with get_db(DB_PATH) as conn:
        seed_all_reference_tables(conn)
        companies_rows = conn.execute("SELECT ticker, isin FROM companies").fetchall()
        patch_rows     = conn.execute("SELECT ticker, isin FROM isin_patch").fetchall()

    ticker_to_isin: dict[str, str]      = {r[0]: r[1] for r in companies_rows}
    patched_by_ticker: dict[str, str]   = {r[0]: r[1] for r in patch_rows}
    log.info("  %d companies, %d already in isin_patch", len(companies_rows), len(patched_by_ticker))

    # Fetch N-PORT: latest accession per index (deduplicated)
    registry = load_index_registry()
    seen_acc: set[str] = set()
    latest_acc_cik: list[tuple[str, str]] = []
    for idx in registry.values():
        cik = idx["cik"]
        if idx["filings"]:
            latest_date = max(idx["filings"].keys())
            acc, _ = idx["filings"][latest_date]
            if acc not in seen_acc:
                seen_acc.add(acc)
                latest_acc_cik.append((acc, cik))

    if not latest_acc_cik:
        raise RuntimeError("No N-PORT accessions found in nport_accessions table.")

    log.info("Fetching %d latest N-PORT filing(s) ...", len(latest_acc_cik))
    nport_isins: set[str] = set()
    for acc, cik in latest_acc_cik:
        try:
            isins = _fetch_nport_all_ec_isins(acc, cik)
            nport_isins.update(isins)
            log.info("  %s: %d EC holdings", acc, len(isins))
        except Exception as e:
            log.warning("  %s: fetch error — %s", acc, e)
        time.sleep(0.3)

    log.info("N-PORT: %d unique EC ISINs in latest filing(s)", len(nport_isins))

    # Suspects: companies whose effective ISIN (patch wins over companies table) is not in N-PORT.
    # Exclude synthetic placeholders — those are handled by --refresh-isins.
    suspect_tickers: list[str] = []
    for ticker, companies_isin in ticker_to_isin.items():
        if companies_isin.startswith("NOISN_"):
            continue
        effective_isin = patched_by_ticker.get(ticker, companies_isin)
        if effective_isin not in nport_isins:
            suspect_tickers.append(ticker)

    log.info("Suspect tickers (effective ISIN not in N-PORT): %d", len(suspect_tickers))
    if not suspect_tickers:
        log.info("No suspects — all company ISINs match N-PORT.")
        return

    for t in sorted(suspect_tickers):
        eff = patched_by_ticker.get(t, ticker_to_isin[t])
        log.info("  %-10s %s", t, eff)

    # ---------- Phase 2: fix via FMP, validated against N-PORT ----------
    api_key = _load_fmp_api_key()
    if not api_key:
        log.warning("No FMP_API_KEY in .env — cannot auto-fix. "
                    "Add FMP_API_KEY=<key> to .env and re-run --fix-isins.")
        return

    log.info("Querying FMP for %d suspect ticker(s) ...", len(suspect_tickers))
    resolved: dict[str, str] = {}   # ticker → FMP ISIN (N-PORT validated)
    unresolved: list[str]    = []
    rate_limited             = False

    for i, ticker in enumerate(sorted(suspect_tickers)):
        if (i + 1) % 5 == 0 or i == 0:
            log.info("  %d/%d ...", i + 1, len(suspect_tickers))
        try:
            fmp_isin = _fmp_fetch_isin(ticker, api_key)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                log.error("[ERROR] FMP rate limit hit after %d requests (HTTP 429). "
                          "Create a new free FMP account, update FMP_API_KEY in .env, and re-run.", i)
                unresolved.extend(sorted(suspect_tickers)[i:])
                rate_limited = True
                break
            raise
        if fmp_isin and fmp_isin in nport_isins:
            resolved[ticker] = fmp_isin
        else:
            unresolved.append(ticker)
        time.sleep(0.05)

    if resolved:
        isin_by_ticker = {r[0]: r[1] for r in companies_rows}
        log.info("Writing %d corrected ISIN(s) to isin_patch ...", len(resolved))
        with get_db(DB_PATH) as conn:
            for ticker, isin in sorted(resolved.items()):
                conn.execute(
                    "INSERT OR REPLACE INTO isin_patch (ticker, isin, note) VALUES (?, ?, ?)",
                    (ticker, isin, "FMP N-PORT validated"),
                )
            conn.commit()
        for ticker, isin in sorted(resolved.items()):
            old = isin_by_ticker.get(ticker, "?")
            log.info("  %-10s  %s  →  %s", ticker, old, isin)

    if unresolved:
        log.warning("%d suspect ticker(s) could not be auto-fixed "
                    "(FMP returned no ISIN present in N-PORT):", len(unresolved))
        for t in unresolved:
            eff = patched_by_ticker.get(t, ticker_to_isin.get(t, "?"))
            log.warning("  %-10s  current=%s", t, eff)

    log.info("Done. %d patched, %d unresolved.", len(resolved), len(unresolved))
    if resolved:
        log.info("Re-run 'python create_universe.py' to rebuild companies table with corrected ISINs.")


# ---------------------------------------------------------------------------
# DB writer
# ---------------------------------------------------------------------------

def write_db(companies: pd.DataFrame, snapshots: pd.DataFrame) -> None:
    with get_db(DB_PATH) as conn:
        # Reference tables are NEVER dropped — user edits survive full rebuilds.
        # seed_all_reference_tables uses CREATE TABLE IF NOT EXISTS + INSERT OR IGNORE.
        seed_all_reference_tables(conn)

        # Preserve recovered delisted/dropped securities (delisted_date IS NOT NULL)
        # across the hard rebuild below — build_companies only knows the *current*
        # index CSVs, so without this the survivorship-bias coverage would be wiped.
        preserved = pd.DataFrame()
        has_companies = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='companies'"
        ).fetchone()
        if has_companies:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(companies)").fetchall()]
            if "delisted_date" in cols:
                preserved = pd.read_sql(
                    "SELECT * FROM companies WHERE delisted_date IS NOT NULL", conn
                )

        conn.executescript("""
            DROP TABLE IF EXISTS universe_snapshots;
            DROP TABLE IF EXISTS companies;
            DROP TABLE IF EXISTS universe;

            CREATE TABLE companies (
                isin                TEXT PRIMARY KEY,
                ticker              TEXT,
                company_name        TEXT,
                gics_sector         TEXT,
                gics_industry_group TEXT,
                gics_industry       TEXT,
                gics_sub_industry   TEXT,
                country             TEXT,
                exchange            TEXT,
                currency            TEXT,
                fiscal_year_end     INTEGER,
                num_employees       INTEGER,
                business_summary    TEXT,
                cik                 TEXT,
                cusip               TEXT,
                simfin_id           INTEGER,
                simfin_sector       TEXT,
                simfin_industry     TEXT,
                data_date           TEXT,
                update_date         TEXT,
                delisted_date       TEXT
            );

            CREATE TABLE universe_snapshots (
                snapshot_date  TEXT NOT NULL,
                isin           TEXT NOT NULL,
                index_name     TEXT NOT NULL,
                weight         REAL,
                market_value   REAL,
                PRIMARY KEY (snapshot_date, isin, index_name),
                FOREIGN KEY (isin) REFERENCES companies(isin)
            );

            CREATE INDEX idx_co_ticker  ON companies(ticker);
            CREATE INDEX idx_co_cik     ON companies(cik);
            CREATE INDEX idx_co_simfin  ON companies(simfin_id);
            CREATE INDEX idx_snap_date  ON universe_snapshots(snapshot_date, index_name);
        """)

        companies.to_sql("companies",         conn, if_exists="append", index=False)
        snapshots.to_sql("universe_snapshots", conn, if_exists="append", index=False)

        # Re-insert preserved delisted rows whose ISIN the current build did not
        # already cover (a name that re-entered the live universe wins).
        if not preserved.empty:
            current_isins = set(companies["isin"].dropna())
            preserved = preserved[~preserved["isin"].isin(current_isins)]
            if not preserved.empty:
                preserved.to_sql("companies", conn, if_exists="append", index=False)
                log.info("Preserved %d delisted/dropped securities across rebuild", len(preserved))
        conn.commit()

        n_alias = conn.execute("SELECT COUNT(*) FROM ticker_alias").fetchone()[0]
        n_patch = conn.execute("SELECT COUNT(*) FROM isin_patch").fetchone()[0]
        n_excl  = conn.execute("SELECT COUNT(*) FROM simfin_exclude").fetchone()[0]
        n_start = conn.execute("SELECT COUNT(*) FROM security_data_start").fetchone()[0]
        n_reg   = conn.execute("SELECT COUNT(*) FROM index_registry").fetchone()[0]
        n_acc   = conn.execute("SELECT COUNT(*) FROM nport_accessions").fetchone()[0]
        log.info("DB written: ticker_alias=%d, isin_patch=%d, simfin_exclude=%d, "
                 "security_data_start=%d, "
                 "index_registry=%d, nport_accessions=%d, "
                 "companies=%s, universe_snapshots=%s",
                 n_alias, n_patch, n_excl, n_start, n_reg, n_acc,
                 f"{len(companies):,}", f"{len(snapshots):,}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(
    companies: pd.DataFrame,
    snapshots: pd.DataFrame,
    patch: dict[str, str] | None = None,
) -> None:
    _patch    = patch if patch is not None else load_isin_patch()
    total     = len(companies)
    synthetic = companies["isin"].str.startswith("NOISN_").sum()
    patched   = companies["isin"].isin(_patch.values()).sum()
    simfin_m  = companies["simfin_id"].notna().sum()
    no_cik    = companies["cik"].isna().sum()

    log.info("=== COMPANIES TABLE  (%s securities) ===", f"{total:,}")
    log.info("  ISIN from patch:  %4d  (FMP / manual override)", patched)
    log.info("  ISIN from SimFin: %4d", total - synthetic - patched)
    log.info("  Synthetic ISIN:   %4d  (run --refresh-isins to resolve)", synthetic)
    if synthetic:
        synt = companies[companies["isin"].str.startswith("NOISN_")][["ticker","company_name"]].values
        for t, n in synt:
            log.info("    %-12s %s", t, n)
    log.info("  SimFin metadata:  %4d / %d", simfin_m, total)
    log.info("  Missing CIK:      %4d  (non-US or new listings)", no_cik)

    log.info("=== UNIVERSE SNAPSHOTS ===")
    for (idx, dt), grp in snapshots.groupby(["index_name", "snapshot_date"]):
        log.info("  %-20s %s   %5d companies", idx, dt, len(grp))

    log.info("GICS sector breakdown (all companies):")
    for sector, cnt in companies["gics_sector"].value_counts().items():
        log.info("  %-30s %4d", sector, cnt)


# ---------------------------------------------------------------------------
# Snapshot-only rebuild  (--rebuild-snapshots flag)
# ---------------------------------------------------------------------------

def _write_snapshots_only(snapshots: pd.DataFrame) -> None:
    """Replace universe_snapshots in place; leave companies and reference tables untouched."""
    with get_db(DB_PATH) as conn:
        seed_all_reference_tables(conn)
        conn.executescript("""
            DROP TABLE IF EXISTS universe_snapshots;
            CREATE TABLE universe_snapshots (
                snapshot_date  TEXT NOT NULL,
                isin           TEXT NOT NULL,
                index_name     TEXT NOT NULL,
                weight         REAL,
                market_value   REAL,
                PRIMARY KEY (snapshot_date, isin, index_name),
                FOREIGN KEY (isin) REFERENCES companies(isin)
            );
            CREATE INDEX idx_snap_date ON universe_snapshots(snapshot_date, index_name);
        """)
        snapshots.to_sql("universe_snapshots", conn, if_exists="append", index=False)
        conn.commit()

        n_reg = conn.execute("SELECT COUNT(*) FROM index_registry").fetchone()[0]
        n_acc = conn.execute("SELECT COUNT(*) FROM nport_accessions").fetchone()[0]

    log.info("Snapshots written: index_registry=%d, nport_accessions=%d, universe_snapshots=%s",
             n_reg, n_acc, f"{len(snapshots):,}")


def _find_latest_nport(
    snap_date_iso: str,
    etf_ticker: str,
    series_id: str,
    cik: str,
    max_candidates: int = 20,
) -> tuple[str, str] | None:
    """
    Discover the N-PORT-P filing for `etf_ticker` with the most recent
    period_of_report ≤ snap_date.

    iShares Trust (CIK 1100663) files one N-PORT-P per fund series per
    reporting period. We disambiguate by reading each candidate's
    primary_doc.xml and matching the <seriesId> element.

    Returns (accession_number, period_of_report_iso) or None if not found.
    """
    try:
        from edgar import set_identity
        from edgar.funds import find_fund
    except ImportError:
        log.error("edgar library not installed — cannot discover N-PORT accessions.")
        return None

    set_identity("universe-builder shivam3125@gmail.com")
    fund = find_fund(etf_ticker)
    df = fund.series.get_filings(form="NPORT-P").to_pandas()
    df["rd"] = pd.to_datetime(df["reportDate"]).dt.date.astype(str)

    candidates = df[df["rd"] <= snap_date_iso].copy()
    if candidates.empty:
        log.warning("No %s N-PORT-P with period_of_report ≤ %s on EDGAR",
                    etf_ticker, snap_date_iso)
        return None
    # Most recent period first; within a period, largest filing first
    candidates = candidates.sort_values(["rd", "size"], ascending=[False, False])

    for _, row in candidates.head(max_candidates).iterrows():
        acc = row["accession_number"]
        try:
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-','')}/primary_doc.xml"
            xml_data = _edgar_fetch_bytes(url, timeout=20)
            root = ET.fromstring(xml_data)
            ns = {"n": "http://www.sec.gov/edgar/nport"}
            sid_el = root.find(".//n:seriesId", ns)
            if sid_el is not None and sid_el.text == series_id:
                return acc, row["rd"]
        except Exception as e:
            log.debug("  skip acc %s (%s)", acc, e)
            continue
        time.sleep(0.15)

    log.warning("Did not find %s filing in top %d candidates for snap %s",
                etf_ticker, max_candidates, snap_date_iso)
    return None


# Indexes we auto-discover N-PORT accessions for.
# Each entry must match a row in index_registry with a valid series_id + cik.
_NPORT_AUTO_INDEXES: list[dict] = [
    {
        "index_name": "russell_1000",
        "etf_ticker":  "IWB",
        "series_id":   "S000004347",
        "cik":         "1100663",
    },
    {
        "index_name": "sp500",
        "etf_ticker":  "IVV",
        "series_id":   "S000004310",
        "cik":         "1100663",
    },
]

# Capped index definitions: index_name -> (base_index, cap).
# For each entry, real N-PORT holdings (from index_registry + nport_accessions) are used
# when available; all other base-index snapshot dates are filled synthetically.
_CAPPED_INDEX_DEFINITIONS: dict[str, tuple[str, float]] = {
    "sp500_3pct_capped": ("sp500", 0.03),
}


def _backfill_capped_index_snapshots(
    index_name: str,
    base_index: str,
    cap: float,
) -> int:
    """Fill universe_snapshots for a capped index for all base-index dates not already covered.

    Real N-PORT rows fetched by rebuild_snapshots are preserved. Synthetic dates
    are rebuilt in place so improvements to the cap algorithm repair old rows.
    Returns the number of synthetic rows inserted.
    """
    from utils import apply_weight_cap

    with get_db(DB_PATH) as conn:
        base_dates = [r[0] for r in conn.execute(
            "SELECT DISTINCT snapshot_date FROM universe_snapshots WHERE index_name=? ORDER BY snapshot_date",
            (base_index,),
        ).fetchall()]

        real_dates = {r[0] for r in conn.execute(
            "SELECT DISTINCT snapshot_date FROM nport_accessions WHERE index_name=?",
            (index_name,),
        ).fetchall()}
        synthetic_dates = [d for d in base_dates if d not in real_dates]

        rows: list[tuple[str, str, str, float, None]] = []
        for snap_date in synthetic_dates:
            base_rows = conn.execute(
                "SELECT isin, weight FROM universe_snapshots WHERE index_name=? AND snapshot_date=?",
                (base_index, snap_date),
            ).fetchall()
            if not base_rows:
                continue
            raw = {isin: w for isin, w in base_rows if w is not None and w > 0}
            capped = apply_weight_cap(raw, cap=cap)
            for isin, w in capped.items():
                rows.append((snap_date, isin, index_name, w, None))

        if synthetic_dates:
            ph = ",".join("?" * len(synthetic_dates))
            conn.execute(
                f"DELETE FROM universe_snapshots "
                f"WHERE index_name=? AND snapshot_date IN ({ph})",
                [index_name] + synthetic_dates,
            )

        if rows:
            conn.executemany(
                "INSERT INTO universe_snapshots "
                "(snapshot_date, isin, index_name, weight, market_value) VALUES (?, ?, ?, ?, ?)",
                rows,
            )
        conn.commit()

    return len(rows)


def ensure_snapshot(snap_date_iso: str, *, include_legacy_csv: bool = False) -> None:
    """
    Make `universe_snapshots` current for `snap_date_iso` across all
    auto-discovered indexes (russell_1000 + sp500).

    For each index in _NPORT_AUTO_INDEXES:
      Phase 1 (discovery): if `nport_accessions` lacks a row for
      (index_name, snap_date), query EDGAR for the latest N-PORT-P with
      period_of_report ≤ snap_date and insert it.

    Phase 2 (rebuild): call `rebuild_snapshots()` once to refresh
    `universe_snapshots` from N-PORT accessions. Legacy CSV snapshots are
    included only when include_legacy_csv=True.

    Idempotent: if all accessions are already known, only Phase 2 runs.
    """
    log.info("=== ENSURE SNAPSHOT %s ===", snap_date_iso)
    with get_db(DB_PATH) as conn:
        for idx in _NPORT_AUTO_INDEXES:
            iname = idx["index_name"]
            row = conn.execute(
                "SELECT accession FROM nport_accessions "
                "WHERE index_name=? AND snapshot_date=?",
                (iname, snap_date_iso),
            ).fetchone()
            if row:
                log.info("[%s] Already have accession %s for %s",
                         iname, row[0], snap_date_iso)
                continue
            log.info("[%s] Looking up %s N-PORT-P on EDGAR for %s ...",
                     iname, idx["etf_ticker"], snap_date_iso)
            result = _find_latest_nport(
                snap_date_iso, idx["etf_ticker"], idx["series_id"], idx["cik"]
            )
            if result is None:
                log.error("[%s] Could not discover N-PORT accession for %s — skipping.",
                          iname, snap_date_iso)
                continue
            accession, period = result
            log.info("[%s] Found: acc=%s  period_of_report=%s", iname, accession, period)
            conn.execute(
                "INSERT INTO nport_accessions "
                "(index_name, snapshot_date, accession, period_ending) "
                "VALUES (?, ?, ?, ?)",
                (iname, snap_date_iso, accession, period),
            )
        conn.commit()

    rebuild_snapshots(include_legacy_csv=include_legacy_csv)


def align_scheduled_universe_snapshots(indexes: list[str] | None = None) -> None:
    """Fill scheduled universe dates from the latest N-PORT-backed as-of snapshot.

    This makes the as-of rule explicit in universe_snapshots and nport_accessions
    instead of relying on downstream "nearest available" logic. It does not fetch
    EDGAR and does not overwrite existing universe rows.
    """
    with get_db(DB_PATH) as conn:
        seed_all_reference_tables(conn)
        if indexes:
            selected_indexes = indexes
        else:
            selected_indexes = [
                row[0] for row in conn.execute(
                    """
                    SELECT DISTINCT us.index_name
                    FROM universe_snapshots us
                    JOIN nport_accessions na
                      ON na.index_name = us.index_name
                     AND na.snapshot_date = us.snapshot_date
                    ORDER BY us.index_name
                    """
                ).fetchall()
            ]
        schedule_dates = [
            row[0] for row in conn.execute(
                "SELECT data_date FROM snapshot_schedule ORDER BY data_date"
            ).fetchall()
        ]
        if not selected_indexes:
            raise RuntimeError("No N-PORT-backed universe indexes found to align.")
        if not schedule_dates:
            raise RuntimeError("snapshot_schedule is empty. Run --rebuild-schedule first.")

        inserted_accessions = inserted_rows = skipped = 0
        for index_name in selected_indexes:
            for target_date in schedule_dates:
                exists = conn.execute(
                    """
                    SELECT 1
                    FROM universe_snapshots
                    WHERE index_name = ? AND snapshot_date = ?
                    LIMIT 1
                    """,
                    (index_name, target_date),
                ).fetchone()
                if exists:
                    continue

                source = conn.execute(
                    """
                    SELECT us.snapshot_date, na.accession, na.period_ending
                    FROM universe_snapshots us
                    JOIN nport_accessions na
                      ON na.index_name = us.index_name
                     AND na.snapshot_date = us.snapshot_date
                    WHERE us.index_name = ?
                      AND us.snapshot_date <= ?
                    GROUP BY us.snapshot_date, na.accession, na.period_ending
                    ORDER BY us.snapshot_date DESC
                    LIMIT 1
                    """,
                    (index_name, target_date),
                ).fetchone()
                if not source:
                    skipped += 1
                    continue
                source_date, accession, period_ending = source

                before = conn.total_changes
                conn.execute(
                    """
                    INSERT OR IGNORE INTO nport_accessions
                        (index_name, snapshot_date, accession, period_ending)
                    VALUES (?, ?, ?, ?)
                    """,
                    (index_name, target_date, accession, period_ending),
                )
                inserted_accessions += conn.total_changes - before

                before = conn.total_changes
                conn.execute(
                    """
                    INSERT OR IGNORE INTO universe_snapshots
                        (snapshot_date, isin, index_name, weight, market_value)
                    SELECT ?, isin, index_name, weight, market_value
                    FROM universe_snapshots
                    WHERE index_name = ? AND snapshot_date = ?
                    """,
                    (target_date, index_name, source_date),
                )
                inserted_rows += conn.total_changes - before

        conn.commit()

    log.info(
        "Aligned scheduled universe snapshots: indexes=%s inserted_rows=%s inserted_accessions=%d skipped=%d",
        ",".join(selected_indexes),
        f"{inserted_rows:,}",
        inserted_accessions,
        skipped,
    )


def _audit_pit_coverage() -> None:
    """
    Log R1000 members at the most recent universe_snapshots date that are missing
    from `companies` (i.e. silently excluded from Barra's PIT R1000 filter).

    Informational only — does not fail. Run at the end of rebuild_snapshots()
    so each weekly --ensure-snapshot run flags new entrants we haven't ingested.
    Backfill missing names with `update_constituents.py --ticker <T>`.
    """
    with get_db(DB_PATH) as conn:
        latest = conn.execute(
            "SELECT MAX(snapshot_date) FROM universe_snapshots WHERE index_name='russell_1000'"
        ).fetchone()[0]
        if latest is None:
            return
        rows = conn.execute(
            "SELECT us.isin FROM universe_snapshots us "
            "LEFT JOIN companies c ON us.isin = c.isin "
            "WHERE us.index_name='russell_1000' AND us.snapshot_date=? AND c.isin IS NULL "
            "ORDER BY us.isin",
            (latest,),
        ).fetchall()
    if not rows:
        log.info("PIT audit (%s): all R1000 members are in companies ✓", latest)
        return
    log.warning(
        "PIT audit (%s): %d R1000 member(s) missing from companies — "
        "they will be silently excluded from Barra. Backfill via "
        "`python update_constituents.py --ticker <T>`.",
        latest, len(rows),
    )
    for (isin,) in rows:
        log.warning("  orphan: %s", isin)


def rebuild_snapshots(*, include_legacy_csv: bool = False) -> None:
    """Rebuild universe_snapshots from EDGAR N-PORT-P. Does not touch companies.

    Local ETF holdings CSVs are legacy/bootstrap inputs and are excluded by
    default. Pass include_legacy_csv=True to preserve old CSV-sourced snapshots
    during a transitional rebuild.
    """
    log.info("=== REBUILD UNIVERSE SNAPSHOTS ===")

    with get_db(DB_PATH) as conn:
        companies = pd.read_sql("SELECT isin, ticker FROM companies", conn)
        seed_all_reference_tables(conn)
    known_isins = set(companies["isin"].dropna())
    log.info("Known ISINs from companies table: %d", len(known_isins))

    registry = load_index_registry()

    snapshots_csv = pd.DataFrame()
    if include_legacy_csv:
        index_files = sorted(INDEX_DIR.glob("*.csv")) if INDEX_DIR.exists() else []
        ishares_frames: list[tuple[pd.DataFrame, str, str]] = []
        for path in index_files:
            eq, snapshot_date, index_name = load_ishares(path)
            log.info("  %-45s  %5d holdings  (%s @ %s)", path.name, len(eq), index_name, snapshot_date)
            ishares_frames.append((eq, snapshot_date, index_name))

        snapshots_csv = build_snapshots(ishares_frames, companies) if ishares_frames else pd.DataFrame()
        log.warning("Legacy CSV snapshots included: %s rows", f"{len(snapshots_csv):,}")
    else:
        log.info("Legacy CSV snapshots excluded (N-PORT-only rebuild).")

    log.info("Fetching N-PORT-P snapshots from EDGAR ...")
    hist = build_historical_snapshots(known_isins, registry=registry)
    log.info("EDGAR snapshots: %s rows", f"{len(hist):,}")

    all_snapshots = pd.concat([snapshots_csv, hist], ignore_index=True)
    all_snapshots = all_snapshots.drop_duplicates(subset=["snapshot_date", "isin", "index_name"])
    log.info("Total unique snapshot rows: %s", f"{len(all_snapshots):,}")

    log.info("Writing to %s ...", DB_PATH)
    _write_snapshots_only(all_snapshots)

    for idx_name in sorted(all_snapshots["index_name"].unique()):
        sub = all_snapshots[all_snapshots["index_name"] == idx_name]
        dates = sorted(sub["snapshot_date"].unique())
        log.info("[%s] %d snapshot dates, %s total rows", idx_name, len(dates), f"{len(sub):,}")
        for d in dates:
            n = (sub["snapshot_date"] == d).sum()
            log.info("  %s: %d companies", d, n)

    _audit_pit_coverage()

    # Fill synthetic capped-index snapshots for all dates not covered by real N-PORT.
    for capped_name, (base_idx, cap) in _CAPPED_INDEX_DEFINITIONS.items():
        n = _backfill_capped_index_snapshots(capped_name, base_idx, cap)
        log.info("[%s] %d synthetic snapshot rows added (cap=%.0f%%)", capped_name, n, cap * 100)

    log.info("Done.")


# ---------------------------------------------------------------------------
# Clean universe materialisation  (--materialize-clean-snapshots flag)
# ---------------------------------------------------------------------------

def _available_snapshot_indexes() -> list[str]:
    with get_db(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT DISTINCT index_name FROM universe_snapshots ORDER BY index_name"
        ).fetchall()
    return [r[0] for r in rows]


def _scheduled_dates(computed_only: bool = True) -> list[str]:
    sql = "SELECT data_date FROM snapshot_schedule"
    if computed_only:
        sql += " WHERE factors_computed_at IS NOT NULL"
    sql += " ORDER BY data_date"
    with get_db(DB_PATH) as conn:
        rows = conn.execute(sql).fetchall()
    return [r[0] for r in rows]


def _clean_snapshot_frame(
    index_name: str,
    snapshot_date: str,
    *,
    mode: str,
) -> pd.DataFrame:
    from universe_loader import load_clean_universe

    require_latest_volume = mode == "live"
    normalize_live_isin = mode == "live"
    min_return_date = snapshot_date
    result = load_clean_universe(
        index_name,
        snapshot_date,
        benchmark_index=None,
        mode=mode,
        normalize_live_isin=normalize_live_isin,
        min_return_date=min_return_date,
        require_latest_volume=require_latest_volume,
        tradable_only=False,
    )
    df = result.members.copy()
    out = pd.DataFrame({
        "mode": mode,
        "requested_snapshot_date": result.requested_snapshot_date,
        "source_snapshot_date": result.source_snapshot_date,
        "index_name": index_name,
        "isin": df["isin"],
        "canonical_isin": df["canonical_isin"],
        "original_isin": df["original_isin"],
        "mapped_from_isin": df["mapped_from_isin"],
        "mapped_to_isin": df["mapped_to_isin"],
        "ticker": df["ticker"],
        "company_name": df["company_name"],
        "weight": df["weight"],
        "raw_weight": df["raw_weight"],
        "market_value": df["market_value"],
        "gics_sector": df["gics_sector"],
        "gics_industry_group": df["gics_industry_group"],
        "gics_industry": df["gics_industry"],
        "gics_sub_industry": df["gics_sub_industry"],
        "simfin_sector": df["simfin_sector"],
        "simfin_industry": df["simfin_industry"],
        "country": df["country"],
        "exchange": df["exchange"],
        "currency": df["currency"],
        "cik": df["cik"],
        "cusip": df["cusip"],
        "delisted_date": df["delisted_date"],
        "last_return_date": df["last_return_date"],
        "latest_close": df["latest_close"],
        "latest_volume": df["latest_volume"],
        "identity_status": df["identity_status"],
        "identity_rule": df["identity_rule"],
        "identity_confidence": df["identity_confidence"],
        "identity_evidence": df["identity_evidence"],
        "is_tradable": df["is_tradable"].astype(int),
        "exclude_reason": df["exclude_reason"],
        "materialized_at": datetime.now().isoformat(timespec="seconds"),
    })
    return out


def materialize_clean_snapshots(
    *,
    indexes: list[str] | None = None,
    mode: str = "point_in_time",
    only_latest: bool = False,
) -> None:
    """
    Build clean_universe_snapshots from raw universe_snapshots + security master.

    Raw universe_snapshots remain untouched.  The clean table is the optimizer-facing
    product with delisted/stale/security-identity flags applied.

    Modes:
      point_in_time — no future ISIN normalisation; suitable for backtests.
      live          — map stale same-name ticker ISINs to the newest current ISIN.
    """
    if mode not in {"point_in_time", "live"}:
        raise ValueError("mode must be 'point_in_time' or 'live'")

    _indexes = indexes if indexes else _available_snapshot_indexes()
    if not _indexes:
        raise RuntimeError("No universe snapshot indexes found.")

    if only_latest:
        with get_db(DB_PATH) as conn:
            rows = conn.execute(
                "SELECT MAX(snapshot_date) FROM universe_snapshots"
            ).fetchall()
        dates = [rows[0][0]] if rows and rows[0][0] else []
    else:
        dates = _scheduled_dates(computed_only=True)

    if not dates:
        raise RuntimeError("No scheduled dates found for clean materialisation.")

    frames: list[pd.DataFrame] = []
    failures: list[tuple[str, str, str]] = []
    log.info(
        "Materialising clean_universe_snapshots: mode=%s indexes=%s dates=%d",
        mode, ",".join(_indexes), len(dates),
    )
    for snapshot_date in dates:
        for index_name in _indexes:
            try:
                frame = _clean_snapshot_frame(index_name, snapshot_date, mode=mode)
            except Exception as exc:
                failures.append((index_name, snapshot_date, str(exc)))
                log.warning("[%s %s] clean materialisation skipped: %s", index_name, snapshot_date, exc)
                continue
            frames.append(frame)

    if not frames:
        raise RuntimeError("Clean materialisation produced no rows.")

    clean = pd.concat(frames, ignore_index=True)
    clean = clean.drop_duplicates(
        subset=["mode", "requested_snapshot_date", "index_name", "isin"],
        keep="last",
    )

    with get_db(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS clean_universe_snapshots (
                mode                    TEXT NOT NULL,
                requested_snapshot_date TEXT NOT NULL,
                source_snapshot_date    TEXT NOT NULL,
                index_name              TEXT NOT NULL,
                isin                    TEXT NOT NULL,
                canonical_isin          TEXT,
                original_isin           TEXT,
                mapped_from_isin        TEXT,
                mapped_to_isin          TEXT,
                ticker                  TEXT,
                company_name            TEXT,
                weight                  REAL,
                raw_weight              REAL,
                market_value            REAL,
                gics_sector             TEXT,
                gics_industry_group     TEXT,
                gics_industry           TEXT,
                gics_sub_industry       TEXT,
                simfin_sector           TEXT,
                simfin_industry         TEXT,
                country                 TEXT,
                exchange                TEXT,
                currency                TEXT,
                cik                     TEXT,
                cusip                   TEXT,
                delisted_date           TEXT,
                last_return_date        TEXT,
                latest_close            REAL,
                latest_volume           REAL,
                identity_status         TEXT,
                identity_rule           TEXT,
                identity_confidence     REAL,
                identity_evidence       TEXT,
                is_tradable             INTEGER NOT NULL,
                exclude_reason          TEXT,
                materialized_at         TEXT,
                PRIMARY KEY (mode, requested_snapshot_date, index_name, isin)
            )
        """)
        existing_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(clean_universe_snapshots)").fetchall()
        }
        for col_name, col_type in {
            "canonical_isin": "TEXT",
            "mapped_from_isin": "TEXT",
            "mapped_to_isin": "TEXT",
            "identity_rule": "TEXT",
            "identity_confidence": "REAL",
            "identity_evidence": "TEXT",
        }.items():
            if col_name not in existing_cols:
                conn.execute(f"ALTER TABLE clean_universe_snapshots ADD COLUMN {col_name} {col_type}")
        ph_indexes = ",".join("?" * len(_indexes))
        ph_dates = ",".join("?" * len(dates))
        conn.execute(
            f"""
            DELETE FROM clean_universe_snapshots
            WHERE mode = ?
              AND index_name IN ({ph_indexes})
              AND requested_snapshot_date IN ({ph_dates})
            """,
            [mode, *_indexes, *dates],
        )
        clean.to_sql("clean_universe_snapshots", conn, if_exists="append", index=False)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_clean_universe_lookup "
            "ON clean_universe_snapshots(mode, requested_snapshot_date, index_name, is_tradable)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_clean_universe_isin "
            "ON clean_universe_snapshots(isin)"
        )
        conn.commit()

    tradable = int(clean["is_tradable"].sum())
    log.info(
        "Clean snapshots written: rows=%s tradable=%s blocked=%s failures=%d",
        f"{len(clean):,}", f"{tradable:,}", f"{len(clean) - tradable:,}", len(failures),
    )
    for index_name, snapshot_date, detail in failures[:20]:
        log.warning("  failure: %s %s — %s", index_name, snapshot_date, detail)
    if len(failures) > 20:
        log.warning("  ... %d more failures", len(failures) - 20)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Delisted / dropped-name recovery  (--recover-delisted)
#
# Historical R1000/S&P500 members that were acquired, went bankrupt, or simply
# fell out of the index are present in universe_snapshots (PIT membership) but
# absent from the companies metadata table and returns.db — the classic
# survivorship hole.  This resolves ISIN→ticker via OpenFIGI (free) and enriches
# metadata via FMP, writing rows into companies so the PIT universe is fully
# covered.  Price history is then backfilled by create_returns.py --backfill-delisted.
# ---------------------------------------------------------------------------

_OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"

# FMP sector taxonomy → the GICS sector strings used by live rows in companies.
# Without this remap, dead names land in phantom sector buckets and the
# optimiser's sector-neutrality constraints break.
_FMP_TO_GICS_SECTOR: dict[str, str] = {
    "Technology":             "Information Technology",
    "Healthcare":             "Health Care",
    "Financial Services":     "Financials",
    "Consumer Cyclical":      "Consumer Discretionary",
    "Consumer Defensive":     "Consumer Staples",
    "Basic Materials":        "Materials",
    "Industrials":            "Industrials",
    "Energy":                 "Energy",
    "Utilities":              "Utilities",
    "Real Estate":            "Real Estate",
    "Communication Services": "Communication Services",
}

# US-listing exchange codes in OpenFIGI (composite "US" + venue codes).
_OPENFIGI_US_EXCH = {"US", "UN", "UW", "UQ", "UA", "UR", "UP", "UV", "UD"}

# Currency suffixes OpenFIGI appends to synthetic foreign listings (MRVLUSD → MRVL).
_CCY_SUFFIX = ("USD", "EUR", "GBP", "CHF", "CAD", "JPY", "AUD", "HKD", "SGD", "SEK", "NOK", "DKK")


def _openfigi_map_isins(isins: list[str], figi_key: str | None = None) -> dict[str, str]:
    """Resolve ISIN→US-listed ticker via OpenFIGI /v3/mapping (free, no paid key).

    Batched (≤10 jobs/request unkeyed) and throttled to stay under the 25 req/min
    public limit.  Prefers a US-composite Common Stock listing.
    Returns {isin: ticker} only for ISINs that resolve to a US equity.
    """
    headers = {"Content-Type": "application/json"}
    if figi_key:
        headers["X-OPENFIGI-APIKEY"] = figi_key
    batch_size = 100 if figi_key else 10
    pause      = 0.3 if figi_key else 2.6  # keyed: 250/min, unkeyed: ~23/min

    def _pick(data: list[dict]) -> str | None:
        eq = [d for d in data if d.get("marketSector") == "Equity"]
        common = [d for d in eq if d.get("securityType2") == "Common Stock"] or eq
        # 1. Clean US composite / venue listing → use its ticker directly.
        for codes in (("US",), tuple(_OPENFIGI_US_EXCH)):
            us = [d for d in common if d.get("exchCode") in codes]
            if us:
                return us[0].get("ticker")
        # 2. Redomiciled / foreign-domiciled name (old ISIN): OpenFIGI only carries
        #    synthetic currency-suffixed lines (e.g. "MRVLUSD" on exch XB/XS). Strip
        #    the trailing currency code to recover the real US ticker (MRVL). Caller
        #    validates it against companies (copy) or FMP.
        for d in common:
            tk = d.get("ticker") or ""
            for suf in _CCY_SUFFIX:
                if tk.endswith(suf) and len(tk) > len(suf):
                    return tk[: -len(suf)]
        return None

    out: dict[str, str] = {}
    for i in range(0, len(isins), batch_size):
        chunk = isins[i:i + batch_size]
        body  = json.dumps([{"idType": "ID_ISIN", "idValue": x} for x in chunk]).encode()
        result = None
        for attempt in range(4):
            try:
                req = urllib.request.Request(_OPENFIGI_URL, data=body, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = json.load(resp)
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    time.sleep(8)
                    continue
                break
            except Exception:
                time.sleep(2)
        if result:
            for isin, rr in zip(chunk, result):
                data = rr.get("data") if isinstance(rr, dict) else None
                if data:
                    tk = _pick(data)
                    if tk:
                        out[isin] = tk
        time.sleep(pause)
    return out


# FMP keys that returned a daily-cap 429 during this run — skipped thereafter so we
# never hammer an exhausted key (the user runs free tiers; quota is precious).
_FMP_EXHAUSTED: set[str] = set()


def _load_fmp_api_keys() -> list[str]:
    """Return all FMP keys from .env (FMP_API_KEY, FMP_API_KEY_SECOND, …), in order.

    Multiple free keys multiply the daily quota; rotation falls through to the next
    when one hits its cap. Tolerant of `KEY = value` spacing.
    """
    env = Path(".env")
    if not env.exists():
        return []
    keys: list[str] = []
    for line in env.read_text().splitlines():
        s = line.strip()
        if s.startswith("FMP_API_KEY") and "=" in s:
            val = s.split("=", 1)[1].strip()
            if val and val not in keys:
                keys.append(val)
    return keys


def _fmp_profile(ticker: str, keys: list[str]) -> dict | None:
    """Fetch FMP /stable/profile, rotating across keys.

    A 429 may be a transient per-minute throttle OR the daily cap. Wait-and-retry
    the same key a few times; only mark it exhausted after sustained 429s, so a
    momentary burst limit doesn't waste the remaining daily quota.
    """
    fmp_ticker = _FMP_TICKER_ALIAS.get(ticker, ticker).replace("/", "-")
    sym = urllib.parse.quote(fmp_ticker)
    for k in keys:
        if k in _FMP_EXHAUSTED:
            continue
        for attempt in range(3):
            try:
                data = _edgar_fetch(f"{_FMP_BASE}/profile?symbol={sym}&apikey={k}", timeout=15)
                return data[0] if isinstance(data, list) and data else None
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    if attempt < 2:
                        time.sleep(15)   # transient per-minute throttle — wait, retry
                        continue
                    _FMP_EXHAUSTED.add(k)  # sustained 429 → daily cap; try next key
                    break
                return None
            except Exception:
                return None
    return None


def _fmp_keys_available(keys: list[str]) -> bool:
    """Probe each not-yet-exhausted key once; returns True if any has quota left."""
    for k in keys:
        if k in _FMP_EXHAUSTED:
            continue
        try:
            _edgar_fetch(f"{_FMP_BASE}/profile?symbol=AAPL&apikey={k}", timeout=15)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 429:
                continue  # maybe transient — don't kill the key here; per-call retry decides
            return True
        except Exception:
            return True
    return False


def _ensure_delisted_column(conn: "sqlite3.Connection") -> None:
    """Idempotently add the delisted_date column to an existing companies table."""
    cols = [r[1] for r in conn.execute("PRAGMA table_info(companies)").fetchall()]
    if "delisted_date" not in cols:
        conn.execute("ALTER TABLE companies ADD COLUMN delisted_date TEXT")
        log.info("Added delisted_date column to companies table")


def recover_delisted_securities(limit: int | None = None) -> None:
    """Resolve and insert historical universe members missing from companies.

    Targets every ISIN that appears in universe_snapshots (any index) but has no
    companies row.  Resolves ticker via OpenFIGI, enriches via FMP, maps the FMP
    sector to GICS, and writes the row with delisted_date set (NULL if the name is
    still actively trading, else its last index-membership date).
    """
    keys = _load_fmp_api_keys()
    if not keys:
        raise RuntimeError("FMP_API_KEY not found in .env")
    figi_key = _load_openfigi_api_key()

    with get_db(DB_PATH) as conn:
        _ensure_delisted_column(conn)
        have = {r[0] for r in conn.execute("SELECT isin FROM companies").fetchall()}
        last_seen = dict(conn.execute(
            "SELECT isin, MAX(snapshot_date) FROM universe_snapshots GROUP BY isin"
        ).fetchall())

    targets = sorted(i for i in last_seen if i not in have)
    if limit:
        targets = targets[:limit]
    log.info("Delisted recovery: %d universe ISINs missing from companies", len(targets))
    if not targets:
        return

    log.info("Resolving ISIN→ticker via OpenFIGI ...")
    isin2ticker = _openfigi_map_isins(targets, figi_key=figi_key)
    log.info("  OpenFIGI resolved %d / %d tickers", len(isin2ticker), len(targets))

    fmp_ok = _fmp_keys_available(keys)
    if not fmp_ok:
        log.warning("FMP quota exhausted — this run inserts only names whose metadata can be "
                    "copied from existing live rows. Re-run after the daily reset to finish.")

    # Many dead ISINs resolve to a ticker that is ALREADY a live company under a
    # different ISIN (corporate action changed the ISIN — the BlackRock pattern).
    # Copy that row's metadata for free and consistently, sparing an FMP call.
    with get_db(DB_PATH) as conn:
        live_by_ticker: dict[str, tuple] = {}
        for row in conn.execute(
            "SELECT ticker, company_name, gics_sector, simfin_industry, country, "
            "       exchange, currency, cik, cusip FROM companies "
            "WHERE delisted_date IS NULL AND ticker IS NOT NULL AND ticker != ''"
        ).fetchall():
            live_by_ticker.setdefault(row[0], row)

    today = datetime.now().strftime("%Y-%m-%d")
    cols = ["isin", "ticker", "company_name", "gics_sector", "simfin_industry",
            "country", "exchange", "currency", "cik", "cusip",
            "data_date", "update_date", "delisted_date"]
    insert_sql = (f"INSERT OR REPLACE INTO companies ({','.join(cols)}) "
                  f"VALUES ({','.join('?' * len(cols))})")

    n_inserted = n_active = n_delisted = n_no_sector = n_copied = n_fmp = 0
    buf: list[tuple] = []
    # Commit incrementally so an interruption (ENOSPC, FMP circuit-breaker) keeps
    # everything resolved so far — the run is then fully resumable.
    with get_db(DB_PATH) as conn:
        _ensure_delisted_column(conn)
        for n, isin in enumerate(targets, 1):
            tk = isin2ticker.get(isin)
            if not tk:
                continue
            live = live_by_ticker.get(tk)
            if live is not None:
                # Alive under a changed ISIN — copy metadata, still trading.
                _, nm, gics, ind, country, exch, ccy, cik, cusip = live
                active = True
                n_copied += 1
            else:
                if not fmp_ok:
                    continue  # defer to a later run when FMP quota is available
                prof = _fmp_profile(tk, keys)
                n_fmp += 1
                if not prof:
                    continue  # resumable: re-run after FMP quota resets
                sector_raw = prof.get("sector") or ""
                gics  = _FMP_TO_GICS_SECTOR.get(sector_raw, sector_raw or None)
                nm    = prof.get("companyName") or ""
                ind   = prof.get("industry") or None
                country = prof.get("country") or None
                exch  = prof.get("exchange") or None
                ccy   = prof.get("currency") or None
                cik   = prof.get("cik") or None
                cusip = prof.get("cusip") or None
                active = bool(prof.get("isActivelyTrading"))
                time.sleep(0.25)  # ~4 req/s — only when we actually call FMP
            if not nm:
                continue  # require at least a name to insert a usable row
            if not gics:
                n_no_sector += 1
            n_active   += int(active)
            n_delisted += int(not active)
            buf.append((isin, tk, nm, gics, ind, country, exch, ccy, cik, cusip,
                        today, today, None if active else last_seen.get(isin)))
            n_inserted += 1
            if len(buf) >= 25:
                conn.executemany(insert_sql, buf)
                conn.commit()
                buf.clear()
                log.info("  inserted %d / %d  (copied=%d, fmp=%d)",
                         n_inserted, len(targets), n_copied, n_fmp)
        if buf:
            conn.executemany(insert_sql, buf)
            conn.commit()

    n_unresolved = len(targets) - n_inserted
    log.info("=== Delisted recovery complete ===")
    log.info("  inserted into companies: %d  (still-trading=%d, delisted=%d)",
             n_inserted, n_active, n_delisted)
    log.info("  metadata via copy=%d, via FMP=%d", n_copied, n_fmp)
    log.info("  no GICS sector resolved:  %d", n_no_sector)
    log.info("  still missing (no ticker / FMP unavailable): %d  — re-run to resume", n_unresolved)


def _load_openfigi_api_key() -> str | None:
    """Read OPENFIGI_API_KEY from .env (optional — raises OpenFIGI rate limit)."""
    env = Path(".env")
    if not env.exists():
        return None
    for line in env.read_text().splitlines():
        line = line.strip()
        if line.startswith("OPENFIGI_API_KEY="):
            return line.split("=", 1)[1].strip() or None
    return None


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build or update universe.db")
    parser.add_argument(
        "--rebuild-snapshots", action="store_true",
        help="Rebuild universe_snapshots from N-PORT accessions only (leaves companies table intact)",
    )
    parser.add_argument(
        "--include-legacy-csv", action="store_true",
        help=(
            "Include local data/universe_index CSV holdings in --rebuild-snapshots "
            "or --ensure-snapshot. Transitional/legacy only."
        ),
    )
    parser.add_argument(
        "--legacy-rebuild-companies", action="store_true",
        help=(
            "Legacy full rebuild of companies from local ETF CSVs + SimFin, then "
            "append N-PORT snapshots. This path is retained for rollback/bootstrap "
            "only while the EDGAR-first security master is completed."
        ),
    )
    parser.add_argument(
        "--ensure-snapshot", metavar="YYYY-MM-DD",
        help=(
            "Discover the latest IVV/IWB N-PORT-P accessions for the given snapshot date "
            "(if missing from nport_accessions), then rebuild universe_snapshots. "
            "Used by daily_ecosystem_update.py weekly cadence."
        ),
    )
    parser.add_argument(
        "--align-scheduled-universe-snapshots", action="store_true",
        help=(
            "Fill missing snapshot_schedule dates in universe_snapshots using the "
            "latest N-PORT-backed snapshot on or before each date. No EDGAR fetch."
        ),
    )
    parser.add_argument(
        "--align-index", action="append", default=[],
        help=(
            "Index for --align-scheduled-universe-snapshots. Repeatable. "
            "Default: all N-PORT-backed universe indexes."
        ),
    )
    parser.add_argument(
        "--refresh-isins", action="store_true",
        help="Legacy: fetch ISINs from FMP for tickers in universe_index CSVs and write to isin_patch",
    )
    parser.add_argument(
        "--fix-isins", action="store_true",
        help=(
            "Find companies whose ISINs differ from N-PORT, resolve correct ticker via "
            "EDGAR EFTS CUSIP search, and write authoritative ISINs to isin_patch"
        ),
    )
    parser.add_argument(
        "--recover-delisted", action="store_true",
        help=(
            "Resolve historical universe members missing from companies (acquired / "
            "bankrupt / dropped from index) via OpenFIGI + FMP and insert them with "
            "delisted_date set. Fixes survivorship coverage. Then run "
            "create_returns.py --backfill-delisted for prices."
        ),
    )
    parser.add_argument(
        "--refresh-nport-metadata", action="store_true",
        help=(
            "Fetch N-PORT holding-level security metadata into "
            "nport_security_metadata. EDGAR-first staging; does not mutate companies."
        ),
    )
    parser.add_argument(
        "--nport-index", action="append", default=[],
        help="Index for --refresh-nport-metadata. Repeatable. Default: all indexes with accessions.",
    )
    parser.add_argument(
        "--nport-latest-only", action="store_true",
        help="Only refresh each selected index's latest N-PORT accession.",
    )
    parser.add_argument(
        "--nport-force", action="store_true",
        help="Re-fetch N-PORT metadata even when an accession is already staged.",
    )
    parser.add_argument(
        "--stage-nport-company-candidates", action="store_true",
        help=(
            "Build nport_company_candidates from staged N-PORT metadata. "
            "Read-only with respect to companies; use for review/audit before backfills."
        ),
    )
    parser.add_argument(
        "--candidate-index", action="append", default=[],
        help=(
            "Index for --stage-nport-company-candidates. Repeatable. "
            "Default: all indexes with staged metadata."
        ),
    )
    parser.add_argument(
        "--candidate-latest-only", action="store_true",
        help="Only stage candidates from each selected index's latest snapshot.",
    )
    parser.add_argument(
        "--resolve-nport-company-candidates", action="store_true",
        help=(
            "Resolve missing nport_company_candidates to likely ticker/CIK in "
            "candidate review columns. Does not mutate companies."
        ),
    )
    parser.add_argument(
        "--promote-nport-company-candidates", action="store_true",
        help=(
            "Insert high-confidence resolved N-PORT candidates into companies by "
            "copying existing same-ticker metadata. Leaves ambiguous rows untouched."
        ),
    )
    parser.add_argument(
        "--repair-company-identifier-continuity", action="store_true",
        help=(
            "Carry known same-ticker/same-company identifier metadata across current "
            "ISIN changes when the source bridge is unambiguous."
        ),
    )
    parser.add_argument(
        "--promote-min-confidence", type=float, default=0.85,
        help="Minimum resolution_confidence for --promote-nport-company-candidates.",
    )
    parser.add_argument(
        "--rebuild-schedule", action="store_true",
        help=(
            "(Re)build the snapshot_schedule table — the single source of truth for "
            "snapshot dates (month-end monthly grid + weekly/legacy tags). Idempotent."
        ),
    )
    parser.add_argument(
        "--materialize-clean-snapshots", action="store_true",
        help=(
            "Build clean_universe_snapshots from raw universe_snapshots without "
            "modifying the raw PIT table."
        ),
    )
    parser.add_argument(
        "--clean-mode", choices=["point_in_time", "live"], default="point_in_time",
        help=(
            "Mode for --materialize-clean-snapshots. point_in_time keeps historical "
            "ISINs; live maps stale same-name ticker ISINs to current rows."
        ),
    )
    parser.add_argument(
        "--clean-index", action="append", default=[],
        help="Index to materialize. Repeat for multiple. Default: all universe indexes.",
    )
    parser.add_argument(
        "--clean-latest-only", action="store_true",
        help="Only materialize the latest raw universe snapshot date.",
    )
    parser.add_argument(
        "--backfill-capped-snapshots", metavar="INDEX_NAME",
        help=(
            "Backfill universe_snapshots for a capped index (e.g. sp500_3pct_capped) "
            "by applying the weight cap to all base-index snapshot dates not already "
            "covered by real N-PORT filings. Safe to re-run; skips existing rows."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of names processed (debug; applies to --recover-delisted)",
    )
    args = parser.parse_args()

    if args.rebuild_schedule:
        log.info("=== REBUILD SNAPSHOT SCHEDULE ===")
        rebuild_snapshot_schedule()
        return

    if args.materialize_clean_snapshots:
        log.info("=== MATERIALIZE CLEAN UNIVERSE SNAPSHOTS ===")
        materialize_clean_snapshots(
            indexes=args.clean_index or None,
            mode=args.clean_mode,
            only_latest=args.clean_latest_only,
        )
        return

    if args.backfill_capped_snapshots:
        iname = args.backfill_capped_snapshots
        if iname not in _CAPPED_INDEX_DEFINITIONS:
            log.error("Unknown capped index '%s'. Known: %s", iname, list(_CAPPED_INDEX_DEFINITIONS))
            sys.exit(1)
        base_idx, cap = _CAPPED_INDEX_DEFINITIONS[iname]
        log.info("=== BACKFILL CAPPED SNAPSHOTS: %s (base=%s cap=%.0f%%) ===", iname, base_idx, cap * 100)
        n = _backfill_capped_index_snapshots(iname, base_idx, cap)
        log.info("Done. %d synthetic rows inserted.", n)
        return

    if args.rebuild_snapshots:
        rebuild_snapshots(include_legacy_csv=args.include_legacy_csv)
        return

    if args.ensure_snapshot:
        ensure_snapshot(args.ensure_snapshot, include_legacy_csv=args.include_legacy_csv)
        return

    if args.align_scheduled_universe_snapshots:
        log.info("=== ALIGN SCHEDULED UNIVERSE SNAPSHOTS ===")
        align_scheduled_universe_snapshots(indexes=args.align_index or None)
        return

    if args.refresh_isins:
        log.info("=== REFRESH ISINs from FMP ===")
        refresh_isins()
        log.info("Done. Re-run without --refresh-isins to rebuild companies table with updated ISINs.")
        return

    if args.fix_isins:
        log.info("=== FIX ISINs via N-PORT + FMP validation ===")
        fix_isins()
        return

    if args.recover_delisted:
        log.info("=== RECOVER DELISTED / DROPPED SECURITIES ===")
        recover_delisted_securities(limit=args.limit)
        return

    if args.refresh_nport_metadata:
        log.info("=== REFRESH N-PORT SECURITY METADATA ===")
        refresh_nport_security_metadata(
            indexes=args.nport_index or None,
            only_latest=args.nport_latest_only,
            force=args.nport_force,
        )
        return

    if args.stage_nport_company_candidates:
        log.info("=== STAGE N-PORT COMPANY CANDIDATES ===")
        stage_nport_company_candidates(
            indexes=args.candidate_index or None,
            only_latest=args.candidate_latest_only,
        )
        return

    if args.resolve_nport_company_candidates:
        log.info("=== RESOLVE N-PORT COMPANY CANDIDATES ===")
        resolve_nport_company_candidates(limit=args.limit)
        return

    if args.repair_company_identifier_continuity:
        log.info("=== REPAIR COMPANY IDENTIFIER CONTINUITY ===")
        repair_company_identifier_continuity()
        consolidate_same_issuer_fundamentals()
        return

    if args.promote_nport_company_candidates:
        log.info("=== PROMOTE N-PORT COMPANY CANDIDATES ===")
        promote_nport_company_candidates(min_confidence=args.promote_min_confidence)
        return

    if not args.legacy_rebuild_companies:
        parser.print_help()
        log.warning(
            "No action selected. The old no-args SimFin/CSV company rebuild is now "
            "behind --legacy-rebuild-companies. Use --rebuild-snapshots for the "
            "N-PORT production path."
        )
        return

    log.warning("=== LEGACY CREATE UNIVERSE: CSV + SIMFIN SECURITY MASTER ===")

    log.info("Loading SimFin data ...")
    simfin = load_simfin()
    log.info("  %s companies loaded", f"{len(simfin):,}")

    index_files = sorted(INDEX_DIR.glob("*.csv")) if INDEX_DIR.exists() else []
    if not index_files:
        log.error("[ERROR] No CSV files found in data/universe_index/")
        return

    log.info("Loading %d index file(s) ...", len(index_files))
    ishares_frames: list[tuple[pd.DataFrame, str, str]] = []
    for path in index_files:
        eq, snapshot_date, index_name = load_ishares(path)
        log.info("  %-45s  %5d holdings  (%s @ %s)", path.name, len(eq), index_name, snapshot_date)
        ishares_frames.append((eq, snapshot_date, index_name))

    # Seed reference tables before first use, then load from DB
    log.info("Loading reference tables from DB ...")
    with get_db(DB_PATH) as conn:
        seed_all_reference_tables(conn)
    patch    = load_isin_patch()
    alias    = load_ticker_alias()
    simfin_exclude = load_simfin_exclude()
    registry = load_index_registry()
    log.info("  isin_patch: %d overrides | ticker_alias: %d | simfin_exclude: %d | indexes: %d",
             len(patch), len(alias), len(simfin_exclude), len(registry))

    log.info("Building companies table ...")
    companies = build_companies(
        ishares_frames,
        simfin,
        patch=patch,
        alias=alias,
        simfin_exclude=simfin_exclude,
    )

    log.info("Enriching metadata from EDGAR ...")
    companies = enrich_edgar_metadata(companies)

    log.info("Building universe_snapshots table ...")
    snapshots = build_snapshots(ishares_frames, companies, alias=alias)

    log.info("Fetching historical universe snapshots from EDGAR N-PORT-P ...")
    known_isins = set(companies["isin"].dropna())
    hist = build_historical_snapshots(known_isins, registry=registry)
    if not hist.empty:
        snapshots = pd.concat([snapshots, hist], ignore_index=True)

    log.info("Writing to %s ...", DB_PATH)
    write_db(companies, snapshots)

    print_report(companies, snapshots, patch=patch)

    log.info("=== Done ===")


if __name__ == "__main__":
    main()
