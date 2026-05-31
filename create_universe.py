"""
create_universe.py

Builds universe.db with two tables:

  companies          — one row per security (ISIN primary key), enriched with
                       metadata from iShares (GICS sector, exchange, country)
                       and SimFin (CIK, fiscal year end, employees, business
                       summary, SimFin sector/industry).

  universe_snapshots — index membership per snapshot date, one row per
                       (snapshot_date, isin, index_name).  Populated from
                       iShares holdings CSVs dropped into data/universe_index/.

Run:
  python create_universe.py

Add a new index snapshot:
  1. Download the iShares holdings CSV for that index and date.
  2. Drop it into data/universe_index/ with a filename convention:
         <index_name>_<YYYY_MM_DD>.csv   e.g. russell_1000_2025_04_01.csv
  3. Re-run this script — it merges all files, deduplicating by ISIN.
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import pandas as pd
import sqlite3
from pathlib import Path
from datetime import datetime

from config import DATA_DIR, SIMFIN_DIR, UNIVERSE_DB as DB_PATH
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
    seed_registry_tables(conn)


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
) -> pd.DataFrame:
    """
    Build the companies table from all iShares snapshots merged with SimFin.
    One row per unique iShares ticker (deduped by ISIN, latest snapshot wins).

    patch: ISIN overrides (ticker → ISIN). If None, load_isin_patch() is used.
    alias: SimFin ticker aliases (iShares ticker → SimFin ticker). If None, load_ticker_alias() is used.
    """
    _patch = patch if patch is not None else load_isin_patch()
    _alias = alias if alias is not None else load_ticker_alias()
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

            # Match to SimFin: direct ticker, then alias from DB
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
    acc_clean = acc.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/primary_doc.xml"
    xml_data = _edgar_fetch_bytes(url, timeout=60)
    root = ET.fromstring(xml_data)
    ns   = {"n": "http://www.sec.gov/edgar/nport"}
    holdings = []
    for inv in root.findall(".//n:invstOrSec", ns):
        isin_el = inv.find("n:identifiers/n:isin", ns)
        val_el  = inv.find("n:valUSD", ns)
        pct_el  = inv.find("n:pctVal", ns)
        cat_el  = inv.find("n:assetCat", ns)
        isin = isin_el.get("value", "") if isin_el is not None else ""
        if not isin or isin == "N/A":
            continue
        if (cat_el.text if cat_el is not None else "") != "EC":
            continue
        holdings.append({
            "isin":         isin,
            "weight":       float(pct_el.text) if pct_el is not None else None,
            "market_value": float(val_el.text) if val_el is not None else None,
        })
    return holdings


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
    Only companies whose ISIN is in the companies table are included.
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
    acc_clean = acc.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/primary_doc.xml"
    xml_data = _edgar_fetch_bytes(url, timeout=60)
    root = ET.fromstring(xml_data)
    ns = {"n": "http://www.sec.gov/edgar/nport"}
    isins: set[str] = set()
    for inv in root.findall(".//n:invstOrSec", ns):
        cat_el = inv.find("n:assetCat", ns)
        if (cat_el.text if cat_el is not None else "") != "EC":
            continue
        isin_el = inv.find("n:identifiers/n:isin", ns)
        isin = isin_el.get("value", "") if isin_el is not None else ""
        if isin and isin != "N/A":
            isins.add(isin)
    return isins


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
        n_reg   = conn.execute("SELECT COUNT(*) FROM index_registry").fetchone()[0]
        n_acc   = conn.execute("SELECT COUNT(*) FROM nport_accessions").fetchone()[0]
        log.info("DB written: ticker_alias=%d, isin_patch=%d, index_registry=%d, nport_accessions=%d, "
                 "companies=%s, universe_snapshots=%s",
                 n_alias, n_patch, n_reg, n_acc, f"{len(companies):,}", f"{len(snapshots):,}")


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


def ensure_snapshot(snap_date_iso: str) -> None:
    """
    Make `universe_snapshots` current for `snap_date_iso` across all
    auto-discovered indexes (russell_1000 + sp500).

    For each index in _NPORT_AUTO_INDEXES:
      Phase 1 (discovery): if `nport_accessions` lacks a row for
      (index_name, snap_date), query EDGAR for the latest N-PORT-P with
      period_of_report ≤ snap_date and insert it.

    Phase 2 (rebuild): call `rebuild_snapshots()` once to refresh
    `universe_snapshots` from all accessions + CSVs.

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

    rebuild_snapshots()


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


def rebuild_snapshots() -> None:
    """Rebuild universe_snapshots from CSVs + EDGAR N-PORT-P. Does not touch companies."""
    log.info("=== REBUILD UNIVERSE SNAPSHOTS ===")

    with get_db(DB_PATH) as conn:
        companies = pd.read_sql("SELECT isin, ticker FROM companies", conn)
        seed_all_reference_tables(conn)
    known_isins = set(companies["isin"].dropna())
    log.info("Known ISINs from companies table: %d", len(known_isins))

    registry = load_index_registry()

    index_files = sorted(INDEX_DIR.glob("russell_*.csv")) if INDEX_DIR.exists() else []
    ishares_frames: list[tuple[pd.DataFrame, str, str]] = []
    for path in index_files:
        eq, snapshot_date, index_name = load_ishares(path)
        log.info("  %-45s  %5d holdings  (%s @ %s)", path.name, len(eq), index_name, snapshot_date)
        ishares_frames.append((eq, snapshot_date, index_name))

    snapshots_csv = build_snapshots(ishares_frames, companies) if ishares_frames else pd.DataFrame()
    log.info("CSV snapshots: %s rows", f"{len(snapshots_csv):,}")

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
    log.info("Done.")


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
        help="Rebuild universe_snapshots from CSVs + EDGAR only (leaves companies table intact)",
    )
    parser.add_argument(
        "--ensure-snapshot", metavar="YYYY-MM-DD",
        help=(
            "Discover the latest IWB N-PORT-P accession for the given snapshot date "
            "(if missing from nport_accessions), then rebuild universe_snapshots. "
            "Used by daily_update.py weekly cadence."
        ),
    )
    parser.add_argument(
        "--refresh-isins", action="store_true",
        help="Fetch ISINs from FMP for all tickers in universe_index CSVs and write to isin_patch table",
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
        "--limit", type=int, default=None,
        help="Cap the number of names processed (debug; applies to --recover-delisted)",
    )
    args = parser.parse_args()

    if args.rebuild_snapshots:
        rebuild_snapshots()
        return

    if args.ensure_snapshot:
        ensure_snapshot(args.ensure_snapshot)
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

    log.info("=== CREATE UNIVERSE ===")

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
    registry = load_index_registry()
    log.info("  isin_patch: %d overrides | ticker_alias: %d | indexes: %d",
             len(patch), len(alias), len(registry))

    log.info("Building companies table ...")
    companies = build_companies(ishares_frames, simfin, patch=patch, alias=alias)

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
