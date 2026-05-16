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
import urllib.request
import xml.etree.ElementTree as ET

import pandas as pd
import sqlite3
from pathlib import Path
from datetime import datetime

from config import DATA_DIR, SIMFIN_DIR, UNIVERSE_DB as DB_PATH
from utils import get_db

INDEX_DIR  = DATA_DIR / "universe_index"

ISHARES_SKIPROWS = 9

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

# iShares ticker → SimFin ticker, for matching metadata only.
# The companies table stores the iShares ticker (market standard).
_SIMFIN_MATCH: dict[str, str] = {
    "BRKB":  "BRK-A",
    "BRKA":  "BRK-A",
    "GOOGL": "GOOG",
    "FOXA":  "FOX",
    "NWSA":  "NWS",
    "UAA":   "UA",
    "LENB":  "LEN",
    "FISV":  "FI",    # iShares uses FISV, SimFin has FI (post-rebrand ticker)
    "CPAY":  "CCCC",  # Corpay — SimFin may not have this yet
    "RVTY":  "PKI",   # Revvity (formerly PerkinElmer)
    "CPAY":  "FLYW",  # Corpay (formerly Flywire? actually formerly Comdata - check)
}

# Known ISINs for companies not matched via SimFin ticker lookup.
# Sourced from iShares Russell 1000 NPORT-P (EDGAR acc 0001004726-26-000805,
# period 2025-12-31) and SEC EDGAR public filings.
_ISIN_PATCH: dict[str, str] = {
    # --- Pre-existing entries (SimFin gaps, renames, non-US incorporations) ---
    "AS":    "US03014X1037",  # Amer Sports Inc
    "CAVA":  "US14943H2085",  # CAVA Group Inc
    "COHR":  "US19247G1076",  # Coherent Corp (formerly II-VI)
    "DINO":  "US40637H1095",  # HF Sinclair Corp
    "DRS":   "US52605T1007",  # Leonardo DRS Inc
    "EG":    "BMG3223R1088",  # Everest Group Ltd (Bermuda)
    "FERG":  "US31482P1003",  # Ferguson Enterprises Inc (US-listed)
    "FLUT":  "IE00BWT6H894",  # Flutter Entertainment plc (Ireland)
    "FRMI":  "US31488E1082",  # Fermi Inc
    "GEHC":  "US36266G1076",  # GE HealthCare Technologies Inc
    "GEN":   "US37290D1054",  # Gen Digital Inc
    "KVUE":  "US49177J1025",  # Kenvue Inc
    "LINE":  "US53567M1071",  # Lineage Inc (REIT, IPO 2024)
    "NIQ":   "IE000S971575",  # NIQ Global Intelligence plc
    "NTRS":  "US6658591044",  # Northern Trust Corp
    "P":     "US30040W1018",  # Everpure Inc Class A
    "PR":    "US71406T1079",  # Permian Resources Corp
    "RITM":  "US77495W1027",  # Rithm Capital Corp
    "SN":    "US82028M1018",  # SharkNinja Inc
    "SNDK":  "US8001991010",  # SanDisk Corp (relisted 2024)
    "FISV":  "US3377381088",  # Fiserv Inc (iShares uses FISV, SimFin uses FI)
    "HOLX":  "US4364401012",  # Hologic Inc (iShares file has blank ticker row)
    # --- Sourced from IWB NPORT-P (EDGAR, period 2025-12-31) ---
    "ALAB":  "US04626A1034",  # Astera Labs Inc
    "AM":    "US03676B1026",  # Antero Midstream Corp
    "AMTM":  "US0239391016",  # Amentum Holdings Inc
    "AU":    "GB00BRXH2664",  # AngloGold Ashanti plc (UK-listed)
    "BFA":   "US1156371007",  # Brown-Forman Corp Class A
    "BFB":   "US1156372096",  # Brown-Forman Corp Class B
    "BIRK":  "JE00BS44BN30",  # Birkenstock Holding plc (Jersey)
    "BLSH":  "KYG169101204",  # Bullish (Cayman)
    "CART":  "US5653941030",  # Maplebear Inc (Instacart)
    "CCC":   "US12510Q1004",  # CCC Intelligent Solutions Holdings Inc
    "CHRD":  "US6742152076",  # Chord Energy Corp
    "CNH":   "NL0010545661",  # CNH Industrial N.V. (Netherlands)
    "CRCL":  "US1725731079",  # Circle Internet Group Inc
    "CXT":   "US2244411052",  # Crane NXT Co
    "DJT":   "US25400Q1058",  # Trump Media & Technology Group Corp
    "ECG":   "US3004261034",  # Everus Construction Group Inc
    "FBIN":  "US34964C1062",  # Fortune Brands Innovations Inc
    "FWONA": "US5312297550",  # Liberty Formula One Group Series A
    "FWONK": "US5312297717",  # Liberty Formula One Group Series C
    "GAP":   "US3647601083",  # The Gap Inc
    "GEV":   "US36828A1016",  # GE Vernova Inc
    "GLIBA": "US36164V8000",  # GCI Liberty Inc Series A
    "GLIBK": "US36164V8000",  # GCI Liberty Inc Series C (same ISIN)
    "GTM":   "US98980F1049",  # ZoomInfo Technologies Inc
    "HEIA":  "US4228061093",  # HEICO Corp Class A (same ISIN as HEICO Corp)
    "HHH":   "US44267T1025",  # Howard Hughes Holdings Inc
    "INGM":  "US4571521065",  # Ingram Micro Holding Corp
    "IOT":   "US79589L1061",  # Samsara Inc Class A
    "KRMN":  "US4859241048",  # Karman Holdings Inc
    "LBRDK": "US5303073051",  # Liberty Broadband Corp Series C
    "LBTYK": "BMG611881019",  # Liberty Global Ltd Class C (Bermuda)
    "LLYVA": "US5309091008",  # Liberty Live Holdings Inc Series A
    "LLYVK": "US5309093087",  # Liberty Live Holdings Inc Series C
    "LOAR":  "US53947R1059",  # Loar Holdings Inc
    "MPT":   "US58463J3041",  # Medical Properties Trust Inc
    "MRP":   "US6011371027",  # Millrose Properties Inc
    "MRSH":  "US5717481023",  # Marsh & McLennan Companies Inc
    "ONTO":  "US6833441057",  # Onto Innovation Inc
    "PRMB":  "US7416231022",  # Primo Brands Corp
    "Q":     "US74743L1008",  # Qnity Electronics Inc
    "QXO":   "US82846H4056",  # QXO Inc
    "RAL":   "US7509401086",  # Ralliant Corp
    "RBC":   "US75524B1044",  # RBC Bearings Inc
    "RBRK":  "US7811541090",  # Rubrik Inc Class A
    "SARO":  "US85423L1035",  # StandardAero Inc
    "SGI":   "US88023U1016",  # Somnigroup International Inc
    "SOLS":  "US83443Q1031",  # Solstice Advanced Materials Inc
    "SOLV":  "US83444M1018",  # Solventum Corp
    "SW":    "IE00028FXN24",  # Smurfit WestRock plc (Ireland)
    "TEM":   "US88023B1035",  # Tempus AI Inc Class A
    "TKO":   "US87256C1018",  # TKO Group Holdings Inc
    "TLN":   "US87422Q1094",  # Talen Energy Corp
    "UHALB": "US0235861004",  # U-Haul Holding Series N
    "VIK":   "BMG93A5A1010",  # Viking Holdings Ltd (Bermuda)
    "VLTO":  "US92338C1036",  # Veralto Corp
    "XYZ":   "US8522341036",  # Block Inc Class A
    "ZG":    "US98954M1018",  # Zillow Group Inc Class A
}


# Historical iShares Russell 1000 ETF N-PORT-P filings (period → EDGAR accession).
# Each entry maps an April 1 snapshot date to the December-period filing available
# ~Feb of that year.  Add new entries when new annual data is needed.
_HISTORICAL_NPORT: dict[str, str] = {
    "2021-04-01": "0001752724-21-040717",   # period 2020-12-31
    "2022-04-01": "0001752724-22-046365",   # period 2021-12-31
    "2023-04-01": "0001752724-23-039564",   # period 2022-12-31
    "2024-04-01": "0001752724-24-034803",   # period 2023-12-31
    "2025-04-01": "0001752724-25-034052",   # period 2024-12-31
}


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


def _parse_ishares_date(path: Path) -> str:
    """Extract snapshot date string (YYYY-MM-DD) from iShares CSV header."""
    with open(path, encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            if i == 1:                          # row 2: 'Fund Holdings as of,"May 04, 2026"'
                parts = line.strip().split(",")
                date_str = " ".join(p.strip().strip('"') for p in parts[1:])
                return datetime.strptime(date_str, "%B %d %Y").strftime("%Y-%m-%d")
    raise ValueError(f"Could not parse date from {path}")


def _infer_index_name(path: Path) -> str:
    """
    Infer the index name from the filename.
    Expects filenames like:  russell_1000_2026_05_04.csv
    Falls back to 'unknown_index' if the pattern isn't recognised.
    """
    stem  = path.stem.lower()                          # russell_1000_2026_05_04
    parts = stem.split("_")
    # Strip trailing date parts: break on a 4-digit calendar year (>= 1900)
    name_parts = []
    for p in parts:
        if len(p) == 4 and p.isdigit() and int(p) >= 1900:
            break
        name_parts.append(p)
    return "_".join(name_parts) if name_parts else "unknown_index"


def load_ishares(path: Path) -> tuple[pd.DataFrame, str, str]:
    """
    Parse an iShares holdings CSV.
    Returns (equity_df, snapshot_date_str, index_name).
    """
    snapshot_date = _parse_ishares_date(path)
    index_name    = _infer_index_name(path)

    df = pd.read_csv(path, skiprows=ISHARES_SKIPROWS, encoding="utf-8-sig")
    eq = df[df["Asset Class"] == "Equity"].copy()

    for col in ["Market Value", "Weight (%)", "Price"]:
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
) -> pd.DataFrame:
    """
    Build the companies table from all iShares snapshots merged with SimFin.
    One row per unique iShares ticker (deduped by ISIN, latest snapshot wins).
    """
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

            # Match to SimFin: direct ticker, then alias
            sf = sf_by_ticker.get(ticker)
            if sf is None:
                sf = sf_by_ticker.get(_SIMFIN_MATCH.get(ticker, ""))

            # Resolve ISIN
            isin: str | None = None
            if sf is not None and pd.notna(sf.get("isin")) and str(sf["isin"]).strip():
                isin = str(sf["isin"]).strip()
            if not isin:
                isin = _ISIN_PATCH.get(ticker)
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
) -> pd.DataFrame:
    """Build universe_snapshots from all iShares frames."""
    isin_by_ticker: dict[str, str] = dict(zip(companies["ticker"], companies["isin"]))

    rows = []
    for eq, snapshot_date, index_name in ishares_frames:
        seen: set[tuple] = set()
        for _, ih in eq.iterrows():
            ticker = str(ih["Ticker"]).upper().strip()
            isin   = isin_by_ticker.get(ticker)
            if isin is None:
                alias = _SIMFIN_MATCH.get(ticker)
                if alias:
                    isin = isin_by_ticker.get(alias)
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

def build_historical_snapshots(known_isins: set[str]) -> pd.DataFrame:
    """
    Fetch historical IWB N-PORT-P filings from EDGAR and return a DataFrame
    of universe_snapshots rows (one per company per snapshot date).

    Only companies whose ISIN is already in the companies table are included —
    historical-only members (since acquired / delisted) are skipped.
    """
    rows = []
    for snap_date, acc in sorted(_HISTORICAL_NPORT.items()):
        acc_clean = acc.replace('-', '')
        xml_url = (
            f"https://www.sec.gov/Archives/edgar/data/1100663/{acc_clean}/primary_doc.xml"
        )
        try:
            xml_data = _edgar_fetch_bytes(xml_url, timeout=60)
            root     = ET.fromstring(xml_data)
        except Exception as e:
            print(f"  [NPORT] {snap_date}: fetch error — {e}")
            continue

        ns = {"n": "http://www.sec.gov/edgar/nport"}
        n_matched = 0
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
            if isin not in known_isins:
                continue

            rows.append({
                "snapshot_date": snap_date,
                "isin":          isin,
                "index_name":    "russell_1000",
                "weight":        float(pct_el.text) if pct_el is not None else None,
                "market_value":  float(val_el.text) if val_el is not None else None,
            })
            n_matched += 1

        print(f"  {snap_date}: {n_matched} companies")
        time.sleep(0.3)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# EDGAR metadata enrichment
# ---------------------------------------------------------------------------

_EDGAR_HEADERS = {"User-Agent": os.getenv("EDGAR_IDENTITY", "your-name your@email.com")}

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
        print(f"  [EDGAR] Could not fetch company_tickers.json: {e}")
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

    print(f"  [EDGAR] Looking up CIKs for {len(missing)} companies ...")
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

    print(f"  [EDGAR] Filled CIK for {filled} / {len(missing)} missing companies")
    return companies


# ---------------------------------------------------------------------------
# DB writer
# ---------------------------------------------------------------------------

def write_db(companies: pd.DataFrame, snapshots: pd.DataFrame) -> None:
    with get_db(DB_PATH) as conn:
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
                update_date         TEXT
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

        companies.to_sql("companies", conn, if_exists="append", index=False)
        snapshots.to_sql("universe_snapshots", conn, if_exists="append", index=False)
        conn.commit()
        print(f"  companies:          {len(companies):,} rows")
        print(f"  universe_snapshots: {len(snapshots):,} rows")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(companies: pd.DataFrame, snapshots: pd.DataFrame) -> None:
    total     = len(companies)
    synthetic = companies["isin"].str.startswith("NOISN_").sum()
    patched   = companies["isin"].isin(_ISIN_PATCH.values()).sum()
    simfin_m  = companies["simfin_id"].notna().sum()
    no_cik    = companies["cik"].isna().sum()

    print(f"\n{'='*55}")
    print(f"COMPANIES TABLE  ({total:,} securities)")
    print(f"{'='*55}")
    print(f"  ISIN from SimFin:    {total - synthetic - patched:>4}")
    print(f"  ISIN from patch:     {patched:>4}  (hardcoded)")
    print(f"  Synthetic ISIN:      {synthetic:>4}  (need lookup)")
    if synthetic:
        synt = companies[companies["isin"].str.startswith("NOISN_")][["ticker","company_name"]].values
        for t, n in synt:
            print(f"    {t:<12} {n}")
    print(f"  SimFin metadata:     {simfin_m:>4} / {total}")
    print(f"  Missing CIK:         {no_cik:>4}  (non-US or new listings)")

    print(f"\n{'='*55}")
    print(f"UNIVERSE SNAPSHOTS")
    print(f"{'='*55}")
    for (idx, dt), grp in snapshots.groupby(["index_name", "snapshot_date"]):
        print(f"  {idx:<20} {dt}   {len(grp):>5} companies")

    print(f"\nGICS sector breakdown (all companies):")
    for sector, cnt in companies["gics_sector"].value_counts().items():
        print(f"  {sector:<30} {cnt:>4}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 55)
    print("CREATE UNIVERSE")
    print("=" * 55)

    print("\nLoading SimFin data ...")
    simfin = load_simfin()
    print(f"  {len(simfin):,} companies loaded")

    index_files = sorted(INDEX_DIR.glob("*.csv")) if INDEX_DIR.exists() else []
    if not index_files:
        print("[ERROR] No CSV files found in data/universe_index/")
        return

    print(f"\nLoading {len(index_files)} index file(s) ...")
    ishares_frames: list[tuple[pd.DataFrame, str, str]] = []
    for path in index_files:
        eq, snapshot_date, index_name = load_ishares(path)
        print(f"  {path.name:<45}  {len(eq):>5} holdings  ({index_name} @ {snapshot_date})")
        ishares_frames.append((eq, snapshot_date, index_name))

    print("\nBuilding companies table ...")
    companies = build_companies(ishares_frames, simfin)

    print("Enriching metadata from EDGAR ...")
    companies = enrich_edgar_metadata(companies)

    print("Building universe_snapshots table ...")
    snapshots = build_snapshots(ishares_frames, companies)

    print("Fetching historical universe snapshots from EDGAR N-PORT-P ...")
    known_isins = set(companies["isin"].dropna())
    hist = build_historical_snapshots(known_isins)
    if not hist.empty:
        snapshots = pd.concat([snapshots, hist], ignore_index=True)

    print(f"\nWriting to {DB_PATH} ...")
    write_db(companies, snapshots)

    print_report(companies, snapshots)

    print("\n" + "=" * 55)
    print("Done.")
    print("=" * 55)


if __name__ == "__main__":
    main()
