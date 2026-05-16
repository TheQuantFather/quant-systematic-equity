"""
validate_constituents.py — Post-fill validation for constituents data.

Checks:
  1. Coverage by fiscal year (how many universe companies have each key metric)
  2. Scale consistency — no >5x jumps between consecutive years per company
  3. SimFin/EDGAR boundary — FY2023→FY2024 continuity for companies with both sources
  4. Known reference values — spot-check AAPL, MSFT, JPM vs published reports
  5. Factors rebuild preview — count of Leverage rows before/after ST Debt fix
"""

import sqlite3
import pandas as pd
import numpy as np

from config import CONSTITUENTS_DB as CONST_DB, UNIVERSE_DB as UNIV_DB, CONSTITUENTS_REF as CONST_REF
from utils import get_db

# Key constituents to check (name → constituent_id)
KEY_FIELDS = {
    "Revenue":                     "9801FC7E",
    "Net Income":                  "CDD1D338",
    "Total Assets":                "3BD29B6F",
    "Total Equity":                "06EF64B2",
    "Total Liabilities":           "3B25F87A",
    "Net Cash from Operating":     "6835500D",
    "Change in CapEx":             "CA8A4027",
    "Shares (Basic)":              "B3C4D5E6",
    "Gross Profit":                "7A1B2BB6",
    "Operating Income":            "80C2558A",
    "Total Current Assets":        "3F897E82",
    "Total Current Liabilities":   "2B0918F0",
    "Short Term Debt":             "2D2B0CAC",
    "Long Term Debt":              "D7815EBF",
    "D&A (Cash Flow)":             "E7754E82",
}

UNIVERSE_TICKERS = {"AAPL", "MSFT", "JPM", "NVDA", "XOM", "JNJ", "AMZN", "META", "GOOG", "BRK.B"}

# Published reference values (in billions USD) for sanity check
KNOWN_VALUES = {
    # (ticker, fiscal_year, field): expected_value_in_USD
    ("AAPL", 2024, "Revenue"):        391_035_000_000,
    ("MSFT", 2024, "Revenue"):        245_122_000_000,
    ("AAPL", 2024, "Total Assets"):   364_980_000_000,
    ("JPM",  2024, "Total Assets"): 4_003_000_000_000,
}

# Updated field → constituent_id mapping for known values check
_KNOWN_VALUE_ID_MAP = {
    "Revenue":      "9801FC7E",
    "Total Assets": "3BD29B6F",
}


def check_coverage(conn, univ_isins: set[str]) -> None:
    print("\n" + "="*60)
    print("1. COVERAGE BY FISCAL YEAR")
    print("="*60)

    ref = pd.read_csv(CONST_REF)
    id_to_name = dict(zip(ref["constituent_id"], ref["constituent_name"]))

    key_ids = list(KEY_FIELDS.values())

    for fy in range(2019, 2026):
        rows = conn.execute("""
            SELECT COUNT(DISTINCT security_id)
            FROM constituents
            WHERE fiscal_year = ? AND fiscal_period IN ('FY','Q4')
              AND security_id IN ({})
        """.format(",".join("?" * len(univ_isins))),
            [fy] + list(univ_isins)
        ).fetchone()
        n_any = rows[0]

        n_rev = conn.execute("""
            SELECT COUNT(DISTINCT security_id) FROM constituents
            WHERE fiscal_year=? AND constituent_id='9801FC7E'
              AND security_id IN ({})
        """.format(",".join("?" * len(univ_isins))),
            [fy] + list(univ_isins)
        ).fetchone()[0]

        n_assets = conn.execute("""
            SELECT COUNT(DISTINCT security_id) FROM constituents
            WHERE fiscal_year=? AND constituent_id='3BD29B6F'
              AND security_id IN ({})
        """.format(",".join("?" * len(univ_isins))),
            [fy] + list(univ_isins)
        ).fetchone()[0]

        n = len(univ_isins)
        print(f"  FY{fy}: {n_any:>4}/{n} any  |  {n_rev:>4} revenue  |  {n_assets:>4} total assets")


def check_scale_jumps(conn, univ_isins: set[str]) -> None:
    print("\n" + "="*60)
    print("2. SCALE CONSISTENCY (>5x year-over-year jumps)")
    print("="*60)

    for name, cid in [("Revenue", "9801FC7E"), ("Total Assets", "3BD29B6F"), ("Net Income", "CDD1D338")]:
        df = pd.read_sql_query("""
            SELECT security_id, fiscal_year, constituent_value
            FROM constituents
            WHERE constituent_id = ?
              AND fiscal_period IN ('FY','Q4')
              AND fiscal_year BETWEEN 2019 AND 2025
            ORDER BY security_id, fiscal_year
        """, conn, params=(cid,))

        if df.empty:
            continue

        df["prev"] = df.groupby("security_id")["constituent_value"].shift(1)
        df = df.dropna(subset=["prev"])
        df = df[df["prev"].abs() > 1e6]  # ignore near-zero denominators
        df["ratio"] = df["constituent_value"].abs() / df["prev"].abs()
        jumps = df[(df["ratio"] > 5) | (df["ratio"] < 0.2)].copy()

        if jumps.empty:
            print(f"  {name}: no scale jumps ✓")
        else:
            with get_db(UNIV_DB) as uc:
                isin_to_ticker = dict(uc.execute("SELECT isin, ticker FROM companies").fetchall())
            jumps["ticker"] = jumps["security_id"].map(isin_to_ticker)
            jumps = jumps[jumps["security_id"].isin(univ_isins)]
            print(f"  {name}: {len(jumps)} jumps in universe companies:")
            for _, r in jumps.iterrows():
                print(f"    {r['ticker'] or r['security_id']}: FY{int(r['fiscal_year'])}  "
                      f"prev={r['prev']/1e9:.1f}B  curr={r['constituent_value']/1e9:.1f}B  "
                      f"ratio={r['ratio']:.1f}x")


def check_source_boundary(conn, univ_isins: set[str]) -> None:
    print("\n" + "="*60)
    print("3. EDGAR/SimFin BOUNDARY (FY2023 → FY2024)")
    print("="*60)

    with get_db(UNIV_DB) as uc:
        isin_to_ticker = dict(uc.execute("SELECT isin, ticker FROM companies").fetchall())

    for name, cid in [("Revenue", "9801FC7E"), ("Total Assets", "3BD29B6F")]:
        df = pd.read_sql_query("""
            SELECT security_id,
              MAX(CASE WHEN fiscal_year=2023 THEN constituent_value END) AS v2023,
              MAX(CASE WHEN fiscal_year=2024 THEN constituent_value END) AS v2024
            FROM constituents
            WHERE constituent_id=? AND fiscal_period IN ('FY','Q4')
              AND fiscal_year IN (2023,2024)
            GROUP BY security_id
            HAVING v2023 IS NOT NULL AND v2024 IS NOT NULL
        """, conn, params=(cid,))

        df = df[df["security_id"].isin(univ_isins)].copy()
        if df.empty:
            continue

        df["ratio"] = df["v2024"].abs() / df["v2023"].abs().replace(0, np.nan)
        df = df.dropna(subset=["ratio"])
        boundary_jumps = df[(df["ratio"] > 5) | (df["ratio"] < 0.2)]
        df["ticker"] = df["security_id"].map(isin_to_ticker)

        median_growth = df["ratio"].median() - 1
        print(f"  {name}: {len(df)} companies with both years | "
              f"median YoY growth={median_growth:.1%} | {len(boundary_jumps)} boundary jumps")
        if not boundary_jumps.empty:
            print("  Boundary jumps:")
            for _, r in boundary_jumps.iterrows():
                print(f"    {r['ticker']}: {r['v2023']/1e9:.1f}B → {r['v2024']/1e9:.1f}B  ({r['ratio']:.1f}x)")


def check_known_values(conn) -> None:
    print("\n" + "="*60)
    print("4. KNOWN REFERENCE VALUES")
    print("="*60)

    with get_db(UNIV_DB) as uc:
        ticker_to_isin = dict(uc.execute("SELECT ticker, isin FROM companies WHERE ticker IS NOT NULL").fetchall())

    field_to_id = _KNOWN_VALUE_ID_MAP

    for (ticker, fy, field), expected in KNOWN_VALUES.items():
        isin = ticker_to_isin.get(ticker)
        if not isin:
            print(f"  {ticker} FY{fy} {field}: ISIN not found")
            continue
        cid = field_to_id.get(field)
        if not cid:
            continue
        row = conn.execute("""
            SELECT constituent_value FROM constituents
            WHERE security_id=? AND constituent_id=? AND fiscal_year=?
              AND fiscal_period IN ('FY','Q4')
        """, (isin, cid, fy)).fetchone()
        if not row:
            print(f"  {ticker} FY{fy} {field}: NOT FOUND")
        else:
            actual = row[0]
            pct_err = abs(actual - expected) / expected * 100
            status = "✓" if pct_err < 5 else "WARN" if pct_err < 15 else "ERROR"
            print(f"  {ticker} FY{fy} {field}: actual={actual/1e9:.1f}B  expected={expected/1e9:.1f}B  err={pct_err:.1f}%  {status}")


def check_gaps_remaining(conn, univ_isins: set[str]) -> None:
    print("\n" + "="*60)
    print("5. REMAINING GAPS SUMMARY")
    print("="*60)

    for fy in [2019, 2020, 2024, 2025]:
        have = conn.execute("""
            SELECT COUNT(DISTINCT security_id) FROM constituents
            WHERE fiscal_year=? AND fiscal_period IN ('FY','Q4')
              AND security_id IN ({})
        """.format(",".join("?" * len(univ_isins))),
            [fy] + list(univ_isins)
        ).fetchone()[0]
        print(f"  FY{fy}: {have}/{len(univ_isins)} universe companies have data "
              f"({100*have/len(univ_isins):.0f}%)")

    # Which companies still have zero data
    have_any = set(r[0] for r in conn.execute("""
        SELECT DISTINCT security_id FROM constituents
        WHERE fiscal_period IN ('FY','Q4')
    """).fetchall())
    missing = univ_isins - have_any
    with get_db(UNIV_DB) as uc:
        isin_to_ticker = dict(uc.execute("SELECT isin, ticker FROM companies").fetchall())
    if missing:
        tickers = [isin_to_ticker.get(i, i) for i in sorted(missing)]
        print(f"\n  Companies with NO data at all ({len(missing)}): {', '.join(tickers[:30])}"
              + (" ..." if len(tickers) > 30 else ""))
    else:
        print("\n  All universe companies have at least some data ✓")


def main():
    with get_db(UNIV_DB) as uc:
        univ_isins = set(r[0] for r in uc.execute(
            "SELECT isin FROM companies WHERE isin IS NOT NULL"
        ).fetchall())

    print(f"Universe: {len(univ_isins)} companies")

    with get_db(CONST_DB) as conn:
        check_coverage(conn, univ_isins)
        check_scale_jumps(conn, univ_isins)
        check_source_boundary(conn, univ_isins)
        check_known_values(conn)
        check_gaps_remaining(conn, univ_isins)

    print("\nValidation complete.")


if __name__ == "__main__":
    main()
