import argparse
import sqlite3
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

from config import (
    DATA_DIR, SIMFIN_DIR, CONSTITUENTS_DB, UNIVERSE_DB, CONSTITUENTS_REF, MACRO_DB,
)
from utils import get_db, get_logger

log = get_logger("create_databases")


def load_constituents_reference():
    if CONSTITUENTS_REF.exists():
        return pd.read_csv(CONSTITUENTS_REF)
    return None


def load_universe():
    with get_db(UNIVERSE_DB) as conn:
        df = pd.read_sql_query("SELECT * FROM companies", conn)
    return df


def _load_simfin_file(path: Path, is_quarterly: bool) -> pd.DataFrame:
    """Read a SimFin CSV. Annual files store Fiscal Year as a date; quarterly as integer."""
    if not path.exists():
        return pd.DataFrame()
    if is_quarterly:
        df = pd.read_csv(path, sep=';', parse_dates=['Report Date', 'Publish Date'])
        df['Fiscal Year'] = df['Fiscal Year'].astype(int)
    else:
        df = pd.read_csv(path, sep=';', parse_dates=['Fiscal Year', 'Report Date', 'Publish Date'])
        df['Fiscal Year'] = df['Fiscal Year'].dt.year
    return df


def _insert_rows(conn, rows):
    conn.executemany('''
        INSERT OR REPLACE INTO constituents
        (data_date, constituent_id, security_id, constituent_value,
         statement_type, report_date, publish_date, available_date,
         update_date, fiscal_year, fiscal_period, currency)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', rows)


def _process_statements(df, stmt_type, stmt_constituents, universe, current_date):
    """Merge df with universe, iterate rows, return insert tuples."""
    df_merged = df.merge(universe[['simfin_id']], left_on='SimFinId', right_on='simfin_id', how='inner')
    rows = []
    type_mapping = {
        'income':   'Income Statement',
        'balance':  'Balance Sheet',
        'cashflow': 'Cash Flow Statement',
    }
    stmt_label = type_mapping[stmt_type]
    relevant = stmt_constituents[stmt_constituents['statement_type'] == stmt_label]

    for _, row in df_merged.iterrows():
        fiscal_year   = int(row['Fiscal Year'])
        fiscal_period = row['Fiscal Period']
        currency      = row['Currency']
        report_date   = row['Report Date'].strftime('%Y-%m-%d') if pd.notna(row['Report Date']) else None
        publish_date  = row['Publish Date'].strftime('%Y-%m-%d') if pd.notna(row['Publish Date']) else None
        available_date = publish_date or report_date

        for _, const in relevant.iterrows():
            col = const['constituent_name']
            if col not in row.index:
                continue
            value = row[col]
            if pd.isna(value) or value == 0:
                continue
            rows.append((
                current_date,
                const['constituent_id'],
                int(row['simfin_id']),
                float(value),
                const['statement_type'],
                report_date,
                publish_date,
                available_date,
                current_date,
                fiscal_year,
                fiscal_period,
                currency,
            ))
    return rows


def _edgar_row_count() -> int:
    """Count EDGAR rows in constituents (security_id = ISIN, starts with a letter)."""
    try:
        with get_db(CONSTITUENTS_DB) as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM constituents WHERE security_id GLOB '[A-Z]*'"
            ).fetchone()[0]
    except Exception:
        return 0  # table doesn't exist yet — safe to create


def create_constituents_database(force: bool = False) -> None:
    edgar_rows = _edgar_row_count()
    if edgar_rows > 0 and not force:
        log.error(
            "constituents.db contains %s EDGAR rows. Re-running will DELETE them "
            "(SimFin rebuild only). Pass --force to override, or use --macro-only "
            "to skip this step entirely.",
            f"{edgar_rows:,}",
        )
        sys.exit(1)

    log.info("Creating constituents database...")

    constituents_ref = load_constituents_reference()
    if constituents_ref is None:
        log.error("constituents_reference.csv not found — aborting.")
        return

    universe     = load_universe()
    current_date = datetime.now().strftime('%Y-%m-%d')

    stmt_files = {
        'income':   ('us-income-annual.csv',   'us-income-quarterly.csv'),
        'balance':  ('us-balance-annual.csv',  'us-balance-quarterly.csv'),
        'cashflow': ('us-cashflow-annual.csv', 'us-cashflow-quarterly.csv'),
    }

    total = 0
    with get_db(CONSTITUENTS_DB) as conn:
        conn.execute('DROP TABLE IF EXISTS constituents')
        conn.execute('''
            CREATE TABLE constituents (
                data_date       TEXT,
                constituent_id  TEXT,
                security_id     TEXT,
                constituent_value REAL,
                statement_type  TEXT,
                report_date     TEXT,
                publish_date    TEXT,
                available_date  TEXT,
                update_date     TEXT,
                fiscal_year     INTEGER,
                fiscal_period   TEXT,
                currency        TEXT,
                PRIMARY KEY (constituent_id, security_id, publish_date)
            )
        ''')

        for stmt_type, (annual_file, quarterly_file) in stmt_files.items():
            for filename, is_quarterly in [(annual_file, False), (quarterly_file, True)]:
                path = SIMFIN_DIR / filename
                if not path.exists():
                    log.warning("%s not found — skipping.", filename)
                    continue

                label = "quarterly" if is_quarterly else "annual"
                df = _load_simfin_file(path, is_quarterly)
                log.info("  %s %s: %s rows", stmt_type, label, f"{len(df):,}")

                rows = _process_statements(df, stmt_type, constituents_ref, universe, current_date)
                if rows:
                    _insert_rows(conn, rows)
                    total += len(rows)
                    log.info("    → inserted %s records", f"{len(rows):,}")

        conn.commit()
    log.info("Done. Total records: %s", f"{total:,}")


def _init_macro_db():
    """Initialize macro.db with signals_reference table and seed initial signals."""
    log.info("Initializing macro database...")

    # Initial seed signals (14 signals: 6 daily rates/spreads, 5 daily commodities/vol, 3 monthly economic)
    signals = [
        ("US10Y", "10-Year Treasury Yield", "FRED", "Rate", "Daily", 0, "%", "US 10-year constant maturity Treasury", "DGS10", 1),
        ("US2Y", "2-Year Treasury Yield", "FRED", "Rate", "Daily", 0, "%", "US 2-year constant maturity Treasury", "DGS2", 1),
        ("US2Y10Y_SPREAD", "2Y10Y Spread", "FRED", "Spread", "Daily", 0, "bps", "10Y - 2Y yield spread", "T10Y2Y", -1),
        ("HY_OAS", "HY OAS", "FRED", "Spread", "Daily", 1, "bps", "High Yield Option-Adjusted Spread", "BAMLH0A0HYM2", -1),
        ("IG_OAS", "IG OAS", "FRED", "Spread", "Daily", 1, "bps", "Investment Grade OAS", "BAMLH0A0HYM2", -1),
        ("VIX", "VIX Index", "CBOE", "Vol", "Daily", 0, "Index", "CBOE Volatility Index", "VIX", -1),
        ("CL_PRICE", "WTI Crude Oil", "Yahoo", "Index", "Daily", 0, "$/bbl", "WTI crude oil futures", "CL=F", 1),
        ("GC_PRICE", "Gold", "Yahoo", "Index", "Daily", 0, "$/oz", "Gold futures", "GC=F", 1),
        ("HG_PRICE", "Copper", "Yahoo", "Index", "Daily", 0, "$/lb", "Copper futures", "HG=F", 1),
        ("USD_INDEX", "DXY USD Index", "Yahoo", "Index", "Daily", 0, "Index", "US Dollar Index", "DXY=F", -1),
        ("CPI_YoY", "CPI YoY Change", "FRED", "Economic", "Monthly", 10, "%", "All Items CPI YoY % Change", "CPIAUCSL", 1),
        ("ISM_MANUF", "ISM Manufacturing", "FRED", "Economic", "Monthly", 1, "Index", "ISM Manufacturing PMI", "MMNRNJ", 1),
        ("ISM_SERVICES", "ISM Services", "FRED", "Economic", "Monthly", 1, "Index", "ISM Services PMI", "ISMCILC", 1),
        ("NONFARM_PAYROLL", "Nonfarm Payroll", "FRED", "Economic", "Monthly", 2, "1000s", "Total Nonfarm Payroll", "PAYEMS", 1),
    ]

    with get_db(MACRO_DB) as conn:
        # Create signals_reference table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS signals_reference (
                signal_id TEXT PRIMARY KEY,
                signal_name TEXT NOT NULL UNIQUE,
                source TEXT NOT NULL,
                data_type TEXT NOT NULL,
                frequency TEXT NOT NULL,
                publication_lag_days INTEGER,
                unit TEXT,
                description TEXT,
                api_endpoint TEXT,
                direction INTEGER DEFAULT 1
            )
        ''')

        # Seed signals if table is empty
        cursor = conn.execute("SELECT COUNT(*) FROM signals_reference")
        if cursor.fetchone()[0] == 0:
            conn.executemany('''
                INSERT INTO signals_reference
                (signal_id, signal_name, source, data_type, frequency,
                 publication_lag_days, unit, description, api_endpoint, direction)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', signals)
            log.info("  Seeded %d signals", len(signals))

        # Create daily_signals table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS daily_signals (
                published_date TEXT NOT NULL,
                signal_id TEXT NOT NULL,
                value REAL NOT NULL,
                update_date TEXT NOT NULL,
                PRIMARY KEY (published_date, signal_id),
                FOREIGN KEY (signal_id) REFERENCES signals_reference(signal_id)
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_signals_published ON daily_signals(published_date)')

        conn.commit()

    log.info("Macro database initialized.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Initialise pipeline databases.")
    ap.add_argument(
        "--macro-only", action="store_true",
        help="Initialise macro.db only — skip constituents rebuild (safe to re-run).",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Rebuild constituents from SimFin even if EDGAR rows exist (destructive).",
    )
    args = ap.parse_args()

    if not args.macro_only:
        create_constituents_database(force=args.force)
    _init_macro_db()
