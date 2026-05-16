import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime

from config import (
    DATA_DIR, SIMFIN_DIR, CONSTITUENTS_DB, UNIVERSE_DB, CONSTITUENTS_REF,
)
from utils import get_db


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


def create_constituents_database():
    print("Creating constituents database...")

    constituents_ref = load_constituents_reference()
    if constituents_ref is None:
        print("constituents_reference.csv not found — aborting.")
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
                    print(f"  {filename} not found — skipping.")
                    continue

                label = "quarterly" if is_quarterly else "annual"
                df = _load_simfin_file(path, is_quarterly)
                print(f"  {stmt_type} {label}: {len(df):,} rows")

                rows = _process_statements(df, stmt_type, constituents_ref, universe, current_date)
                if rows:
                    _insert_rows(conn, rows)
                    total += len(rows)
                    print(f"    → inserted {len(rows):,} records")

        conn.commit()
    print(f"\nDone. Total records: {total:,}")


if __name__ == "__main__":
    create_constituents_database()
