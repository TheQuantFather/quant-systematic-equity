"""
create_models.py — Build factor models by combining pre-computed z-scores from factors.db.

Models: Quality, Value, Growth, Momentum, Size (base models) and Alpha (composite).

Factor z-scores are read from factors.db (already winsorized cross-sectionally per
data_date/factor).  Model scores are also winsorized and z-scored within each
(data_date, model_id) cross-section so all model scores are on a comparable scale.

Usage:
  python create_models.py              # process all dates in factors.db
  python create_models.py --date DATE  # single date only

Run after create_factors.py.
"""

import argparse
import csv
import math
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd

from config import FACTORS_DB, MODELS_DB, MODELS_REF as MODELS_CSV, FACTORS_REF as FACTORS_CSV
from utils import get_db, winsorized_zscore


# ---------------------------------------------------------------------------
# Reference loading
# ---------------------------------------------------------------------------

def load_factor_directions() -> dict:
    """Returns {factor_id: direction (1 or -1)} from factors_reference.csv."""
    df = pd.read_csv(FACTORS_CSV)
    return dict(zip(df['factor_id'], df['direction'].astype(int)))


def load_models_reference() -> dict:
    """
    Returns {model_name: {model_id, is_composite, weights: {factor_id: weight}}}.
    """
    models: dict = {}
    with open(MODELS_CSV) as f:
        for row in csv.DictReader(f):
            name         = row['Model']
            model_id     = row['ModelID']
            factor_id    = row['Factors']
            weight       = float(row['Weights'])
            is_composite = bool(int(row['IsComposite']))
            if name not in models:
                models[name] = {'model_id': model_id, 'is_composite': is_composite, 'weights': {}}
            models[name]['weights'][factor_id] = weight
    return models


def get_base_models(models: dict) -> dict:
    model_ids = {info['model_id'] for info in models.values()}
    return {n: m for n, m in models.items()
            if not any(f in model_ids for f in m['weights'])}


def get_alpha_models(models: dict) -> dict:
    model_ids = {info['model_id'] for info in models.values()}
    return {n: m for n, m in models.items()
            if any(f in model_ids for f in m['weights'])}


# ---------------------------------------------------------------------------
# Factor data loading
# ---------------------------------------------------------------------------

def load_factors_zscores(date_filter: str = None) -> dict:
    """
    Returns {data_date: {security_id: {factor_id: z_score}}}.

    date_filter — if provided, only load rows for that specific date.
    """
    with get_db(FACTORS_DB) as conn:
        if date_filter:
            rows = conn.execute(
                "SELECT data_date, security_id, factor_id, factor_value_z "
                "FROM factors WHERE factor_value_z IS NOT NULL AND data_date = ?",
                (date_filter,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT data_date, security_id, factor_id, factor_value_z "
                "FROM factors WHERE factor_value_z IS NOT NULL"
            ).fetchall()

    data: dict = {}
    for data_date, security_id, factor_id, z in rows:
        security_id = str(security_id)
        data.setdefault(data_date, {}).setdefault(security_id, {})[factor_id] = z
    return data


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def setup_models_db(conn: sqlite3.Connection, clean: bool = False) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    existing = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='models'"
    ).fetchone()
    if existing:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(models)").fetchall()]
        if 'fiscal_year' in cols or clean:
            reason = "old schema (had fiscal_year column)" if 'fiscal_year' in cols else "--clean flag"
            print(f"[INFO] Dropping models table ({reason}) — will rebuild from scratch")
            conn.execute("DROP TABLE models")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS models (
            data_date     TEXT    NOT NULL,
            model_id      TEXT    NOT NULL,
            security_id   TEXT    NOT NULL,
            model_value   REAL    NOT NULL,
            model_value_z REAL,
            is_composite  INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (data_date, model_id, security_id)
        )
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Model computation
# ---------------------------------------------------------------------------

def compute_base_models(
    conn: sqlite3.Connection,
    base_models: dict,
    factors_data: dict,
    factor_directions: dict,
) -> int:
    """Score each base model as a weighted sum of direction-adjusted factor z-scores."""
    rows = []
    for data_date, securities in factors_data.items():
        for model_name, model_info in base_models.items():
            model_id     = model_info['model_id']
            is_composite = int(model_info['is_composite'])
            weights      = model_info['weights']
            for security_id, factor_zscores in securities.items():
                score        = 0.0
                valid_weight = 0.0
                for factor_id, weight in weights.items():
                    z = factor_zscores.get(factor_id)
                    if z is not None and not math.isnan(z):
                        direction     = factor_directions.get(factor_id, 1)
                        score        += z * weight * direction
                        valid_weight += weight
                if valid_weight > 0:
                    rows.append((data_date, model_id, security_id, score, is_composite))

    conn.executemany(
        "INSERT OR REPLACE INTO models "
        "(data_date, model_id, security_id, model_value, is_composite) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    # No commit here — caller commits after alpha models and z-scores are also written,
    # so base + alpha + z-score updates land in one atomic transaction.
    return len(rows)


def compute_alpha_models(
    conn: sqlite3.Connection,
    alpha_models: dict,
    factors_data: dict,
) -> int:
    """Score composite models as a weighted sum of base model scores."""
    rows = []
    for data_date, securities in factors_data.items():
        base_scores: dict = {}
        for sid, model_id, model_value in conn.execute(
            "SELECT security_id, model_id, model_value FROM models "
            "WHERE data_date = ? AND is_composite = 0",
            (data_date,),
        ).fetchall():
            sid = str(sid)
            base_scores.setdefault(sid, {})[model_id] = model_value

        for model_name, model_info in alpha_models.items():
            model_id     = model_info['model_id']
            is_composite = int(model_info['is_composite'])
            weights      = model_info['weights']
            for security_id in securities:
                sid_scores   = base_scores.get(security_id, {})
                score        = 0.0
                valid_weight = 0.0
                for base_model_id, weight in weights.items():
                    v = sid_scores.get(base_model_id)
                    if v is not None and not math.isnan(v):
                        score        += v * weight
                        valid_weight += weight
                if valid_weight > 0:
                    rows.append((data_date, model_id, security_id, score, is_composite))

    conn.executemany(
        "INSERT OR REPLACE INTO models "
        "(data_date, model_id, security_id, model_value, is_composite) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    # No commit here — caller commits once after z-scores are written.
    return len(rows)


def compute_model_zscores(conn: sqlite3.Connection) -> None:
    """
    Add cross-sectional winsorized z-scores to model_value_z.
    Grouped by (data_date, model_id) so each snapshot's scores are normalised
    within their own peer group.
    """
    df = pd.read_sql_query(
        "SELECT rowid, data_date, model_id, model_value FROM models", conn
    )
    df['model_value_z'] = (
        df.groupby(['data_date', 'model_id'])['model_value']
        .transform(winsorized_zscore)
    )
    conn.executemany(
        "UPDATE models SET model_value_z = ? WHERE rowid = ?",
        df[['model_value_z', 'rowid']].itertuples(index=False, name=None),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute model scores from factor z-scores and write to models.db"
    )
    parser.add_argument('--date', metavar='YYYY-MM-DD',
                        help='Process a single date only (default: all dates in factors.db)')
    parser.add_argument('--clean', action='store_true',
                        help='Drop and rebuild the models table before running')
    args = parser.parse_args()

    models            = load_models_reference()
    base_models       = get_base_models(models)
    alpha_models      = get_alpha_models(models)
    factor_directions = load_factor_directions()

    print(f"Models to create : {list(models.keys())}")
    print(f"Base models      : {list(base_models.keys())}")
    print(f"Alpha models     : {list(alpha_models.keys())}")

    print("\nLoading pre-computed z-scores from factors.db ...")
    factors_data = load_factors_zscores(date_filter=args.date)
    dates = sorted(factors_data.keys())
    print(f"Loaded z-scores for {len(dates)} date(s): {dates}")

    with get_db(MODELS_DB) as conn:
        setup_models_db(conn, clean=args.clean)

        print("\n=== Stage 1: Base models from factor z-scores ===")
        n1 = compute_base_models(conn, base_models, factors_data, factor_directions)
        print(f"  {n1:,} base model records written")

        if alpha_models:
            print("\n=== Stage 2: Alpha models from base model scores ===")
            n2 = compute_alpha_models(conn, alpha_models, factors_data)
            print(f"  {n2:,} alpha model records written")
        else:
            n2 = 0

        print("\nComputing cross-sectional z-scores for model scores ...")
        compute_model_zscores(conn)  # commits inside here

    print(f"\nDone — {n1 + n2:,} total records | {len(models)} models × {len(dates)} date(s)")


if __name__ == "__main__":
    main()
