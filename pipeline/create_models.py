"""
create_models.py — Build factor models by combining pre-computed z-scores from factors.db.

Models: Profitability, Defensive Quality, Value, Growth, Momentum, Size (base models) and Alpha (composite).

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


from config import (
    FACTORS_DB, MODELS_DB, UNIVERSE_DB,
    MODELS_REF as MODELS_CSV, FACTORS_REF as FACTORS_CSV,
)
from utils import (
    get_db, get_logger, winsorized_zscore,
    classify_sector, factor_applies_to_company,
)

log = get_logger("create_models")

# A model score is renormalised by the weight of factors actually present, but the
# divisor is floored at MIN_COVERAGE × (weight applicable to the security's sector).
# So a name with ≥50% coverage gets full conviction, while a sparsely-covered name
# is divided by the floor — shrinking its score toward neutral rather than giving
# one or two factors full conviction or muting whole sectors structurally.
MIN_COVERAGE = 0.5


# ---------------------------------------------------------------------------
# Reference loading
# ---------------------------------------------------------------------------

def load_factor_directions() -> dict:
    """Returns {factor_id: direction (1 or -1)} from factors_reference.csv."""
    df = pd.read_csv(FACTORS_CSV)
    return dict(zip(df['factor_id'], df['direction'].astype(int)))


def load_models_reference() -> dict:
    """
    Returns {model_name: {model_id, is_composite,
                          weights:  {factor_id: weight},
                          sectors:  {factor_id: frozenset|None}}}.

    `sectors[factor_id]` is the optional `sector_type` override from
    models_reference.csv: a pipe-separated set of company sector_types the factor
    applies to within this model (e.g. earnings yield = general|financial, FFO
    yield = reit). None means no override — fall back to the factor's own
    sector gating from factors_reference.csv.
    """
    models: dict = {}
    with open(MODELS_CSV) as f:
        for row in csv.DictReader(f):
            name         = row['Model']
            model_id     = row['ModelID']
            factor_id    = row['Factors']
            weight       = float(row['Weights'])
            is_composite = bool(int(row['IsComposite']))
            override_raw = (row.get('sector_type') or '').strip()
            override     = frozenset(s.strip() for s in override_raw.split('|')) if override_raw else None
            if name not in models:
                models[name] = {'model_id': model_id, 'is_composite': is_composite,
                                'weights': {}, 'sectors': {}}
            models[name]['weights'][factor_id] = weight
            models[name]['sectors'][factor_id] = override
    return models


def load_factor_sector_types() -> dict:
    """{factor_id: factor-layer sector_type} from factors_reference.csv (default 'all')."""
    df = pd.read_csv(FACTORS_CSV)
    return dict(zip(df['factor_id'], df['sector_type'].fillna('all')))


def load_security_sector_types() -> dict:
    """{security_id (isin): company sector_type} via universe.db + classify_sector."""
    with get_db(UNIVERSE_DB) as conn:
        rows = conn.execute(
            "SELECT isin, simfin_sector, simfin_industry FROM companies"
        ).fetchall()
    return {isin: classify_sector(sec, ind) for isin, sec, ind in rows}


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

def load_factors_zscores(date_filter: list[str] | str | None = None) -> dict:
    """
    Returns {data_date: {security_id: {factor_id: z_score}}}.

    date_filter — if provided, only load rows for that specific date or list of dates.
    """
    with get_db(FACTORS_DB) as conn:
        if date_filter:
            dates = [date_filter] if isinstance(date_filter, str) else list(date_filter)
            placeholders = ','.join('?' * len(dates))
            rows = conn.execute(
                f"SELECT data_date, security_id, factor_id, factor_value_z "
                f"FROM factors WHERE factor_value_z IS NOT NULL AND data_date IN ({placeholders})",
                dates
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
            log.info("Dropping models table (%s) — will rebuild from scratch", reason)
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
    factor_sector_types: dict,
    security_sector_types: dict,
) -> int:
    """Score each base model as a coverage-renormalised weighted sum of
    direction-adjusted factor z-scores.

    For each security only the factors that apply to its sector contribute to the
    denominator (`applicable_weight`); the score is divided by
    max(valid_weight, MIN_COVERAGE × applicable_weight). This keeps a well-covered
    name at full conviction, shrinks a sparsely-covered name toward neutral, and
    stops whole sectors (financials/REITs, which are structurally gated out of many
    factors) from being systematically muted.
    """
    rows = []
    for data_date, securities in factors_data.items():
        for model_name, model_info in base_models.items():
            model_id     = model_info['model_id']
            is_composite = int(model_info['is_composite'])
            weights      = model_info['weights']
            overrides    = model_info['sectors']
            for security_id, factor_zscores in securities.items():
                ctype = security_sector_types.get(security_id, 'general')
                score             = 0.0
                valid_weight      = 0.0
                applicable_weight = 0.0
                for factor_id, weight in weights.items():
                    override = overrides.get(factor_id)
                    if override is not None:
                        applies = ctype in override
                    else:
                        applies = factor_applies_to_company(
                            factor_sector_types.get(factor_id, 'all'), ctype)
                    if not applies:
                        continue
                    applicable_weight += weight
                    z = factor_zscores.get(factor_id)
                    if z is not None and not math.isnan(z):
                        direction     = factor_directions.get(factor_id, 1)
                        score        += z * weight * direction
                        valid_weight += weight
                if applicable_weight > 0 and valid_weight > 0:
                    denom = max(valid_weight, MIN_COVERAGE * applicable_weight)
                    rows.append((data_date, model_id, security_id, score / denom, is_composite))

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
                sid_scores        = base_scores.get(security_id, {})
                score             = 0.0
                valid_weight      = 0.0
                applicable_weight = sum(weights.values())  # all base models apply to all names
                for base_model_id, weight in weights.items():
                    v = sid_scores.get(base_model_id)
                    if v is not None and not math.isnan(v):
                        score        += v * weight
                        valid_weight += weight
                if valid_weight > 0:
                    denom = max(valid_weight, MIN_COVERAGE * applicable_weight)
                    rows.append((data_date, model_id, security_id, score / denom, is_composite))

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
    parser.add_argument('--date', metavar='YYYY-MM-DD', action='append', dest='dates',
                        help='Process a specific date (repeatable: --date D1 --date D2); default: all dates in factors.db')
    parser.add_argument('--clean', action='store_true',
                        help='Drop and rebuild the models table before running')
    args = parser.parse_args()

    models                = load_models_reference()
    base_models           = get_base_models(models)
    alpha_models          = get_alpha_models(models)
    factor_directions     = load_factor_directions()
    factor_sector_types   = load_factor_sector_types()
    security_sector_types = load_security_sector_types()

    log.info("Models to create : %s", list(models.keys()))
    log.info("Base models      : %s", list(base_models.keys()))
    log.info("Alpha models     : %s", list(alpha_models.keys()))

    log.info("Loading pre-computed z-scores from factors.db ...")
    factors_data = load_factors_zscores(date_filter=args.dates)
    dates = sorted(factors_data.keys())
    log.info("Loaded z-scores for %d date(s): %s", len(dates), dates)

    with get_db(MODELS_DB) as conn:
        setup_models_db(conn, clean=args.clean)

        log.info("=== Stage 1: Base models from factor z-scores ===")
        n1 = compute_base_models(conn, base_models, factors_data, factor_directions,
                                 factor_sector_types, security_sector_types)
        log.info("  %s base model records written", f"{n1:,}")

        if alpha_models:
            log.info("=== Stage 2: Alpha models from base model scores ===")
            n2 = compute_alpha_models(conn, alpha_models, factors_data)
            log.info("  %s alpha model records written", f"{n2:,}")
        else:
            n2 = 0

        log.info("Computing cross-sectional z-scores for model scores ...")
        compute_model_zscores(conn)  # commits inside here

    log.info("Done — %s total records | %d models × %d date(s)", f"{n1 + n2:,}", len(models), len(dates))


if __name__ == "__main__":
    main()
