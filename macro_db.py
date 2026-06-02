"""
macro_db.py — Utilities for macro signal database.

Provides signal metadata loading for macro.db. DB access uses get_db(MACRO_DB) from utils.
"""

from config import MACRO_DB
from utils import get_db


def load_signals_reference() -> dict:
    """Load signal metadata from signals_reference table. Returns dict: signal_id -> dict."""
    signals = {}
    with get_db(MACRO_DB) as conn:
        cursor = conn.execute(
            "SELECT signal_id, signal_name, source, data_type, frequency, "
            "publication_lag_days, unit, api_endpoint, fred_units, "
            "category, display_order, display_transform "
            "FROM signals_reference"
        )
        for row in cursor.fetchall():
            signal_id = row[0]
            signals[signal_id] = {
                "signal_name": row[1],
                "source": row[2],
                "data_type": row[3],
                "frequency": row[4],
                "publication_lag_days": row[5],
                "unit": row[6],
                "api_endpoint": row[7],
                "fred_units": row[8],
                "category": row[9],
                "display_order": row[10],
                "display_transform": row[11],  # None | "diff_mom"
            }
    return signals
