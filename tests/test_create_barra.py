import sqlite3

import pytest

from config import FACTORS_REF, MODELS_REF

# create_barra reads the reference CSVs at import time (module-level FACTOR_NAMES).
# Those files are intentionally untracked in git, so they are absent in CI.
# Skip this module when the reference data is unavailable — it runs locally.
if not (FACTORS_REF.exists() and MODELS_REF.exists()):
    pytest.skip(
        "requires local reference data (data/*_reference.csv)",
        allow_module_level=True,
    )

import pipeline.create_barra as create_barra


def test_load_returns_wide_clips_extreme_returns(tmp_path, monkeypatch):
    db_path = tmp_path / "returns.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE returns (date TEXT, isin TEXT, total_return REAL)"
        )
        conn.executemany(
            "INSERT INTO returns VALUES (?,?,?)",
            [
                ("2026-01-02", "A", 0.10),
                ("2026-01-03", "A", 2.50),
                ("2026-01-03", "B", -3.00),
            ],
        )
        conn.commit()

    monkeypatch.setattr(create_barra, "RETURNS_DB", db_path)
    monkeypatch.setattr(create_barra, "BARRA_RETURN_CLIP", 0.50)

    wide = create_barra._load_returns_wide()

    assert wide.loc["2026-01-02", "A"] == pytest.approx(0.10)
    assert wide.loc["2026-01-03", "A"] == pytest.approx(0.50)
    assert wide.loc["2026-01-03", "B"] == pytest.approx(-0.50)
