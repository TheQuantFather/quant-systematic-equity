"""Tests for selected utils helpers."""

import numpy as np
import pandas as pd
import pytest

from utils import apply_weight_cap, classify_sector, winsorized_zscore


# ── winsorized_zscore ────────────────────────────────────────────────────────

def test_winsorized_zscore_basic_shape_and_centering():
    s = pd.Series(np.arange(100, dtype=float))
    z = winsorized_zscore(s)
    assert len(z) == 100
    assert z.notna().all()
    # After winsorization at [1%, 99%] the mean is essentially zero
    assert abs(z.mean()) < 1e-9
    # And the std of the clipped values is 1 by construction
    assert abs(z.std() - 1.0) < 1e-9


def test_winsorized_zscore_clips_extremes():
    s = pd.Series(list(range(100)) + [1_000_000])  # one extreme outlier
    z = winsorized_zscore(s)
    # The outlier should be clipped to the same value as the next-highest
    assert z.iloc[-1] == z.iloc[-2]


def test_winsorized_zscore_too_few_obs():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])  # n=5 < 10
    z = winsorized_zscore(s)
    assert z.isna().all()
    assert len(z) == 5


def test_winsorized_zscore_zero_variance():
    s = pd.Series([7.0] * 20)
    z = winsorized_zscore(s)
    assert z.isna().all()


def test_winsorized_zscore_preserves_index():
    idx = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]
    s = pd.Series(range(11), index=idx, dtype=float)
    z = winsorized_zscore(s)
    assert list(z.index) == idx


def test_winsorized_zscore_handles_nan():
    # NaNs in the input — quantile / mean / std must still produce a valid output
    vals = list(range(20)) + [np.nan, np.nan]
    s = pd.Series(vals, dtype=float)
    z = winsorized_zscore(s)
    # NaN positions should still be NaN
    assert z.iloc[-1] != z.iloc[-1]  # NaN != NaN
    # Non-NaN positions should be finite
    assert z.iloc[:-2].notna().all()


# ── classify_sector ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("sector,industry,expected", [
    ("Real Estate",        "Office REITs",        "reit"),
    ("Real Estate",        "",                    "reit"),
    ("Financial Services", "Mortgage REITs",      "reit"),     # industry beats sector
    ("Financial Services", "Investment Banking",  "financial"), # investment banks ≠ depository 'bank'
    ("Financial Services", "Banks",               "bank"),      # depository banks
    ("Financial Services", "Credit Services",     "bank"),      # consumer lenders (AmEx, COF, SLM)
    ("Financial Services", "Asset Management",    "financial"),
    ("Financial Services", "Insurance - Life",    "financial"),
    ("Technology",         "Software",            "general"),
    ("Health Care",        "Pharmaceuticals",     "general"),
    (None,                 None,                  "general"),
    ("",                   "",                    "general"),
    ("REAL ESTATE",        "",                    "reit"),     # case-insensitive
    ("financial services", "",                    "financial"),
])
def test_classify_sector(sector, industry, expected):
    assert classify_sector(sector, industry) == expected


# ── apply_weight_cap ─────────────────────────────────────────────────────────

def test_apply_weight_cap_preserves_sum_when_uncapped_names_hit_cap():
    weights = {
        "mega_a": 0.08,
        "mega_b": 0.07,
        "large_a": 0.04,
        "large_b": 0.035,
        **{f"name_{i}": 0.775 / 20 for i in range(20)},
    }

    capped = apply_weight_cap(weights, cap=0.05)

    assert abs(sum(capped.values()) - 1.0) < 1e-12
    assert max(capped.values()) <= 0.05 + 1e-12


def test_apply_weight_cap_normalises_percent_inputs():
    weights = {"a": 8.0, "b": 7.0, **{f"name_{i}": 85.0 / 40 for i in range(40)}}

    capped = apply_weight_cap(weights, cap=0.03)

    assert abs(sum(capped.values()) - 1.0) < 1e-12
    assert max(capped.values()) <= 0.03 + 1e-12


def test_apply_weight_cap_rejects_infeasible_cap():
    with pytest.raises(ValueError):
        apply_weight_cap({"a": 0.6, "b": 0.4}, cap=0.49)
