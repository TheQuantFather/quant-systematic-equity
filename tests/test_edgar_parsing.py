"""Tests for the EDGAR fiscal-period helpers in update_constituents.py.

These functions decide which constituent rows land in which fiscal year/quarter
bucket.  Bugs here cause silent misclassification (e.g. NVDA Jan-FYE quarterlies
being labelled under the wrong fiscal year), which propagates into wrong LTMs
and wrong factor values.
"""

from datetime import date

import pytest

from pipeline.update_constituents import (
    _derive_working_capital_change,
    _latest_expected_sk,
    _quarter_from_period,
)


# ── _derive_working_capital_change ──────────────────────────────────────────

def test_wc_change_both_components_present():
    cf = {"6E42C12C": 100.0, "BF654FC5": -25.0}
    assert _derive_working_capital_change(cf) == 75.0


def test_wc_change_only_receivables():
    assert _derive_working_capital_change({"6E42C12C": 100.0}) == 100.0


def test_wc_change_only_other():
    assert _derive_working_capital_change({"BF654FC5": -25.0}) == -25.0


def test_wc_change_neither_present_returns_none():
    assert _derive_working_capital_change({"some_other_key": 100.0}) is None
    assert _derive_working_capital_change({}) is None


# ── _quarter_from_period — December FYE ─────────────────────────────────────

@pytest.mark.parametrize("period,expected", [
    ("2024-03-31", ("Q1", 2024)),
    ("2024-06-30", ("Q2", 2024)),
    ("2024-09-30", ("Q3", 2024)),
    ("2024-12-31",  None),                    # Q4 — covered by 10-K, skipped
])
def test_quarter_from_period_calendar_fye(period, expected):
    assert _quarter_from_period(period, fye_month=12) == expected


def test_quarter_from_period_calendar_fye_spillover():
    # 52/53-week filers whose Q1 ends in early April (e.g. GD Q1 FY2026 ended
    # Apr 5 rather than Mar 31) — the +1 month spillover must still map to Q1.
    assert _quarter_from_period("2024-04-05", fye_month=12) == ("Q1", 2024)
    assert _quarter_from_period("2024-07-03", fye_month=12) == ("Q2", 2024)
    assert _quarter_from_period("2024-10-03", fye_month=12) == ("Q3", 2024)


# ── _quarter_from_period — non-calendar FYE ─────────────────────────────────

def test_quarter_from_period_apple_september_fye():
    # AAPL: FYE = September.  Q1 ends December, Q2 March, Q3 June, Q4 September.
    # A period ending Dec 2024 belongs to fiscal year 2025 (ends Sep 2025).
    assert _quarter_from_period("2024-12-28", fye_month=9) == ("Q1", 2025)
    assert _quarter_from_period("2025-03-29", fye_month=9) == ("Q2", 2025)
    assert _quarter_from_period("2025-06-28", fye_month=9) == ("Q3", 2025)
    assert _quarter_from_period("2024-09-28", fye_month=9) is None   # Q4 — skip


def test_quarter_from_period_nvda_january_fye():
    # NVDA: FYE = January.  Q1 ends April, Q2 July, Q3 October.
    # A period ending May 2025 belongs to FY2026 (ends Jan 2026).
    assert _quarter_from_period("2025-05-04", fye_month=1) == ("Q1", 2026)
    assert _quarter_from_period("2025-07-28", fye_month=1) == ("Q2", 2026)
    assert _quarter_from_period("2025-10-27", fye_month=1) == ("Q3", 2026)
    assert _quarter_from_period("2025-01-26", fye_month=1) is None    # Q4 — skip


def test_quarter_from_period_unparseable_input():
    assert _quarter_from_period("", fye_month=12) is None
    assert _quarter_from_period("garbage", fye_month=12) is None
    assert _quarter_from_period("2024-XX-31", fye_month=12) is None


def test_quarter_from_period_unrecognised_month():
    # February has no quarter end for a December FYE; must return None.
    assert _quarter_from_period("2024-02-28", fye_month=12) is None
    # ditto August
    assert _quarter_from_period("2024-08-15", fye_month=12) is None


# ── _latest_expected_sk ─────────────────────────────────────────────────────

def test_latest_expected_sk_mid_year_december_fye():
    # As of mid-August 2025 for a Dec-FYE company:
    #   Q1 (Mar 2025) → filing due ~May → available
    #   Q2 (Jun 2025) → filing due ~Aug → available (boundary)
    #   Q3 (Sep 2025) → too early, not yet expected
    # Expected most-recent = FY2025 Q2 → sort_key = 20252.
    assert _latest_expected_sk(date(2025, 8, 31), fye_month=12) == 2025 * 10 + 2


def test_latest_expected_sk_after_q3():
    # December 2025 — Q3 (Sep 30) is well past its 2-month filing lag.
    assert _latest_expected_sk(date(2025, 12, 15), fye_month=12) == 2025 * 10 + 3


def test_latest_expected_sk_january_fallback():
    # Early January 2025 — no Q3 (Sep 2024) is yet 2 months past… actually
    # Sep 30 + 2 months = Nov 30, before Jan 2025, so Q3 FY2024 is expected.
    assert _latest_expected_sk(date(2025, 1, 5), fye_month=12) == 2024 * 10 + 3


def test_latest_expected_sk_apple_fye_september():
    # AAPL FYE = September.  As of mid-August 2025:
    #   Q1 FY2025 (Dec 2024) → filed in Feb 2025 → available
    #   Q2 FY2025 (Mar 2025) → filed in May 2025 → available
    #   Q3 FY2025 (Jun 2025) → filed in Aug 2025 → just available (boundary)
    # Expected most-recent: FY2025 Q3 → sort_key 20253.
    assert _latest_expected_sk(date(2025, 8, 31), fye_month=9) == 2025 * 10 + 3
