"""Tests for the EDGAR fiscal-period helpers in update_constituents.py.

These functions decide which constituent rows land in which fiscal year/quarter
bucket.  Bugs here cause silent misclassification (e.g. NVDA Jan-FYE quarterlies
being labelled under the wrong fiscal year), which propagates into wrong LTMs
and wrong factor values.
"""

from datetime import date
import sqlite3

import pytest

from pipeline.update_constituents import (
    _derive_working_capital_change,
    _effective_fye_year_and_month,
    _infer_fiscal_year_offset_from_existing,
    _infer_fye_month_for_period,
    _infer_fye_month_from_existing,
    _latest_expected_sk,
    _parse_statement,
    _quarter_from_period,
    _resolve_quarter_label,
)


class _Stmt:
    def __init__(self, df):
        self._df = df

    def to_dataframe(self):
        return self._df


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


def test_parse_statement_falls_back_to_raw_facts_for_retail_16_week_q1():
    # KR-style 4-5-4 Q1: edgartools' statement table exposes stale period
    # columns ending before the filing period, but raw facts contain the current
    # 16-week quarter ending on period_of_report.  The parser must not store the
    # stale statement-column values.
    pd = pytest.importorskip("pandas")
    stmt_df = pd.DataFrame([
        {
            "concept": "us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax",
            "standard_concept": "Revenue",
            "2025-02-01 (Q1)": None,
        },
        {
            "concept": "us-gaap_ProfitLoss",
            "standard_concept": "ProfitLoss",
            "2025-02-01 (Q1)": 634.0,
        },
    ])
    facts_df = pd.DataFrame([
        {
            "concept": "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
            "numeric_value": 45_118.0,
            "period_start": "2025-02-02",
            "period_end": "2025-05-24",
            "is_dimensioned": False,
        },
        {
            "concept": "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
            "numeric_value": 44_781.0,
            "period_start": "2025-02-02",
            "period_end": "2025-05-24",
            "is_dimensioned": True,
        },
        {
            "concept": "us-gaap:ProfitLoss",
            "numeric_value": 868.0,
            "period_start": "2025-02-02",
            "period_end": "2025-05-24",
            "is_dimensioned": False,
        },
    ])

    result, _ = _parse_statement(
        _Stmt(stmt_df),
        {"Revenue": "REV", "ProfitLoss": "NI"},
        prefer_standalone=True,
        fallback_facts=facts_df,
        period_end="2025-05-24",
    )

    assert result["REV"] == 45_118.0
    assert result["NI"] == 868.0


def test_parse_statement_uses_raw_us_gaap_facts_when_statement_has_no_period_cols():
    pd = pytest.importorskip("pandas")
    stmt_df = pd.DataFrame([
        {"label": "Income Statement"},
    ])
    facts_df = pd.DataFrame([
        {
            "concept": "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
            "numeric_value": 24_880.8,
            "period_start": "2025-02-23",
            "period_end": "2025-06-14",
            "is_dimensioned": False,
        },
        {
            "concept": "us-gaap:CostOfRevenue",
            "numeric_value": 18_142.5,
            "period_start": "2025-02-23",
            "period_end": "2025-06-14",
            "is_dimensioned": False,
        },
        {
            "concept": "us-gaap:NetIncomeLoss",
            "numeric_value": 236.4,
            "period_start": "2025-02-23",
            "period_end": "2025-06-14",
            "is_dimensioned": False,
        },
    ])

    result, _ = _parse_statement(
        _Stmt(stmt_df),
        {
            "Revenue": "REV",
            "CostOfGoodsAndServicesSold": "COR",
            "NetIncome": "NI",
        },
        prefer_standalone=True,
        fallback_facts=facts_df,
        period_end="2025-06-14",
        raw_concept_map={
            "RevenueFromContractWithCustomerExcludingAssessedTax": "Revenue",
            "CostOfRevenue": "CostOfGoodsAndServicesSold",
            "NetIncomeLoss": "NetIncome",
        },
    )

    assert result == {
        "REV": 24_880.8,
        "COR": 18_142.5,
        "NI": 236.4,
    }


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
    # Quarters ending after the January FYE belong to the fiscal year ending the
    # following January.
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


def test_effective_fye_treats_first_week_january_as_december_spillover():
    pd = pytest.importorskip("pandas")
    assert _effective_fye_year_and_month(pd.Timestamp("2025-01-03")) == (2024, 12)
    assert _effective_fye_year_and_month(pd.Timestamp("2025-01-26")) == (2025, 1)


def test_resolve_quarter_label_keeps_derived_quarter_when_declared_is_stale():
    assert _resolve_quarter_label(("Q2", 2023), (2023, "Q1")) == ("Q2", 2023)


def test_resolve_quarter_label_uses_declared_year_when_quarter_agrees():
    assert _resolve_quarter_label(("Q1", 2025), (2026, "Q1")) == ("Q1", 2026)


def test_resolve_quarter_label_rejects_implausible_declared_year():
    assert _resolve_quarter_label(("Q1", 2026), (2024, "Q1")) == ("Q1", 2026)


def test_resolve_quarter_label_prefers_annual_offset_over_declared_year():
    assert _resolve_quarter_label(("Q1", 2026), (2026, "Q1"), fiscal_year_offset=-1) == ("Q1", 2025)


def test_infer_fye_month_from_existing_annual_rows():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE constituents (
            security_id TEXT,
            fiscal_period TEXT,
            statement_type TEXT,
            report_date TEXT,
            publish_date TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO constituents VALUES (?, ?, ?, ?, ?)",
        [
            ("US123", "FY", "Income Statement", "2025-09-27", "2025-11-15"),
            ("US123", "FY", "Cash Flow Statement", "2024-09-28", "2024-11-15"),
            ("US999", "FY", "Income Statement", "2025-12-31", "2026-02-15"),
        ],
    )

    assert _infer_fye_month_from_existing(conn, "US123") == 9


def test_infer_fiscal_year_offset_from_existing_annual_rows():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE constituents (
            security_id TEXT,
            fiscal_year INTEGER,
            fiscal_period TEXT,
            statement_type TEXT,
            report_date TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO constituents VALUES (?, ?, ?, ?, ?)",
        [
            ("HD-LIKE", 2025, "FY", "Income Statement", "2026-02-01"),
            ("HD-LIKE", 2024, "FY", "Income Statement", "2025-02-02"),
            ("NVDA-LIKE", 2026, "FY", "Income Statement", "2026-01-25"),
            ("LDOS-LIKE", 2024, "FY", "Income Statement", "2025-01-03"),
        ],
    )

    assert _infer_fiscal_year_offset_from_existing(conn, "HD-LIKE", "2025-05-04") == -1
    assert _infer_fiscal_year_offset_from_existing(conn, "NVDA-LIKE", "2025-04-27") == 0
    assert _infer_fiscal_year_offset_from_existing(conn, "LDOS-LIKE", "2024-03-29") == 0


def test_infer_fye_month_for_period_uses_nearest_future_annual():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE constituents (
            security_id TEXT,
            fiscal_period TEXT,
            statement_type TEXT,
            report_date TEXT,
            publish_date TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO constituents VALUES (?, ?, ?, ?, ?)",
        [
            ("US123", "FY", "Income Statement", "2021-06-30", "2021-08-15"),
            ("US123", "FY", "Income Statement", "2022-12-31", "2023-02-15"),
            ("US123", "FY", "Income Statement", "2023-12-31", "2024-02-15"),
        ],
    )

    assert _infer_fye_month_for_period(conn, "US123", "2021-03-31") == 6
    assert _infer_fye_month_for_period(conn, "US123", "2022-03-31") == 12


def test_infer_fye_month_for_period_normalises_january_spillover():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE constituents (
            security_id TEXT,
            fiscal_period TEXT,
            statement_type TEXT,
            report_date TEXT,
            publish_date TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO constituents VALUES (?, ?, ?, ?, ?)",
        ("US123", "FY", "Income Statement", "2025-01-03", "2025-02-11"),
    )

    assert _infer_fye_month_for_period(conn, "US123", "2024-03-29") == 12


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


def test_latest_expected_sk_honours_annual_fiscal_year_offset():
    assert _latest_expected_sk(date(2026, 6, 5), fye_month=1, fiscal_year_offset=-1) == 2026 * 10 + 1
