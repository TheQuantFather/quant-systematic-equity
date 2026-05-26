"""Tests for the pure math helpers in create_factors.py.

These cover the functions where bad numbers hide longest:
  _ebitda, _enterprise_value     — accounting identities
  _fix_ytd_quarters              — YTD decomposition heuristic
  compute_quality_factors        — 19 ratios from raw financial inputs
  compute_value_factors          — market-cap-relative valuations
  compute_growth_factors         — YoY changes with sign rules
  select_ltm_data                — LTM building incl. annual-only and gap warning
"""

from datetime import date

import pandas as pd
import pytest

from create_factors import (
    _ebitda,
    _enterprise_value,
    _fix_ytd_quarters,
    compute_growth_factors,
    compute_quality_factors,
    compute_value_factors,
    select_ltm_data,
)


# ── _ebitda ─────────────────────────────────────────────────────────────────

def test_ebitda_positive_da():
    assert _ebitda(100.0, 20.0) == 120.0


def test_ebitda_negative_da_uses_abs():
    # Some filers store D&A as a negative cash outflow on the CF statement;
    # _ebitda must normalise the sign.
    assert _ebitda(100.0, -20.0) == 120.0


def test_ebitda_missing_inputs():
    assert _ebitda(None, 20.0) is None
    assert _ebitda(100.0, None) is None
    assert _ebitda(None, None) is None


# ── _enterprise_value ───────────────────────────────────────────────────────

def test_enterprise_value_basic():
    ev = _enterprise_value(market_cap=1000, short_debt=100, long_debt=200, cash=50)
    assert ev == 1000 + 100 + 200 - 50


def test_enterprise_value_treats_none_as_zero():
    ev = _enterprise_value(market_cap=1000, short_debt=None, long_debt=200, cash=None)
    assert ev == 1200


def test_enterprise_value_returns_none_when_nonpositive():
    # Net-cash company whose cash exceeds market cap + debt → EV ≤ 0, skip.
    assert _enterprise_value(market_cap=100, short_debt=0, long_debt=0, cash=1000) is None


# ── _fix_ytd_quarters ───────────────────────────────────────────────────────

def _make_ytd_data(q1, q2, q3=None, kind="Flow"):
    """Build the minimal nested dict that _fix_ytd_quarters expects."""
    sid = "ISIN-TEST"
    sid_data = {(2025, "Q1"): {"Revenue": q1},
                (2025, "Q2"): {"Revenue": q2}}
    if q3 is not None:
        sid_data[(2025, "Q3")] = {"Revenue": q3}
    return {sid: sid_data}, {"Revenue": kind}


def test_fix_ytd_decomposes_h1_cumulative():
    # Q2 = H1 cumulative (Q1 + actual_Q2).  Q3 = 9M cumulative.
    data, kind_map = _make_ytd_data(q1=100, q2=210, q3=320)  # ratios 2.1 and 3.2
    fixed = _fix_ytd_quarters(data, kind_map)
    assert fixed == 2
    assert data["ISIN-TEST"][(2025, "Q2")]["Revenue"] == 110   # 210 - 100
    assert data["ISIN-TEST"][(2025, "Q3")]["Revenue"] == 110   # 320 - 210 (stored Q2)


def test_fix_ytd_leaves_seasonal_uplift_alone():
    # Q2 only 50% above Q1 — well below 1.65 threshold; should not be touched.
    data, kind_map = _make_ytd_data(q1=100, q2=150, q3=160)
    fixed = _fix_ytd_quarters(data, kind_map)
    assert fixed == 0
    assert data["ISIN-TEST"][(2025, "Q2")]["Revenue"] == 150
    assert data["ISIN-TEST"][(2025, "Q3")]["Revenue"] == 160


def test_fix_ytd_skips_negative_q1():
    # The heuristic relies on a positive Q1; for losses, we can't infer the sign.
    data, kind_map = _make_ytd_data(q1=-100, q2=300)
    fixed = _fix_ytd_quarters(data, kind_map)
    assert fixed == 0
    assert data["ISIN-TEST"][(2025, "Q2")]["Revenue"] == 300  # unchanged


def test_fix_ytd_ignores_stock_items():
    data, kind_map = _make_ytd_data(q1=100, q2=300, kind="Stock")
    fixed = _fix_ytd_quarters(data, kind_map)
    assert fixed == 0
    assert data["ISIN-TEST"][(2025, "Q2")]["Revenue"] == 300  # unchanged


def test_fix_ytd_q3_left_alone_when_below_threshold():
    # Q2 looks YTD-cumulative but Q3 doesn't — only Q2 should be corrected.
    data, kind_map = _make_ytd_data(q1=100, q2=210, q3=150)  # Q3/Q1 = 1.5 < 1.65
    fixed = _fix_ytd_quarters(data, kind_map)
    assert fixed == 1
    assert data["ISIN-TEST"][(2025, "Q2")]["Revenue"] == 110
    assert data["ISIN-TEST"][(2025, "Q3")]["Revenue"] == 150  # unchanged


# ── compute_quality_factors ─────────────────────────────────────────────────

def _baseline_cdata():
    """Synthetic LTM dict producing a reasonable, well-behaved company."""
    return {
        "Revenue":                                   1000.0,
        "Cost of Revenue":                            600.0,
        "Gross Profit":                               400.0,
        "Operating Income (Loss)":                    200.0,
        "Net Income":                                 150.0,
        "Total Equity":                              1500.0,
        "Total Assets":                              3000.0,
        "Total Liabilities":                         1500.0,
        "Net Cash from Operating Activities":         180.0,
        "Change in Fixed Assets & Intangibles":      -100.0,   # capex stored negative
        "Total Current Assets":                       500.0,
        "Total Current Liabilities":                  250.0,
        "Short Term Debt":                            100.0,
        "Long Term Debt":                             400.0,
        "Change in Working Capital":                  -30.0,
        "Pretax Income (Loss)":                       190.0,
        "Income Tax (Expense) Benefit, Net":          -40.0,
        "Cash, Cash Equivalents & Short Term Investments": 200.0,
    }


def test_quality_factors_basic_ratios():
    f = compute_quality_factors(_baseline_cdata())
    assert f["Gross Margin"]      == pytest.approx(0.40)
    assert f["Operating Margin"]  == pytest.approx(0.20)
    assert f["Net Margin"]        == pytest.approx(0.15)
    assert f["ROE"]               == pytest.approx(0.10)
    assert f["ROA"]               == pytest.approx(0.05)
    assert f["Current Ratio"]     == pytest.approx(2.0)
    assert f["Debt-to-Assets"]    == pytest.approx(0.50)
    assert f["Asset Turnover"]    == pytest.approx(1000 / 3000)
    assert f["Equity Ratio"]      == pytest.approx(0.50)


def test_quality_factors_fcf_margin_sign_convention():
    # capex is stored negative; FCF Margin must be (op_cf + capex) / revenue,
    # equivalent to (op_cf - |capex|) / revenue.
    c = _baseline_cdata()
    f = compute_quality_factors(c)
    assert f["FCF Margin"] == pytest.approx((180 - 100) / 1000)


def test_quality_factors_net_margin_filter():
    # A REIT-style company with near-zero revenue produces wild net margins;
    # the code should exclude any |Net Margin| > 2 to prevent z-score distortion.
    c = _baseline_cdata()
    c["Revenue"]    = 50.0
    c["Net Income"] = 200.0          # nm = 4.0
    f = compute_quality_factors(c)
    assert "Net Margin" not in f
    # But Operating Margin (2 / 50 wait — let me recompute) is 4.0 also, so it
    # should still be present (Net Margin's |.|>2 filter is bespoke to that key).
    assert "Operating Margin" in f


def test_quality_factors_interest_coverage_edgar_path():
    # EDGAR convention: InterestExpense is stored negative (expense sign).
    # Net interest = |expense| - investment_income.  Skip if net ≤ 0.
    c = _baseline_cdata()
    c["Interest Expense, Net"]      = -50.0  # EDGAR: gross expense, negative
    c["Investment Income, Interest"] = 10.0
    f = compute_quality_factors(c)
    assert f["Interest Coverage"] == pytest.approx(200 / 40)


def test_quality_factors_interest_coverage_simfin_path():
    # SimFin convention: already net, positive value = expense.
    c = _baseline_cdata()
    c["Interest Expense, Net"]      = 40.0   # positive = net expense
    c["Investment Income, Interest"] = None  # ignored on this path
    f = compute_quality_factors(c)
    assert f["Interest Coverage"] == pytest.approx(200 / 40)


def test_quality_factors_roic_with_loss_makers():
    # Pretax negative → effective tax rate falls back to 0.21 statutory.
    c = _baseline_cdata()
    c["Pretax Income (Loss)"]            = -50.0
    c["Income Tax (Expense) Benefit, Net"] = 10.0
    f = compute_quality_factors(c)
    nopat = 200.0 * (1.0 - 0.21)
    invested_capital = 1500 + 100 + 400 - 200
    assert f["ROIC"] == pytest.approx(nopat / invested_capital)


def test_quality_factors_accruals_and_gp_to_assets():
    c = _baseline_cdata()
    f = compute_quality_factors(c)
    assert f["Accruals Ratio"]        == pytest.approx((150 - 180) / 3000)
    assert f["Gross Profit to Assets"] == pytest.approx(400 / 3000)


def test_quality_factors_empty_input_returns_empty():
    assert compute_quality_factors({}) == {}


def test_quality_factors_zero_denominators_skipped():
    # Any factor whose denominator is 0 should be silently absent (not raise).
    c = _baseline_cdata()
    c["Revenue"]      = 0.0
    c["Total Equity"] = 0.0   # ROE skipped (equity > 0 required)
    f = compute_quality_factors(c)
    assert "Gross Margin" not in f
    assert "Operating Margin" not in f
    assert "Net Margin" not in f
    assert "ROE" not in f


# ── compute_value_factors ───────────────────────────────────────────────────

def _make_prices(isin, close):
    dates  = pd.to_datetime(["2025-04-01"]).values
    rets   = pd.Series([0.0]).values
    closes = pd.Series([close]).values
    vols   = pd.Series([1e6]).values.astype(float)
    return {isin: (dates, rets, closes, vols)}


def test_value_factors_basic():
    cdata = {
        "Shares (Basic)":                                1_000_000,
        "Net Income":                                       50_000_000,
        "Total Equity":                                  1_000_000_000,
        "Revenue":                                         800_000_000,
        "Net Cash from Operating Activities":               75_000_000,
        "Operating Income (Loss)":                          80_000_000,
        "Depreciation & Amortization":                      30_000_000,
        "Short Term Debt":                                  20_000_000,
        "Long Term Debt":                                  100_000_000,
        "Cash, Cash Equivalents & Short Term Investments":  50_000_000,
        "Dividends Paid":                                  -10_000_000,
    }
    isin = "ISIN-VAL"
    prices = _make_prices(isin, close=500.0)
    f = compute_value_factors(cdata, isin, prices, ref_date=date(2025, 4, 1))

    mc = 500.0 * 1_000_000   # market cap = 500m
    assert f["Earnings Yield"]   == pytest.approx(50_000_000 / mc)
    assert f["Book-to-Price"]    == pytest.approx(1_000_000_000 / mc)
    assert f["Sales-to-Price"]   == pytest.approx(800_000_000 / mc)
    assert f["Cash Yield"]       == pytest.approx(75_000_000 / mc)
    assert f["Dividend Yield"]   == pytest.approx(10_000_000 / mc)

    ev = mc + 20_000_000 + 100_000_000 - 50_000_000
    assert f["EV-to-EBIT"]       == pytest.approx(ev / 80_000_000)
    assert f["EV/EBITDA"]        == pytest.approx(ev / (80_000_000 + 30_000_000))


def test_value_factors_missing_price_returns_empty():
    cdata = {"Shares (Basic)": 1_000_000, "Net Income": 50.0}
    assert compute_value_factors(cdata, "NOSUCH", {}, ref_date=date(2025, 4, 1)) == {}


def test_value_factors_dividend_yield_only_for_payers():
    # Companies that do not pay dividends (Dividends Paid >= 0 or missing) should
    # not get a Dividend Yield factor at all.
    isin = "ISIN-NODIV"
    prices = _make_prices(isin, close=100.0)
    cdata = {
        "Shares (Basic)": 1_000_000,
        "Net Income":     5_000_000,
        # No Dividends Paid key
    }
    f = compute_value_factors(cdata, isin, prices, ref_date=date(2025, 4, 1))
    assert "Dividend Yield" not in f


# ── compute_growth_factors ──────────────────────────────────────────────────

def test_growth_factors_basic_yoy():
    cur   = {"Revenue": 1100.0, "Net Income": 120.0,
             "Net Cash from Operating Activities": 130.0,
             "Total Assets": 3300.0, "Total Equity": 1650.0,
             "Operating Income (Loss)": 220.0}
    prior = {"Revenue": 1000.0, "Net Income": 100.0,
             "Net Cash from Operating Activities": 100.0,
             "Total Assets": 3000.0, "Total Equity": 1500.0,
             "Operating Income (Loss)": 200.0}
    f = compute_growth_factors(cur, prior)
    assert f["Revenue Growth"]  == pytest.approx(0.10)
    assert f["Earnings Growth"] == pytest.approx(0.20)
    assert f["Cash Flow Growth"] == pytest.approx(0.30)
    assert f["Asset Growth"]    == pytest.approx(0.10)
    assert f["Equity Growth"]   == pytest.approx(0.10)
    assert f["Operating Income Growth"] == pytest.approx(0.10)


def test_growth_factors_negative_base_skipped_when_required():
    # Earnings/CF/OpInc growth all require positive base.  Asset/Equity/Revenue
    # do not — their growth is computed against |prior|.
    cur   = {"Revenue": 800.0, "Net Income": 50.0,
             "Net Cash from Operating Activities": 60.0,
             "Total Assets": 3000.0, "Total Equity": 1500.0,
             "Operating Income (Loss)": 200.0}
    prior = {"Revenue": 1000.0, "Net Income": -100.0,    # loss
             "Net Cash from Operating Activities": -50.0,
             "Total Assets": 2800.0, "Total Equity": 1400.0,
             "Operating Income (Loss)": -10.0}
    f = compute_growth_factors(cur, prior)
    assert "Earnings Growth" not in f
    assert "Cash Flow Growth" not in f
    assert "Operating Income Growth" not in f
    # Revenue contraction still computed (no require_positive_base):
    assert f["Revenue Growth"] == pytest.approx(-0.20)
    # Asset / Equity growth still computed:
    assert f["Asset Growth"]  == pytest.approx((3000 - 2800) / 2800)
    assert f["Equity Growth"] == pytest.approx((1500 - 1400) / 1400)


def test_growth_factors_zero_base_skipped():
    cur   = {"Revenue": 100.0}
    prior = {"Revenue": 0.0}
    assert "Revenue Growth" not in compute_growth_factors(cur, prior)


def test_growth_factors_ebitda_growth_uses_abs_da():
    cur   = {"Operating Income (Loss)": 220.0,
             "Depreciation & Amortization": -30.0}    # negative D&A
    prior = {"Operating Income (Loss)": 200.0,
             "Depreciation & Amortization":  30.0}
    f = compute_growth_factors(cur, prior)
    # current EBITDA = 250, prior = 230
    assert f["EBITDA Growth"] == pytest.approx((250 - 230) / 230)


# ── select_ltm_data ─────────────────────────────────────────────────────────

def _quarter(fy, q, pub, **vals):
    """Build a quarter bucket with the _publish_date / _sort_key meta keys."""
    q_num = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}[q]
    bucket = {"_publish_date": pd.Timestamp(pub), "_sort_key": fy * 10 + q_num}
    bucket.update(vals)
    return bucket


def test_select_ltm_normal_quarterly_sum():
    sid_data = {
        (2024, "Q1"): _quarter(2024, "Q1", "2024-05-01", Revenue=250, **{"Total Assets": 9800}),
        (2024, "Q2"): _quarter(2024, "Q2", "2024-08-01", Revenue=260, **{"Total Assets": 9900}),
        (2024, "Q3"): _quarter(2024, "Q3", "2024-11-01", Revenue=270, **{"Total Assets": 9950}),
        (2024, "Q4"): _quarter(2024, "Q4", "2025-02-15", Revenue=280, **{"Total Assets": 10_000}),
    }
    kind_map = {"Revenue": "Flow", "Total Assets": "Stock"}

    ltm, prior = select_ltm_data(sid_data, kind_map, snapshot=date(2025, 4, 1))
    assert ltm["Revenue"] == 250 + 260 + 270 + 280     # Flow → summed
    assert ltm["Total Assets"] == 10_000               # Stock → most recent only
    assert prior == {}                                 # < 8 quarters available


def test_select_ltm_respects_publish_date_cutoff():
    # Q4 was published AFTER the snapshot — must be excluded.
    sid_data = {
        (2024, "Q1"): _quarter(2024, "Q1", "2024-05-01", Revenue=250),
        (2024, "Q2"): _quarter(2024, "Q2", "2024-08-01", Revenue=260),
        (2024, "Q3"): _quarter(2024, "Q3", "2024-11-01", Revenue=270),
        (2024, "Q4"): _quarter(2024, "Q4", "2025-05-01", Revenue=280),  # after snap
    }
    ltm, _ = select_ltm_data(sid_data, {"Revenue": "Flow"}, snapshot=date(2025, 4, 1))
    assert ltm["Revenue"] == 250 + 260 + 270            # Q4 excluded


def test_select_ltm_annual_only_uses_single_period():
    # SimFin annual-only filers: each period is a full FY (sort_key gaps >= 10).
    # Summing them would N×-overstate; the code must collapse to the latest only.
    sid_data = {
        (2022, "Q4"): _quarter(2022, "Q4", "2023-02-15", Revenue=1000),
        (2023, "Q4"): _quarter(2023, "Q4", "2024-02-15", Revenue=1100),
        (2024, "Q4"): _quarter(2024, "Q4", "2025-02-15", Revenue=1200),
    }
    ltm, prior = select_ltm_data(sid_data, {"Revenue": "Flow"},
                                 snapshot=date(2025, 4, 1))
    assert ltm["Revenue"] == 1200              # latest only — not summed
    assert prior["Revenue"] == 1100            # the year before


def test_select_ltm_empty_when_no_data():
    ltm, prior = select_ltm_data({}, {"Revenue": "Flow"}, snapshot=date(2025, 4, 1))
    assert ltm == {} and prior == {}


def test_select_ltm_excludes_bs_only_buckets():
    # An orphaned EDGAR annual BS-only bucket should NOT overwrite the prior
    # quarter's BS value — the _has_flow filter excludes it from the window.
    sid_data = {
        (2024, "Q1"): _quarter(2024, "Q1", "2024-05-01",
                                Revenue=250, **{"Total Assets": 9800}),
        (2024, "Q2"): _quarter(2024, "Q2", "2024-08-01",
                                Revenue=260, **{"Total Assets": 9900}),
        (2024, "Q3"): _quarter(2024, "Q3", "2024-11-01",
                                Revenue=270, **{"Total Assets": 9950}),
        # BS-only bucket — no Flow data; must be skipped:
        (2024, "Q4"): _quarter(2024, "Q4", "2025-02-15", **{"Total Assets": 11_111}),
    }
    kind_map = {"Revenue": "Flow", "Total Assets": "Stock"}
    ltm, _ = select_ltm_data(sid_data, kind_map, snapshot=date(2025, 4, 1))
    # Latest *Flow*-bearing bucket is Q3; its BS value wins, not the orphan Q4.
    assert ltm["Total Assets"] == 9950
    assert ltm["Revenue"]      == 250 + 260 + 270


def test_select_ltm_derives_gross_profit_per_quarter():
    # build_ltm() derives Gross Profit when Revenue and CoR are present but GP
    # isn't.  Verify the per-quarter derivation and subsequent summation.
    sid_data = {
        (2024, "Q1"): _quarter(2024, "Q1", "2024-05-01",
                                Revenue=1000, **{"Cost of Revenue": 600}),
        (2024, "Q2"): _quarter(2024, "Q2", "2024-08-01",
                                Revenue=1100, **{"Cost of Revenue": 650}),
        (2024, "Q3"): _quarter(2024, "Q3", "2024-11-01",
                                Revenue=1200, **{"Cost of Revenue": 700}),
        (2024, "Q4"): _quarter(2024, "Q4", "2025-02-15",
                                Revenue=1300, **{"Cost of Revenue": 750}),
    }
    kind_map = {"Revenue": "Flow", "Cost of Revenue": "Flow", "Gross Profit": "Flow"}
    ltm, _ = select_ltm_data(sid_data, kind_map, snapshot=date(2025, 4, 1))
    assert ltm["Gross Profit"] == (1000 - 600) + (1100 - 650) + (1200 - 700) + (1300 - 750)
