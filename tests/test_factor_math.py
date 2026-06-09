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

from pipeline.create_factors import (
    _apply_security_data_starts,
    _ebitda,
    _dedup_constituent_rows,
    _enterprise_value,
    _fix_ytd_quarters,
    _q4_inputs_available_before_annual,
    compute_growth_factors,
    compute_quality_factors,
    compute_reit_factors,
    compute_value_factors,
    select_growth_series,
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


def test_dedup_constituent_rows_prefers_native_isin_over_mapped_legacy():
    df = pd.DataFrame([
        {
            "security_id": "US-ISIN",
            "constituent_id": "REV",
            "fiscal_year": 2025,
            "fiscal_period": "Q1",
            "publish_date": pd.Timestamp("2025-05-28"),
            "source_priority": 0,
            "constituent_value": 44_062.0,
        },
        {
            "security_id": "US-ISIN",
            "constituent_id": "REV",
            "fiscal_year": 2025,
            "fiscal_period": "Q1",
            "publish_date": pd.Timestamp("2024-05-29"),
            "source_priority": 1,
            "constituent_value": 26_044.0,
        },
    ])

    out = _dedup_constituent_rows(df)
    assert len(out) == 1
    assert out.iloc[0]["constituent_value"] == 26_044.0
    assert out.iloc[0]["publish_date"] == pd.Timestamp("2024-05-29")


def test_q4_temporal_guard_uses_constituent_publish_dates_not_bucket_max():
    annual_pub = pd.Timestamp("2025-02-26")
    q1 = {
        "Revenue": 26_044.0,
        "Some Other Flow": 1.0,
        "_publish_date": pd.Timestamp("2025-05-28"),
        "_publish_by_name": {
            "Revenue": pd.Timestamp("2024-05-29"),
            "Some Other Flow": pd.Timestamp("2025-05-28"),
        },
    }
    q2 = {"Revenue": 30_040.0, "_publish_by_name": {"Revenue": pd.Timestamp("2024-08-28")}}
    q3 = {"Revenue": 35_082.0, "_publish_by_name": {"Revenue": pd.Timestamp("2024-11-20")}}

    assert _q4_inputs_available_before_annual(q1, q2, q3, "Revenue", annual_pub)
    assert not _q4_inputs_available_before_annual(q1, q2, q3, "Some Other Flow", annual_pub)


def test_apply_security_data_starts_filters_old_issuer_rows():
    df = pd.DataFrame([
        {"security_id": "US-LINE", "report_date": "2018-12-31", "constituent_value": 1.0},
        {"security_id": "US-LINE", "report_date": "2024-12-31", "constituent_value": 2.0},
        {"security_id": "US-OTHER", "report_date": "2018-12-31", "constituent_value": 3.0},
    ])

    out = _apply_security_data_starts(
        df,
        {"US-LINE": pd.Timestamp("2024-01-01")},
    )

    assert out["constituent_value"].tolist() == [2.0, 3.0]


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


# ── compute_growth_factors (multi-year trend) ───────────────────────────────

def test_growth_trend_basic_slope_over_mean():
    # Trend growth = OLS slope of the annual series ÷ mean(|level|).
    # Net Income [100,110,120]: slope=10, mean=110 → 0.0909
    series = {
        "Revenue":                            [1000.0, 1100.0, 1200.0],
        "Net Income":                         [100.0, 110.0, 120.0],
        "Net Cash from Operating Activities": [200.0, 230.0, 260.0],
        "Operating Income (Loss)":            [180.0, 190.0, 200.0],
        "EBITDA":                             [300.0, 330.0, 360.0],
    }
    f = compute_growth_factors(series)
    assert f["Revenue Growth"]          == pytest.approx(100 / 1100)
    assert f["Earnings Growth"]         == pytest.approx(10 / 110)
    assert f["Cash Flow Growth"]        == pytest.approx(30 / 230)
    assert f["Operating Income Growth"] == pytest.approx(10 / 190)
    assert f["EBITDA Growth"]           == pytest.approx(30 / 330)
    # Asset / equity growth are no longer growth factors (asset-growth anomaly):
    assert "Asset Growth" not in f
    assert "Equity Growth" not in f


def test_growth_trend_requires_three_points():
    # Two points is below the minimum window — no trend emitted.
    assert compute_growth_factors({"Revenue": [1000.0, 1100.0]}) == {}
    # Three points emit.
    assert "Revenue Growth" in compute_growth_factors({"Revenue": [1000.0, 1050.0, 1100.0]})


def test_growth_trend_no_explosion_on_depressed_base():
    # KEY-like net income: volatile, dips negative, recovers. The 1-yr ratio
    # exploded (+76x); the trend must stay bounded and read flat-to-negative.
    series = {"Net Income": [2625.0, 1917.0, 967.0, -161.0, 1829.0]}
    g = compute_growth_factors(series)["Earnings Growth"]
    assert abs(g) < 1.0          # not an explosive multiple
    assert g < 0.2               # a down-then-recover series is not strong growth


def test_growth_trend_degenerate_zero_series_skipped():
    # All-zero series → mean level 0 → undefined, skipped rather than div-by-zero.
    assert "Revenue Growth" not in compute_growth_factors({"Revenue": [0.0, 0.0, 0.0]})


def test_growth_series_ebitda_uses_abs_da():
    # select_growth_series derives the per-year EBITDA series with abs(D&A).
    # Annual-only filer: one FY (Q4) period per year, three years.
    def yr(fy, pub, op, da):
        return _quarter(fy, "Q4", pub,
                        **{"Operating Income (Loss)": op,
                           "Depreciation & Amortization": da,
                           "Net Income": op})
    sid_data = {
        0: yr(2021, "2022-02-01", 200.0,  30.0),
        1: yr(2022, "2023-02-01", 210.0, -30.0),   # negative D&A sign
        2: yr(2023, "2024-02-01", 220.0,  30.0),
    }
    kind_map = {"Operating Income (Loss)": "Flow",
                "Depreciation & Amortization": "Flow",
                "Net Income": "Flow"}
    series = select_growth_series(sid_data, kind_map, date(2024, 6, 1))
    assert series["EBITDA"] == pytest.approx([230.0, 240.0, 250.0])  # op + |da|, oldest→newest
    # FFO series = Net Income + |D&A| (NI set equal to op in this fixture):
    assert series["FFO"] == pytest.approx([230.0, 240.0, 250.0])


def test_reit_ffo_growth_is_trend_not_yoy():
    # FFO Growth now uses the multi-year trend (slope/mean), bounded even on a
    # volatile base — no 1yr-ratio explosion.
    series = {"FFO": [500.0, 520.0, 30.0, 560.0]}   # one trough year
    f = compute_reit_factors(cdata={"Net Income": 100.0, "Depreciation & Amortization": 400.0},
                             growth_series=series, isin="X", prices={})
    assert "FFO Growth" in f
    assert abs(f["FFO Growth"]) < 1.0               # trend stays bounded
    # No FFO series → no FFO Growth (but other FFO factors absent too without price)
    assert "FFO Growth" not in compute_reit_factors(
        cdata={"Net Income": 100.0, "Depreciation & Amortization": 400.0},
        growth_series={}, isin="X", prices={})


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
    assert ltm["Revenue"] is None                       # Q4 excluded; no full LTM


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
    assert ltm["Revenue"] is None


def test_select_ltm_requires_four_flow_quarters_for_non_annual_filers():
    sid_data = {
        (2025, "Q2"): _quarter(2025, "Q2", "2025-08-01", Revenue=260),
        (2025, "Q3"): _quarter(2025, "Q3", "2025-11-01", Revenue=270),
        (2026, "Q1"): _quarter(2026, "Q1", "2026-05-01", Revenue=290),
    }

    ltm, _ = select_ltm_data(sid_data, {"Revenue": "Flow"}, snapshot=date(2026, 5, 29))
    assert ltm["Revenue"] is None


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
