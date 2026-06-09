from __future__ import annotations

import pandas as pd
import pytest

from portfolio_analytics import (
    daily_price_performance,
    enrich_snapshot_items,
    nearest_risk_date,
    split_table,
    weighted_factor_exposures,
)


def test_nearest_risk_date_uses_prior_snapshot_or_earliest_available():
    dates = ["2026-04-01", "2025-04-01", "2024-04-01"]

    assert nearest_risk_date("2026-06-02", dates) == "2026-04-01"
    assert nearest_risk_date("2025-10-01", dates) == "2025-04-01"
    assert nearest_risk_date("2023-12-31", dates) == "2024-04-01"
    assert nearest_risk_date("2026-06-02", []) is None


def test_enrich_snapshot_items_prefers_gics_and_falls_back_to_simfin_or_unknown():
    items = pd.DataFrame({
        "symbol": ["AAPL", "XYZ", "CASH"],
        "isin": ["US0378331005", "US0000000001", None],
        "name": [None, "Broker Name", "EUR Cash"],
    })
    meta = pd.DataFrame({
        "isin": ["US0378331005", "US0000000001"],
        "ticker": ["AAPL", "XYZ"],
        "company_name": ["Apple Inc", "XYZ Corp"],
        "gics_sector": ["Information Technology", None],
        "gics_industry": ["Technology Hardware", None],
        "simfin_sector": ["Technology", "Industrials"],
        "simfin_industry": ["Consumer Electronics", "Machinery"],
    })

    out = enrich_snapshot_items(items, meta).set_index("symbol")

    assert out.loc["AAPL", "display_name"] == "Apple Inc"
    assert out.loc["AAPL", "sector"] == "Information Technology"
    assert out.loc["AAPL", "industry"] == "Technology Hardware"
    assert out.loc["XYZ", "display_name"] == "Broker Name"
    assert out.loc["XYZ", "sector"] == "Industrials"
    assert out.loc["XYZ", "industry"] == "Machinery"
    assert out.loc["CASH", "sector"] == "Unknown"
    assert out.loc["CASH", "industry"] == "Unknown"


def test_split_table_aggregates_weight_value_and_name_count():
    positions = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "JNJ"],
        "sector": ["Technology", "Technology", "Health Care"],
        "weight": [0.20, 0.15, 0.10],
        "market_value_base": [200.0, 150.0, 100.0],
    })

    out = split_table(positions, "sector").set_index("sector")

    assert out.loc["Technology", "weight"] == 0.35
    assert out.loc["Technology", "market_value_base"] == 350.0
    assert out.loc["Technology", "names"] == 2
    assert out.loc["Health Care", "weight"] == 0.10


def test_daily_price_performance_matches_by_isin_and_symbol_fallback():
    current = pd.DataFrame({
        "symbol": ["AAPL", "OUTU", "NEW"],
        "isin": ["US0378331005", None, None],
        "display_name": ["Apple", "Out Universe", "New Name"],
        "sector": ["Technology", "Unknown", "Unknown"],
        "industry": ["Hardware", "Unknown", "Unknown"],
        "currency": ["USD", "USD", "USD"],
        "quantity": [2.0, 3.0, 1.0],
        "market_value_base": [220.0, 330.0, 50.0],
        "weight": [0.22, 0.33, 0.05],
    })
    prior = pd.DataFrame({
        "symbol": ["AAPL", "OUTU"],
        "isin": ["US0378331005", None],
        "currency": ["USD", "USD"],
        "quantity": [2.0, 2.0],
        "market_value_base": [200.0, 200.0],
    })

    perf, summary = daily_price_performance(current, prior, prior_net_liq_value=1000.0)
    by_symbol = perf.set_index("symbol")

    assert by_symbol.loc["AAPL", "daily_pnl_base"] == 20.0
    assert by_symbol.loc["AAPL", "daily_return"] == pytest.approx(0.10)
    assert by_symbol.loc["OUTU", "daily_pnl_base"] == 30.0
    assert by_symbol.loc["OUTU", "quantity_change"] == 1.0
    assert pd.isna(by_symbol.loc["NEW", "daily_pnl_base"])
    assert summary["pnl_base"] == 50.0
    assert summary["return_pct"] == 0.05
    assert summary["matched_names"] == 2
    assert summary["unmatched_names"] == 1
    assert summary["changed_quantity_names"] == 2
    assert summary["coverage_weight"] == pytest.approx(550.0 / 600.0)


def test_weighted_factor_exposures_sums_security_exposures_and_reports_coverage():
    positions = pd.DataFrame({
        "isin": ["US0378331005", "US5949181045", "US0000000001"],
        "weight": [0.25, 0.15, 0.10],
    })
    exposures = pd.DataFrame({
        "isin": ["US0378331005", "US5949181045", "US0378331005"],
        "factor_id": ["beta_60d", "beta_60d", "quality"],
        "exposure": [1.2, 0.8, -0.4],
    })

    out, coverage = weighted_factor_exposures(
        positions,
        exposures,
        pretty_names={"beta_60d": "Beta (60d)", "quality": "Quality"},
        factor_groups={"beta_60d": "Style", "quality": "Style"},
    )
    by_factor = out.set_index("factor_id")

    assert coverage == 0.40
    assert by_factor.loc["beta_60d", "exposure"] == 0.42
    assert by_factor.loc["quality", "exposure"] == -0.10
    assert by_factor.loc["quality", "abs_exposure"] == 0.10
    assert by_factor.loc["quality", "factor_name"] == "Quality"
