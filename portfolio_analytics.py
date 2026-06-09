from __future__ import annotations

import pandas as pd


def nearest_risk_date(data_date: str, risk_dates: list[str]) -> str | None:
    """Return the nearest available risk snapshot on or before data_date."""
    if not risk_dates:
        return None
    dates = sorted(risk_dates)
    before = [d for d in dates if d <= data_date]
    return before[-1] if before else dates[0]


def enrich_snapshot_items(items: pd.DataFrame, universe_meta: pd.DataFrame) -> pd.DataFrame:
    """Attach display name, sector, and industry metadata to snapshot rows."""
    out = items.copy()
    if not universe_meta.empty:
        out = out.merge(universe_meta, on="isin", how="left", suffixes=("", "_univ"))
    else:
        out["company_name"] = None
        out["gics_sector"] = None
        out["gics_industry"] = None
        out["simfin_sector"] = None
        out["simfin_industry"] = None

    out["display_name"] = out["name"].fillna(out["company_name"]).fillna(out["symbol"])
    out["sector"] = out["gics_sector"].fillna(out["simfin_sector"]).fillna("Unknown")
    out["industry"] = out["gics_industry"].fillna(out["simfin_industry"]).fillna("Unknown")
    return out


def split_table(positions: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Aggregate portfolio weight and market value by a metadata column."""
    if positions.empty:
        return pd.DataFrame(columns=[group_col, "weight", "market_value_base", "names"])
    return (
        positions.groupby(group_col, dropna=False)
        .agg(
            weight=("weight", "sum"),
            market_value_base=("market_value_base", "sum"),
            names=("symbol", "nunique"),
        )
        .reset_index()
        .sort_values("weight", ascending=False)
    )


def daily_price_performance(
    current_positions: pd.DataFrame,
    prior_positions: pd.DataFrame,
    prior_net_liq_value: float | None,
) -> tuple[pd.DataFrame, dict[str, float | int | None]]:
    """Estimate daily price P&L by comparing current holdings to prior snapshot prices.

    This intentionally avoids transaction assumptions. It measures mark-to-market
    movement for positions held in the current snapshot that also have a prior
    snapshot price.
    """
    empty_summary = {
        "pnl_base": 0.0,
        "return_pct": None,
        "coverage_weight": 0.0,
        "matched_names": 0,
        "unmatched_names": 0,
        "changed_quantity_names": 0,
    }
    if current_positions.empty or prior_positions.empty:
        return pd.DataFrame(), empty_summary

    current = _performance_frame(current_positions, "current")
    prior = _performance_frame(prior_positions, "prior")
    merged = current.merge(
        prior[["perf_key", "quantity_prior", "unit_value_base_prior"]],
        on="perf_key",
        how="left",
    )
    matched = (
        merged["unit_value_base_current"].notna()
        & merged["unit_value_base_prior"].notna()
        & (merged["unit_value_base_prior"] != 0)
    )
    merged["daily_pnl_base"] = pd.NA
    merged["daily_return"] = pd.NA
    merged["daily_contribution"] = pd.NA
    merged.loc[matched, "daily_pnl_base"] = (
        merged.loc[matched, "quantity_current"]
        * (merged.loc[matched, "unit_value_base_current"] - merged.loc[matched, "unit_value_base_prior"])
    )
    merged.loc[matched, "daily_return"] = (
        merged.loc[matched, "unit_value_base_current"] / merged.loc[matched, "unit_value_base_prior"] - 1.0
    )
    if prior_net_liq_value:
        merged.loc[matched, "daily_contribution"] = merged.loc[matched, "daily_pnl_base"] / prior_net_liq_value
    merged["quantity_change"] = merged["quantity_current"] - merged["quantity_prior"].fillna(0.0)
    merged["has_prior_price"] = matched

    current_value = merged["market_value_base"].sum()
    covered_value = merged.loc[matched, "market_value_base"].sum()
    pnl_base = merged.loc[matched, "daily_pnl_base"].sum()
    summary = {
        "pnl_base": float(pnl_base),
        "return_pct": float(pnl_base / prior_net_liq_value) if prior_net_liq_value else None,
        "coverage_weight": float(covered_value / current_value) if current_value else 0.0,
        "matched_names": int(matched.sum()),
        "unmatched_names": int((~matched).sum()),
        "changed_quantity_names": int((merged["quantity_change"].abs() > 1e-9).sum()),
    }
    return merged.sort_values("daily_pnl_base", ascending=False, na_position="last"), summary


def weighted_factor_exposures(
    positions: pd.DataFrame,
    exposures: pd.DataFrame,
    pretty_names: dict[str, str],
    factor_groups: dict[str, str],
) -> tuple[pd.DataFrame, float]:
    """Calculate portfolio-level factor exposures from security-level exposures."""
    if positions.empty or exposures.empty:
        return pd.DataFrame(), 0.0

    factor_positions = positions.dropna(subset=["isin", "weight"]).copy()
    if factor_positions.empty:
        return pd.DataFrame(), 0.0

    weights = factor_positions[["isin", "weight"]].drop_duplicates("isin")
    coverage = weights[weights["isin"].isin(exposures["isin"].unique())]["weight"].sum()
    merged = exposures.merge(weights, on="isin", how="inner")
    if merged.empty:
        return pd.DataFrame(), 0.0

    merged["weighted_exposure"] = merged["exposure"] * merged["weight"]
    out = merged.groupby("factor_id", as_index=False).agg(exposure=("weighted_exposure", "sum"))
    out["abs_exposure"] = out["exposure"].abs()
    out["factor_name"] = out["factor_id"].map(pretty_names).fillna(out["factor_id"])
    out["group"] = out["factor_id"].map(factor_groups).fillna("Model")
    return out.sort_values("abs_exposure", ascending=False), float(coverage)


def _performance_frame(positions: pd.DataFrame, suffix: str) -> pd.DataFrame:
    out = positions.copy()
    out["perf_key"] = out.apply(_performance_key, axis=1)
    out = out.dropna(subset=["perf_key", "quantity", "market_value_base"])
    out = out[out["quantity"] != 0].copy()
    out[f"quantity_{suffix}"] = out["quantity"].astype(float)
    out[f"unit_value_base_{suffix}"] = out["market_value_base"].astype(float) / out[f"quantity_{suffix}"]
    keep = [
        "perf_key",
        f"quantity_{suffix}",
        f"unit_value_base_{suffix}",
    ]
    if suffix == "current":
        for column in ["display_name", "sector", "industry", "weight"]:
            if column not in out.columns:
                out[column] = pd.NA
        keep = [
            "perf_key", "symbol", "isin", "display_name", "sector", "industry",
            "currency", "market_value_base", "weight", f"quantity_{suffix}",
            f"unit_value_base_{suffix}",
        ]
    return out[keep]


def _performance_key(row: pd.Series) -> str | None:
    isin = row.get("isin")
    if pd.notna(isin) and str(isin).strip():
        return f"isin:{str(isin).strip().upper()}"
    symbol = row.get("symbol")
    if pd.isna(symbol) or not str(symbol).strip():
        return None
    currency = row.get("currency")
    currency_part = "" if pd.isna(currency) else str(currency).strip().upper()
    return f"symbol:{str(symbol).strip().upper()}:{currency_part}"
