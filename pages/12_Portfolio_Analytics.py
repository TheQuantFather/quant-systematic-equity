"""
Portfolio analytics page for broker-synced portfolio snapshots.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brokers.schema import PORTFOLIO_ANALYTICS_DB
from config import RISK_DB, UNIVERSE_DB
from portfolio_analytics import (
    daily_price_performance,
    enrich_snapshot_items,
    nearest_risk_date,
    split_table,
    weighted_factor_exposures,
)
from utils import get_barra_layout, get_db, inject_css


st.set_page_config(page_title="Portfolio Analytics", layout="wide")
inject_css()
st.title("Portfolio Analytics")

_LAYOUT = get_barra_layout()
_PRETTY = _LAYOUT["pretty"]
_FACTOR_GROUP = _LAYOUT["factor_group"]


@st.cache_data(ttl=300)
def load_accounts() -> pd.DataFrame:
    if not PORTFOLIO_ANALYTICS_DB.exists():
        return pd.DataFrame()
    with get_db(PORTFOLIO_ANALYTICS_DB) as conn:
        return pd.read_sql(
            """
            SELECT portfolio_id, broker, account_id, portfolio_name, portfolio_type,
                   base_currency, strategy_id, benchmark_id, is_active
            FROM accounts
            ORDER BY portfolio_name
            """,
            conn,
        )


@st.cache_data(ttl=300)
def load_snapshots(portfolio_id: str) -> pd.DataFrame:
    with get_db(PORTFOLIO_ANALYTICS_DB) as conn:
        return pd.read_sql(
            """
            SELECT snapshot_id, portfolio_id, data_date, snapshot_at, base_currency,
                   net_liq_value, gross_position_value, cash_value, source
            FROM portfolio_snapshots
            WHERE portfolio_id = ?
            ORDER BY data_date DESC, snapshot_at DESC
            """,
            conn,
            params=(portfolio_id,),
        )


@st.cache_data(ttl=300)
def load_snapshot_items(snapshot_id: str) -> pd.DataFrame:
    with get_db(PORTFOLIO_ANALYTICS_DB) as conn:
        return pd.read_sql(
            """
            SELECT snapshot_id, item_type, symbol, isin, name, currency, quantity,
                   price, market_value, market_value_base, weight, fx_rate_to_base,
                   cost_basis
            FROM portfolio_snapshot_items
            WHERE snapshot_id = ?
            ORDER BY ABS(COALESCE(weight, 0)) DESC
            """,
            conn,
            params=(snapshot_id,),
        )


@st.cache_data(ttl=300)
def load_prior_snapshot(portfolio_id: str, data_date: str) -> pd.DataFrame:
    with get_db(PORTFOLIO_ANALYTICS_DB) as conn:
        return pd.read_sql(
            """
            SELECT snapshot_id, portfolio_id, data_date, snapshot_at, base_currency,
                   net_liq_value, gross_position_value, cash_value, source
            FROM portfolio_snapshots
            WHERE portfolio_id = ?
              AND data_date < ?
            ORDER BY data_date DESC, snapshot_at DESC
            LIMIT 1
            """,
            conn,
            params=(portfolio_id, data_date),
        )


@st.cache_data(ttl=300)
def load_universe_meta() -> pd.DataFrame:
    if not UNIVERSE_DB.exists():
        return pd.DataFrame()
    with get_db(UNIVERSE_DB) as conn:
        return pd.read_sql(
            """
            SELECT isin, ticker, company_name, gics_sector, gics_industry,
                   simfin_sector, simfin_industry
            FROM companies
            """,
            conn,
        )


@st.cache_data(ttl=300)
def load_risk_dates() -> list[str]:
    if not RISK_DB.exists():
        return []
    with get_db(RISK_DB) as conn:
        rows = conn.execute(
            "SELECT DISTINCT snapshot_date FROM factor_exposures ORDER BY snapshot_date"
        ).fetchall()
    return [r[0] for r in rows]


@st.cache_data(ttl=300)
def load_factor_exposures(risk_date: str, isins: tuple[str, ...]) -> pd.DataFrame:
    if not isins or not RISK_DB.exists():
        return pd.DataFrame()
    placeholders = ",".join("?" * len(isins))
    with get_db(RISK_DB) as conn:
        return pd.read_sql(
            f"""
            SELECT security_id AS isin, factor_id, exposure
            FROM factor_exposures
            WHERE snapshot_date = ?
              AND security_id IN ({placeholders})
            """,
            conn,
            params=(risk_date, *isins),
        )


def enrich_items(items: pd.DataFrame) -> pd.DataFrame:
    return enrich_snapshot_items(items, load_universe_meta())


def factor_exposure_table(positions: pd.DataFrame, risk_date: str | None) -> tuple[pd.DataFrame, float]:
    if positions.empty or risk_date is None:
        return pd.DataFrame(), 0.0

    factor_positions = positions.dropna(subset=["isin", "weight"]).copy()
    isins = tuple(sorted(factor_positions["isin"].dropna().unique()))
    exposures = load_factor_exposures(risk_date, isins)
    if exposures.empty:
        return pd.DataFrame(), 0.0
    return weighted_factor_exposures(factor_positions, exposures, _PRETTY, _FACTOR_GROUP)


def fmt_pct(x: float | None) -> str:
    if pd.isna(x):
        return ""
    return f"{x:.2%}"


accounts = load_accounts()
if accounts.empty:
    st.info("No portfolio analytics database found yet. Run broker snapshot sync first.")
    st.stop()
    raise SystemExit

active = accounts[accounts["is_active"].fillna(1).astype(int) == 1].copy()
label_map = {
    f"{row.portfolio_name} ({row.broker})": row.portfolio_id
    for row in active.itertuples(index=False)
}

left, right = st.columns([2, 1])
with left:
    selected_label = st.selectbox("Portfolio", list(label_map.keys()))
if selected_label not in label_map:
    selected_label = next(iter(label_map))
portfolio_id = label_map[selected_label]
portfolio_row = active[active["portfolio_id"] == portfolio_id].iloc[0]
snapshots = load_snapshots(portfolio_id)
if snapshots.empty:
    st.info("No snapshots available for this portfolio yet.")
    st.stop()
    raise SystemExit

with right:
    snapshot_label = st.selectbox(
        "Snapshot",
        snapshots["snapshot_at"].tolist(),
        format_func=lambda x: (
            f"{snapshots.loc[snapshots['snapshot_at'] == x, 'data_date'].iloc[0]} | "
            f"{snapshots.loc[snapshots['snapshot_at'] == x, 'source'].iloc[0]} | pulled {x[11:19]}"
        ),
    )

snapshot_match = snapshots[snapshots["snapshot_at"] == snapshot_label]
snapshot = snapshot_match.iloc[0] if not snapshot_match.empty else snapshots.iloc[0]
items = enrich_items(load_snapshot_items(snapshot["snapshot_id"]))
positions = items[items["item_type"] == "POSITION"].copy()
cash = items[items["item_type"] == "CASH"].copy()

st.caption(
    f"{portfolio_row['portfolio_type']} | {portfolio_row['broker']} | "
    f"{snapshot['data_date']} | base currency {snapshot['base_currency']}"
)

metric_cols = st.columns(6)
metric_cols[0].metric("Net Liq", f"{snapshot['net_liq_value']:,.0f} {snapshot['base_currency']}")
metric_cols[1].metric("Gross", f"{snapshot['gross_position_value']:,.0f} {snapshot['base_currency']}")
metric_cols[2].metric("Cash", f"{snapshot['cash_value']:,.0f} {snapshot['base_currency']}")
metric_cols[3].metric("Positions", f"{len(positions):,}")
metric_cols[4].metric("Gross Weight", fmt_pct(positions["weight"].sum()))
metric_cols[5].metric("Cash Weight", fmt_pct(cash["weight"].sum() if not cash.empty else 0.0))

st.divider()

tab_holdings, tab_performance, tab_splits, tab_factors = st.tabs([
    "Holdings", "Performance", "Splits", "Factor Exposure"
])

with tab_holdings:
    chart_df = positions.sort_values("weight", ascending=False).head(25)
    if not chart_df.empty:
        fig = px.bar(
            chart_df.sort_values("weight"),
            x="weight",
            y="symbol",
            orientation="h",
            color="sector",
            labels={"weight": "Weight", "symbol": ""},
            hover_data=["display_name", "industry", "market_value_base", "quantity"],
        )
        fig.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10))
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    table = items[[
        "item_type", "symbol", "isin", "display_name", "sector", "industry",
        "currency", "quantity", "price", "market_value_base", "weight",
    ]].copy()
    table["weight"] = table["weight"] * 100
    table = table.rename(columns={
        "item_type": "Type",
        "symbol": "Symbol",
        "isin": "ISIN",
        "display_name": "Name",
        "sector": "Sector",
        "industry": "Industry",
        "currency": "Currency",
        "quantity": "Quantity",
        "price": "Price",
        "market_value_base": "Market Value",
        "weight": "Weight",
    })
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Weight": st.column_config.NumberColumn(format="%.2f%%"),
            "Market Value": st.column_config.NumberColumn(format="%.2f"),
            "Price": st.column_config.NumberColumn(format="%.2f"),
        },
    )

with tab_performance:
    prior_snapshot_df = load_prior_snapshot(portfolio_id, str(snapshot["data_date"]))
    if prior_snapshot_df.empty:
        st.info("No prior portfolio snapshot is available for this portfolio.")
    else:
        prior_snapshot = prior_snapshot_df.iloc[0]
        prior_items = enrich_items(load_snapshot_items(prior_snapshot["snapshot_id"]))
        prior_positions = prior_items[prior_items["item_type"] == "POSITION"].copy()
        perf, perf_summary = daily_price_performance(
            positions,
            prior_positions,
            float(prior_snapshot["net_liq_value"]) if pd.notna(prior_snapshot["net_liq_value"]) else None,
        )

        st.caption(
            f"Compared with {prior_snapshot['data_date']} | {prior_snapshot['source']} | "
            "price movement on matched current holdings"
        )
        perf_cols = st.columns(5)
        perf_cols[0].metric("P&L", f"{perf_summary['pnl_base']:,.2f} {snapshot['base_currency']}")
        perf_cols[1].metric("Return", fmt_pct(perf_summary["return_pct"]))
        perf_cols[2].metric("Coverage", fmt_pct(perf_summary["coverage_weight"]))
        perf_cols[3].metric("Matched", f"{perf_summary['matched_names']:,}")
        perf_cols[4].metric("Qty Changed", f"{perf_summary['changed_quantity_names']:,}")

        if perf.empty:
            st.info("No matched holdings with prior prices are available.")
        else:
            chart_df = perf.dropna(subset=["daily_pnl_base"]).copy()
            if not chart_df.empty:
                top_chart = pd.concat([
                    chart_df.sort_values("daily_pnl_base", ascending=False).head(10),
                    chart_df.sort_values("daily_pnl_base", ascending=True).head(10),
                ]).drop_duplicates("perf_key")
                fig = px.bar(
                    top_chart.sort_values("daily_pnl_base"),
                    x="daily_pnl_base",
                    y="symbol",
                    orientation="h",
                    color="sector",
                    labels={"daily_pnl_base": "P&L", "symbol": ""},
                    hover_data=["display_name", "daily_return", "quantity_change"],
                )
                fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

            perf_table = perf[[
                "symbol", "isin", "display_name", "sector", "industry", "currency",
                "quantity_current", "quantity_prior", "quantity_change",
                "market_value_base", "daily_pnl_base", "daily_return",
                "daily_contribution", "has_prior_price",
            ]].copy()
            for pct_col in ["daily_return", "daily_contribution"]:
                perf_table[pct_col] = pd.to_numeric(perf_table[pct_col], errors="coerce") * 100
            perf_table = perf_table.rename(columns={
                "symbol": "Symbol",
                "isin": "ISIN",
                "display_name": "Name",
                "sector": "Sector",
                "industry": "Industry",
                "currency": "Currency",
                "quantity_current": "Quantity",
                "quantity_prior": "Prior Quantity",
                "quantity_change": "Quantity Change",
                "market_value_base": "Market Value",
                "daily_pnl_base": "Daily P&L",
                "daily_return": "Daily Return",
                "daily_contribution": "Contribution",
                "has_prior_price": "Matched",
            })
            st.dataframe(
                perf_table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Market Value": st.column_config.NumberColumn(format="%.2f"),
                    "Daily P&L": st.column_config.NumberColumn(format="%.2f"),
                    "Daily Return": st.column_config.NumberColumn(format="%.2f%%"),
                    "Contribution": st.column_config.NumberColumn(format="%.2f%%"),
                },
            )

with tab_splits:
    sector_df = split_table(positions, "sector")
    industry_df = split_table(positions, "industry").head(25)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Sector")
        if not sector_df.empty:
            fig = px.bar(
                sector_df.sort_values("weight"),
                x="weight",
                y="sector",
                orientation="h",
                labels={"weight": "Weight", "sector": ""},
                hover_data=["market_value_base", "names"],
            )
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            sector_df.assign(weight=lambda x: x["weight"] * 100).rename(
                columns={"sector": "Sector", "weight": "Weight", "market_value_base": "Market Value", "names": "Names"}
            ),
            use_container_width=True,
            hide_index=True,
            column_config={"Weight": st.column_config.NumberColumn(format="%.2f%%")},
        )

    with col_b:
        st.subheader("Industry")
        if not industry_df.empty:
            fig = px.bar(
                industry_df.sort_values("weight"),
                x="weight",
                y="industry",
                orientation="h",
                labels={"weight": "Weight", "industry": ""},
                hover_data=["market_value_base", "names"],
            )
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            industry_df.assign(weight=lambda x: x["weight"] * 100).rename(
                columns={"industry": "Industry", "weight": "Weight", "market_value_base": "Market Value", "names": "Names"}
            ),
            use_container_width=True,
            hide_index=True,
            column_config={"Weight": st.column_config.NumberColumn(format="%.2f%%")},
        )

with tab_factors:
    risk_date = nearest_risk_date(str(snapshot["data_date"]), load_risk_dates())
    exposures, coverage = factor_exposure_table(positions, risk_date)
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Risk Snapshot", risk_date or "N/A")
    col_b.metric("Exposure Coverage", fmt_pct(coverage))
    col_c.metric("Factors", f"{len(exposures):,}")

    if exposures.empty:
        st.info("No factor exposures available for the selected snapshot.")
    else:
        top = exposures.head(30)
        fig = px.bar(
            top.sort_values("abs_exposure"),
            x="abs_exposure",
            y="factor_name",
            orientation="h",
            color="group",
            labels={"abs_exposure": "Absolute Exposure", "factor_name": ""},
            hover_data=["factor_id", "exposure", "group"],
        )
        fig.update_layout(height=640, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        factor_table = exposures[["group", "factor_id", "factor_name", "exposure", "abs_exposure"]].copy()
        factor_table = factor_table.rename(columns={
            "group": "Group",
            "factor_id": "Factor ID",
            "factor_name": "Factor",
            "exposure": "Exposure",
            "abs_exposure": "Abs Exposure",
        })
        st.dataframe(
            factor_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Exposure": st.column_config.NumberColumn(format="%.4f"),
                "Abs Exposure": st.column_config.NumberColumn(format="%.4f"),
            },
        )
