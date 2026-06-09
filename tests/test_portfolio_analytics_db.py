from __future__ import annotations

import sqlite3

import pytest

from brokers.broker_sync import build_degiro_snapshot, build_ibkr_flex_snapshot, build_ibkr_snapshot
from brokers.models import AccountValue, CashBalance, Position
from brokers.persistence import get_account, insert_snapshot
from brokers.schema import init_db


def test_schema_seeds_accounts_and_inserts_snapshot(tmp_path):
    db_path = tmp_path / "portfolio_analytics.db"
    init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        account = get_account(conn, "ibkr_us_quant")
        assert account is not None
        assert account["broker"] == "ibkr"
        assert account["strategy_id"] == "core_active"

        snapshot = {
            "snapshot_id": "test_snapshot",
            "portfolio_id": "ibkr_us_quant",
            "data_date": "2026-06-02",
            "snapshot_at": "2026-06-02T12:00:00+00:00",
            "base_currency": "EUR",
            "net_liq_value": 1000.0,
            "gross_position_value": 900.0,
            "cash_value": 100.0,
            "source": "test",
            "created_at": "2026-06-02T12:00:00+00:00",
        }
        items = [
            {
                "snapshot_id": "test_snapshot",
                "item_type": "POSITION",
                "symbol": "AAPL",
                "isin": None,
                "name": None,
                "currency": "USD",
                "quantity": 2.0,
                "price": 100.0,
                "market_value": 200.0,
                "market_value_base": 180.0,
                "weight": 0.18,
                "fx_rate_to_base": 0.9,
                "cost_basis": None,
            },
            {
                "snapshot_id": "test_snapshot",
                "item_type": "CASH",
                "symbol": "CASH",
                "isin": None,
                "name": "EUR Cash",
                "currency": "EUR",
                "quantity": 100.0,
                "price": 1.0,
                "market_value": 100.0,
                "market_value_base": 100.0,
                "weight": 0.10,
                "fx_rate_to_base": 1.0,
                "cost_basis": None,
            },
        ]

        insert_snapshot(conn, snapshot, items)
        conn.commit()

        assert conn.execute("SELECT COUNT(*) FROM portfolio_snapshots").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM portfolio_snapshot_items").fetchone()[0] == 2
        assert tuple(conn.execute(
            "SELECT market_value_base, weight FROM portfolio_snapshot_items WHERE symbol = 'AAPL'"
        ).fetchone()) == (180.0, 0.18)

        replacement_items = [
            {
                "snapshot_id": "test_snapshot",
                "item_type": "POSITION",
                "symbol": "MSFT",
                "isin": "US5949181045",
                "name": "Microsoft",
                "currency": "USD",
                "quantity": 1.0,
                "price": 200.0,
                "market_value": 200.0,
                "market_value_base": 190.0,
                "weight": 0.19,
                "fx_rate_to_base": 0.95,
                "cost_basis": None,
            },
        ]
        insert_snapshot(conn, snapshot, replacement_items)
        conn.commit()

        assert conn.execute("SELECT COUNT(*) FROM portfolio_snapshot_items").fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM portfolio_snapshot_items WHERE symbol = 'AAPL'"
        ).fetchone()[0] == 0
        assert tuple(conn.execute(
            "SELECT market_value_base, weight FROM portfolio_snapshot_items WHERE symbol = 'MSFT'"
        ).fetchone()) == (190.0, 0.19)


def test_schema_migrates_existing_snapshot_items_table_without_weight(tmp_path):
    db_path = tmp_path / "portfolio_analytics.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE portfolio_snapshot_items (
                snapshot_id       TEXT NOT NULL,
                item_type         TEXT NOT NULL,
                symbol            TEXT,
                isin              TEXT,
                name              TEXT,
                currency          TEXT,
                quantity          REAL NOT NULL,
                price             REAL,
                market_value      REAL,
                market_value_base REAL,
                fx_rate_to_base   REAL,
                cost_basis        REAL,
                PRIMARY KEY (snapshot_id, item_type, symbol, currency)
            )
            """
        )
        conn.commit()

    init_db(db_path, seed=False)

    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(portfolio_snapshot_items)")}
        assert "weight" in columns


def test_insert_snapshot_replaces_same_portfolio_date_source(tmp_path):
    db_path = tmp_path / "portfolio_analytics.db"
    init_db(db_path)

    def snapshot(snapshot_id: str, snapshot_at: str, nav: float) -> dict:
        return {
            "snapshot_id": snapshot_id,
            "portfolio_id": "ibkr_us_quant",
            "data_date": "2026-06-03",
            "snapshot_at": snapshot_at,
            "base_currency": "EUR",
            "net_liq_value": nav,
            "gross_position_value": nav,
            "cash_value": 0.0,
            "source": "ibkr_flex",
            "created_at": snapshot_at,
        }

    def item(snapshot_id: str, symbol: str) -> dict:
        return {
            "snapshot_id": snapshot_id,
            "item_type": "POSITION",
            "symbol": symbol,
            "isin": None,
            "name": None,
            "currency": "USD",
            "quantity": 1.0,
            "price": 100.0,
            "market_value": 100.0,
            "market_value_base": 100.0,
            "weight": 0.10,
            "fx_rate_to_base": 1.0,
            "cost_basis": None,
        }

    with sqlite3.connect(db_path) as conn:
        insert_snapshot(
            conn,
            snapshot("ibkr_us_quant_old", "2026-06-04T20:00:00+00:00", 1000.0),
            [item("ibkr_us_quant_old", "AAPL")],
        )
        insert_snapshot(
            conn,
            snapshot("ibkr_us_quant_new", "2026-06-04T22:00:00+00:00", 1001.0),
            [item("ibkr_us_quant_new", "MSFT")],
        )
        conn.commit()

        rows = conn.execute(
            "SELECT snapshot_id, net_liq_value FROM portfolio_snapshots"
        ).fetchall()
        assert rows == [("ibkr_us_quant_new", 1001.0)]
        item_rows = conn.execute(
            "SELECT snapshot_id, symbol FROM portfolio_snapshot_items"
        ).fetchall()
        assert item_rows == [("ibkr_us_quant_new", "MSFT")]


class FakeIBKRClient:
    def get_account_value(self):
        return AccountValue(amount=1000.0, currency="EUR", amount_usd=1100.0, fx_rate_to_usd=1.1)

    def get_positions(self):
        return [
            Position(symbol="AAPL", quantity=2.0, currency="USD"),
            Position(symbol="SAP", quantity=1.0, currency="EUR", isin="DE0007164600", name="SAP"),
        ]

    def get_price(self, symbol):
        return {"AAPL": 100.0, "SAP": 150.0}[symbol]

    def get_cash(self):
        return [
            CashBalance(currency="USD", amount=55.0),
            CashBalance(currency="USD", amount=55.0),
            CashBalance(currency="EUR", amount=50.0),
            CashBalance(currency="BASE", amount=1000.0),
        ]


def test_build_ibkr_snapshot_enriches_metadata_aggregates_cash_and_calculates_weights(monkeypatch):
    monkeypatch.setattr(
        "brokers.broker_sync.load_symbol_metadata",
        lambda: {"AAPL": {"isin": "US0378331005", "company_name": "Apple Inc"}},
    )
    account = {"portfolio_id": "ibkr_us_quant", "base_currency": "EUR"}

    snapshot, items = build_ibkr_snapshot(FakeIBKRClient(), account, "2026-06-02T12:00:00+00:00")

    assert snapshot["snapshot_id"] == "ibkr_us_quant_20260602T120000Z0000"
    assert snapshot["data_date"] == "2026-06-02"
    assert snapshot["gross_position_value"] == pytest.approx(331.8181818181818)
    assert snapshot["cash_value"] == 150.0

    by_key = {(item["item_type"], item["symbol"], item["currency"]): item for item in items}
    aapl = by_key[("POSITION", "AAPL", "USD")]
    assert aapl["isin"] == "US0378331005"
    assert aapl["name"] == "Apple Inc"
    assert aapl["market_value_base"] == pytest.approx(181.8181818181818)
    assert aapl["weight"] == pytest.approx(0.1818181818181818)
    assert by_key[("CASH", "CASH", "USD")]["quantity"] == 110.0
    assert by_key[("CASH", "CASH", "USD")]["weight"] == 0.10
    assert ("CASH", "CASH", "BASE") not in by_key


class FakeDegiroClient:
    def get_account_value(self):
        return AccountValue(amount=1000.0, currency="EUR", amount_usd=None, fx_rate_to_usd=None)

    def get_position_rows(self):
        return [
            {
                "product_id": 1,
                "symbol": "ASML",
                "isin": "NL0010273215",
                "name": "ASML Holding",
                "currency": "EUR",
                "quantity": 1.0,
                "price": 500.0,
                "market_value": 500.0,
                "market_value_base": 500.0,
                "fx_rate_to_base": 1.0,
            },
            {
                "product_id": 2,
                "symbol": "NVDA",
                "isin": "US67066G1040",
                "name": "NVIDIA",
                "currency": "USD",
                "quantity": 2.0,
                "price": 200.0,
                "market_value": 400.0,
                "market_value_base": 360.0,
                "fx_rate_to_base": 0.9,
            },
        ]

    def get_cash(self):
        return [CashBalance(currency="EUR", amount=140.0)]


def test_build_degiro_snapshot_uses_base_values_and_calculates_weights():
    account = {"portfolio_id": "degiro_us_opportunistic", "base_currency": "EUR"}

    snapshot, items = build_degiro_snapshot(FakeDegiroClient(), account, "2026-06-02T12:00:00+00:00")

    assert snapshot["source"] == "degiro"
    assert snapshot["net_liq_value"] == 1000.0
    assert snapshot["gross_position_value"] == 860.0
    assert snapshot["cash_value"] == 140.0

    by_symbol = {item["symbol"]: item for item in items}
    assert by_symbol["ASML"]["weight"] == 0.50
    assert by_symbol["NVDA"]["weight"] == 0.36
    assert by_symbol["CASH"]["weight"] == 0.14


def test_build_ibkr_flex_snapshot_parses_positions_cash_and_net_liq():
    xml = """
    <FlexQueryResponse>
      <FlexStatements>
        <FlexStatement accountId="DUQ516120" toDate="20260603">
          <NetAssetValue currency="EUR" total="1000.00" />
          <OpenPositions>
            <OpenPosition symbol="AAPL" description="Apple Inc" isin="US0378331005"
                          currency="USD" position="2" markPrice="100"
                          positionValue="200" positionValueInBase="184"
                          fxRateToBase="0.92" costBasisMoney="150" />
            <OpenPosition symbol="SAP" description="SAP SE" isin="DE0007164600"
                          currency="EUR" position="1" markPrice="150"
                          positionValue="150" />
          </OpenPositions>
          <CashReport>
            <CashReportCurrency currency="USD" endingCash="20" endingCashInBase="18.4" fxRateToBase="0.92" />
            <CashReportCurrency currency="EUR" endingCash="647.6" />
          </CashReport>
        </FlexStatement>
      </FlexStatements>
    </FlexQueryResponse>
    """
    account = {"portfolio_id": "ibkr_us_quant", "base_currency": "EUR"}

    snapshot, items = build_ibkr_flex_snapshot(xml, account, "2026-06-03T20:00:00+00:00")

    assert snapshot["snapshot_id"] == "ibkr_us_quant_20260603T200000Z0000"
    assert snapshot["source"] == "ibkr_flex"
    assert snapshot["data_date"] == "2026-06-03"
    assert snapshot["net_liq_value"] == 1000.0
    assert snapshot["gross_position_value"] == 334.0
    assert snapshot["cash_value"] == 666.0

    by_key = {(item["item_type"], item["symbol"], item["currency"]): item for item in items}
    assert by_key[("POSITION", "AAPL", "USD")]["market_value_base"] == 184.0
    assert by_key[("POSITION", "AAPL", "USD")]["weight"] == 0.184
    assert by_key[("POSITION", "SAP", "EUR")]["market_value_base"] == 150.0
    assert by_key[("CASH", "CASH", "USD")]["market_value_base"] == 18.4
    assert by_key[("CASH", "CASH", "EUR")]["weight"] == pytest.approx(0.6476)


def test_build_ibkr_flex_snapshot_prefers_base_cash_and_infers_net_liq_from_percent_nav():
    xml = """
    <FlexQueryResponse>
      <FlexStatements>
        <FlexStatement accountId="DUQ516120" toDate="20260603">
          <OpenPositions>
            <OpenPosition symbol="AAPL" description="Apple Inc" isin="US0378331005"
                          currency="USD" position="5" markPrice="315.2"
                          positionValue="1576" fxRateToBase="0.85975"
                          percentOfNAV="4.516553333333333" />
            <OpenPosition symbol="SAP" description="SAP SE" isin="DE0007164600"
                          currency="EUR" position="1" markPrice="150"
                          positionValue="150" percentOfNAV="0.50" />
          </OpenPositions>
          <CashReport>
            <CashReportCurrency currency="BASE_SUMMARY" levelOfDetail="BaseCurrency" endingCash="-94.08" />
            <CashReportCurrency currency="EUR" levelOfDetail="Currency" endingCash="30000" />
            <CashReportCurrency currency="USD" levelOfDetail="Currency" endingCash="-35003.29" />
          </CashReport>
        </FlexStatement>
      </FlexStatements>
    </FlexQueryResponse>
    """
    account = {"portfolio_id": "ibkr_us_quant", "base_currency": "EUR"}

    with pytest.raises(RuntimeError, match="recognized NAV section"):
        build_ibkr_flex_snapshot(xml, account, "2026-06-03T20:00:00+00:00")

    snapshot, items = build_ibkr_flex_snapshot(
        xml,
        account,
        "2026-06-03T20:00:00+00:00",
        allow_inferred_nav=True,
    )

    assert snapshot["net_liq_value"] == pytest.approx(30000.0)
    assert snapshot["cash_value"] == -94.08
    cash = [item for item in items if item["item_type"] == "CASH"]
    assert len(cash) == 1
    assert cash[0]["currency"] == "EUR"
    assert cash[0]["market_value_base"] == -94.08


def test_build_ibkr_flex_snapshot_uses_equity_summary_in_base_for_nav():
    xml = """
    <FlexQueryResponse>
      <FlexStatements>
        <FlexStatement accountId="DUQ516120" toDate="20260603">
          <EquitySummaryInBase>
            <EquitySummaryByReportDateInBase currency="EUR" reportDate="20260602" total="30027.49" />
          </EquitySummaryInBase>
          <OpenPositions>
            <OpenPosition symbol="AAPL" description="Apple Inc" isin="US0378331005"
                          currency="USD" position="5" markPrice="315.2"
                          positionValue="1576" fxRateToBase="0.85975"
                          percentOfNAV="4.51" />
          </OpenPositions>
          <CashReport>
            <CashReportCurrency currency="BASE_SUMMARY" levelOfDetail="BaseCurrency" endingCash="-94.08" />
            <CashReportCurrency currency="EUR" levelOfDetail="Currency" endingCash="30000" />
            <CashReportCurrency currency="USD" levelOfDetail="Currency" endingCash="-35003.29" />
          </CashReport>
        </FlexStatement>
      </FlexStatements>
    </FlexQueryResponse>
    """
    account = {"portfolio_id": "ibkr_us_quant", "base_currency": "EUR"}

    snapshot, items = build_ibkr_flex_snapshot(xml, account, "2026-06-03T20:00:00+00:00")

    assert snapshot["net_liq_value"] == 30027.49
    assert snapshot["cash_value"] == -94.08
    cash = [item for item in items if item["item_type"] == "CASH"]
    assert len(cash) == 1
