import pandas as pd

from pipeline.create_universe import build_companies, _parse_nport_ec_holdings


def _ishares_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Ticker": "LINE",
            "Name": "LINEAGE INC",
            "Sector": "Real Estate",
            "Location": "United States",
            "Exchange": "NASDAQ",
            "Currency": "USD",
            "Weight (%)": 0.1,
            "Market Value": 100.0,
        }
    ])


def _simfin_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "ticker": "LINE",
            "company_name": "LINN ENERGY, INC.",
            "isin": "US000OLD0000",
            "cik": 1326428,
            "fiscal_year_end": 12,
            "num_employees": 100,
            "business_summary": "Old issuer",
            "simfin_id": 640898,
            "simfin_sector": "Energy",
            "simfin_industry": "Oil & Gas",
        }
    ])


def test_build_companies_can_exclude_stale_simfin_ticker_reuse():
    companies = build_companies(
        [(_ishares_frame(), "2026-05-04", "russell_1000")],
        _simfin_frame(),
        patch={"LINE": "US53566V1061"},
        alias={},
        simfin_exclude={"LINE"},
    )

    row = companies.iloc[0]
    assert row["isin"] == "US53566V1061"
    assert row["ticker"] == "LINE"
    assert row["company_name"] == "LINEAGE INC"
    assert pd.isna(row["cik"])
    assert pd.isna(row["simfin_id"])


def test_build_companies_uses_simfin_when_ticker_not_excluded():
    companies = build_companies(
        [(_ishares_frame(), "2026-05-04", "russell_1000")],
        _simfin_frame(),
        patch={"LINE": "US53566V1061"},
        alias={},
        simfin_exclude=set(),
    )

    row = companies.iloc[0]
    assert row["isin"] == "US53566V1061"
    assert row["company_name"] == "LINN ENERGY, INC."
    assert row["cik"] == "1326428"
    assert row["simfin_id"] == 640898


def test_parse_nport_ec_holdings_extracts_security_metadata():
    xml = b"""
    <edgarSubmission xmlns="http://www.sec.gov/edgar/nport">
      <formData>
        <invstOrSecs>
          <invstOrSec>
            <name>Example Corp</name>
            <lei>ABC123</lei>
            <title>Example Corp Class A</title>
            <cusip>123456789</cusip>
            <identifiers>
              <isin value="US1234567890"/>
            </identifiers>
            <balance>100.5</balance>
            <units>NS</units>
            <curCd>USD</curCd>
            <valUSD>2500.25</valUSD>
            <pctVal>0.42</pctVal>
            <payoffProfile>Long</payoffProfile>
            <assetCat>EC</assetCat>
            <issuerCat>CORP</issuerCat>
            <invCountry>US</invCountry>
            <isRestrictedSec>N</isRestrictedSec>
            <fairValLevel>1</fairValLevel>
          </invstOrSec>
          <invstOrSec>
            <name>Bond Corp</name>
            <identifiers><isin value="US0000000001"/></identifiers>
            <assetCat>DBT</assetCat>
          </invstOrSec>
        </invstOrSecs>
      </formData>
    </edgarSubmission>
    """

    holdings = _parse_nport_ec_holdings(xml)

    assert len(holdings) == 1
    row = holdings[0]
    assert row["isin"] == "US1234567890"
    assert row["security_name"] == "Example Corp"
    assert row["security_title"] == "Example Corp Class A"
    assert row["cusip"] == "123456789"
    assert row["lei"] == "ABC123"
    assert row["balance"] == 100.5
    assert row["units"] == "NS"
    assert row["currency"] == "USD"
    assert row["market_value"] == 2500.25
    assert row["weight"] == 0.42
    assert row["issuer_category"] == "CORP"
    assert row["investment_country"] == "US"
