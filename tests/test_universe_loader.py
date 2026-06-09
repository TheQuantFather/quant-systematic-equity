import pandas as pd

import universe_loader


def _snapshot_member() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "index_name": "sp500",
            "source_snapshot_date": "2026-06-05",
            "original_isin": "USOLD",
            "raw_weight": 1.0,
            "market_value": 100.0,
            "snapshot_ticker": "ABC",
            "snapshot_company_name": "ABC INC",
            "snapshot_gics_sector": "Information Technology",
            "snapshot_gics_industry_group": None,
            "snapshot_gics_industry": None,
            "snapshot_gics_sub_industry": None,
            "snapshot_simfin_sector": "Technology",
            "snapshot_simfin_industry": "Software",
            "snapshot_country": "United States",
            "snapshot_exchange": "NASDAQ",
            "snapshot_currency": "USD",
            "snapshot_cik": "0000000001",
            "snapshot_cusip": "123456789",
            "snapshot_company_data_date": "2026-05-18",
            "snapshot_company_update_date": "2026-05-18",
            "snapshot_delisted_date": None,
        }
    ])


def _security_master() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "isin": "USNEW",
            "ticker": "ABC",
            "company_name": "ABC INC",
            "gics_sector": "Information Technology",
            "gics_industry_group": "Software & Services",
            "gics_industry": "Software",
            "gics_sub_industry": "Application Software",
            "simfin_sector": "Technology",
            "simfin_industry": "Software",
            "country": "United States",
            "exchange": "NASDAQ",
            "currency": "USD",
            "cik": "0000000001",
            "cusip": "987654321",
            "data_date": "2026-05-30",
            "update_date": "2026-05-30",
            "delisted_date": None,
        }
    ])


def test_live_identity_maps_same_name_ticker_isin_change(monkeypatch):
    monkeypatch.setattr(universe_loader, "_load_security_master", lambda mode, snapshot_date: _security_master())

    out = universe_loader._apply_security_identity(
        _snapshot_member(),
        mode="live",
        snapshot_date="2026-06-05",
        normalize_live_isin=True,
    )

    row = out.iloc[0]
    assert row["isin"] == "USNEW"
    assert row["canonical_isin"] == "USNEW"
    assert row["mapped_from_isin"] == "USOLD"
    assert row["mapped_to_isin"] == "USNEW"
    assert row["identity_status"] == "mapped_to_current_isin"
    assert row["identity_rule"] == "live_same_ticker_same_name_current_isin"
    assert row["identity_confidence"] == 0.9


def test_dual_class_ticker_collision_not_collapsed(monkeypatch):
    """Two co-listed share classes sharing a ticker must stay distinct ISINs.

    Regression for the Clearway (CWEN) class A/C collapse: a live remap of the
    stale class ISIN onto the current canonical ISIN would otherwise produce a
    duplicate ISIN in the member set.
    """
    monkeypatch.setattr(
        universe_loader, "_load_security_master", lambda mode, snapshot_date: _security_master()
    )

    classes = _snapshot_member()
    second = _snapshot_member().iloc[0].to_dict()
    second["original_isin"] = "USNEW"  # the current/canonical ISIN, already a member
    classes = pd.concat([classes, pd.DataFrame([second])], ignore_index=True)

    out = universe_loader._apply_security_identity(
        classes,
        mode="live",
        snapshot_date="2026-06-05",
        normalize_live_isin=True,
    )

    assert out["isin"].nunique() == 2, "dual-class ISINs must not collapse onto one"
    by_orig = out.set_index("original_isin")
    assert by_orig.loc["USOLD", "isin"] == "USOLD"
    assert by_orig.loc["USOLD", "identity_status"] == "ticker_collision_kept_snapshot_isin"
    assert by_orig.loc["USOLD", "identity_rule"] == "live_remap_suppressed_target_already_present"
    assert by_orig.loc["USNEW", "isin"] == "USNEW"
    assert by_orig.loc["USNEW", "identity_status"] == "current_isin"


def test_point_in_time_identity_keeps_snapshot_isin(monkeypatch):
    monkeypatch.setattr(universe_loader, "_load_security_master", lambda mode, snapshot_date: _security_master())

    out = universe_loader._apply_security_identity(
        _snapshot_member(),
        mode="point_in_time",
        snapshot_date="2026-06-05",
        normalize_live_isin=False,
    )

    row = out.iloc[0]
    assert row["isin"] == "USOLD"
    assert row["canonical_isin"] == "USOLD"
    assert row["mapped_from_isin"] == ""
    assert row["mapped_to_isin"] == ""
    assert row["identity_status"] == "snapshot_isin"
    assert row["identity_rule"] == "point_in_time_snapshot_isin"
    assert row["identity_confidence"] == 1.0


def test_snapshot_master_without_current_ticker_match_is_not_missing(monkeypatch):
    monkeypatch.setattr(
        universe_loader,
        "_load_security_master",
        lambda mode, snapshot_date: pd.DataFrame(columns=_security_master().columns),
    )

    member = _snapshot_member()
    member["snapshot_has_security_master"] = True
    out = universe_loader._apply_security_identity(
        member,
        mode="live",
        snapshot_date="2026-06-05",
        normalize_live_isin=True,
    )

    row = out.iloc[0]
    assert row["isin"] == "USOLD"
    assert row["identity_status"] == "snapshot_isin"
    assert row["identity_rule"] == "snapshot_isin_no_current_ticker_match"
    assert row["identity_confidence"] == 0.8


def test_missing_snapshot_master_stays_blocked(monkeypatch):
    monkeypatch.setattr(
        universe_loader,
        "_load_security_master",
        lambda mode, snapshot_date: pd.DataFrame(columns=_security_master().columns),
    )

    member = _snapshot_member()
    member["snapshot_has_security_master"] = False
    member["snapshot_ticker"] = None
    member["snapshot_company_name"] = None
    out = universe_loader._apply_security_identity(
        member,
        mode="live",
        snapshot_date="2026-06-05",
        normalize_live_isin=True,
    )

    row = out.iloc[0]
    assert row["identity_status"] == "missing_security_master"
    assert row["identity_rule"] == "missing_security_master"
    assert row["identity_confidence"] == 0.0
