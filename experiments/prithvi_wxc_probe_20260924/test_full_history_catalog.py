from full_history_catalog import build_coverage, select_required_granules


def granule(date, name="sample"):
    return {"id": name, "time_start": date + "T00:00:00Z", "time_end": date + "T23:59:59Z"}


def test_dates_are_selected_by_decoded_utc_date():
    rows = [granule("2020-01-01"), granule("2020-01-02", "other")]
    assert len(select_required_granules(rows, {"2020-01-01"})) == 1


def test_missing_is_separate_from_failed_query():
    dates = ["2020-01-01", "2020-01-02"]
    result = build_coverage(dates, {"ok": {"complete": True, "granules": [granule(dates[0])]}, "failed": {"complete": False, "granules": []}})
    assert result["products"]["ok"]["missing_nominal_dates"] == [dates[1]]
    assert result["products"]["failed"]["missing_nominal_dates"] == []
    assert result["products"]["failed"]["unresolved_nominal_dates"] == dates
    assert result["all_products_metadata_dates"] == []


def test_metadata_never_claims_download_or_preprocessing():
    result = build_coverage(["2020-01-01"], {"a": {"complete": True, "granules": [granule("2020-01-01")]}})
    assert result["data_downloaded"] is False
    assert result["preprocessing_complete"] is False
    assert result["dependency_scope"] == "nominal-required UTC dates only"
    assert result["all_products_metadata_dates"] == ["2020-01-01"]
