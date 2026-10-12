import importlib.util
from pathlib import Path

import pandas as pd
import pytest

PATH = Path(__file__).with_name("support_catalog.py")
spec = importlib.util.spec_from_file_location("support_catalog", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_instant_and_average_dates_include_cross_day_support():
    frame = pd.DataFrame({"weather_time_utc": ["2020-01-02T00:00Z"],
                          "source_left_utc": ["2020-01-01T23:30Z"],
                          "source_right_utc": ["2020-01-02T00:30Z"]})
    result = module.required_dates_by_product(frame)
    assert result["M2I3NVASM"] == ["2020-01-02"]
    assert result["M2I1NXASM"] == ["2020-01-02"]
    for name in module.AVERAGED:
        assert result[name] == ["2020-01-01", "2020-01-02"]


def test_partial_query_does_not_claim_missing_or_complete():
    required = {name: ["2020-01-01"] for name in module.PRODUCTS}
    products = {"M2I1NXASM": {"complete": False, "granules": []}}
    report = module.build_coverage(required, products)
    assert not report["all_expected_products_query_complete"]
    assert not report["all_required_dates_have_metadata"]
    assert report["products"]["M2I1NXASM"]["missing_required_dates"] == []
    assert report["products"]["M2I1NXASM"]["unresolved_required_dates"] == ["2020-01-01"]
    products["M2I1NXASM"]["complete"] = True
    assert module.build_coverage(required, products)["products"]["M2I1NXASM"]["missing_required_dates"] == ["2020-01-01"]


def test_duplicate_pagination_is_rejected():
    granule = {"id": "one", "time_start": "2020-01-01T00:00:00Z"}
    def fetch(*args):
        return {"feed": {"entry": [granule]}}, {"CMR-Hits": "2"}
    product = {"complete": False, "granules": []}
    with pytest.raises(ValueError, match="Duplicate"):
        module.collect_pages(None, "collection", ["2020-01-01"], product, fetch)
    assert not product["complete"]


def test_changed_hit_count_is_rejected():
    calls = iter([1, 2])
    def fetch(*args):
        page = next(calls)
        return {"feed": {"entry": [{"id": str(page), "time_start": "2020-01-01T00:00:00Z"}]}}, {"CMR-Hits": str(page + 1)}
    with pytest.raises(ValueError, match="changed"):
        module.collect_pages(None, "collection", ["2020-01-01"], {"complete": False, "granules": []}, fetch)


def test_complete_products_and_year_counts():
    dates = {name: ["2019-12-31", "2020-01-01"] for name in module.PRODUCTS}
    products = {name: {"complete": True, "granules": [
        {"id": "a", "time_start": "2019-12-31T00:00Z"},
        {"id": "b", "time_start": "2020-01-01T00:00Z"}]} for name in module.PRODUCTS}
    report = module.build_coverage(dates, products)
    assert report["all_required_dates_have_metadata"]
    assert not report["data_downloaded"]
    assert not report["preprocessing_complete"]
    assert report["products"]["M2T1NXLND"]["years"]["2019"]["required_dates"] == 1
