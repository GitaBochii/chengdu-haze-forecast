from __future__ import annotations

from datetime import date

from haze_forecast.cnemc_download import cnemc_daily_url, inclusive_dates


def test_cnemc_daily_url_uses_the_public_daily_station_contract() -> None:
    assert cnemc_daily_url(date(2025, 1, 2)) == (
        "https://quotsoft.net/air/data/china_sites_20250102.csv"
    )


def test_inclusive_dates_includes_both_requested_boundaries() -> None:
    assert list(inclusive_dates(date(2024, 2, 28), date(2024, 3, 1))) == [
        date(2024, 2, 28),
        date(2024, 2, 29),
        date(2024, 3, 1),
    ]
