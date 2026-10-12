from zipfile import ZipFile

import pandas as pd
import pytest

from haze_forecast.cnemc_2014 import (
    aggregate_cnemc_hourly,
    load_cnemc_daily_directory,
    load_cnemc_zip,
)


def test_cnemc_aggregation_uses_chengdu_station_median_and_local_timezone():
    raw = pd.DataFrame(
        {
            "date": [20140513, 20140513, 20140513, 20140513],
            "hour": [0, 0, 1, 1],
            "type": ["AQI", "PM2.5", "AQI", "PM2.5"],
            "1431A": [80, 30, 100, 40],
            "1432A": [100, 50, None, 60],
            "other": [999, 999, 999, 999],
        }
    )

    hourly = aggregate_cnemc_hourly(raw, station_ids=["1431A", "1432A"])

    assert hourly["timestamp"].tolist() == [
        pd.Timestamp("2014-05-13 00:00:00+08:00"),
        pd.Timestamp("2014-05-13 01:00:00+08:00"),
    ]
    assert hourly["aqi"].tolist() == [90.0, 100.0]
    assert hourly["pm2_5"].tolist() == [40.0, 50.0]
    assert hourly["pm10"].isna().all()


def test_cnemc_aggregation_rejects_missing_required_columns_or_station_ids():
    with pytest.raises(ValueError, match="missing"):
        aggregate_cnemc_hourly(pd.DataFrame({"date": [20140513]}), station_ids=["1431A"])
    with pytest.raises(ValueError, match="station"):
        aggregate_cnemc_hourly(
            pd.DataFrame({"date": [20140513], "hour": [0], "type": ["AQI"]}),
            station_ids=["1431A"],
        )


def test_cnemc_zip_loader_combines_daily_files_in_time_order(tmp_path):
    archive = tmp_path / "cnemc.zip"
    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr(
            "2014/china_sites_20140514.csv",
            "date,hour,type,1431A\n20140514,0,AQI,80\n",
        )
        zip_file.writestr(
            "2014/china_sites_20140513.csv",
            "date,hour,type,1431A\n20140513,0,AQI,70\n",
        )

    hourly = load_cnemc_zip(archive, station_ids=["1431A"])

    assert hourly["aqi"].tolist() == [70.0, 80.0]
    assert hourly["timestamp"].tolist()[0] == pd.Timestamp("2014-05-13 00:00:00+08:00")


def test_cnemc_zip_loader_accepts_other_years_and_nonstandard_archive_roots(tmp_path):
    archive = tmp_path / "cnemc-2022.zip"
    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr(
            "archive_root/china_sites_20220102.csv",
            "date,hour,type,1431A\n20220102,0,AQI,90\n",
        )

    hourly = load_cnemc_zip(archive, station_ids=["1431A"])

    assert hourly["timestamp"].tolist() == [pd.Timestamp("2022-01-02 00:00:00+08:00")]


def test_cnemc_daily_directory_loader_combines_downloaded_files_in_time_order(tmp_path):
    (tmp_path / "china_sites_20230102.csv").write_text(
        "date,hour,type,1431A\n20230102,0,AQI,90\n", encoding="utf-8"
    )
    (tmp_path / "china_sites_20230101.csv").write_text(
        "date,hour,type,1431A\n20230101,0,AQI,80\n", encoding="utf-8"
    )

    hourly = load_cnemc_daily_directory(tmp_path, station_ids=["1431A"])

    assert hourly["aqi"].tolist() == [80.0, 90.0]
