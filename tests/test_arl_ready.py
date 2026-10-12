from datetime import date

import pandas as pd
import pytest

from haze_forecast.arl_ready import (
    arl_f003_request_times,
    download_ready_gfs_day,
    parse_ready_download_path,
    parse_ready_job_id,
    ready_gfs_filename,
)


def test_ready_gfs_filename_uses_utc_archive_day():
    assert ready_gfs_filename(date(2020, 12, 23)) == "20201223_gfs0p25"


def test_ready_html_parsers_find_the_job_and_completed_zip():
    job_id = parse_ready_job_id('<input type="HIDDEN" name="proc" value="27399">')

    assert job_id == "27399"
    assert parse_ready_download_path('<a href="/extractout/user/extract_27399.zip">', job_id) == (
        "/extractout/user/extract_27399.zip"
    )
    assert parse_ready_download_path("still running", job_id) is None


def test_ready_job_parser_rejects_an_unexpected_response():
    with pytest.raises(ValueError, match="job id"):
        parse_ready_job_id("missing")


def test_arl_f003_planner_only_keeps_times_before_the_next_three_hour_valid_time():
    timestamps = pd.to_datetime(
        ["2020-12-23T18:00:00Z", "2020-12-23T20:00:00Z", "2020-12-23T21:00:00Z"]
    )

    keep, runs, valid_times = arl_f003_request_times(timestamps)

    assert keep.tolist() == [True, True, False]
    assert runs.tolist() == [pd.Timestamp("2020-12-23T18:00:00Z")] * 3
    assert valid_times.tolist() == [pd.Timestamp("2020-12-23T21:00:00Z")] * 3


def test_arl_f003_planner_accepts_a_timestamp_series():
    keep, runs, valid_times = arl_f003_request_times(
        pd.Series(pd.to_datetime(["2020-12-23T18:00:00Z", "2020-12-23T21:00:00Z"]))
    )

    assert keep.tolist() == [True, False]
    assert runs.tolist() == [pd.Timestamp("2020-12-23T18:00:00Z")] * 2
    assert valid_times.tolist() == [pd.Timestamp("2020-12-23T21:00:00Z")] * 2


def test_ready_subset_rejects_non_three_hour_request_bounds(tmp_path):
    with pytest.raises(ValueError, match="three-hour"):
        download_ready_gfs_day(date(2020, 12, 23), tmp_path, start_hour=1, end_hour=3)
