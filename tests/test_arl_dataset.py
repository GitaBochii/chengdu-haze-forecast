from datetime import date

import pandas as pd
import pytest

from haze_forecast.arl_dataset import (
    ARL_FEATURE_COLUMNS,
    ARL_SOURCE_COLUMN,
    ARL_VALID_COLUMN,
    arl_feature_names,
    candidate_arl_rows,
    merge_arl_feature_batches,
    recommend_archive_days,
    resolve_archive_days,
    validate_arl_feature_frame,
)


def _candidate_frame() -> pd.DataFrame:
    timestamps = pd.to_datetime(
        [
            "2020-01-04T00:00:00Z",
            "2020-01-04T01:00:00Z",
            "2020-01-04T03:00:00Z",
            "2020-01-05T18:00:00Z",
        ]
    )
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "aqi": [150.0, 140.0, 130.0, 90.0],
            "event_id": [7, 7, 7, 8],
            "remaining_hours": [5.0, 4.0, 3.0, 2.0],
        }
    )


def _batch_frame(times: list[str], event_id: int = 7, temperature: float = 10.0) -> pd.DataFrame:
    timestamps = pd.to_datetime(times)
    runs = timestamps.floor("6h")
    valid = runs + pd.offsets.Hour(3)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "event_id": event_id,
            "remaining_hours": [float(index + 1) for index in range(len(timestamps))],
            "arl_forecast_run_utc": runs,
            "arl_forecast_valid_utc": valid,
            "arl_grid_longitude": 104.0,
            "arl_grid_latitude": 30.75,
        }
    )
    for index, column in enumerate(ARL_FEATURE_COLUMNS):
        frame[column] = temperature + index
    return frame


def test_candidate_rows_keep_only_hours_before_the_next_valid_time():
    candidates = candidate_arl_rows(_candidate_frame(), "2020-01-01", "2020-02-01")

    assert candidates["timestamp"].dt.strftime("%H:%M").tolist() == ["00:00", "01:00"]
    assert candidates["arl_forecast_run_utc"].tolist() == [pd.Timestamp("2020-01-04T00:00:00Z")] * 2
    assert candidates["arl_forecast_valid_utc"].tolist() == [pd.Timestamp("2020-01-04T03:00:00Z")] * 2
    assert candidates["arl_archive_day"].tolist() == [date(2020, 1, 4)] * 2


def test_candidate_rows_drop_unpolluted_and_unlabeled_hours():
    candidates = candidate_arl_rows(_candidate_frame(), "2020-01-01", "2020-02-01")

    assert set(candidates["event_id"]) == {7}


def test_candidate_rows_reject_an_empty_or_inverted_window():
    with pytest.raises(ValueError, match="missing columns"):
        candidate_arl_rows(pd.DataFrame({"timestamp": []}), "2020-01-01", "2020-02-01")
    with pytest.raises(ValueError, match="earlier than"):
        candidate_arl_rows(_candidate_frame(), "2020-02-01", "2020-01-01")


def test_resolve_archive_days_supports_explicit_days_and_bounded_slices():
    days = [date(2020, 1, 5), date(2020, 1, 4), date(2020, 1, 6)]

    assert resolve_archive_days(days, days=["2020-01-05", date(2020, 1, 4)]) == [
        date(2020, 1, 4),
        date(2020, 1, 5),
    ]
    assert resolve_archive_days(days, day_offset=1, max_days=2) == [
        date(2020, 1, 5),
        date(2020, 1, 6),
    ]


def test_resolve_archive_days_reject_unknown_days_and_empty_slices():
    with pytest.raises(ValueError, match="without candidate rows"):
        resolve_archive_days([date(2020, 1, 4)], days=["2020-02-02"])
    with pytest.raises(ValueError, match="no candidate day"):
        resolve_archive_days([date(2020, 1, 4)], day_offset=5, max_days=1)
    with pytest.raises(ValueError, match="must be positive"):
        resolve_archive_days([date(2020, 1, 4)], max_days=0)


def test_merge_drops_repeated_rows_and_keeps_the_first_provenance():
    first = _batch_frame(["2020-01-04T00:00:00Z", "2020-01-04T01:00:00Z"])
    second = _batch_frame(["2020-01-04T01:00:00Z", "2020-01-04T07:00:00Z"])

    merged = merge_arl_feature_batches({"batch_01": first, "batch_02": second})

    assert len(merged) == 3
    assert merged.loc[merged["timestamp"].eq(pd.Timestamp("2020-01-04T01:00:00Z")), ARL_SOURCE_COLUMN].item() == "batch_01"
    assert merged[ARL_SOURCE_COLUMN].tolist() == ["batch_01", "batch_01", "batch_02"]


def test_merge_rejects_schema_changes_and_conflicting_duplicates():
    base = _batch_frame(["2020-01-04T00:00:00Z"])
    renamed = base.rename(columns={"arl_grid_latitude": "arl_grid_lat"})
    with pytest.raises(ValueError, match="missing columns"):
        merge_arl_feature_batches({"batch_01": base, "batch_02": renamed})

    conflicting = _batch_frame(["2020-01-04T00:00:00Z"], temperature=99.0)
    with pytest.raises(ValueError, match="disagree on feature values"):
        merge_arl_feature_batches({"batch_01": base, "batch_02": conflicting})


def test_merge_requires_at_least_one_batch():
    with pytest.raises(ValueError, match="at least one"):
        merge_arl_feature_batches({})


def test_validation_summarises_a_clean_no_leak_frame():
    merged = merge_arl_feature_batches({"batch_01": _batch_frame(["2020-01-04T00:00:00Z"])})

    summary = validate_arl_feature_frame(merged)

    assert summary["rows"] == 1
    assert summary["event_ids"] == [7]
    assert summary["no_future_run_check"] is True
    assert summary["missing_feature_values"] == 0


def test_validation_rejects_a_cycle_issued_after_the_event_timestamp():
    frame = _batch_frame(["2020-01-04T00:00:00Z"])
    frame["arl_forecast_run_utc"] = pd.Timestamp("2020-01-04T06:00:00Z")

    with pytest.raises(ValueError, match="issued after the event timestamp"):
        validate_arl_feature_frame(frame)


def test_validation_rejects_a_valid_time_that_is_not_in_the_future():
    frame = _batch_frame(["2020-01-04T00:00:00Z"])
    frame[ARL_VALID_COLUMN] = pd.Timestamp("2020-01-04T00:00:00Z")

    with pytest.raises(ValueError, match="not in the future"):
        validate_arl_feature_frame(frame)


def test_validation_rejects_a_forecast_horizon_other_than_three_hours():
    frame = _batch_frame(["2020-01-04T00:00:00Z"])
    frame[ARL_VALID_COLUMN] = pd.Timestamp("2020-01-04T06:00:00Z")

    with pytest.raises(ValueError, match=r"\+3 hour"):
        validate_arl_feature_frame(frame)


def test_validation_rejects_missing_feature_values_and_empty_frames():
    frame = _batch_frame(["2020-01-04T00:00:00Z"])
    frame.loc[0, "arl_temperature_c_f003"] = None
    with pytest.raises(ValueError, match="missing feature values"):
        validate_arl_feature_frame(frame)
    with pytest.raises(ValueError, match="empty"):
        validate_arl_feature_frame(frame.iloc[0:0])


def test_recommendation_prefers_days_that_unlock_new_events():
    candidates = candidate_arl_rows(
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2020-01-04T00:00:00Z",
                        "2020-01-05T00:00:00Z",
                        "2020-01-05T01:00:00Z",
                        "2020-01-06T00:00:00Z",
                    ]
                ),
                "aqi": [150.0, 160.0, 170.0, 180.0],
                "event_id": [7, 8, 8, 9],
                "remaining_hours": [3.0, 4.0, 3.0, 2.0],
            }
        ),
        "2020-01-01",
        "2020-02-01",
    )

    plan = recommend_archive_days(candidates, covered_days=[date(2020, 1, 5)], limit=5)

    assert [item["archive_day"] for item in plan] == ["2020-01-04", "2020-01-06"]
    assert plan[0]["new_event_ids"] == [7]
    assert plan[0]["rows"] == 1


def test_recommendation_accounts_for_events_already_covered():
    candidates = candidate_arl_rows(
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2020-01-04T00:00:00Z", "2020-01-05T00:00:00Z"]),
                "aqi": [150.0, 160.0],
                "event_id": [7, 8],
                "remaining_hours": [3.0, 4.0],
            }
        ),
        "2020-01-01",
        "2020-02-01",
    )

    plan = recommend_archive_days(candidates, covered_events=[7, 8])

    assert plan == []


def test_arl_feature_names_follow_the_canonical_order():
    frame = _batch_frame(["2020-01-04T00:00:00Z"]).drop(columns=["arl_wind_v_ms_f003"])

    assert arl_feature_names(frame) == [
        "arl_temperature_c_f003",
        "arl_relative_humidity_pct_f003",
        "arl_wind_u_ms_f003",
        "arl_boundary_layer_height_m_f003",
    ]
