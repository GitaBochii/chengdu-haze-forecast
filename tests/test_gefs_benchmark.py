from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from haze_forecast.features import MEASUREMENT_COLUMNS
from haze_forecast.gefs_benchmark import (
    benchmark_gefs_multivariable_features,
    decide_gefs_multivariable_candidate,
    prepare_gefs_benchmark_frame,
)
from haze_forecast.gefs_dataset import GEFS_SURFACE_FEATURE_COLUMNS


def _city_rows(events: int = 5) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start = pd.Timestamp("2015-01-01T00:00:00Z")
    for event_id in range(1, events + 1):
        event_start = start + pd.Timedelta(event_id * 2, unit="D")
        for hour in range(3):
            rows.append(
                {
                    "timestamp": event_start + pd.Timedelta(hour, unit="h"),
                    "event_id": event_id,
                    "remaining_hours": float(3 - hour),
                    "aqi": 150.0 - hour,
                    "pm2_5": 90.0 - hour,
                    "pm10": 170.0 - hour,
                    "temperature_c": float(3 - hour),
                    "particle_primary": True,
                }
            )
    return pd.DataFrame(rows)


def _pilot_for(city: pd.DataFrame) -> pd.DataFrame:
    pilot = city.loc[:, ["timestamp", "event_id"]].copy()
    pilot["gefs_run_utc"] = pilot["timestamp"].dt.normalize()
    pilot["gefs_forecast_valid_utc"] = pilot["timestamp"] + pd.Timedelta(6, unit="h")
    for offset, name in enumerate(GEFS_SURFACE_FEATURE_COLUMNS):
        pilot[name] = float(offset) + np.arange(len(pilot), dtype=float)
    return pilot


def test_prepare_gefs_benchmark_frame_keeps_only_time_safe_joined_rows():
    city = _city_rows()
    pilot = _pilot_for(city)

    prepared, gefs_features = prepare_gefs_benchmark_frame(city, pilot)

    assert gefs_features == list(GEFS_SURFACE_FEATURE_COLUMNS)
    assert len(prepared) == len(city)
    assert (prepared["gefs_run_utc"] <= prepared["timestamp"]).all()
    assert (prepared["timestamp"] < prepared["gefs_forecast_valid_utc"]).all()


def test_prepare_gefs_benchmark_frame_selects_only_the_pilot_subset_from_training_rows():
    city = _city_rows(events=40)
    pilot_source = city.loc[city["event_id"].isin([4, 17])].reset_index(drop=True)
    pilot = _pilot_for(pilot_source)

    prepared, _ = prepare_gefs_benchmark_frame(city, pilot)

    assert len(prepared) == len(pilot)
    assert set(prepared["event_id"]) == {4, 17}
    assert set(map(tuple, prepared[["timestamp", "event_id"]].to_numpy())) == set(
        map(tuple, pilot[["timestamp", "event_id"]].to_numpy())
    )


def test_prepare_gefs_benchmark_frame_rejects_a_forecast_valid_at_or_before_issue_time():
    city = _city_rows()
    pilot = _pilot_for(city)
    pilot.loc[pilot.index[0], "gefs_forecast_valid_utc"] = pilot.loc[pilot.index[0], "timestamp"]

    with pytest.raises(ValueError, match="run <= timestamp < valid"):
        prepare_gefs_benchmark_frame(city, pilot)


def test_prepare_gefs_benchmark_frame_rejects_forecast_valid_less_than_six_hours_ahead():
    city = _city_rows()
    pilot = _pilot_for(city)
    pilot.loc[pilot.index[0], "gefs_forecast_valid_utc"] = (
        pilot.loc[pilot.index[0], "timestamp"] + pd.Timedelta(3, unit="h")
    )

    with pytest.raises(ValueError, match="at least 6 hours"):
        prepare_gefs_benchmark_frame(city, pilot)


def test_prepare_gefs_benchmark_frame_requires_the_complete_surface_feature_set():
    city = _city_rows()
    pilot = _pilot_for(city).drop(columns=[GEFS_SURFACE_FEATURE_COLUMNS[-1]])

    with pytest.raises(ValueError, match="complete GEFS surface feature set"):
        prepare_gefs_benchmark_frame(city, pilot)


def test_prepare_gefs_benchmark_frame_rejects_pilot_rows_without_a_training_row_match():
    city = _city_rows()
    city.loc[city.index[0], "aqi"] = 95.0
    city.loc[city.index[0], "particle_primary"] = False
    pilot = _pilot_for(city)

    with pytest.raises(ValueError, match="one-to-one.*training rows"):
        prepare_gefs_benchmark_frame(city, pilot)


def test_benchmark_gefs_multivariable_uses_disjoint_complete_event_rolling_folds(monkeypatch):
    city = _city_rows(events=15)
    frame, gefs_features = prepare_gefs_benchmark_frame(city, _pilot_for(city))
    calls: list[list[str]] = []

    def fake_fit_predict(train_x, train_y, valid_x, seed):  # type: ignore[no-untyped-def]
        calls.append(train_x.columns.tolist())
        return valid_x["temperature_c"].to_numpy(dtype=float)

    monkeypatch.setattr("haze_forecast.gefs_benchmark._fit_predict_aft", fake_fit_predict)

    report, predictions = benchmark_gefs_multivariable_features(
        frame,
        baseline_features=[name for name in MEASUREMENT_COLUMNS if name in frame],
        gefs_features=gefs_features,
        n_splits=3,
        seed=7,
    )

    assert {record["variant"] for record in report["models"]} == {
        "baseline",
        "gefs_multivariable",
    }
    assert len(calls) == 6
    assert all(
        set(fold["train_event_ids"]).isdisjoint(fold["validation_event_ids"])
        for fold in report["folds"]
    )
    assert report["decision"]["decision"] == "reject_insufficient_mae_improvement"
    assert set(predictions["variant"]) == {"baseline", "gefs_multivariable"}


def test_benchmark_gefs_multivariable_rejects_missing_gefs_values():
    city = _city_rows(events=15)
    frame, gefs_features = prepare_gefs_benchmark_frame(city, _pilot_for(city))
    frame.loc[frame.index[0], gefs_features[0]] = np.nan

    with pytest.raises(ValueError, match="missing GEFS feature values"):
        benchmark_gefs_multivariable_features(
            frame,
            baseline_features=[name for name in MEASUREMENT_COLUMNS if name in frame],
            gefs_features=gefs_features,
            n_splits=3,
        )


def test_benchmark_gefs_multivariable_requires_exactly_fifteen_independent_events():
    city = _city_rows(events=14)
    frame, gefs_features = prepare_gefs_benchmark_frame(city, _pilot_for(city))

    with pytest.raises(ValueError, match="exactly 15 independent events"):
        benchmark_gefs_multivariable_features(
            frame,
            baseline_features=[name for name in MEASUREMENT_COLUMNS if name in frame],
            gefs_features=gefs_features,
            n_splits=3,
        )


def test_benchmark_gefs_multivariable_rejects_gefs_metadata_as_a_model_feature(monkeypatch):
    city = _city_rows(events=15)
    frame, gefs_features = prepare_gefs_benchmark_frame(city, _pilot_for(city))
    frame["gefs_lead_hours"] = 6.0

    with pytest.raises(ValueError, match="must be the complete GEFS surface feature set"):
        benchmark_gefs_multivariable_features(
            frame,
            baseline_features=[name for name in MEASUREMENT_COLUMNS if name in frame],
            gefs_features=[*gefs_features, "gefs_lead_hours"],
            n_splits=3,
        )


def test_multivariable_extractor_outputs_time_safe_event_samples(monkeypatch, tmp_path):
    script_path = Path("scripts/extract_gefs_multivariable_event_features.py")
    spec = importlib.util.spec_from_file_location("extract_gefs_multivariable", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    city = _city_rows(events=3)
    output = tmp_path / "pilot.parquet"
    manifest = tmp_path / "report.json"

    def fake_enrich(event_rows, **kwargs):  # type: ignore[no-untyped-def]
        enriched = event_rows.copy()
        enriched["gefs_run_utc"] = enriched["timestamp"].dt.normalize()
        enriched["gefs_forecast_valid_utc"] = enriched["timestamp"] + pd.Timedelta(6, unit="h")
        for offset, name in enumerate(GEFS_SURFACE_FEATURE_COLUMNS):
            enriched[name] = float(offset)
        return enriched

    monkeypatch.setattr(module, "enrich_event_rows_with_gefs_surface", fake_enrich)
    result = module.extract_multivariable_event_features(
        city,
        year=2015,
        max_events=2,
        samples_per_event=2,
        horizon_hours=6,
        longitude=104.0665,
        latitude=30.5723,
        wgrib2=Path("unused.exe"),
        timeout_s=1,
        output=output,
        manifest=manifest,
    )

    assert result["event_ids"] == [1, 2]
    assert (result["frame"]["gefs_run_utc"] <= result["frame"]["timestamp"]).all()
    assert (result["frame"]["timestamp"] < result["frame"]["gefs_forecast_valid_utc"]).all()
    assert result["manifest"]["time_safety"]["run_lte_timestamp_lt_valid"] is True


def test_multivariable_extractor_excludes_labeled_rows_after_particle_pollution_ends(monkeypatch, tmp_path):
    script_path = Path("scripts/extract_gefs_multivariable_event_features.py")
    spec = importlib.util.spec_from_file_location("extract_gefs_multivariable_predictable_rows", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    city = _city_rows(events=2)
    ended = city.index[city["event_id"].eq(1) & city["timestamp"].eq(city.loc[2, "timestamp"])]
    city.loc[ended, "aqi"] = 90.0
    city.loc[ended, "particle_primary"] = False

    def fake_enrich(event_rows, **_kwargs):  # type: ignore[no-untyped-def]
        assert not ((event_rows["aqi"] <= 100) | ~event_rows["particle_primary"]).any()
        enriched = event_rows.copy()
        enriched["gefs_run_utc"] = enriched["timestamp"].dt.normalize()
        enriched["gefs_forecast_valid_utc"] = enriched["timestamp"] + pd.Timedelta(6, unit="h")
        for offset, name in enumerate(GEFS_SURFACE_FEATURE_COLUMNS):
            enriched[name] = float(offset)
        return enriched

    monkeypatch.setattr(module, "enrich_event_rows_with_gefs_surface", fake_enrich)
    result = module.extract_multivariable_event_features(
        city,
        year=2015,
        max_events=2,
        samples_per_event=2,
        horizon_hours=6,
        longitude=104.0665,
        latitude=30.5723,
        wgrib2=Path("unused.exe"),
        timeout_s=1,
        output=tmp_path / "pilot.parquet",
        manifest=tmp_path / "report.json",
    )

    assert result["event_ids"] == [1, 2]
    assert not ((result["frame"]["aqi"] <= 100) | ~result["frame"]["particle_primary"]).any()


def test_multivariable_extractor_rejects_insufficient_events_before_any_download(monkeypatch, tmp_path):
    script_path = Path("scripts/extract_gefs_multivariable_event_features.py")
    spec = importlib.util.spec_from_file_location("extract_gefs_multivariable_event_count", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    city = _city_rows(events=2)
    city.loc[city["event_id"].eq(2), "aqi"] = 90.0
    output = tmp_path / "pilot.parquet"
    manifest = tmp_path / "report.json"

    def should_not_download(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("extractor must validate event count before download")

    monkeypatch.setattr(module, "enrich_event_rows_with_gefs_surface", should_not_download)

    with pytest.raises(ValueError, match="exactly 2"):
        module.extract_multivariable_event_features(
            city,
            year=2015,
            max_events=2,
            samples_per_event=2,
            horizon_hours=6,
            longitude=104.0665,
            latitude=30.5723,
            wgrib2=Path("unused.exe"),
            timeout_s=1,
            output=output,
            manifest=manifest,
        )

    assert not output.exists()
    assert not manifest.exists()


@pytest.mark.parametrize("candidate_event_ids", [[1, 3], [1, 1]])
def test_candidate_decision_rejects_nonidentical_or_nonunique_event_sets(candidate_event_ids):
    baseline = pd.DataFrame(
        {
            "event_id": [1, 2],
            "mae": [2.0, 4.0],
            "absolute_error": [4.0, 8.0],
        }
    )
    candidate = pd.DataFrame(
        {
            "event_id": candidate_event_ids,
            "mae": [1.0, 3.0],
            "absolute_error": [2.0, 6.0],
        }
    )

    with pytest.raises(ValueError, match="unique.*same event_id set"):
        decide_gefs_multivariable_candidate(3.0, 2.0, baseline, candidate)


@pytest.mark.parametrize(
    ("year", "max_events", "samples_per_event"),
    [(2014, 1, 1), (2015, 16, 1), (2015, 1, 5)],
)
def test_multivariable_extractor_rejects_out_of_scope_research_limits(
    monkeypatch, tmp_path, year, max_events, samples_per_event
):
    script_path = Path("scripts/extract_gefs_multivariable_event_features.py")
    spec = importlib.util.spec_from_file_location("extract_gefs_multivariable_limits", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "enrich_event_rows_with_gefs_surface", lambda rows, **_kwargs: rows)

    with pytest.raises(ValueError, match="research scope"):
        module.extract_multivariable_event_features(
            _city_rows(events=3),
            year=year,
            max_events=max_events,
            samples_per_event=samples_per_event,
            horizon_hours=6,
            longitude=104.0665,
            latitude=30.5723,
            wgrib2=Path("unused.exe"),
            timeout_s=1,
            output=tmp_path / "pilot.parquet",
            manifest=tmp_path / "report.json",
        )


def test_multivariable_extractor_cli_rejects_a_year_outside_the_fixed_research_scope(monkeypatch):
    script_path = Path("scripts/extract_gefs_multivariable_event_features.py")
    spec = importlib.util.spec_from_file_location("extract_gefs_multivariable_cli", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", ["extract", "--year", "2016"])

    with pytest.raises(SystemExit):
        module.parse_args()


def test_multivariable_extractor_rejects_forecast_valid_less_than_six_hours_ahead(monkeypatch, tmp_path):
    script_path = Path("scripts/extract_gefs_multivariable_event_features.py")
    spec = importlib.util.spec_from_file_location("extract_gefs_multivariable_horizon", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def fake_enrich(event_rows, **_kwargs):  # type: ignore[no-untyped-def]
        enriched = event_rows.copy()
        enriched["gefs_run_utc"] = enriched["timestamp"].dt.normalize()
        enriched["gefs_forecast_valid_utc"] = enriched["timestamp"] + pd.Timedelta(3, unit="h")
        return enriched

    monkeypatch.setattr(module, "enrich_event_rows_with_gefs_surface", fake_enrich)
    with pytest.raises(RuntimeError, match="at least six hours"):
        module.extract_multivariable_event_features(
            _city_rows(events=3),
            year=2015,
            max_events=2,
            samples_per_event=2,
            horizon_hours=6,
            longitude=104.0665,
            latitude=30.5723,
            wgrib2=Path("unused.exe"),
            timeout_s=1,
            output=tmp_path / "pilot.parquet",
            manifest=tmp_path / "report.json",
        )


def test_benchmark_cli_uses_only_existing_measurement_columns_for_its_baseline(monkeypatch, tmp_path):
    script_path = Path("scripts/benchmark_gefs_multivariable_features.py")
    spec = importlib.util.spec_from_file_location("benchmark_gefs_multivariable_cli", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prepared = _city_rows(events=15)
    prepared["aqi_lag_1"] = 149.0
    for offset, name in enumerate(GEFS_SURFACE_FEATURE_COLUMNS):
        prepared[name] = float(offset)
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(module.pd, "read_parquet", lambda _path: pd.DataFrame())
    monkeypatch.setattr(
        module,
        "prepare_gefs_benchmark_frame",
        lambda _city, _pilot: (prepared, list(GEFS_SURFACE_FEATURE_COLUMNS)),
    )

    def fake_benchmark(_frame, *, baseline_features, gefs_features, **_kwargs):  # type: ignore[no-untyped-def]
        captured["baseline"] = list(baseline_features)
        captured["gefs"] = list(gefs_features)
        return {"decision": {"decision": "reject"}}, pd.DataFrame()

    monkeypatch.setattr(module, "benchmark_gefs_multivariable_features", fake_benchmark)
    monkeypatch.setattr(module, "write_gefs_benchmark_artifacts", lambda *_args: None)
    monkeypatch.setattr(sys, "argv", ["benchmark", "--output", str(tmp_path)])

    module.main()

    assert captured["baseline"] == [name for name in MEASUREMENT_COLUMNS if name in prepared]
    assert captured["gefs"] == list(GEFS_SURFACE_FEATURE_COLUMNS)
