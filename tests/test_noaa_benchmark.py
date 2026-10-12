import hashlib
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from haze_forecast import noaa_benchmark


def _city_labeled() -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=12, freq="h", tz="Asia/Shanghai")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "event_id": np.repeat(np.arange(1, 7), 2),
            "remaining_hours": np.tile([2.0, 1.0], 6),
            "aqi": 130.0,
            "particle_primary": True,
            "history_feature": np.arange(12, dtype=float),
        }
    )


def _hourly() -> pd.DataFrame:
    timestamps = pd.date_range("2024-12-31T16:00:00Z", periods=12, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "noaa_temperature_c": np.arange(12, dtype=float),
            "noaa_wind_speed_ms": 2.0,
        }
    )


def _eligible_audit() -> dict[str, object]:
    years = list(range(2015, 2022))
    return {
        "eligible": True,
        "minimum_coverage": 0.70,
        "years": years,
        "eligible_station_count": 1,
        "city_hourly_rows": 12,
        "stations": [
            {
                "eligible": True,
                "years": {str(year): {"sha256": "a" * 64, "coverage": 1.0} for year in years},
                "boundary_sources": {str(year): {"sha256": "b" * 64} for year in years},
            }
        ],
        "city_hourly_sha256": "c" * 64,
    }


def _write_audit(tmp_path, report=None):  # type: ignore[no-untyped-def]
    path = tmp_path / "audit.json"
    audit = _eligible_audit() if report is None else report
    path.write_text(json.dumps(audit), encoding="utf-8")
    return audit, path, noaa_benchmark.audit_report_sha256(path)


def _write_hourly(tmp_path, hourly=None):  # type: ignore[no-untyped-def]
    path = tmp_path / "hourly.parquet"
    (_hourly() if hourly is None else hourly).to_parquet(path, index=False)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _audited_inputs(tmp_path, hourly=None, audit=None):  # type: ignore[no-untyped-def]
    hourly_path, hourly_digest = _write_hourly(tmp_path, hourly)
    report = _eligible_audit() if audit is None else audit
    report["city_hourly_sha256"] = hourly_digest
    report, audit_path, audit_digest = _write_audit(tmp_path, report)
    return report, audit_path, audit_digest, hourly_path


def test_ineligible_audit_rejects_before_training(monkeypatch):
    def should_not_train(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("benchmark must not run for an ineligible audit")

    monkeypatch.setattr(noaa_benchmark, "benchmark_spatial_features", should_not_train)

    with pytest.raises(ValueError, match="eligible"):
        noaa_benchmark.benchmark_noaa_isd_features(
            _city_labeled(), _hourly(), {"eligible": False}, audit_report_sha256="a" * 64
        )


@pytest.mark.parametrize("eligible", [False, None])
def test_audit_is_eligible_requires_an_explicit_true_top_level_status(eligible):
    audit = _eligible_audit()
    if eligible is None:
        del audit["eligible"]
    else:
        audit["eligible"] = eligible

    assert noaa_benchmark.audit_is_eligible(audit) is False


def test_eligible_audit_requires_a_verified_hourly_path(tmp_path):
    audit, path, digest, _ = _audited_inputs(tmp_path)

    with pytest.raises(ValueError, match="hourly parquet path"):
        noaa_benchmark.benchmark_noaa_isd_features(
            _city_labeled(),
            None,
            audit,
            audit_report_sha256=digest,
            audit_report_path=path,
        )


@pytest.mark.parametrize(
    ("hourly", "reason"),
    [
        (_hourly().assign(timestamp=lambda frame: frame["timestamp"] + pd.Timedelta(days=30)), "non-overlap"),
        (
            _hourly().assign(noaa_temperature_c=np.nan, noaa_wind_speed_ms=np.nan),
            "all-empty",
        ),
        (
            _hourly().assign(
                noaa_temperature_c=[1.0] * 8 + [np.nan] * 4,
                noaa_wind_speed_ms=[2.0] * 8 + [np.nan] * 4,
            ),
            "low-coverage",
        ),
    ],
)
def test_noaa_coverage_gate_blocks_training_before_benchmark(tmp_path, monkeypatch, hourly, reason):
    monkeypatch.setattr(
        noaa_benchmark,
        "benchmark_spatial_features",
        lambda *args, **kwargs: pytest.fail("benchmark must not run below NOAA coverage gates"),
    )

    audit, path, digest, hourly_path = _audited_inputs(tmp_path, hourly)
    with pytest.raises(ValueError, match="coverage"):
        noaa_benchmark.benchmark_noaa_isd_features(
            _city_labeled(),
            hourly,
            audit,
            audit_report_sha256=digest,
            audit_report_path=path,
            hourly_path=hourly_path,
        )


def test_noaa_benchmark_excludes_long_term_empty_columns_before_training(tmp_path, monkeypatch):
    calls = []

    def fake_benchmark(frame, baseline_features, enhanced_features, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append((frame.copy(), list(baseline_features), list(enhanced_features)))
        return (
            {
                "protocol": "test",
                "folds": [],
                "models": {"baseline": {}, "directional": {}},
                "decision": {},
            },
            pd.DataFrame({"variant": ["baseline", "directional"]}),
        )

    monkeypatch.setattr(noaa_benchmark, "benchmark_spatial_features", fake_benchmark)
    hourly = _hourly().assign(noaa_dewpoint_c=np.nan)
    audit, path, digest, hourly_path = _audited_inputs(tmp_path, hourly)

    report, _ = noaa_benchmark.benchmark_noaa_isd_features(
        _city_labeled(),
        hourly,
        audit,
        audit_report_sha256=digest,
        audit_report_path=path,
        hourly_path=hourly_path,
    )

    frame, baseline, enhanced = calls[0]
    assert "noaa_dewpoint_c" in frame.columns
    assert "noaa_dewpoint_c" not in enhanced
    assert baseline == [name for name in enhanced if not name.startswith("noaa_")]
    assert report["noaa_features"] == ["noaa_temperature_c", "noaa_wind_speed_ms"]
    assert report["noaa_feature_complete_rate"] == 1.0
    assert report["excluded_noaa_features"] == {
        "noaa_dewpoint_c": {
            "reason": "non_null_rate_below_0.70",
            "non_null_rate": 0.0,
        }
    }


def test_noaa_benchmark_requires_selected_columns_to_be_complete_often_enough(tmp_path, monkeypatch):
    monkeypatch.setattr(
        noaa_benchmark,
        "benchmark_spatial_features",
        lambda *args, **kwargs: pytest.fail("benchmark must not run below selected NOAA coverage"),
    )
    hourly = _hourly().assign(
        noaa_temperature_c=[1.0] * 9 + [np.nan] * 3,
        noaa_wind_speed_ms=[np.nan] * 3 + [2.0] * 9,
    )
    audit, path, digest, hourly_path = _audited_inputs(tmp_path, hourly)

    with pytest.raises(ValueError, match="complete_rate"):
        noaa_benchmark.benchmark_noaa_isd_features(
            _city_labeled(),
            hourly,
            audit,
            audit_report_sha256=digest,
            audit_report_path=path,
            hourly_path=hourly_path,
        )


def test_noaa_benchmark_uses_same_complete_event_folds_without_leakage(tmp_path, monkeypatch):
    calls = []

    def fake_benchmark(frame, baseline_features, enhanced_features, n_splits, seed):  # type: ignore[no-untyped-def]
        calls.append((frame.copy(), list(baseline_features), list(enhanced_features), n_splits, seed))
        return (
            {
                "protocol": "expanding chronological complete-event rolling validation",
                "directional_features": ["noaa_temperature_c", "noaa_wind_speed_ms"],
                "directional_feature_coverage": 1.0,
                "folds": [
                    {"train_event_ids": [1, 2], "validation_event_ids": [3]},
                    {"train_event_ids": [1, 2, 3], "validation_event_ids": [4]},
                    {"train_event_ids": [1, 2, 3, 4], "validation_event_ids": [5]},
                ],
                "models": {
                    "baseline": {"metrics": {"mae": 10.0}, "fold_mae": [10.0] * 3},
                    "directional": {"metrics": {"mae": 9.0}, "fold_mae": [9.0] * 3},
                },
                "decision": {"decision": "candidate_for_replacement_review"},
            },
            pd.DataFrame({"variant": ["baseline", "directional"]}),
        )

    monkeypatch.setattr(noaa_benchmark, "benchmark_spatial_features", fake_benchmark)

    audit, path, digest, hourly_path = _audited_inputs(tmp_path)
    report, predictions = noaa_benchmark.benchmark_noaa_isd_features(
        _city_labeled(),
        _hourly(),
        audit,
        audit_report_sha256=digest,
        audit_report_path=path,
        hourly_path=hourly_path,
        n_splits=3,
        seed=7,
    )

    frame, baseline, enhanced, splits, seed = calls[0]
    assert frame["timestamp"].dt.tz.zone == "Asia/Shanghai"
    assert frame["timestamp"].is_unique
    assert all(set(fold["train_event_ids"]).isdisjoint(fold["validation_event_ids"]) for fold in report["folds"])
    assert baseline == [name for name in enhanced if not name.startswith("noaa_")]
    assert [name for name in enhanced if name.startswith("noaa_")] == [
        "noaa_temperature_c",
        "noaa_wind_speed_ms",
    ]
    assert (splits, seed) == (3, 7)
    assert set(report["models"]) == {"baseline", "noaa_isd"}
    assert report["variants"] == ["baseline", "noaa_isd"]
    assert set(predictions["variant"]) == {"baseline", "noaa_isd"}
    assert report["input_audit_report_sha256"] == digest
    assert report["noaa_feature_non_null_rate"] == {
        "noaa_temperature_c": 1.0,
        "noaa_wind_speed_ms": 1.0,
    }


def test_noaa_benchmark_uses_the_verified_hourly_path_not_a_supplied_dataframe(tmp_path, monkeypatch):
    calls = []

    def fake_benchmark(frame, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(frame.copy())
        return (
            {
                "protocol": "test",
                "folds": [],
                "models": {"baseline": {}, "directional": {}},
                "decision": {},
            },
            pd.DataFrame({"variant": ["baseline", "directional"]}),
        )

    monkeypatch.setattr(noaa_benchmark, "benchmark_spatial_features", fake_benchmark)
    audit, path, digest, hourly_path = _audited_inputs(tmp_path)
    supplied = _hourly().assign(noaa_temperature_c=999.0)

    noaa_benchmark.benchmark_noaa_isd_features(
        _city_labeled(),
        supplied,
        audit,
        audit_report_sha256=digest,
        audit_report_path=path,
        hourly_path=hourly_path,
    )

    assert calls[0]["noaa_temperature_c"].tolist() == list(range(12))


def test_noaa_benchmark_rejects_audit_that_does_not_cover_exactly_2015_to_2021(tmp_path, monkeypatch):
    monkeypatch.setattr(
        noaa_benchmark,
        "benchmark_spatial_features",
        lambda *args, **kwargs: pytest.fail("benchmark must not run for invalid audit years"),
    )
    audit = _eligible_audit()
    audit["years"] = [2025]
    audit["stations"][0]["years"] = {"2025": {"sha256": "a" * 64, "coverage": 1.0}}
    audit["stations"][0]["boundary_sources"] = {"2025": {"sha256": "b" * 64}}
    audit, path, digest, hourly_path = _audited_inputs(tmp_path, audit=audit)

    with pytest.raises(ValueError, match="eligible"):
        noaa_benchmark.benchmark_noaa_isd_features(
            _city_labeled(),
            _hourly(),
            audit,
            audit_report_sha256=digest,
            audit_report_path=path,
            hourly_path=hourly_path,
        )


@pytest.mark.parametrize("audit_sha256", [None, "abc", "g" * 64])
def test_noaa_benchmark_requires_a_valid_audit_report_sha256(tmp_path, monkeypatch, audit_sha256):
    monkeypatch.setattr(
        noaa_benchmark,
        "benchmark_spatial_features",
        lambda *args, **kwargs: pytest.fail("benchmark must not run without a valid audit hash"),
    )
    audit, path, _, hourly_path = _audited_inputs(tmp_path)

    with pytest.raises(ValueError, match="sha256"):
        noaa_benchmark.benchmark_noaa_isd_features(
            _city_labeled(),
            _hourly(),
            audit,
            audit_report_sha256=audit_sha256,
            audit_report_path=path,
            hourly_path=hourly_path,
        )


@pytest.mark.parametrize("splits", [2, 4])
def test_noaa_benchmark_only_allows_three_folds(tmp_path, splits):
    audit, path, digest, hourly_path = _audited_inputs(tmp_path)
    with pytest.raises(ValueError, match="exactly 3"):
        noaa_benchmark.benchmark_noaa_isd_features(
            _city_labeled(),
            _hourly(),
            audit,
            audit_report_sha256=digest,
            audit_report_path=path,
            hourly_path=hourly_path,
            n_splits=splits,
        )


def test_low_minimum_coverage_audit_rejects_before_training(tmp_path, monkeypatch):
    monkeypatch.setattr(
        noaa_benchmark,
        "benchmark_spatial_features",
        lambda *args, **kwargs: pytest.fail("benchmark must not run for a weak audit"),
    )
    audit = _eligible_audit() | {"minimum_coverage": 0.1}
    audit, path, digest, hourly_path = _audited_inputs(tmp_path, audit=audit)

    with pytest.raises(ValueError, match="eligible"):
        noaa_benchmark.benchmark_noaa_isd_features(
            _city_labeled(),
            _hourly(),
            audit,
            audit_report_sha256=digest,
            audit_report_path=path,
            hourly_path=hourly_path,
        )


def test_low_eligible_station_annual_coverage_rejects_before_training(tmp_path, monkeypatch):
    monkeypatch.setattr(
        noaa_benchmark,
        "benchmark_spatial_features",
        lambda *args, **kwargs: pytest.fail("benchmark must not run for a low annual coverage audit"),
    )
    audit = _eligible_audit()
    audit["stations"][0]["years"]["2015"]["coverage"] = 0.1
    audit, path, digest, hourly_path = _audited_inputs(tmp_path, audit=audit)

    with pytest.raises(ValueError, match="eligible"):
        noaa_benchmark.benchmark_noaa_isd_features(
            _city_labeled(),
            _hourly(),
            audit,
            audit_report_sha256=digest,
            audit_report_path=path,
            hourly_path=hourly_path,
        )


def test_hourly_sha256_mismatch_rejects_before_training(tmp_path, monkeypatch):
    monkeypatch.setattr(
        noaa_benchmark,
        "benchmark_spatial_features",
        lambda *args, **kwargs: pytest.fail("benchmark must not run for a mismatched hourly file"),
    )
    hourly_path, hourly_digest = _write_hourly(tmp_path)
    audit = _eligible_audit() | {"city_hourly_sha256": hourly_digest}
    audit, audit_path, audit_digest = _write_audit(tmp_path, audit)
    _hourly().assign(noaa_temperature_c=99.0).to_parquet(hourly_path, index=False)

    with pytest.raises(ValueError, match="hourly sha256"):
        noaa_benchmark.benchmark_noaa_isd_features(
            _city_labeled(),
            _hourly(),
            audit,
            audit_report_sha256=audit_digest,
            audit_report_path=audit_path,
            hourly_path=hourly_path,
        )


def test_noaa_benchmark_requires_the_exact_audit_file_hash_and_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(
        noaa_benchmark,
        "benchmark_spatial_features",
        lambda *args, **kwargs: pytest.fail("benchmark must not run without audit provenance"),
    )
    audit, path, digest, hourly_path = _audited_inputs(tmp_path)

    with pytest.raises(ValueError, match="audit report sha256"):
        noaa_benchmark.benchmark_noaa_isd_features(
            _city_labeled(),
            _hourly(),
            audit,
            audit_report_sha256="c" * 64,
            audit_report_path=path,
            hourly_path=hourly_path,
        )
    with pytest.raises(ValueError, match="audit report path"):
        noaa_benchmark.benchmark_noaa_isd_features(
            _city_labeled(), _hourly(), audit, audit_report_sha256=digest, hourly_path=hourly_path
        )

    missing_source = _eligible_audit()
    missing_source["stations"][0]["boundary_sources"]["2015"]["sha256"] = None
    missing_source, missing_path, missing_digest, missing_hourly_path = _audited_inputs(
        tmp_path, audit=missing_source
    )
    with pytest.raises(ValueError, match="source sha256"):
        noaa_benchmark.benchmark_noaa_isd_features(
            _city_labeled(),
            _hourly(),
            missing_source,
            audit_report_sha256=missing_digest,
            audit_report_path=missing_path,
            hourly_path=missing_hourly_path,
        )


def test_noaa_benchmark_artifacts_are_auditable(tmp_path):
    output = tmp_path / "noaa"
    predictions = pd.DataFrame(
        {
            "fold": [1, 1],
            "variant": ["baseline", "noaa_isd"],
            "event_id": [1, 1],
            "actual_remaining_hours": [2.0, 2.0],
            "predicted_remaining_hours": [2.1, 1.9],
        }
    )

    noaa_benchmark.write_noaa_benchmark_artifacts({"models": {}}, predictions, output)

    report_path, prediction_path = noaa_benchmark.resolve_noaa_benchmark_artifacts(output)
    assert json.loads(report_path.read_text(encoding="utf-8")) == {"models": {}}
    assert pd.read_csv(prediction_path, encoding="utf-8-sig").to_dict(
        orient="records"
    ) == predictions.to_dict(orient="records")
    assert (output / "current.json").is_file()


def test_noaa_artifact_publish_failure_restores_the_previous_pair(tmp_path, monkeypatch):
    output = tmp_path / "noaa"
    old_report = {"version": "old"}
    old_predictions = pd.DataFrame(
        {
            "fold": [1],
            "variant": ["baseline"],
            "event_id": [1],
            "actual_remaining_hours": [2.0],
            "predicted_remaining_hours": [2.0],
        }
    )
    noaa_benchmark.write_noaa_benchmark_artifacts(old_report, old_predictions, output)
    old_current_bytes = (output / "current.json").read_bytes()
    original_replace = Path.replace

    def fail_pointer_publish(source, target):  # type: ignore[no-untyped-def]
        if source.name == "current.json.part" and target.name == "current.json":
            raise OSError("simulated pointer publish failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_pointer_publish)
    with pytest.raises(OSError, match="simulated"):
        noaa_benchmark.write_noaa_benchmark_artifacts({"version": "new"}, old_predictions, output)

    assert (output / "current.json").read_bytes() == old_current_bytes
    report_path, prediction_path = noaa_benchmark.resolve_noaa_benchmark_artifacts(output)
    assert json.loads(report_path.read_text(encoding="utf-8")) == old_report
    assert pd.read_csv(prediction_path, encoding="utf-8-sig").to_dict(orient="records") == (
        old_predictions.to_dict(orient="records")
    )


def test_unpointed_half_pair_is_not_a_successful_noaa_artifact(tmp_path):
    output = tmp_path / "noaa"
    output.mkdir()
    (output / "report.json").write_text("{}", encoding="utf-8")
    (output / "fold_predictions.csv").write_text("fold\n1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="current"):
        noaa_benchmark.resolve_noaa_benchmark_artifacts(output)


def test_cli_rejects_ineligible_audit_without_calling_training(tmp_path, monkeypatch, capsys):
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"eligible": False}), encoding="utf-8")
    output = tmp_path / "output"

    monkeypatch.setattr(noaa_benchmark, "benchmark_noaa_isd_features", lambda *args, **kwargs: pytest.fail("called"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_noaa_isd_features.py",
            "--audit",
            str(audit),
            "--hourly",
            str(tmp_path / "missing.parquet"),
            "--city-data",
            str(tmp_path / "missing-city.parquet"),
            "--output",
            str(output),
        ],
    )

    with pytest.raises(SystemExit) as error:
        runpy.run_path("scripts/benchmark_noaa_isd_features.py", run_name="__main__")

    assert error.value.code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "rejected"
    assert not output.exists()


def test_cli_only_accepts_exactly_three_splits(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["benchmark_noaa_isd_features.py", "--splits", "4"])
    cli = runpy.run_path("scripts/benchmark_noaa_isd_features.py", run_name="noaa_cli_test")

    with pytest.raises(SystemExit) as error:
        cli["parse_args"]()

    assert error.value.code == 2


@pytest.mark.parametrize(
    "hourly",
    [
        pd.DataFrame(columns=["timestamp", "noaa_temperature_c"]),
        pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=1, freq="h")}),
        _hourly().assign(noaa_temperature_c=np.nan, noaa_wind_speed_ms=np.nan),
    ],
)
def test_cli_rejects_unusable_eligible_noaa_hourly_without_output(
    tmp_path, monkeypatch, capsys, hourly
):
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps(_eligible_audit()), encoding="utf-8")
    hourly_path = tmp_path / "hourly.parquet"
    city_path = tmp_path / "city.parquet"
    hourly.to_parquet(hourly_path, index=False)
    _city_labeled().to_parquet(city_path, index=False)
    output = tmp_path / "output"
    monkeypatch.setattr(
        noaa_benchmark,
        "benchmark_spatial_features",
        lambda *args, **kwargs: pytest.fail("benchmark must not run"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_noaa_isd_features.py",
            "--audit",
            str(audit),
            "--hourly",
            str(hourly_path),
            "--city-data",
            str(city_path),
            "--output",
            str(output),
        ],
    )

    with pytest.raises(SystemExit) as error:
        runpy.run_path("scripts/benchmark_noaa_isd_features.py", run_name="__main__")

    assert error.value.code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "rejected"
    assert not output.exists()
