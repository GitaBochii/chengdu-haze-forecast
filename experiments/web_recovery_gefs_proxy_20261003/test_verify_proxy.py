"""Synthetic verifier regressions; these tests do not establish model accuracy."""

import ast
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent


@pytest.fixture
def v():
    path = HERE / "verify_proxy.py"
    assert path.is_file(), "proxy verifier not implemented"
    spec = importlib.util.spec_from_file_location("independent_proxy_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshots():
    times = pd.date_range("2024-01-01", periods=3, freq="6h", tz="UTC")
    result = pd.DataFrame({"known_at": times + pd.Timedelta(hours=4)})
    for i in range(9):
        result[f"weather_{i}_f024"] = np.arange(3, dtype=float) + i
    result["run_at"] = times
    result["valid_at"] = times + pd.Timedelta(hours=24)
    return result


def evaluation():
    return pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=4, freq="h", tz="Asia/Shanghai"),
        "year": [2024, 2024, 2025, 2025], "continuity_block": [1, 1, 2, 2], "event_key": ["a", "a", "b", "b"],
        "lower_bound_hours": [30., 2., 30., 4.], "upper_bound_hours": [30., 2., 30., 8.],
        "exact_hours": [30., 2., 30., np.nan]})


def test_no_runner_import_and_frozen_bridge(v):
    source = (HERE / "verify_proxy.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any("runner" in item for item in imports)
    assert v.bridge().sha256(HERE.parent / "web_recovery_bridge_20261002/verify_bridge.py") == v.BRIDGE_SHA256


def test_latest_snapshot_numpy_with_timezone_conversion_and_ages(v):
    times = pd.DatetimeIndex(["2024-01-01 03:00Z", "2024-01-01 04:00Z", "2024-01-01 11:00Z", "2024-01-02 12:00Z"])
    values, names = v.match_snapshots(times.tz_convert("Asia/Shanghai"), snapshots())
    assert len(names) == 9
    assert np.isnan(values.loc[0, names].to_numpy(dtype=float)).all()
    assert values.loc[1, names[0]] == 0
    assert values.loc[2, names[0]] == 1
    assert np.isnan(values.loc[3, names].to_numpy(dtype=float)).all()
    np.testing.assert_array_equal(values.loc[2, list(v.AGE_COLUMNS)].to_numpy(dtype=float), [1., 5., 19.])
    assert values.loc[2, "known_at"] == pd.Timestamp("2024-01-01 10:00Z")


@pytest.mark.parametrize("mutation", ["naive_origin", "naive_known", "duplicate", "future_known", "bad_valid", "nan_weather", "inf_weather"])
def test_invalid_snapshot_inputs_fail(v, mutation):
    data = snapshots()
    origins = pd.date_range("2024-01-01 04:00", periods=2, freq="h", tz="UTC")
    if mutation == "naive_origin":
        origins = origins.tz_localize(None)
    elif mutation == "naive_known":
        data["known_at"] = data.known_at.dt.tz_localize(None)
    elif mutation == "duplicate":
        data.loc[1, "known_at"] = data.loc[0, "known_at"]
    elif mutation == "future_known":
        data.loc[0, "known_at"] = data.loc[0, "valid_at"]
    elif mutation == "bad_valid":
        data.loc[0, "valid_at"] += pd.Timedelta(hours=1)
    elif mutation == "nan_weather":
        data.loc[0, "weather_0_f024"] = np.nan
    else:
        data.loc[0, "weather_0_f024"] = np.inf
    with pytest.raises(ValueError):
        v.match_snapshots(origins, data)


def test_control_preserves_mask_original_columns_bounds_and_age(v):
    base = pd.DataFrame({"timestamp": pd.date_range("2024-01-01 03:00", periods=2, freq="h", tz="UTC"),
                         "aqi": [np.nan, 123.], "lower_bound_hours": [1., 2.]})
    actual, control, names = v.rebuild_population(base, snapshots(), ["aqi"])
    assert names[0] == "aqi" and len(names) == 13
    np.testing.assert_array_equal(actual.aqi, base.aqi)
    np.testing.assert_array_equal(actual.lower_bound_hours, base.lower_bound_hours)
    assert np.isnan(control.loc[0, "weather_0_f024"])
    assert control.loc[1, "weather_0_f024"] == 0
    np.testing.assert_array_equal(np.isnan(actual[names]), np.isnan(control[names]))
    v.validate_population(actual, actual.copy(), base)
    for column in ("aqi", "lower_bound_hours", "forecast_run_age_hours"):
        changed = actual.copy()
        changed.loc[1, column] += 1
        with pytest.raises(ValueError):
            v.validate_population(actual, changed, base)


def test_inventory_requires_64_models_and_correct_id(v):
    frame = pd.DataFrame({"continuity_block": np.arange(32)})
    records = [{"arm": arm, "continuity_block": block, "model_id": f"{arm}_b{block}"}
               for arm in v.NEW_ARMS for block in range(32)]
    v.validate_inventory(records, frame)
    for changed in (records[:-1], records + records[:1]):
        with pytest.raises(ValueError):
            v.validate_inventory(changed, frame)
    changed = copy.deepcopy(records)
    changed[0]["model_id"] = "foreign"
    with pytest.raises(ValueError):
        v.validate_inventory(changed, frame)


def test_weather_information_gate_cannot_pass_when_only_metadata_improves(v):
    frame = evaluation()
    predictions = {arm: np.array([40., 4., 40., 10.]) for arm in v.ARMS}
    predictions["metadata_control"] = np.array([30., 2., 30., 6.])
    predictions["weather_values"] = predictions["metadata_control"].copy()
    score, _ = v.compute_scores(frame, predictions)
    assert not score["gates"]["passed"]
    assert score["gates"]["annual"]["2024"]["primary"]["passed"]
    assert not score["gates"]["annual"]["2024"]["weather_information"]["passed"]


def test_both_gates_pass_only_when_weather_adds_value_each_year(v):
    frame = evaluation()
    predictions = {arm: np.array([40., 4., 40., 10.]) for arm in v.ARMS}
    predictions["metadata_control"] = np.array([35., 3., 35., 8.])
    predictions["weather_values"] = np.array([30., 2., 30., 6.])
    scores, processes = v.compute_scores(frame, predictions)
    assert scores["gates"]["passed"]
    assert len(processes) == 8
    rows = v.prediction_rows(frame, predictions)
    assert "weather_values_vs_metadata_control_delta_upper" in rows
    predictions["weather_values"][2] = 40.
    scores, _ = v.compute_scores(frame, predictions)
    assert not scores["gates"]["passed"]


def test_output_hash_inventory_detects_tamper_and_extras(v, tmp_path):
    (tmp_path / "predictions.parquet").write_bytes(b"locked")
    hashes = {"predictions.parquet": v.bridge().sha256(tmp_path / "predictions.parquet")}
    v.validate_output_inventory(tmp_path, hashes, set(hashes))
    (tmp_path / "predictions.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        v.validate_output_inventory(tmp_path, hashes, set(hashes))


def test_existing_verification_refused_before_parent_work(v, tmp_path):
    run = tmp_path / "experiments/web_recovery_gefs_proxy_20261003/runs/test"
    run.mkdir(parents=True)
    (run / "verification.json").write_text("{}")
    with pytest.raises(ValueError, match="already exists"):
        v.verify_run(run, tmp_path)


def test_native_configuration_checks_99_and_seed(v):
    path = HERE.parent / "web_recovery_factors_20261002/runs/20261002T161429405383Z/models.json"
    config = json.loads(path.read_text())["models"][0]["actual_trained_booster_config"]
    config["learner"]["learner_model_param"]["num_feature"] = "99"
    v.bridge().validate_trained_config(config, 99)
    config["learner"]["generic_param"]["seed"] = "43"
    with pytest.raises(ValueError):
        v.bridge().validate_trained_config(config, 99)


def test_exact_valid_boundary_and_future_known_are_never_selected(v):
    data = snapshots().iloc[:1]
    origins = pd.DatetimeIndex(["2024-01-01 03:59Z", "2024-01-01 04:00Z", "2024-01-01 23:59Z", "2024-01-02 00:00Z"])
    result, names = v.match_snapshots(origins, data)
    assert result[names[0]].notna().tolist() == [False, True, True, False]
    assert result.known_at.notna().tolist() == [False, True, True, False]


def test_wrong_timezone_reinterpretation_is_detected_in_population(v):
    base = pd.DataFrame({"timestamp": pd.DatetimeIndex(["2024-01-01 10:00Z"]), "aqi": [120.]})
    expected, _, _ = v.rebuild_population(base, snapshots(), ["aqi"])
    changed = expected.copy()
    changed["known_at"] = changed.known_at.dt.tz_localize(None).dt.tz_localize("Asia/Shanghai")
    with pytest.raises(ValueError):
        v.validate_population(expected, changed, base)


@pytest.mark.parametrize("mutation", ["feature_order", "metadata_value", "missingmask", "origin_clock"])
def test_npz_rejects_schema_mask_control_value_and_timestamp_drift(v, tmp_path, mutation):
    base = pd.DataFrame({"timestamp": pd.date_range("2024-01-01 03:00", periods=2, freq="h", tz="UTC"), "aqi": [123., 120.]})
    true, control, names = v.rebuild_population(base, snapshots(), ["aqi"])
    expected = {"metadata_control": control[names].to_numpy(float), "weather_values": true[names].to_numpy(float)}
    output = {arm: values.copy() for arm, values in expected.items()}
    output.update(timestamps=pd.DatetimeIndex(base.timestamp).asi8.copy(), feature_names=np.array(names))
    if mutation == "feature_order":
        output["feature_names"] = output["feature_names"][::-1]
    elif mutation == "metadata_value":
        output["metadata_control"][1, 1] = 2.
    elif mutation == "missingmask":
        output["metadata_control"][0, 1] = 0.
    else:
        output["timestamps"][0] += 1
    np.savez(tmp_path / "features.npz", **output)
    with pytest.raises(ValueError):
        v.validate_feature_archive(tmp_path / "features.npz", base, names, expected)


def test_parent_members_are_compared_by_value_not_counts(v):
    parent = pd.DataFrame({"arm": ["grid_all_legacy_origins"] * 2, "model_id": ["grid_all_legacy_origins_b1"] * 2,
        "evaluation_block": [1, 1], "population_row": [0, 1], "lower_bound_hours": [2., 4.]})
    parts = []
    for arm in v.NEW_ARMS:
        part = parent.copy()
        part["arm"], part["model_id"] = arm, f"{arm}_b1"
        parts.append(part)
    members = pd.concat(parts, ignore_index=True)
    v.validate_same_members(members, parent)
    changed = members.copy()
    changed.loc[3, "lower_bound_hours"] += 1
    with pytest.raises(ValueError):
        v.validate_same_members(changed, parent)
    with pytest.raises(ValueError):
        v.validate_same_members(pd.concat([members, members.iloc[:1]]), parent)


def test_maturity_excludes_equal_timestamp_equal_label_and_same_block(v):
    cutoff = pd.Timestamp("2024-01-01", tz="Asia/Shanghai")
    earlier = cutoff - pd.Timedelta(hours=1)
    population = pd.DataFrame({"timestamp": [earlier, cutoff, earlier, earlier],
        "label_available_at": [earlier, earlier, cutoff, earlier], "continuity_block": [1, 1, 1, 2]})
    expected = v.bridge().expected_members(population, cutoff, 2)
    assert expected.index.tolist() == [0]


@pytest.mark.parametrize("field,value", [("nthread", "2"), ("seed", "0")])
def test_trained_configuration_rejects_wrong_runtime_parameters(v, field, value):
    path = HERE.parent / "web_recovery_factors_20261002/runs/20261002T161429405383Z/models.json"
    config = json.loads(path.read_text())["models"][0]["actual_trained_booster_config"]
    config["learner"]["learner_model_param"]["num_feature"] = "99"
    config["learner"]["generic_param"][field] = value
    with pytest.raises(ValueError):
        v.bridge().validate_trained_config(config, 99)


def test_native_file_validation_rejects_parent_87_feature_model(v):
    folder = HERE.parent / "web_recovery_factors_20261002/runs/20261002T161429405383Z"
    record = json.loads((folder / "models.json").read_text())["models"][0]
    with pytest.raises(ValueError):
        v.validate_native_file(folder / record["model_path"], ["feature"] * 99)


def test_scores_do_not_cap_long_duration_or_substitute_interval_midpoint(v):
    frame = evaluation()
    frame.loc[0, ["lower_bound_hours", "upper_bound_hours", "exact_hours"]] = [100., 100., 100.]
    predictions = {arm: np.array([68., 2., 30., 0.]) for arm in v.ARMS}
    scores, _ = v.compute_scores(frame, predictions)
    assert scores["annual"]["2024"]["arms"]["archive"]["exact_row_mae"] == 16.
    rows = v.prediction_rows(frame, predictions)
    assert np.isnan(rows.loc[3, "archive_exact_error"])
    assert rows.loc[3, "archive_error_lower"] == 4.
    assert rows.loc[3, "archive_error_upper"] == 8.


def test_protected_inventory_rejects_extra_file(v, tmp_path):
    hashes = {}
    for name in ("artifacts/cnemc_2025_deployment_candidate", "artifacts/strict_history_backtest", "project-showcase", "sites-haze/dist", "releases/runtime_20261002"):
        path = tmp_path / name / "protected.dat"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"protected")
        hashes[str(path.resolve())] = v.bridge().sha256(path)
    v.validate_protected_inventory(tmp_path, hashes)
    extra = tmp_path / "sites-haze/dist/extra.dat"
    extra.write_bytes(b"extra")
    with pytest.raises(ValueError, match="exact inventory"):
        v.validate_protected_inventory(tmp_path, hashes)


@pytest.mark.parametrize("mutation", [None, "feature_order", "rounds", "feature_count", "objective", "scale", "nan", "shape"])
def test_native_reload_predict_mock_contract_without_fit(v, monkeypatch, mutation):
    import xgboost as xgb

    names = [f"feature_{index}" for index in range(99)]
    queries = []

    class FakeBooster:
        def __init__(self, params):
            assert params == {"nthread": 1}
            self.feature_names = names[::-1] if mutation == "feature_order" else names

        def load_model(self, path):
            assert str(path) == "synthetic_model.json"

        def num_boosted_rounds(self):
            return 499 if mutation == "rounds" else 500

        def num_features(self):
            return 87 if mutation == "feature_count" else 99

        def save_config(self):
            return json.dumps({"learner": {"objective": {"name": "reg:squarederror" if mutation == "objective" else "survival:aft",
                "aft_loss_param": {"aft_loss_distribution": "normal", "aft_loss_distribution_scale": "2" if mutation == "scale" else "1.2"}}}})

        def predict(self, query):
            queries.append(query)
            if mutation == "shape":
                return np.array([1.])
            return np.array([np.nan, 2.]) if mutation == "nan" else np.array([1., 2.])

    monkeypatch.setattr(xgb, "Booster", FakeBooster)
    monkeypatch.setattr(xgb, "DMatrix", lambda frame, missing: frame)
    monkeypatch.setattr(v, "validate_native_file", lambda path, columns: None)
    query = np.arange(198, dtype=float).reshape(2, 99)
    if mutation is None:
        np.testing.assert_array_equal(v.reload_native_prediction(Path("synthetic_model.json"), names, query), [1., 2.])
        assert list(queries[0]) == names
        np.testing.assert_array_equal(queries[0].to_numpy(), query)
    else:
        with pytest.raises(ValueError):
            v.reload_native_prediction(Path("synthetic_model.json"), names, query)
