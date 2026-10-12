"""Synthetic independent-verifier regressions, not accuracy evidence."""

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
    path = HERE / "verify_state.py"
    assert path.is_file(), "independent state verifier is not implemented"
    spec = importlib.util.spec_from_file_location("independent_state_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def grid(values):
    return pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=len(values), freq="h", tz="Asia/Shanghai"),
                         "aqi": pd.Series(values)})


def evaluation():
    return pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=4, freq="h", tz="Asia/Shanghai"),
        "year": [2024, 2024, 2025, 2025], "continuity_block": [1, 1, 2, 2], "event_key": ["a", "a", "b", "b"],
        "lower_bound_hours": [30., 2., 30., 4.], "upper_bound_hours": [30., 2., 30., 8.],
        "exact_hours": [30., 2., 30., np.nan]})


def test_no_runner_import_or_mutable_parent_globals(v):
    source = (HERE / "verify_state.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any("runner" in item for item in imports)
    assert not any("runner" in alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)
    assert v.proxy().bridge().sha256(HERE.parent / "web_recovery_gefs_proxy_20261003/verify_proxy.py") == v.PROXY_SHA256
    assert v.proxy().ARMS == ("archive", "grid_all_legacy_origins", "metadata_control", "weather_values")


def test_state_threshold_missing_left_end_and_start_flag(v):
    values = [130., 150., 100., 101., 160., np.nan, 166., 180., 99., np.nan, 100., 140.]
    state = v.observed_aqi_state(grid(values))
    assert list(state) == ["timestamp", *v.STATE_FIELDS]
    np.testing.assert_array_equal(state[v.STATE_FIELDS[0]], [1., 2., 0., 1., 2., np.nan, 1., 2., 0., np.nan, 0., 1.])
    np.testing.assert_array_equal(state[v.STATE_FIELDS[1]], [0., 0., np.nan, 1., 1., np.nan, 0., 0., np.nan, np.nan, np.nan, 1.])


def test_complex_missing_object_is_rejected_before_unknown_conversion(v):
    data = grid([90., 140., 160.])
    data["aqi"] = pd.Series([90., complex(np.nan, 0.), 140.], dtype=object)
    with pytest.raises(ValueError, match="real"):
        v.observed_aqi_state(data)


@pytest.mark.parametrize("missing", [pd.NaT, np.datetime64("NaT")])
def test_non_numeric_missing_object_is_not_aqi_nan(v, missing):
    data = grid([90., 140., 160.])
    data["aqi"] = pd.Series([90., missing, 140.], dtype=object)
    with pytest.raises(ValueError, match="numeric"):
        v.observed_aqi_state(data)


def test_declared_output_inventory_is_exactly_64_models_plus_ten_artifacts(v):
    frame = pd.DataFrame({"continuity_block": np.arange(32)})
    expected = v.expected_output_inventory(frame)
    assert len(expected) == 74
    assert "execution_record_at_start.md" in expected
    assert "models/state_values_b31.json" in expected
    assert "models/weather_values_b31.json" not in expected


def test_short_clean_and_sixhour_clean_do_not_change_observation_contract(v):
    data = grid([140.] * 35 + [90.] + [130.] * 3 + [90.] * 6 + [160.] * 3)
    state = v.observed_aqi_state(data)
    assert state.loc[34, v.STATE_FIELDS[0]] == 35.
    assert state.loc[35, v.STATE_FIELDS[0]] == 0.
    assert state.loc[36:38, v.STATE_FIELDS[1]].eq(1).all()
    assert state.loc[45:47, v.STATE_FIELDS[1]].eq(1).all()


def test_state_only_uses_timestamp_and_aqi_not_labels_particle_or_id(v):
    data = grid([90., 166., 155., np.nan, 140.])
    expected = v.observed_aqi_state(data)
    for name in ("pm2_5", "pm10", "pm2_5_iaqi", "eligible", "continuity_block", "episode_id", "exact_hours", "label_available_at"):
        data[name] = [object()] * len(data)
    actual = v.observed_aqi_state(data)
    pd.testing.assert_frame_equal(expected, actual)


def test_every_prefix_and_future_perturbation_are_causal(v):
    data = grid([np.nan, 130., 170., 95., 120., 180., np.nan, 90., 101., 102.])
    expected = v.observed_aqi_state(data)
    for end in range(1, len(data) + 1):
        pd.testing.assert_frame_equal(expected.iloc[:end].reset_index(drop=True), v.observed_aqi_state(data.iloc[:end]))
        changed = data.copy()
        changed.loc[end:, "aqi"] = np.nan if end % 2 else 200.
        pd.testing.assert_frame_equal(expected.iloc[:end].reset_index(drop=True), v.observed_aqi_state(changed).iloc[:end])


@pytest.mark.parametrize("mutation", ["empty", "naive", "duplicate", "unsorted", "gap", "fractional", "missing_time", "missing_aqi", "inf", "negative_inf", "bool", "mixed_bool", "complex", "text"])
def test_invalid_time_or_aqi_fails_closed(v, mutation):
    data = grid([90., 140., 160.])
    if mutation == "empty":
        data = data.iloc[:0]
    elif mutation == "naive":
        data["timestamp"] = data.timestamp.dt.tz_localize(None)
    elif mutation == "duplicate":
        data.loc[1, "timestamp"] = data.loc[0, "timestamp"]
    elif mutation == "unsorted":
        data = data.iloc[::-1]
    elif mutation == "gap":
        data.loc[2, "timestamp"] += pd.Timedelta(hours=1)
    elif mutation == "fractional":
        data["timestamp"] += pd.Timedelta(minutes=1)
    elif mutation == "missing_time":
        data.loc[1, "timestamp"] = pd.NaT
    elif mutation == "missing_aqi":
        data = data.drop(columns="aqi")
    elif mutation in ("inf", "negative_inf"):
        data.loc[1, "aqi"] = np.inf if mutation == "inf" else -np.inf
    elif mutation == "bool":
        data["aqi"] = [True, False, True]
    elif mutation == "mixed_bool":
        data["aqi"] = pd.Series([90., True, 140.], dtype=object)
    elif mutation == "complex":
        data["aqi"] = [90 + 0j, 140 + 0j, 160 + 0j]
    else:
        data["aqi"] = ["90", "140", "160"]
    with pytest.raises(ValueError):
        v.observed_aqi_state(data)


def test_full_grid_mapping_99_columns_and_equal_control_masks(v):
    data = grid([90., 140., np.nan, 170., 180.])
    names = [f"f_{i}" for i in range(99)]
    parent = pd.DataFrame({"timestamp": data.timestamp.iloc[[1, 3, 4]].reset_index(drop=True), "population_row": [0, 1, 2]})
    for i, name in enumerate(names):
        parent[name] = [float(i), np.nan, float(i + 1)]
    state, control, all_names = v.rebuild_population(parent, data, names)
    assert all_names == [*names, *v.STATE_FIELDS]
    pd.testing.assert_frame_equal(state[list(parent)], parent)
    np.testing.assert_array_equal(state[v.STATE_FIELDS[0]], [1., 1., 2.])
    np.testing.assert_array_equal(state[v.STATE_FIELDS[1]], [1., 0., 0.])
    np.testing.assert_array_equal(np.isnan(state[all_names]), np.isnan(control[all_names]))
    assert control[list(v.STATE_FIELDS)].eq(0).all().all()
    v.validate_population(state, state.copy(), parent)
    changed = state.copy()
    changed.loc[0, "f_0"] += 1.
    with pytest.raises(ValueError):
        v.validate_population(state, changed, parent)
    parent.loc[1, "timestamp"] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError):
        v.rebuild_population(parent, data, names)


def test_population_rejects_duplicate_origin_or_state_name_collision(v):
    data = grid([90., 140.])
    names = [f"f_{i}" for i in range(99)]
    base = pd.DataFrame({"timestamp": data.timestamp, **{name: [0., 1.] for name in names}})
    duplicate = pd.concat([base, base.iloc[:1]])
    with pytest.raises(ValueError):
        v.rebuild_population(duplicate, data, names)
    names[-1] = v.STATE_FIELDS[0]
    with pytest.raises(ValueError):
        v.rebuild_population(base, data, names)


def test_model_inventory_exact_64_and_identity(v):
    frame = pd.DataFrame({"continuity_block": np.arange(32)})
    records = [{"arm": arm, "continuity_block": b, "model_id": f"{arm}_b{b}"} for arm in v.NEW_ARMS for b in range(32)]
    v.validate_inventory(records, frame)
    for changed in (records[:-1], records + records[:1]):
        with pytest.raises(ValueError):
            v.validate_inventory(changed, frame)
    changed = copy.deepcopy(records)
    changed[0]["model_id"] = "foreign"
    with pytest.raises(ValueError):
        v.validate_inventory(changed, frame)


def test_state_gate_does_not_credit_mask_control_only(v):
    frame = evaluation()
    predictions = {arm: np.array([40., 4., 40., 10.]) for arm in v.ARMS}
    predictions["state_control"] = np.array([30., 2., 30., 6.])
    predictions["state_values"] = predictions["state_control"].copy()
    scores, _ = v.compute_scores(frame, predictions)
    assert scores["gates"]["annual"]["2024"]["primary"]["passed"]
    assert not scores["gates"]["annual"]["2024"]["state_information"]["passed"]
    assert not scores["gates"]["passed"]


def scope_report(v):
    return {"schema_version": 1, "status": "pending_independent_verification", "production_ready": False,
        "deployed": False, "blind_test": False, "historical_shadow_evaluation": True,
        "archive_is_primary_reference": True, "protected_unchanged": True, "parent_unchanged": True,
        "primary_candidate": "state_values", "diagnostic_only_arms": ["state_control", "weather_values"],
        "new_refitted_arms": list(v.NEW_ARMS), "inherited_arms": list(v.INHERITED_ARMS),
        "years": [2024, 2025], "forecast_time_semantics": "archive_timestamp_proxy",
        "scope": v.SCOPE, "state_time_semantics": v.STATE_TIME_SEMANTICS,
        "state_scope": v.STATE_SCOPE, "state_fields": list(v.STATE_FIELDS)}


def test_scope_keeps_inherited_weather_diagnostic_not_primary(v):
    report = scope_report(v)
    v.validate_scope(report)
    for key, value in (("production_ready", True), ("deployed", True), ("primary_candidate", "weather_values"),
                       ("blind_test", True), ("inherited_arms", ["archive"]), ("state_scope", "live"),
                       ("state_fields", list(v.STATE_FIELDS)[::-1]), ("years", [2025])):
        changed = copy.deepcopy(report)
        changed[key] = value
        with pytest.raises(ValueError):
            v.validate_scope(changed)


def test_scope_boolean_one_cannot_masquerade_as_true(v):
    report = scope_report(v)
    report["protected_unchanged"] = 1
    with pytest.raises(ValueError):
        v.validate_scope(report)


def test_both_years_and_both_gates_are_required(v):
    frame = evaluation()
    predictions = {arm: np.array([40., 4., 40., 10.]) for arm in v.ARMS}
    predictions["state_control"] = np.array([35., 3., 35., 8.])
    predictions["state_values"] = np.array([30., 2., 30., 6.])
    scores, process = v.compute_scores(frame, predictions)
    assert scores["gates"]["passed"]
    assert len(process) == 8
    assert "state_values_vs_weather_values_delta_upper" in v.prediction_rows(frame, predictions)
    predictions["state_values"][2] = 40.
    assert not v.compute_scores(frame, predictions)[0]["gates"]["passed"]


def test_full_duration_and_interval_bounds_not_caps_or_midpoints(v):
    frame = evaluation()
    frame.loc[0, ["lower_bound_hours", "upper_bound_hours", "exact_hours"]] = [100., 100., 100.]
    predictions = {arm: np.array([68., 2., 30., 0.]) for arm in v.ARMS}
    scores, _ = v.compute_scores(frame, predictions)
    assert scores["annual"]["2024"]["arms"]["archive"]["exact_row_mae"] == 16.
    rows = v.prediction_rows(frame, predictions)
    assert np.isnan(rows.loc[3, "archive_exact_error"])
    assert rows.loc[3, "archive_error_lower"] == 4.
    assert rows.loc[3, "archive_error_upper"] == 8.


@pytest.mark.parametrize("mutation", ["feature_order", "control_value", "missingmask", "timestamp"])
def test_features_npz_exact_schema_and_all_values(v, tmp_path, mutation):
    names = [f"f_{i}" for i in range(99)] + list(v.STATE_FIELDS)
    frame = evaluation().iloc[:2]
    expected = {arm: np.arange(202., dtype=float).reshape(2, 101) for arm in v.NEW_ARMS}
    expected["state_control"][:, -2:] = 0.
    expected["state_values"][0, 100] = np.nan
    expected["state_control"][0, 100] = np.nan
    output = {arm: values.copy() for arm, values in expected.items()}
    output.update(timestamps=pd.DatetimeIndex(frame.timestamp).asi8.copy(), feature_names=np.array(names))
    if mutation == "feature_order":
        output["feature_names"] = output["feature_names"][::-1]
    elif mutation == "control_value":
        output["state_control"][1, 100] = 2.
    elif mutation == "missingmask":
        output["state_control"][0, 100] = 0.
    else:
        output["timestamps"][0] += 1
    np.savez(tmp_path / "features.npz", **output)
    with pytest.raises(ValueError):
        v.validate_feature_archive(tmp_path / "features.npz", frame, names, expected)


def test_same_members_compared_by_value_not_counts(v):
    parent = pd.DataFrame({"arm": ["weather_values"] * 2, "model_id": ["weather_values_b1"] * 2,
        "evaluation_block": [1, 1], "population_row": [0, 1], "lower_bound_hours": [2., 4.]})
    parts = []
    for arm in v.NEW_ARMS:
        part = parent.copy()
        part["arm"], part["model_id"] = arm, f"{arm}_b1"
        parts.append(part)
    members = pd.concat(parts, ignore_index=True)
    v.validate_same_members(members, parent)
    changed = members.copy()
    changed.loc[3, "lower_bound_hours"] += 1.
    for bad in (changed, pd.concat([members, members.iloc[:1]])):
        with pytest.raises(ValueError):
            v.validate_same_members(bad, parent)


def test_existing_verification_rejected_before_parent_checks(v, tmp_path):
    run = tmp_path / "experiments/web_recovery_aqi_state_20261004/runs/test"
    run.mkdir(parents=True)
    (run / "verification.json").write_text("{}")
    with pytest.raises(ValueError, match="already exists"):
        v.verify_run(run, tmp_path)


def test_output_inventory_rejects_extras_and_tamper(v, tmp_path):
    path = tmp_path / "predictions.parquet"
    path.write_bytes(b"locked")
    hashes = {"predictions.parquet": v.bridge().sha256(path)}
    v.validate_output_inventory(tmp_path, hashes, set(hashes))
    extra = tmp_path / "extra.txt"
    extra.write_bytes(b"extra")
    with pytest.raises(ValueError):
        v.validate_output_inventory(tmp_path, hashes, set(hashes))
    extra.unlink()
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError):
        v.validate_output_inventory(tmp_path, hashes, set(hashes))


def test_native_parameters_are_fixed_101_seed_and_single_thread(v):
    parent = HERE.parent / "web_recovery_gefs_proxy_20261003/runs/20261003T145140578727Z/models.json"
    config = json.loads(parent.read_text())["models"][0]["actual_trained_booster_config"]
    config["learner"]["learner_model_param"]["num_feature"] = "101"
    v.bridge().validate_trained_config(config, 101)
    config["learner"]["generic_param"]["seed"] = "43"
    with pytest.raises(ValueError):
        v.bridge().validate_trained_config(config, 101)


def test_parent_99_native_model_is_not_valid_101_model(v):
    parent = HERE.parent / "web_recovery_gefs_proxy_20261003/runs/20261003T145140578727Z"
    record = json.loads((parent / "models.json").read_text())["models"][0]
    with pytest.raises(ValueError):
        v.validate_native_file(parent / record["model_path"], [f"f_{i}" for i in range(101)])


@pytest.mark.parametrize("mutation", [None, "feature_order", "rounds", "feature_count", "objective", "scale", "nan", "shape"])
def test_native_reload_mock_contract_not_real_accuracy(v, monkeypatch, mutation):
    import xgboost as xgb

    names = [f"f_{i}" for i in range(101)]
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
            return 99 if mutation == "feature_count" else 101

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
    query = np.arange(202., dtype=float).reshape(2, 101)
    if mutation is None:
        np.testing.assert_array_equal(v.reload_native_prediction(Path("synthetic_model.json"), names, query), [1., 2.])
        assert list(queries[0]) == names
        np.testing.assert_array_equal(queries[0].to_numpy(), query)
    else:
        with pytest.raises(ValueError):
            v.reload_native_prediction(Path("synthetic_model.json"), names, query)
