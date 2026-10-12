import pandas as pd

from haze_forecast.app import (
    build_history_prediction_input,
    build_station_history_prediction_input,
    candidate_prediction_resource_paths,
    constrain_live_prediction,
    dissipation_criterion,
    evaluation_metric_key,
    external_history_data_filename,
    history_backtest_feature_frame,
    history_backtest_source_year,
    history_error_label,
    manual_feature_frame,
    model_specification,
    model_version_options,
    pollutant_display_frame,
    populate_missingness_indicators,
    prediction_message,
    prediction_resource_paths,
    primary_particle_text,
    target_resource_paths,
    training_history_backtest_data,
    training_sample_count,
    trend_data_filename,
    uncertainty_band,
)


def test_prediction_message_rejects_non_pollution_state():
    assert prediction_message(False, 3.2, "2021-01-01 12:00") == "当前不满足颗粒物超标污染条件"


def test_dissipation_criterion_exposes_target_specific_recovery_rules():
    overall = dissipation_criterion("整体空气质量恢复")
    particle = dissipation_criterion("颗粒物雾霾消散")

    assert overall.threshold_label == "AQI ≤ 100"
    assert "6" in overall.detail
    assert overall.aqi_threshold == 100.0
    assert particle.threshold_label == "不设固定 AQI 阈值"
    assert "IAQI" in particle.detail
    assert particle.aqi_threshold is None


def test_pollutant_display_uses_chinese_labels_and_units():
    displayed = pollutant_display_frame(pd.Series({"aqi": 130.0, "pm2_5": 90.0, "co": 1.1}))

    assert displayed.to_dict("records") == [
        {"污染物": "AQI", "数值": 130.0, "单位": ""},
        {"污染物": "PM2.5", "数值": 90.0, "单位": "µg/m³"},
        {"污染物": "CO", "数值": 1.1, "单位": "mg/m³"},
    ]


def test_primary_particle_text_identifies_all_exceeding_particles():
    assert primary_particle_text({"pm2_5": 80, "pm10": 180}) == "PM2.5、PM10"
    assert primary_particle_text({"pm2_5": 70, "pm10": 120}) == "无颗粒物超标"


def test_manual_mode_uses_snapshot_model_features_and_validation_interval():
    metadata = {
        "feature_names": ["aqi", "aqi_lag_1"],
        "snapshot_feature_names": ["aqi"],
        "selected_history_model_filename": "aft_model.joblib",
        "selected_history_validation_mae": 16.0,
        "selected_snapshot_model_filename": "static_catboost_model.joblib",
        "selected_snapshot_validation_mae": 20.0,
    }
    metrics = {}

    feature_names, validation_mae, model_filename = model_specification(metadata, metrics, manual=True)

    assert feature_names == ["aqi"]
    assert validation_mae == 20.0
    assert model_filename == "static_catboost_model.joblib"


def test_manual_feature_frame_does_not_add_unrequested_features():
    values = {
        "aqi": 130.0,
        "pm2_5": 90.0,
        "pm10": 120.0,
        "wind_u_ms": 1.0,
        "wind_v_ms": 0.0,
    }

    frame = manual_feature_frame(["aqi", "pm2_5", "wind_speed_ms"], values, pd.Timestamp("2021-01-01"))

    assert frame.columns.tolist() == ["aqi", "pm2_5", "wind_speed_ms"]


def test_missingness_indicator_follows_its_source_value():
    result = populate_missingness_indicators(
        pd.DataFrame(
            {
                "boundary_layer_height_m": [250.0, float("nan")],
                "boundary_layer_height_m_is_missing": [float("nan"), float("nan")],
            }
        )
    )

    assert result["boundary_layer_height_m_is_missing"].tolist() == [0.0, 1.0]


def test_missingness_indicator_derives_change_3_from_its_two_inputs():
    result = populate_missingness_indicators(
        pd.DataFrame(
            {
                "boundary_layer_height_m": [250.0, 180.0, float("nan")],
                "boundary_layer_height_m_lag_3": [300.0, float("nan"), 210.0],
                "boundary_layer_height_m_change_3_is_missing": [float("nan")] * 3,
            }
        )
    )

    assert result["boundary_layer_height_m_change_3_is_missing"].tolist() == [0.0, 1.0, 1.0]


def test_history_backtest_feature_frame_populates_candidate_missingness_indicators():
    row = pd.Series(
        {
            "aqi": 130.0,
            "boundary_layer_height_m": float("nan"),
            "boundary_layer_height_m_lag_1": float("nan"),
        }
    )
    feature_names = [
        "aqi",
        "boundary_layer_height_m",
        "boundary_layer_height_m_is_missing",
        "boundary_layer_height_m_lag_1",
        "boundary_layer_height_m_lag_1_is_missing",
    ]

    result = history_backtest_feature_frame(row, feature_names)

    assert result.columns.tolist() == feature_names
    assert pd.isna(result.iloc[0]["boundary_layer_height_m"])
    assert result.iloc[0]["boundary_layer_height_m_is_missing"] == 1.0
    assert result.iloc[0]["boundary_layer_height_m_lag_1_is_missing"] == 1.0


def test_training_sample_count_supports_baseline_and_candidate_metadata_schemas():
    assert training_sample_count({"rows": {"train": 3872}}) == 3872
    assert training_sample_count({"final_training_rows": 6905}) == 6905


def test_particle_target_uses_its_own_data_and_model_artifact_directory():
    data_name, artifact_directory = target_resource_paths("颗粒物雾霾消散")

    assert data_name == "particle_dissipation_training_features.parquet"
    assert artifact_directory == "particle_dissipation"


def test_external_history_data_filename_maps_the_new_forward_years():
    assert external_history_data_filename(2022) == "cnemc_chengdu_2022_training_features.parquet"
    assert external_history_data_filename(2023) == "cnemc_chengdu_2023_training_features.parquet"
    assert external_history_data_filename(2024) == "cnemc_chengdu_2024_training_features.parquet"
    assert external_history_data_filename(2025) == "cnemc_chengdu_2025_training_features.parquet"


def test_history_backtest_source_year_maps_replay_sources_without_claiming_external_accuracy():
    assert history_backtest_source_year("历史记录（2014-2021）") is None
    assert history_backtest_source_year("历史记录（2022）") == 2022
    assert history_backtest_source_year("历史记录（2023）") == 2023
    assert history_backtest_source_year("历史记录（2024）") == 2024
    assert history_backtest_source_year("历史记录（2025）") == 2025


def test_training_history_backtest_data_does_not_mix_later_years_into_the_2014_to_2021_choice():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2021-12-31 23:00", "2022-01-01 00:00"]),
            "aqi": [120.0, 130.0],
        }
    )

    result = training_history_backtest_data(frame)

    assert result["timestamp"].dt.year.tolist() == [2021]


def test_history_error_label_distinguishes_external_backtest_from_training_replay():
    assert history_error_label(False) == "本次绝对误差"
    assert history_error_label(True) == "历史回放差异"


def test_trend_data_uses_continuous_labeled_observations_not_training_rows():
    assert trend_data_filename("整体空气质量恢复", use_spatial=False) == "city_labeled.parquet"
    assert trend_data_filename("整体空气质量恢复", use_spatial=True) == "spatial_city_labeled.parquet"
    assert (
        trend_data_filename("颗粒物雾霾消散", use_spatial=False)
        == "particle_dissipation_city_labeled.parquet"
    )


def test_spatial_prediction_uses_its_own_feature_data_and_artifacts():
    data_name, artifact_directory = prediction_resource_paths("整体空气质量恢复", use_spatial=True)

    assert data_name == "spatial_training_features.parquet"
    assert artifact_directory == "spatial_transport"


def test_2014_candidate_uses_its_own_model_artifacts_without_changing_input_schema():
    data_name, artifact_directory = candidate_prediction_resource_paths(
        "整体空气质量恢复", use_spatial=False, use_cnemc_2014_candidate=True
    )

    assert data_name == "training_features.parquet"
    assert artifact_directory == "cnemc_2014_candidate"


def test_full_history_candidate_uses_the_2014_to_2025_training_view_and_isolated_artifacts():
    data_name, artifact_directory = candidate_prediction_resource_paths(
        "整体空气质量恢复",
        use_spatial=False,
        use_cnemc_2014_candidate=False,
        use_full_history_deployment_candidate=True,
    )

    assert data_name == "cnemc_chengdu_2014_2025_deployment_training_features.parquet"
    assert artifact_directory == "cnemc_2025_deployment_candidate"


def test_main_prediction_target_exposes_only_the_full_history_deployment_model():
    assert model_version_options(candidate_available=True) == ("2025全量部署模型（预测未来）",)
    assert model_version_options(candidate_available=False) == ("基准模型",)


def test_history_input_builds_latest_lagged_feature_row_from_continuous_observations():
    timestamps = pd.date_range("2021-01-01", periods=25, freq="h", tz="Asia/Shanghai")
    history = pd.DataFrame(
        {
            "timestamp": timestamps,
            "aqi": [130.0] * 25,
            "pm2_5": list(range(80, 105)),
            "pm10": [120.0] * 25,
            "so2": [12.0] * 25,
            "no2": [35.0] * 25,
            "co": [1.0] * 25,
            "o3": [50.0] * 25,
            "temperature_c": [12.0] * 25,
            "relative_humidity_pct": [70.0] * 25,
            "wind_u_ms": [1.0] * 25,
            "wind_v_ms": [0.0] * 25,
            "boundary_layer_height_m": [300.0] * 25,
        }
    )

    latest, features = build_history_prediction_input(history, ["pm2_5", "pm2_5_lag_1", "pm2_5_lag_24"])

    assert latest["timestamp"] == timestamps[-1]
    assert features.iloc[0].to_dict() == {"pm2_5": 104.0, "pm2_5_lag_1": 103.0, "pm2_5_lag_24": 80.0}


def test_history_input_populates_candidate_missingness_indicators():
    timestamps = pd.date_range("2021-01-01", periods=25, freq="h", tz="Asia/Shanghai")
    history = pd.DataFrame(
        {
            "timestamp": timestamps,
            "aqi": [130.0] * 25,
            "pm2_5": [90.0] * 25,
            "pm10": [120.0] * 25,
            "so2": [12.0] * 25,
            "no2": [35.0] * 25,
            "co": [1.0] * 25,
            "o3": [50.0] * 25,
            "temperature_c": [12.0] * 25,
            "relative_humidity_pct": [70.0] * 25,
            "wind_u_ms": [1.0] * 25,
            "wind_v_ms": [0.0] * 25,
            "boundary_layer_height_m": [float("nan")] * 25,
        }
    )

    _, features = build_history_prediction_input(
        history,
        ["boundary_layer_height_m", "boundary_layer_height_m_is_missing"],
    )

    assert pd.isna(features.iloc[0]["boundary_layer_height_m"])
    assert features.iloc[0]["boundary_layer_height_m_is_missing"] == 1.0


def test_station_history_input_builds_spatial_features_from_the_latest_network_snapshot():
    timestamps = pd.date_range("2021-01-01", periods=25, freq="h", tz="Asia/Shanghai")
    rows = []
    for index, timestamp in enumerate(timestamps):
        for station_id, offset, wind_u in (("west", 80.0, 0.0), ("east", 100.0, 1.0)):
            rows.append(
                {
                    "timestamp": timestamp,
                    "station_id": station_id,
                    "aqi": 130.0,
                    "pm2_5": offset + index,
                    "pm10": offset + index + 40.0,
                    "so2": 12.0,
                    "no2": 35.0,
                    "co": 1.0,
                    "o3": 50.0,
                    "temperature_c": 12.0,
                    "relative_humidity_pct": 70.0,
                    "wind_u_ms": wind_u,
                    "wind_v_ms": 0.0,
                    "boundary_layer_height_m": 300.0,
                }
            )

    latest, features = build_station_history_prediction_input(
        pd.DataFrame(rows),
        ["pm2_5", "pm2_5_lag_1", "spatial_pm2_5_range", "spatial_upwind_pm2_5_mean"],
        {"west": [103.0, 30.0], "east": [104.0, 30.0]},
    )

    assert latest["timestamp"] == timestamps[-1]
    assert features.iloc[0].to_dict() == {
        "pm2_5": 114.0,
        "pm2_5_lag_1": 113.0,
        "spatial_pm2_5_range": 20.0,
        "spatial_upwind_pm2_5_mean": 104.0,
    }


def test_live_constraint_reduces_only_a_consecutive_hour_upward_revision():
    previous_time = pd.Timestamp("2021-01-01 12:00", tz="Asia/Shanghai")

    constrained = constrain_live_prediction(
        previous_time, 10.0, previous_time + pd.Timedelta(hours=1), 15.0
    )

    assert constrained == 9.0
    assert constrain_live_prediction(previous_time, 10.0, previous_time + pd.Timedelta(hours=2), 15.0) == 15.0


def test_sequence_mode_reports_the_state_constrained_evaluation_metric():
    assert evaluation_metric_key(manual=False, sequence=True, model_filename="aft_model.joblib") == "xgboost_aft"
    assert (
        evaluation_metric_key(manual=True, sequence=False, model_filename="static_catboost_model.joblib")
        == "catboost_snapshot"
    )


def _calibration_payload() -> dict[str, object]:
    return {
        "default_coverage": 0.9,
        "calibration": {
            "0.90": {"coverage": 0.9, "lower_ratio": 0.25, "upper_ratio": 4.0, "samples": 100}
        },
        "evaluation": [
            {
                "source": "base_holdout",
                "coverage_target": 0.9,
                "rows": 100,
                "observed_coverage": 0.927,
                "mean_width_hours": 153.4,
                "used_for_calibration": True,
            },
            {
                "source": "external_2023",
                "coverage_target": 0.9,
                "rows": 100,
                "observed_coverage": 0.857,
                "mean_width_hours": 87.1,
                "used_for_calibration": False,
            },
        ],
    }


def test_uncertainty_band_uses_the_calibrated_asymmetric_interval():
    text, caption = uncertainty_band(20.0, 22.26, _calibration_payload())

    assert text == "5.0 - 80.0 小时"
    assert "90%" in caption
    assert "86%" in caption
    assert "93%" in caption


def test_full_history_snapshot_does_not_borrow_history_validation():
    metadata = {
        "deployment_role": "full_history_refit_for_future_predictions",
        "snapshot_feature_names": ["aqi"],
        "selected_snapshot_model_filename": "static_aft_model.joblib",
        "selected_snapshot_validation_mae": 26.267,
    }
    names, width, filename = model_specification(metadata, {}, manual=True)
    assert names == ["aqi"]
    assert filename == "static_aft_model.joblib"
    assert width is None
    text, caption = uncertainty_band(20., width, _calibration_payload())
    assert text == "暂不可用"
    assert "单独验证" in caption


def test_uncertainty_band_falls_back_to_symmetric_width_without_calibration():
    text, caption = uncertainty_band(20.0, 22.26, None)

    assert text == "0.0 - 42.3 小时"
    assert "未经覆盖率校准" in caption


def test_uncertainty_band_tolerates_a_calibration_document_without_levels():
    text, caption = uncertainty_band(30.0, 10.0, {"default_coverage": 0.9})

    assert text == "20.0 - 40.0 小时"
    assert "未经覆盖率校准" in caption
