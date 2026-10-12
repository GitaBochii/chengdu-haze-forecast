import pandas as pd
import pytest


def test_probability_output_rejects_a_model_without_accepted_mae_decision():
    from haze_forecast.probability import build_probability_output

    predictions = pd.DataFrame(
        {
            "fold": [1, 1],
            "actual_remaining_hours": [12.0, 24.0],
            "predicted_remaining_hours": [10.0, 20.0],
        }
    )

    with pytest.raises(ValueError, match="accepted"):
        build_probability_output(predictions, decision={"decision": "reject"})


def test_probability_output_requires_three_rolling_validation_folds():
    from haze_forecast.probability import build_probability_output

    predictions = pd.DataFrame(
        {
            "fold": [1, 1, 2, 2],
            "actual_remaining_hours": [8.0, 20.0, 32.0, 49.0],
            "predicted_remaining_hours": [10.0, 20.0, 30.0, 45.0],
        }
    )

    with pytest.raises(ValueError, match="at least three"):
        build_probability_output(
            predictions, decision={"decision": "candidate_for_replacement_review"}
        )


def test_probability_output_uses_rolling_residuals_for_intervals_probabilities_and_coverage():
    from haze_forecast.probability import build_probability_output

    predictions = pd.DataFrame(
        {
            "fold": [1, 2, 3, 3],
            "actual_remaining_hours": [8.0, 20.0, 32.0, 49.0],
            "predicted_remaining_hours": [10.0, 20.0, 30.0, 45.0],
        }
    )

    result = build_probability_output(
        predictions, decision={"decision": "candidate_for_replacement_review"}
    )
    output = result["predictions"]

    assert output["median_remaining_hours"].tolist() == [11.0, 21.0, 31.0, 46.0]
    assert output["interval_50_low_hours"].tolist() == [9.5, 19.5, 29.5, 44.5]
    assert output["interval_50_high_hours"].tolist() == [12.5, 22.5, 32.5, 47.5]
    assert output["interval_80_low_hours"].tolist() == [8.6, 18.6, 28.6, 43.6]
    assert output["interval_80_high_hours"].tolist() == [13.4, 23.4, 33.4, 48.4]
    assert output["probability_recovery_within_24h"].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert output["probability_recovery_within_48h"].tolist() == [1.0, 1.0, 1.0, 0.75]
    assert result["coverage"] == {"rows": 4, "interval_50": 0.5, "interval_80": 0.5}
