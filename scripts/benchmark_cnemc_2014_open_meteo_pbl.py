"""Benchmark a 2014 PBL-history candidate on untouched 2022 and 2023 labels."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from haze_forecast.cnemc_candidate import build_cnemc_2014_candidate
from haze_forecast.evaluation import regression_metrics
from haze_forecast.features import add_missingness_indicators

ROOT = Path(__file__).resolve().parents[1]
PBL_COLUMNS = [
    "boundary_layer_height_m",
    "boundary_layer_height_m_lag_1",
    "boundary_layer_height_m_lag_3",
    "boundary_layer_height_m_lag_24",
    "boundary_layer_height_m_roll_3",
    "boundary_layer_height_m_roll_24",
    "boundary_layer_height_m_change_3",
]


def _with_pbl_flags(frame: pd.DataFrame) -> pd.DataFrame:
    return add_missingness_indicators(frame, PBL_COLUMNS)


def _predict(model_path: Path, metadata_path: Path, frame: pd.DataFrame) -> tuple[pd.Series, dict[str, float]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    features = metadata["feature_names"]
    model = joblib.load(model_path)
    prediction = pd.Series(model.predict(_with_pbl_flags(frame)[features]), index=frame.index).clip(lower=0)
    return prediction, regression_metrics(frame["remaining_hours"], prediction)


def _relative_improvement(reference_mae: float, candidate_mae: float) -> float:
    return float(1 - candidate_mae / reference_mae)


def main() -> None:
    base = pd.read_parquet(ROOT / "data" / "processed" / "training_features.parquet")
    prefix = pd.read_parquet(
        ROOT / "data" / "processed" / "cnemc_chengdu_2014_pbl_training_features.parquet"
    ).copy()
    prefix["event_id"] = "cnemc_2014_pbl_" + prefix["event_id"].astype(str)
    candidate_dir = ROOT / "artifacts" / "cnemc_2014_open_meteo_pbl_candidate"
    candidate_result = build_cnemc_2014_candidate(base, prefix, candidate_dir)

    reference_dir = ROOT / "artifacts" / "cnemc_2014_candidate"
    external_results: dict[str, object] = {}
    improvements: list[float] = []
    for year in (2022, 2023):
        external = pd.read_parquet(
            ROOT / "data" / "processed" / f"cnemc_chengdu_{year}_training_features.parquet"
        )
        _, reference_metrics = _predict(
            reference_dir / "aft_model.joblib", reference_dir / "model_metadata.json", external
        )
        _, pbl_metrics = _predict(
            candidate_dir / "aft_model.joblib", candidate_dir / "model_metadata.json", external
        )
        improvement = _relative_improvement(reference_metrics["mae"], pbl_metrics["mae"])
        improvements.append(improvement)
        external_results[str(year)] = {
            "rows": len(external),
            "reference_2014_candidate": reference_metrics,
            "open_meteo_pbl_candidate": pbl_metrics,
            "relative_mae_improvement": improvement,
        }

    average_improvement = float(sum(improvements) / len(improvements))
    max_relative_worsening = float(max(-improvement for improvement in improvements))
    accepted = average_improvement >= 0.03 and max_relative_worsening <= 0.05
    report = {
        "protocol": "2022 and 2023 are untouched external label sets; historical Open-Meteo PBL is candidate-only",
        "comparison": "2014 PBL candidate against the already validated 2014 candidate",
        "base_holdout_metrics": candidate_result["metrics"],
        "external_results": external_results,
        "average_external_relative_mae_improvement": average_improvement,
        "max_external_relative_mae_worsening": max_relative_worsening,
        "decision": "accept_for_further_validation" if accepted else "reject_no_default_replacement",
        "acceptance_gate": "mean external MAE improvement >= 3% and no external year worsens by > 5%",
    }
    report_path = ROOT / "artifacts" / "cnemc_2014_open_meteo_pbl_candidate" / "external_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
