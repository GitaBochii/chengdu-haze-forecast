"""Select operational GEFS features on 2023 and confirm on 2024--2025."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.features import add_missingness_indicators
from haze_forecast.gfs_external_validation import event_equal_mae
from haze_forecast.operational_gefs_benchmark import (
    VARIANTS,
    extracted_keys_are_complete,
    publication_audited_features,
    safe_event_year_split,
    select_2023_variant,
    should_accept_for_deployment,
)
from haze_forecast.operational_gefs_dataset import OPERATIONAL_GEFS_FEATURES
from haze_forecast.training import AFTSurvivalRegressor

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
PBL_COLUMNS = [
    "boundary_layer_height_m", "boundary_layer_height_m_lag_1", "boundary_layer_height_m_lag_3",
    "boundary_layer_height_m_lag_24", "boundary_layer_height_m_roll_3",
    "boundary_layer_height_m_roll_24", "boundary_layer_height_m_change_3",
]


def _prepare(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="raise")
    result["event_key"] = source + ":" + result["event_id"].astype(str)
    result = add_missingness_indicators(result, PBL_COLUMNS)
    result["boundary_layer_height_m_change_3_is_missing"] = (
        result["boundary_layer_height_m"].isna() | result["boundary_layer_height_m_lag_3"].isna()
    ).astype(float)
    return result


def _attach_operational_gefs(frame: pd.DataFrame, year: int, directory: Path) -> pd.DataFrame:
    path = directory / f"operational_gefs_event_features_{year}.parquet"
    result = frame.copy()
    for column in OPERATIONAL_GEFS_FEATURES:
        result[column] = np.nan
    if not path.exists():
        if year:
            raise ValueError(f"operational GEFS extraction is missing for {year}")
        return result
    extracted = publication_audited_features(pd.read_parquet(path))
    extracted['timestamp'] = pd.to_datetime(extracted.timestamp, utc=True)
    eligible = result.loc[
        result["event_id"].gt(0)
        & result["remaining_hours"].notna()
        & result["particle_primary"].fillna(False)
    ]
    expected_keys = {
        (int(row.event_id), pd.Timestamp(row.timestamp).isoformat())
        for row in eligible.itertuples(index=False)
    }
    observed_keys = {
        (int(row.event_id), pd.Timestamp(row.timestamp).isoformat())
        for row in extracted.itertuples(index=False)
    }
    if not extracted_keys_are_complete(expected_keys, observed_keys):
        raise ValueError(f"operational GEFS extraction is incomplete for {year}")
    return result.drop(columns=list(OPERATIONAL_GEFS_FEATURES)).merge(
        extracted, on=["timestamp", "event_id"], how="left", validate="one_to_one"
    )


def _score(train: pd.DataFrame, test: pd.DataFrame, feature_names: list[str],
           output: Path) -> dict[str, float]:
    model = AFTSurvivalRegressor(seed=42, nthread=1).fit(train[feature_names], train["remaining_hours"])
    prediction = np.asarray(model.predict(test[feature_names]), dtype=float).clip(min=0.0)
    actual = test["remaining_hours"].to_numpy(float)
    records = test[['timestamp', 'event_key', 'remaining_hours']].copy()
    records['predicted_hours'] = prediction
    records.to_parquet(output, index=False)
    return {
        "row_mae": float(np.abs(prediction - actual).mean()),
        "event_equal_mae": event_equal_mae(actual, prediction, test["event_key"]),
        "train_rows": len(train), "test_rows": len(test),
        "train_events": int(train.event_key.nunique()), "test_events": int(test.event_key.nunique()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "operational_gefs_ensemble" / "report.json")
    parser.add_argument('--gefs-dir', type=Path, default=PROCESSED / 'operational_gefs_audited')
    args = parser.parse_args()
    selection_path = args.output.with_name(args.output.stem + '_selection_2023.json')
    records_dir = args.output.parent / (args.output.stem + '_predictions')
    if args.output.exists() or selection_path.exists() or records_dir.exists():
        raise FileExistsError('choose a new output; benchmark evidence is immutable')
    deployed = ROOT / 'artifacts' / 'cnemc_2025_deployment_candidate'
    def hashes(directory):
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(directory.glob('*')) if p.is_file()}
    before = hashes(deployed)
    metadata = json.loads((ROOT / "artifacts" / "cnemc_2025_deployment_candidate" / "model_metadata.json").read_text(encoding="utf-8"))
    base_features = list(metadata["feature_names"])
    sources = {
        "prefix_2014": PROCESSED / "cnemc_chengdu_2014_training_features.parquet",
        "base": PROCESSED / "training_features.parquet",
        "external_2022": PROCESSED / "cnemc_chengdu_2022_training_features.parquet",
        "external_2023": PROCESSED / "cnemc_chengdu_2023_training_features.parquet",
        "external_2024": PROCESSED / "cnemc_chengdu_2024_training_features.parquet",
        "external_2025": PROCESSED / "cnemc_chengdu_2025_training_features.parquet",
    }
    def source_hashes():
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources.values()}
    source_snapshot, gefs_snapshot = source_hashes(), hashes(args.gefs_dir)
    frames = []
    for source, path in sources.items():
        frame = _prepare(pd.read_parquet(path), source)
        year = int(source[-4:]) if source.startswith("external_") else 0
        frames.append(_attach_operational_gefs(frame, year, args.gefs_dir))
    rows = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    rows = add_missingness_indicators(rows, list(OPERATIONAL_GEFS_FEATURES))
    records_dir.mkdir(parents=True)
    (records_dir / 'deployment_before.json').write_text(json.dumps(before, indent=2), encoding='utf-8')
    scores: dict[int, dict[str, dict[str, float]]] = {}
    selected = None
    for year in (2023, 2024, 2025):
        train, test = safe_event_year_split(rows, year)
        scores[year] = {}
        names = list(VARIANTS) if year == 2023 else list(dict.fromkeys(['aft', selected]))
        for name in names:
            fields = VARIANTS[name]
            features = [*base_features, *fields, *[f"{field}_is_missing" for field in fields]]
            scores[year][name] = _score(train, test, features, records_dir / f'{name}_{year}.parquet')
        if year == 2023:
            selected = select_2023_variant({name: v['row_mae'] for name, v in scores[2023].items()})
            selection_path.write_text(json.dumps({'selection_year': 2023, 'selected_variant': selected,
                                                   'scores': scores[2023]}, indent=2), encoding='utf-8')
            frozen_hash = hashlib.sha256(selection_path.read_bytes()).hexdigest()
            print(f'2023 selection frozen: {selected}', flush=True)
        elif hashlib.sha256(selection_path.read_bytes()).hexdigest() != frozen_hash:
            raise RuntimeError('2023 selection changed during confirmation')
    confirmations = {
        year: {
            "row_improved": scores[year][selected]["row_mae"] < scores[year]["aft"]["row_mae"],
            "event_improved": scores[year][selected]["event_equal_mae"] < scores[year]["aft"]["event_equal_mae"],
        }
        for year in (2024, 2025)
    }
    report = {
        "protocol": "2023 GEFS selection followed by event-isolated 2024 and 2025 confirmation",
        'training_policy': 'Expanding-window annual refit from 2014; feature choice, not weights, is frozen.',
        'base_feature_asof_verified': False,
        "selection_year": 2023,
        "scores": {str(year): value for year, value in scores.items()},
        "selected_variant": selected,
        'selection_sha256': frozen_hash,
        'source_data_sha256': source_snapshot,
        'audited_gefs_sha256': gefs_snapshot,
        'availability_policy': 'archive_timestamp_proxy',
        'limits': [('S3 timestamps are a conservative archive-availability proxy, not verified '
                    'historical first-publication/ACL evidence or operational validation.'),
                   '2024 and 2025 were previously inspected in other experiments.',
                   'Unavailable GEFS fields are masked; rows are not selected by forecast availability.',
                   'Historical base meteorology/PBL publication availability is not independently verified.',
                   'Saved predictions support metric reconstruction; experiment model weights are not saved.',
                   'This is full-duration MAE, not the neural pilot 68-hour capped metric.'],
        "confirmations": {str(year): value for year, value in confirmations.items()},
        "decision": {
            "status": "research_candidate_only" if should_accept_for_deployment(confirmations) else "reject",
            'metric_gate_passed': should_accept_for_deployment(confirmations),
            "deployment_artifact_changed": False,
        },
    }
    after = hashes(deployed)
    (records_dir / 'deployment_after.json').write_text(json.dumps(after, indent=2), encoding='utf-8')
    if before != after:
        raise RuntimeError('deployment artifacts changed during benchmark')
    if source_snapshot != source_hashes() or gefs_snapshot != hashes(args.gefs_dir):
        raise RuntimeError('benchmark inputs changed after loading')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False))


if __name__ == "__main__":
    main()
