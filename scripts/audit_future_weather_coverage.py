"""Audit existing GEFS availability at neural origins without fitting any model."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from haze_forecast.operational_gefs_benchmark import publication_audited_features
from haze_forecast.operational_gefs_dataset import OPERATIONAL_GEFS_FEATURES
from scripts.benchmark_recovery_head import RecoveryWindows, load_experiment
from scripts.benchmark_recovery_selection import ROOT, file_hash
from scripts.benchmark_regional_air_models import hashes, write_json

FIELDS = {lead: [f'gefs_mean_{field}_f{lead:03d}' for field in (
    'temperature_c', 'relative_humidity_pct', 'wind_u_ms', 'wind_v_ms')] for lead in (6, 24)}


def align_coverage(origins, forecast, fields):
    if origins.timestamp.duplicated().any() or forecast.timestamp.duplicated().any():
        raise ValueError('origin timestamps must be unique in each source')
    merged = origins.merge(forecast, on='timestamp', how='left', validate='one_to_one', indicator=True)
    merged['matched'] = merged['_merge'].eq('both')
    for lead, columns in fields.items():
        merged[f'available_{lead}'] = np.isfinite(merged[columns]).all(axis=1)
        merged[f'ahead_{lead}'] = (merged[f'gefs_forecast_valid_utc_f{lead:03d}']
                                  - merged.timestamp).dt.total_seconds() / 3600
    return merged.drop(columns='_merge')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    args.data, args.history, args.horizon, args.validation_year = (
        'data/processed/regional_air_models_full_history', 24, 72, 2024)
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    directory = ROOT / 'data/processed/operational_gefs_audited'
    before, deployed = hashes(directory), hashes(ROOT / 'artifacts/cnemc_2025_deployment_candidate')
    neural_before = hashes(ROOT / args.data)
    old_report = json.loads((ROOT / 'artifacts/operational_gefs_ensemble/report.json').read_text('utf-8'))
    assert before == old_report['audited_gefs_sha256']
    frames, raw_summary = [], {}
    for year in (2022, 2023, 2024, 2025):
        frame = pd.read_parquet(directory / f'operational_gefs_event_features_{year}.parquet')
        audited = publication_audited_features(frame)
        assert len(frame) == len(audited)
        frame['timestamp'] = pd.to_datetime(frame.timestamp, utc=True)
        selected = frame[['timestamp', 'aqi', 'gefs_run_utc',
                          'gefs_forecast_valid_utc_f006', 'gefs_forecast_valid_utc_f024',
                          *[field for columns in FIELDS.values() for field in columns]]].rename(
                              columns={'aqi': 'gefs_source_aqi'})
        frames.append(selected)
        raw_summary[year] = {'rows': len(frame),
                             'retained_field_values': int(np.isfinite(frame[list(OPERATIONAL_GEFS_FEATURES)]).sum().sum()),
                             'total_field_values': len(frame) * 18,
                             'four_means_available': {lead: int(np.isfinite(frame[cols]).all(axis=1).sum())
                                                      for lead, cols in FIELDS.items()}}
    forecast = pd.concat(frames, ignore_index=True)
    panel, city, _, eligible, partitions = load_experiment(args)
    rows = []
    for year, origins in partitions.items():
        dataset = RecoveryWindows(panel, origins, 24, 72, city.aqi.to_numpy(), eligible)
        rows.append(pd.DataFrame({'timestamp': panel['timestamps'][origins].tz_convert('UTC'),
                                  'year': year, 'neural_aqi': city.aqi.to_numpy()[origins],
                                  'eligible': dataset.eligible,
                                  'informative': dataset.eligible & dataset.risk_mask.any(axis=1)}))
    joined = align_coverage(pd.concat(rows, ignore_index=True), forecast, FIELDS)
    joined['aqi_conflict'] = (joined.matched & joined.neural_aqi.notna() & joined.gefs_source_aqi.notna()
                              & ~np.isclose(joined.neural_aqi, joined.gefs_source_aqi, equal_nan=True))
    joined.to_parquet(output / 'origin_coverage.parquet', index=False)
    summary = []
    for year, group in joined.groupby('year'):
        for population, subset in (('all_origins', group), ('informative_survival', group.loc[group.informative])):
            summary.append({'year': int(year), 'population': population, 'origins': len(subset),
                            'matched_existing_event_rows': int(subset.matched.sum()),
                            'available_mean_f006': int(subset.available_6.sum()),
                            'available_mean_f024': int(subset.available_24.sum()),
                            'available_both': int((subset.available_6 & subset.available_24).sum()),
                            'aqi_conflicts': int(subset.aqi_conflict.sum())})
    pd.DataFrame(summary).to_csv(output / 'coverage_by_year.csv', index=False)
    full_offsets = {lead: ((forecast[f'gefs_forecast_valid_utc_f{lead:03d}'] - forecast.timestamp)
                           .dt.total_seconds() / 3600).value_counts().sort_index().to_dict() for lead in (6, 24)}
    origin_offsets = {lead: joined.loc[joined.matched, f'ahead_{lead}'].value_counts().sort_index().to_dict()
                      for lead in (6, 24)}
    report = {'configuration': vars(args), 'audited_gefs_sha256': before, 'neural_inputs_sha256': neural_before,
              'deployment_sha256': deployed, 'old_gefs_report_sha256': file_hash(
                  ROOT / 'artifacts/operational_gefs_ensemble/report.json'),
              'existing_gefs': raw_summary, 'coverage': summary,
              'all_event_rows_ahead_hours': full_offsets, 'matched_neural_origins_ahead_hours': origin_offsets,
              'join_key': 'exact_timestamp_only_not_event_id_across_different_label_tables',
              'limitations': ['2024_cutoff_panel_for_inventory_not_new_performance_evaluation',
                              'availability_is_archive_proxy_not_verified_first_publication',
                              'existing_gefs_is_completed_particle_event_subset_not_all_causal_origins',
                              'two_leads_not_a_72_hour_weather_sequence',
                              'unmatched_rows_preserved_no_nearest_neighbor_or_forward_fill'],
              'inputs_unchanged': hashes(directory) == before and hashes(ROOT / args.data) == neural_before,
              'deployment_unchanged': hashes(ROOT / 'artifacts/cnemc_2025_deployment_candidate') == deployed}
    assert report['inputs_unchanged'] and report['deployment_unchanged']
    write_json(output / 'report.json', report)
    print(json.dumps({'existing_gefs': raw_summary, 'coverage': summary,
                      'matched_neural_origins_ahead_hours': origin_offsets}, indent=2))


if __name__ == '__main__':
    main()
