"""Record the verified ISD coverage break and prospective GHCNh replacement."""

import hashlib
import json
from pathlib import Path

import pandas as pd


def main():
    out = Path('artifacts/task01_recovery_head/weather_source_retirement_v1')
    if out.exists():
        raise FileExistsError(out)
    paths = {
        'old': Path('data/raw/noaa_isd/2025/56294099999.csv'),
        'refresh': Path('data/raw/noaa_isd_refresh_20260906/2025/56294099999.csv'),
        'ghcnh': Path('data/raw/ghcnh_discovery_20260906/GHCNh_CHI0000ZUUU_2025.parquet'),
        'stations': Path('data/raw/ghcnh_discovery_20260906/stations.txt'),
        'documentation': Path('data/raw/ghcnh_discovery_20260906/documentation.pdf'),
        'weather': Path('data/interim/cnemc_chengdu_2024_2025_weather.parquet'),
        'predictions': Path('artifacts/task01_recovery_head/observed_hazard_crossyear_v1/2025_predictions.parquet'),
    }
    old, refreshed = pd.read_csv(paths['old']), pd.read_csv(paths['refresh'])
    new = pd.read_parquet(paths['ghcnh'])
    weather = pd.read_parquet(paths['weather'])
    pred = pd.read_parquet(paths['predictions'])
    hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}
    assert old.DATE.max() == refreshed.DATE.max()
    assert new.DATE.is_unique
    exact = pred.loc[pred.exact_hours.notna()].copy()
    exact['delta_error'] = (exact.hazard - exact.exact_hours).abs() - (exact.legacy_training - exact.exact_hours).abs()
    blocks = exact.groupby('continuity_block').agg(rows=('timestamp', 'size'),
        start=('timestamp', 'min'), delta_mae=('delta_error', 'mean'),
        truth_mean=('exact_hours', 'mean'), hazard_mean=('hazard', 'mean'))
    result = {
        'status': 'verified_input_coverage_break_replacement_not_yet_integrated',
        'paths': {k: str(v) for k, v in paths.items()}, 'sha256': hashes,
        'old_raw_rows': len(old), 'old_last_utc': old.DATE.max(),
        'refresh_rows': len(refreshed), 'refresh_last_utc': refreshed.DATE.max(),
        'refresh_identical_bytes': hashes['old'] == hashes['refresh'],
        'ghcnh_rows': len(new), 'ghcnh_last_utc': new.DATE.max(),
        'station_match': {'old_name': old.NAME.iloc[0], 'new_name': new.Station_name.iloc[0],
            'old_lat': float(old.LATITUDE.iloc[0]), 'new_lat': float(new.LATITUDE.iloc[0]),
            'old_lon': float(old.LONGITUDE.iloc[0]), 'new_lon': float(new.LONGITUDE.iloc[0])},
        'monthly_weather': weather.loc[weather.timestamp.dt.year.eq(2025)].groupby(
            weather.timestamp.dt.strftime('%Y-%m')).temperature_c.agg(['count', 'size']).to_dict('index'),
        'degraded_exact_blocks': int(blocks.delta_mae.gt(0).sum()),
        'block_diagnostics': json.loads(blocks.to_json(orient='index', date_format='iso')),
        'sources': ['https://www.nesdis.noaa.gov/news/service-location-change-integrated-surface-data-global-hourly',
                    'https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly'],
        'limitations': ['weather gap not proven sole cause of error', 'GHCNh QC and units need explicit adapter',
                       'archive revisions do not prove real-time availability', 'no model accuracy claim'],
    }
    out.mkdir(parents=True)
    (out / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['status', 'old_last_utc', 'ghcnh_last_utc', 'refresh_identical_bytes']}, indent=2))


if __name__ == '__main__':
    main()
