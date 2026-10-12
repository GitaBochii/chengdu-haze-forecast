"""Lazy, support-aligned offline batches; no implicit training authorization."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

from raw_dynamic_adapter import HERE
from raw_inference_batch import auxiliary, pack, climate_key
from prepare_full_dynamic_cache import boundary_flag, cache_name, validate_cube
from import_climatology_regions import cache_valid, REVISION
from scripts.import_merra_static import validate as validate_static
from scripts.download_merra_regional_sample import file_hash
from audit_full_import import parquet, derive_requirements


def describe_origin(value):
    origin = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if origin.tzinfo is None:
        raise ValueError('Timezone-aware origin required')
    origin = origin.astimezone(timezone.utc)
    if origin.minute or origin.second or origin.microsecond:
        raise ValueError('Hourly pollution origin required')
    shifted = origin - timedelta(hours=1)
    latest = shifted.replace(hour=shifted.hour // 6 * 6)
    earlier = latest - timedelta(hours=6)
    pair = [earlier.isoformat(), latest.isoformat()]
    return {'origin': origin.isoformat(), 'earlier': pair[0], 'latest': pair[1],
            'support_end': (latest + timedelta(hours=1)).isoformat(),
            'dynamic_files': [cache_name(t) for t in pair],
            'climate_key': list(climate_key(latest)),
            'year_start_boundary_uncertain': any(boundary_flag(t) for t in pair)}


def validate_origin_mapping(origins, candidates):
    expected = {}
    for row in candidates:
        metadata = describe_origin(row['timestamp'])
        key = metadata['origin']
        if key in expected:
            raise ValueError('Duplicate original candidate')
        expected[key] = (row['partition'], metadata['dynamic_files'], metadata['year_start_boundary_uncertain'])
    if set(origins) != set(expected):
        raise ValueError('Cached candidate set differs from original plan')
    for key, values in expected.items():
        row = origins[key]
        if (row['partition'], row['files'], row['year_start_boundary_uncertain']) != values:
            raise ValueError('Cached partition or dependencies differ from original plan')


class FullHistoryBatchLoader:
    def __init__(self):
        self.base = HERE / 'full_history'
        self.hashes = {}
        self.config_sha = self.verify(HERE / 'vendor/config.yaml')
        self.config = yaml.safe_load((HERE / 'vendor/config.yaml').read_text())['data']
        manifest = self.read(self.base / 'dynamic_cubes_centered_v1/manifest.json')
        if manifest['status'] != 'dynamic_cache_complete_not_training_ready' or manifest['policy'] != 'centered_float32_no_clamp_v1':
            raise ValueError('Complete centered dynamic cache required')
        for path, digest in manifest['source_sha256'].items():
            self.verify(Path(path), digest)
        self.cubes = {row['path']: row for row in manifest['files']}
        self.origins = {describe_origin(row['origin'])['origin']: row for row in manifest['candidates']}
        if len(self.cubes) != len(manifest['files']) or len(self.origins) != len(manifest['candidates']):
            raise ValueError('Duplicate cache identifiers')
        source = self.base / 'support_aligned_plan/candidate_hours.parquet'
        self.verify(source)
        candidates = parquet(source)
        derive_requirements(candidates)
        validate_origin_mapping(self.origins, candidates)
        inventory = self.read(self.base / 'climatology_inventory.json')
        if inventory['revision'] != REVISION:
            raise ValueError('Unexpected climatology revision')
        self.climate_entries = {row['path']: row for row in inventory['files']}
        report = self.read(self.base / 'static_import/report.json')
        path = self.base / 'static_import/M2C0NXCTM.nc4'
        self.verify(path, report['products']['M2C0NXCTM']['sha256'])
        with xr.open_dataset(path, engine='h5netcdf') as dataset:
            if not validate_static(dataset, 'M2C0NXCTM')['all_months_equal']:
                raise ValueError('Static monthly inconsistency')
            self.constants = {name: dataset[name].isel(time=0).transpose('lat', 'lon').values.copy()
                              for name in self.config['static_surface_vars']}

    def verify(self, path, expected=None):
        digest = file_hash(path)
        if expected is not None and digest != expected:
            raise ValueError('Checksum mismatch: ' + str(path))
        if str(path) in self.hashes and self.hashes[str(path)] != digest:
            raise ValueError('Source changed during session: ' + str(path))
        self.hashes[str(path)] = digest
        return digest

    def read(self, path):
        self.verify(path)
        return json.loads(path.read_text('utf-8'))

    def load(self, origin):
        metadata = describe_origin(origin)
        row = self.origins.get(metadata['origin'])
        if row is None:
            raise ValueError('Origin outside frozen candidate plan')
        if row['files'] != metadata['dynamic_files'] or row['year_start_boundary_uncertain'] != metadata['year_start_boundary_uncertain']:
            raise ValueError('Candidate dependency mapping mismatch')
        arrays = []
        for time, name in zip((metadata['earlier'], metadata['latest']), metadata['dynamic_files']):
            entry = self.cubes[name]
            if datetime.fromisoformat(entry['time']) != datetime.fromisoformat(time):
                raise ValueError('Dynamic time label mismatch')
            path = self.base / 'dynamic_cubes_centered_v1' / name
            self.verify(path, entry['sha256'])
            with np.load(path, allow_pickle=False) as values:
                if values.files != ['x']:
                    raise ValueError('Unexpected cube fields')
                validate_cube(values['x'])
                arrays.append(values['x'].copy())
        day, hour = metadata['climate_key']
        climate = {}
        for kind in ('surface', 'vertical'):
            relative = f'climatology/climate_{kind}_doy{day:03}_hour{hour:02}.nc'
            path = self.base / 'climatology_regions' / (Path(relative).stem + '.npz')
            if not cache_valid(path, self.climate_entries[relative], self.config_sha,
                               self.config['surface_vars' if kind == 'surface' else 'vertical_vars'],
                               None if kind == 'surface' else self.config['levels']):
                raise ValueError('Invalid climate artifact')
            self.verify(path)
            self.verify(path.with_suffix('.json'))
            with np.load(path, allow_pickle=False) as values:
                climate[kind] = {name: values[name].copy() for name in values.files}
        static, clim = auxiliary(datetime.fromisoformat(metadata['latest']), self.constants,
                                 climate['surface'], climate['vertical'], self.config)
        batch = pack(np.stack(arrays), static, clim)
        metadata.update(partition=row['partition'], training_allowed=False,
                        operational_availability_verified=False)
        return batch, metadata

    def verify_unchanged(self):
        for path, digest in self.hashes.items():
            if file_hash(Path(path)) != digest:
                raise ValueError('Input changed during batch validation: ' + path)
