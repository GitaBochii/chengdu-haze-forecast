"""Compare available raw half-hour pairs with every pinned local reference."""
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from scripts.download_merra_averaged_sample import PRODUCTS, compare_averaged
from scripts.import_merra_history import atomic_json
from scripts.download_merra_regional_sample import file_hash


def main():
    base = HERE / 'full_history'
    manifest_path = base / 'dynamic_import/manifest.json'
    pilot_path = HERE / 'pilot_batch_report.json'
    pins = {str(p): file_hash(p) for p in (manifest_path, pilot_path)}
    manifest = json.loads(manifest_path.read_text('utf-8'))
    pilot = json.loads(pilot_path.read_text('utf-8'))
    references = {}
    for row in pilot['rows']:
        for path, digest in row.get('source_sha256', {}).items():
            if path.startswith('merra-2/MERRA2_sfc_'):
                if path in references and references[path] != digest:
                    raise ValueError('Conflicting reference hashes')
                references[path] = digest
    rows, skipped = [], []
    source_hashes = {}
    for relative, digest in sorted(references.items()):
        reference_path = HERE / 'real_sample' / relative
        if file_hash(reference_path) != digest:
            raise ValueError('Reference hash mismatch: ' + relative)
        source_hashes[str(reference_path)] = digest
        with xr.open_dataset(reference_path, engine='h5netcdf') as reference:
            dates = sorted({str(t.astype('datetime64[D]') + np.timedelta64(offset, 'D'))
                            for t in reference.time.values for offset in (-1, 0)})
            for product, names in PRODUCTS.items():
                with ExitStack() as stack:
                    pieces = []
                    for date in dates:
                        key = product + '/' + date
                        if key not in manifest['files']:
                            continue
                        path = base / 'dynamic_import' / product / (date + '.nc4')
                        record = manifest['files'][key]
                        actual = file_hash(path)
                        if actual != record['sha256'] or path.stat().st_size != record['bytes']:
                            raise ValueError('Raw source checksum mismatch: ' + key)
                        source_hashes[str(path)] = actual
                        pieces.append(stack.enter_context(xr.open_dataset(path, engine='h5netcdf')))
                    if not pieces:
                        skipped.append({'reference': relative, 'product': product, 'reason': 'no_raw_days'})
                        continue
                    raw = xr.concat(pieces, dim='time').sortby('time')
                    available = set(raw.time.values.astype('datetime64[m]').astype('int64').tolist())
                    for time in reference.time.values.astype('datetime64[m]'):
                        pair = [time - np.timedelta64(30, 'm'), time + np.timedelta64(30, 'm')]
                        if not all(int(t.astype('int64')) in available for t in pair):
                            skipped.append({'reference': relative, 'product': product,
                                            'time': str(time), 'reason': 'half_hour_pair_absent'})
                            continue
                    # Reuse the tested two-time comparison without duplicating cells.
                    valid = [t for t in reference.time.values.astype('datetime64[m]')
                             if all(int((t + np.timedelta64(m, 'm')).astype('int64')) in available
                                    for m in (-30, 30))]
                    for index in range(0, len(valid) - 1, 2):
                        times = np.array(valid[index:index+2])
                        result = compare_averaged(raw, reference, names, times)
                        rows.append({'reference': relative, 'product': product,
                                     'times': [str(t) for t in times], 'comparison': result})
                    if len(valid) % 2:
                        skipped.append({'reference': relative, 'product': product,
                                        'time': str(valid[-1]), 'reason': 'unpaired_comparison_time'})
    if any(file_hash(Path(p)) != digest for p, digest in {**pins, **source_hashes}.items()):
        raise ValueError('Source changed during audit')
    result = {'audited_at_utc': datetime.now(timezone.utc).isoformat(),
              'reference_files': len(references), 'comparisons': rows, 'skipped': skipped,
              'source_sha256': source_hashes, 'manifest_sha256': pins,
              'compared_cells': sum(v['cells'] for row in rows for v in row['comparison']['variables'].values()),
              'full_history_preprocessing_verified': False,
              'scope': 'Available pinned 2020 references only; no imputation, training or downloads.'}
    atomic_json(base / 'averaged_overlap_audit.json', result)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ('comparisons', 'skipped', 'source_sha256', 'manifest_sha256')}, indent=2))
    print('comparison_groups', len(rows), 'skipped', len(skipped))


if __name__ == '__main__':
    main()
