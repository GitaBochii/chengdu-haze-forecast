"""Compare raw-source batches with frozen pilot batches; never train or deploy."""
from contextlib import ExitStack
from datetime import datetime, timedelta
import json
from pathlib import Path
import subprocess

import numpy as np
import xarray as xr
import yaml

from raw_dynamic_adapter import assemble, HERE, PRODUCTS
from raw_inference_batch import auxiliary, pack, climate_key, utc, gate_status
from import_climatology_regions import cache_valid
from scripts.import_merra_history import atomic_json
from scripts.import_merra_static import validate as validate_static
from scripts.download_merra_regional_sample import file_hash


def main():
    base = HERE / 'full_history'
    out = base / 'raw_inference_gate'
    out.mkdir(exist_ok=True)
    hashes = {}

    def verify(path, expected=None):
        digest = file_hash(path)
        if expected is not None and digest != expected:
            raise ValueError('Checksum mismatch: ' + str(path))
        if str(path) in hashes and hashes[str(path)] != digest:
            raise ValueError('Source changed: ' + str(path))
        hashes[str(path)] = digest
        return digest

    def read(path):
        verify(path)
        return json.loads(path.read_text('utf-8'))

    pilot = read(HERE / 'pilot_batch_report.json')
    manifest = read(base / 'dynamic_import/manifest.json')
    inventory = {r['path']: r for r in read(base / 'climatology_inventory.json')['files']}
    config_sha = verify(HERE / 'vendor/config.yaml')
    config = yaml.safe_load((HERE / 'vendor/config.yaml').read_text())['data']
    for name in ('raw_dynamic_adapter.py', 'raw_inference_batch.py', 'audit_raw_inference_batch.py'):
        verify(HERE / name)
    static_report = read(base / 'static_import/report.json')
    static_path = base / 'static_import/M2C0NXCTM.nc4'
    verify(static_path, static_report['products']['M2C0NXCTM']['sha256'])
    with xr.open_dataset(static_path, engine='h5netcdf') as dataset:
        if not validate_static(dataset, 'M2C0NXCTM')['all_months_equal']:
            raise ValueError('Static fields are not constant')
        constants = {name: dataset[name].isel(time=0).transpose('lat', 'lon').values.copy()
                     for name in config['static_surface_vars']}
    rows = []
    for candidate in pilot['rows']:
        row = {'date': candidate['date'], 'status': 'pending'}
        latest = utc(datetime.fromisoformat(candidate['origin_utc']))
        times = [np.datetime64((latest - timedelta(hours=h)).replace(tzinfo=None), 'ns') for h in (6, 0)]
        reference = HERE / candidate['batch_path']
        verify(reference, candidate['batch_sha256'])
        with ExitStack() as stack:
            products, missing = {}, []
            for product in PRODUCTS:
                required = [t + np.timedelta64(m, 'm') for t in times for m in ((-30, 30) if product.startswith('M2T') else (0,))]
                pieces = []
                for day in sorted({str(t.astype('datetime64[D]')) for t in required}):
                    record = manifest['files'].get(product + '/' + day)
                    if record is None:
                        continue
                    path = base / 'dynamic_import' / product / (day + '.nc4')
                    verify(path, record['sha256'])
                    pieces.append(stack.enter_context(xr.open_dataset(path, engine='h5netcdf')))
                if pieces:
                    products[product] = xr.concat(pieces, dim='time').sortby('time')
                if product not in products or not all(t in products[product].time.values for t in required):
                    missing.append(product)
            day, hour = climate_key(latest)
            climate = {}
            for kind in ('surface', 'vertical'):
                relative = f'climatology/climate_{kind}_doy{day:03}_hour{hour:02}.nc'
                path = base / 'climatology_regions' / (Path(relative).stem + '.npz')
                if relative not in inventory or not path.exists():
                    missing.append(relative)
                    continue
                if not cache_valid(path, inventory[relative], config_sha,
                                   config['surface_vars' if kind == 'surface' else 'vertical_vars'],
                                   None if kind == 'surface' else config['levels']):
                    raise ValueError('Invalid climatology cache')
                verify(path)
                verify(path.with_suffix('.json'))
                with np.load(path, allow_pickle=False) as values:
                    climate[kind] = {key: values[key].copy() for key in values.files}
            if missing:
                row.update(status='missing_overlap', missing=missing)
            else:
                dynamic = np.stack([assemble(products, t, config) for t in times])
                static, clim = auxiliary(latest, constants, climate['surface'], climate['vertical'], config)
                batch = pack(dynamic, static, clim)
                path = out / (candidate['date'] + '.npz')
                temp = path.with_suffix('.npz.partial')
                with temp.open('wb') as stream:
                    np.savez_compressed(stream, **batch)
                temp.replace(path)
                row.update(status='assembled', batch_path=str(path), batch_sha256=file_hash(path),
                           reference_path=str(reference), reference_sha256=candidate['batch_sha256'])
        rows.append(row)
    if any(file_hash(Path(path)) != digest for path, digest in hashes.items()):
        raise ValueError('Source changed during assembly')
    report_path = out / 'report.json'
    atomic_json(report_path, {'rows': rows, 'source_sha256': hashes, 'training_performed': False,
        'scope': 'Frozen pilot-time compatibility only; pilot origins are not the support-aligned deployment mapping.'})
    # Keep the scientific and Torch environments isolated; exchange only arrays.
    code = '''import json,sys,hashlib,numpy as np,torch
from pathlib import Path
p=Path(sys.argv[1]); report=json.loads(p.read_text('utf-8'))
for row in report['rows']:
 if row['status']!='assembled': continue
 for key in ('batch','reference'):
  if hashlib.sha256(Path(row[key+'_path']).read_bytes()).hexdigest()!=row[key+'_sha256']: raise ValueError('Artifact changed')
 reference=torch.load(row['reference_path'],weights_only=True,map_location='cpu')['batch']
 with np.load(row['batch_path'],allow_pickle=False) as batch:
  if set(batch.files)!=set(reference): raise ValueError('Batch key mismatch')
  checks={}
  for key in batch.files:
   a=batch[key]; b=reference[key].numpy()
   if a.shape!=b.shape or a.dtype!=b.dtype: raise ValueError('Shape or dtype mismatch')
   checks[key]={'equal':bool(np.array_equal(a,b)),'cells':int(a.size),'equal_cells':int((a==b).sum()),'max_abs_difference':float(np.abs(a.astype('float64')-b).max())}
 row['checks']=checks; row['status']='exact' if all(v['equal'] for v in checks.values()) else 'different'
report['counts']={s:sum(r['status']==s for r in report['rows']) for s in ('exact','different','missing_overlap')}
report['full_history_preprocessing_verified']=False
temp=p.with_suffix('.tmp'); temp.write_text(json.dumps(report,indent=2,allow_nan=False),'utf-8'); temp.replace(p)
print(json.dumps(report['counts']))
'''
    subprocess.run([str(HERE.parents[1] / '.venv_air_models/Scripts/python.exe'), '-X', 'utf8', '-c', code, str(report_path)], check=True)
    result = json.loads(report_path.read_text('utf-8'))
    result['pilot_contract_pass'] = gate_status(result['rows'])
    result['compared_batches'] = sum(row['status'] in ('exact', 'different') for row in result['rows'])
    atomic_json(report_path, result)
    if not result['pilot_contract_pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
