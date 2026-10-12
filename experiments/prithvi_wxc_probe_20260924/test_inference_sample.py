"""Prove target weather is never read and real inputs retain official parity."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'vendor'))
from PrithviWxC.dataloaders.merra2 import Merra2Dataset, SampleSpec
from experiments.prithvi_wxc_probe_20260924.inference_sample import load_inference_batch, load_inference_sample


class RecordingLoader:
    dtype = torch.float32
    _roll_longitudes = [0]
    _require_clim = True

    def __init__(self, spec):
        self.spec, self.read_times, self.static_times = spec, [], []

    def data_file_surface(self, time): return 'surface'
    def data_file_vertical(self, time): return 'vertical'
    def data_file_surface_climate(self, **kwargs): return ('surface_climate', kwargs)
    def data_file_vertical_climate(self, **kwargs): return ('vertical_climate', kwargs)

    def _read_data(self, files, time):
        if time > self.spec.inputs[-1]:
            raise AssertionError('future actual weather requested')
        self.read_times.append(time)
        return {'surf': np.ones((2, 4, 4)), 'vert': np.ones((2, 3, 4, 4))}

    def _read_static_data(self, file, doy, hour):
        self.static_times.append((doy, hour))
        return np.zeros((10, 4, 4))

    def _read_climate(self, paths):
        self.climate_paths = paths
        return {'surf': np.ones((2, 4, 4)), 'vert': np.ones((2, 3, 4, 4))}

    def _to_torch(self, data, dtype):
        return {key: torch.as_tensor(value, dtype=dtype) for key, value in data.items()}


class InferenceTests(unittest.TestCase):
    def test_reads_only_inputs(self):
        spec = SampleSpec.get(pd.Timestamp('2020-01-01T06:00:00'), 6, 6)
        loader = RecordingLoader(spec)
        sample = load_inference_sample(loader, spec)
        self.assertEqual(loader.read_times, list(spec.inputs))
        self.assertEqual(loader.static_times, [(1, 6)])
        self.assertEqual(loader.climate_paths[0][1], {'dayofyear': 1, 'hourofday': 12})
        self.assertEqual(sample['sur_tars'].count_nonzero().item(), 0)
        self.assertEqual(sample['ulv_tars'].count_nonzero().item(), 0)

    def test_future_static_spec_rejected(self):
        spec = SampleSpec.get(pd.Timestamp('2020-01-01T06:00:00'), 6, 6)
        loader = RecordingLoader(spec)
        spec.stat_times = [spec.target]
        with self.assertRaises(ValueError):
            load_inference_sample(loader, spec)
        self.assertFalse(loader.read_times)

    def test_real_january_parity(self):
        cfg = yaml.safe_load((HERE / 'vendor/config.yaml').read_text())['data']
        data = HERE / 'real_sample'
        dataset = Merra2Dataset(
            time_range=('2020-01-01T00:00:00', '2020-01-01T23:59:59'),
            lead_times=[6], input_times=[-6], data_path_surface=data / 'merra-2',
            data_path_vertical=data / 'merra-2', climatology_path_surface=data / 'climatology',
            climatology_path_vertical=data / 'climatology', surface_vars=cfg['surface_vars'],
            static_surface_vars=cfg['static_surface_vars'], vertical_vars=cfg['vertical_vars'],
            levels=cfg['levels'], positional_encoding='fourier')
        spec = SampleSpec.get(pd.Timestamp('2020-01-01T06:00:00'), 6, 6)
        with np.load(HERE / 'regional_raw_sample.npz') as raw:
            iy = int(np.where(dataset.lats == raw['lat'][0])[0][0])
            ix = int(np.where(dataset.lons == raw['lon'][0])[0][0])
        original_reader = dataset._read_data
        observed_times = []
        def guarded_reader(files, time):
            self.assertLessEqual(time, spec.inputs[-1])
            observed_times.append(time)
            return original_reader(files, time)
        dataset._read_data = guarded_reader
        batch = load_inference_batch(dataset, spec, slice(iy, iy+30), slice(ix, ix+32))
        expected = torch.load(HERE / 'official_batch.pt', weights_only=True)['batch']
        self.assertEqual(observed_times, list(spec.inputs))
        self.assertNotIn('y', batch)
        self.assertEqual(set(batch), {'x', 'static', 'climate', 'input_time', 'lead_time'})
        for key in batch:
            torch.testing.assert_close(batch[key], expected[key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
