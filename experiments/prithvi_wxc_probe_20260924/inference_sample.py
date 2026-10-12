"""Official-loader inference inputs without reading actual target weather."""
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / 'vendor'))
from PrithviWxC.dataloaders.merra2 import preproc


def load_inference_sample(dataset, spec):
    """Read only spec.inputs; target climatology is allowed, target weather is not.

    `spec` must follow official SampleSpec conventions (UTC-naive timestamps).
    Returned zero target placeholders exist only to satisfy official preproc.
    """
    if (len(spec.inputs) != 2 or not spec.inputs[0] < spec.inputs[1]
            or list(spec.stat_times) != [spec.inputs[-1]]):
        raise ValueError('two ascending inputs and origin-only static time required')
    if any(time.tzinfo is not None for time in spec.inputs):
        raise ValueError('official reader requires UTC-naive SampleSpec timestamps')
    if any(value != 0 for value in dataset._roll_longitudes):
        raise ValueError('random longitude rolling is prohibited for fixed regional inference')
    if not dataset._require_clim:
        raise ValueError('this fixed pretrained configuration requires climatology')
    observed = []
    for time in spec.inputs:
        files = dataset.data_file_surface(time), dataset.data_file_vertical(time)
        observed.append(dataset._read_data(files, time))
    surface = np.stack([value['surf'] for value in observed], axis=1)
    vertical = np.stack([value['vert'] for value in observed], axis=2)
    origin = spec.stat_times[0]
    static = dataset._read_static_data(dataset.data_file_surface(origin), origin.dayofyear, origin.hour)
    day, hour = spec.climatology_info
    climate = dataset._read_climate((
        dataset.data_file_surface_climate(dayofyear=day, hourofday=hour),
        dataset.data_file_vertical_climate(dayofyear=day, hourofday=hour)))
    sample = {'sur_vals': surface, 'ulv_vals': vertical,
              'sur_tars': np.zeros_like(surface[:, :1]),
              'ulv_tars': np.zeros_like(vertical[:, :, :1]),
              'sur_static': static, 'sur_climate': climate['surf'], 'ulv_climate': climate['vert']}
    sample = dataset._to_torch(sample, dtype=dataset.dtype)
    sample['input_time'] = spec.input_time
    sample['lead_time'] = spec.lead_time
    return sample


def load_inference_batch(dataset, spec, latitude_slice, longitude_slice):
    """Prepare a single regional batch with no `y` key or future truth access."""
    for value in (latitude_slice, longitude_slice):
        if (not isinstance(value, slice) or value.start is None or value.stop is None
                or value.start < 0 or value.stop <= value.start or value.step not in (None, 1)):
            raise ValueError('explicit increasing unit-step regional slices required')
    sample = load_inference_sample(dataset, spec)
    shape = sample['sur_vals'].shape[-2:]
    if latitude_slice.stop > shape[0] or longitude_slice.stop > shape[1]:
        raise ValueError('regional slice exceeds source grid')
    cropped = {key: value[..., latitude_slice, longitude_slice].clone()
               if isinstance(value, torch.Tensor) else value for key, value in sample.items()}
    batch = preproc([cropped], {'level': (0, 0), 'lat': (0, 0), 'lon': (0, 0)})
    del batch['y']
    if not all(torch.isfinite(value).all() for value in batch.values()):
        raise ValueError('nonfinite regional inference inputs')
    return batch
