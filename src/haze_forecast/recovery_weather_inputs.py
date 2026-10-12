"""Issue-relative three-hour weather inputs with explicit archive-proxy masks."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

WEATHER_FEATURES = ('temperature_c', 'relative_humidity_pct', 'wind_u_ms', 'wind_v_ms')
WEATHER_AHEAD_HOURS = tuple(range(3, 73, 3))


def _utc(value):
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError('weather timestamps must be present and timezone aware')
    return timestamp.tz_convert('UTC')


def build_weather_inputs(frame, origins):
    """Return values/mask [origin, 24, 4]; unavailable cells have value zero.

    available_at_utc is an audited archive timestamp proxy, not proof of first
    publication. Callers must supply only provenance-verified source rows.
    No interpolation is performed and unrequested origins are rejected.
    """
    origins = [_utc(value) for value in origins]
    if not origins or len(set(origins)) != len(origins):
        raise ValueError('origins must be nonempty and unique')
    positions = {value: index for index, value in enumerate(origins)}
    values = np.zeros((len(origins), 24, 4), dtype=np.float64)
    mask = np.zeros_like(values, dtype=bool)
    seen = set()
    for row in frame.itertuples(index=False):
        issue, valid, run, available = [_utc(getattr(row, key)) for key in
                                      ('issue_utc', 'valid_utc', 'run_utc', 'available_at_utc')]
        ahead = (valid - issue).total_seconds() / 3600
        if issue not in positions or ahead not in WEATHER_AHEAD_HOURS:
            raise ValueError('unexpected origin or forecast valid time')
        if run > issue or run > available:
            raise ValueError('invalid run or archive timestamp')
        key = (issue, valid)
        if key in seen:
            raise ValueError('duplicate weather origin/valid key')
        seen.add(key)
        numeric = np.asarray([getattr(row, name) for name in WEATHER_FEATURES], dtype=float)
        observed = np.isfinite(numeric) & (np.abs(numeric) < 1e10) & (available <= issue)
        index = (positions[issue], WEATHER_AHEAD_HOURS.index(ahead))
        values[index] = np.where(observed, numeric, 0.)
        mask[index] = observed
    return values, mask


def _validate_arrays(values, mask):
    if (values.ndim != 3 or values.shape[-1] != 4 or mask.shape != values.shape
            or mask.dtype != bool or not np.isfinite(values[mask]).all()):
        raise ValueError('expected finite observed values and boolean matching [batch, time, 4] mask')


@dataclass(frozen=True)
class WeatherScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values, mask, training_rows):
        """Explicit training-origin selector prevents implicit full-panel fitting."""
        _validate_arrays(values, mask)
        training_rows = np.asarray(training_rows)
        if training_rows.dtype != bool or training_rows.shape != (len(values),) or not training_rows.any():
            raise ValueError('explicit nonempty boolean training selector required')
        selected = mask & training_rows[:, None, None]
        if not selected.any(axis=(0, 1)).all():
            raise ValueError('each weather feature needs observed training values')
        columns = [values[..., i][selected[..., i]] for i in range(4)]
        mean = np.array([column.mean() for column in columns])
        scale = np.array([column.std() for column in columns])
        scale = np.where(scale > 0, scale, 1.)
        return cls(mean, scale)

    def transform(self, values, mask):
        _validate_arrays(values, mask)
        if (self.mean.shape != (4,) or self.scale.shape != (4,)
                or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all()
                or not (self.scale > 0).all()):
            raise ValueError('invalid weather scaler')
        safe = np.where(mask, values, self.mean)
        return np.where(mask, (safe - self.mean) / self.scale, 0.)
