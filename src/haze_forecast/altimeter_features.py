"""Clock-aligned delayed altimeter observations; never interpolate missing hours."""

import pandas as pd

COLUMNS = ['altimeter_lag3_hpa', 'altimeter_change3_hpa', 'altimeter_change24_hpa']


def altimeter_features(hourly, timestamps):
    source = pd.DatetimeIndex(hourly.timestamp)
    target = pd.DatetimeIndex(timestamps)
    if (source.has_duplicates or source.hasnans or target.hasnans
            or source.tz is None or target.tz is None):
        raise ValueError('Require unique source times and timezone-aware nonmissing clocks')
    values = pd.Series(hourly.altimeter_setting_hpa.to_numpy(dtype=float), index=source)
    past = [values.reindex(target-pd.Timedelta(hours=h)).to_numpy() for h in (3, 6, 27)]
    return pd.DataFrame(dict(zip(COLUMNS, [past[0], past[0]-past[1], past[0]-past[2]])))
