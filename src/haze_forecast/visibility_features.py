"""Saturated visibility with the same frozen causal clock lookup as pressure."""

from haze_forecast.altimeter_features import altimeter_features

COLUMNS = ['visibility_lag3_m', 'visibility_change3_m', 'visibility_change24_m']


def visibility_features(hourly, timestamps):
    temporary = hourly[['timestamp']].copy()
    temporary['altimeter_setting_hpa'] = hourly.visibility_m.clip(upper=5000)
    result = altimeter_features(temporary, timestamps)
    result.columns = COLUMNS
    return result
