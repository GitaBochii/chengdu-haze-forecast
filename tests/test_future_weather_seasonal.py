import pandas as pd

from scripts.probe_future_weather_seasonal import select_seasonal_origins


def test_selection_excludes_test_year_and_retains_empty_quarter():
    frame = pd.DataFrame({'timestamp': pd.to_datetime([
        '2022-01-02T03:00Z', '2022-01-01T03:00Z', '2022-04-01T03:00Z',
        '2023-01-01T03:00Z', '2025-01-01T03:00Z']),
        'eligible': [True, True, False, True, True]})
    rows = select_seasonal_origins(frame)
    assert len(rows) == 8
    assert rows[0]['issue_utc'] == '2022-01-01T03:00:00+00:00'
    assert rows[1]['issue_utc'] is None
    assert rows[4]['issue_utc'] == '2023-01-01T03:00:00+00:00'
    assert {r['year'] for r in rows} == {2022, 2023}
