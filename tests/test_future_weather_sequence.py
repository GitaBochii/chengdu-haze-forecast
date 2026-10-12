import pandas as pd
import pytest

from scripts.probe_future_weather_sequence import sequence_candidates


def test_sequence_is_issue_relative_and_complete():
    issue = pd.Timestamp('2023-01-03T09:00Z')
    rows = sequence_candidates(issue)
    assert len(rows) == 24
    assert [r['ahead_hours'] for r in rows] == list(range(3, 73, 3))
    assert [r['lead_hours'] for r in rows] == list(range(12, 82, 3))
    assert {r['run_utc'] for r in rows} == {'2023-01-03T00:00:00+00:00'}
    assert rows[-1]['valid_utc'] == '2023-01-06T09:00:00+00:00'
    assert rows[-1]['url'].endswith('f081')


def test_reject_non_aligned_issue():
    with pytest.raises(ValueError):
        sequence_candidates(pd.Timestamp('2023-01-03T10:00Z'))
