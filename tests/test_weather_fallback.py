import pandas as pd
import pytest

from scripts.extract_future_weather_pilot import validate_candidate
from scripts.resolve_weather_fallback import fallback_candidates


def test_older_cycle_keeps_valid_times_and_increases_lead():
    issue = pd.Timestamp('2022-01-06T15:00Z')
    first = fallback_candidates(issue, 2)
    last = fallback_candidates(issue, 4)
    assert len(first) == len(last) == 24
    assert [r['valid_utc'] for r in first] == [r['valid_utc'] for r in last]
    assert last[0]['run_utc'] == '2022-01-05T12:00:00+00:00'
    assert last[0]['lead_hours'] == first[0]['lead_hours'] + 12
    assert last[-1]['ahead_hours'] == 72


def test_fallback_bound_is_explicit():
    with pytest.raises(ValueError):
        fallback_candidates(pd.Timestamp('2022-01-06T15:00Z'), 5)


def test_extractor_accepts_audited_four_cycle_fallback():
    row = fallback_candidates(pd.Timestamp('2022-01-06T15:00Z'), 4)[0]
    validate_candidate(row)
    with pytest.raises(ValueError):
        validate_candidate({**row, 'cycles_back': 1})
