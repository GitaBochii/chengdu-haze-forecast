import pandas as pd

from scripts.probe_future_weather_cycles import cycle_candidate


def test_fallback_keeps_valid_time_and_adjusts_lead():
    run = pd.Timestamp('2023-01-01T06:00:00Z')
    issue = pd.Timestamp('2023-01-01T09:00:00Z')
    for lead in (6, 24):
        current = cycle_candidate(issue, run, lead, 0)
        prior = cycle_candidate(issue, run, lead, 1)
        assert current[1] == prior[1]
        assert prior[2] == lead + 6
        assert prior[0] == pd.Timestamp('2023-01-01T00:00:00Z')
        assert prior[3].endswith(f'f{lead+6:03d}')
