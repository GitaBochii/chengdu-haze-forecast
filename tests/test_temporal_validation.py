import pandas as pd

from haze_forecast.temporal_validation import event_isolated_year_split


def test_year_split_keeps_a_cross_year_event_out_of_the_training_rows():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2017-12-30 00:00",
                    "2018-01-01 00:00",
                    "2017-12-20 00:00",
                    "2017-12-21 00:00",
                    "2018-02-01 00:00",
                ]
            ),
            "event_key": ["cross", "cross", "finished", "finished", "new"],
            "remaining_hours": [2.0, 1.0, 2.0, 1.0, 1.0],
        }
    )

    train, test = event_isolated_year_split(frame, test_year=2018)

    assert set(train["event_key"]) == {"finished"}
    assert set(test["event_key"]) == {"cross", "new"}
    assert test["timestamp"].dt.year.eq(2018).all()
    assert not set(train["event_key"]).intersection(test["event_key"])
