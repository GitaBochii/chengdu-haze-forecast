import pandas as pd
import pytest
from experiments.prithvi_wxc_probe_20260924 import weather_support_plan as plan


def rows(stamps):
    times=pd.Series(pd.to_datetime(stamps,utc=True))
    return pd.DataFrame({'timestamp':times,'partition':'train',
                         'weather_latest_utc':times.dt.floor('6h')})


def test_completed_interval_not_only_nominal_time_controls_alignment():
    original=rows(['2020-01-01T06:00Z','2020-01-01T07:00Z'])
    aligned,required=plan.align(original)
    assert aligned.support_latest_utc.tolist()==list(pd.to_datetime(
        ['2020-01-01T00:00Z','2020-01-01T06:00Z'],utc=True))
    assert aligned.support_alignment_changed.tolist()==[True,False]
    assert (aligned.support_end_utc<=aligned.timestamp).all()
    pd.testing.assert_frame_equal(aligned[original.columns],original)
    assert (required.source_right_support_end_utc==required.weather_time_utc+pd.Timedelta(hours=1)).all()


def test_midnight_sources_include_previous_day():
    aligned,required=plan.align(rows(['2020-01-01T01:00Z']))
    current=required.loc[required.weather_time_utc==pd.Timestamp('2020-01-01T00:00Z')].iloc[0]
    assert current.source_left_utc==pd.Timestamp('2019-12-31T23:30Z')
    assert current.source_right_utc==pd.Timestamp('2020-01-01T00:30Z')
    assert aligned.support_earlier_utc.iloc[0]==pd.Timestamp('2019-12-31T18:00Z')


def test_naive_or_duplicate_origins_rejected():
    naive=pd.DataFrame({'timestamp':pd.to_datetime(['2020-01-01']),
                        'weather_latest_utc':pd.to_datetime(['2020-01-01'])})
    with pytest.raises(ValueError): plan.align(naive)
    with pytest.raises(ValueError): plan.align(rows(['2020-01-01T00:00Z']*2))
