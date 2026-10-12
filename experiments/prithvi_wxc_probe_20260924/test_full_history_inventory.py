import pandas as pd
from experiments.prithvi_wxc_probe_20260924.full_history_inventory import weather_times, assign_partition, inventory, HERE


def test_weather_inputs_never_use_future_and_cross_day_correctly():
    origin = pd.Timestamp('2020-01-02 03:00', tz='Asia/Shanghai')
    earlier, latest = weather_times(origin)
    assert latest == pd.Timestamp('2020-01-01 18:00', tz='UTC')
    assert earlier == pd.Timestamp('2020-01-01 12:00', tz='UTC')
    assert latest <= origin < latest + pd.Timedelta(hours=6)


def test_old_pilot_hour_keeps_exact_weather_times():
    assert weather_times(pd.Timestamp('2020-01-01 14:00', tz='Asia/Shanghai')) == (
        pd.Timestamp('2020-01-01 00:00', tz='UTC'), pd.Timestamp('2020-01-01 06:00', tz='UTC'))


def test_calendar_partitions_not_random_rows():
    assert [assign_partition(year) for year in [2014,2023,2024,2025,2026]] == [
        'train','train','dev','retrospective_test','recent_audit']


def test_full_real_inventory_preserves_all_hours_and_flags_boundaries():
    report = inventory()
    rows = pd.read_parquet(HERE/'full_history/candidate_hours.parquet')
    assert rows.timestamp.is_unique
    assert len(rows) == report['total']['eligible_hours'] == 8080
    assert rows.timestamp.dt.year.min() == 2014 and rows.timestamp.dt.year.max() == 2026
    assert len(rows[rows.existing_pilot_origin]) == 25
    assert (rows.weather_latest_utc <= rows.timestamp).all()
    assert rows.weather_age_hours.between(0,5).all()
    assert (rows.weather_latest_utc-rows.weather_earlier_utc).eq(pd.Timedelta(hours=6)).all()
    assert rows.cross_partition_block.sum() == 252
    assert (~rows.history_within_partition).sum() == 30
    assert report['status'] == 'inventory_only_not_trained'
