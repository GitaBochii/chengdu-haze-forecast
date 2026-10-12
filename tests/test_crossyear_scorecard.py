import pandas as pd
import pytest

from scripts.evaluate_crossyear_scorecard import canonical_cohort


def frames():
    labels = pd.DataFrame({
        'timestamp': pd.date_range('2024-02-01', periods=2, freq='h', tz='Asia/Shanghai'),
        'continuity_block': [1, 1], 'lower_bound_hours': [2., 1.],
        'upper_bound_hours': [2., 1.], 'exact_hours': [2., 1.],
        'label_available_at': pd.to_datetime(['2024-02-01 07:00'] * 2).tz_localize('Asia/Shanghai'),
    })
    return labels, labels.copy()


def test_aligns_labels_by_timestamp_and_replaces_stale_maturity():
    labels, saved = frames()
    saved['label_available_at'] = saved.timestamp
    canonical, stale = canonical_cohort(labels.iloc[::-1], saved, 2024)
    pd.testing.assert_frame_equal(canonical, labels)
    assert stale == 2


def test_rejects_duplicate_or_absent_origin():
    labels, saved = frames()
    with pytest.raises(ValueError):
        canonical_cohort(pd.concat([labels, labels]), saved, 2024)
    with pytest.raises(ValueError):
        canonical_cohort(labels.iloc[:1], saved, 2024)


def test_rejects_changed_labels_and_immature_labels():
    labels, saved = frames()
    saved.loc[0, 'exact_hours'] = 3.
    with pytest.raises(AssertionError):
        canonical_cohort(labels, saved, 2024)
    labels, saved = frames()
    labels['label_available_at'] = pd.Timestamp('2025-01-01', tz='Asia/Shanghai')
    with pytest.raises(ValueError):
        canonical_cohort(labels, saved, 2024)
