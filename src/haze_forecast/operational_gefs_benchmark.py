"""Decision guards for operational GEFS ensemble feature experiments."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from haze_forecast.operational_gefs_dataset import (
    OPERATIONAL_GEFS_BASE_FEATURES,
    OPERATIONAL_GEFS_FEATURES,
)
from haze_forecast.temporal_validation import event_isolated_year_split


def publication_audited_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Require explicit archive-availability audit, not just forecast initialization time."""
    flags = {'gefs_availability_policy', 'gefs_availability_audited'}
    if not flags.issubset(frame.columns):
        raise ValueError('GEFS availability audit missing; raw shards cannot enter this benchmark')
    if not frame.gefs_availability_policy.eq('archive_timestamp_proxy').all() or not (
        frame.gefs_availability_audited.eq(True).all()
    ):
        raise ValueError('invalid GEFS availability policy')
    if 'gefs_extraction_verified' not in frame or not frame.gefs_extraction_verified.eq(True).all():
        raise ValueError('GEFS extraction must be verified before masking unavailable features')
    if frame.duplicated(['timestamp', 'event_id']).any():
        raise ValueError('duplicate audited GEFS keys')
    return frame[['timestamp', 'event_id', *OPERATIONAL_GEFS_FEATURES]].copy()


def safe_event_year_split(frame: pd.DataFrame, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use Chengdu year boundaries and labels fully confirmed before training cutoff."""
    local = frame.copy()
    local['timestamp'] = pd.to_datetime(local.timestamp, utc=True).dt.tz_convert('Asia/Shanghai')
    train, test = event_isolated_year_split(local, year, event_column='event_key')
    confirmed = train.timestamp + pd.to_timedelta(train.remaining_hours + 5, unit='h')
    cutoff = pd.Timestamp(f'{year}-01-01', tz='Asia/Shanghai')
    bad = set(train.loc[confirmed.isna() | confirmed.ge(cutoff), 'event_key'])
    train = train.loc[~train.event_key.isin(bad)].copy()
    if train.empty or test.empty:
        raise ValueError('no rows after confirmed-label chronological split')
    return train, test

VARIANTS = {
    "aft": (),
    "mean_6": tuple(f"gefs_mean_{name}_f006" for name in OPERATIONAL_GEFS_BASE_FEATURES),
    "ensemble_6": tuple(
        f"gefs_{stat}_{name}_f006"
        for stat in ("mean", "spread")
        for name in OPERATIONAL_GEFS_BASE_FEATURES
    ),
    "ensemble_6_24": OPERATIONAL_GEFS_FEATURES,
}


def extracted_keys_are_complete(
    expected_keys: set[tuple[int, str]], observed_keys: set[tuple[int, str]]
) -> bool:
    """Return whether every eligible source event timestamp has an extracted GEFS row."""
    return expected_keys.issubset(observed_keys)


def select_2023_variant(row_mae: Mapping[str, float]) -> str:
    """Select exactly one predeclared candidate using only 2023 row-level MAE."""
    if "aft" not in row_mae or not set(row_mae).issubset(VARIANTS):
        raise ValueError("2023 selection must score aft and only predeclared GEFS variants")
    return min(row_mae, key=row_mae.__getitem__)


def should_accept_for_deployment(confirmations: Mapping[int, Mapping[str, bool]]) -> bool:
    """Require both 2024 and 2025 to improve both MAE views before any promotion."""
    if set(confirmations) != {2024, 2025}:
        raise ValueError("operational GEFS confirmation must contain only 2024 and 2025")
    return all(
        values.get("row_improved", False) and values.get("event_improved", False)
        for values in confirmations.values()
    )
