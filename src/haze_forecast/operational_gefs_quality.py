"""Strict source-quality consolidation for immutable operational GEFS retries."""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from haze_forecast.operational_gefs import gefs_data_url
from haze_forecast.operational_gefs_dataset import OPERATIONAL_GEFS_FEATURES

ENSEMBLE_FIELDS = ('temperature_2m', 'relative_humidity_2m', 'u_wind_10m', 'v_wind_10m')
EXPECTED_MESSAGES = {
    (product, field, lead) for lead in (6, 24) for product in ('geavg', 'gespr', 'gec00')
    for field in (('pblh',) if product == 'gec00' else ENSEMBLE_FIELDS)
}


def _utc(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError('GEFS timestamps must be nonmissing and timezone-aware')
    return timestamp.tz_convert('UTC')


def verified_row_signature(row: pd.Series) -> tuple:
    """Validate extraction evidence, not historical publication availability."""
    error = row.get('gefs_extraction_error')
    if not pd.isna(error) and str(error).strip():
        raise ValueError('nonempty GEFS extraction error')
    values = pd.to_numeric(row.reindex(OPERATIONAL_GEFS_FEATURES), errors='coerce').to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError('GEFS requires 18 finite features')
    run, issue = _utc(row['gefs_run_utc']), _utc(row['timestamp'])
    if run.hour % 6 or any((run.minute, run.second, run.microsecond)):
        raise ValueError('GEFS run must be a six-hour cycle')
    if run > issue:
        raise ValueError('GEFS run follows prediction time')
    valid_times = []
    for lead in (6, 24):
        valid = _utc(row[f'gefs_forecast_valid_utc_f{lead:03d}'])
        if valid != run + pd.Timedelta(lead, unit='h') or valid <= issue:
            raise ValueError('GEFS valid time does not match run/lead/issue')
        valid_times.append(valid.isoformat())
    manifest = json.loads(row['gefs_source_manifest_json'])
    if _utc(manifest['run_utc']) != run:
        raise ValueError('GEFS source manifest run mismatch')
    messages = manifest['messages']
    if not isinstance(messages, list) or len(messages) != 18:
        raise ValueError('GEFS source manifest requires 18 messages')
    identities, signatures = set(), []
    for message in messages:
        identity = (message['product'], message['field'], message['lead_hours'])
        if identity not in EXPECTED_MESSAGES or identity in identities:
            raise ValueError('GEFS source message identity missing or duplicated')
        identities.add(identity)
        product, _, lead = identity
        url = gefs_data_url(run.to_pydatetime(), product=product, lead_hours=lead)
        if message['source_url'] != url or message['index_url'] != url + '.idx':
            raise ValueError('GEFS source URL mismatch')
        if not isinstance(message['sha256'], str) or not re.fullmatch(r'[0-9a-fA-F]{64}', message['sha256']):
            raise ValueError('invalid GEFS source SHA256')
        byte_range = message['byte_range']
        if (not isinstance(byte_range, list) or len(byte_range) != 2
                or any(type(value) is not int for value in byte_range)
                or byte_range[0] < 0 or byte_range[1] < byte_range[0]):
            raise ValueError('invalid GEFS source byte range')
        signatures.append((*identity, url, message['sha256'].lower(), *byte_range))
    return (tuple(values), run.isoformat(), tuple(valid_times), tuple(sorted(signatures)))


def consolidate_verified_shards(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Prefer successful retries per UTC key; reject conflicting successful versions.

    Failed keys remain present with ``gefs_row_verified=False`` for repair planning.
    A consumer claiming complete extraction must require that flag for every key.
    """
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=['timestamp', 'event_id', 'gefs_row_verified',
                                     'gefs_quality_error'])
    combined = pd.concat(frames, ignore_index=True)
    if not {'timestamp', 'event_id'}.issubset(combined):
        raise ValueError('GEFS shards require timestamp/event_id keys')
    combined['timestamp'] = pd.to_datetime([_utc(value) for value in combined.timestamp], utc=True)
    ids = pd.to_numeric(combined.event_id, errors='raise')
    if ids.isna().any() or not ids.gt(0).all() or not ids.mod(1).eq(0).all():
        raise ValueError('GEFS event IDs must be positive integers')
    combined['event_id'] = ids.astype('int64')
    verified, reasons, signatures = [], [], []
    for _, row in combined.iterrows():
        try:
            signature = verified_row_signature(row)
        except (ValueError, KeyError, TypeError, OverflowError) as error:
            verified.append(False)
            reasons.append(str(error))
            signatures.append(None)
        else:
            verified.append(True)
            reasons.append('')
            signatures.append(signature)
    combined['gefs_row_verified'] = verified
    combined['gefs_quality_error'] = reasons
    selected = []
    for key, group in combined.groupby(['timestamp', 'event_id'], sort=True):
        successful = group.loc[group.gefs_row_verified]
        if len(successful) > 1:
            first = signatures[successful.index[0]]
            if any(signatures[index] != first for index in successful.index[1:]):
                raise ValueError(f'conflicting successful GEFS retries for {key}')
        selected.append(successful.index[0] if len(successful) else group.index[0])
    return combined.loc[selected].reset_index(drop=True)
