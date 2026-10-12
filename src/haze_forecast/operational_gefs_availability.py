"""Mask archive GEFS inputs using object timestamps, not claimed public release times."""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import requests

from haze_forecast.operational_gefs import gefs_data_url, validate_operational_gefs_timing
from haze_forecast.operational_gefs_dataset import OPERATIONAL_GEFS_FEATURES

POLICY = "archive_timestamp_proxy"
LIMITATIONS = [
    "S3 Last-Modified is an archive object timestamp, not independently verified public release time.",
    "Historical ACL and first public availability are unverified; no automatic operational promotion.",
    "Only quoted single-part hexadecimal ETags are accepted; multipart or missing metadata is unknown.",
    "Forecast initialization time is not publication time; no inference from cycle hour alone.",
]
_FIELD_NAMES = {"temperature_2m": "temperature_c", "relative_humidity_2m": "relative_humidity_pct",
                "u_wind_10m": "wind_u_ms", "v_wind_10m": "wind_v_ms", "pblh": "boundary_layer_height_m"}
_STATS = {"geavg": "mean", "gespr": "spread", "gec00": "control"}


def head_evidence(url: str, cache_dir: Path, *, head=None, timeout_s: int = 30,
                  attempts: int = 3, sleeper=time.sleep) -> dict:
    """Reuse successful HEAD snapshots; failed attempts are immutable and retryable."""
    if not url.startswith("https://noaa-gefs-pds.s3.amazonaws.com/gefs."):
        raise ValueError("only official GEFS S3 object URLs are accepted")
    if attempts not in (1, 2, 3) or timeout_s <= 0:
        raise ValueError("HEAD attempts must be 1-3 and timeout positive")
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode()).hexdigest()
    def read(path):
        content = path.read_bytes()
        value = json.loads(content)
        if value.get("url") != url:
            raise ValueError("HEAD cache URL identity mismatch")
        return {**value, "cache_path": str(path.resolve()),
                "cache_sha256": hashlib.sha256(content).hexdigest()}
    for path in sorted(cache_dir.glob(f"{key}_*.json")):
        value = read(path)
        if value.get("http_status") == 200:
            return value
    request = head or requests.head
    for attempt in range(attempts):
        value = {"url": url, "checked_at_utc": datetime.now(UTC).isoformat(),
                 "http_status": None, "last_modified": None, "etag": None}
        try:
            response = request(url, timeout=timeout_s, allow_redirects=False)
            value.update(http_status=response.status_code,
                         last_modified=response.headers.get("Last-Modified"),
                         etag=response.headers.get("ETag"),
                         content_length=response.headers.get("Content-Length"),
                         version_id=response.headers.get("x-amz-version-id"))
        except requests.RequestException as error:
            value["error"] = str(error)
        path = cache_dir / f"{key}_{datetime.now(UTC):%Y%m%dT%H%M%S%f}_{uuid4().hex}.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
        saved = read(path)
        if value["http_status"] == 200:
            return saved
        if attempt + 1 < attempts:
            sleeper(2 ** attempt)
    return saved


def _utc(value, name: str) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if pd.isna(parsed) or parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"{name} must be timezone-aware UTC")
    return parsed


def _raw_messages(row: pd.Series) -> dict[str, dict]:
    issue, run = _utc(row.timestamp, "timestamp"), _utc(row.gefs_run_utc, "run")
    if not np.isfinite(pd.to_numeric(row[list(OPERATIONAL_GEFS_FEATURES)], errors="coerce")).all():
        raise ValueError("raw extraction contains missing/non-finite GEFS features; repair first")
    error = row.get("gefs_extraction_error")
    if pd.notna(error) and str(error).strip():
        raise ValueError("raw extraction has a recorded error; repair first")
    try:
        manifest = json.loads(row.gefs_source_manifest_json)
    except (TypeError, ValueError) as error:
        raise ValueError("raw source manifest missing/invalid; repair first") from error
    if _utc(manifest.get("run_utc"), "manifest run") != run:
        raise ValueError("manifest run differs from raw row")
    for lead in (6, 24):
        valid = _utc(row[f"gefs_forecast_valid_utc_f{lead:03d}"], "valid")
        if valid != run + pd.Timedelta(hours=lead):
            raise ValueError("raw forecast lead/valid mismatch")
        validate_operational_gefs_timing(issue.to_pydatetime(), run.to_pydatetime(), valid.to_pydatetime())
    messages = {}
    for message in manifest.get("messages", []):
        product, lead, field = (message.get(key) for key in ("product", "lead_hours", "field"))
        if product not in _STATS or lead not in (6, 24) or field not in _FIELD_NAMES:
            raise ValueError("unexpected source message identity")
        if (product == "gec00") != (field == "pblh"):
            raise ValueError("unexpected product/field pairing")
        url = gefs_data_url(run.to_pydatetime(), product=product, lead_hours=lead)
        if message.get("source_url") != url or message.get("index_url") != url + ".idx":
            raise ValueError("source URL does not match run/product/lead")
        byte_range = message.get("byte_range")
        if (not isinstance(byte_range, list) or len(byte_range) != 2
                or not all(isinstance(value, int) for value in byte_range)
                or not 0 <= byte_range[0] <= byte_range[1]
                or not re.fullmatch(r"[0-9a-f]{64}", str(message.get("sha256", "")))):
            raise ValueError("source byte range or SHA256 invalid")
        feature = f"gefs_{_STATS[product]}_{_FIELD_NAMES[field]}_f{lead:03d}"
        if feature in messages:
            raise ValueError("duplicate source feature")
        messages[feature] = message
    if set(messages) != set(OPERATIONAL_GEFS_FEATURES):
        raise ValueError("raw manifest does not cover all 18 extracted features; repair first")
    return messages


def availability_state(evidence: dict, issue: pd.Timestamp,
                       run: pd.Timestamp | None = None) -> tuple[str, str]:
    if evidence.get("http_status") != 200:
        return "unknown", "HEAD was not HTTP 200"
    if not re.fullmatch(r'"[0-9a-fA-F]{32}"', str(evidence.get("etag", ""))):
        return "unknown", "ETag missing or not single-part hexadecimal"
    try:
        modified = parsedate_to_datetime(evidence.get("last_modified"))
        modified = _utc(modified, "Last-Modified")
    except (TypeError, ValueError, OverflowError):
        return "unknown", "Last-Modified missing, invalid or not UTC"
    if run is not None and modified < run:
        return "unknown", "object timestamp predates model initialization"
    if modified <= issue:
        return "available", "archive_timestamp_proxy only"
    return "not_yet_available", "object timestamp follows prediction issue"


def audit_frame(frame: pd.DataFrame, evidence_fetcher) -> tuple[pd.DataFrame, dict]:
    """Preserve rows and raw provenance; mask each product/lead independently."""
    if frame.empty or frame.duplicated(["timestamp", "event_id"]).any():
        raise ValueError("expected non-empty unique timestamp/event_id rows")
    # Validate the whole input before making any external request.
    row_messages = [_raw_messages(row) for _, row in frame.iterrows()]
    result = frame.copy()
    masks = {feature: [] for feature in OPERATIONAL_GEFS_FEATURES}
    evidence_rows, all_available, local_cache = [], [], {}
    counts = {state: 0 for state in ("available", "not_yet_available", "unknown")}
    affected, unknown_rows = 0, 0
    per_feature = {feature: dict.fromkeys(counts, 0) for feature in OPERATIONAL_GEFS_FEATURES}
    for (_, row), messages in zip(frame.iterrows(), row_messages, strict=True):
        issue = _utc(row.timestamp, "timestamp")
        objects, row_states = {}, []
        for feature, message in messages.items():
            url = message["source_url"]
            if url not in local_cache:
                local_cache[url] = evidence_fetcher(url)
            evidence = local_cache[url]
            if evidence.get("url") != url:
                raise ValueError("evidence URL identity mismatch")
            state, reason = availability_state(evidence, issue, _utc(row.gefs_run_utc, "run"))
            masks[feature].append(state != "available")
            counts[state] += 1
            per_feature[feature][state] += 1
            row_states.append(state)
            objects[url] = {**evidence, "state": state, "reason": reason}
        all_available.append(all(state == "available" for state in row_states))
        affected += not all_available[-1]
        unknown_rows += "unknown" in row_states
        evidence_rows.append(json.dumps(objects, sort_keys=True, separators=(",", ":")))
    for feature, mask in masks.items():
        result.loc[np.asarray(mask), feature] = np.nan
    result["gefs_availability_policy"] = POLICY
    result["gefs_availability_audited"] = True
    result["gefs_extraction_verified"] = True
    result["gefs_availability_all_available"] = all_available
    result["gefs_availability_evidence_json"] = evidence_rows
    total = len(frame) * len(OPERATIONAL_GEFS_FEATURES)
    report = {"policy": POLICY, "limitations": LIMITATIONS, "rows": len(frame),
              "unique_objects": len(local_cache), "values": total, "state_counts": counts,
              "available_fraction": counts["available"] / total,
              "unknown_fraction": counts["unknown"] / total,
              "masked_values": total - counts["available"], "unknown_values": counts["unknown"],
              "affected_rows": int(affected), "unknown_rows": int(unknown_rows),
              "fully_available_rows": int(sum(all_available)), "features": per_feature}
    return result, report
