from pathlib import Path


def test_preflight_reports_missing_or_truncated_required_storage(tmp_path):
    from run_pretrained_encoder import storage_preflight
    descriptions = {
        "encoder.weight": {"storage": {"key": "1"}},
        "decoder.weight": {"storage": {"key": "2"}},
    }
    records = {"1": {"bytes": 8}, "2": {"bytes": 8}}
    result = storage_preflight(descriptions, records, tmp_path)
    assert result == {"required_records": 1, "missing_records": ["1"], "invalid_size_records": []}
    (tmp_path / "1.bin").write_bytes(b"short")
    assert storage_preflight(descriptions, records, tmp_path)["invalid_size_records"] == ["1"]
    (tmp_path / "1.bin").write_bytes(b"12345678")
    assert storage_preflight(descriptions, records, tmp_path)["missing_records"] == []
    assert storage_preflight(descriptions, records, tmp_path)["invalid_size_records"] == []
