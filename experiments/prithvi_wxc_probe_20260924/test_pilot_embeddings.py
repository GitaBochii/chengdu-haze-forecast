import torch
import pytest
from pathlib import Path


def test_cached_embedding_requires_exact_provenance_and_valid_values():
    from encode_pilot_batches import valid_cache
    identity = {"timestamp": "2020-01-01T14:00:00+08:00", "batch_sha256": "a"*64,
                "candidate_manifest_sha256": "b"*64, "weights_inventory_sha256": "c"*64}
    value = {**identity, "embedding": torch.arange(2560.).reshape(1, -1)}
    assert valid_cache(value, identity)
    assert not valid_cache(value, {**identity, "batch_sha256": "different"})
    assert not valid_cache({**value, "embedding": torch.zeros(1, 2560)}, identity)
    assert not valid_cache({**value, "embedding": torch.full((1, 2560), float('nan'))}, identity)
    assert not valid_cache({**value, "embedding": torch.ones(1, 2)}, identity)


def test_duplicate_report_rows_are_rejected_before_mapping():
    from encode_pilot_batches import validate_report_rows
    frozen = [{"timestamp": "a"}, {"timestamp": "b"}]
    with pytest.raises(ValueError):
        validate_report_rows([*frozen, frozen[0]], frozen)
    assert len(validate_report_rows(frozen, frozen)) == 2


def test_dependencies_change_cache_identity(tmp_path):
    from encode_pilot_batches import dependency_identity
    paths = ["official_batch.pt", "real_sample/config.yaml", "encoder_loading.py",
             "storage_loader.py", "spatial_encoder.py", "tensor_descriptors.json",
             "vendor/PrithviWxC/model.py", "encode_pilot_batches.py"]
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"first")
    before = dependency_identity(tmp_path)
    (tmp_path / "encoder_loading.py").write_bytes(b"changed")
    assert before != dependency_identity(tmp_path)


def test_real_batch_identity_and_future_flag():
    from encode_pilot_batches import validate_batch_identity
    root = Path(__file__).resolve().parent
    value = torch.load(root / "pilot_batches/20200101.pt", weights_only=True)
    origin = {"timestamp": value["timestamp"], "partition": value["partition"]}
    validate_batch_identity(value, origin, value["candidate_manifest_sha256"])
    for key, altered in [("future_actual_weather_read", True), ("revision", "main"),
                         ("origin_utc", "2020-01-02T06:00:00+00:00")]:
        with pytest.raises(ValueError):
            validate_batch_identity({**value, key: altered}, origin, value["candidate_manifest_sha256"])
