import copy
import zlib

import pytest
import torch

from probe import make_model


def test_meta_encoder_keeps_concrete_regional_geometry():
    from encoder_loading import build_encoder
    values = {"mu": torch.zeros(160), "sigma": torch.ones(160),
              "static_mu": torch.zeros(11), "static_sigma": torch.ones(11),
              "output_sigma": torch.ones(160)}
    config = dict(in_channels=160, input_size_time=2, in_channels_static=8,
                  input_scalers_epsilon=0., static_input_scalers_epsilon=0.,
                  n_lats_px=360, n_lons_px=576, patch_size_px=(2, 2),
                  mask_unit_size_px=(30, 32), embed_dim=128,
                  n_blocks_encoder=1, n_blocks_decoder=2, mlp_multiplier=4,
                  n_heads=4, dropout=0., drop_path=0., parameter_dropout=0.)
    model = build_encoder(config, values, (30, 32))
    assert next(model.parameters()).device.type == "meta"
    assert not hasattr(model, "decoder")
    assert model._global_idx.device.type == "cpu"
    assert model._global_idx.numel() == 1
    assert model.encoder_shifter.local_mask.device.type == "cpu"
    assert model.encoder.lgl_block.shifter is model.encoder_shifter


def make_records(model, tmp_path):
    descriptions, records = {}, {}
    for index, (name, tensor) in enumerate(model.state_dict().items()):
        value = tensor.detach().contiguous()
        raw = value.numpy().tobytes()
        key = str(index)
        (tmp_path / f"{key}.bin").write_bytes(raw)
        descriptions[name] = {"storage": {"key": key, "elements": value.numel(),
                                          "dtype": str(value.dtype).split('.')[-1]},
                              "offset": 0, "shape": list(value.shape), "stride": list(value.stride())}
        records[key] = {"key": key, "bytes": len(raw), "crc": zlib.crc32(raw)}
    return descriptions, records


def test_strict_encoder_loading_and_named_decoder_exclusion(tmp_path):
    from encoder_loading import load_encoder_state
    original = make_model(False)
    descriptions, records = make_records(original, tmp_path)
    encoder = copy.deepcopy(original)
    del encoder.decoder, encoder.unembed, encoder.decoder_shifter
    for param in encoder.parameters():
        param.data.zero_()
    result = load_encoder_state(encoder, descriptions, records, tmp_path, "cpu", torch.float32)
    assert result["loaded_tensors"] == len(encoder.state_dict())
    assert result["excluded_decoder_tensors"] > 0
    for name, value in encoder.state_dict().items():
        torch.testing.assert_close(value, original.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("fault", ["missing", "extra", "shape"])
def test_rejects_incompatible_encoder(tmp_path, fault):
    from encoder_loading import load_encoder_state
    model = make_model(False)
    descriptions, records = make_records(model, tmp_path)
    del model.decoder, model.unembed, model.decoder_shifter
    name = next(iter(model.state_dict()))
    if fault == "missing":
        descriptions.pop(name)
    elif fault == "extra":
        descriptions["unrecognized_encoder_weight"] = descriptions[name]
    else:
        descriptions[name]["shape"] = [999]
    with pytest.raises(ValueError):
        load_encoder_state(model, descriptions, records, tmp_path, "cpu", torch.float32)
