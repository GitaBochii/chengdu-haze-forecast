"""Strict encoder loading from individually verified checkpoint storages."""
from pathlib import Path
import sys

import torch

from storage_loader import load_tensor


def build_encoder(config, values, spatial_shape):
    sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
    from PrithviWxC.model import PrithviWxC, SWINShiftNoBuffer
    config = dict(config)
    config.update(n_lats_px=spatial_shape[0], n_lons_px=spatial_shape[1], n_blocks_decoder=0)
    with torch.device("meta"):
        model = PrithviWxC(
            **config, input_scalers_mu=values["mu"], input_scalers_sigma=values["sigma"],
            static_input_scalers_mu=values["static_mu"], static_input_scalers_sigma=values["static_sigma"],
            output_scalers=values["output_sigma"], mask_ratio_inputs=0., mask_ratio_targets=0.,
            residual="climate", masking_mode="global", positional_encoding="fourier",
            encoder_shifting=True, decoder_shifting=False)
    # These upstream geometry tensors are not registered buffers. Reconstruct
    # them on CPU for the regional grid rather than copying meta placeholders.
    model._global_idx = torch.arange(model._nglobal_mu)
    model._local_idx = torch.arange(model._nlocal_mu)
    shifter = SWINShiftNoBuffer(model.mask_unit_size_px, model.global_shape_mu,
                               model.local_shape_mu, model.patch_size_px, n_context_tokens=0)
    model.encoder_shifter = shifter
    model.encoder.lgl_block.shifter = shifter
    return model


def load_encoder_state(model, descriptions, records, folder, device, dtype):
    expected = model.state_dict()
    excluded = {name for name in descriptions
                if name.startswith(("decoder.", "decoder_shifter.", "unembed."))}
    selected = descriptions.keys()-excluded
    if set(expected) != selected:
        raise ValueError("Encoder key mismatch; refusing partial or unexpected weights")
    for name, target in expected.items():
        if list(target.shape) != descriptions[name]["shape"]:
            raise ValueError(f"Encoder shape mismatch: {name}")
    converted = {}
    parameters = set(dict(model.named_parameters()))
    for name in expected:
        descriptor = descriptions[name]
        tensor = load_tensor(folder, descriptor, records[descriptor["storage"]["key"]])
        if tensor.dtype != expected[name].dtype:
            raise ValueError(f"Encoder dtype mismatch: {name}")
        # Keep normalization buffers in FP32 even when transformer weights use BF16.
        destination_dtype = dtype if name in parameters and tensor.is_floating_point() else tensor.dtype
        converted[name] = tensor.to(device=device, dtype=destination_dtype)
    model.load_state_dict(converted, strict=True, assign=True)
    model.requires_grad_(False).eval()
    return {"loaded_tensors": len(converted), "excluded_decoder_tensors": len(excluded),
            "strict_load": True, "parameter_dtype": str(dtype), "device": str(device)}
