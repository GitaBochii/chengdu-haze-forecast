"""Encoder-only path, checked against the pinned official full forward."""
import torch


def encode_spatial(model, batch):
    if model.residual != "climate" or model.positional_encoding != "fourier":
        raise ValueError("Only the pinned climate/Fourier configuration is supported")
    if model.mask_ratio_inputs != 0:
        raise ValueError("Feature extraction requires unmasked inputs")
    x = batch["x"]
    if x.shape[1:] != (model.input_size_time, model.in_channels, model.n_lats_px, model.n_lons_px):
        raise ValueError("Unexpected weather cube dimensions")
    # Mirror upstream preprocessing, but never accept/use future truth 'y'.
    normalized = (x-model.input_scalers_mu)/(model.input_scalers_sigma+model.input_scalers_epsilon)
    static = (batch["static"][:, 2:]-model.static_input_scalers_mu[:, 3:])/(
        model.static_input_scalers_sigma[:, 3:]+model.static_input_scalers_epsilon)
    climate = (batch["climate"]-model.input_scalers_mu.view(1, -1, 1, 1))/(
        model.input_scalers_sigma.view(1, -1, 1, 1)+model.input_scalers_epsilon)
    embedded = model.patch_embedding(model.parameter_dropout(normalized.flatten(1, 2)))
    static_embedded = model.patch_embedding_static(torch.cat((static, climate), dim=1))
    static_embedded += model.fourier_pos_encoding(batch["static"])
    tokens = model.to_patching(embedded)+model.to_patching(static_embedded)
    tokens += model.time_encoding(batch["input_time"], batch["lead_time"])
    masked, unmasked = model.generate_mask((x.shape[0], model._nglobal_mu))
    unmasked = unmasked.to(tokens.device)
    dim = masked.ndim
    view = (*unmasked.shape, *[1]*(tokens.ndim-dim))
    selected = torch.gather(tokens, dim=dim-1,
                            index=unmasked.view(*view).expand(*unmasked.shape, *tokens.shape[dim:]))
    return model.encoder(selected)
