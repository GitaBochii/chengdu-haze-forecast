"""Parameter-free origin-observation mask for the regional AirFormer adapter."""

from types import MethodType

import torch
from torch import nn


def masked_attention(module, x):
    support = module._origin_support
    if support is None or support.all():
        return module._unmasked_forward(x)
    batch, nodes, channels = x.shape
    if batch % len(support):
        raise ValueError('Spatial batch must preserve origin-major time order')
    active = support.repeat_interleave(batch // len(support), dim=0).to(x.dtype)
    total = torch.einsum('bn,mnr->bmr', active, module.assignment)
    values = torch.einsum('bnc,mnr->bmrc', x * active[:, :, None], module.assignment)
    values = values / total.clamp_min(torch.finfo(x.dtype).tiny)[..., None]
    q = module.q_linear(x.reshape(-1, 1, channels)).reshape(
        batch * nodes, 1, module.num_heads, channels // module.num_heads).permute(0, 2, 1, 3)
    kv = module.kv_linear(values.reshape(-1, module.num_sector, channels)).reshape(
        batch * nodes, module.num_sector, 2, module.num_heads, channels // module.num_heads).permute(2, 0, 3, 1, 4)
    k, v = kv[0], kv[1]
    attention = ((q @ k.transpose(-2, -1)) * module.scale).reshape(
        batch, nodes, module.num_heads, 1, module.num_sector) + module.relative_bias
    attention = attention.masked_fill(total.eq(0)[:, :, None, None, :], float('-inf'))
    attention = attention.reshape(batch * nodes, module.num_heads, 1, module.num_sector).softmax(dim=-1)
    output = (attention @ v).transpose(1, 2).reshape(batch, nodes, channels)
    return module.proj_drop(module.proj(output))


class MaskedAirFormer(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        for module in self.base.modules():
            if module.__class__.__name__ == 'SpatialAttention':
                module._unmasked_forward = module.forward
                module._origin_support = None
                module.forward = MethodType(masked_attention, module)

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[-1] != self.base.input_dim + 1:
            raise ValueError('Expected original numerical features plus observation flag')
        flags = inputs[..., -1]
        if not ((flags == 0) | (flags == 1)).all():
            raise ValueError('Observation flags must be binary')
        active = flags.bool().any(dim=1)
        if not active.any(dim=1).all():
            raise ValueError('No observed AQI station at origin')
        modules = [m for m in self.base.modules() if m.__class__.__name__ == 'SpatialAttention']
        try:
            for module in modules:
                module._origin_support = active
            return self.base(inputs[..., :-1])
        finally:
            for module in modules:
                module._origin_support = None
