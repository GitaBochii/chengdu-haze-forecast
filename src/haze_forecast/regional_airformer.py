"""Local research adaptation of https://github.com/yoshall/AirFormer.

The vendored official deterministic AirFormer supplies DS-MSA, CT-MSA, batch
normalization and the forecast decoder. Changes: continuous-feature 1x1 input
projection instead of dataset-specific categorical embeddings; regional geometry;
registered device buffers; a smaller decoder; and a [B,P,N,C] output interface.
This is not a full-paper reproduction or a claim of upstream redistribution rights.
"""

from __future__ import annotations

import ast
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
from torch import nn


def _load_source(name: str, path: Path, remap: dict[str, str] | None = None) -> ModuleType:
    """Load upstream code under private names, rewriting only absolute imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in (remap or {}):
            node.module = remap[node.module]
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)  # noqa: S102 - pinned vendor source
    return module


@lru_cache(maxsize=1)
def _official_airformer() -> type[nn.Module]:
    root = Path(__file__).resolve().parents[2] / "experiments/regional_air_models/vendor/AirFormer"
    paths = {
        "src.base.model": root / "src/base/model.py",
        "src.layers.embedding": root / "src/layers/embedding.py",
    }
    main = root / "src/models/airformer.py"
    for path in (*paths.values(), main):
        if not path.is_file():
            raise FileNotFoundError(f"Official AirFormer source is required: {path}")
    remap = {name: "_haze_airformer_" + name.replace(".", "_") for name in paths}
    for name, path in paths.items():
        _load_source(remap[name], path)
    return _load_source("_haze_airformer_model", main, remap).AirFormer


def _dartboard(coords: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Self + 8 bearing sectors in each of <50 km and >=50 km rings.

    Bearings use a regional equirectangular approximation in kilometers. Each
    occupied sector averages its nodes, empty sectors are masked. Unlike national
    partitions, the outer ring is unbounded so every regional node is represented.
    """
    longitude, latitude = np.deg2rad(coords).T
    east = 6371.0 * (longitude[None, :] - longitude[:, None]) * np.cos(
        (latitude[None, :] + latitude[:, None]) / 2
    )
    north = 6371.0 * (latitude[None, :] - latitude[:, None])
    distance = np.hypot(east, north)
    bearing_sector = np.floor((np.arctan2(north, east) % (2 * np.pi)) / (np.pi / 4)).astype(int)
    sectors = 1 + bearing_sector + 8 * (distance >= 50.0)
    np.fill_diagonal(sectors, 0)
    assignment = np.eye(17, dtype=np.float32)[sectors]
    counts = assignment.sum(axis=1)
    assignment /= np.maximum(counts[:, None, :], 1)
    return torch.from_numpy(assignment), torch.from_numpy(counts == 0)


def create_airformer(
    input_dim: int,
    node_coords: np.ndarray,
    history: int = 24,
    horizon: int = 72,
    output_dim: int = 2,
    hidden: int = 16,
) -> nn.Module:
    """Create an untrained deterministic regional adaptation of official AirFormer."""
    coords = np.asarray(node_coords, dtype=float)
    if (
        coords.ndim != 2 or coords.shape[1] != 2 or len(coords) == 0
        or not np.isfinite(coords).all()
        or (np.abs(coords[:, 0]) > 180).any() or (np.abs(coords[:, 1]) > 90).any()
    ):
        raise ValueError("node coordinates must be finite [N,2] longitude/latitude degrees")
    if any(not isinstance(value, int) or value < 1 for value in (input_dim, history, horizon, output_dim, hidden)):
        raise ValueError("model dimensions must be positive integers")
    if hidden % 2:
        raise ValueError("hidden must be divisible by the two attention heads")
    assignment, mask = _dartboard(coords)
    blocks = 1
    while blocks < 4 and history % (2 ** blocks) == 0:
        blocks += 1

    class RegionalAirFormer(_official_airformer()):
        def get_dartboard_info(self, dartboard):
            self.register_buffer("assignment", assignment)
            self.register_buffer("mask", mask)

        def forward(self, inputs):
            expected = (self.seq_len, self.num_nodes, self.input_dim)
            if inputs.ndim != 4 or tuple(inputs.shape[1:]) != expected:
                raise ValueError(f"inputs must have shape [B,{expected[0]},{expected[1]},{expected[2]}]")
            # Preserve the official deterministic path, bypassing AirEmbedding.
            state = self.start_conv(inputs.permute(0, 3, 2, 1))
            states = []
            for spatial, temporal, norm in zip(self.s_modules, self.t_modules, self.bn, strict=True):
                state = norm(temporal(spatial(state)))
                states.append(state)
            last_states = torch.cat(states, dim=1)[..., -1:]
            decoded = self.end_conv_2(torch.relu(self.end_conv_1(torch.relu(last_states))))
            return decoded.squeeze(-1).reshape(
                inputs.shape[0], self.horizon, self.output_dim, self.num_nodes
            ).permute(0, 1, 3, 2).contiguous()

    model = RegionalAirFormer(
        name="regional_airformer_deterministic", dataset="chengdu_region",
        device=torch.device("cpu"), num_nodes=len(coords), seq_len=history,
        horizon=horizon, input_dim=input_dim, output_dim=output_dim,
        hidden_channels=hidden, end_channels=hidden * 4, blocks=blocks,
        num_heads=2, spatial_flag=True, stochastic_flag=False,
    )
    del model.embedding_air
    # Upstream attention stores these tensors as plain attributes, so .to() does
    # not move them. Register every consumer's references as device-aware buffers.
    for module in model.modules():
        for name in ("assignment", "mask"):
            value = getattr(module, name, None)
            if isinstance(value, torch.Tensor) and name not in module._buffers:
                delattr(module, name)
                module.register_buffer(name, value)
    return model
