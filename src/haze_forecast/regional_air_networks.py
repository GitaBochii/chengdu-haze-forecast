"""Source-backed PM2.5-GNN adaptation and a matched simple sequence baseline.

The vendor files stay unmodified. Adaptations: native torch scatter, radians in
transport weights, hourly transport coefficient, FROM-bearing geometry, dynamic batches,
persisted origin features, and an independent AQI head using the same architecture.
This is not a reproduction of the paper's original data/forecast-weather protocol.
"""

from __future__ import annotations

import ast
import importlib.util
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
from torch import nn

VENDOR = Path(__file__).resolve().parents[2] / "experiments/regional_air_models/vendor/PM25-GNN"


def directional_edges(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 2:
        raise ValueError("at least two longitude/latitude coordinates are required")
    if not np.isfinite(coords).all():
        raise ValueError("coordinates must be finite")
    edges, attributes = [], []
    for source, (lon1, lat1) in enumerate(np.radians(coords)):
        for target, (lon2, lat2) in enumerate(np.radians(coords)):
            if source == target:
                continue
            dlon, dlat = lon2 - lon1, lat2 - lat1
            a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
            distance = 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
            if distance < 1e-3:
                raise ValueError("distinct nodes must have distinct coordinates")
            bearing = np.degrees(np.arctan2(
                np.sin(dlon) * np.cos(lat2),
                np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)))
            edges.append((source, target))
            attributes.append((distance, (bearing + 180) % 360))
    return np.asarray(edges, dtype=np.int64).T, np.asarray(attributes, dtype=np.float32)


def transport_weight(speed, wind_from_degrees, distance, edge_from_degrees, step_hours=1.0):
    return torch.relu(step_hours * speed * torch.cos(torch.deg2rad(
        edge_from_degrees - wind_from_degrees)) / distance.clamp_min(1e-3))


def _scatter_add(source, index, dim=1, dim_size=None):
    shape = list(source.shape)
    shape[dim] = dim_size if dim_size is not None else int(index.max()) + 1
    return source.new_zeros(shape).index_add(dim, index, source)


@lru_cache(maxsize=4)
def _official_module(step_hours: float = 1.0) -> ModuleType:
    cell_path = VENDOR / "model/cells.py"
    model_path = VENDOR / "model/PM25_GNN.py"
    if not cell_path.is_file() or not model_path.is_file():
        raise FileNotFoundError("official PM2.5-GNN sources are missing; run environment setup")
    spec = importlib.util.spec_from_file_location("_regional_pm25_cells", cell_path)
    cells = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cells)
    tree = ast.parse(model_path.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and node.module in ("model.cells", "torch_scatter"))]

    class RadianCorrection(ast.NodeTransformer):
        def visit_BinOp(self, node):
            self.generic_visit(node)
            if (isinstance(node.op, ast.Mult) and isinstance(node.left, ast.Constant)
                    and node.left.value == 3 and isinstance(node.right, ast.Name)
                    and node.right.id == "src_wind_speed"):
                node.left = ast.Constant(value=step_hours)
            return node

        def visit_Call(self, node):
            self.generic_visit(node)
            if (isinstance(node.func, ast.Attribute) and node.func.attr == "cos"
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "torch"):
                node.args[0] = ast.Call(func=ast.Attribute(value=ast.Name(id="torch", ctx=ast.Load()),
                                                         attr="deg2rad", ctx=ast.Load()),
                                        args=[node.args[0]], keywords=[])
            return node

    tree = ast.fix_missing_locations(RadianCorrection().visit(tree))
    module = ModuleType("_regional_pm25_official")
    module.GRUCell = cells.GRUCell
    module.scatter_add = _scatter_add
    exec(compile(tree, str(model_path), "exec"), module.__dict__)  # noqa: S102 - pinned vendor source
    return module


class PM25GNNAdapter(nn.Module):
    """Official GNN/GRU core, with origin-persisted exogenous input for each lead."""

    def __init__(self, input_dim, node_coords, wind_mean, wind_scale,
                 history=24, horizon=72, target_indices=(0, 2), step_hours=1.0):
        super().__init__()
        self.history, self.horizon = history, horizon
        self.target_indices = target_indices
        edges, attrs = directional_edges(node_coords)
        if step_hours <= 0:
            raise ValueError("transport step must be positive")
        official = _official_module(step_hours)
        self.heads = nn.ModuleList()
        for _ in target_indices:
            model = official.PM25_GNN(history, horizon, input_dim + 1, len(node_coords), 1,
                                     "cpu", edges, attrs, wind_mean, wind_scale)
            edge = model.graph_gnn.edge_attr
            model.graph_gnn.edge_attr_norm = (edge - edge.mean(0)) / edge.std(
                dim=0, unbiased=False).clamp_min(1e-6)
            for name in ("edge_index", "edge_attr", "edge_attr_norm", "wind_mean", "wind_std"):
                value = getattr(model.graph_gnn, name)
                delattr(model.graph_gnn, name)
                model.graph_gnn.register_buffer(name, value)
            self.heads.append(model)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.history:
            raise ValueError("expected [batch, history, node, feature] inputs")
        future_features = x[:, -1:].expand(-1, self.history + self.horizon, -1, -1)
        predictions = []
        for index, model in zip(self.target_indices, self.heads, strict=True):
            model.batch_size = x.shape[0]
            model.device = x.device
            model.graph_gnn.device = x.device
            model.graph_gnn.wind_mean = model.graph_gnn.wind_mean.to(x.device)
            model.graph_gnn.wind_std = model.graph_gnn.wind_std.to(x.device)
            predictions.append(model(x[..., index:index + 1], future_features))
        return torch.cat(predictions, dim=-1)


class PM25GNNHistoryAdapter(PM25GNNAdapter):
    """Warm the same graph/GRU state with observed history; add no parameters."""

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.history:
            raise ValueError("expected [batch, history, node, feature] inputs")
        batch, _, nodes, _ = x.shape
        predictions = []
        for index, model in zip(self.target_indices, self.heads, strict=True):
            model.graph_gnn.device = x.device
            hidden = x.new_zeros(batch * nodes, model.hid_dim)

            def step(values, state, model=model):
                spatial = model.graph_gnn(values.contiguous())
                return model.gru_cell(torch.cat([spatial, values], dim=-1), state)

            for hour in range(self.history - 1):
                observed = torch.cat([x[:, hour, :, index:index + 1], x[:, hour]], dim=-1)
                hidden = step(observed, hidden)
            current = x[:, -1, :, index:index + 1]
            decoded = []
            for _ in range(self.horizon):
                hidden = step(torch.cat([current, x[:, -1]], dim=-1), hidden)
                current = model.fc_out(hidden.view(batch, nodes, model.hid_dim))
                decoded.append(current)
            predictions.append(torch.stack(decoded, dim=1))
        return torch.cat(predictions, dim=-1)


class SequenceGRU(nn.Module):
    """Shared per-node GRU; no spatial mixing, same history/features/targets."""

    def __init__(self, input_dim: int, horizon: int = 72, output_dim: int = 2, hidden: int = 32):
        super().__init__()
        self.horizon, self.output_dim = horizon, output_dim
        self.gru = nn.GRU(input_dim, hidden, batch_first=True)
        self.output = nn.Linear(hidden, horizon * output_dim)

    def forward(self, x):
        batch, history, nodes, features = x.shape
        sequence = x.permute(0, 2, 1, 3).reshape(batch * nodes, history, features)
        _, hidden = self.gru(sequence)
        return self.output(hidden[-1]).reshape(
            batch, nodes, self.horizon, self.output_dim).permute(0, 2, 1, 3)
