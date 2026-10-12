"""Label-free rules for deciding when issued GFS paths inform a prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd

GATE_NAMES = ("never", "pbl_rise", "wind_rise", "pbl_or_wind", "drying")
_REQUIRED_COLUMNS = {
    "gfs_boundary_layer_height_m_f006",
    "gfs_boundary_layer_height_m_f024",
    "gfs_wind_u_ms_f006",
    "gfs_wind_v_ms_f006",
    "gfs_wind_u_ms_f024",
    "gfs_wind_v_ms_f024",
    "gfs_relative_humidity_pct_f006",
    "gfs_relative_humidity_pct_f024",
}


def gate_mask(frame: pd.DataFrame, name: str) -> pd.Series:
    """Return rows where a complete forecast signals stronger dispersion."""
    if name not in GATE_NAMES:
        raise ValueError(f"unknown gate: {name}")
    missing = _REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"GFS gate columns missing: {sorted(missing)}")

    complete = frame.loc[:, sorted(_REQUIRED_COLUMNS)].notna().all(axis=1)
    pbl_rise = (
        frame["gfs_boundary_layer_height_m_f024"] - frame["gfs_boundary_layer_height_m_f006"]
    ).ge(100)
    wind_gain = (
        np.hypot(frame["gfs_wind_u_ms_f024"], frame["gfs_wind_v_ms_f024"])
        - np.hypot(frame["gfs_wind_u_ms_f006"], frame["gfs_wind_v_ms_f006"])
    ).ge(1.0)
    drying = (
        frame["gfs_relative_humidity_pct_f024"] - frame["gfs_relative_humidity_pct_f006"]
    ).le(-10)
    rule = {
        "never": pd.Series(False, index=frame.index),
        "pbl_rise": pbl_rise,
        "wind_rise": wind_gain,
        "pbl_or_wind": pbl_rise | wind_gain,
        "drying": drying,
    }[name]
    return (complete & rule).astype(bool)


def select_gate(
    selection_mae: dict[str, float],
    confirmation_mae: dict[str, float],
    coverage: dict[str, float],
) -> str | None:
    """Accept a 2024-selected rule only after a bounded 2025 improvement."""
    if "never" not in selection_mae or "never" not in confirmation_mae:
        raise ValueError("selection and confirmation must include the never baseline")
    candidates = [
        name
        for name in selection_mae
        if name != "never" and 0.05 <= coverage.get(name, 0.0) <= 0.80
    ]
    if not candidates:
        return None
    selected = min(candidates, key=selection_mae.__getitem__)
    if (
        selection_mae[selected] >= selection_mae["never"]
        or confirmation_mae.get(selected, float("inf")) >= confirmation_mae["never"]
    ):
        return None
    return selected
