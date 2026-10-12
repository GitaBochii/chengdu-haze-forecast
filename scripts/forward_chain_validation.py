"""前向链时间验证：用 <= cutoff 的数据训练，预测 cutoff+1 年。

直接回答 P0：模型对近年（尤其 2023）的 3 倍低估，到底是"训练数据太旧"能修，
还是"特征对消散时长的信息上限被时间漂移打破"这种结构性问题。

事件均为年内完整，按年切分不会把同一事件拆到训练/测试两侧。
特征集与已部署模型一致（87 = 80 基础 + 7 个 PBL 缺失标志）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from haze_forecast.features import add_missingness_indicators
from haze_forecast.temporal_validation import event_isolated_year_split
from haze_forecast.training import AFTSurvivalRegressor

ART = Path("artifacts")
BASE = pd.read_parquet("data/processed/training_features.parquet")
PRE = pd.read_parquet("data/processed/cnemc_chengdu_2014_training_features.parquet")
Y22 = pd.read_parquet("data/processed/cnemc_chengdu_2022_training_features.parquet")
Y23 = pd.read_parquet("data/processed/cnemc_chengdu_2023_training_features.parquet")

PBL_COLS = [
    "boundary_layer_height_m",
    "boundary_layer_height_m_lag_1",
    "boundary_layer_height_m_lag_3",
    "boundary_layer_height_m_lag_24",
    "boundary_layer_height_m_roll_3",
    "boundary_layer_height_m_roll_24",
    "boundary_layer_height_m_change_3",
]


def prep(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    d = frame.copy()
    d["_year"] = d["timestamp"].dt.year
    d["event_key"] = source + ":" + d["event_id"].astype(str)
    d = add_missingness_indicators(d, PBL_COLS)
    # change_3 标志推导（与 app.populate_missingness_indicators 一致）
    flag = "boundary_layer_height_m_change_3_is_missing"
    if flag not in d.columns:
        d[flag] = (
            d["boundary_layer_height_m"].isna() | d["boundary_layer_height_m_lag_3"].isna()
        ).astype(float)
    return d


BASE = prep(BASE, "base")
PRE = prep(PRE, "prefix_2014")
Y22 = prep(Y22, "external_2022")
Y23 = prep(Y23, "external_2023")

ALL = pd.concat([PRE, BASE, Y22, Y23], ignore_index=True)

# 与已部署模型严格一致：直接读其元数据特征列表（87 个）
with open(ART / "model_metadata.json", encoding="utf-8") as fh:
    FEATURES = json.load(fh)["feature_names"]
assert len(FEATURES) == 87, len(FEATURES)


def evaluate(train_rows: pd.DataFrame, test_rows: pd.DataFrame, seed: int = 42):
    model = AFTSurvivalRegressor(seed=seed, nthread=1).fit(
        train_rows[FEATURES], train_rows["remaining_hours"]
    )
    pred = np.asarray(model.predict(test_rows[FEATURES]), dtype=float).clip(min=0.0)
    act = test_rows["remaining_hours"].to_numpy()
    mae = float(np.abs(pred - act).mean())
    return {
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "mae": round(mae, 3),
        "pred_mean": round(float(pred.mean()), 2),
        "act_mean": round(float(act.mean()), 2),
        "pred_std": round(float(pred.std()), 2),
    }


def forward_year_split(test_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train only on events completed before this year's first prediction."""
    return event_isolated_year_split(ALL, test_year, event_column="event_key")


print(f"{'cutoff':<7} {'训练行':>7} {'测试年':<7} {'测试行':>7} {'MAE':>8} {'预测均值':>9} {'实际均值':>9} {'预测std':>8}")
print("-" * 62)
records = []
for cutoff in (2017, 2018, 2019, 2020, 2021, 2022):
    train, test = forward_year_split(cutoff + 1)
    if len(test) == 0:
        continue
    t0 = time.time()
    r = evaluate(train, test)
    print(
        f"{cutoff:<7} {r['n_train']:>7} {cutoff + 1:<7} {r['n_test']:>7} "
        f"{r['mae']:>8.3f} {r['pred_mean']:>9.2f} {r['act_mean']:>9.2f} {r['pred_std']:>8.2f}"
    )
    records.append({"cutoff": cutoff, "test_year": cutoff + 1, **r})

# 对照：部署时可用的已完成事件在 2023 的表现
print("-" * 62)
deploy_train, deploy_test = forward_year_split(2023)
dep = evaluate(deploy_train, deploy_test)
print(
    f"{'deploy':<7} {dep['n_train']:>7} {'2023':<7} {dep['n_test']:>7} "
    f"{dep['mae']:>8.3f} {dep['pred_mean']:>9.2f} {dep['act_mean']:>9.2f} {dep['pred_std']:>8.2f}"
)

# 对照：直接用全部数据（含 2023）拟合 2023 = 样本内，仅看拟合上限
in23 = evaluate(ALL, Y23)
print(
    f"{'all-in':<7} {in23['n_train']:>7} {'2023':<7} {in23['n_test']:>7} "
    f"{in23['mae']:>8.3f} {in23['pred_mean']:>9.2f} {in23['act_mean']:>9.2f} {in23['pred_std']:>8.2f}"
)

out = {
    "protocol": (
        "event-isolated forward chain: training events must end before the test year's start; "
        "test rows are observed during that year; 87 deployed-aligned features"
    ),
    "runs": records,
    "deployed_model_on_2023": dep,
    "all_data_in_sample_on_2023": in23,
}
(ART / "forward_chain_validation_event_isolated.json").write_text(
    json.dumps(out, indent=2), encoding="utf-8"
)
print("\n已写入 artifacts/forward_chain_validation_event_isolated.json")
