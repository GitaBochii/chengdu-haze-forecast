"""回归模型的真实保留集评估与图表。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
    }


def write_evaluation_charts(predictions: pd.DataFrame, importance: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scatter = px.scatter(
        predictions,
        x="actual_remaining_hours",
        y="predicted_remaining_hours",
        hover_data=["timestamp"],
        title="保留测试集：预测消散剩余时间与真实值",
    )
    scatter.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=float(max(predictions["actual_remaining_hours"].max(), predictions["predicted_remaining_hours"].max())),
        y1=float(max(predictions["actual_remaining_hours"].max(), predictions["predicted_remaining_hours"].max())),
        line={"dash": "dash"},
    )
    scatter.write_html(output_dir / "predicted_vs_actual.html", include_plotlyjs="cdn")

    residuals = predictions.assign(
        residual=predictions["predicted_remaining_hours"] - predictions["actual_remaining_hours"]
    )
    px.scatter(
        residuals,
        x="timestamp",
        y="residual",
        title="保留测试集残差",
    ).write_html(output_dir / "residuals.html", include_plotlyjs="cdn")

    px.bar(
        importance.sort_values("importance", ascending=True).tail(20),
        x="importance",
        y="feature",
        orientation="h",
        title="LightGBM 特征重要性（前 20）",
    ).write_html(output_dir / "feature_importance.html", include_plotlyjs="cdn")
