import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


def test_prefix_benchmark_script_uses_base_feature_schema_and_namespaces_old_events(
    monkeypatch, capsys
):
    script_path = Path(__file__).parents[1] / "scripts" / "benchmark_cnemc_2014_prefix.py"
    spec = importlib.util.spec_from_file_location("benchmark_cnemc_2014_prefix", script_path)
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    base = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-01 00:00"], utc=True),
            "event_id": [1],
            "remaining_hours": [1.0],
            "baseline_feature": [4.0],
        }
    )
    prefix = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2014-01-01 00:00"], utc=True),
            "event_id": [1],
            "remaining_hours": [2.0],
            "baseline_feature": [3.0],
            "source_only_feature": [99.0],
        }
    )
    captured = {}
    monkeypatch.setattr(
        script.pd,
        "read_parquet",
        lambda path: base if path == Path("base.parquet") else prefix,
    )
    monkeypatch.setattr(script, "model_feature_names", lambda frame: ["baseline_feature"])
    monkeypatch.setattr(
        script,
        "benchmark_prefix_data_effect",
        lambda received_base, received_prefix, features, n_splits, seed: (
            captured.update(
                base=received_base,
                prefix=received_prefix,
                features=features,
                splits=n_splits,
                seed=seed,
            )
            or ({"with_historical_prefix": {}}, pd.DataFrame({"fold": [1]}))
        ),
    )
    monkeypatch.setattr(
        script,
        "write_benchmark_artifacts",
        lambda report, predictions, output: captured.update(output=output),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script_path),
            "--base-data",
            "base.parquet",
            "--prefix-data",
            "prefix.parquet",
            "--output",
            "result",
            "--splits",
            "3",
            "--seed",
            "7",
        ],
    )

    script.main()

    assert captured["features"] == ["baseline_feature"]
    assert captured["prefix"]["event_id"].tolist() == ["cnemc_2014_1"]
    assert "source_only_feature" in captured["prefix"].columns
    assert captured["output"] == Path("result")
    assert json.loads(capsys.readouterr().out) == {
        "base_data": "base.parquet",
        "prefix_data": "prefix.parquet",
        "output_dir": "result",
        "base_rows": 1,
        "prefix_rows": 1,
        "prediction_rows": 1,
    }
