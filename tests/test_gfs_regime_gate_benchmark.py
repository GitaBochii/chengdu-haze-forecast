import json

from haze_forecast.gfs_regime_gate import select_gate


def test_select_gate_requires_2024_selection_and_2025_improvement():
    selection = {"never": 16.44, "pbl_or_wind": 15.90}
    confirmation = {"never": 26.27, "pbl_or_wind": 24.50}
    coverage = {"pbl_or_wind": 0.25}

    assert select_gate(selection, confirmation, coverage) == "pbl_or_wind"
    assert select_gate(selection, {"never": 26.27, "pbl_or_wind": 26.5}, coverage) is None
    assert select_gate(selection, confirmation, {"pbl_or_wind": 0.90}) is None


def test_report_decision_records_that_rejected_gate_does_not_change_deployment(tmp_path):
    report = {"decision": {"status": "reject", "deployment_artifact_changed": False}}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    assert json.loads(path.read_text(encoding="utf-8"))["decision"]["deployment_artifact_changed"] is False
