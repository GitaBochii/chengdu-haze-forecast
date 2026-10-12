import json

from haze_forecast.event_residual_calibration import select_candidate


def test_select_candidate_uses_2024_only_and_requires_stable_confirmation():
    selection = {"none": 16.4, "state": 15.0}
    confirmation = {"none": 26.27, "state": 25.0}

    assert select_candidate(selection, confirmation, baseline_name="none") == "state"
    assert select_candidate(selection, {"none": 26.27, "state": 27.0}, baseline_name="none") is None


def test_benchmark_report_marks_rejected_candidate_when_2025_does_not_improve(tmp_path):
    report = {"decision": {"status": "reject", "reason": "2025_confirmation_not_better_than_baseline"}}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert loaded["decision"]["status"] == "reject"
