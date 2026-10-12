from pathlib import Path

from haze_forecast.showcase_server import STATIC_FILES


def test_effect_resources_are_served_without_opening_other_paths():
    root = Path(__file__).resolve().parents[1] / "project-showcase"
    for name in ("effects.js", "effects.css"):
        assert name in STATIC_FILES
        assert (root / name).is_file()
    assert "qa/effects.test.cjs" not in STATIC_FILES
