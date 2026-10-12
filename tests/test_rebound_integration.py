from pathlib import Path
from haze_forecast.showcase_server import STATIC_FILES

ROOT = Path(__file__).resolve().parents[1]


def test_rebound_assets_are_served_without_exposing_model_files():
    assert {'rebound.js', 'rebound-analysis.js', 'rebound.css', 'rebound-cases.json'} <= STATIC_FILES
    assert not any(name.endswith(('.npy', '.parquet', '.joblib')) for name in STATIC_FILES)


def test_rebound_panel_is_separate_from_original_model_results():
    html = (ROOT / 'project-showcase/workbench.html').read_text('utf-8')
    assert 'id="tab-rebound"' in html
    assert 'id="rebound-panel"' in html
    assert 'id="rebound-controls"' in html
    assert 'id="rebound-view" hidden' in html
    assert html.index('src="rebound-analysis.js"') < html.index('src="rebound.js"')
    script = (ROOT / 'project-showcase/workbench.js').read_text('utf-8')
    assert 'window.hideReboundView?.()' in script
    assert 'window.openReboundView?.()' in script
    assert "if(mode!=='rebound')clearResult()" in script
