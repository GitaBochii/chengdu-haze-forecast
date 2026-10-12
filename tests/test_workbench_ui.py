"""Exercise the redesigned workbench with the real local prediction resources."""

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from haze_forecast.workbench_ui import readings_table, series_summary


def test_series_summary_keeps_missing_current_value_and_observed_statistics():
    context = pd.DataFrame({"aqi": [100.0, 140.0, float("nan")],
                            "pm2_5": [10.0, 20.0, 30.0], "pm10": [float("nan")] * 3})
    markup = series_summary(context)
    assert "<dt>当前</dt><dd>暂无</dd>" in markup
    assert "<dt>均值</dt><dd>120.0</dd>" in markup
    assert "<dt>当前</dt><dd>30.0</dd>" in markup
    assert "AQI</strong> <small>指数" in markup
    assert "nan" not in markup


def test_readings_table_preserves_values_units_and_escapes_labels():
    markup = readings_table([{"污染物": "PM2.5", "单位": "µg/m³", "数值": 98.0},
                             {"污染物": "<sample>", "单位": "", "数值": 1.5}])
    assert "98</td>" in markup
    assert "1.5</td>" in markup
    assert "µg/m³" in markup
    assert "&lt;sample&gt;" in markup
    assert 'scope="col"' in markup
    assert 'tabindex="0"' in markup


def test_workbench_navigation_and_strict_result():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "src/haze_forecast/app.py"))
    app.run(timeout=90)
    assert not app.exception
    assert app.radio[0].value == "严格历史回测"
    assert any("23.1 小时" in item.value for item in app.markdown)
    assert any("3.5 - 164.4" in item.value for item in app.markdown)
    app.selectbox[1].select(app.selectbox[1].options[-2]).run(timeout=90)
    assert not app.exception
    assert app.selectbox[2].value.strftime("%Y-%m-%d") != "2025-12-30"
    app.radio[0].set_value("24小时序列输入").run(timeout=90)
    assert not app.exception
    assert any("上传数据后" in item.value for item in app.info)
    app.radio[0].set_value("24小时多站点序列输入").run(timeout=90)
    assert not app.exception
    assert any("上传各站点数据后" in item.value for item in app.info)
    app.radio[0].set_value("手动输入").run(timeout=90)
    assert not app.exception
    assert len(app.number_input) == 12
    app.button[0].click().run(timeout=90)
    assert not app.exception
    assert any("一个小时" in item.value for item in app.info)
    app.radio[0].set_value("历史回放").run(timeout=90)
    assert not app.exception
