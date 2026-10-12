import pandas as pd

from haze_forecast.app import build_trend_figure, prepare_trend_context


def test_single_observation_has_visible_marker_and_full_day_axis():
    now = pd.Timestamp('2025-04-12 16:00', tz='Asia/Shanghai')
    frame = pd.DataFrame({'timestamp': [now], 'aqi': [274], 'pm2_5': [220], 'pm10': [400]})
    context = prepare_trend_context(frame, now)
    figure = build_trend_figure(context, now, 100)
    assert len(context) == 25
    assert context.aqi.notna().sum() == 1
    assert all(trace.mode == 'lines+markers' for trace in figure.data)
    assert pd.Timestamp(figure.layout.xaxis.range[1]) - pd.Timestamp(figure.layout.xaxis.range[0]) == pd.Timedelta(hours=24)
    assert pd.Timestamp(figure.layout.xaxis.range[1]).hour == 16
    assert figure.layout.font.color == '#22352b'
    assert figure.layout.xaxis.tickfont.color == '#22352b'
    assert figure.layout.legend.font.color == '#22352b'


def test_context_preserves_missing_hours_and_never_uses_future():
    now = pd.Timestamp('2025-12-30 00:00', tz='Asia/Shanghai')
    frame = pd.DataFrame({
        'timestamp': [now - pd.Timedelta(hours=2), now, now + pd.Timedelta(hours=1)],
        'aqi': [130, 129, 999], 'pm2_5': [100, 98, 999], 'pm10': [128, 126, 999],
    })
    context = prepare_trend_context(frame, now)
    assert context.aqi.notna().sum() == 2
    assert context.iloc[-2].aqi != context.iloc[-2].aqi
    assert context.iloc[-1].aqi == 129
    assert context.timestamp.max() == now
    figure = build_trend_figure(context, now, None)
    assert all(not trace.connectgaps for trace in figure.data)
    assert len(figure.layout.shapes) == 0
    assert figure.data[0].yaxis == 'y'
    assert figure.data[1].yaxis == 'y2'


def test_upload_local_timestamps_and_empty_context_are_supported():
    now = pd.Timestamp('2025-01-01 00:00', tz='Asia/Shanghai')
    frame = pd.DataFrame({'timestamp': ['2025-01-01 00:00'], 'aqi': [110], 'pm2_5': [80], 'pm10': [100]})
    assert prepare_trend_context(frame, now).iloc[-1].aqi == 110
    assert prepare_trend_context(frame.iloc[:0], now).aqi.isna().all()
