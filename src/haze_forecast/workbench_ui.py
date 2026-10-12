"""Presentation-only observation workbench; model calculations stay in app.py."""

from html import escape

import pandas as pd
import streamlit as st


def apply_style():
    from pathlib import Path

    css = Path(__file__).with_name("workbench.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_header():
    with st.container(key="appnav"):
        brand, navigation = st.columns([1.15, 3])
        brand.markdown("### 成都雾霾消散预测")
        with navigation:
            return st.radio(
                "预测方式",
                ["严格历史回测", "历史回放", "手动输入", "24小时序列输入", "24小时多站点序列输入"],
                horizontal=True, label_visibility="collapsed",
            )


def result_line(label, value, *, date=False, primary=False):
    css = "result-value primary" if primary else "result-value date" if date else "result-value"
    st.markdown(f'<div class="result-line"><div class="result-label">{escape(label)}</div>'
                f'<div class="{css}">{escape(value)}</div></div>', unsafe_allow_html=True)


def render_results(*, row, context, figure, mode, prediction, dissipates_at, band_text,
                   band_caption, actual_text, error, error_label, criterion, observed_count,
                   particle, pollutants, is_pollution_event, download_slot):
    historical = mode in {"严格历史回测", "历史回放"}
    note = ("当前为严格历史回测结果，用于历史数据分析与验证。" if mode == "严格历史回测"
            else "当前为历史回放，单次差异不代表样本外预测精度。" if historical
            else "当前为输入观测的模型估计，并非实时监测数据。")
    st.markdown(f'<div class="status-note">{note}</div>', unsafe_allow_html=True)
    with st.container(key="rail"):
        st.markdown('[:material/monitoring:](#observations "观测趋势")')
        st.markdown('[:material/table_chart:](#readings "污染物数据")')
        st.markdown('[:material/description:](#criteria "消散判定")')
    left, right = st.columns([2.45, 1], gap="small")
    with left, st.container(key="observation"):
        st.markdown('<div id="observations"></div>', unsafe_allow_html=True)
        st.subheader("过去 24 小时污染物观测")
        st.caption(f"{context['timestamp'].iloc[0]:%Y-%m-%d %H:%M} 至 {row['timestamp']:%Y-%m-%d %H:%M} · 北京时间")
        if observed_count:
            figure.update_layout(title=None, height=370, margin={"l": 44, "r": 44, "t": 48, "b": 48},
                                 legend={"y": 1.15, "font": {"size": 12}},
                                 font={"color": "#43546b", "size": 12})
            figure.update_xaxes(showgrid=True, gridcolor="#e8edf3")
            figure.update_yaxes(gridcolor="#e8edf3")
            for trace, color in zip(figure.data, ["#087f8c", "#2c7bd6", "#bc780d"], strict=True):
                trace.line.color = color
                trace.line.width = 2
                trace.marker.size = 4
            st.plotly_chart(figure, width="stretch", theme=None,
                            config={"displaylogo": False, "responsive": True})
            st.markdown(series_summary(context), unsafe_allow_html=True)
            st.caption(f"观测完整度 {observed_count}/25 · AQI 左轴 / 颗粒物右轴 · 颗粒物判定：{particle}")
            if observed_count == 1:
                st.info("当前只有一个小时的观测，尚不足以判断变化趋势。")
        else:
            st.info("该时刻之前24小时没有可显示的污染物观测。")
    with right, st.container(key="prediction"):
        st.subheader("预测结果（历史回测）" if mode == "严格历史回测" else "预测结果（历史回放）" if historical else "预测结果")
        if is_pollution_event:
            result_line("预计剩余消散时间", f"{prediction:.1f} 小时", primary=True)
            result_line("预计消散时刻", dissipates_at.strftime("%Y-%m-%d %H:%M"), date=True)
            result_line("预测区间", band_text)
            st.caption(band_caption)
        else:
            st.info("当前未判定为颗粒物污染过程。")
        if error is not None:
            with st.container(key="backtest"):
                st.markdown("#### 回测对照")
                first, second = st.columns(2)
                first.metric("历史真实剩余时间", actual_text)
                second.metric(error_label, f"{error:.1f} 小时")
    with st.container(key="readings"):
        st.markdown('<div id="readings"></div>', unsafe_allow_html=True)
        st.subheader(f"当前污染物数据（{row['timestamp']:%Y-%m-%d %H:%M}）")
        st.markdown(readings_table(pollutants.to_dict("records")), unsafe_allow_html=True)
    with download_slot:
        st.caption("观测数据")
        st.download_button("下载观测数据", context.to_csv(index=False).encode("utf-8-sig"),
                           "observations.csv", "text/csv", icon=":material/download:")
    st.markdown('<div id="criteria"></div>', unsafe_allow_html=True)
    st.caption(f"消散判定规则：{criterion.detail}")


def series_summary(context):
    """Summarize only the displayed causal observations, never fill missing data."""
    items = []
    for key, label, unit in [("aqi", "AQI", "指数"), ("pm2_5", "PM2.5", "µg/m³"),
                             ("pm10", "PM10", "µg/m³")]:
        values = context[key]
        stats = [values.iloc[-1], values.min(), values.mean(), values.max()]
        cells = "".join(f'<div><dt>{name}</dt><dd>{value:.1f}</dd></div>'
                        if pd.notna(value) else f'<div><dt>{name}</dt><dd>暂无</dd></div>'
                        for name, value in zip(["当前", "最小", "均值", "最大"], stats, strict=True))
        items.append(f'<div class="series-item"><strong>{label}</strong> <small>{unit}</small>'
                     f'<dl>{cells}</dl></div>')
    return '<div class="series-summary">' + "".join(items) + '</div>'


def readings_table(records):
    """Render existing measured values without changing units or classifying them."""
    headings = "".join(f'<th scope="col">{escape(item["污染物"])}'
                       f'<small>{escape(item["单位"]) or "指数"}</small></th>' for item in records)
    values = "".join(f'<td>{item["数值"]:g}</td>' for item in records)
    return ('<div class="readings-scroll" role="region" aria-label="当前污染物数据" tabindex="0">'
            f'<table class="reading-table"><thead><tr>{headings}</tr></thead>'
            f'<tbody><tr>{values}</tr></tbody></table></div>')
