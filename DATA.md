# 数据说明

## 常用文件

| 路径 | 内容与用途 |
| --- | --- |
| `data/processed/regional_air_models_full_history/city_hourly.parquet` | 成都城市小时观测，包含 AQI、PM2.5、PM10 等；用于查看连续变化和网页历史曲线 |
| `data/processed/regional_air_models_full_history/station_hourly.parquet` | 分监测站的小时观测，用于空间与多站点研究 |
| `data/processed/regional_air_models_full_history/station_metadata.csv` | 站点标识和位置等元数据 |
| `data/processed/regional_air_models_full_history/weather_proxy.parquet` | 独立的历史气象代理数据，用于离线天气研究 |
| `artifacts/strict_history_backtest/predictions.parquet` | 历史起点对应的预测归档，不是原始污染物观测 |
| `artifacts/cnemc_2025_deployment_candidate/model_metadata.json` | 普通网页模型的输入特征、训练信息与模型配置 |
| `project-showcase/rebound-cases.json` | 反弹案例的预测和观测曲线 |
| `project-showcase/weather-inputs.json` | 多模态案例实际使用的 MERRA-2 气象网格，可用于绘制输入气象场 |

城市与站点小时表的时间轴覆盖 **2014-05-13 至 2026-09-03**，按北京时间 `Asia/Shanghai` 对齐。2014 年和 2026 年是部分年份；缺测保留为空值，不能视为零污染。时间轴覆盖与实际有效观测覆盖是两个概念。

城市数据按各污染物在可用站点间的中位数聚合。其中的城市 AQI 是站点 AQI 的中位数，不应标注为官方发布的全市 AQI。臭氧的类型与单位应按字段及相应处理代码确认，不把不同平均时长的臭氧指标混用。

## 读取 Parquet

Parquet 是保留数据类型、压缩存储的表格格式。依赖安装完成后，在项目根目录可用 pandas 读取：

```python
from pathlib import Path

import pandas as pd

root = Path.cwd()
path = root / "data/processed/regional_air_models_full_history/city_hourly.parquet"
data = pd.read_parquet(path)

print(data.columns.tolist())
print(data[["timestamp", "aqi", "pm2_5", "pm10"]].head())
print(data["timestamp"].min(), data["timestamp"].max())
print(data[["aqi", "pm2_5", "pm10"]].isna().sum())
```

需要 Excel 查看时可导出所需列或日期，不必展开全部训练数据：

```python
columns = ["timestamp", "aqi", "pm2_5", "pm10"]
data[columns].to_csv("city_pollution.csv", index=False, encoding="utf-8-sig")
```

AQI 无单位；PM2.5、PM10、SO2、NO2 与相应臭氧浓度通常使用微克每立方米，CO 使用毫克每立方米。导入序列应以操作页提供的字段和模板为准，不自行替换字段名称或单位。

## 来源与使用条件

污染物历史来自多个来源，不是一个统一许可的数据集。

1. **Figshare 数据集**：[Air pollutant and meteorological data of five provinces in China](https://figshare.com/articles/dataset/Air_pollutant_and_meteorological_data_of_five_provinces_in_China/24152163)。本项目原始来源清单记录其下载文件为 `data.rar`，下载地址为 [Figshare 文件 42374340](https://ndownloader.figshare.com/files/42374340)，记录的许可为 CC BY 4.0。该记录只对应这一来源，不自动覆盖后续加入的监测数据和气象数据。
2. **CNEMC 历史站点观测**：项目包含中国环境监测总站来源的历史空气质量资料。2014 年扩展使用 [Zenodo 记录 10836299](https://zenodo.org/records/10836299)；近期日数据通过 [quotsoft 空气质量历史镜像](https://quotsoft.net/air/) 获取，下载代码使用 `https://quotsoft.net/air/data/china_sites_YYYYMMDD.csv`。镜像下载地址不等于取得新的再分发许可，使用时应保留原始来源并核查相应条款。
3. **历史气象代理**：气象代理构建代码使用 Open-Meteo 历史接口，并将其作为离线再分析代理。它不是当时监测站气象观测，也不能直接作为历史实时可得数据的证据。
4. **多模态气象网格**：展示输入来自 MERRA-2 再分析。网页中的网格图是模型输入数值的可视化，不是卫星自然色照片；已提取的空间特征用于固定历史案例的融合预测。

本仓库不为所有合并数据统一声明 CC BY 许可。对外再发布或用于其他项目时，应分别核查各来源的数据使用条件；代码许可也不自动授予第三方数据的使用权。CNEMC 镜像、气象数据及第三方模型的授权范围，以相应提供方的条款为准。

## 结果解释

观测表、训练标签和预测表分别表示已经发生的数值、事后确认的恢复时间以及模型估计，三者不能互相替代。历史回测按起点与年份解释；手动输入没有对应的事后实测结果，不能把回测误差当成该次输入的已验证误差。

多模态网页归档覆盖 2020 年的 25 个起点，采用 68 小时封顶目标。LightGBM 反弹案例的未来 24 小时曲线与 AFT 的剩余时间预测不是同一个输出，评估时应分别说明目标、样本与时间范围。
