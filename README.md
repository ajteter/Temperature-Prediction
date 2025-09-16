# GISTEMP Anomaly Forecaster / 全球气温异常预测器

## Description / 项目描述

This project provides a scientific forecasting tool for predicting the Global Land-Ocean Temperature Index (GISTEMP) anomaly. It leverages a sophisticated, adaptive time-series modeling approach using Python to deliver a robust, probabilistic forecast.

The system automatically fetches and processes historical data for both the GISTEMP anomaly and the El Niño-Southern Oscillation (ENSO) index, treating ENSO as a critical external predictor. The final output is not just a single prediction, but a complete probability distribution across user-defined temperature bins.

本项目是一个用于预测全球陆地-海洋温度指数（GISTEMP）异常值的科学预测工具。它利用一套复杂的、自适应的时间序列建模方法，通过Python提供稳健的、概率化的预测。

该系统会自动获取并处理GISTEMP温度异常和ENSO（厄尔尼诺-南方涛动）指数的历史数据，并将ENSO作为关键的外部预测因子。最终的输出不仅是一个单一的预测值，而是一个覆盖用户自定义温度区间的完整概率分布。

---

## Data Sources / 数据来源

This project utilizes two primary data sources:
本项目利用了两个主要数据来源：

1.  **GISTEMP v4:** The Global Land-Ocean Temperature Index.
    - **Provider:** NASA's Goddard Institute for Space Studies (GISS).
    - **Usage:** Provided locally as `GLB.Ts+dSST.txt`.
    - **Original Source:** [https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.txt](https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.txt)

2.  **ENSO (Niño 3.4 Index):** Monthly sea surface temperature anomalies.
    - **Provider:** NOAA's Climate Prediction Center (CPC).
    - **Usage:** Fetched automatically by the script from the URL specified in `config.py`.
    - **Source:** [https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/detrend.nino34.ascii.txt](https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/detrend.nino34.ascii.txt)

---

## Core Workflow: The Adaptive Forecasting Strategy / 核心流程：自适应预测策略

The script implements a two-stage adaptive forecasting system to ensure the final prediction is based on the most effective model configuration for the current climate context.

脚本实现了一个两阶段的自适应预测系统，以确保最终预测是基于对当前气候背景最有效的模型配置。

### Part 1: The "Horse Race" Experiment / 第一部分：“赛马”实验

The system first runs a series of backtests to find the optimal training period. It tests several historical cutoff points (e.g., -18, -21, -24, -27, -30 months from the target date) and calculates the Root Mean Squared Error (RMSE) for each. This process identifies the model configuration that shows the best predictive performance on recent data.

系统首先会运行一系列的回测，以寻找最优的训练周期。它会测试多个历史数据截止点（例如，距离目标日期-18、-21、-24、-27、-30个月），并为每个场景计算均方根误差（RMSE）。这个过程旨在识别出在近期数据上表现出最佳预测性能的模型配置。

### Part 2: The "Champion" Forecast / 第二部分：“冠军”预测

The system then selects the "champion" model—the one with the lowest RMSE from the horse race. It uses this champion's specific configuration (its training period and model parameters) to perform the final, definitive forecast for the target month. This ensures the final prediction is based on the most effective and empirically validated modeling strategy.

然后，系统会选出“赛马”中的“冠军”——即RMSE最低的那个模型。它将使用这个冠军模型的特定配置（包括其训练周期和在那个周期上找到的最佳模型参数）来执行最终的、决定性的目标月份预测。这确保了最终的预测是基于经验上被验证为最有效的建模策略。

---

## Setup and Usage / 安装与使用

### 1. Prerequisites / 环境要求
- Python 3.8+

### 2. Installation / 安装
Clone the repository and install the required packages:
克隆本仓库并安装所需依赖包：
```bash
git clone <repository_url>
cd Temperature-Prediction
pip install -r requirements.txt
```

### 3. Configuration / 配置
Modify the `config.py` file to set your desired forecast target and prediction bins.
修改 `config.py` 文件以设置您希望预测的目标年月和结果的分类区间。
```python
# 1. Target year and month to predict, format: YYYYMM
# 1. 需要预测的年月，格式：YYYYMM
TARGET_YYYYMM = "202509"

# ... (other configurations) ...

# 5. Definition of prediction bins (unit: 0.01 degrees Celsius)
# 5. 预测区间的定义 (单位: 0.01摄氏度)
PREDICTION_BINS = {
    '<100': (None, 99),
    '100-104': (100, 104),
    # ... add more bins as needed ...
    '>119': (119, None),
}
```

### 4. Execution / 执行
Run the main script from the project's root directory:
在项目根目录运行主脚本：
```bash
python main.py
```

---

## Understanding the Output / 理解输出结果

The script will perform all steps automatically and print a detailed report.

脚本会自动执行所有步骤并打印详细的报告。

1.  **Data Preparation:** It will first show the status of data loading and merging.
    **数据准备：** 首先会显示数据加载和合并的状态。

2.  **The "Horse Race":** It will then run the backtesting experiments for each defined cutoff point and output the RMSE for each.
    **“赛马”实验：** 接着，它会运行每个分割点的回测实验，并输出各自的RMSE。

3.  **The "Champion" Announcement:** A summary table will be displayed, and the best-performing model configuration will be declared the "champion".
    **“冠军”揭晓：** 一个总结表会显示“赛马”的结果，并宣布表现最佳的模型为“冠军”。

4.  **Final Forecast:** Finally, it will use the champion model to perform the definitive forecast, showing the dynamically predicted ENSO value it used, the final predicted mean temperature anomaly, and the full probability distribution across your defined bins.
    **最终预测：** 最后，它会使用冠军模型执行最终预测，显示其动态推算出的ENSO值、最终预测的期望均值，以及覆盖您所定义区间的一个完整概率分布报告。

---

## Example Output / 输出示例

```bash
--- 数据准备阶段 (Data Preparation Stage) ---
所有训练和测试将基于1970年之后的数据 (All training and testing will be based on data since 1970). 范围 (Range): 1970-01 to 2025-08

--- [第一部分 / Part 1] 开始执行“赛马”实验，寻找最优训练周期 (Starting "Horse Race" to find the optimal training period) ---

--- 正在执行 (Executing): -18个月 赛道 (Lane) ---
      训练集截止 (Train End): 2024-03, 测试集 (Test Set): 2024-04 to 2025-08
      ...完成 (Completed)。RMSE: 10.13

--- 正在执行 (Executing): -21个月 赛道 (Lane) ---
      训练集截止 (Train End): 2023-12, 测试集 (Test Set): 2024-01 to 2025-08
      ...完成 (Completed)。RMSE: 9.86

--- 正在执行 (Executing): -24个月 赛道 (Lane) ---
      训练集截止 (Train End): 2023-09, 测试集 (Test Set): 2023-10 to 2025-08
      ...完成 (Completed)。RMSE: 8.78

--- 正在执行 (Executing): -27个月 赛道 (Lane) ---
      训练集截止 (Train End): 2023-06, 测试集 (Test Set): 2023-07 to 2025-08
      ...完成 (Completed)。RMSE: 22.09

--- 正在执行 (Executing): -30个月 赛道 (Lane) ---
      训练集截止 (Train End): 2023-03, 测试集 (Test Set): 2023-04 to 2025-08
      ...完成 (Completed)。RMSE: 21.90

--- [第二部分 / Part 2] “赛马”实验完成，执行最终预测 (Horse Race complete, executing final forecast) ---

--- “赛马”结果总结 (Horse Race Results Summary) ---
回测偏移 (Offset) | RMSE   | 胜出? (Champion?)
-----------------|--------|-----------------
-18              | 10.13  |
-21              | 9.86   |
-24              | 8.78   |  * 
-27              | 22.09  |
-30              | 21.90  |

冠军模型 (Champion Model): -24个月的训练周期 (training period), 训练至 (trained until) 2023-09).
现在将使用此‘冠军’配置进行最终预测 (Now using this 'champion' configuration for the final forecast)...

正在为ENSO数据本身建立预测模型 (Building forecast model for ENSO data itself)...
      ...动态预测出未来 1 个月的ENSO异常值为 (Dynamically forecasted ENSO anomaly for the next 1 month(s) is): -0.37

正在使用冠军配置训练最终模型并进行预测 (Training and forecasting with champion configuration)...

--- 最终预测结果分析 (冠军模型) / Final Forecast Analysis (Champion Model) ---
> 目标月份 (Target Month): 202509
> 预测的期望值 (Predicted Mean): 123.73

--- 模型预测概率分布 (Model's Predicted Probability Distribution) ---
  - 区间 (Bin) '<100': 1.04%
  - 区间 (Bin) '100-104': 1.93%
  - 区间 (Bin) '105-109': 4.43%
  - 区间 (Bin) '110-114': 8.19%
  - 区间 (Bin) '115-119': 12.21%
  - 区间 (Bin) '>119': 67.08%
```
