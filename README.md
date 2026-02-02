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

## Recent Updates: The "Robust" Methodology (Nov 2025) / 最新更新：“稳健”方法论 (2025.11)

Based on the findings in `20251128.md` and `validation_report.md`, the project has evolved to a "Robust Edition" (`main_robust.py`) to address key physical and statistical disconnects found in earlier versions.

基于 `20251128.md` 和 `validation_report.md` 的发现，本项目已升级为“稳健版” (`main_robust.py`)，以解决早期版本中发现的关键物理和统计脱节问题。

### Key Improvements / 关键改进

1.  **Physics-Informed Lag (Lag-3) / 基于物理的滞后 (Lag-3)**
    *   **Problem:** Previous models used "same-month" ENSO data, ignoring the thermal inertia required for Pacific sea surface temperatures to affect global atmospheric temperatures. This caused a ~3-month lag in predicting sudden climate shifts.
    *   **Solution:** Analysis (`analyze_lags.py`) confirmed a max correlation at **3 months**. The model now correctly uses ENSO signals from 3 months prior to predict the current month's temperature.
    *   **问题：** 之前的模型使用“当月”ENSO数据，忽略了太平洋海温影响全球大气温度所需的热惯性，导致对气候突变的预测滞后约3个月。
    *   **方案：** 分析确认相关性峰值在 **3个月**。模型现在正确使用3个月前的ENSO信号来预测当月气温。

2.  **Critical Scaling Fix / 关键缩放修复**
    *   **Fix:** Corrected a data parsing error where raw GISTEMP values (0.01°C) were treated as full degrees Celsius. All calculations now meaningfully reflect real-world physical units (°C).
    *   **修复：** 修正了一个数据解析错误，该错误将原始GISTEMP值（0.01°C）视为摄氏度。现在的计算能正确反映真实的物理量级。

3.  **Hybrid Evaluation Strategy / 混合评估策略**
    *   **Innovation:** Instead of relying solely on global RMSE, the "Horse Race" now weights **Seasonal RMSE** (performance in the same target month) heavily (default 70%). This ensures the selected model is not just generally good, but specifically good for the season being predicted.
    *   **创新：** “赛马”不再仅依赖全局RMSE，而是高度加权 **季节性RMSE**（目标月份的性能，默认70%）。这确保了所选模型不仅整体表现好，而且特别适合预测当前的季节。

4.  **Residual Bias Correction / 残差偏差修正**
    *   **Feature:** The model now looks back at the last 6 months of forecast performance. If a systematic bias (over/under-prediction) is detected, it automatically calibrates the future forecast to compensate.
    *   **特性：** 模型现在会回溯过去6个月的预测表现。如果检测到系统性偏差（高估/低估），它会自动校准未来的预测以进行补偿。

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
### 3. Configuration / 配置
Modify the `config_robust.py` file to set your desired forecast target and prediction bins.
修改 `config_robust.py` 文件以设置您希望预测的目标年月和结果的分类区间。
```python
# 1. Target year and month to predict, format: YYYYMM
# 1. 需要预测的年月，格式：YYYYMM
TARGET_YYYYMM = "202601"

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
python main_robust.py
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

=== 气候温度预测系统: 稳健增强版 (Robust Edition) ===

ENSO Lag: 3 months
数据范围 (Model Train): 1970-01 to 2025-12

目标月份: 2026年01月 (第1月)
=== 阶段1: 混合评估赛马 (Hybrid Evaluation) ===
策略: 启用混合RMSE (季节性RMSE + 全局RMSE)
基础季节性权重: 0.7
------------------------------------------------------------
窗口 -18月 | 样本数:1 | 全局RMSE: 0.13 | 同月RMSE: 0.24 | 混合RMSE: 0.16
窗口 -24月 | 样本数:1 | 全局RMSE: 0.10 | 同月RMSE: 0.18 | 混合RMSE: 0.12
窗口 -30月 | 样本数:2 | 全局RMSE: 0.21 | 同月RMSE: 0.24 | 混合RMSE: 0.23
窗口 -36月 | 样本数:2 | 全局RMSE: 0.32 | 同月RMSE: 0.38 | 混合RMSE: 0.35

--- 选入集成的 Top 4 模型 ---
偏移 | 混合RMSE | 权重   
-----|----------|--------
-24月 |     0.12 |  50.8%
-18月 |     0.16 |  28.3%
-30月 |     0.23 |  14.7%
-36月 |     0.35 |   6.2%

=== 阶段2: 准备预测变量 (Steps=1) ===
利用最新ENSO数据进行智能拼接...

=== 阶段3.5: 残差偏差修正 (回溯 6 月) ===
计算偏差窗口: 2025-07 到 2025-12

=== 阶段3: 执行集成预测 & 偏差计算 ===
模型1 (-24月):   1.10 (SD:0.11) | 近期偏差(6月): +0.017
模型2 (-18月):   1.10 (SD:0.11) | 近期偏差(6月): +0.018
模型3 (-30月):   1.06 (SD:0.11) | 近期偏差(6月): +0.121
模型4 (-36月):   0.91 (SD:0.11) | 近期偏差(6月): +0.216

基础集成预测值: 1.08
加权残差偏差: +0.045

=== 阶段4: 趋势稳定性检查与调整 ===
短期趋势 (6月): 0.007/月
长期趋势 (12月): -0.017/月
✓ 趋势相对稳定，无需调整。

==================================================
=== 最终稳健预测结果 (Robust Forecast) ===
==================================================
目标月份: 202601
基础预测: 1.08
偏差修正: +0.04
最终预测: 1.13
95%置信区间: [0.92, 1.33]

--- 预测概率分布 ---
        <100: 10.27% █████
     100-104:  9.21% ████
     105-109: 13.04% ██████
     110-114: 14.86% ███████
     115-119: 13.63% ██████
        >119: 27.11% █████████████