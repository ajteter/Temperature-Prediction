# 预测配置 (改进版 - Robust)
# ==========================================

# 1. 预测目标
TARGET_YYYYMM = "202601"

# 2. 数据文件的本地路径
DATA_FILE_PATH = "GLB.Ts+dSST.txt"

# 3. ENSO(厄尔尼诺)数据源地址
ENSO_DATA_URL = "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/detrend.nino34.ascii.txt"

# 4. 赛马实验的训练周期选项（单位：月）
# 优化：仅保留最近3年的窗口 (关键年份)，加快验证速度
RACE_SPLIT_MONTHS = [18, 24, 30, 36] 

# 5. ENSO 滞后月数
OPTIMAL_ENSO_LAG = 3 

# 5. 预测区间的定义 (单位: 0.01摄氏度)
PREDICTION_BINS = {
    '<100': (None, 99),
    '100-104': (100, 104),
    '105-109': (105, 109),
    '110-114': (110, 114),
    '115-119': (115, 119),
    '>119': (119, None),
}

# 6. 主流观点对各区间的概率预测
MAINSTREAM_PROBS = {
    '<100': 0.01,
    '100-104': 0.01,
    '105-109': 0.54,
    '110-114': 0.37,
    '115-119': 0.05,
    '>119': 0.02,
}

# ==========================================
# 改进方案配置 (Robust Configuration)
# ==========================================

# --- A. 混合评估策略 ---
USE_HYBRID_EVALUATION = True  # 是否启用混合RMSE计算

# 基础季节性权重 (当样本充足时，多大程度上依赖同月表现)
# 0.7 表示 70% 看同月RMSE，30% 看全局RMSE
SEASONAL_WEIGHT_BASE = 0.7 

# 权重衰减规则：根据验证集中的同月样本数量动态调整季节性权重
# 样本少 -> 降低季节性权重，更多依赖全局表现
SEASONAL_WEIGHT_DECAY = {
    0: 0.0, # 无样本（通常会跳过，但作为兜底）
    1: 0.3, # 只有1个同月样本，季节性权重降至 30%
    2: 0.5, # 只有2个同月样本，季节性权重降至 50%
    # >=3 个样本时使用 SEASONAL_WEIGHT_BASE (0.7)
}
SEASONAL_WEIGHT_MIN_SAMPLES_FOR_BASE = 3

# --- B. 近期趋势调整 ---
USE_TREND_ADJUSTMENT = True   # 是否启用趋势调整

# 趋势加速阈值：如果 6个月趋势斜率 > 12个月趋势斜率 * 1.5，则认为趋势加速
TREND_ACCELERATION_THRESHOLD = 1.5 

# 调整权重：将计算出的额外趋势增量乘以此系数加到最终结果中
# 防止过度调整
TREND_ADJUSTMENT_WEIGHT = 0.15 

# 趋势计算的回溯窗口（月）
TREND_LOOKBACK_SHORT = 6
TREND_LOOKBACK_LONG = 12

# --- C. 残差偏差修正 (Residual Bias Correction) ---
USE_BIAS_CORRECTION = True # 是否启用偏差修正（通过近期误差校准未来）
BIAS_LOOKBACK_MONTHS = 6   # 计算偏差的回溯窗口（月）
