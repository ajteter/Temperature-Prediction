# 预测配置 (Statsmodels 增强版)
# ==========================================
# 继承 config_robust 的基础设置，新增 statsmodels 增强配置

# --- 基础配置 (继承自 config_robust) ---
TARGET_YYYYMM = "202603"
DATA_FILE_PATH = "GLB.Ts+dSST.txt"
ENSO_DATA_URL = "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/detrend.nino34.ascii.txt"
RACE_SPLIT_MONTHS = [18, 24, 30, 36]
OPTIMAL_ENSO_LAG = 3

PREDICTION_BINS = {
    '<110': (None, 105),
    '110-114': (110, 114),
    '115-119': (115, 119),
    '120-124': (120, 124),
    '125-129': (120, 124), 
    '>129': (124, None),
}

MAINSTREAM_PROBS = {
    '<110': 0.07,
    '110-114': 0.14,
    '115-119': 0.29,
    '120-124': 0.30,
    '125-129': 0.15, 
    '>129': 0.06,
}

# --- A. 混合评估策略 (继承) ---
USE_HYBRID_EVALUATION = True
SEASONAL_WEIGHT_BASE = 0.7
SEASONAL_WEIGHT_DECAY = {0: 0.0, 1: 0.3, 2: 0.5}
SEASONAL_WEIGHT_MIN_SAMPLES_FOR_BASE = 3

# --- B. 趋势调整 (继承) ---
USE_TREND_ADJUSTMENT = True
TREND_ACCELERATION_THRESHOLD = 1.5
TREND_ADJUSTMENT_WEIGHT = 0.15
TREND_LOOKBACK_SHORT = 6
TREND_LOOKBACK_LONG = 12

# --- C. 偏差修正 (继承) ---
USE_BIAS_CORRECTION = True
BIAS_LOOKBACK_MONTHS = 3

# ==========================================
# Statsmodels 增强配置
# ==========================================

# --- D. 平稳性诊断 ---
PRINT_STATIONARITY_DIAGNOSTICS = True  # 是否在启动时打印 ADF/KPSS 诊断

# --- E. 多算法竞赛 ---
USE_ETS_CANDIDATE = True  # 是否将 ETS (Holt-Winters) 纳入赛马竞赛
ETS_SEASONAL_PERIODS = 12  # ETS 的季节周期

# --- F. 残差诊断 ---
PRINT_RESIDUAL_DIAGNOSTICS = True  # 是否为每个入选模型打印 Ljung-Box/Jarque-Bera 结果
LJUNGBOX_LAGS = 10  # Ljung-Box 检验的滞后阶数

# --- G. 模型比较表 ---
PRINT_MODEL_COMPARISON_TABLE = True  # 是否输出结构化 AIC/BIC 对比表

# --- H. 集成 Top-N ---
ENSEMBLE_TOP_N = 4  # 从赛马中选入集成的模型数量
