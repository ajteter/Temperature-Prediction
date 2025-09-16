
# 预测配置
# ==========================================

# 1. 需要预测的年月，格式：YYYYMM
TARGET_YYYYMM = "202509"

# 2. 数据文件的本地路径
# https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.txt
DATA_FILE_PATH = "GLB.Ts+dSST.txt"

# 3. ENSO(厄尔尼诺)数据源地址
ENSO_DATA_URL = "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/detrend.nino34.ascii.txt"



# 5. 预测区间的定义 (单位: 0.01摄氏度)
#    字典的键是区间的名称，值是一个元组 (下限, 上限)
#    None 表示没有边界
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
    '<100': 0.006,
    '100-104': 0.015,
    '105-109': 0.014,
    '110-114': 0.14,
    '115-119': 0.41,
    '>119': 0.49,
}
