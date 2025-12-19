import pandas as pd
import numpy as np
import data_handler
import config

# 加载数据
gistemp_data = data_handler.get_clean_data(config.DATA_FILE_PATH)
enso_data = data_handler.get_enso_data(config.ENSO_DATA_URL)
full_df = pd.concat([gistemp_data, enso_data], axis=1)
full_df.dropna(inplace=True)

df_modern = full_df.loc["1970-01-01":].copy()

# 添加月份信息
df_modern['Month'] = df_modern.index.month

# 按月份统计
print("=== 各月份温度异常值统计 (1970-至今) ===\n")
monthly_stats = df_modern.groupby('Month')['Anomaly'].agg(['mean', 'std', 'min', 'max', 'count'])
monthly_stats.index = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']
print(monthly_stats.round(2))

# 季节性分析
print("\n=== 季节性分组统计 ===\n")
season_map = {12: '冬', 1: '冬', 2: '冬', 3: '春', 4: '春', 5: '春', 
              6: '夏', 7: '夏', 8: '夏', 9: '秋', 10: '秋', 11: '秋'}
df_modern['Season'] = df_modern['Month'].map(season_map)
season_stats = df_modern.groupby('Season')['Anomaly'].agg(['mean', 'std'])
print(season_stats.round(2))

# 分析ENSO与温度的季节性相关
print("\n=== ENSO与温度异常的季节性相关系数 ===\n")
for month in range(1, 13):
    month_data = df_modern[df_modern['Month'] == month]
    if len(month_data) > 10:
        corr = month_data['ENSO_ANOM'].corr(month_data['Anomaly'])
        print(f"{month:>2}月: {corr:>6.3f}")

# 近期趋势分析（2020年后）
print("\n=== 近期各月份平均值 (2020-至今) ===\n")
recent = df_modern.loc["2020-01-01":]
recent_monthly = recent.groupby('Month')['Anomaly'].mean()
for month in range(1, 13):
    if month in recent_monthly.index:
        print(f"{month:>2}月: {recent_monthly[month]:>6.2f}")
