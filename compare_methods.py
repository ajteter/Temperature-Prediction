import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta

import config
import data_handler
import model

print("="*60)
print("对比：全局RMSE vs 季节性RMSE")
print("="*60)

# 加载数据
gistemp_data = data_handler.get_clean_data(config.DATA_FILE_PATH)
enso_data = data_handler.get_enso_data(config.ENSO_DATA_URL)
full_df = pd.concat([gistemp_data, enso_data], axis=1)
full_df.dropna(inplace=True)
df_modern = full_df.loc["1970-01-01":].copy()

target_date = pd.to_datetime(config.TARGET_YYYYMM, format='%Y%m')
target_month = target_date.month

print(f"\n目标：预测 {target_date.strftime('%Y年%m月')} (第{target_month}月)\n")

# 以-24月窗口为例
months = 24
split_date = target_date - relativedelta(months=months)

train_df = df_modern.loc[:split_date]
test_df = df_modern.loc[split_date + relativedelta(months=1):df_modern.index.max()]

print(f"训练截止：{split_date.strftime('%Y-%m')}")
print(f"测试期间：{test_df.index.min().strftime('%Y-%m')} 到 {test_df.index.max().strftime('%Y-%m')}")
print(f"测试集共 {len(test_df)} 个月\n")

# 训练模型
train_endog = train_df['Anomaly']
train_exog = train_df[['ENSO_ANOM']]
test_endog = test_df['Anomaly']
test_exog = test_df[['ENSO_ANOM']]

best_order, best_seasonal_order = model.find_best_sarimax_params(train_endog, train_exog)
mod = sm.tsa.statespace.SARIMAX(train_endog, exog=train_exog, order=best_order, seasonal_order=best_seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
fit_results = mod.fit(disp=False)
forecast_results = fit_results.get_forecast(steps=len(test_df), exog=test_exog)

predictions = pd.Series(forecast_results.predicted_mean.values, index=test_df.index)

# 计算全局RMSE
rmse_all = np.sqrt(mean_squared_error(test_endog, predictions))

# 提取11月的数据
test_nov = test_df[test_df.index.month == target_month]
pred_nov = predictions[predictions.index.month == target_month]

print("="*60)
print("测试集中的11月数据：")
print("="*60)
print("日期      | 实际值 | 预测值 | 误差")
print("----------|--------|--------|-------")
for date in test_nov.index:
    actual = test_nov.loc[date, 'Anomaly']
    pred = pred_nov.loc[date]
    error = pred - actual
    print(f"{date.strftime('%Y-%m')} | {actual:>6.2f} | {pred:>6.2f} | {error:>+6.2f}")

rmse_nov = np.sqrt(mean_squared_error(test_nov['Anomaly'], pred_nov))

print("\n" + "="*60)
print("RMSE对比：")
print("="*60)
print(f"RMSE(全部{len(test_df)}个月)：{rmse_all:.2f}")
print(f"RMSE(仅{len(test_nov)}个11月)：{rmse_nov:.2f}")
print("\n" + "="*60)
print("结论：")
print("="*60)
print(f"• 全局RMSE看：这个模型表现{'好' if rmse_all < 12 else '一般'} (RMSE={rmse_all:.2f})")
print(f"• 季节性RMSE看：这个模型对11月预测{'非常准确' if rmse_nov < 5 else '较准确' if rmse_nov < 10 else '不够准确'} (RMSE={rmse_nov:.2f})")
print(f"\n如果目标是预测11月，应该优先选择RMSE(11月)最小的模型！")
