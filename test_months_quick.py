import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta

import data_handler
import model

# 加载数据
print("加载数据...")
gistemp_data = data_handler.get_clean_data('GLB.Ts+dSST.txt')
enso_data = data_handler.get_enso_data('https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/detrend.nino34.ascii.txt')
full_df = pd.concat([gistemp_data, enso_data], axis=1)
full_df.dropna(inplace=True)
df_modern = full_df.loc["1970-01-01":].copy()

print("\n" + "="*70)
print("快速测试：季节性加权方法对不同月份的有效性")
print("="*70)
print("\n策略：只训练一次模型，然后分析对不同月份的预测准确度\n")

# 使用固定的训练窗口进行一次性测试
target_date = pd.to_datetime("202510", format='%Y%m')  # 使用2025-10作为基准
windows_to_test = [18, 21, 24]  # 只测试3个窗口以加快速度

print(f"测试窗口：{windows_to_test}")
print(f"基准日期：{target_date.strftime('%Y-%m')}\n")

# 存储每个窗口的预测结果
window_predictions = {}

for months in windows_to_test:
    print(f"训练 -{months}月窗口模型...")
    split_date = target_date - relativedelta(months=months)
    
    train_df = df_modern.loc[:split_date]
    test_df = df_modern.loc[split_date + relativedelta(months=1):df_modern.index.max()]
    
    train_endog = train_df['Anomaly']
    train_exog = train_df[['ENSO_ANOM']]
    test_endog = test_df['Anomaly']
    test_exog = test_df[['ENSO_ANOM']]
    
    best_order, best_seasonal_order = model.find_best_sarimax_params(train_endog, train_exog)
    mod = sm.tsa.statespace.SARIMAX(train_endog, exog=train_exog, order=best_order, 
                                     seasonal_order=best_seasonal_order, 
                                     enforce_stationarity=False, enforce_invertibility=False)
    fit_results = mod.fit(disp=False)
    forecast_results = fit_results.get_forecast(steps=len(test_df), exog=test_exog)
    
    predictions = pd.Series(forecast_results.predicted_mean.values, index=test_df.index)
    window_predictions[months] = {
        'predictions': predictions,
        'actuals': test_endog,
        'train_end': split_date
    }
    print(f"  完成。训练截止：{split_date.strftime('%Y-%m')}")

# 分析每个月份
print("\n" + "="*70)
print("各月份的最优窗口分析")
print("="*70)

month_names = ['1月', '2月', '3月', '4月', '5月', '6月', 
               '7月', '8月', '9月', '10月', '11月', '12月']

results_summary = []

for target_month in range(1, 13):
    month_rmse = {}
    month_samples = {}
    
    for months, data in window_predictions.items():
        # 筛选出该月份的数据
        mask = data['predictions'].index.month == target_month
        if mask.sum() == 0:
            continue
        
        pred_month = data['predictions'][mask]
        actual_month = data['actuals'][mask]
        
        rmse = np.sqrt(mean_squared_error(actual_month, pred_month))
        month_rmse[months] = rmse
        month_samples[months] = len(pred_month)
    
    if month_rmse:
        best_window = min(month_rmse.keys(), key=lambda x: month_rmse[x])
        worst_window = max(month_rmse.keys(), key=lambda x: month_rmse[x])
        
        improvement = (month_rmse[worst_window] - month_rmse[best_window]) / month_rmse[worst_window] * 100
        
        results_summary.append({
            'month': target_month,
            'best_window': best_window,
            'best_rmse': month_rmse[best_window],
            'worst_window': worst_window,
            'worst_rmse': month_rmse[worst_window],
            'improvement': improvement,
            'samples': month_samples[best_window]
        })

# 显示结果
print("\n月份 | 最优窗口 | 最优RMSE | 最差窗口 | 最差RMSE | 改进幅度 | 样本数")
print("-----|----------|----------|----------|----------|----------|-------")
for r in results_summary:
    print(f"{month_names[r['month']-1]:>4} | -{r['best_window']:>2}月    | {r['best_rmse']:>8.2f} | "
          f"-{r['worst_window']:>2}月    | {r['worst_rmse']:>8.2f} | {r['improvement']:>7.1f}% | {r['samples']:>6}")

# 统计
print("\n" + "="*70)
print("统计分析")
print("="*70)

different_choices = sum(1 for r in results_summary if r['best_window'] != r['worst_window'])
avg_improvement = np.mean([r['improvement'] for r in results_summary])
max_improvement = max([r['improvement'] for r in results_summary])
max_improvement_month = month_names[[r['improvement'] for r in results_summary].index(max_improvement)]

print(f"\n测试月份数：{len(results_summary)}")
print(f"不同窗口表现有差异的月份：{different_choices} ({different_choices/len(results_summary)*100:.1f}%)")
print(f"平均改进幅度：{avg_improvement:.1f}%")
print(f"最大改进幅度：{max_improvement:.1f}% ({max_improvement_month})")

# 窗口偏好统计
window_preference = {}
for r in results_summary:
    w = r['best_window']
    window_preference[w] = window_preference.get(w, 0) + 1

print(f"\n各窗口被选为最优的次数：")
for w in sorted(window_preference.keys()):
    count = window_preference[w]
    print(f"  -{w}月窗口：{count}次 ({count/len(results_summary)*100:.1f}%)")

print("\n" + "="*70)
print("结论")
print("="*70)
if avg_improvement > 10:
    print("✓ 季节性加权方法有效！不同月份确实需要不同的训练窗口")
    print(f"  选择针对性窗口平均可改进 {avg_improvement:.1f}%")
elif avg_improvement > 5:
    print("• 季节性加权方法有一定效果，部分月份受益明显")
else:
    print("• 不同窗口对各月份的表现差异较小")
