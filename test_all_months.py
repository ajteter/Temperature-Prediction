import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta

import data_handler
import model

# 加载数据
gistemp_data = data_handler.get_clean_data('GLB.Ts+dSST.txt')
enso_data = data_handler.get_enso_data('https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/detrend.nino34.ascii.txt')
full_df = pd.concat([gistemp_data, enso_data], axis=1)
full_df.dropna(inplace=True)
df_modern = full_df.loc["1970-01-01":].copy()

print("="*80)
print("测试季节性加权方法对12个月份的有效性")
print("="*80)
print("\n方法：对每个月份，测试不同窗口在该月份的历史预测准确度\n")

# 测试每个月份
results_by_month = {}

for target_month in range(1, 13):
    print(f"\n{'='*80}")
    print(f"目标月份：{target_month}月")
    print(f"{'='*80}")
    
    # 找一个该月份的目标日期（使用2025年）
    target_date = pd.to_datetime(f"2025{target_month:02d}", format='%Y%m')
    last_available_date = df_modern.index.max()
    
    # 如果目标日期超过当前数据，往前推一年
    if target_date > last_available_date:
        target_date = target_date - relativedelta(years=1)
    
    split_months_options = [12, 15, 18, 21, 24, 27, 30]
    month_results = []
    
    for months in split_months_options:
        split_date = target_date - relativedelta(months=months)
        
        if split_date < df_modern.index.min() + relativedelta(months=36):
            continue
            
        train_df = df_modern.loc[:split_date]
        test_df = df_modern.loc[split_date + relativedelta(months=1):last_available_date]
        
        if test_df.empty or len(train_df) < 36:
            continue
        
        # 只看目标月份的数据
        test_df_same_month = test_df[test_df.index.month == target_month]
        
        if len(test_df_same_month) == 0:
            continue
        
        train_endog = train_df['Anomaly']
        train_exog = train_df[['ENSO_ANOM']]
        test_endog = test_df['Anomaly']
        test_exog = test_df[['ENSO_ANOM']]
        
        try:
            best_order, best_seasonal_order = model.find_best_sarimax_params(train_endog, train_exog)
            mod = sm.tsa.statespace.SARIMAX(train_endog, exog=train_exog, order=best_order, seasonal_order=best_seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
            fit_results = mod.fit(disp=False)
            forecast_results = fit_results.get_forecast(steps=len(test_df), exog=test_exog)
            
            # 全部RMSE
            rmse_all = np.sqrt(mean_squared_error(test_endog, forecast_results.predicted_mean))
            
            # 同月RMSE
            predictions_all = pd.Series(forecast_results.predicted_mean.values, index=test_df.index)
            predictions_same_month = predictions_all[predictions_all.index.month == target_month]
            actuals_same_month = test_df_same_month['Anomaly']
            rmse_same_month = np.sqrt(mean_squared_error(actuals_same_month, predictions_same_month))
            
            month_results.append({
                'window': months,
                'rmse_all': rmse_all,
                'rmse_same_month': rmse_same_month,
                'sample_count': len(test_df_same_month)
            })
        except:
            continue
    
    if month_results:
        # 找出全局RMSE最小和季节性RMSE最小的窗口
        best_global = min(month_results, key=lambda x: x['rmse_all'])
        best_seasonal = min(month_results, key=lambda x: x['rmse_same_month'])
        
        print(f"\n窗口 | RMSE(全) | RMSE({target_month}月) | 样本数")
        print("-"*50)
        for r in sorted(month_results, key=lambda x: x['rmse_same_month']):
            mark_g = " ← 全局最优" if r['window'] == best_global['window'] else ""
            mark_s = " ← 季节最优" if r['window'] == best_seasonal['window'] else ""
            print(f"-{r['window']:>2}月 | {r['rmse_all']:>8.2f} | {r['rmse_same_month']:>13.2f} | {r['sample_count']:>6}{mark_g}{mark_s}")
        
        # 判断是否有差异
        same_choice = (best_global['window'] == best_seasonal['window'])
        improvement = ((best_global['rmse_same_month'] - best_seasonal['rmse_same_month']) / best_global['rmse_same_month'] * 100) if not same_choice else 0
        
        results_by_month[target_month] = {
            'best_global_window': best_global['window'],
            'best_seasonal_window': best_seasonal['window'],
            'same_choice': same_choice,
            'improvement': improvement,
            'global_rmse_on_month': best_global['rmse_same_month'],
            'seasonal_rmse_on_month': best_seasonal['rmse_same_month']
        }
        
        if same_choice:
            print(f"\n✓ 两种方法选择相同窗口：-{best_global['window']}月")
        else:
            print(f"\n✗ 两种方法选择不同：")
            print(f"  全局方法：-{best_global['window']}月 → 对{target_month}月RMSE={best_global['rmse_same_month']:.2f}")
            print(f"  季节方法：-{best_seasonal['window']}月 → 对{target_month}月RMSE={best_seasonal['rmse_same_month']:.2f}")
            print(f"  改进：{improvement:.1f}%")

# 总结
print("\n" + "="*80)
print("总结：季节性加权方法的有效性")
print("="*80)

different_count = sum(1 for r in results_by_month.values() if not r['same_choice'])
total_count = len(results_by_month)

print(f"\n测试月份数：{total_count}")
print(f"两种方法选择不同窗口的月份数：{different_count} ({different_count/total_count*100:.1f}%)")
print(f"两种方法选择相同窗口的月份数：{total_count - different_count} ({(total_count-different_count)/total_count*100:.1f}%)")

if different_count > 0:
    avg_improvement = np.mean([r['improvement'] for r in results_by_month.values() if not r['same_choice']])
    print(f"\n当选择不同时，季节性方法平均改进：{avg_improvement:.1f}%")
    
    print("\n各月份详情：")
    print("月份 | 全局选择 | 季节选择 | 改进")
    print("-----|----------|----------|-------")
    for month in range(1, 13):
        if month in results_by_month:
            r = results_by_month[month]
            if r['same_choice']:
                print(f"{month:>2}月 | -{r['best_global_window']:>2}月    | -{r['best_seasonal_window']:>2}月    | 相同")
            else:
                print(f"{month:>2}月 | -{r['best_global_window']:>2}月    | -{r['best_seasonal_window']:>2}月    | {r['improvement']:>+5.1f}%")

print("\n" + "="*80)
print("结论：")
print("="*80)
if different_count > total_count * 0.3:
    print("✓ 季节性加权方法对多数月份有效，能找到更适合特定月份的模型")
else:
    print("• 季节性加权方法在少数月份有差异，整体与全局方法接近")
