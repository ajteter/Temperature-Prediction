import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta

import config
import data_handler
import model

def rolling_window_validation(full_df, window_sizes=[18, 21, 24, 27, 30], n_tests=6):
    """
    滚动窗口验证：在多个时间点测试模型性能
    返回每个窗口大小在不同测试点的平均表现
    """
    df_modern = full_df.loc["1970-01-01":].copy()
    target_date = pd.to_datetime(config.TARGET_YYYYMM, format='%Y%m')
    last_available_date = df_modern.index.max()
    
    print("=== 滚动窗口验证 ===")
    print(f"数据范围: {df_modern.index.min().strftime('%Y-%m')} to {last_available_date.strftime('%Y-%m')}\n")
    
    results = {ws: [] for ws in window_sizes}
    
    # 在过去n_tests个不同时间点进行测试
    for test_idx in range(n_tests):
        test_offset = test_idx * 3  # 每3个月一个测试点
        test_end = last_available_date - relativedelta(months=test_offset)
        
        print(f"\n--- 测试点 {test_idx+1}: 预测 {test_end.strftime('%Y-%m')} ---")
        
        for window_size in window_sizes:
            train_end = test_end - relativedelta(months=window_size)
            
            if train_end < df_modern.index.min() + relativedelta(months=36):
                continue
                
            train_df = df_modern.loc[:train_end]
            actual_value = df_modern.loc[test_end, 'Anomaly']
            
            train_endog = train_df['Anomaly']
            train_exog = train_df[['ENSO_ANOM']]
            test_exog = df_modern.loc[[test_end], ['ENSO_ANOM']]
            
            try:
                best_order, best_seasonal_order = model.find_best_sarimax_params(train_endog, train_exog)
                mod = sm.tsa.statespace.SARIMAX(
                    train_endog, exog=train_exog, 
                    order=best_order, seasonal_order=best_seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False
                )
                fit_results = mod.fit(disp=False)
                forecast = fit_results.get_forecast(steps=1, exog=test_exog)
                predicted = forecast.predicted_mean.iloc[0]
                error = abs(predicted - actual_value)
                
                results[window_size].append(error)
                print(f"  窗口-{window_size}月: 预测={predicted:.2f}, 实际={actual_value:.2f}, 误差={error:.2f}")
            except:
                continue
    
    # 汇总结果
    print("\n=== 验证结果汇总 ===")
    print("窗口大小 | 平均误差 | 标准差 | 测试次数")
    print("---------|----------|--------|----------")
    
    summary = {}
    for ws in window_sizes:
        if results[ws]:
            mean_err = np.mean(results[ws])
            std_err = np.std(results[ws])
            count = len(results[ws])
            summary[ws] = {'mean': mean_err, 'std': std_err, 'count': count}
            print(f"{ws:>8}月 | {mean_err:>8.2f} | {std_err:>6.2f} | {count:>8}")
    
    best_window = min(summary.keys(), key=lambda x: summary[x]['mean'])
    print(f"\n最优窗口: {best_window}月 (平均误差最小)")
    
    return summary, best_window

def out_of_sample_test(full_df, train_end_date, test_months=12):
    """
    样本外测试：使用指定日期之前的数据训练，预测之后的数据
    """
    df_modern = full_df.loc["1970-01-01":].copy()
    train_end = pd.to_datetime(train_end_date)
    
    print(f"\n=== 样本外测试 ===")
    print(f"训练截止: {train_end.strftime('%Y-%m')}")
    
    train_df = df_modern.loc[:train_end]
    test_start = train_end + relativedelta(months=1)
    test_end = min(train_end + relativedelta(months=test_months), df_modern.index.max())
    test_df = df_modern.loc[test_start:test_end]
    
    train_endog = train_df['Anomaly']
    train_exog = train_df[['ENSO_ANOM']]
    test_endog = test_df['Anomaly']
    test_exog = test_df[['ENSO_ANOM']]
    
    best_order, best_seasonal_order = model.find_best_sarimax_params(train_endog, train_exog)
    mod = sm.tsa.statespace.SARIMAX(
        train_endog, exog=train_exog,
        order=best_order, seasonal_order=best_seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False
    )
    fit_results = mod.fit(disp=False)
    forecast_results = fit_results.get_forecast(steps=len(test_df), exog=test_exog)
    predictions = forecast_results.predicted_mean
    
    rmse = np.sqrt(mean_squared_error(test_endog, predictions))
    mae = mean_absolute_error(test_endog, predictions)
    
    print(f"测试期间: {test_start.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    
    print("\n逐月对比:")
    print("月份      | 预测值 | 实际值 | 误差")
    print("----------|--------|--------|------")
    for date, pred, actual in zip(test_df.index, predictions, test_endog):
        error = pred - actual
        print(f"{date.strftime('%Y-%m')} | {pred:>6.2f} | {actual:>6.2f} | {error:>+5.2f}")
    
    return rmse, mae, predictions, test_endog

def compare_model_configurations(full_df):
    """
    比较不同模型配置的性能
    """
    df_modern = full_df.loc["1970-01-01":].copy()
    target_date = pd.to_datetime(config.TARGET_YYYYMM, format='%Y%m')
    
    print("\n=== 模型配置对比 ===")
    
    configs = [
        {'name': '仅ENSO', 'use_enso': True, 'use_trend': False},
        {'name': '无外生变量', 'use_enso': False, 'use_trend': False},
    ]
    
    results = []
    
    for cfg in configs:
        print(f"\n测试配置: {cfg['name']}")
        
        # 使用最近24个月作为训练窗口
        train_end = target_date - relativedelta(months=24)
        train_df = df_modern.loc[:train_end]
        test_df = df_modern.loc[train_end + relativedelta(months=1):df_modern.index.max()]
        
        train_endog = train_df['Anomaly']
        test_endog = test_df['Anomaly']
        
        if cfg['use_enso']:
            train_exog = train_df[['ENSO_ANOM']]
            test_exog = test_df[['ENSO_ANOM']]
        else:
            train_exog = None
            test_exog = None
        
        try:
            best_order, best_seasonal_order = model.find_best_sarimax_params(train_endog, train_exog)
            mod = sm.tsa.statespace.SARIMAX(
                train_endog, exog=train_exog,
                order=best_order, seasonal_order=best_seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False
            )
            fit_results = mod.fit(disp=False)
            forecast_results = fit_results.get_forecast(steps=len(test_df), exog=test_exog)
            rmse = np.sqrt(mean_squared_error(test_endog, forecast_results.predicted_mean))
            
            results.append({'config': cfg['name'], 'rmse': rmse})
            print(f"  RMSE: {rmse:.2f}")
        except Exception as e:
            print(f"  失败: {e}")
    
    print("\n配置排名:")
    results_sorted = sorted(results, key=lambda x: x['rmse'])
    for i, r in enumerate(results_sorted, 1):
        print(f"{i}. {r['config']}: RMSE={r['rmse']:.2f}")

if __name__ == "__main__":
    print("=== 模型验证系统 ===\n")
    
    gistemp_data = data_handler.get_clean_data(config.DATA_FILE_PATH)
    enso_data = data_handler.get_enso_data(config.ENSO_DATA_URL)
    full_df = pd.concat([gistemp_data, enso_data], axis=1)
    full_df.dropna(inplace=True)
    
    # 1. 滚动窗口验证
    summary, best_window = rolling_window_validation(full_df, window_sizes=[15, 18, 21, 24, 27, 30], n_tests=8)
    
    # 2. 样本外测试
    out_of_sample_test(full_df, train_end_date='2024-02', test_months=12)
    
    # 3. 模型配置对比
    compare_model_configurations(full_df)
