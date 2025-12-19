import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta

import config
import data_handler
import model

def calculate_recent_trend(data, months=12):
    """计算最近N个月的趋势"""
    recent = data.tail(months)
    x = np.arange(len(recent))
    y = recent.values
    trend = np.polyfit(x, y, 1)[0]  # 线性趋势斜率
    return trend

def calculate_bin_probabilities(forecast_results, bins):
    """根据预测的分布计算其落入每个区间的概率。"""
    mu = forecast_results.predicted_mean.iloc[-1]
    conf = forecast_results.conf_int().iloc[-1]
    std_err = (conf.iloc[1] - conf.iloc[0]) / (2 * 1.95996)
    probabilities = {}
    sorted_bins = sorted(bins.items(), key=lambda item: item[1][0] if item[1][0] is not None else -np.inf)
    for name, (lower, upper) in sorted_bins:
        if lower is None: prob = norm.cdf(upper, loc=mu, scale=std_err)
        elif upper is None: prob = 1 - norm.cdf(lower, loc=mu, scale=std_err)
        else: prob = norm.cdf(upper, loc=mu, scale=std_err) - norm.cdf(lower, loc=mu, scale=std_err)
        probabilities[name] = prob
    return probabilities, mu

def run_trend_adjusted_forecast(full_df, trend_weight=0.3):
    """趋势调整版本：在赛马结果基础上考虑近期趋势"""
    
    df_modern = full_df.loc["1970-01-01":].copy()
    print(f"数据范围: {df_modern.index.min().strftime('%Y-%m')} to {df_modern.index.max().strftime('%Y-%m')}")

    # 计算近期趋势
    recent_trend = calculate_recent_trend(df_modern['Anomaly'], months=12)
    print(f"\n最近12个月趋势斜率: {recent_trend:.3f} (每月)")
    
    print('\n--- [第一部分] "赛马"实验 ---')
    target_date = pd.to_datetime(config.TARGET_YYYYMM, format='%Y%m')
    last_available_date = df_modern.index.max()
    split_months_options = [18, 21, 24, 27, 30]
    race_results = []

    for months in split_months_options:
        split_date = target_date - relativedelta(months=months)
        print(f'\n--- -{months}个月 赛道 ---')
        
        train_df = df_modern.loc[:split_date]
        test_df = df_modern.loc[split_date + relativedelta(months=1):last_available_date]

        if test_df.empty or len(train_df) < 36:
            continue

        train_endog = train_df['Anomaly']
        train_exog = train_df[['ENSO_ANOM']]
        test_endog = test_df['Anomaly']
        test_exog = test_df[['ENSO_ANOM']]

        print(f'      训练截止: {train_df.index.max().strftime("%Y-%m")}')
        
        best_order, best_seasonal_order = model.find_best_sarimax_params(train_endog, train_exog)
        mod = sm.tsa.statespace.SARIMAX(train_endog, exog=train_exog, order=best_order, seasonal_order=best_seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        fit_results = mod.fit(disp=False)
        forecast_results = fit_results.get_forecast(steps=len(test_df), exog=test_exog)
        rmse = np.sqrt(mean_squared_error(test_endog, forecast_results.predicted_mean))
        
        race_results.append({
            'offset': months, 
            'rmse': rmse, 
            'order': best_order, 
            'seasonal_order': best_seasonal_order, 
            'train_end_date': split_date
        })
        print(f'      RMSE: {rmse:.2f}')

    champion = min(race_results, key=lambda x: x['rmse'])
    
    print('\n--- [第二部分] 趋势调整预测 ---')
    print(f"冠军模型: -{champion['offset']}个月")

    # 标准预测
    final_train_df = df_modern.loc[:champion['train_end_date']]
    final_train_endog = final_train_df['Anomaly']
    final_train_exog = final_train_df[['ENSO_ANOM']]

    steps_to_forecast = (target_date.year - last_available_date.year) * 12 + (target_date.month - last_available_date.month)
    future_index = pd.date_range(start=last_available_date + pd.DateOffset(months=1), periods=steps_to_forecast, freq='MS')

    print("\n预测ENSO...")
    enso_order, enso_seasonal_order = model.find_best_sarimax_params(df_modern['ENSO_ANOM'], exog=None)
    enso_forecast_results = model.train_and_forecast_sarimax(df_modern['ENSO_ANOM'], enso_order, enso_seasonal_order, steps=steps_to_forecast)
    future_enso_values = enso_forecast_results.predicted_mean
    future_exog = pd.DataFrame(future_enso_values.values, index=future_index, columns=['ENSO_ANOM'])
    print(f"      ENSO预测值: {future_enso_values.iloc[-1]:.2f}")

    final_mod = sm.tsa.statespace.SARIMAX(
        final_train_endog, 
        exog=final_train_exog, 
        order=champion['order'], 
        seasonal_order=champion['seasonal_order'], 
        enforce_stationarity=False, 
        enforce_invertibility=False
    )
    final_fit_results = final_mod.fit(disp=False)
    final_forecast_results = final_fit_results.get_forecast(steps=len(future_exog), exog=future_exog)

    # 原始预测
    original_prediction = final_forecast_results.predicted_mean.iloc[-1]
    
    # 趋势调整
    months_from_train_end = (target_date.year - champion['train_end_date'].year) * 12 + (target_date.month - champion['train_end_date'].month)
    trend_adjustment = recent_trend * months_from_train_end * trend_weight
    
    adjusted_prediction = original_prediction + trend_adjustment
    
    print("\n--- 预测结果对比 ---")
    print(f"> 目标月份: {config.TARGET_YYYYMM}")
    print(f"> 原始预测: {original_prediction:.2f}")
    print(f"> 趋势调整量: {trend_adjustment:+.2f} (权重={trend_weight})")
    print(f"> 调整后预测: {adjusted_prediction:.2f}")
    
    # 使用调整后的预测计算概率
    conf = final_forecast_results.conf_int().iloc[-1]
    std_err = (conf.iloc[1] - conf.iloc[0]) / (2 * 1.95996)
    
    probabilities = {}
    sorted_bins = sorted(config.PREDICTION_BINS.items(), key=lambda item: item[1][0] if item[1][0] is not None else -np.inf)
    for name, (lower, upper) in sorted_bins:
        if lower is None: prob = norm.cdf(upper, loc=adjusted_prediction, scale=std_err)
        elif upper is None: prob = 1 - norm.cdf(lower, loc=adjusted_prediction, scale=std_err)
        else: prob = norm.cdf(upper, loc=adjusted_prediction, scale=std_err) - norm.cdf(lower, loc=adjusted_prediction, scale=std_err)
        probabilities[name] = prob
    
    print("\n--- 调整后概率分布 ---")
    for bin_name, prob in probabilities.items():
        print(f"  - 区间 '{bin_name}': {prob:.2%}")

if __name__ == "__main__":
    try:
        print("--- 数据准备 ---")
        gistemp_data = data_handler.get_clean_data(config.DATA_FILE_PATH)
        enso_data = data_handler.get_enso_data(config.ENSO_DATA_URL)
        full_df = pd.concat([gistemp_data, enso_data], axis=1)
        full_df.dropna(inplace=True)
        
        # 可以调整trend_weight参数 (0-1之间)
        run_trend_adjusted_forecast(full_df, trend_weight=0.2)

    except Exception as e:
        print(f"错误: {e}")
