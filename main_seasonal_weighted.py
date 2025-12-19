import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta

import config
import data_handler
import model

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

def run_seasonal_weighted_forecast(full_df):
    """
    季节性加权赛马：只在相同月份的历史数据上计算RMSE
    这样可以更准确地评估模型对特定月份的预测能力
    """
    
    df_modern = full_df.loc["1970-01-01":].copy()
    print(f"数据范围: {df_modern.index.min().strftime('%Y-%m')} to {df_modern.index.max().strftime('%Y-%m')}\n")

    target_date = pd.to_datetime(config.TARGET_YYYYMM, format='%Y%m')
    target_month = target_date.month
    last_available_date = df_modern.index.max()
    
    print(f"目标月份: {target_date.strftime('%Y年%m月')} (第{target_month}月)")
    print("使用季节性加权：仅在相同月份的历史数据上评估模型\n")
    
    print('--- [第一部分] 季节性赛马实验 ---')
    split_months_options = list(range(12, 37, 3))
    race_results = []

    for months in split_months_options:
        split_date = target_date - relativedelta(months=months)
        
        train_df = df_modern.loc[:split_date]
        test_df = df_modern.loc[split_date + relativedelta(months=1):last_available_date]

        if test_df.empty or len(train_df) < 36:
            continue

        # 只在目标月份的数据上计算RMSE
        test_df_same_month = test_df[test_df.index.month == target_month]
        
        if len(test_df_same_month) == 0:
            continue

        train_endog = train_df['Anomaly']
        train_exog = train_df[['ENSO_ANOM']]
        test_endog = test_df['Anomaly']
        test_exog = test_df[['ENSO_ANOM']]
        
        best_order, best_seasonal_order = model.find_best_sarimax_params(train_endog, train_exog)
        mod = sm.tsa.statespace.SARIMAX(train_endog, exog=train_exog, order=best_order, seasonal_order=best_seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        fit_results = mod.fit(disp=False)
        forecast_results = fit_results.get_forecast(steps=len(test_df), exog=test_exog)
        
        # 计算全部RMSE和同月RMSE
        rmse_all = np.sqrt(mean_squared_error(test_endog, forecast_results.predicted_mean))
        
        # 提取同月的预测和实际值
        predictions_all = pd.Series(forecast_results.predicted_mean.values, index=test_df.index)
        predictions_same_month = predictions_all[predictions_all.index.month == target_month]
        actuals_same_month = test_df_same_month['Anomaly']
        rmse_same_month = np.sqrt(mean_squared_error(actuals_same_month, predictions_same_month))
        
        race_results.append({
            'offset': months, 
            'rmse_all': rmse_all,
            'rmse_same_month': rmse_same_month,
            'order': best_order, 
            'seasonal_order': best_seasonal_order, 
            'train_end_date': split_date,
            'same_month_count': len(test_df_same_month)
        })
        print(f"  -{months:>2}月窗口: RMSE(全部)={rmse_all:>6.2f}, RMSE({target_month}月)={rmse_same_month:>6.2f}, 样本数={len(test_df_same_month)}")

    if not race_results:
        raise ValueError("所有赛马实验均失败")

    # 使用同月RMSE选择最好的4个模型
    good_models = sorted(race_results, key=lambda x: x['rmse_same_month'])[:4]
    
    rmse_values = np.array([m['rmse_same_month'] for m in good_models])
    weights = 1 / (rmse_values ** 2)
    weights = weights / weights.sum()
    
    print('\n--- [第二部分] 智能集成预测 ---')
    print(f"基于{target_month}月的历史表现，选择最好的4个模型\n")
    
    print("偏移 | RMSE(全) | RMSE(同月) | 权重   | 状态")
    print("-----|----------|------------|--------|------")
    race_results_sorted = sorted(race_results, key=lambda x: x['rmse_same_month'])
    for res in race_results_sorted:
        if res in good_models:
            weight = weights[good_models.index(res)]
            print(f"-{res['offset']:>2}月 | {res['rmse_all']:>8.2f} | {res['rmse_same_month']:>10.2f} | {weight:>6.1%} | ✓ 入选")
        else:
            print(f"-{res['offset']:>2}月 | {res['rmse_all']:>8.2f} | {res['rmse_same_month']:>10.2f} |        | ")

    # 预测ENSO
    steps_to_forecast = (target_date.year - last_available_date.year) * 12 + (target_date.month - last_available_date.month)
    future_index = pd.date_range(start=last_available_date + pd.DateOffset(months=1), periods=steps_to_forecast, freq='MS')
    
    print(f"\n预测未来 {steps_to_forecast} 个月的ENSO值...")
    enso_order, enso_seasonal_order = model.find_best_sarimax_params(df_modern['ENSO_ANOM'], exog=None)
    enso_forecast_results = model.train_and_forecast_sarimax(df_modern['ENSO_ANOM'], enso_order, enso_seasonal_order, steps=steps_to_forecast)
    future_enso_values = enso_forecast_results.predicted_mean
    future_exog = pd.DataFrame(future_enso_values.values, index=future_index, columns=['ENSO_ANOM'])
    print(f"  ENSO预测值: {future_enso_values.iloc[-1]:.2f}")

    # 集成预测
    ensemble_predictions = []
    ensemble_std_errs = []
    
    print(f"\n各模型预测结果:")
    for i, m in enumerate(good_models):
        final_train_df = df_modern.loc[:m['train_end_date']]
        final_train_endog = final_train_df['Anomaly']
        final_train_exog = final_train_df[['ENSO_ANOM']]
        
        final_mod = sm.tsa.statespace.SARIMAX(
            final_train_endog, 
            exog=final_train_exog, 
            order=m['order'], 
            seasonal_order=m['seasonal_order'], 
            enforce_stationarity=False, 
            enforce_invertibility=False
        )
        final_fit_results = final_mod.fit(disp=False)
        final_forecast_results = final_fit_results.get_forecast(steps=len(future_exog), exog=future_exog)
        
        pred_mean = final_forecast_results.predicted_mean.iloc[-1]
        conf = final_forecast_results.conf_int().iloc[-1]
        std_err = (conf.iloc[1] - conf.iloc[0]) / (2 * 1.95996)
        
        ensemble_predictions.append(pred_mean)
        ensemble_std_errs.append(std_err)
        print(f"  模型{i+1} (-{m['offset']:>2}月): {pred_mean:>6.2f} (权重 {weights[i]:>5.1%})")
    
    final_prediction = np.average(ensemble_predictions, weights=weights)
    final_std_err = np.sqrt(np.average(np.array(ensemble_std_errs)**2, weights=weights))
    
    print("\n" + "="*50)
    print("=== 最终预测结果 ===")
    print("="*50)
    print(f"目标月份: {config.TARGET_YYYYMM}")
    print(f"集成预测均值: {final_prediction:.2f}")
    print(f"预测标准误差: {final_std_err:.2f}")
    print(f"95%置信区间: [{final_prediction - 1.96*final_std_err:.2f}, {final_prediction + 1.96*final_std_err:.2f}]")
    
    probabilities = {}
    sorted_bins = sorted(config.PREDICTION_BINS.items(), key=lambda item: item[1][0] if item[1][0] is not None else -np.inf)
    for name, (lower, upper) in sorted_bins:
        if lower is None: prob = norm.cdf(upper, loc=final_prediction, scale=final_std_err)
        elif upper is None: prob = 1 - norm.cdf(lower, loc=final_prediction, scale=final_std_err)
        else: prob = norm.cdf(upper, loc=final_prediction, scale=final_std_err) - norm.cdf(lower, loc=final_prediction, scale=final_std_err)
        probabilities[name] = prob
    
    print("\n--- 预测概率分布 ---")
    for bin_name, prob in probabilities.items():
        bar = '█' * int(prob * 50)
        print(f"  {bin_name:>10}: {prob:>6.2%} {bar}")
    
    return final_prediction, probabilities

if __name__ == "__main__":
    try:
        print("=== 季节性加权赛马系统 ===\n")
        
        gistemp_data = data_handler.get_clean_data(config.DATA_FILE_PATH)
        enso_data = data_handler.get_enso_data(config.ENSO_DATA_URL)
        full_df = pd.concat([gistemp_data, enso_data], axis=1)
        full_df.dropna(inplace=True)
        
        prediction, probabilities = run_seasonal_weighted_forecast(full_df)

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
