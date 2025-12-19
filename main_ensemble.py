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

def run_ensemble_forecast_system(full_df, top_n=3):
    """集成学习版本：使用表现最好的N个模型进行加权平均预测"""
    
    df_modern = full_df.loc["1970-01-01":].copy()
    print(f"所有训练和测试将基于1970年之后的数据. 范围: {df_modern.index.min().strftime('%Y-%m')} to {df_modern.index.max().strftime('%Y-%m')}")

    print('\n--- [第一部分] 开始执行"赛马"实验 ---')
    target_date = pd.to_datetime(config.TARGET_YYYYMM, format='%Y%m')
    last_available_date = df_modern.index.max()
    split_months_options = [18, 21, 24, 27, 30]
    race_results = []

    for months in split_months_options:
        split_date = target_date - relativedelta(months=months)
        print(f'\n--- 执行: -{months}个月 赛道 ---')
        
        train_df = df_modern.loc[:split_date]
        test_df = df_modern.loc[split_date + relativedelta(months=1):last_available_date]

        if test_df.empty or len(train_df) < 36:
            print(f'      警告: 数据不足，跳过')
            continue

        train_endog = train_df['Anomaly']
        train_exog = train_df[['ENSO_ANOM']]
        test_endog = test_df['Anomaly']
        test_exog = test_df[['ENSO_ANOM']]

        print(f'      训练截止: {train_df.index.max().strftime("%Y-%m")}, 测试集: {test_df.index.min().strftime("%Y-%m")} to {test_df.index.max().strftime("%Y-%m")}')
        
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
        print(f'      完成。RMSE: {rmse:.2f}')

    if not race_results:
        raise ValueError("所有赛马实验均失败")

    # 选择表现最好的top_n个模型
    race_results_sorted = sorted(race_results, key=lambda x: x['rmse'])
    top_models = race_results_sorted[:top_n]
    
    # 计算权重（RMSE越小权重越大）
    rmse_values = np.array([m['rmse'] for m in top_models])
    weights = 1 / rmse_values
    weights = weights / weights.sum()
    
    print('\n--- [第二部分] 集成预测 ---')
    print('\n--- "赛马"结果总结 ---')
    print("偏移 | RMSE   | 权重   | 入选?")
    print("-----|--------|--------|------")
    for res in race_results_sorted:
        in_ensemble = res in top_models
        weight = weights[top_models.index(res)] if in_ensemble else 0
        mark = f"  {weight:.1%}" if in_ensemble else ""
        print(f"-{res['offset']:<4} | {res['rmse']:<6.2f} | {mark}")
    
    print(f"\n使用Top-{top_n}模型进行集成预测...")

    # 预测ENSO
    steps_to_forecast = (target_date.year - last_available_date.year) * 12 + (target_date.month - last_available_date.month)
    future_index = pd.date_range(start=last_available_date + pd.DateOffset(months=1), periods=steps_to_forecast, freq='MS')
    
    print("\n预测ENSO值...")
    enso_order, enso_seasonal_order = model.find_best_sarimax_params(df_modern['ENSO_ANOM'], exog=None)
    enso_forecast_results = model.train_and_forecast_sarimax(df_modern['ENSO_ANOM'], enso_order, enso_seasonal_order, steps=steps_to_forecast)
    future_enso_values = enso_forecast_results.predicted_mean
    future_exog = pd.DataFrame(future_enso_values.values, index=future_index, columns=['ENSO_ANOM'])
    print(f"      预测ENSO值: {future_enso_values.iloc[-1]:.2f}")

    # 对每个top模型进行预测并加权平均
    ensemble_predictions = []
    ensemble_std_errs = []
    
    for i, m in enumerate(top_models):
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
        print(f"      模型{i+1} (偏移-{m['offset']}月): 预测={pred_mean:.2f}, 权重={weights[i]:.1%}")
    
    # 加权平均
    final_prediction = np.average(ensemble_predictions, weights=weights)
    final_std_err = np.sqrt(np.average(np.array(ensemble_std_errs)**2, weights=weights))
    
    print("\n--- 集成预测结果 ---")
    print(f"> 目标月份: {config.TARGET_YYYYMM}")
    print(f"> 集成预测均值: {final_prediction:.2f}")
    print(f"> 标准误差: {final_std_err:.2f}")
    
    # 计算概率分布
    probabilities = {}
    sorted_bins = sorted(config.PREDICTION_BINS.items(), key=lambda item: item[1][0] if item[1][0] is not None else -np.inf)
    for name, (lower, upper) in sorted_bins:
        if lower is None: prob = norm.cdf(upper, loc=final_prediction, scale=final_std_err)
        elif upper is None: prob = 1 - norm.cdf(lower, loc=final_prediction, scale=final_std_err)
        else: prob = norm.cdf(upper, loc=final_prediction, scale=final_std_err) - norm.cdf(lower, loc=final_prediction, scale=final_std_err)
        probabilities[name] = prob
    
    print("\n--- 集成模型预测概率分布 ---")
    for bin_name, prob in probabilities.items():
        print(f"  - 区间 '{bin_name}': {prob:.2%}")

if __name__ == "__main__":
    try:
        print("--- 数据准备阶段 ---")
        gistemp_data = data_handler.get_clean_data(config.DATA_FILE_PATH)
        enso_data = data_handler.get_enso_data(config.ENSO_DATA_URL)
        full_df = pd.concat([gistemp_data, enso_data], axis=1)
        full_df.dropna(inplace=True)
        
        run_ensemble_forecast_system(full_df, top_n=3)

    except Exception as e:
        print(f"错误: {e}")
