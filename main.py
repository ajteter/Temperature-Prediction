import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

def run_adaptive_forecast_system(full_df):
    """执行一个自适应预测系统：先通过回测找到最优模型，然后用该模型进行最终预测。"""
    
    df_modern = full_df.loc["1970-01-01":].copy()
    print(f"所有训练和测试将基于1970年之后的数据 (All training and testing will be based on data since 1970). 范围 (Range): {df_modern.index.min().strftime('%Y-%m')} to {df_modern.index.max().strftime('%Y-%m')}")

    print('\n--- [第一部分 / Part 1] 开始执行“赛马”实验，寻找最优训练周期 (Starting "Horse Race" to find the optimal training period) ---')
    target_date = pd.to_datetime(config.TARGET_YYYYMM, format='%Y%m')
    last_available_date = df_modern.index.max()
    split_months_options = [18, 21, 24, 27, 30]
    race_results = []

    for months in split_months_options:
        split_date = target_date - relativedelta(months=months)
        name = f"-{months}个月"
        print(f'\n--- 正在执行 (Executing): {name} 赛道 (Lane) ---')
        
        train_df = df_modern.loc[:split_date]
        test_df = df_modern.loc[split_date + relativedelta(months=1):last_available_date]

        if test_df.empty or len(train_df) < 36:
            print(f'      警告 (Warning): 数据不足，跳过此实验 (Insufficient data, skipping this experiment).')
            continue

        train_endog = train_df['Anomaly']
        train_exog = train_df[['ENSO_ANOM']]
        test_endog = test_df['Anomaly']
        test_exog = test_df[['ENSO_ANOM']]

        print(f'      训练集截止 (Train End): {train_df.index.max().strftime("%Y-%m")}, 测试集 (Test Set): {test_df.index.min().strftime("%Y-%m")} to {test_df.index.max().strftime("%Y-%m")}')
        
        best_order, best_seasonal_order = model.find_best_sarimax_params(train_endog, train_exog)
        mod = sm.tsa.statespace.SARIMAX(train_endog, exog=train_exog, order=best_order, seasonal_order=best_seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        fit_results = mod.fit(disp=False)
        forecast_results = fit_results.get_forecast(steps=len(test_df), exog=test_exog)
        rmse = np.sqrt(mean_squared_error(test_endog, forecast_results.predicted_mean))
        
        race_results.append({'offset': months, 'rmse': rmse, 'order': best_order, 'seasonal_order': best_seasonal_order, 'train_end_date': split_date})
        print(f'      ...完成 (Completed)。RMSE: {rmse:.2f}')

    if not race_results:
        raise ValueError("所有“赛马”实验均失败，无法进行最终预测 (All 'Horse Race' experiments failed, cannot proceed to final forecast).")

    champion = min(race_results, key=lambda x: x['rmse'])
    print('\n--- [第二部分 / Part 2] “赛马”实验完成，执行最终预测 (Horse Race complete, executing final forecast) ---')
    print('\n--- “赛马”结果总结 (Horse Race Results Summary) ---')
    print("回测偏移 (Offset) | RMSE   | 胜出? (Champion?)")
    print("-----------------|--------|-----------------")
    for res in race_results:
        is_champion = "  * " if res['offset'] == champion['offset'] else ""
        print(f"-{res['offset']:<15} | {res['rmse']:<6.2f} |{is_champion}")
    
    print(f"\n冠军模型 (Champion Model): {-champion['offset']}个月的训练周期 (training period), 训练至 (trained until) {champion['train_end_date'].strftime('%Y-%m')}).")
    print("现在将使用此‘冠军’配置进行最终预测 (Now using this 'champion' configuration for the final forecast)...")

    final_train_df = df_modern.loc[:champion['train_end_date']]
    final_train_endog = final_train_df['Anomaly']
    final_train_exog = final_train_df[['ENSO_ANOM']]

    steps_to_forecast = (target_date.year - last_available_date.year) * 12 + (target_date.month - last_available_date.month)
    future_index = pd.date_range(start=last_available_date + pd.DateOffset(months=1), periods=steps_to_forecast, freq='MS')

    print("\n正在为ENSO数据本身建立预测模型 (Building forecast model for ENSO data itself)...")
    enso_order, enso_seasonal_order = model.find_best_sarimax_params(df_modern['ENSO_ANOM'], exog=None)
    enso_forecast_results = model.train_and_forecast_sarimax(df_modern['ENSO_ANOM'], enso_order, enso_seasonal_order, steps=steps_to_forecast)
    future_enso_values = enso_forecast_results.predicted_mean
    future_exog = pd.DataFrame(future_enso_values.values, index=future_index, columns=['ENSO_ANOM'])
    print(f"      ...动态预测出未来 {steps_to_forecast} 个月的ENSO异常值为 (Dynamically forecasted ENSO anomaly for the next {steps_to_forecast} month(s) is): {future_enso_values.iloc[-1]:.2f}")

    print("\n正在使用冠军配置训练最终模型并进行预测 (Training and forecasting with champion configuration)...")
    final_mod = sm.tsa.statespace.SARIMAX(final_train_endog, exog=final_train_exog, order=champion['order'], seasonal_order=champion['seasonal_order'], enforce_stationarity=False, enforce_invertibility=False)
    final_fit_results = final_mod.fit(disp=False)
    final_forecast_results = final_fit_results.get_forecast(steps=len(future_exog), exog=future_exog)

    print("\n--- 最终预测结果分析 (冠军模型) / Final Forecast Analysis (Champion Model) ---")
    probabilities, predicted_mean = calculate_bin_probabilities(final_forecast_results, config.PREDICTION_BINS)
    print(f"> 目标月份 (Target Month): {config.TARGET_YYYYMM}")
    print(f"> 预测的期望值 (Predicted Mean): {predicted_mean:.2f}")
    print("\n--- 模型预测概率分布 (Model's Predicted Probability Distribution) ---")
    for bin_name, prob in probabilities.items():
        print(f"  - 区间 (Bin) '{bin_name}': {prob:.2%}")

if __name__ == "__main__":
    try:
        print("--- 数据准备阶段 (Data Preparation Stage) ---")
        gistemp_data = data_handler.get_clean_data(config.DATA_FILE_PATH)
        enso_data = data_handler.get_enso_data(config.ENSO_DATA_URL)
        full_df = pd.concat([gistemp_data, enso_data], axis=1)
        full_df.dropna(inplace=True)
        
        run_adaptive_forecast_system(full_df)

    except FileNotFoundError as e:
        print(f"错误 (Error): 数据文件未找到 (Data file not found)。 {e}")
    except Exception as e:
        print(f"发生了一个未预料的错误 (An unexpected error occurred): {e}")
