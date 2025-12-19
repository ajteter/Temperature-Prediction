import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta
import sys
import io

# 导入改进版的配置文件
import config_robust as config
import data_handler
import model

# 捕获标准输出以避免打印过多日志
class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = io.StringIO()
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout

def calculate_trend_slope(series):
    """计算时间序列的线性趋势斜率"""
    if len(series) < 2:
        return 0
    x = np.arange(len(series))
    y = series.values
    slope = np.polyfit(x, y, 1)[0]
    return slope


def prepare_exog_for_forecast(last_train_date, steps, lag, enso_series, enso_order=None, enso_seasonal_order=None):
    """
    Copy of the helper from main_robust.py to ensure consistent validation.
    """
    future_dates = [last_train_date + relativedelta(months=i+1) for i in range(steps)]
    required_enso_dates = [d - relativedelta(months=lag) for d in future_dates]
    
    exog_values = []
    last_enso_date = enso_series.index.max()
    
    max_required_date = max(required_enso_dates)
    enso_forecast_needed = max_required_date > last_enso_date
    enso_forecast_series = None
    
    if enso_forecast_needed:
        months_to_forecast = (max_required_date.year - last_enso_date.year) * 12 + (max_required_date.month - last_enso_date.month)
        with SuppressOutput():
            if enso_order is None:
                 enso_order, enso_seasonal_order = model.find_best_sarimax_params(enso_series, exog=None)
            forecast_res = model.train_and_forecast_sarimax(enso_series, enso_order, enso_seasonal_order, steps=months_to_forecast)
            enso_forecast_series = forecast_res.predicted_mean
        
    for d in required_enso_dates:
        if d <= last_enso_date:
            exog_values.append(enso_series.loc[d])
        else:
            exog_values.append(enso_forecast_series.loc[d])
            
    return pd.DataFrame(exog_values, index=future_dates, columns=['ENSO_ANOM_LAGGED'])

def run_backtest_for_month(target_yyyymm, full_df, optimal_enso_lag):
    """
    针对特定月份执行回测 (Logic synced with main_robust.py)
    """
    target_date = pd.to_datetime(target_yyyymm, format='%Y%m')
    target_month = target_date.month
    
    # 定义数据截止日期（目标月的前一个月）
    cutoff_date = target_date - relativedelta(months=1)
    
    # 模拟当时的数据环境
    # 重要: 这里必须小心处理 enso_series。full_df 包含了"未来"的ENSO数据（相对于cutoff_date），
    # 我们必须剥离出来，传给 prepare_exog_for_forecast 的只能是 cutoff_date 之前的。
    
    # 分离原始序列
    gistemp_series = full_df['Anomaly']
    enso_series_full = full_df['ENSO_ANOM'] # Use raw unshifted for helper
    
    df_modern = pd.DataFrame({'Anomaly': gistemp_series})
    df_modern['ENSO_ANOM_LAGGED'] = enso_series_full.shift(optimal_enso_lag)
    df_modern = df_modern.loc["1970-01-01":cutoff_date].copy()
    df_modern.dropna(inplace=True) 

    # 获取真实值（如果存在）
    actual_value = None
    if target_date in full_df.index:
        actual_value = full_df.loc[target_date, 'Anomaly']
    
    race_results = []
    last_available_date = df_modern.index.max()
    
    # 为回测准备的模拟ENSO历史
    # 注意: df_modern已经根据cutoff_date截断了，但enso_series_full还是全量的。
    # 我们需要一个 strictly cutoff 的 enso series
    sim_enso_history_main = enso_series_full.loc[:last_available_date]

    for months in config.RACE_SPLIT_MONTHS:
        split_date = target_date - relativedelta(months=months)
        
        if split_date < df_modern.index.min() + relativedelta(months=36):
            continue
            
        train_df = df_modern.loc[:split_date]
        test_df_period = df_modern.loc[split_date + relativedelta(months=1):last_available_date]

        if test_df_period.empty:
            continue
            
        # Horse Race 内部的回测：模拟在 split_date 的情况
        sim_enso_history_inner = enso_series_full.loc[:split_date]

        steps_test = len(test_df_period)
        
        test_df_same_month = test_df_period[test_df_period.index.month == target_month]
        same_month_count = len(test_df_same_month)
        
        if same_month_count == 0:
            continue

        train_endog = train_df['Anomaly']
        train_exog = train_df[['ENSO_ANOM_LAGGED']]
        
        with SuppressOutput(): # 抑制模型训练输出
            best_order, best_seasonal_order = model.find_best_sarimax_params(train_endog, train_exog)
            mod = sm.tsa.statespace.SARIMAX(train_endog, exog=train_exog, order=best_order, seasonal_order=best_seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
            fit_results = mod.fit(disp=False)
            
            # 使用 Helper 准备测试集 Exog
            test_exog_hybrid = prepare_exog_for_forecast(
                last_train_date=split_date, 
                steps=steps_test, 
                lag=optimal_enso_lag, 
                enso_series=sim_enso_history_inner
            )
            
            forecast_results = fit_results.get_forecast(steps=steps_test, exog=test_exog_hybrid)
        
        test_endog = test_df_period['Anomaly']
        rmse_all = np.sqrt(mean_squared_error(test_endog, forecast_results.predicted_mean))
        
        predictions_all = pd.Series(forecast_results.predicted_mean.values, index=test_df_period.index)
        predictions_same_month = predictions_all[predictions_all.index.month == target_month]
        actuals_same_month = test_df_same_month['Anomaly']
        rmse_same_month = np.sqrt(mean_squared_error(actuals_same_month, predictions_same_month))
        
        if config.USE_HYBRID_EVALUATION:
            if same_month_count >= config.SEASONAL_WEIGHT_MIN_SAMPLES_FOR_BASE:
                alpha = config.SEASONAL_WEIGHT_BASE
            else:
                alpha = config.SEASONAL_WEIGHT_DECAY.get(same_month_count, 0.3)
            rmse_hybrid = alpha * rmse_same_month + (1 - alpha) * rmse_all
            eval_metric = rmse_hybrid
        else:
            rmse_hybrid = rmse_same_month
            eval_metric = rmse_same_month

        race_results.append({
            'offset': months, 
            'eval_metric': eval_metric, 
            'order': best_order, 
            'seasonal_order': best_seasonal_order, 
            'train_end_date': split_date
        })

    if not race_results:
        return None, actual_value

    good_models = sorted(race_results, key=lambda x: x['eval_metric'])[:4]
    metric_values = np.array([m['eval_metric'] for m in good_models])
    weights = 1 / (metric_values ** 2)
    weights = weights / weights.sum()

    # 最终预测 (1步)
    # 我们需要预测的是 target_date 这一步
    # 此时我们有 sim_enso_history_main (截止到 cutoff_date)
    
    future_exog_final = prepare_exog_for_forecast(
        last_train_date=last_available_date,
        steps=1,
        lag=optimal_enso_lag,
        enso_series=sim_enso_history_main
    )

    # === 阶段3: 集成预测 & 偏差计算 ===
    ensemble_predictions = []
    ensemble_recent_errors = [] 
    
    # 验证窗口 (相对于 cutoff_date 的最近 N 个月)
    validation_end_date = last_available_date
    validation_start_date = last_available_date - relativedelta(months=config.BIAS_LOOKBACK_MONTHS - 1)
    val_idx = df_modern.index
    validation_dates = val_idx[(val_idx >= validation_start_date) & (val_idx <= validation_end_date)]

    for i, m in enumerate(good_models):
        final_train_df = df_modern.loc[:m['train_end_date']]
        final_train_endog = final_train_df['Anomaly']
        final_train_exog = final_train_df[['ENSO_ANOM_LAGGED']]
        
        with SuppressOutput():
            final_mod = sm.tsa.statespace.SARIMAX(
                final_train_endog, 
                exog=final_train_exog, 
                order=m['order'], 
                seasonal_order=m['seasonal_order'], 
                enforce_stationarity=False, 
                enforce_invertibility=False
            )
            final_fit_results = final_mod.fit(disp=False)
            final_forecast_results = final_fit_results.get_forecast(steps=1, exog=future_exog_final)
        
        ensemble_predictions.append(final_forecast_results.predicted_mean.iloc[0])
        
        # --- Bias Calculation ---
        if config.USE_BIAS_CORRECTION and not validation_dates.empty:
            try:
                # 1. In-Sample Part
                is_mask = validation_dates <= m['train_end_date']
                preds_is = []
                if is_mask.any():
                    start_is = validation_dates[is_mask][0]
                    end_is = validation_dates[is_mask][-1]
                    # In-sample prediction does not require external exog (uses internal)
                    preds_is = final_fit_results.get_prediction(start=start_is, end=end_is).predicted_mean.values
                
                # 2. Out-of-Sample Part
                # MUST cover from train_end+1 to valid_end to satisfy state space requirements
                preds_oos = []
                oos_target_end = validation_dates[-1] 
                
                if oos_target_end > m['train_end_date']:
                    last_train = m['train_end_date']
                    # Construct exog from last_train+1 to oos_target_end
                    oos_range = pd.date_range(start=last_train + relativedelta(months=1), end=oos_target_end, freq='MS')
                    
                    if not oos_range.empty:
                        oos_exog = df_modern.loc[oos_range, ['ENSO_ANOM_LAGGED']]
                        # Using get_forecast to bridge the gap
                        full_oos_pred = final_fit_results.get_forecast(steps=len(oos_range), exog=oos_exog).predicted_mean
                        
                        # Only keep the parts that are in validation_dates
                        valid_oos_dates = validation_dates[validation_dates > m['train_end_date']]
                        preds_oos = full_oos_pred.loc[valid_oos_dates].values

                combined_preds = np.concatenate([preds_is, preds_oos])
                
                # Verify length matches. Actuals is indexed by validation_dates
                actuals = df_modern.loc[validation_dates, 'Anomaly']
                
                if len(combined_preds) == len(actuals):
                     current_errors = actuals.values - combined_preds
                     ensemble_recent_errors.append(np.mean(current_errors))
                else:
                     print(f"Length mismatch in bias calc: Preds {len(combined_preds)} vs Actuals {len(actuals)}")
                     ensemble_recent_errors.append(0.0)
                     
            except Exception as e:
                print(f"Error in bias calc: {e}")
                ensemble_recent_errors.append(0.0)
        else:
             ensemble_recent_errors.append(0.0)

    base_prediction = np.average(ensemble_predictions, weights=weights)

    bias_val = 0.0
    if config.USE_BIAS_CORRECTION and ensemble_recent_errors:
        bias_val = np.average(ensemble_recent_errors, weights=weights)

    final_prediction = base_prediction + bias_val
    
    # 趋势调整
    adjustment_val = 0.0
    if config.USE_TREND_ADJUSTMENT:
        recent_data = df_modern['Anomaly']
        slope_short = calculate_trend_slope(recent_data.tail(config.TREND_LOOKBACK_SHORT))
        slope_long = calculate_trend_slope(recent_data.tail(config.TREND_LOOKBACK_LONG))
        
        if (slope_short * slope_long > 0) and (abs(slope_short) > abs(slope_long) * config.TREND_ACCELERATION_THRESHOLD):
            adjustment_val = slope_short * 1 * config.TREND_ADJUSTMENT_WEIGHT
            final_prediction += adjustment_val
            
    return final_prediction, actual_value

if __name__ == "__main__":
    # 从命令行参数获取要测试的月份列表
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python validate_robust_methods.py YYYYMM YYYYMM ...")
        sys.exit(1)
        
    months_to_test = sys.argv[1:]
    
    gistemp_data = data_handler.get_clean_data(config.DATA_FILE_PATH)
    enso_data = data_handler.get_enso_data(config.ENSO_DATA_URL)
    full_df = pd.concat([gistemp_data, enso_data], axis=1)
    full_df.dropna(inplace=True)
    
    # --- Lag Config ---
    optimal_enso_lag = 3
    # No global lagging here; run_backtest_for_month handles it per iteration to avoid leakage
    # But we need to ensure the columns exist?
    # full_df already has 'Anomaly' and 'ENSO_ANOM'.
    
    
    print(f"| 月份 | 预测值 | 实际值 | 误差 | 状态 |")
    print(f"|---|---|---|---|---|")
    
    mape_sum = 0
    count = 0
    
    for yyyymm in months_to_test:
        try:
            pred, actual = run_backtest_for_month(yyyymm, full_df, optimal_enso_lag)
            
            if pred is None:
                print(f"| {yyyymm} | 失败 | {actual} | - | ❌ |")
                continue
                
            error = pred - actual
            status = "✅" if abs(error) < 10 else "⚠️" # 误差小于0.1度算优秀
            
            print(f"| {yyyymm} | {pred:.2f} | {actual:.2f} | {error:+.2f} | {status} |")
            
            mape_sum += abs(error)
            count += 1
            
        except Exception as e:
             print(f"| {yyyymm} | Error | - | {str(e)} | ❌ |")
    
    if count > 0:
        print(f"\n平均绝对误差 (MAE): {mape_sum/count:.2f}")
