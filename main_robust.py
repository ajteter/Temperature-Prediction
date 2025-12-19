import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta

# 导入改进版的配置文件
import config_robust as config
import data_handler
import model

def calculate_bin_probabilities(prediction, std_err, bins):
    """根据预测值和标准误计算落入每个区间的概率"""
    probabilities = {}
    sorted_bins = sorted(bins.items(), key=lambda item: item[1][0] if item[1][0] is not None else -np.inf)
    for name, (lower, upper) in sorted_bins:
        if lower is None: prob = norm.cdf(upper, loc=prediction, scale=std_err)
        elif upper is None: prob = 1 - norm.cdf(lower, loc=prediction, scale=std_err)
        else: prob = norm.cdf(upper, loc=prediction, scale=std_err) - norm.cdf(lower, loc=prediction, scale=std_err)
        probabilities[name] = prob
    return probabilities

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
    准备用于预测的外生变量 (ENSO)，智能拼接已知数据和预测数据。
    
    Args:
        last_train_date: 训练数据的最后日期 (Temperature)
        steps: 需要预测的步数
        lag: 滞后期 (config.OPTIMAL_ENSO_LAG)
        enso_series: 截至目前的完整ENSO历史数据 (未滞后)
        enso_order, enso_seasonal_order: ENSO SARIMA模型的参数 (如果需要预测)
    """
    # 预测的目标日期范围
    future_dates = [last_train_date + relativedelta(months=i+1) for i in range(steps)]
    
    # 每个目标日期需要的ENSO数据日期 (T - lag)
    required_enso_dates = [d - relativedelta(months=lag) for d in future_dates]
    
    exog_values = []
    
    # 获取现有ENSO数据的截止日期
    last_enso_date = enso_series.index.max()
    
    # 检查是否需要预测ENSO (即: 需要的日期是否超出了现有的ENSO数据)
    max_required_date = max(required_enso_dates)
    
    enso_forecast_needed = max_required_date > last_enso_date
    enso_forecast_series = None
    
    if enso_forecast_needed:
        # 需要预测的月数
        months_to_forecast = (max_required_date.year - last_enso_date.year) * 12 + (max_required_date.month - last_enso_date.month)
        
        # 训练ENSO模型并预测
        # print(f"    [Info] Forecasting ENSO for {months_to_forecast} months...")
        if enso_order is None:
             enso_order, enso_seasonal_order = model.find_best_sarimax_params(enso_series, exog=None)
        
        forecast_res = model.train_and_forecast_sarimax(enso_series, enso_order, enso_seasonal_order, steps=months_to_forecast)
        enso_forecast_series = forecast_res.predicted_mean
        
    # 拼接数据
    from_real_count = 0
    from_forecast_count = 0
    
    for d in required_enso_dates:
        if d <= last_enso_date:
            exog_values.append(enso_series.loc[d])
            from_real_count += 1
        else:
            exog_values.append(enso_forecast_series.loc[d])
            from_forecast_count += 1
            
    # print(f"    [Exog Prep] Real: {from_real_count}, Forecast: {from_forecast_count}")
    
    return pd.DataFrame(exog_values, index=future_dates, columns=['ENSO_ANOM_LAGGED'])

def run_robust_forecast(gistemp_series, enso_series, optimal_enso_lag):
    """
    改进版预测流程：
    1. 混合评估策略 (Hybrid Evaluation)
    2. 智能外生变量拼接 (Smart Exogenous Stitching)
    3. 趋势加速调整 (Trend Adjustment)
    """
    
    print(f"ENSO Lag: {optimal_enso_lag} months")
    
    # 构造基础DataFrame用于训练 (History)
    # 注意：这里我们构造一个 align 好的 DataFrame，但仅用于方便切片
    # 实际预测时的 exogenous 会通过 prepare_exog_for_forecast 动态构建
    
    df_modern = pd.DataFrame({'Anomaly': gistemp_series})
    df_modern['ENSO_ANOM_LAGGED'] = enso_series.shift(optimal_enso_lag)
    df_modern = df_modern.loc["1970-01-01":]
    df_modern.dropna(inplace=True)
    
    print(f"数据范围 (Model Train): {df_modern.index.min().strftime('%Y-%m')} to {df_modern.index.max().strftime('%Y-%m')}\n")

    target_date = pd.to_datetime(config.TARGET_YYYYMM, format='%Y%m')
    target_month = target_date.month
    last_available_date = df_modern.index.max()
    
    print(f"目标月份: {target_date.strftime('%Y年%m月')} (第{target_month}月)")
    print("=== 阶段1: 混合评估赛马 (Hybrid Evaluation) ===")
    if config.USE_HYBRID_EVALUATION:
        print("策略: 启用混合RMSE (季节性RMSE + 全局RMSE)")
        print(f"基础季节性权重: {config.SEASONAL_WEIGHT_BASE}")
    else:
        print("策略: 仅使用季节性RMSE (旧版逻辑)")
    print("-" * 60)
    
    race_results = []

    # 遍历配置文件中定义的窗口选项
    for months in config.RACE_SPLIT_MONTHS:
        split_date = target_date - relativedelta(months=months)
        
        # 确保训练数据足够长 (至少3年)
        if split_date < df_modern.index.min() + relativedelta(months=36):
            continue
            
        train_df = df_modern.loc[:split_date]
        test_df_period = df_modern.loc[split_date + relativedelta(months=1):last_available_date]
        
        if test_df_period.empty:
            continue
            
        # 模拟回测环境：假设我们在 split_date 这一天
        # 1. Temperature 数据截止到 split_date
        # 2. ENSO 数据截止到 split_date (我们必须严格限制 ENSO 的获取)
        
        sim_enso_history = enso_series.loc[:split_date]
        
        # 准备测试集的 Exogenous 数据
        # 注意：这里是关键改进点。我们不能直接用 test_df_period 里的 ENSO_ANOM_LAGGED,
        # 因为那里面包含了一部分未来的真实ENSO数据（如果 lag 窗口外）。
        # 我们必须用 prepare_exog_for_forecast 来模拟真实的预测过程。
        
        steps_test = len(test_df_period)
        
        # 只有当测试集有数据时才进行
        test_df_same_month = test_df_period[test_df_period.index.month == target_month]
        same_month_count = len(test_df_same_month)
        
        if same_month_count == 0:
            continue

        train_endog = train_df['Anomaly']
        train_exog = train_df[['ENSO_ANOM_LAGGED']]
        
        # 训练模型
        best_order, best_seasonal_order = model.find_best_sarimax_params(train_endog, train_exog)
        mod = sm.tsa.statespace.SARIMAX(train_endog, exog=train_exog, order=best_order, seasonal_order=best_seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        fit_results = mod.fit(disp=False)
        
        # 构建测试用的 Exog
        # 这里会根据 lag=3，自动把前3个月的 exog 设为已知，后面的设为预测
        test_exog_hybrid = prepare_exog_for_forecast(
            last_train_date=split_date, 
            steps=steps_test, 
            lag=optimal_enso_lag, 
            enso_series=sim_enso_history
        )
        
        # 预测
        forecast_results = fit_results.get_forecast(steps=steps_test, exog=test_exog_hybrid)
        
        # --- 计算各类误差 ---
        test_endog = test_df_period['Anomaly']
        
        # A. 全局RMSE
        rmse_all = np.sqrt(mean_squared_error(test_endog, forecast_results.predicted_mean))
        
        # B. 同月RMSE
        predictions_all = pd.Series(forecast_results.predicted_mean.values, index=test_df_period.index)
        predictions_same_month = predictions_all[predictions_all.index.month == target_month]
        actuals_same_month = test_df_same_month['Anomaly']
        rmse_same_month = np.sqrt(mean_squared_error(actuals_same_month, predictions_same_month))
        
        # C. 混合RMSE
        if config.USE_HYBRID_EVALUATION:
            if same_month_count >= config.SEASONAL_WEIGHT_MIN_SAMPLES_FOR_BASE:
                alpha = config.SEASONAL_WEIGHT_BASE
            else:
                alpha = config.SEASONAL_WEIGHT_DECAY.get(same_month_count, 0.3)
            rmse_hybrid = alpha * rmse_same_month + (1 - alpha) * rmse_all
            eval_metric = rmse_hybrid
        else:
            alpha = 1.0
            rmse_hybrid = rmse_same_month
            eval_metric = rmse_same_month

        race_results.append({
            'offset': months,
            'rmse_all': rmse_all,
            'rmse_same_month': rmse_same_month,
            'eval_metric': eval_metric,
            'order': best_order, 
            'seasonal_order': best_seasonal_order, 
            'train_end_date': split_date,
            'alpha': alpha,
            'same_month_count': same_month_count
        })
        
        print(f"窗口 -{months:>2}月 | 样本数:{same_month_count} | 全局RMSE:{rmse_all:>5.2f} | 同月RMSE:{rmse_same_month:>5.2f} | 混合RMSE:{rmse_hybrid:>5.2f}")

    if not race_results:
        raise ValueError("所有赛马实验均失败，请检查数据范围")

    # 根据评估指标前4
    good_models = sorted(race_results, key=lambda x: x['eval_metric'])[:4]
    
    # 权重计算
    metric_values = np.array([m['eval_metric'] for m in good_models])
    weights = 1 / (metric_values ** 2)
    weights = weights / weights.sum()
    
    print('\n--- 选入集成的 Top 4 模型 ---')
    print("偏移 | 混合RMSE | 权重   ")
    print("-----|----------|--------")
    for i, m in enumerate(good_models):
        print(f"-{m['offset']:>2}月 | {m['eval_metric']:>8.2f} | {weights[i]:>6.1%}")

    # === 阶段2: 最终预测 Prep ===
    steps_to_forecast = (target_date.year - last_available_date.year) * 12 + (target_date.month - last_available_date.month)
    
    # 准备最终预测用的 Exogenous (使用当前所有可用的 ENSO 数据)
    print(f"\n=== 阶段2: 准备预测变量 (Steps={steps_to_forecast}) ===")
    print("利用最新ENSO数据进行智能拼接...")
    
    future_exog_final = prepare_exog_for_forecast(
        last_train_date=last_available_date,
        steps=steps_to_forecast,
        lag=optimal_enso_lag,
        enso_series=enso_series # 使用完整的 real ENSO history
    )
    
    # === 阶段3: 集成预测 ===
    ensemble_predictions = []
    ensemble_std_errs = []
    
    # === 阶段 3.5: 残差偏差调整 (Residual Bias Correction) ===
    bias_correction_val = 0.0
    if config.USE_BIAS_CORRECTION:
        print(f"\n=== 阶段3.5: 残差偏差修正 (回溯 {config.BIAS_LOOKBACK_MONTHS} 月) ===")
        
        # 定义验证窗口 (最近 N 个月)
        validation_end_date = last_available_date
        validation_start_date = last_available_date - relativedelta(months=config.BIAS_LOOKBACK_MONTHS - 1)
        validation_range = pd.date_range(start=validation_start_date, end=validation_end_date, freq='MS')
        
        print(f"计算偏差窗口: {validation_start_date.strftime('%Y-%m')} 到 {validation_end_date.strftime('%Y-%m')}")
        
        model_biases = []
        
        for i, m in enumerate(good_models):
            # 获取该模型对验证窗口的预测
            # 注意: final_fit_results 已经在阶段3中训练好了，我们不需要重新fit，但我们需要它的对象
            # 为了代码整洁，我们不得不在这里重新通过 'm' 来获取预测，或者在阶段3保存 results
            # 我们选择在阶段3循环中顺便计算，或者重构。
            # 鉴于代码结构，我们直接在这里重新构建预测 (虽然有点重复计算，但逻辑清晰)
            
            final_train_df = df_modern.loc[:m['train_end_date']]
            final_train_endog = final_train_df['Anomaly']
            final_train_exog = final_train_df[['ENSO_ANOM_LAGGED']]

            # 我们需要验证窗口的 exog
            # 验证窗口的数据都在 df_modern 中
            validation_exog = df_modern.loc[validation_range, ['ENSO_ANOM_LAGGED']]
             
            with model.SuppressOutput():
                 # 重新加载模型 (因为fit_results在循环中被覆盖了，或者我们可以重构上面的循环)
                 # 为了避免重复训练带来的性能损耗，我们应该重构阶段3
                 pass
        
        # --- 重构: 将偏差计算合并到阶段3循环中 ---
        
    # 重写 阶段3 包括 偏差计算
    print(f"\n=== 阶段3: 执行集成预测 & 偏差计算 ===")
    
    ensemble_predictions = []
    ensemble_std_errs = []
    ensemble_recent_errors = [] # 存储每个模型在最近窗口的平均误差 (Actual - Predict)

    # 准备验证窗口数据 (用于计算偏差)
    validation_end_date = last_available_date
    validation_start_date = last_available_date - relativedelta(months=config.BIAS_LOOKBACK_MONTHS - 1)
    # 确保 data_range 在 df_modern 范围内
    val_idx = df_modern.index
    validation_dates = val_idx[(val_idx >= validation_start_date) & (val_idx <= validation_end_date)]
    
    if len(validation_dates) < config.BIAS_LOOKBACK_MONTHS:
        print(f"⚠️ 警告: 可用历史数据不足以进行完整的 {config.BIAS_LOOKBACK_MONTHS} 个月偏差回测")

    for i, m in enumerate(good_models):
        final_train_df = df_modern.loc[:m['train_end_date']]
        final_train_endog = final_train_df['Anomaly']
        final_train_exog = final_train_df[['ENSO_ANOM_LAGGED']]
        
        final_mod = sm.tsa.statespace.SARIMAX(
            final_train_endog, 
            exog=final_train_exog, 
            order=m['order'], 
            seasonal_order=m['seasonal_order'], 
            enforce_stationarity=False, 
            enforce_invertibility=False
        )
        final_fit_results = final_mod.fit(disp=False)
        final_forecast_results = final_fit_results.get_forecast(steps=steps_to_forecast, exog=future_exog_final)
        
        pred_mean = final_forecast_results.predicted_mean.iloc[-1]
        conf = final_forecast_results.conf_int().iloc[-1]
        std_err = (conf.iloc[1] - conf.iloc[0]) / (2 * 1.95996)
        
        ensemble_predictions.append(pred_mean)
        ensemble_std_errs.append(std_err)
        
        # --- 计算该模型的近期偏差 ---
        if config.USE_BIAS_CORRECTION and not validation_dates.empty:
            try:
                # 1. In-Sample Part
                is_mask = validation_dates <= m['train_end_date']
                preds_is = []
                if is_mask.any():
                    start_is = validation_dates[is_mask][0]
                    end_is = validation_dates[is_mask][-1]
                    preds_is = final_fit_results.get_prediction(start=start_is, end=end_is).predicted_mean.values
                
                # 2. Out-of-Sample Part
                preds_oos = []
                oos_target_end = validation_dates[-1] 
                
                if oos_target_end > m['train_end_date']:
                    last_train = m['train_end_date']
                    oos_range = pd.date_range(start=last_train + relativedelta(months=1), end=oos_target_end, freq='MS')
                    
                    if not oos_range.empty:
                        oos_exog = df_modern.loc[oos_range, ['ENSO_ANOM_LAGGED']]
                        full_oos_pred = final_fit_results.get_forecast(steps=len(oos_range), exog=oos_exog).predicted_mean
                        valid_oos_dates = validation_dates[validation_dates > m['train_end_date']]
                        preds_oos = full_oos_pred.loc[valid_oos_dates].values

                combined_preds = np.concatenate([preds_is, preds_oos])
                
                actuals = df_modern.loc[validation_dates, 'Anomaly']
                
                if len(combined_preds) == len(actuals):
                     current_errors = actuals.values - combined_preds
                     avg_error = np.mean(current_errors)
                     ensemble_recent_errors.append(avg_error)
                else:
                     print(f"  [Bias Warning] Length mismatch: Preds {len(combined_preds)} vs Actuals {len(actuals)}")
                     ensemble_recent_errors.append(0.0)
                     
            except Exception as e:
                print(f"  [Bias Warning] Error: {e}")
                ensemble_recent_errors.append(0.0)

        else:
             ensemble_recent_errors.append(0.0) # No validation data

        print(f"模型{i+1} (-{m['offset']:>2}月): {pred_mean:>6.2f} (SD:{std_err:.2f}) | 近期偏差({len(validation_dates)}月): {ensemble_recent_errors[-1]:+.3f}")

    base_prediction = np.average(ensemble_predictions, weights=weights)
    final_std_err = np.sqrt(np.average(np.array(ensemble_std_errs)**2, weights=weights))
    
    # 计算加权平均偏差
    weighted_bias = 0.0
    if config.USE_BIAS_CORRECTION and ensemble_recent_errors:
        weighted_bias = np.average(ensemble_recent_errors, weights=weights)
        print(f"\n基础集成预测值: {base_prediction:.2f}")
        print(f"加权残差偏差: {weighted_bias:+.3f}")
        
    bias_correction_val = weighted_bias

    # === 阶段4: 趋势调整 (Trend Adjustment) ===
    final_prediction = base_prediction + bias_correction_val
    adjustment_val = 0.0
    
    if config.USE_TREND_ADJUSTMENT:
        print(f"\n=== 阶段4: 趋势稳定性检查与调整 ===")
        # 获取最新的数据进行趋势检测
        # 注意: 这里的趋势检测是针对 df_modern 全量数据的尾部
        recent_data = df_modern.loc[:last_available_date, 'Anomaly']
        
        slope_short = calculate_trend_slope(recent_data.tail(config.TREND_LOOKBACK_SHORT))
        slope_long = calculate_trend_slope(recent_data.tail(config.TREND_LOOKBACK_LONG))
        
        print(f"短期趋势 ({config.TREND_LOOKBACK_SHORT}月): {slope_short:.3f}/月")
        print(f"长期趋势 ({config.TREND_LOOKBACK_LONG}月): {slope_long:.3f}/月")
        
        is_accelerating = False
        # 只有当两者同号，且短期趋势绝对值显著大于长期趋势绝对值时，才认为是加速
        
        if (slope_short * slope_long > 0) and (abs(slope_short) > abs(slope_long) * config.TREND_ACCELERATION_THRESHOLD):
            # ...
            final_prediction += adjustment_val # Add Trend Adjustment
        else:
            print("✓ 趋势相对稳定，无需调整。")
            
    # === 结果输出 ===
    print("\n" + "="*50)
    print("=== 最终稳健预测结果 (Robust Forecast) ===")
    print("="*50)
    print(f"目标月份: {config.TARGET_YYYYMM}")
    print(f"基础预测: {base_prediction:.2f}")
    if config.USE_BIAS_CORRECTION and bias_correction_val != 0:
        print(f"偏差修正: {bias_correction_val:+.2f}")
    if config.USE_TREND_ADJUSTMENT and adjustment_val != 0:
        print(f"趋势调整: {adjustment_val:+.2f}")
    print(f"最终预测: {final_prediction:.2f}")
    print(f"95%置信区间: [{final_prediction - 1.96*final_std_err:.2f}, {final_prediction + 1.96*final_std_err:.2f}]")
    
    probabilities = calculate_bin_probabilities(final_prediction * 100, final_std_err * 100, config.PREDICTION_BINS)
    
    print("\n--- 预测概率分布 ---")
    for bin_name, prob in probabilities.items():
        bar = '█' * int(prob * 50)
        print(f"  {bin_name:>10}: {prob:>6.2%} {bar}")
    
    return final_prediction, probabilities

if __name__ == "__main__":
    try:
        print("=== 气候温度预测系统: 稳健增强版 (Robust Edition) ===\n")
        
        gistemp_data = data_handler.get_clean_data(config.DATA_FILE_PATH)
        enso_data = data_handler.get_enso_data(config.ENSO_DATA_URL)
        run_robust_forecast(gistemp_data, enso_data, optimal_enso_lag=config.OPTIMAL_ENSO_LAG)

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
