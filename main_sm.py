"""
main_sm.py — Statsmodels 增强版预测主流程
==========================================
基于 main_robust.py 核心策略 (赛马 → 集成 → 偏差修正 → 趋势调整),
引入 statsmodels skill 最佳实践:
  - 阶段 0: ADF/KPSS 平稳性诊断
  - 阶段 1: SARIMAX + ETS 多算法赛马
  - 阶段 2: 集成预测 + 残差诊断
  - 阶段 3: 偏差修正 + 趋势调整
  - 阶段 4: 结构化模型比较表 + 概率分布
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta

import config_sm as config
import data_handler
import model_sm as msm


# ══════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════

def calculate_bin_probabilities(prediction, std_err, bins):
    """根据预测值和标准误计算落入每个区间的概率"""
    probabilities = {}
    sorted_bins = sorted(bins.items(), key=lambda item: item[1][0] if item[1][0] is not None else -np.inf)
    for name, (lower, upper) in sorted_bins:
        if lower is None:
            prob = norm.cdf(upper, loc=prediction, scale=std_err)
        elif upper is None:
            prob = 1 - norm.cdf(lower, loc=prediction, scale=std_err)
        else:
            prob = norm.cdf(upper, loc=prediction, scale=std_err) - norm.cdf(lower, loc=prediction, scale=std_err)
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


def prepare_exog_for_forecast(last_train_date, steps, lag, enso_series):
    """
    准备用于预测的外生变量 (ENSO)，智能拼接已知数据和预测数据。
    """
    future_dates = [last_train_date + relativedelta(months=i + 1) for i in range(steps)]
    required_enso_dates = [d - relativedelta(months=lag) for d in future_dates]

    exog_values = []
    last_enso_date = enso_series.index.max()
    max_required_date = max(required_enso_dates)

    enso_forecast_series = None

    if max_required_date > last_enso_date:
        months_to_forecast = (max_required_date.year - last_enso_date.year) * 12 + \
                             (max_required_date.month - last_enso_date.month)
        # 用 SARIMAX 自动搜参预测 ENSO
        params = msm.find_best_sarimax_params_sm(enso_series, exog=None)
        if params:
            forecast_res = msm.forecast_sarimax(
                msm.train_sarimax(enso_series, exog=None, order=params['order'], seasonal_order=params['seasonal_order']),
                steps=months_to_forecast
            )
            enso_forecast_series = forecast_res.predicted_mean

    for d in required_enso_dates:
        if d <= last_enso_date:
            exog_values.append(enso_series.loc[d])
        else:
            exog_values.append(enso_forecast_series.loc[d])

    return pd.DataFrame(exog_values, index=future_dates, columns=['ENSO_ANOM_LAGGED'])


# ══════════════════════════════════════════════════════════════════
# 赛马中的模型评估器
# ══════════════════════════════════════════════════════════════════

def evaluate_candidate_on_window(
    candidate_type, train_endog, train_exog, test_df_period,
    split_date, target_month, optimal_enso_lag, sim_enso_history
):
    """
    在一个窗口上评估单个候选模型，返回结果 dict 或 None。

    candidate_type: 'SARIMAX' 或 'ETS'
    """
    steps_test = len(test_df_period)
    test_df_same_month = test_df_period[test_df_period.index.month == target_month]
    same_month_count = len(test_df_same_month)

    if same_month_count == 0:
        return None

    result = {
        'candidate_type': candidate_type,
        'train_end_date': split_date,
        'same_month_count': same_month_count,
    }

    try:
        if candidate_type == 'SARIMAX':
            # 网格搜索
            params = msm.find_best_sarimax_params_sm(train_endog, train_exog)
            if params is None:
                return None

            result['order'] = params['order']
            result['seasonal_order'] = params['seasonal_order']
            result['aic'] = params['aic']
            result['bic'] = params['bic']

            # 训练
            fit_results = msm.train_sarimax(train_endog, train_exog, params['order'], params['seasonal_order'])
            result['fit_results'] = fit_results

            # 构建测试 Exog
            test_exog = prepare_exog_for_forecast(
                last_train_date=split_date,
                steps=steps_test,
                lag=optimal_enso_lag,
                enso_series=sim_enso_history
            )

            # 预测
            forecast_results = fit_results.get_forecast(steps=steps_test, exog=test_exog)
            predicted = forecast_results.predicted_mean

            # 残差诊断
            result['diagnostics'] = msm.diagnose_residuals(fit_results.resid, lags=config.LJUNGBOX_LAGS)

        elif candidate_type == 'ETS':
            ets_result = msm.fit_ets_model(train_endog, seasonal_periods=config.ETS_SEASONAL_PERIODS)
            if ets_result is None:
                return None

            result['order'] = ets_result['model_type']
            result['seasonal_order'] = f"s={config.ETS_SEASONAL_PERIODS}"
            result['aic'] = ets_result['aic']
            result['bic'] = ets_result['bic']
            result['ets_results'] = ets_result['results']

            # ETS 预测 (不支持 exog)
            predicted = ets_result['results'].forecast(steps_test)

            # 残差诊断
            result['diagnostics'] = msm.diagnose_residuals(ets_result['results'].resid, lags=config.LJUNGBOX_LAGS)

        else:
            return None

        # 通用评估
        test_endog = test_df_period['Anomaly']

        # A. 全局 RMSE
        rmse_all = np.sqrt(mean_squared_error(test_endog, predicted.values[:len(test_endog)]))

        # B. 同月 RMSE
        predictions_series = pd.Series(predicted.values[:len(test_endog)], index=test_df_period.index)
        predictions_same_month = predictions_series[predictions_series.index.month == target_month]
        actuals_same_month = test_df_same_month['Anomaly']
        rmse_same_month = np.sqrt(mean_squared_error(actuals_same_month, predictions_same_month))

        # C. 混合 RMSE
        if config.USE_HYBRID_EVALUATION:
            if same_month_count >= config.SEASONAL_WEIGHT_MIN_SAMPLES_FOR_BASE:
                alpha = config.SEASONAL_WEIGHT_BASE
            else:
                alpha = config.SEASONAL_WEIGHT_DECAY.get(same_month_count, 0.3)
            rmse_hybrid = alpha * rmse_same_month + (1 - alpha) * rmse_all
        else:
            alpha = 1.0
            rmse_hybrid = rmse_same_month

        result['rmse_all'] = rmse_all
        result['rmse_same_month'] = rmse_same_month
        result['eval_metric'] = rmse_hybrid
        result['alpha'] = alpha

        return result

    except Exception as e:
        # 静默失败，返回 None
        return None


# ══════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════

def run_sm_forecast(gistemp_series, enso_series, optimal_enso_lag):
    """
    Statsmodels 增强版预测流程。
    """

    print(f"ENSO Lag: {optimal_enso_lag} months")

    # 构造训练用 DataFrame
    df_modern = pd.DataFrame({'Anomaly': gistemp_series})
    df_modern['ENSO_ANOM_LAGGED'] = enso_series.shift(optimal_enso_lag)
    df_modern = df_modern.loc["1970-01-01":]
    df_modern.dropna(inplace=True)

    print(f"数据范围 (Model Train): {df_modern.index.min().strftime('%Y-%m')} to {df_modern.index.max().strftime('%Y-%m')}\n")

    target_date = pd.to_datetime(config.TARGET_YYYYMM, format='%Y%m')
    target_month = target_date.month
    last_available_date = df_modern.index.max()

    print(f"目标月份: {target_date.strftime('%Y年%m月')} (第{target_month}月)")

    # ════════════════════════════════════════════════════
    # 阶段 0: 平稳性诊断
    # ════════════════════════════════════════════════════
    if config.PRINT_STATIONARITY_DIAGNOSTICS:
        print("\n=== 阶段0: 平稳性诊断 (ADF + KPSS) ===")
        print("-" * 50)

        report_temp = msm.check_stationarity(df_modern['Anomaly'], name="Temperature Anomaly")
        msm.print_stationarity_report(report_temp)

        report_enso = msm.check_stationarity(df_modern['ENSO_ANOM_LAGGED'], name="ENSO (Lagged)")
        msm.print_stationarity_report(report_enso)

    # ════════════════════════════════════════════════════
    # 阶段 1: 多模型赛马
    # ════════════════════════════════════════════════════
    print(f"\n=== 阶段1: 多模型赛马 (Hybrid Evaluation) ===")
    model_types = ['SARIMAX']
    if config.USE_ETS_CANDIDATE:
        model_types.append('ETS')
    print(f"候选模型类型: {', '.join(model_types)}")
    if config.USE_HYBRID_EVALUATION:
        print(f"策略: 混合RMSE (季节性权重={config.SEASONAL_WEIGHT_BASE})")
    print("-" * 80)

    race_results = []

    for months in config.RACE_SPLIT_MONTHS:
        split_date = target_date - relativedelta(months=months)

        if split_date < df_modern.index.min() + relativedelta(months=36):
            continue

        train_df = df_modern.loc[:split_date]
        test_df_period = df_modern.loc[split_date + relativedelta(months=1):last_available_date]

        if test_df_period.empty:
            continue

        sim_enso_history = enso_series.loc[:split_date]

        train_endog = train_df['Anomaly']
        train_exog = train_df[['ENSO_ANOM_LAGGED']]

        for ctype in model_types:
            result = evaluate_candidate_on_window(
                candidate_type=ctype,
                train_endog=train_endog,
                train_exog=train_exog,
                test_df_period=test_df_period,
                split_date=split_date,
                target_month=target_month,
                optimal_enso_lag=optimal_enso_lag,
                sim_enso_history=sim_enso_history
            )

            if result is not None:
                result['offset'] = months
                race_results.append(result)

                diag_str = ""
                if config.PRINT_RESIDUAL_DIAGNOSTICS and 'diagnostics' in result:
                    d = result['diagnostics']
                    lb = f"LB-p={d['ljungbox_pvalue']:.2f}{'✓' if d['ljungbox_pass'] else '✗'}"
                    jb = f"JB-p={d['jarque_bera_pvalue']:.2f}{'✓' if d['jarque_bera_pass'] else '✗'}"
                    diag_str = f" | {lb} {jb}"

                order_str = str(result['order'])
                print(f"  -{months:>2}月 {ctype:<7} | {order_str:<14} | "
                      f"AIC:{result['aic']:>8.1f} | "
                      f"全局:{result['rmse_all']:>5.2f} 同月:{result['rmse_same_month']:>5.2f} "
                      f"混合:{result['eval_metric']:>5.2f}{diag_str}")

    if not race_results:
        raise ValueError("所有赛马实验均失败，请检查数据范围")

    # ════════════════════════════════════════════════════
    # 选择 Top-N
    # ════════════════════════════════════════════════════
    good_models = sorted(race_results, key=lambda x: x['eval_metric'])[:config.ENSEMBLE_TOP_N]

    metric_values = np.array([m['eval_metric'] for m in good_models])
    weights = 1 / (metric_values ** 2)
    weights = weights / weights.sum()

    print(f"\n--- 选入集成的 Top {config.ENSEMBLE_TOP_N} 模型 ---")
    print(f"{'类型':<8} {'偏移':>4} {'混合RMSE':>8} {'权重':>8} {'AIC':>10}")
    print("-" * 45)
    for i, m in enumerate(good_models):
        print(f"{m['candidate_type']:<8} -{m['offset']:>2}月 {m['eval_metric']:>8.2f} {weights[i]:>7.1%} {m['aic']:>10.1f}")

    # ════════════════════════════════════════════════════
    # 阶段 2: 集成预测
    # ════════════════════════════════════════════════════
    steps_to_forecast = (target_date.year - last_available_date.year) * 12 + \
                        (target_date.month - last_available_date.month)

    print(f"\n=== 阶段2: 集成预测 (Steps={steps_to_forecast}) ===")

    # 预先准备最终 Exog
    future_exog_final = prepare_exog_for_forecast(
        last_train_date=last_available_date,
        steps=steps_to_forecast,
        lag=optimal_enso_lag,
        enso_series=enso_series
    )

    ensemble_predictions = []
    ensemble_std_errs = []
    ensemble_recent_errors = []

    # 偏差验证窗口
    validation_end_date = last_available_date
    validation_start_date = last_available_date - relativedelta(months=config.BIAS_LOOKBACK_MONTHS - 1)
    val_idx = df_modern.index
    validation_dates = val_idx[(val_idx >= validation_start_date) & (val_idx <= validation_end_date)]

    for i, m in enumerate(good_models):
        final_train_df = df_modern.loc[:m['train_end_date']]
        final_train_endog = final_train_df['Anomaly']
        final_train_exog = final_train_df[['ENSO_ANOM_LAGGED']]

        if m['candidate_type'] == 'SARIMAX':
            # 重新训练 SARIMAX
            fit_res = msm.train_sarimax(final_train_endog, final_train_exog, m['order'], m['seasonal_order'])
            forecast_res = fit_res.get_forecast(steps=steps_to_forecast, exog=future_exog_final)

            pred_mean = forecast_res.predicted_mean.iloc[-1]
            conf = forecast_res.conf_int().iloc[-1]
            std_err = (conf.iloc[1] - conf.iloc[0]) / (2 * 1.95996)

            # 偏差计算
            bias = _compute_bias_for_model(fit_res, m, df_modern, validation_dates)

        elif m['candidate_type'] == 'ETS':
            # 重新训练 ETS
            ets_result = msm.fit_ets_model(final_train_endog, seasonal_periods=config.ETS_SEASONAL_PERIODS)
            if ets_result is None:
                continue
            forecast_vals = ets_result['results'].forecast(steps_to_forecast)
            pred_mean = forecast_vals.iloc[-1]
            # ETS 没有内置的 conf_int，用残差标准差估计
            std_err = np.std(ets_result['results'].resid.dropna())

            # ETS 偏差计算 (简化: 用最近残差均值)
            recent_resid = ets_result['results'].resid.dropna().tail(config.BIAS_LOOKBACK_MONTHS)
            bias = recent_resid.mean() if len(recent_resid) > 0 else 0.0

        ensemble_predictions.append(pred_mean)
        ensemble_std_errs.append(std_err)
        ensemble_recent_errors.append(bias)

        print(f"  模型{i + 1} {m['candidate_type']:<7} (-{m['offset']:>2}月): "
              f"{pred_mean:>6.2f} (SD:{std_err:.2f}) | 偏差: {bias:+.3f}")

    base_prediction = np.average(ensemble_predictions, weights=weights)
    final_std_err = np.sqrt(np.average(np.array(ensemble_std_errs) ** 2, weights=weights))

    # ════════════════════════════════════════════════════
    # 阶段 3: 偏差修正 + 趋势调整
    # ════════════════════════════════════════════════════
    bias_correction_val = 0.0
    if config.USE_BIAS_CORRECTION and ensemble_recent_errors:
        bias_correction_val = np.average(ensemble_recent_errors, weights=weights)

    final_prediction = base_prediction + bias_correction_val
    adjustment_val = 0.0

    if config.USE_TREND_ADJUSTMENT:
        print(f"\n=== 阶段3: 趋势调整 ===")
        recent_data = df_modern.loc[:last_available_date, 'Anomaly']
        slope_short = calculate_trend_slope(recent_data.tail(config.TREND_LOOKBACK_SHORT))
        slope_long = calculate_trend_slope(recent_data.tail(config.TREND_LOOKBACK_LONG))

        print(f"  短期趋势 ({config.TREND_LOOKBACK_SHORT}月): {slope_short:.4f}/月")
        print(f"  长期趋势 ({config.TREND_LOOKBACK_LONG}月): {slope_long:.4f}/月")

        if (slope_short * slope_long > 0) and \
           (abs(slope_short) > abs(slope_long) * config.TREND_ACCELERATION_THRESHOLD):
            adjustment_val = slope_short * steps_to_forecast * config.TREND_ADJUSTMENT_WEIGHT
            final_prediction += adjustment_val
            print(f"  ⚡ 趋势加速，调整: {adjustment_val:+.3f}")
        else:
            print("  ✓ 趋势稳定，无需调整。")

    # ════════════════════════════════════════════════════
    # 阶段 4: 结果输出
    # ════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("=== 最终预测结果 (Statsmodels Enhanced Edition) ===")
    print("=" * 60)
    print(f"目标月份: {config.TARGET_YYYYMM}")
    print(f"基础集成预测: {base_prediction:.2f}")
    if config.USE_BIAS_CORRECTION and bias_correction_val != 0:
        print(f"偏差修正: {bias_correction_val:+.3f}")
    if config.USE_TREND_ADJUSTMENT and adjustment_val != 0:
        print(f"趋势调整: {adjustment_val:+.3f}")
    print(f"最终预测: {final_prediction:.2f}")
    print(f"95%置信区间: [{final_prediction - 1.96 * final_std_err:.2f}, {final_prediction + 1.96 * final_std_err:.2f}]")

    # 概率分布
    probabilities = calculate_bin_probabilities(
        final_prediction * 100, final_std_err * 100, config.PREDICTION_BINS
    )

    print("\n--- 预测概率分布 ---")
    for bin_name, prob in probabilities.items():
        bar = '█' * int(prob * 50)
        print(f"  {bin_name:>10}: {prob:>6.2%} {bar}")

    # ════════════════════════════════════════════════════
    # 模型比较表
    # ════════════════════════════════════════════════════
    if config.PRINT_MODEL_COMPARISON_TABLE:
        print("\n--- 集成模型对比表 ---")
        header = f"{'#':>2} {'类型':<8} {'偏移':>4} {'阶数':<16} {'AIC':>10} {'BIC':>10} {'RMSE':>6} {'权重':>7}"
        print(header)
        print("-" * len(header))
        for i, m in enumerate(good_models):
            order_str = str(m['order'])
            bic_val = m.get('bic', np.nan)
            print(f"{i + 1:>2} {m['candidate_type']:<8} -{m['offset']:>2}月 {order_str:<16} "
                  f"{m['aic']:>10.1f} {bic_val:>10.1f} {m['eval_metric']:>6.2f} {weights[i]:>6.1%}")

    return final_prediction, probabilities


# ══════════════════════════════════════════════════════════════════
# 偏差计算辅助 (SARIMAX)
# ══════════════════════════════════════════════════════════════════

def _compute_bias_for_model(fit_results, model_info, df_modern, validation_dates):
    """
    计算 SARIMAX 模型在验证窗口上的平均偏差(Actual - Predicted)。
    """
    if not config.USE_BIAS_CORRECTION or validation_dates.empty:
        return 0.0

    try:
        # In-sample 部分
        is_mask = validation_dates <= model_info['train_end_date']
        preds_is = []
        if is_mask.any():
            start_is = validation_dates[is_mask][0]
            end_is = validation_dates[is_mask][-1]
            preds_is = fit_results.get_prediction(start=start_is, end=end_is).predicted_mean.values

        # Out-of-sample 部分
        preds_oos = []
        oos_target_end = validation_dates[-1]

        if oos_target_end > model_info['train_end_date']:
            last_train = model_info['train_end_date']
            oos_range = pd.date_range(
                start=last_train + relativedelta(months=1),
                end=oos_target_end, freq='MS'
            )
            if not oos_range.empty:
                oos_exog = df_modern.loc[oos_range, ['ENSO_ANOM_LAGGED']]
                full_oos_pred = fit_results.get_forecast(
                    steps=len(oos_range), exog=oos_exog
                ).predicted_mean
                valid_oos_dates = validation_dates[validation_dates > model_info['train_end_date']]
                preds_oos = full_oos_pred.loc[valid_oos_dates].values

        combined_preds = np.concatenate([preds_is, preds_oos])
        actuals = df_modern.loc[validation_dates, 'Anomaly']

        if len(combined_preds) == len(actuals):
            return np.mean(actuals.values - combined_preds)
        else:
            return 0.0

    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════

def get_enso_with_cache(url, cache_path="enso_cache.txt"):
    """
    带本地缓存的 ENSO 数据获取。
    1. 尝试从 URL 获取（成功后自动缓存到本地文件）
    2. 如果网络失败，回退到本地缓存
    3. 如果缓存也不存在，给出明确提示
    """
    import os
    from io import StringIO

    # 尝试在线获取
    try:
        print("  [ENSO] 尝试从 NOAA 在线获取...", end="")
        enso_data = data_handler.get_enso_data(url)
        print(" ✓")

        # 成功后，缓存原始数据到本地
        try:
            import requests
            response = requests.get(url, timeout=15)
            with open(cache_path, 'w') as f:
                f.write(response.text)
            print(f"  [ENSO] 数据已缓存到 {cache_path}")
        except Exception:
            pass  # 缓存失败不影响主流程

        return enso_data

    except Exception as e:
        print(f" ✗ ({type(e).__name__})")

        # 回退到本地缓存
        if os.path.exists(cache_path):
            print(f"  [ENSO] 使用本地缓存: {cache_path}")
            enso_df = pd.read_csv(cache_path, sep=r'\s+', engine='python')
            enso_df['Date'] = pd.to_datetime(
                enso_df['YR'].astype(str) + '-' + enso_df['MON'].astype(str)
            )
            enso_df.set_index('Date', inplace=True)
            enso_series = enso_df['ANOM'].rename('ENSO_ANOM')
            return enso_series
        else:
            raise RuntimeError(
                f"无法获取 ENSO 数据且本地无缓存文件。\n"
                f"请手动下载 ENSO 数据并保存为 '{cache_path}':\n"
                f"  URL: {url}\n"
                f"  或者在网络正常时运行一次脚本以自动缓存。"
            )


if __name__ == "__main__":
    try:
        print("=" * 60)
        print("  气候温度预测系统: Statsmodels 增强版")
        print("  (SARIMAX + ETS | ADF/KPSS | Ljung-Box/Jarque-Bera)")
        print("=" * 60 + "\n")

        gistemp_data = data_handler.get_clean_data(config.DATA_FILE_PATH)
        enso_data = get_enso_with_cache(config.ENSO_DATA_URL)
        run_sm_forecast(gistemp_data, enso_data, optimal_enso_lag=config.OPTIMAL_ENSO_LAG)

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
