"""
model_sm.py — Statsmodels 增强版模型模块
=========================================
基于 statsmodels skill 最佳实践，提供：
  1. ADF + KPSS 双重平稳性检验
  2. SARIMAX 网格搜索 (附带 AIC/BIC 记录)
  3. ETS (Holt-Winters) 候选模型
  4. Ljung-Box / Jarque-Bera 残差诊断
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera
import itertools
import warnings
import sys
import os


# ── 辅助：抑制输出 ──────────────────────────────────────────────
class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


# ══════════════════════════════════════════════════════════════════
# 1. 平稳性检验
# ══════════════════════════════════════════════════════════════════

def check_stationarity(series, name="Series"):
    """
    联合 ADF + KPSS 双重检验，返回诊断报告 dict。

    解读矩阵:
      ADF reject + KPSS fail-to-reject → 平稳 ✓
      ADF fail    + KPSS reject        → 非平稳，需差分
      两者矛盾                          → 可能存在趋势平稳，建议差分
    """
    report = {"name": name}

    # ADF 检验 (H0: 单位根/非平稳)
    adf_result = adfuller(series.dropna(), autolag='AIC')
    report["adf_statistic"] = adf_result[0]
    report["adf_pvalue"] = adf_result[1]
    report["adf_reject"] = adf_result[1] <= 0.05  # True = 平稳

    # KPSS 检验 (H0: 平稳)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_result = kpss(series.dropna(), regression='c', nlags='auto')
    report["kpss_statistic"] = kpss_result[0]
    report["kpss_pvalue"] = kpss_result[1]
    report["kpss_reject"] = kpss_result[1] <= 0.05  # True = 非平稳

    # 综合判断
    if report["adf_reject"] and not report["kpss_reject"]:
        report["conclusion"] = "平稳 ✓"
        report["suggested_d"] = 0
    elif not report["adf_reject"] and report["kpss_reject"]:
        report["conclusion"] = "非平稳 → 建议 d=1"
        report["suggested_d"] = 1
    elif report["adf_reject"] and report["kpss_reject"]:
        report["conclusion"] = "趋势平稳 → 建议差分或去趋势"
        report["suggested_d"] = 1
    else:
        report["conclusion"] = "两者均未拒绝 → 可能平稳，但需谨慎"
        report["suggested_d"] = 0

    return report


def print_stationarity_report(report):
    """格式化打印平稳性诊断报告"""
    print(f"  序列: {report['name']}")
    print(f"    ADF  统计量: {report['adf_statistic']:>8.4f}  p={report['adf_pvalue']:.4f}  {'拒绝H0(平稳)' if report['adf_reject'] else '未拒绝H0(非平稳)'}")
    print(f"    KPSS 统计量: {report['kpss_statistic']:>8.4f}  p={report['kpss_pvalue']:.4f}  {'拒绝H0(非平稳)' if report['kpss_reject'] else '未拒绝H0(平稳)'}")
    print(f"    结论: {report['conclusion']}")


# ══════════════════════════════════════════════════════════════════
# 2. SARIMAX 网格搜索 (增强版)
# ══════════════════════════════════════════════════════════════════

def find_best_sarimax_params_sm(endog, exog=None, p_range=range(0, 2), d_range=range(0, 2), q_range=range(0, 2)):
    """
    SARIMAX 网格搜索，返回最优参数及其 AIC/BIC。

    Returns:
        dict: {
            'order': (p,d,q),
            'seasonal_order': (P,D,Q,12),
            'aic': float,
            'bic': float
        }
    """
    pdq = list(itertools.product(p_range, d_range, q_range))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]

    warnings.filterwarnings("ignore")

    best_aic = float("inf")
    best_result = None

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(
                    endog, exog=exog,
                    order=param,
                    seasonal_order=param_seasonal,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                results = mod.fit(disp=False)

                if results.aic < best_aic:
                    best_aic = results.aic
                    best_result = {
                        'order': param,
                        'seasonal_order': param_seasonal,
                        'aic': results.aic,
                        'bic': results.bic,
                    }
            except Exception:
                continue

    return best_result


# ══════════════════════════════════════════════════════════════════
# 3. ETS (Holt-Winters) 模型候选
# ══════════════════════════════════════════════════════════════════

def fit_ets_model(endog, seasonal_periods=12):
    """
    拟合 ETS (Holt-Winters) 加法和乘法季节性模型，返回最优者。

    注意: ETS 不支持外生变量 (exog)，因此仅作为纯内生基准模型。

    Returns:
        dict: {
            'model_type': str,  # 'ETS-Add' or 'ETS-Mul'
            'results': HoltWintersResults,
            'aic': float,
            'bic': float
        }
        或 None（若拟合失败）
    """
    candidates = []

    for seasonal_type, label in [('add', 'ETS-Add'), ('mul', 'ETS-Mul')]:
        try:
            model = ExponentialSmoothing(
                endog,
                trend='add',
                seasonal=seasonal_type,
                seasonal_periods=seasonal_periods,
                initialization_method='estimated'
            )
            results = model.fit(disp=False, optimized=True)
            candidates.append({
                'model_type': label,
                'results': results,
                'aic': results.aic,
                'bic': results.bic,
            })
        except Exception:
            continue

    if not candidates:
        return None

    # 按 AIC 选最优
    best = min(candidates, key=lambda x: x['aic'])
    return best


# ══════════════════════════════════════════════════════════════════
# 4. 残差诊断
# ══════════════════════════════════════════════════════════════════

def diagnose_residuals(residuals, lags=10):
    """
    对模型残差进行诊断检验。

    Returns:
        dict: {
            'ljungbox_pvalue': float (最大滞后阶的 p 值),
            'ljungbox_pass': bool  (p > 0.05 → 无显著自相关 → 通过),
            'jarque_bera_pvalue': float,
            'jarque_bera_pass': bool  (p > 0.05 → 正态 → 通过),
        }
    """
    report = {}

    # Ljung-Box 检验 (H0: 无自相关)
    try:
        lb_result = acorr_ljungbox(residuals.dropna(), lags=lags, return_df=True)
        # 取最大滞后阶的 p 值作为汇总指标
        max_lag_pvalue = lb_result['lb_pvalue'].iloc[-1]
        report['ljungbox_pvalue'] = max_lag_pvalue
        report['ljungbox_pass'] = max_lag_pvalue > 0.05
    except Exception:
        report['ljungbox_pvalue'] = np.nan
        report['ljungbox_pass'] = False

    # Jarque-Bera 检验 (H0: 正态分布)
    try:
        jb_stat, jb_pvalue = jarque_bera(residuals.dropna())
        report['jarque_bera_pvalue'] = jb_pvalue
        report['jarque_bera_pass'] = jb_pvalue > 0.05
    except Exception:
        report['jarque_bera_pvalue'] = np.nan
        report['jarque_bera_pass'] = False

    return report


def print_diagnostics_summary(diag, model_label=""):
    """简洁打印诊断结果"""
    lb_icon = "✓" if diag['ljungbox_pass'] else "✗"
    jb_icon = "✓" if diag['jarque_bera_pass'] else "✗"
    print(f"    {model_label} 残差诊断: "
          f"Ljung-Box p={diag['ljungbox_pvalue']:.3f} {lb_icon}  "
          f"Jarque-Bera p={diag['jarque_bera_pvalue']:.3f} {jb_icon}")


# ══════════════════════════════════════════════════════════════════
# 5. 训练 + 预测 (SARIMAX 封装)
# ══════════════════════════════════════════════════════════════════

def train_sarimax(endog, exog, order, seasonal_order):
    """
    训练 SARIMAX 模型，返回 fit results。
    """
    mod = sm.tsa.statespace.SARIMAX(
        endog, exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = mod.fit(disp=False)
    return results


def forecast_sarimax(fit_results, steps, exog=None):
    """
    基于已训练的 SARIMAX results 执行预测。
    """
    return fit_results.get_forecast(steps=steps, exog=exog)
