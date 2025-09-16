import pandas as pd
import statsmodels.api as sm
import itertools
import warnings

def find_best_sarimax_params(endog, exog=None):
    """
    使用网格搜索为SARIMA(X)模型寻找最优参数。
    如果exog为None，则作为纯SARIMA模型进行搜索。
    """
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]

    warnings.filterwarnings("ignore")

    best_aic = float("inf")
    best_pdq = None
    best_seasonal_pdq = None

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(endog,
                                                exog=exog,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit(disp=False)
                
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_pdq = param
                    best_seasonal_pdq = param_seasonal
            except Exception as e:
                continue
    
    return best_pdq, best_seasonal_pdq

def train_and_forecast_sarimax(endog, order, seasonal_order, steps, exog=None, future_exog=None):
    """
    使用给定的参数训练SARIMA(X)模型并进行预测。
    此版本明确接收steps参数，并能处理带或不带exog变量的情况。
    """
    mod = sm.tsa.statespace.SARIMAX(endog,
                                    exog=exog,
                                    order=order,
                                    seasonal_order=seasonal_order,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit(disp=False)

    forecast_results = results.get_forecast(steps=steps, exog=future_exog)
    
    return forecast_results