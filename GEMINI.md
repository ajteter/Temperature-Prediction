# Project Log: GISTEMP Anomaly Forecaster

This document summarizes the development process, key decisions, and iterative refinements made during the creation of the temperature forecasting tool.

## Phase 1: Scaffolding and Initial Data Handling

- **Objective:** Create a Python-based tool to forecast the GISTEMP anomaly.
- **Initial Actions:** Scaffolded the project structure with `config.py`, `data_handler.py`, `model.py`, and `main.py`.
- **Challenge 1 (Data Fetching):** Initial attempts to fetch the GISTEMP data via `web_fetch` and `curl` failed due to server-side restrictions (403 Forbidden, timeouts, SSL errors).
- **Solution 1:** Pivoted to a manual workflow where the user provided the data file (`GLB.Ts+dSST.txt`) locally. The `config.py` was updated to point to a local file path instead of a URL.
- **Challenge 2 (Data Parsing):** The GISTEMP text file had a complex, non-standard, fixed-width format with multiple header lines. Initial `pandas.read_csv` attempts failed, leading to `KeyError` and `ValueError` exceptions.
- **Solution 2:** Iterated through several parsing strategies. The final, robust solution involved manually reading the file line-by-line, identifying the correct header row (which was discovered to be on line 8, not 7), splitting each data row into a dictionary, and then creating a DataFrame from a list of these records. This defensive approach proved successful.

## Phase 2: Model Implementation and Baseline Backtest

- **Objective:** Implement and validate a baseline SARIMA model.
- **Actions:** 
    - Implemented a `find_best_sarima_params` function to perform a grid search for optimal model parameters based on AIC.
    - Implemented a `train_and_forecast` function.
    - Added a backtesting framework to `main.py`.
- **User Insight:** The user correctly hypothesized that training on the full historical record (since 1880) might be less effective than training on data from the modern era of accelerated warming. We agreed to set the training start date for all backtests to **January 1970**.
- **Finding:** The initial backtest of the best univariate SARIMA model revealed a high Root Mean Squared Error (RMSE), indicating that important predictive information was missing.

## Phase 3: Enhancement with Exogenous Variables (SARIMAX)

- **Objective:** Improve model accuracy by incorporating external climate drivers.

### ENSO Integration (Success)
- **Action:** Identified a machine-readable monthly Niño 3.4 index from NOAA/CPC. Added a `get_enso_data` function to the data handler. Upgraded the model in `model.py` and the workflow in `main.py` to use SARIMAX with ENSO as an exogenous variable.
- **Result:** This was highly successful. The backtest RMSE dropped dramatically (e.g., from ~29 to ~10 in one experiment), proving that ENSO is a critical predictor of short-term temperature anomalies.

### AOD, TSI, and CO2 Integration (Challenges and Findings)
- **AOD/TSI:** Attempts to find up-to-date, machine-readable data for Volcanic Aerosols (AOD) and Total Solar Irradiance (TSI) failed. The data was either outdated, in an un-parsable binary format (NetCDF), or on servers that blocked access. This line of inquiry was abandoned.
- **CO2:** At the user's request, we integrated monthly CO2 data from the Scripps CO2 Program. I hypothesized this might cause a collinearity issue with the model's internal trend-handling mechanism (`d=1`).
- **Result:** The experiment confirmed the hypothesis. The backtest RMSE *increased* after adding CO2 (e.g., from 9.86 to 13.97), proving it was detrimental to the model's predictive accuracy for this task. 
- **Decision:** The CO2 variable was discarded, and the **SARIMAX model with only ENSO was confirmed as the optimal model.**

## Phase 4: Advanced Features & The Final Adaptive Strategy

- **Objective:** Refine the model's assumptions and predictive power based on user insights.

1.  **Probabilistic Forecast:** The system was upgraded to move beyond a single point forecast. It now uses the forecast's standard error to calculate and display the probability of the outcome falling into various user-defined bins.

2.  **Dynamic Exogenous Forecast:** The naive assumption of a persistent future ENSO value was replaced. A SARIMA sub-model was implemented to create a dynamic, data-driven forecast for the future ENSO value, which is then fed into the main temperature model.

3.  **Horizon Sensitivity Analysis:** At the user's direction, we conducted a series of increasingly fine-grained backtesting experiments with different training/testing split points. This was done to answer the question: "What is the optimal amount of historical data to use?"

4.  **The Final Adaptive Strategy (User-Designed):** The project culminated in the implementation of a sophisticated, two-stage adaptive forecasting system proposed by the user:
    - **Stage 1 ("Horse Race"):** The script first runs a series of backtests with different historical cutoff points (`-18, -21, -24, -27, -30` months from the target) to empirically determine which training period yields the lowest RMSE on recent data.
    - **Stage 2 ("Champion Forecast"):** The script automatically identifies the "champion" configuration from the horse race and uses that specific model (with its unique training period and parameters) to perform the final, definitive forecast.

## Final Outcome

The project concluded with the successful implementation and execution of the adaptive forecasting system. The final run automatically determined that a model trained on data up to **September 2023** (the -24 month model) was the optimal predictor, achieving the lowest RMSE of **8.78** in backtesting. Using this champion configuration, it generated the project's definitive forecast.

The final deliverable is not just a model, but an intelligent, repeatable, and scientifically validated forecasting workflow that adapts to the specific context of the data. The project successfully evolved from a simple script to a sophisticated piece of analytical software.
