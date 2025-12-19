import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import pearsonr

# Import existing data handling and config
import data_handler
import config

def analyze_enso_lags(full_df, anomaly_col='Anomaly', enso_col='ENSO_ANOM', max_lag=12):
    """
    Analyzes the correlation between temperature anomaly and lagged ENSO anomaly.

    Args:
        full_df (pd.DataFrame): DataFrame containing 'Anomaly' and 'ENSO_ANOM'.
        anomaly_col (str): Name of the temperature anomaly column.
        enso_col (str): Name of the ENSO anomaly column.
        max_lag (int): Maximum lag in months to analyze.

    Returns:
        pd.DataFrame: A DataFrame with 'Lag' and 'Pearson_Correlation' for each lag.
    """
    
    # Ensure all data is numeric and fill NaNs for correlation calculation
    df = full_df[[anomaly_col, enso_col]].copy()
    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)

    correlations = []
    print(f"Analyzing correlations between {anomaly_col} and {enso_col} for lags 0 to {max_lag} months...")

    for lag in range(max_lag + 1):
        # Shift ENSO data by 'lag' months
        # A positive lag means ENSO from 'lag' months *ago* is correlated with *current* Anomaly
        enso_lagged = df[enso_col].shift(lag)
        
        # Create a temporary DataFrame for correlation calculation, dropping NaNs introduced by shifting
        temp_df = pd.DataFrame({
            'Anomaly': df[anomaly_col],
            'ENSO_Lagged': enso_lagged
        }).dropna()

        if len(temp_df) > 0:
            # Calculate Pearson correlation coefficient
            corr, _ = pearsonr(temp_df['Anomaly'], temp_df['ENSO_Lagged'])
            correlations.append({'Lag': lag, 'Pearson_Correlation': corr})
        else:
            correlations.append({'Lag': lag, 'Pearson_Correlation': np.nan}) # No data for correlation

    results_df = pd.DataFrame(correlations)
    return results_df

if __name__ == "__main__":
    print("=== ENSO Lag Correlation Analysis ===")
    try:
        # Load GISTEMP and ENSO data
        gistemp_data = data_handler.get_clean_data(config.DATA_FILE_PATH)
        enso_data = data_handler.get_enso_data(config.ENSO_DATA_URL)
        
        # Merge them based on index (Date)
        full_df = pd.concat([gistemp_data, enso_data], axis=1)
        full_df.dropna(inplace=True) # Drop rows where either GISTEMP or ENSO is missing
        
        print(f"Data loaded from {full_df.index.min().strftime('%Y-%m')} to {full_df.index.max().strftime('%Y-%m')}")
        
        # Run the analysis
        correlation_results = analyze_enso_lags(full_df, max_lag=12)
        
        # Print results
        print("\nCorrelation Results:")
        print(correlation_results.to_string(index=False))
        
        # Find the best lag
        best_lag_row = correlation_results.loc[correlation_results['Pearson_Correlation'].idxmax()]
        print(f"\nOptimal Lag: {int(best_lag_row['Lag'])} months with Pearson Correlation: {best_lag_row['Pearson_Correlation']:.3f}")

    except Exception as e:
        print(f"Error during correlation analysis: {e}")
        import traceback
        traceback.print_exc()
