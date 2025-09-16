import pandas as pd
import numpy as np
import requests
from io import StringIO

def get_clean_data(file_path):
    """
    从本地文件路径加载、清理并准备GISTEMP时间序列数据。
    """
    with open(file_path, 'r') as f:
        all_lines = f.readlines()
    header = all_lines[7].split()
    records = []
    for line in all_lines[8:]:
        if not line.strip() or ('Year' in line and 'Jan' in line):
            continue
        values = line.split()
        if len(values) >= len(header):
            records.append(dict(zip(header, values)))
    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise ValueError("GISTEMP数据解析失败。")
    if 'Year.1' in df.columns:
        df = df.drop(columns=['Year.1'])
    df.replace(r'\*+', np.nan, regex=True, inplace=True)
    month_columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    required_cols = ['Year'] + month_columns
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"GISTEMP数据中缺少必需的列。")
    df = df[required_cols]
    df_long = df.melt(id_vars=['Year'], var_name='Month', value_name='Anomaly')
    month_map = {month: i+1 for i, month in enumerate(month_columns)}
    df_long['Month_Num'] = df_long['Month'].map(month_map)
    df_long.dropna(subset=['Year', 'Month_Num'], inplace=True)
    df_long['Year'] = df_long['Year'].astype(int)
    df_long['Month_Num'] = df_long['Month_Num'].astype(int)
    df_long['Date'] = pd.to_datetime(df_long['Year'].astype(str) + '-' + df_long['Month_Num'].astype(str))
    df_long.set_index('Date', inplace=True)
    df_long['Anomaly'] = pd.to_numeric(df_long['Anomaly'], errors='coerce')
    df_long.sort_index(inplace=True)
    last_valid_index = df_long['Anomaly'].last_valid_index()
    if last_valid_index is not None:
        df_long = df_long.loc[:last_valid_index]
    df_long['Anomaly'] = df_long['Anomaly'].interpolate(method='time')
    df_long.dropna(inplace=True)
    return df_long['Anomaly']

def get_enso_data(url):
    """
    从NOAA网站获取月度ENSO(Nino 3.4)数据。
    """
    response = requests.get(url)
    response.raise_for_status()
    data_io = StringIO(response.text)
    enso_df = pd.read_csv(data_io, sep='\s+', engine='python')
    enso_df['Date'] = pd.to_datetime(enso_df['YR'].astype(str) + '-' + enso_df['MON'].astype(str))
    enso_df.set_index('Date', inplace=True)
    enso_series = enso_df['ANOM'].rename('ENSO_ANOM')
    return enso_series


