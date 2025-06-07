import numpy as np
import pandas as pd
from utils import model_manager as mm

def run_forecasting_models(df, metadata):
    df = df.copy()
    date_col = metadata['date_column']
    target_col = metadata['target_column']
    df = df.set_index(date_col).resample("M").sum()
    series = df[target_col]

    sarima_forecast, _ = mm.sarima_forecast(series, periods=12)
    arima_forecast, _ = mm.arima_forecast(series, periods=12)
    lstm_forecast, _ = mm.lstm_forecast(series, periods=12)
    xgb_forecast, _ = mm.xgb_forecast(series.to_frame(), periods=12)

    return {
        "sarima": sarima_forecast,
        "arima": arima_forecast,
        "lstm": lstm_forecast,
        "xgboost": xgb_forecast,
        "index": pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods=12, freq="MS")
    }
