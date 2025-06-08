import numpy as np
import pandas as pd
from utils import model_manager as mm

def run_forecasting_models(df, metadata):
    df = df.copy()
    date_col = metadata['date_column']
    target_col = metadata['target_column']
    if isinstance(target_col, list):
        target_col = target_col[0]
    df = df.set_index(date_col).resample("M").sum()
    series = df[target_col]

    # simple train/test split for evaluation
    n_test = min(3, max(1, len(series) // 4))
    train = series[:-n_test] if len(series) > n_test else series
    test = series[-n_test:]

    forecasts = {}
    metrics = {}

    for name, func, data in [
        ("sarima", mm.sarima_forecast, train),
        ("arima", mm.arima_forecast, train),
        ("lstm", mm.lstm_forecast, train),
        ("xgboost", mm.xgb_forecast, train.to_frame()),
    ]:
        preds, _ = func(data, periods=n_test)
        rmse = np.sqrt(np.mean((preds[:len(test)] - test.values[:len(preds)]) ** 2))
        metrics[name] = rmse
        final_preds, _ = func(series if name != "xgboost" else series.to_frame(), periods=12)
        forecasts[name] = final_preds

    best_model = min(metrics, key=metrics.get)

    forecasts['index'] = pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods=12, freq="MS")
    forecasts['metrics'] = metrics
    forecasts['best_model'] = best_model

    return forecasts
