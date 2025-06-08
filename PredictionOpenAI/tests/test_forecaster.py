import os
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.')))
from PredictionOpenAI.app.utils.Forecaster import run_forecasting_models
from PredictionOpenAI.app.utils.Visualizer import plot_and_save


def sample_df():
    dates = pd.date_range('2022-01-01', periods=15, freq='M')
    sales = pd.Series(range(1, 16))
    return pd.DataFrame({'date': dates, 'sales': sales})


def test_run_forecasting_models(tmp_path):
    df = sample_df()
    metadata = {'date_column': 'date', 'target_column': ['sales']}
    result = run_forecasting_models(df, metadata)
    assert 'best_model' in result
    assert 'metrics' in result
    assert len(result['index']) == 12


def test_plot_and_save(tmp_path):
    df = sample_df()
    metadata = {'date_column': 'date', 'target_column': ['sales']}
    forecasts = run_forecasting_models(df, metadata)
    plot_and_save(df, forecasts, metadata)
    assert os.path.exists('logs/forecast_plot.png')
    os.remove('logs/forecast_plot.png')
