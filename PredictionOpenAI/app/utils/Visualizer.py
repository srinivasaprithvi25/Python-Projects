import matplotlib.pyplot as plt
import os

def plot_and_save(df, forecasts, metadata):
    target = metadata['target_column']
    index = forecasts['index']

    plt.figure(figsize=(10, 6))
    df_plot = df.set_index(metadata['date_column']).resample("M").sum()
    plt.plot(df_plot.index, df_plot[target], label='Historical')

    plt.plot(index, forecasts['sarima'], label='SARIMA')
    plt.plot(index, forecasts['arima'], label='ARIMA')
    plt.plot(index, forecasts['lstm'], label='LSTM')
    plt.plot(index, forecasts['xgboost'], label='XGBoost')

    plt.title("Forecast Comparison")
    plt.xlabel("Date")
    plt.ylabel(target)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs("logs", exist_ok=True)
    plt.savefig("logs/forecast_plot.png")
    plt.close()
