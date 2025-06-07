import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from xgboost import XGBRegressor
from pmdarima import auto_arima

def run_forecasting_models(df, metadata):
    df = df.copy()
    date_col = metadata['date_column']
    target_col = metadata['target_column']
    df = df.set_index(date_col).resample("M").sum()

    # SARIMA
    sarima = SARIMAX(df[target_col], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
    sarima_forecast = sarima.forecast(12)

    # ARIMA
    arima_model = auto_arima(df[target_col])
    arima_forecast = arima_model.predict(n_periods=12)

    # LSTM
    values = df[target_col].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    X, y = [], []
    for i in range(3, len(scaled)):
        X.append(scaled[i-3:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)

    model = Sequential([LSTM(50, activation='relu', input_shape=(3, 1)), Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=100, verbose=0)

    input_seq = scaled[-3:].reshape(1, 3, 1)
    lstm_forecast = []
    for _ in range(12):
        pred = model.predict(input_seq, verbose=0)
        lstm_forecast.append(pred[0, 0])
        input_seq = np.append(input_seq[:, 1:, :], [[[pred[0, 0]]]], axis=1)
    lstm_forecast = scaler.inverse_transform(np.array(lstm_forecast).reshape(-1, 1)).flatten()

    # XGBoost
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['value'] = df[target_col]
    df['t'] = np.arange(len(df))

    model_xgb = XGBRegressor()
    model_xgb.fit(df[['t', 'month', 'year']], df['value'])

    future = pd.DataFrame({
        't': np.arange(len(df), len(df)+12),
        'month': [(df.index[-1] + pd.DateOffset(months=i)).month for i in range(1, 13)],
        'year': [(df.index[-1] + pd.DateOffset(months=i)).year for i in range(1, 13)]
    })

    xgb_forecast = model_xgb.predict(future)

    return {
        "sarima": sarima_forecast,
        "arima": arima_forecast,
        "lstm": lstm_forecast,
        "xgboost": xgb_forecast,
        "index": pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods=12, freq="MS")
    }
