import os
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import ARIMA
from sklearn.metrics import mean_squared_error

# === Model Save/Load Paths ===
MODEL_DIR = "models"


def ensure_model_dir():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)


# === Save/Load Utility ===

def save_model(model, model_name):
    ensure_model_dir()
    path = os.path.join(MODEL_DIR, model_name)
    if hasattr(model, 'save'):
        model.save(path)
    else:
        joblib.dump(model, path + '.pkl')


def load_model_file(model_name, keras=False):
    path = os.path.join(MODEL_DIR, model_name)
    if keras:
        return load_model(path)
    else:
        return joblib.load(path + '.pkl')


# === Anomaly Detection ===

def detect_anomalies(actual, predicted, threshold=2.0):
    residuals = np.abs(actual - predicted)
    std = residuals.std()
    anomalies = residuals > threshold * std
    return anomalies


# === Forecasting Models ===

def sarima_forecast(df, periods=12):
    model = SARIMAX(df, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    result = model.fit(disp=False)
    forecast = result.forecast(steps=periods)
    save_model(result, "sarima_model")
    return forecast, result.aic


def arima_forecast(df, periods=12):
    model = ARIMA(order=(1, 1, 1))
    model_fit = model.fit(df)
    forecast = model_fit.predict(n_periods=periods)
    save_model(model_fit, "arima_model")
    return forecast, model_fit.aic()


def lstm_forecast(df, periods=12):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from sklearn.preprocessing import MinMaxScaler

    values = df.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    X, y = [], []
    for i in range(3, len(scaled)):
        X.append(scaled[i - 3:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(X.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=30, verbose=0, callbacks=[EarlyStopping(patience=5)])

    last_steps = scaled[-3:].reshape(1, 3, 1)
    forecast = []
    for _ in range(periods):
        pred = model.predict(last_steps, verbose=0)[0]
        forecast.append(pred)
        last_steps = np.roll(last_steps, -1, axis=1)
        last_steps[0, -1] = pred
    forecast = scaler.inverse_transform(forecast)
    save_model(model, "lstm_model")
    return forecast.flatten(), None


def xgb_forecast(df, periods=12):
    df = df.reset_index()
    df['time'] = np.arange(len(df))
    X, y = df[['time']], df[df.columns[1]]

    model = XGBRegressor()
    model.fit(X, y)

    future_time = np.arange(len(df), len(df) + periods).reshape(-1, 1)
    forecast = model.predict(future_time)
    save_model(model, "xgb_model")
    return forecast, mean_squared_error(y, model.predict(X), squared=False)
