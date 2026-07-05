"""
lstm_model.py
-------------
LSTM model: sequence preparation, 2-layer architecture,
training with EarlyStopping, and forecast generation.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
WINDOW_SIZE   = 60


def prepare_sequences(train_series, test_series, window=WINDOW_SIZE):
    """
    Scale data with MinMaxScaler and build sliding-window sequences.
    Returns X_train, y_train, X_test, y_test, and the fitted scaler.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_series.values.reshape(-1, 1))
    test_scaled  = scaler.transform(test_series.values.reshape(-1, 1))

    def make_sequences(data, w):
        X, y = [], []
        for i in range(w, len(data)):
            X.append(data[i - w:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X_train, y_train = make_sequences(train_scaled, window)

    # Prepend last `window` train values to test for continuity
    combined = np.concatenate([train_scaled[-window:], test_scaled])
    X_test, y_test = make_sequences(combined, window)

    # Reshape to (samples, timesteps, features)
    X_train = X_train.reshape(*X_train.shape, 1)
    X_test  = X_test.reshape(*X_test.shape, 1)

    return X_train, y_train, X_test, y_test, scaler


def build_model(window=WINDOW_SIZE):
    """Build and compile a 2-layer LSTM model."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window, 1)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_model(model, X_train, y_train, epochs=100, batch_size=32, val_split=0.1):
    """Train the LSTM model with EarlyStopping."""
    from tensorflow.keras.callbacks import EarlyStopping

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=[early_stop],
        verbose=1
    )
    return history


def generate_forecast(model, X_test, scaler):
    """Generate inverse-transformed price forecasts from the LSTM model."""
    pred_scaled = model.predict(X_test)
    return scaler.inverse_transform(pred_scaled).flatten()


def plot_training_history(history, save_name='lstm_training_history.png'):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.history['loss'],     label='Train Loss')
    ax.plot(history.history['val_loss'], label='Val Loss')
    ax.set_title('LSTM Training History', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, save_name), dpi=150, bbox_inches='tight')
    plt.show()


def plot_forecast(train, test, forecast_values, save_name='lstm_forecast.png'):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(train.index[-120:], train.iloc[-120:], label='Train (last 120 days)', color='steelblue')
    ax.plot(test.index, test.values,       label='Actual',        color='darkorange')
    ax.plot(test.index, forecast_values,   label='LSTM Forecast', color='purple', linestyle='--')
    ax.set_title('LSTM Forecast vs Actual', fontsize=13, fontweight='bold')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, save_name), dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    import tensorflow as tf
    tf.random.set_seed(42)
    np.random.seed(42)

    from data_loader import load_cleaned

    tsla  = load_cleaned('TSLA')[['Adj Close']].rename(columns={'Adj Close': 'Close'})
    train = tsla[tsla.index < '2025-01-01']['Close']
    test  = tsla[tsla.index >= '2025-01-01']['Close']

    X_train, y_train, X_test, y_test, scaler = prepare_sequences(train, test)
    model   = build_model()
    history = train_model(model, X_train, y_train)
    preds   = generate_forecast(model, X_test, scaler)

    plot_training_history(history)
    plot_forecast(train, test, preds)
