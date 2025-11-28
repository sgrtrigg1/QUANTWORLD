# quant_pipeline/deep.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import MODELS_DIR, PLOTS_DIR, METRICS_DIR
from .utils import ensure_dir
from .data import FinancialDataCollector


def make_supervised_series(
    series: np.ndarray,
    window: int = 20,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Turn a 1D series into supervised (X, y) pairs using sliding windows.
    """
    X, y = [], []
    for i in range(len(series) - window - horizon + 1):
        X.append(series[i : i + window])
        y.append(series[i + window + horizon - 1])
    return np.array(X), np.array(y)


@dataclass
class ANNRegressor:
    """
    Simple feed-forward neural network for 1-step-ahead forecasting
    of Close prices using a sliding window of past prices.
    """

    collector: FinancialDataCollector
    window: int = 20
    plots_dir: Path = PLOTS_DIR
    metrics_dir: Path = METRICS_DIR
    models_dir: Path = MODELS_DIR

    def __post_init__(self) -> None:
        ensure_dir(self.plots_dir)
        ensure_dir(self.metrics_dir)
        ensure_dir(self.models_dir)

        self.df = self.collector.process()
        self.ticker = self.collector.ticker
        self.year = self.collector.year

        self.scaler = MinMaxScaler()
        self.model = None
        self.history = None

        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.y_pred = None

    def prepare_data(self, test_size: float = 0.3) -> None:
        close = self.df["Close"].values.reshape(-1, 1)
        scaled = self.scaler.fit_transform(close).flatten()

        X, y = make_supervised_series(scaled, window=self.window, horizon=1)

        n = len(X)
        split_idx = int(n * (1 - test_size))

        self.X_train, self.X_test = X[:split_idx], X[split_idx:]
        self.y_train, self.y_test = y[:split_idx], y[split_idx:]

    def build_model(self) -> None:
        self.model = Sequential(
            [
                Dense(64, activation="relu", input_shape=(self.window,)),
                Dense(32, activation="relu"),
                Dense(1),
            ]
        )
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

    def fit(self, epochs: int = 50, batch_size: int = 32, verbose: int = 0) -> None:
        if self.X_train is None:
            self.prepare_data()

        if self.model is None:
            self.build_model()

        es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es],
            verbose=verbose,
        )

        # save model
        path = self.models_dir / f"{self.ticker}_{self.year}_ann.keras"
        self.model.save(path)
        print(f"[ANN] Saved model to: {path}")

    def predict(self) -> np.ndarray:
        y_pred_scaled = self.model.predict(self.X_test)
        # inverse scale
        y_true_full = self.scaler.inverse_transform(
            self.y_test.reshape(-1, 1)
        ).flatten()
        y_pred_full = self.scaler.inverse_transform(y_pred_scaled).flatten()

        self.y_pred = y_pred_full
        self.y_test_unscaled = y_true_full
        return y_pred_full

    def evaluate(self) -> pd.DataFrame:
        if self.y_pred is None:
            self.predict()

        y_true = self.y_test_unscaled
        y_pred = self.y_pred

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mask = y_true != 0
        mape = (
            np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            if mask.sum()
            else np.nan
        )

        dy_true = np.diff(y_true)
        dy_pred = np.diff(y_pred)
        da = np.mean(np.sign(dy_true) == np.sign(dy_pred))

        metrics = pd.DataFrame(
            [
                {
                    "Model": "ANN",
                    "Ticker": self.ticker,
                    "Year": self.year,
                    "Window": self.window,
                    "MAE": mae,
                    "RMSE": rmse,
                    "MAPE": mape,
                    "Directional_Accuracy": da,
                }
            ]
        )

        out = self.metrics_dir / f"{self.ticker}_{self.year}_ann_metrics.csv"
        metrics.to_csv(out, index=False)
        print(f"[ANN] Saved metrics to: {out}")
        return metrics

    def plot_predictions(self, show: bool = True) -> None:
        if self.y_pred is None:
            self.predict()

        idx = self.df.index[-len(self.y_pred) :]
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(idx, self.y_test_unscaled, label="Actual", linewidth=1)
        ax.plot(idx, self.y_pred, label="ANN forecast", linewidth=1.5)
        ax.set_title(f"{self.ticker} ANN Forecast ({self.year})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price")
        ax.legend()
        ax.grid(True)

        out = self.plots_dir / f"{self.ticker}_{self.year}_ann_forecast.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[ANN] Saved forecast plot to: {out}")

        if show:
            plt.show()
        else:
            plt.close(fig)


@dataclass
class LSTMForecaster:
    """
    LSTM model for 1-step-ahead forecasting of Close prices.
    """

    collector: FinancialDataCollector
    window: int = 20
    plots_dir: Path = PLOTS_DIR
    metrics_dir: Path = METRICS_DIR
    models_dir: Path = MODELS_DIR

    def __post_init__(self) -> None:
        ensure_dir(self.plots_dir)
        ensure_dir(self.metrics_dir)
        ensure_dir(self.models_dir)

        self.df = self.collector.process()
        self.ticker = self.collector.ticker
        self.year = self.collector.year

        self.scaler = MinMaxScaler()
        self.model = None
        self.history = None

        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.y_pred = None

    def prepare_data(self, test_size: float = 0.3) -> None:
        close = self.df["Close"].values.reshape(-1, 1)
        scaled = self.scaler.fit_transform(close).flatten()

        X, y = make_supervised_series(scaled, window=self.window, horizon=1)

        n = len(X)
        split_idx = int(n * (1 - test_size))

        self.X_train, self.X_test = X[:split_idx], X[split_idx:]
        self.y_train, self.y_test = y[:split_idx], y[split_idx:]

        # reshape for LSTM: (samples, timesteps, features)
        self.X_train = self.X_train[..., np.newaxis]
        self.X_test = self.X_test[..., np.newaxis]

    def build_model(self) -> None:
        self.model = Sequential(
            [
                LSTM(50, input_shape=(self.window, 1), return_sequences=False),
                Dropout(0.2),
                Dense(1),
            ]
        )
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

    def fit(self, epochs: int = 50, batch_size: int = 32, verbose: int = 0) -> None:
        if self.X_train is None:
            self.prepare_data()

        if self.model is None:
            self.build_model()

        es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es],
            verbose=verbose,
        )

        path = self.models_dir / f"{self.ticker}_{self.year}_lstm.keras"
        self.model.save(path)
        print(f"[LSTM] Saved model to: {path}")

    def predict(self) -> np.ndarray:
        y_pred_scaled = self.model.predict(self.X_test)
        y_true_full = self.scaler.inverse_transform(
            self.y_test.reshape(-1, 1)
        ).flatten()
        y_pred_full = self.scaler.inverse_transform(y_pred_scaled).flatten()

        self.y_pred = y_pred_full
        self.y_test_unscaled = y_true_full
        return y_pred_full

    def evaluate(self) -> pd.DataFrame:
        if self.y_pred is None:
            self.predict()

        y_true = self.y_test_unscaled
        y_pred = self.y_pred

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mask = y_true != 0
        mape = (
            np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            if mask.sum()
            else np.nan
        )

        dy_true = np.diff(y_true)
        dy_pred = np.diff(y_pred)
        da = np.mean(np.sign(dy_true) == np.sign(dy_pred))

        metrics = pd.DataFrame(
            [
                {
                    "Model": "LSTM",
                    "Ticker": self.ticker,
                    "Year": self.year,
                    "Window": self.window,
                    "MAE": mae,
                    "RMSE": rmse,
                    "MAPE": mape,
                    "Directional_Accuracy": da,
                }
            ]
        )

        out = self.metrics_dir / f"{self.ticker}_{self.year}_lstm_metrics.csv"
        metrics.to_csv(out, index=False)
        print(f"[LSTM] Saved metrics to: {out}")
        return metrics

    def plot_predictions(self, show: bool = True) -> None:
        if self.y_pred is None:
            self.predict()

        idx = self.df.index[-len(self.y_pred) :]
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(idx, self.y_test_unscaled, label="Actual", linewidth=1)
        ax.plot(idx, self.y_pred, label="LSTM forecast", linewidth=1.5)
        ax.set_title(f"{self.ticker} LSTM Forecast ({self.year})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price")
        ax.legend()
        ax.grid(True)

        out = self.plots_dir / f"{self.ticker}_{self.year}_lstm_forecast.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[LSTM] Saved forecast plot to: {out}")

        if show:
            plt.show()
        else:
            plt.close(fig)
