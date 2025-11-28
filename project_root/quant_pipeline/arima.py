# quant_pipeline/arima.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA

from .config import PLOTS_DIR, METRICS_DIR
from .utils import ensure_dir, train_test_split_time_series
from .data import FinancialDataCollector


@dataclass
class ARIMAForecaster:
    """
    ARIMA baseline model for 1-step-ahead forecasting of daily Close prices.

    Parameters
    ----------
    collector : FinancialDataCollector
        Data source for a single ticker-year.
    order : tuple
        (p, d, q) order for ARIMA.
    """

    collector: FinancialDataCollector
    order: Tuple[int, int, int] = (1, 1, 1)
    plots_dir: Path = PLOTS_DIR
    metrics_dir: Path = METRICS_DIR

    def __post_init__(self) -> None:
        ensure_dir(self.plots_dir)
        ensure_dir(self.metrics_dir)

        self.df = self.collector.process()
        self.ticker = self.collector.ticker
        self.year = self.collector.year

        self.model = None
        self.fitted_result = None
        self.y_train = None
        self.y_test = None
        self.pred = None

    def fit(self, test_size: float = 0.3) -> None:
        """
        Fit ARIMA(p,d,q) model on training subset of Close prices.
        """
        df_close = self.df[["Close"]].copy()
        train, test = train_test_split_time_series(df_close, test_size=test_size)

        self.y_train = train["Close"]
        self.y_test = test["Close"]

        # statsmodels prefers a regular frequency index
        self.y_train = self.y_train.asfreq("B")

        self.model = ARIMA(self.y_train, order=self.order)
        self.fitted_result = self.model.fit()
        print(self.fitted_result.summary())

    def forecast(self) -> pd.Series:
        """
        1-step-ahead rolling forecast over the test period.

        Uses positional indices for start/end and then
        aligns the predictions to y_test's DatetimeIndex.
        """
        if self.fitted_result is None:
            self.fit()

        steps = len(self.y_test)

        # use integer positions: start at end of training sample
        start = len(self.y_train)  # first out-of-sample point
        end = start + steps - 1  # last out-of-sample point

        pred = self.fitted_result.predict(start=start, end=end, typ="levels")

        # align to test dates
        pred.index = self.y_test.index

        self.pred = pred
        return pred

    def evaluate(self) -> pd.DataFrame:
        """
        Compute standard regression metrics: MAE, RMSE, MAPE, Directional Accuracy.
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        if self.pred is None:
            self.forecast()

        y_true = self.y_test.values
        y_pred = self.pred.values

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        mask = y_true != 0
        mape = (
            np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            if mask.sum()
            else np.nan
        )

        # directional accuracy
        dy_true = np.diff(y_true)
        dy_pred = np.diff(y_pred)
        da = np.mean(np.sign(dy_true) == np.sign(dy_pred))

        metrics = pd.DataFrame(
            [
                {
                    "Model": "ARIMA",
                    "Order": str(self.order),
                    "Ticker": self.ticker,
                    "Year": self.year,
                    "N_Obs": len(y_true),
                    "MAE": mae,
                    "RMSE": rmse,
                    "MAPE": mape,
                    "Directional_Accuracy": da,
                }
            ]
        )

        out = self.metrics_dir / f"{self.ticker}_{self.year}_arima_metrics.csv"
        metrics.to_csv(out, index=False)
        print(f"[ARIMA] Saved metrics to: {out}")
        return metrics

    def plot_forecast(self, show: bool = True) -> None:
        """
        Plot training data, test data, and ARIMA forecasts.
        """
        if self.pred is None:
            self.forecast()

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(self.y_train.index, self.y_train, label="Train")
        ax.plot(self.y_test.index, self.y_test, label="Test", color="black")
        ax.plot(self.pred.index, self.pred, label="ARIMA forecast", color="red")

        ax.set_title(f"{self.ticker} ARIMA{self.order} Forecast ({self.year})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price")
        ax.legend()
        ax.grid(True)

        out = self.plots_dir / f"{self.ticker}_{self.year}_arima_forecast.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[ARIMA] Saved forecast plot to: {out}")

        if show:
            plt.show()
        else:
            plt.close(fig)
