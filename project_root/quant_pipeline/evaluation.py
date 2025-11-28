# quant_pipeline/evaluation.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import METRICS_DIR
from .utils import ensure_dir


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mask = y_true != 0
    mape = (
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        if mask.sum()
        else np.nan
    )

    dy_true = np.diff(y_true)
    dy_pred = np.diff(y_pred)
    da = (
        float(np.mean(np.sign(dy_true) == np.sign(dy_pred))) if len(dy_true) else np.nan
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "Directional_Accuracy": da,
    }


@dataclass
class ModelComparison:
    """
    Collects metrics from multiple models into a single table
    for the Results & Discussion chapter.
    """

    metrics_dir: Path = METRICS_DIR
    rows: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        ensure_dir(self.metrics_dir)

    def add_row(
        self, model_name: str, ticker: str, year: int, metrics: Dict[str, Any]
    ) -> None:
        entry = {"Model": model_name, "Ticker": ticker, "Year": year}
        entry.update(metrics)
        self.rows.append(entry)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def save(self, filename: str = "model_comparison.csv") -> Path:
        df = self.to_dataframe()
        path = self.metrics_dir / filename
        df.to_csv(path, index=False)
        print(f"[ModelComparison] Saved comparison table to: {path}")
        return path
