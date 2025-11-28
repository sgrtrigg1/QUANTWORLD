# quant_pipeline/classification.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from .config import RESULTS_DIR, PLOTS_DIR, METRICS_DIR
from .utils import ensure_dir
from .data import FinancialDataCollector
from .features import FeatureEngineering


@dataclass
class Classification:
    """
    Market Regime Classification using Logistic Regression.

    Features:
      - rolling_return
      - rolling_vol
      - ma_dist

    Labels:
      -1 bear, 0 sideways, 1 bull
    """

    collector: FinancialDataCollector
    results_dir: Path = RESULTS_DIR

    def __post_init__(self) -> None:
        self.df = self.collector.process()
        self.ticker = self.collector.ticker
        self.year = self.collector.year

        self.results_dir = Path(self.results_dir)
        self.plots_dir = self.results_dir / "classification_plots"
        self.metrics_dir = self.results_dir / "classification_metrics"

        ensure_dir(self.plots_dir)
        ensure_dir(self.metrics_dir)

        self.features: Optional[pd.DataFrame] = None
        self.labels: Optional[pd.Series] = None

        self.X_train = self.X_test = None
        self.y_train = self.y_test = None

        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[LogisticRegression] = None

    # --------------------------
    # Feature Engineering + Labels
    # --------------------------
    def make_features_and_labels(
        self,
        window_return: int = 20,
        window_vol: int = 20,
        window_ma: int = 50,
        upper: float = 0.005,
        lower: float = -0.005,
    ) -> None:
        fe = FeatureEngineering(self.df.copy(), self.ticker, self.year)
        feat_df = fe.build_regime_features(
            window_return=window_return,
            window_vol=window_vol,
            window_ma=window_ma,
            upper=upper,
            lower=lower,
        )

        self.features = feat_df[["rolling_return", "rolling_vol", "ma_dist"]]
        self.labels = feat_df["regime"].astype(int)

    # --------------------------
    # Train / Test Split
    # --------------------------
    def train_test_split(self, train_size: float = 0.7) -> None:
        n = len(self.features)
        idx = int(n * train_size)

        self.X_train = self.features.iloc[:idx]
        self.y_train = self.labels.iloc[:idx]
        self.X_test = self.features.iloc[idx:]
        self.y_test = self.labels.iloc[idx:]

    # --------------------------
    # Model Fit
    # --------------------------
    def fit_model(self) -> None:
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train)

        self.model = LogisticRegression(
            multi_class="multinomial",
            max_iter=1000,
            class_weight="balanced",
        )
        self.model.fit(X_train_scaled, self.y_train)

    # --------------------------
    # Evaluation
    # --------------------------
    def evaluate(self, show: bool = True) -> None:
        X_test_scaled = self.scaler.transform(self.X_test)
        y_pred = self.model.predict(X_test_scaled)

        report = classification_report(self.y_test, y_pred)
        print("\nClassification Report:\n")
        print(report)

        # Save text report
        report_path = (
            self.metrics_dir / f"{self.ticker}_{self.year}_classification_report.txt"
        )
        with open(report_path, "w") as f:
            f.write(report)
        print(f"[Classification] Saved classification report to: {report_path}")

        cm = confusion_matrix(self.y_test, y_pred, labels=[-1, 0, 1])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Bear (-1)", "Sideways (0)", "Bull (1)"],
        )

        fig, ax = plt.subplots(figsize=(5, 4))
        disp.plot(ax=ax)
        plt.title("Market Regime Classification - Confusion Matrix")
        plt.tight_layout()

        # Save confusion matrix
        cm_path = self.plots_dir / f"{self.ticker}_{self.year}_confusion_matrix.png"
        plt.savefig(cm_path, dpi=150, bbox_inches="tight")
        print(f"[Classification] Saved confusion matrix to: {cm_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    # --------------------------
    # Full Pipeline
    # --------------------------
    def run_pipeline(self, train_size: float = 0.7, show: bool = True) -> None:
        self.make_features_and_labels()
        self.train_test_split(train_size=train_size)
        self.fit_model()
        self.evaluate(show=show)
