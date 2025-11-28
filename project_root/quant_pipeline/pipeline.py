# quant_pipeline/pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .data import FinancialDataCollector
from .eda import ExploratoryAnalysis
from .ols import OLSRegression
from .classification import Classification
from .arima import ARIMAForecaster
from .unsupervised import KMeansClusterer, PCAEngine, AnomalyDetector
from .deep import ANNRegressor, LSTMForecaster
from .rl import BanditTrader
from .evaluation import ModelComparison


@dataclass
class QuantPipeline:
    """
    High-level orchestrator for the full dissertation pipeline.

    Usage in notebook:

        qp = QuantPipeline(ticker="AAPL", year=2020)
        qp.run_all()

    or selectively:

        qp.run_baselines()
        qp.run_unsupervised()
        qp.run_deep()
        qp.run_classification()
        qp.run_rl()
    """

    ticker: str
    year: int

    def __post_init__(self) -> None:
        self.collector = FinancialDataCollector(self.ticker, self.year)
        self.df = self.collector.process()
        self.mc = ModelComparison()

    # ----------------------------- Baselines -------------------------- #

    def run_baselines(self) -> None:
        # OLS
        ols = OLSRegression(self.collector)
        ols.fit()
        ols.plot_global_fit(show=False)
        ols.first_half_vs_second_half(show=False)
        ols.segmented_plot(n_segments=4, show=False)
        ols.sliding_window_slope(window_size=60, show=False)

        # If you pasted the evaluate_model() method into OLSRegression:
        try:
            df_ols_global = ols.evaluate_model(window_size=None)
            row = df_ols_global.iloc[0].to_dict()
            self.mc.add_row(
                "OLS_Global", self.collector.ticker, self.collector.year, row
            )
        except Exception:
            pass

        # ARIMA
        ar = ARIMAForecaster(self.collector, order=(1, 1, 1))
        ar.fit()
        ar.plot_forecast(show=False)
        metrics_ar = ar.evaluate().iloc[0].to_dict()
        self.mc.add_row(
            "ARIMA(1,1,1)", self.collector.ticker, self.collector.year, metrics_ar
        )

    # -------------------------- Unsupervised -------------------------- #

    def run_unsupervised(self) -> None:
        # K-Means
        km = KMeansClusterer(self.collector, n_clusters=3)
        km.fit()
        km.plot_clusters(show=False)

        # PCA
        pca_engine = PCAEngine(self.df, self.collector.ticker, self.collector.year)
        pca_engine.fit()
        pca_engine.plot_scree(show=False)
        pca_engine.plot_projection(show=False)

        # Anomaly detection
        ad = AnomalyDetector(self.df, self.collector.ticker, self.collector.year)
        ad.detect_zscore()
        ad.detect_isolation_forest()
        ad.plot_anomalies(use_iforest=True, show=False)

    # --------------------------- Deep models -------------------------- #

    def run_deep(self) -> None:
        ann = ANNRegressor(self.collector, window=20)
        ann.fit(verbose=0)
        ann.predict()
        ann.plot_predictions(show=False)
        metrics_ann = ann.evaluate().iloc[0].to_dict()
        self.mc.add_row("ANN", self.collector.ticker, self.collector.year, metrics_ann)

        lstm = LSTMForecaster(self.collector, window=20)
        lstm.fit(verbose=0)
        lstm.predict()
        lstm.plot_predictions(show=False)
        metrics_lstm = lstm.evaluate().iloc[0].to_dict()
        self.mc.add_row(
            "LSTM", self.collector.ticker, self.collector.year, metrics_lstm
        )

    # ------------------------ Classification -------------------------- #

    def run_classification(self) -> None:
        clf = Classification(self.collector)
        clf.run_pipeline(train_size=0.7, show=False)
        # Classification metrics are already saved in its module

    # --------------------------- RL module ---------------------------- #

    def run_rl(self) -> None:
        rl_agent = BanditTrader(self.collector)
        rl_agent.run()
        rl_agent.plot_rewards(show=False)

    # --------------------------- All together ------------------------- #

    def run_all(self) -> None:
        self.run_baselines()
        self.run_unsupervised()
        self.run_deep()
        self.run_classification()
        self.run_rl()
        self.mc.save()
