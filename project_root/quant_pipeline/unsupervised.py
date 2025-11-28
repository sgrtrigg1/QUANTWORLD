# quant_pipeline/unsupervised.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from .config import PLOTS_DIR, METRICS_DIR
from .utils import ensure_dir
from .data import FinancialDataCollector
from .features import FeatureEngineering


# ------------------------ K-MEANS CLUSTERING ------------------------ #


@dataclass
class KMeansClusterer:
    """
    K-Means clustering on regime-style features
    (rolling_return, rolling_vol, ma_dist).
    """

    collector: FinancialDataCollector
    n_clusters: int = 3
    plots_dir: Path = PLOTS_DIR
    metrics_dir: Path = METRICS_DIR

    def __post_init__(self) -> None:
        self.df = self.collector.process()
        self.ticker = self.collector.ticker
        self.year = self.collector.year

        self.plots_dir = Path(self.plots_dir) / "unsupervised"
        self.metrics_dir = Path(self.metrics_dir) / "unsupervised"
        ensure_dir(self.plots_dir)
        ensure_dir(self.metrics_dir)

        self.features = None
        self.labels_ = None
        self.model = None

    def prepare_features(self) -> None:
        fe = FeatureEngineering(self.df.copy(), self.ticker, self.year)
        feat_df = fe.build_regime_features()
        self.features = feat_df[["rolling_return", "rolling_vol", "ma_dist"]]

    def fit(self) -> None:
        if self.features is None:
            self.prepare_features()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.features)

        self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init="auto")
        self.labels_ = self.model.fit_predict(X_scaled)
        self.features["cluster"] = self.labels_

        out = self.metrics_dir / f"{self.ticker}_{self.year}_kmeans_clusters.csv"
        self.features.to_csv(out)
        print(f"[KMeans] Saved clustered features to: {out}")

    def plot_clusters(self, show: bool = True) -> None:
        if self.labels_ is None:
            self.fit()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=self.features,
            x="rolling_return",
            y="rolling_vol",
            hue="cluster",
            palette="tab10",
            ax=ax,
        )
        ax.set_title(f"K-Means Clusters (k={self.n_clusters})")
        ax.set_xlabel("Rolling Return")
        ax.set_ylabel("Rolling Volatility")
        ax.grid(True)

        out = self.plots_dir / f"{self.ticker}_{self.year}_kmeans_clusters.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[KMeans] Saved cluster plot to: {out}")

        if show:
            plt.show()
        else:
            plt.close(fig)


# ----------------------------- PCA ---------------------------------- #


@dataclass
class PCAEngine:
    """
    PCA on numeric price/feature data to identify latent factors.
    """

    df: pd.DataFrame
    ticker: str
    year: int
    n_components: int = 3
    plots_dir: Path = PLOTS_DIR
    metrics_dir: Path = METRICS_DIR

    def __post_init__(self) -> None:
        self.ticker = self.ticker.upper()
        self.plots_dir = Path(self.plots_dir) / "pca"
        self.metrics_dir = Path(self.metrics_dir) / "pca"
        ensure_dir(self.plots_dir)
        ensure_dir(self.metrics_dir)

        self.model = None
        self.components_ = None
        self.explained_variance_ratio_ = None
        self.X_scaled = None

    def fit(self) -> None:
        df_num = self.df.select_dtypes(include="number").dropna()
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(df_num)

        self.model = PCA(n_components=self.n_components, random_state=42)
        self.components_ = self.model.fit_transform(self.X_scaled)
        self.explained_variance_ratio_ = self.model.explained_variance_ratio_

        # Save variance ratios
        ev = pd.DataFrame(
            {
                "Component": np.arange(1, self.n_components + 1),
                "ExplainedVarianceRatio": self.explained_variance_ratio_,
            }
        )
        out = self.metrics_dir / f"{self.ticker}_{self.year}_pca_variance.csv"
        ev.to_csv(out, index=False)
        print(f"[PCA] Saved explained variance ratios to: {out}")

    def plot_scree(self, show: bool = True) -> None:
        if self.explained_variance_ratio_ is None:
            self.fit()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(
            np.arange(1, len(self.explained_variance_ratio_) + 1),
            self.explained_variance_ratio_,
            marker="o",
        )
        ax.set_title("PCA Scree Plot")
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Explained Variance Ratio")
        ax.grid(True)

        out = self.plots_dir / f"{self.ticker}_{self.year}_pca_scree.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[PCA] Saved scree plot to: {out}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_projection(
        self, comp_x: int = 1, comp_y: int = 2, show: bool = True
    ) -> None:
        if self.components_ is None:
            self.fit()

        cx, cy = comp_x - 1, comp_y - 1
        z = self.components_

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(z[:, cx], z[:, cy], s=10, alpha=0.7)
        ax.set_title(f"PCA Projection: PC{comp_x} vs PC{comp_y}")
        ax.set_xlabel(f"PC{comp_x}")
        ax.set_ylabel(f"PC{comp_y}")
        ax.grid(True)

        out = (
            self.plots_dir / f"{self.ticker}_{self.year}_pca_pc{comp_x}_pc{comp_y}.png"
        )
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[PCA] Saved projection plot to: {out}")

        if show:
            plt.show()
        else:
            plt.close(fig)


# ------------------------ Anomaly Detection ------------------------- #


@dataclass
class AnomalyDetector:
    """
    Simple anomaly detection on returns using z-score + IsolationForest.
    """

    df: pd.DataFrame
    ticker: str
    year: int
    plots_dir: Path = PLOTS_DIR
    metrics_dir: Path = METRICS_DIR

    def __post_init__(self) -> None:
        self.ticker = self.ticker.upper()
        self.plots_dir = Path(self.plots_dir) / "anomalies"
        self.metrics_dir = Path(self.metrics_dir) / "anomalies"
        ensure_dir(self.plots_dir)
        ensure_dir(self.metrics_dir)

        self.ret = self.df["Close"].pct_change().dropna()
        self.anomalies_z = None
        self.anomalies_iforest = None

    def detect_zscore(self, threshold: float = 3.0) -> pd.DataFrame:
        z = (self.ret - self.ret.mean()) / self.ret.std()
        mask = np.abs(z) > threshold
        out_df = pd.DataFrame({"return": self.ret, "zscore": z, "is_anomaly": mask})
        self.anomalies_z = out_df[out_df["is_anomaly"]]

        path = self.metrics_dir / f"{self.ticker}_{self.year}_zscore_anomalies.csv"
        out_df.to_csv(path)
        print(f"[Anomaly] Saved z-score anomalies to: {path}")
        return out_df

    def detect_isolation_forest(self, contamination: float = 0.01) -> pd.DataFrame:
        X = self.ret.values.reshape(-1, 1)
        model = IsolationForest(contamination=contamination, random_state=42)
        labels = model.fit_predict(X)  # -1 anomaly, 1 normal

        out_df = pd.DataFrame(
            {"return": self.ret, "label": labels}, index=self.ret.index
        )
        out_df["is_anomaly"] = out_df["label"] == -1
        self.anomalies_iforest = out_df[out_df["is_anomaly"]]

        path = self.metrics_dir / f"{self.ticker}_{self.year}_iforest_anomalies.csv"
        out_df.to_csv(path)
        print(f"[Anomaly] Saved IsolationForest anomalies to: {path}")
        return out_df

    def plot_anomalies(self, use_iforest: bool = True, show: bool = True) -> None:
        if use_iforest:
            if self.anomalies_iforest is None:
                self.detect_isolation_forest()
            anomalies = self.anomalies_iforest
            method_name = "IsolationForest"
        else:
            if self.anomalies_z is None:
                self.detect_zscore()
            anomalies = self.anomalies_z
            method_name = "Z-score"

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(self.ret.index, self.ret, label="Return", linewidth=1)
        ax.scatter(
            anomalies.index,
            anomalies["return"],
            color="red",
            label="Anomaly",
            zorder=5,
        )
        ax.set_title(f"Return Anomalies ({method_name})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Return")
        ax.grid(True)
        ax.legend()

        suffix = "iforest" if use_iforest else "zscore"
        out = self.plots_dir / f"{self.ticker}_{self.year}_anomalies_{suffix}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[Anomaly] Saved anomaly plot to: {out}")

        if show:
            plt.show()
        else:
            plt.close(fig)
