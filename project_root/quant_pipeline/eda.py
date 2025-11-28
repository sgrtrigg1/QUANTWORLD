# quant_pipeline/eda.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .config import PLOTS_DIR
from .utils import ensure_dir


@dataclass
class ExploratoryAnalysis:
    """
    EDA module: plots OHLC, returns, volatility, correlation heatmap,
    and distributions. Outputs are used in the dissertation figures.
    """

    df: pd.DataFrame
    ticker: str
    year: int
    plots_dir: Path = PLOTS_DIR

    def __post_init__(self) -> None:
        self.ticker = self.ticker.upper()
        ensure_dir(self.plots_dir)

    def _save_show(self, fig: plt.Figure, name: str, show: bool) -> None:
        out = self.plots_dir / f"{self.ticker}_{self.year}_{name}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[EDA] Saved: {out}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_ohlc(self, show: bool = True) -> None:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(self.df.index, self.df["Close"], label="Close")
        ax.set_title(f"{self.ticker} Close Price {self.year}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True)
        ax.legend()
        self._save_show(fig, "ohlc_close", show)

    def plot_returns_and_cumulative(self, show: bool = True) -> None:
        ret = self.df["Close"].pct_change().dropna()
        cum_ret = (1 + ret).cumprod()

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        axes[0].plot(ret.index, ret, linewidth=1)
        axes[0].set_title("Daily Returns")
        axes[0].set_ylabel("Return")

        axes[1].plot(cum_ret.index, cum_ret, linewidth=1.2)
        axes[1].set_title("Cumulative Buy-and-Hold Return")
        axes[1].set_ylabel("Cumulative Growth")
        axes[1].set_xlabel("Date")

        for ax in axes:
            ax.grid(True)

        self._save_show(fig, "returns_cumulative", show)

    def plot_rolling_volatility(self, window: int = 20, show: bool = True) -> None:
        ret = self.df["Close"].pct_change()
        vol = ret.rolling(window).std()

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(vol.index, vol)
        ax.set_title(f"Rolling {window}-Day Volatility")
        ax.set_xlabel("Date")
        ax.set_ylabel("Volatility")
        ax.grid(True)

        self._save_show(fig, f"rolling_vol_{window}", show)

    def plot_correlation_heatmap(self, show: bool = True) -> None:
        df_num = self.df.select_dtypes(include="number").dropna()
        corr = df_num.corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, ax=ax, annot=False, cmap="coolwarm", fmt=".2f")
        ax.set_title("Feature Correlation Heatmap")

        self._save_show(fig, "corr_heatmap", show)

    def plot_return_distribution(self, show: bool = True) -> None:
        ret = self.df["Close"].pct_change().dropna()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(ret, bins=50, density=True, alpha=0.7)
        ax.set_title("Return Distribution")
        ax.set_xlabel("Daily Return")
        ax.set_ylabel("Density")
        self._save_show(fig, "return_distribution", show)
