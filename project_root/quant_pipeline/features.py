# quant_pipeline/features.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

from .config import PROCESSED_DIR
from .utils import ensure_dir


@dataclass
class FeatureEngineering:
    """
    Feature engineering module.

    - Computes returns (log or pct)
    - Normalised price series
    - Rolling indicators (MA, volatility, MA distance)
    - Optional exports to processed CSV
    """

    df: pd.DataFrame
    ticker: str
    year: int
    processed_dir: Path = PROCESSED_DIR

    def __post_init__(self) -> None:
        self.ticker = self.ticker.upper()
        ensure_dir(self.processed_dir)

    def add_returns(self, log: bool = True) -> None:
        close = self.df["Close"]
        if log:
            self.df["ret"] = np.log(close / close.shift(1))
        else:
            self.df["ret"] = close.pct_change()

    def add_moving_averages(self, windows: List[int] = [10, 20, 50]) -> None:
        for w in windows:
            self.df[f"sma_{w}"] = self.df["Close"].rolling(w).mean()
            self.df[f"ema_{w}"] = self.df["Close"].ewm(span=w, adjust=False).mean()
            self.df[f"ma_dist_{w}"] = self.df["Close"] / self.df[f"sma_{w}"] - 1.0

    def add_volatility(self, window: int = 20) -> None:
        if "ret" not in self.df.columns:
            self.add_returns(log=False)
        self.df[f"rolling_vol_{window}"] = self.df["ret"].rolling(window).std()

    def build_regime_features(
        self,
        window_return: int = 20,
        window_vol: int = 20,
        window_ma: int = 50,
        upper: float = 0.005,
        lower: float = -0.005,
    ) -> pd.DataFrame:
        """
        Features & labels consistent with Classification class:
        rolling_return, rolling_vol, ma_dist, regime.
        """
        close = self.df["Close"]
        ret = close.pct_change()

        roll_ret = ret.rolling(window_return).mean()
        roll_vol = ret.rolling(window_vol).std()
        ma = close.rolling(window_ma).mean()
        ma_dist = close / ma - 1.0

        feat_df = pd.DataFrame(
            {
                "rolling_return": roll_ret,
                "rolling_vol": roll_vol,
                "ma_dist": ma_dist,
            },
            index=self.df.index,
        )

        regime = np.where(
            roll_ret > upper,
            1,
            np.where(roll_ret < lower, -1, 0),
        )
        feat_df["regime"] = regime

        feat_df = feat_df.dropna()
        return feat_df

    def save(self, suffix: str = "features") -> Path:
        """Save the feature-enriched DataFrame."""
        out_path = self.processed_dir / f"{self.ticker}_{self.year}_{suffix}.csv"
        self.df.to_csv(out_path)
        print(f"[FeatureEngineering] Saved features to: {out_path}")
        return out_path
