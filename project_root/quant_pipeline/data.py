# quant_pipeline/data.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from .config import RAW_DIR, PROCESSED_DIR, PLOTS_DIR
from .utils import ensure_dir


@dataclass
class FinancialDataCollector:
    """
    Handles data collection and basic cleaning for a single ticker-year slice.

    This is based on the original Colab implementation but refactored into a
    reusable class without interactive input() calls.
    """

    ticker: str
    year: int
    data_dir: Path = RAW_DIR.parent
    results_dir: Path = PLOTS_DIR.parent

    raw_data_file: Optional[Path] = None
    processed_file: Optional[Path] = None
    df: Optional[pd.DataFrame] = None

    def __post_init__(self) -> None:
        self.ticker = self.ticker.upper()
        self.year = int(self.year)

        self.raw_data_dir = self.data_dir / "Raw_Data"
        self.processed_data_dir = self.data_dir / "Processed"

        ensure_dir(self.raw_data_dir)
        ensure_dir(self.processed_data_dir)
        ensure_dir(self.results_dir)
        ensure_dir(PLOTS_DIR)

        self.raw_data_file = self.raw_data_dir / f"Raw_{self.ticker}_{self.year}.csv"
        self.processed_file = (
            self.processed_data_dir / f"Processed_{self.ticker}_{self.year}.csv"
        )

    # ------------------------------------------------------------------ #
    # Data collection and processing
    # ------------------------------------------------------------------ #
    def collect(self) -> Path:
        """
        Download raw OHLCV data from Yahoo! Finance and save as CSV
        if it does not already exist. Returns the raw CSV path.
        """
        if self.raw_data_file.exists():
            print(f"[DataCollector] Raw data already collected: {self.raw_data_file}")
            return self.raw_data_file

        print(
            f"[DataCollector] Downloading Yahoo! Finance data for {self.ticker} {self.year}..."
        )
        data = yf.download(
            self.ticker,
            start=f"{self.year}-01-01",
            end=f"{self.year}-12-31",
        )
        if data.empty:
            raise ValueError(f"No data returned for {self.ticker} in {self.year}.")

        data.to_csv(self.raw_data_file, index=True)
        print(f"[DataCollector] Raw data saved to: {self.raw_data_file}")
        return self.raw_data_file

    def process(self) -> pd.DataFrame:
        """
        Process the downloaded data and save cleaned data to processed CSV.
        - Flattens potential multi-index columns
        - Keeps Open, High, Low, Close, Volume
        - Ensures DatetimeIndex
        """
        # Ensure raw data
        raw_file = self.collect()

        # If processed already available: load and return
        if self.processed_file.exists():
            print(
                f"[DataCollector] Using existing processed file: {self.processed_file}"
            )
            self.df = pd.read_csv(self.processed_file, index_col=0, parse_dates=True)
            return self.df

        # Otherwise process from raw
        df = pd.read_csv(raw_file, header=[0, 1], index_col=0, parse_dates=True)
        # Flatten multi-index and keep main OHLCV
        df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Rename axis for clarity (columns axis)
        df = df.rename_axis(self.ticker, axis=1)

        # Basic cleaning: drop duplicates
        df = df[~df.index.duplicated(keep="first")]

        df.to_csv(self.processed_file)
        print(f"[DataCollector] Processed data saved to: {self.processed_file}")

        self.df = df
        return self.df

    # ------------------------------------------------------------------ #
    # Simple diagnostic plot
    # ------------------------------------------------------------------ #
    def plot_close_price(self, show: bool = True, save: bool = True) -> pd.Series:
        """
        Plot the Close price over time and optionally save to /results/plots.

        Returns the Close series for convenience.
        """
        if self.df is None:
            self.df = self.process()

        close = self.df["Close"]
        stats = close.describe()

        plt.figure(figsize=(12, 6))
        plt.plot(self.df.index, close, label="Close Price")
        plt.title(f"{self.ticker} Close Price - {self.year}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if save:
            ensure_dir(PLOTS_DIR)
            plot_path = PLOTS_DIR / f"ClosePrice_{self.ticker}_{self.year}.png"
            plt.savefig(plot_path, dpi=150)
            print(f"[DataCollector] Close price plot saved to: {plot_path}")

        if show:
            plt.show()
        else:
            plt.close()

        print("[DataCollector] Close price summary statistics:")
        print(stats)
        print(f"Variance: {close.var():.4f}")
        print(f"Std Dev:  {close.std():.4f}")
        print(f"Mean:     {close.mean():.4f}")

        return close
