# quant_pipeline/utils.py
from pathlib import Path
import pandas as pd
from typing import Tuple


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def train_test_split_time_series(
    df: pd.DataFrame, test_size: float = 0.3
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological train/test split for time series.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe sorted by time.
    test_size : float
        Fraction for test set (0 < test_size < 1).

    Returns
    -------
    train, test : (pd.DataFrame, pd.DataFrame)
    """
    n = len(df)
    split_idx = int(n * (1 - test_size))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test
