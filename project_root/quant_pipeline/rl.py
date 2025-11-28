# quant_pipeline/rl.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import PLOTS_DIR, METRICS_DIR
from .utils import ensure_dir
from .data import FinancialDataCollector


@dataclass
class BanditTrader:
    """
    Very simple contextual bandit-style trader.

    At each timestep, the agent chooses among:
        0 = stay flat, 1 = long, 2 = short
    Reward is next-day return * position.

    This is intentionally lightweight and illustrative, to support the
    RL discussion in the dissertation rather than a full Gym environment.
    """

    collector: FinancialDataCollector
    epsilon: float = 0.1
    plots_dir: Path = PLOTS_DIR
    metrics_dir: Path = METRICS_DIR

    def __post_init__(self) -> None:
        ensure_dir(self.plots_dir)
        ensure_dir(self.metrics_dir)

        self.df = self.collector.process()
        self.ticker = self.collector.ticker
        self.year = self.collector.year

        self.returns = self.df["Close"].pct_change().dropna().values
        self.n_actions = 3  # flat, long, short
        self.q_values = np.zeros(self.n_actions)
        self.action_counts = np.zeros(self.n_actions)
        self.rewards_history = []

    def choose_action(self) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_values))

    def action_to_position(self, action: int) -> int:
        return {0: 0, 1: 1, 2: -1}[action]

    def run(self) -> pd.DataFrame:
        """
        Run bandit learning over the available return series.
        """
        for t in range(len(self.returns) - 1):
            r_next = self.returns[t + 1]
            action = self.choose_action()
            position = self.action_to_position(action)
            reward = position * r_next

            # incremental update
            self.action_counts[action] += 1
            alpha = 1.0 / self.action_counts[action]
            self.q_values[action] += alpha * (reward - self.q_values[action])

            self.rewards_history.append(reward)

        cum_rewards = np.cumsum(self.rewards_history)
        df_out = pd.DataFrame(
            {
                "reward": self.rewards_history,
                "cum_reward": cum_rewards,
            },
            index=self.df.index[-len(self.rewards_history) :],
        )

        path = self.metrics_dir / f"{self.ticker}_{self.year}_bandit_rewards.csv"
        df_out.to_csv(path)
        print(f"[BanditRL] Saved reward history to: {path}")
        return df_out

    def plot_rewards(self, show: bool = True) -> None:
        if not self.rewards_history:
            self.run()

        df_out = pd.read_csv(
            self.metrics_dir / f"{self.ticker}_{self.year}_bandit_rewards.csv",
            index_col=0,
            parse_dates=True,
        )

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df_out.index, df_out["cum_reward"])
        ax.set_title("Bandit Trader Cumulative Reward")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Reward")
        ax.grid(True)

        out = self.plots_dir / f"{self.ticker}_{self.year}_bandit_cum_rewards.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[BanditRL] Saved cumulative reward plot to: {out}")

        if show:
            plt.show()
        else:
            plt.close(fig)
