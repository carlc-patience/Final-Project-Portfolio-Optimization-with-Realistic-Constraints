from dataclasses import dataclass
from typing import List, Optional

"""
Global configuration for the portfolio optimization project, including settings for both simulated and real market data.
Citation: Used chatgpt in helping me writing the code(80% chatgpt, 20% self modification)
"""
@dataclass
class DataConfig:
    use_simulated: bool = True
    n_assets: int = 50                # Number of assets in the simulated universe
    n_periods: int = 200              # Number of time periods
    sim_seed: int = 42                # Random seed for reproducibility
    n_clusters: int = 3               # Number of industry clusters
    within_corr: float = 0.8          # Correlation within the same cluster
    between_corr: float = 0.2         # Correlation between different clusters

    # Volatility range for each asset (used to construct the covariance matrix)
    sim_vol_low: float = 0.1
    sim_vol_high: float = 0.3
    annualized_ret_avg_low: float = 0.02
    annualized_ret_avg_high: float = 0.15


@dataclass
class BacktestConfig:
    """Configuration for portfolio backtesting."""
    rebalance_freq: int = 7
    lookBackWindow: int = 128
    initial_nav: float = 1.0

