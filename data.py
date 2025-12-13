# data.py
"""
Data manager supporting two modes:
1) Simulated data:
       - Cluster-based correlation matrix
       - True covariance Σ_true
       - Multivariate normal log-returns
       - Simulated price paths and arithmetic returns

Citation: 1. Chatgpt generated 60% of the code
"""

from typing import Optional, List
import numpy as np
import pandas as pd

from config import DataConfig


class MarketData:
    """
    Data interface for the project.
    """

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg

        self.tickers: List[str] = []
        self.prices: Optional[pd.DataFrame] = None
        self.returns: Optional[pd.DataFrame] = None
        self.Sigma_true: Optional[np.ndarray] = None  # Only for simulated data

        self._generate_simulated_data()


    def _generate_simulated_data(self) -> None:
        """
        Generate simulated data with a cluster-based correlation matrix.

        Structure similar to López de Prado (2016):
        - Assets belong to clusters
        - Within-cluster correlation is high
        - Between-cluster correlation is low
        - Volatilities differ across assets

        citation: chatgpt generated the code after I specified the data generating process
        """

        N = self.cfg.n_assets
        T = self.cfg.n_periods
        n_clusters = self.cfg.n_clusters
        rho_in = self.cfg.within_corr
        rho_out = self.cfg.between_corr
        vol_low = self.cfg.sim_vol_low
        vol_high = self.cfg.sim_vol_high
        annualized_ret_avg_low = self.cfg.annualized_ret_avg_low
        annualized_ret_avg_high = self.cfg.annualized_ret_avg_high
        seed = self.cfg.sim_seed

        rng = np.random.default_rng(seed)

        base_size = N // n_clusters
        remainder = N % n_clusters

        cluster_sizes = []
        for k in range(n_clusters):
            curClusterSize = base_size + (1 if k < remainder else 0)
            cluster_sizes.append(curClusterSize)

        cluster_labels = np.empty(N, dtype=int)
        start = 0
        for c, size_c in enumerate(cluster_sizes):
            end = start + size_c
            cluster_labels[start:end] = c
            start = end

        corr_matrix = np.where(
            cluster_labels[:, None] == cluster_labels[None, :],
            rho_in,
            rho_out
        )
        np.fill_diagonal(corr_matrix, 1.0)

        vols = rng.uniform(vol_low, vol_high, size=N)
        D = np.diag(vols)
        Sigma_true = D @ corr_matrix @ D
        self.Sigma_true = Sigma_true

        daily_ret_low = annualized_ret_avg_low / 252
        daily_ret_high = annualized_ret_avg_high / 252

        mu = rng.uniform(daily_ret_low, daily_ret_high, size=N)
        arith_rets = rng.multivariate_normal(
            mean=mu,
            cov=Sigma_true,
            size=T
        )  # shape: (T, N)

        arith_rets = np.clip(arith_rets, -0.99, None)

        S0 = 100.0
        cum_growth = np.cumprod(1.0 + arith_rets, axis=0)   # shape: (T, N)
        prices = S0 * cum_growth

        # 7. Wrap into DataFrames
        dates = [f"date{i+1}" for i in range(T)]
        tickers = [f"Asset_{i+1}" for i in range(N)]

        self.tickers = tickers
        self.prices = pd.DataFrame(prices, index=dates, columns=tickers)
        self.returns = pd.DataFrame(arith_rets, index=dates, columns=tickers)

        # print(f"[MarketData] Simulated data generated: {N} assets, {T} periods.")
        # print(f"[MarketData] Cluster sizes: {cluster_sizes}")
        # print(f"[MarketData] True covariance shape: {Sigma_true.shape}")
    

    def get_returns(self) -> pd.DataFrame:
        if self.returns is None:
            raise ValueError("Returns are not available.")
        return self.returns.copy()

    def get_prices(self) -> pd.DataFrame:
        if self.prices is None:
            raise ValueError("Prices are not available.")
        return self.prices.copy()

    def get_true_covariance(self) -> Optional[np.ndarray]:
        """Return Σ_true only for simulated data."""
        return self.Sigma_true
    


