from config import *
from data import MarketData
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from estimator import *
from backtest import Backtest
from portfolio import *
from collections import defaultdict
from tqdm import tqdm

def monte_carlo_annualized_volatility(n_assets: int,lookBackWindow: int,trials: int = 200, savefig: bool = False):
    """
    Monte Carlo Simulation
    Citation: I gave instructions for chatgpt to generate the code
    """
    n_periods = 2000
    rebalance_freq = 7
    base_seed = 66
    n_clusters = 5
    within_corr = 0.6
    between_corr = 0.2
    sim_vol_low = 0.02
    sim_vol_high = 0.10
    annualized_ret_avg_low = 0.02
    annualized_ret_avg_high = 0.15

    vols_dict = defaultdict(list)
    strat_index = None

    for k in range(trials):
        sim_seed = base_seed + k
        data_cfg = DataConfig(
            use_simulated=True,
            n_assets=n_assets,
            n_periods=n_periods,
            sim_seed=sim_seed,
            n_clusters=n_clusters,
            within_corr=within_corr,
            between_corr=between_corr,
            sim_vol_low=sim_vol_low,
            sim_vol_high=sim_vol_high,
            annualized_ret_avg_low=annualized_ret_avg_low,
            annualized_ret_avg_high=annualized_ret_avg_high,
        )
        data = MarketData(data_cfg)
        rets = data.get_returns()   
        bt_cfg = BacktestConfig(
            rebalance_freq=rebalance_freq,
            lookBackWindow=lookBackWindow,
        )
        bt = Backtest(bt_cfg)

        bt.calculateReturns(rets.T)
        metrics = bt.calculateEvalMetrics()  

        if strat_index is None:
            strat_index = metrics.index 
        for idx, vol in metrics["annualized_volatility"].items():
            vols_dict[idx].append(float(vol))

    for key in vols_dict:
        vols_dict[key] = np.array(vols_dict[key])

    baseline_label = None
    for key in vols_dict.keys():
        if isinstance(key, tuple) and len(key) >= 2:
            if (key[0] == "sample") and (key[1] == "minVar"):
                baseline_label = key
                break

    if baseline_label is None:
        raise ValueError(
            "Cannot find baseline strategy ('sample','minVar') in vols_dict.\n"
            f"Available keys: {list(vols_dict.keys())}"
        )

    baseline_mean = vols_dict[baseline_label].mean()

    n_strats = len(strat_index)
    ncols = 2
    nrows = int(np.ceil(n_strats / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows), sharex=True)
    axes = axes.flatten()

    for ax, idx in zip(axes, strat_index):
        vals = vols_dict[idx]

        ax.hist(vals, bins=20, alpha=0.7, edgecolor="black")
        ax.axvline(baseline_mean, linestyle="--", linewidth=2)

        if isinstance(idx, tuple) and len(idx) >= 2:
            title = f"{idx[0]}-{idx[1]}"
        else:
            title = str(idx)

        ax.set_title(title)
        ax.set_xlabel("Annualized Volatility")
        ax.set_ylabel("Frequency")
        ax.grid(alpha=0.3)

    for j in range(len(strat_index), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(
        f"Monte Carlo distribution of annualized volatility\n"
        f"(baseline = sample-minVar mean = {baseline_mean:.4f})",
        y=1.02,
        fontsize=14,
    )
    plt.tight_layout()
    if savefig:
        plt.savefig(f'./MC_N{n_assets}_T{lookBackWindow}_trials{trials}')
    plt.show()

    return vols_dict, baseline_mean
