from config import *
from data import *
from estimator import *
from portfolio import *
from itertools import product
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cvxpy

"""
Backtest Class:
1. input the generated data(return data, prices data, true covariance matrix)
2. do rolling window backtest using 4 portfolio constructors(sample covariance + minVar, LinearShrinkage + minVar, sample covariance + Hrp optimizer, LinearShrinkage + Hrp optimizer), calculkate the portfolio weights, return serires
3. evaluation metrics
4. visualization plots
"""

class Backtest:
    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg
        self.estimators = ['sample', 'shrinkage']
        self.optimizers = ['minVar', 'hrp']
        self.estimatorOptimizerComb = list(product(self.estimators, self.optimizers))
        self.retSeries = pd.DataFrame(index=self.estimatorOptimizerComb)
        self.equalWeightLabel = ("equal", "ew")
        self.constructor = portfolioConstructor()

    
    def calculateReturns(self, retData):

        assert isinstance(retData, pd.DataFrame), 'Return Data must be a pandas DataFrame with shape (N, T)'

        allRows = self.estimatorOptimizerComb + [self.equalWeightLabel]
        N, T = retData.shape[0], retData.shape[1]
        start = self.cfg.lookBackWindow
        self.rebalanceDates = retData.columns[start::self.cfg.rebalance_freq]
        self.rebalanceDatesIndex = list(range(start, T, self.cfg.rebalance_freq))
        row_index = pd.MultiIndex.from_tuples(allRows, names=["estimator", "optimizer"])
        self.retSeries = pd.DataFrame(index=row_index, columns=self.rebalanceDates, dtype=float)

        for date, dateIndex in zip(self.rebalanceDates, self.rebalanceDatesIndex):
            pastRetData = retData.iloc[:, dateIndex - self.cfg.lookBackWindow:dateIndex].values  # (N, window)
            futureRetData = retData.iloc[:, dateIndex: dateIndex + self.cfg.rebalance_freq].values  # (N, k)

            for estimator, optimizer in self.estimatorOptimizerComb:
                weights = self.constructor.fit(pastRetData, estimator, optimizer)
                portRet = futureRetData.T @ weights
                cumRet = np.prod(1 + portRet) - 1
                self.retSeries.loc[(estimator, optimizer), date] = cumRet
            equalWeights = np.ones(N) / N
            portRet = futureRetData.T @ equalWeights
            cumRet = np.prod(1+portRet) - 1
            self.retSeries.loc[self.equalWeightLabel, date] = cumRet

        return self.retSeries
    

    
    def calculateEvalMetrics(self):
        """
        this function calculates the 'annualized return', 'annualized volatility', 'sharpe', 'max drawdown' for portfolios constructed by different estimators
        """
        assert not self.retSeries.empty, "Portfolio returns havenn't been calculated yet, run calculateReturns() first."
        numPeriods = int(252 / self.cfg.rebalance_freq)
        metrics = []

        for idx, row in self.retSeries.iterrows():
            retArithm = row.values
            retAnnualized = np.mean(retArithm) * numPeriods
            vol = np.std(retArithm, ddof=1) * np.sqrt(numPeriods)
            sharpe = retAnnualized / vol if vol > 0 else np.nan
            cumRet = (1 + retArithm).cumprod()           
            cumMax = np.maximum.accumulate(cumRet)
            drawdown = cumRet / cumMax - 1.0  
            max_dd = drawdown.min()             

            metrics.append({
                "index": idx,
                "annualized_ret": retAnnualized,
                "annualized_volatility": vol,
                "sharpe": sharpe,
                "max_drawdown": max_dd, 
            })

        self.metrics_df = pd.DataFrame(metrics)
        self.metrics_df.set_index("index", inplace=True)
        return self.metrics_df


    def plotReturnCurves(self, ax=None, max_xticks=10):
        """
        plot the net asset value of different portfolios, 80% handcrafted, consulted chatgpt for plotting function details
        """
        assert not self.retSeries.empty, "Portfolio returns havenn't been calculated yet, run calculateReturns() first."
        cumReturn = (1 + self.retSeries).cumprod(axis=1)

        if ax is None:
            plt.figure(figsize=(10, 6))
            ax = plt.gca()

        for estimator, optimizer in cumReturn.index:
            label = f"{estimator}-{optimizer}" 
            ax.plot(self.rebalanceDates, cumReturn.loc[(estimator, optimizer)].values, label=label)

        num_points = len(self.rebalanceDates)
        if num_points > 0:
            step = max(1, num_points // max_xticks)
            tick_positions = np.arange(0, num_points, step)
            tick_labels = [self.rebalanceDates[i] for i in tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45)
        ax.set_xlabel("Date")
        ax.set_ylabel("Net Asset Value")
        ax.set_title("Portfolio Return Curves")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()

        return ax

