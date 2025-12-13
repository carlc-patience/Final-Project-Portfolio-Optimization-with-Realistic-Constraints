# Final-Project-Portfolio-Optimization-with-Realistic-Constraints

## 1. Project Overview:
- This project focuses on implementing and comparing several portfolio construction techniques based on different covariance estimators, with an emphasis on the Hierarchical Risk Parity (HRP) methodology. The goal is to analyze how alternative covariance estimation methods and risk allocation frameworks influence portfolio robustness, diversification, and out-of-sample performance.

---
## 2. Citations: The project is base on the following 2 academic papers: 
1. Building Diversified Portfolios that Outperform Out-of-Sample:https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://papers.ssrn.com/sol3/papers.cfm%3Fabstract_id%3D2708678&ved=2ahUKEwias8eTyqWRAxXxEFkFHbQzOAgQFnoECEIQAQ&usg=AOvVaw3vhm-B0nmRd-gIlgVgkzxZ
2. Essays on risk and return in the stock market(Chatpter1): http://hdl.handle.net/1721.1/11875


---

## 3. File Components of the Project
1. config.py: defines the parameters used in other files, to make the code look cleaner
2. data.py: defines a class that specifies the data generating process for simulation, and generate teh price and return paths for the dataset
3. estimator.py: This file construct sample covariance estimator and linear shrinkage covariance estimator, using class methods
4. portfolio.py: This file consist of a portfolioConstructer class, which achieves the fowllowing function: given the return data for a universe of stocks, use different estimators to estimate the covariance matrix, construct either the minimum-variance portfolio weights, or the weights constructed by Hierachical Risk Parity(HRP). The algorithm for HRP is a bit complicated
5. backtest.py: The pipeline that generates the data, conduct porfolio optimization task on a rolling basis, form the portfolio constructed by different optimizers, calculate and visualize the results obtained by different optimizers
6. MC.py: Runs Monte Carlo simulations to compare the annualized volatility of portfolios constructed by different methods, visualizes the simulation result
7. portfolio_optimization.ipynb: The user interface for the project

## 4. User Interface(files to run)
1. portfolio_optimization.ipynb: have 3 sections: 
   1. runs the full pipeline once, visualizes the cumulative portfolio returns and calculates relevant metrics
   2. Monte Carlo Simulation: Visualizes the risk(measured by annualized volatility) of different optimizers under different hyperparameter settings(can clearly see that Hrp performs better than traditional sample covariance + minimum variance framework)
   3. test-cases for single linkage algorithm(used in Hrp algorithm): I implemented the single linkage algorithm using priority queue and union find, and compared my version with that of the scipy built-in package to ensure correctness
- Interpretation: The key parameters that distinguishes the performance of different techniques are **num_assets** and **lookBackWindow**. You can verify that when **num_assets << lookBackWindow**, the traditional Samplke Covariance + Minimum Variance Optimizer works well. However, with **num_assets comparable or larger than lookBackWindow**, the performance of shrinkage method and HRP becomes much better, this is becomes the traditional sample covaraince matrix generated in this case is very likely to be singular, and bad to be used for minimum variance optimizer.


## 5. Dependencies(No specific version control required)
- scipy
- dataclasses
- typing
- heapq
- cvxpy
- collections