# Monte Carlo Risk Simulation for Portfolio Management

## Overview
This project implements a Monte Carlo simulation framework to assess
the risk of a multi-asset investment portfolio. Using historical market
data, the model estimates portfolio risk through Value at Risk (VaR)
and Conditional Value at Risk (CVaR), explicitly accounting for asset
correlations and diversification effects.

The project is designed as a reproducible, educational risk analytics
tool suitable for portfolio management and financial risk analysis.

---

## Methodology
1. Historical daily prices are downloaded from Yahoo Finance.
2. Log returns are computed and used to estimate:
   - the mean return vector (μ)
   - the covariance matrix (Σ)
3. Correlated asset returns are simulated using a multivariate normal
   distribution and Cholesky decomposition.
4. Portfolio PnL is generated from simulated returns and portfolio weights.
5. Risk metrics are computed:
   - Value at Risk (VaR)
   - Conditional Value at Risk (CVaR / Expected Shortfall)

---

## Key Concepts
- Monte Carlo simulation
- Multivariate normal returns
- Portfolio VaR and CVaR
- Correlation and diversification effects
- Stress testing of volatility and correlation

---

## Tools Used
- Python
- NumPy
- Pandas
- Matplotlib
- yfinance

---

## Results
The simulation demonstrates how portfolio risk is driven not only by
individual asset volatility but also by cross-asset correlations.
Stress scenarios show that increases in volatility and correlation
can significantly worsen tail risk, highlighting the importance of
diversification and dependency modeling in portfolio risk management.

---

## Data Source
Historical price data is obtained from Yahoo Finance via the `yfinance`
library and is used strictly for educational and research purposes.

---

## How to Run
```bash
pip install -r requirements.txt
