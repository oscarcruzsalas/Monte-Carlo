import yfinance as yf
import numpy as np
import pandas as pd

def download_prices(
    tickers,
    start="2018-01-01",
    end=None
):
    """
    Download adjusted close prices from Yahoo Finance.
    """
    df = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False
    )

    prices = df["Adj Close"].dropna(how="all")
    return prices

def compute_log_returns(prices: pd.DataFrame):
    """
    Compute daily log returns.
    """
    return np.log(prices / prices.shift(1)).dropna()

def estimate_mu_cov(returns: pd.DataFrame):
    """
    Estimate mean vector and covariance matrix.
    """
    mu = returns.mean().values
    cov = returns.cov().values
    return mu, cov
