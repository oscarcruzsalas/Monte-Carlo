import numpy as np

def simulate_correlated_returns(mu, cov, n_sims=200_000, seed=42):
    """
    Multivariate normal simulation using Cholesky.
    """
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(cov)

    z = rng.standard_normal(size=(n_sims, len(mu)))
    returns = z @ L.T + mu
    return returns

def portfolio_pnl(returns, weights, portfolio_value=1_000_000):
    """
    Compute portfolio PnL from simulated returns.
    """
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()

    portfolio_returns = returns @ w
    pnl = portfolio_value * portfolio_returns
    return pnl

def var_cvar(pnl, alpha=0.05):
    """
    Compute VaR and CVaR (Expected Shortfall).
    """
    q = np.quantile(pnl, alpha)
    var = -q
    tail = pnl[pnl <= q]
    cvar = -tail.mean()
    return var, cvar

def summarize_risk(pnl, alpha=0.05):
    """
    Summary risk statistics.
    """
    var, cvar = var_cvar(pnl, alpha)

    return {
        "Expected PnL": pnl.mean(),
        "PnL Volatility": pnl.std(),
        "Prob(Loss)": (pnl < 0).mean(),
        f"VaR {int((1-alpha)*100)}%": var,
        f"CVaR {int((1-alpha)*100)}%": cvar
    }
