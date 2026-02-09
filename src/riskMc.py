import numpy as np

import numpy as np

def simulate_correlated_returns(mu, cov, n_sims=200_000, seed=42):
    """
    Multivariate normal simulation using a PD-safe Cholesky.
    """
    rng = np.random.default_rng(seed)
    cov = np.asarray(cov, dtype=float)
    cov = (cov + cov.T) / 2  # ensure symmetric

    # Try Cholesky; if it fails, progressively add diagonal jitter
    jitter = 1e-12
    for _ in range(10):
        try:
            L = np.linalg.cholesky(cov + jitter * np.eye(cov.shape[0]))
            break
        except np.linalg.LinAlgError:
            jitter *= 10
    else:
        # Last resort: eigenvalue repair (nearest PD)
        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, 1e-10, None)
        cov_pd = vecs @ np.diag(vals) @ vecs.T
        cov_pd = (cov_pd + cov_pd.T) / 2
        L = np.linalg.cholesky(cov_pd)

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
