import numpy as np

def stress_volatility(cov, multiplier=1.5):
    """
    Increase volatility by a multiplier.
    """
    return cov * multiplier**2

def stress_correlation(cov, corr_bump=0.2):
    """
    Increase correlations while keeping variances constant.
    """
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)

    corr_stressed = corr.copy()
    for i in range(len(corr)):
        for j in range(len(corr)):
            if i != j:
                corr_stressed[i, j] = min(corr[i, j] + corr_bump, 0.99)

    cov_stressed = corr_stressed * np.outer(std, std)
    return cov_stressed
