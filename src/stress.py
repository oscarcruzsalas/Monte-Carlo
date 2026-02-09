import numpy as np

def stress_volatility(cov, multiplier=1.5):
    """
    Increase volatility by a multiplier.
    If vol -> m*vol, then cov -> m^2 * cov.
    """
    cov = np.asarray(cov, dtype=float)
    return cov * (multiplier ** 2)

def make_pos_def(A, eps=1e-10):
    """
    Force a symmetric matrix to be positive definite via eigenvalue clipping.
    This ensures Cholesky works.
    """
    A = np.asarray(A, dtype=float)
    A = (A + A.T) / 2  # symmetrize

    # Eigen-decomposition for symmetric matrix
    vals, vecs = np.linalg.eigh(A)

    # Clip eigenvalues to be at least eps
    vals = np.clip(vals, eps, None)

    A_pd = vecs @ np.diag(vals) @ vecs.T
    return (A_pd + A_pd.T) / 2  # resymmetrize

def stress_correlation(cov, corr_bump=0.2, cap=0.99):
    """
    Increase off-diagonal correlations by corr_bump while preserving variances.
    Then repair the covariance matrix to be positive definite.
    """
    cov = np.asarray(cov, dtype=float)

    # std devs
    std = np.sqrt(np.diag(cov))
    outer_std = np.outer(std, std)

    # correlation matrix
    corr = cov / outer_std
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)

    # bump correlations
    corr_stressed = corr.copy()
    k = corr.shape[0]
    for i in range(k):
        for j in range(k):
            if i != j:
                corr_stressed[i, j] = np.clip(corr[i, j] + corr_bump, -cap, cap)

    # rebuild covariance with original variances
    cov_stressed = corr_stressed * outer_std

    # Repair so Cholesky works
    cov_stressed = make_pos_def(cov_stressed)

    return cov_stressed
