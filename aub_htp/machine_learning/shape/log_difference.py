import numpy as np

from .utils import divide_by_sigma, warn_pairs_without_samples


# Var(log|Z_i| - log|Z_j|) at rho = 0, for standard normals Z_i, Z_j.
MAX_LOG_DIFFERENCE_VARIANCE: float = np.pi ** 2 / 4.0


def estimate_correlation_log_difference(
    data: np.ndarray,
    small_sigma: np.ndarray,
    *,
    log_eps: float = 2.0 ** -20,
) -> np.ndarray:
    """
    Estimate the correlation matrix R of centered data.

    Each feature is divided by small_sigma first, so the floor log_eps is relative to the feature's scale.
    This only shifts log|x_i| by a constant, so the variance below is unchanged.

    Each row is sqrt(A) G with G_i = sigma_i Z_i and Z ~ N(0, R), so A cancels in
    log|x_i| - log|x_j| = log(sigma_i / sigma_j) + log|Z_i| - log|Z_j|, and

        Var(log|x_i| - log|x_j|) = pi^2 / 4 - arcsin(rho_ij)^2

    which gives |rho_ij|. The sign comes from Sheppard's formula:
    P(x_i x_j > 0) = 1/2 + arcsin(rho_ij) / pi, so sgn(rho_ij) = sgn(2 p - 1).

    log|0| is not defined, so rows with x_i = 0 or x_j = 0 are left out of the variance.
    A pair with fewer than 2 rows left gets rho_ij = 0, with a warning.
    """
    _, n_features = data.shape
    data = divide_by_sigma(data, small_sigma)

    log_abs = np.log(np.maximum(np.abs(data), log_eps))
    is_nonzero = data != 0

    absolute_correlation = np.eye(n_features, dtype=float)
    has_pair_without_samples = False
    for i in range(n_features):
        for j in range(i + 1, n_features):
            mask = is_nonzero[:, i] & is_nonzero[:, j]
            if mask.sum() < 2:
                has_pair_without_samples = True
                continue
            difference_variance = np.var(log_abs[mask, i] - log_abs[mask, j], ddof=1)
            arcsin_squared = np.clip(
                MAX_LOG_DIFFERENCE_VARIANCE - difference_variance, 0.0, MAX_LOG_DIFFERENCE_VARIANCE
            )
            absolute_correlation[i, j] = absolute_correlation[j, i] = np.sin(np.sqrt(arcsin_squared))

    if has_pair_without_samples:
        warn_pairs_without_samples()

    # (S^T S)_ij = #(x_i x_j > 0) - #(x_i x_j < 0), which has the sign of 2 p - 1.
    signs = np.sign(data)
    correlation_sign = np.sign(signs.T @ signs)

    correlation_matrix = correlation_sign * absolute_correlation
    np.fill_diagonal(correlation_matrix, 1.0)
    return correlation_matrix
