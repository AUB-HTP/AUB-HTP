import numpy as np


# Var(log|Z_i| - log|Z_j|) at rho = 0, for standard normals Z_i, Z_j.
MAX_LOG_DIFFERENCE_VARIANCE: float = np.pi ** 2 / 4.0


def estimate_correlation_log_difference(data: np.ndarray, *, log_eps: float = 2.0 ** -20) -> np.ndarray:
    """
    Estimate the correlation matrix R of centered data.

    Each row is sqrt(A) G with G_i = sigma_i Z_i and Z ~ N(0, R), so A cancels in
    log|x_i| - log|x_j| = log(sigma_i / sigma_j) + log|Z_i| - log|Z_j|, and

        Var(log|x_i| - log|x_j|) = pi^2 / 4 - arcsin(rho_ij)^2

    which gives |rho_ij|. The sign comes from Sheppard's formula:
    P(x_i x_j > 0) = 1/2 + arcsin(rho_ij) / pi, so sgn(rho_ij) = sgn(2 p - 1).
    """
    _, n_features = data.shape

    log_abs = np.log(np.maximum(np.abs(data), log_eps))
    log_covariance = np.atleast_2d(np.cov(log_abs, rowvar=False))
    log_variance = np.diag(log_covariance)
    # Var(L_i - L_j) = Var(L_i) + Var(L_j) - 2 Cov(L_i, L_j)
    difference_variance = log_variance[:, None] + log_variance[None, :] - 2.0 * log_covariance

    arcsin_squared = np.clip(
        MAX_LOG_DIFFERENCE_VARIANCE - difference_variance, 0.0, MAX_LOG_DIFFERENCE_VARIANCE
    )
    absolute_correlation = np.sin(np.sqrt(arcsin_squared))

    # (S^T S)_ij = #(x_i x_j > 0) - #(x_i x_j < 0), which has the sign of 2 p - 1.
    signs = np.sign(data)
    correlation_sign = np.sign(signs.T @ signs)

    correlation_matrix = correlation_sign * absolute_correlation
    correlation_matrix = 0.5 * (correlation_matrix + correlation_matrix.T)
    np.fill_diagonal(correlation_matrix, 1.0)
    return correlation_matrix
