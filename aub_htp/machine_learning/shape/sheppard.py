import numpy as np

from .utils import warn_pairs_without_samples


def estimate_correlation_sheppard(data: np.ndarray) -> np.ndarray:
    """
    Estimate the correlation matrix R of centered data from the signs of x_i x_j.

    Each row is sqrt(A) G with G_i = sigma_i Z_i and Z ~ N(0, R). A and sigma are
    positive, so sgn(x_i x_j) = sgn(Z_i Z_j), and Sheppard's formula gives

        p = P(x_i x_j > 0) = 1/2 + arcsin(rho_ij) / pi
        rho_ij = sin(pi (p - 1/2))

    p is estimated as #(x_i x_j > 0) / #(x_i x_j != 0). Rows where the product is
    exactly 0 are left out, so they don't count as negative. A pair with no rows left
    gets rho_ij = 0, with a warning.
    """
    signs = np.sign(data)
    # (S^T S)_ij = #(x_i x_j > 0) - #(x_i x_j < 0)
    positive_minus_negative = signs.T @ signs
    is_nonzero = np.abs(signs)
    # (|S|^T |S|)_ij = #(x_i x_j != 0)
    nonzero = is_nonzero.T @ is_nonzero

    has_samples = nonzero > 0
    if not has_samples.all():
        warn_pairs_without_samples()

    # p - 1/2 = (#positive - #negative) / (2 #nonzero), or 0 for a pair with no samples.
    p_minus_half = np.divide(
        positive_minus_negative, 2.0 * nonzero, out=np.zeros_like(nonzero), where=has_samples
    )
    correlation_matrix = np.sin(np.pi * p_minus_half)

    correlation_matrix = 0.5 * (correlation_matrix + correlation_matrix.T)
    np.fill_diagonal(correlation_matrix, 1.0)
    return correlation_matrix
