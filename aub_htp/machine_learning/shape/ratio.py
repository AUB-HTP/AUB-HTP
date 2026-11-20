import numpy as np
from aub_htp.statistics import alpha_location, alpha_power


def estimate_correlation_ratio(
    data: np.ndarray,
    *,
    min_denominator: float = 0.0,
    min_keep_frac: float = 0.90,
) -> np.ndarray:
    """
    Estimate correlation matrix R using the ratio trick and Eq. (5.12):

        z_ij = X_i / X_j  (assuming centered data)
        Fit Cauchy(z_ij) → (μ_ij, γ_ij)
        ρ_ij = μ_ij / √(μ_ij² + γ_ij²)
    Returns
    -------
    correlation_matrix: (d, d) correlation matrix.
    """
    n, d = data.shape
    correlation_matrix = np.eye(d, dtype=float)
    eps = np.finfo(float).eps

    def _mask_denom(den: np.ndarray) -> np.ndarray:
        thr = max(min_denominator, eps)
        return np.abs(den) > thr

    # Full pairwise computation
    for i in range(d):
        num_i = data[:, i]
        for j in range(i + 1, d):
            den_j = data[:, j]
            mask = _mask_denom(den_j)
            if mask.sum() < max(5, int(min_keep_frac * n)):
                rho = 0.0
            else:
                z = num_i[mask] / den_j[mask]
                mu_ij, gam_ij = fit_cauchy_1d(z)
                rho = compute_correlation_element(mu_ij, gam_ij)
            correlation_matrix[i, j] = correlation_matrix[j, i] = float(np.clip(rho, -1.0, 1.0))

    # Symmetrize & normalize
    correlation_matrix = 0.5 * (correlation_matrix + correlation_matrix.T)
    np.fill_diagonal(correlation_matrix, 1.0)
    return correlation_matrix


def fit_cauchy_1d(data: np.ndarray) -> tuple[float, float]:
    """
    Fit a Cauchy(μ, γ) to 1-D data using our α=1 location/power solver.

    For α=1, the α-power P* equals the Cauchy scale γ at optimum μ*.
    Returns (μ̂, γ̂).
    """
    mu_hat = alpha_location(data, alpha=1.0)
    gamma_hat = alpha_power(data, alpha=1.0)
    return float(np.squeeze(mu_hat)), gamma_hat


def compute_correlation_element(mu_ij: float, gamma_ij: float) -> float:
    """
    Compute correlation ρ_ij using Eq. (5.12):

        ρ_ij = μ_ij / √(μ_ij² + γ_ij²)
    """
    mu = float(mu_ij)
    gam = float(max(gamma_ij, np.finfo(float).eps))
    rho = mu / np.hypot(mu, gam)
    return float(np.clip(rho, -1.0, 1.0))
