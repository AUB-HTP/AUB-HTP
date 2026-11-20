import numpy as np
from aub_htp.statistics import alpha_location, alpha_power


def estimate_correlation_ratio(data: np.ndarray) -> np.ndarray:
    """
    Estimate correlation matrix R using the ratio trick and Eq. (5.12):

        z_ij = X_i / X_j  (assuming centered data; rows with X_j = 0 are left out)
        Fit Cauchy(z_ij) → (μ_ij, γ_ij)
        ρ_ij = μ_ij / √(μ_ij² + γ_ij²)
    Returns
    -------
    correlation_matrix: (d, d) correlation matrix.
    """
    _, d = data.shape
    correlation_matrix = np.eye(d, dtype=float)

    # Full pairwise computation
    for i in range(d):
        num_i = data[:, i]
        for j in range(i + 1, d):
            den_j = data[:, j]
            mask = den_j != 0
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

    Returns (μ̂, γ̂).
    """
    mu_hat = alpha_location(data, alpha=1.0)
    mu_hat = float(np.squeeze(mu_hat))

    # For α=1, the α-power of the data centered at μ̂ equals the Cauchy scale γ.
    gamma_hat = alpha_power(data - mu_hat, alpha=1.0)
    
    return mu_hat, gamma_hat


def compute_correlation_element(mu_ij: float, gamma_ij: float) -> float:
    """
    Compute correlation ρ_ij using Eq. (5.12):

        ρ_ij = μ_ij / √(μ_ij² + γ_ij²)
    """
    mu = float(mu_ij)
    gam = float(max(gamma_ij, np.finfo(float).eps))
    rho = mu / np.hypot(mu, gam)
    return float(np.clip(rho, -1.0, 1.0))
