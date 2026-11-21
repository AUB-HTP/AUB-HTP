from typing import Literal
import numpy as np
from aub_htp.statistics import alpha_location, alpha_power

RatioEquationLiteral = Literal[1, 2, 3]


def estimate_correlation_ratio(
    data: np.ndarray,
    small_sigma: np.ndarray,
    *,
    equation: RatioEquationLiteral = 1,
) -> np.ndarray:
    """
    Estimate correlation matrix R using the ratio trick:

        z_ij = X_i / X_j  (assuming centered data; rows with X_j = 0 are left out)
        Fit Cauchy(z_ij) → (μ_ij, γ_ij)
        Convert to ρ_ij with the selected equation (see compute_correlation_element).

    small_sigma holds σ_i = √Σ_ii for each feature. Only equations 2 and 3 use it.

    Returns
    -------
    correlation_matrix: (d, d) correlation matrix.
    """
    _, d = data.shape
    small_sigma = np.asarray(small_sigma, dtype=float).reshape(-1)
    correlation_matrix = np.eye(d, dtype=float)

    # Full pairwise computation
    for i in range(d):
        num_i = data[:, i]
        for j in range(i + 1, d):
            den_j = data[:, j]
            mask = den_j != 0
            z = num_i[mask] / den_j[mask]
            mu_ij, gam_ij = fit_cauchy_1d(z)
            rho = compute_correlation_element(mu_ij, gam_ij, small_sigma[i], small_sigma[j], equation)
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


def compute_correlation_element(
    mu_ij: float,
    gamma_ij: float,
    sigma_i: float,
    sigma_j: float,
    equation: RatioEquationLiteral,
) -> float:
    """
    Compute correlation ρ_ij from the Cauchy fit of z_ij = X_i / X_j, which has

        μ_ij = ρ_ij σ_i / σ_j,    γ_ij = (σ_i / σ_j) √(1 − ρ_ij²)

    using one of:

      (1)  ρ_ij = μ_ij / √(μ_ij² + γ_ij²)
      (2)  ρ_ij = sgn(μ_ij) √(1 − (σ_j² / σ_i²) γ_ij²)
      (3)  ρ_ij = μ_ij σ_j / σ_i
    """
    mu = float(mu_ij)
    gam = float(max(gamma_ij, np.finfo(float).eps))
    # Only the ratio of the sigmas matters, so it has no scale. sigma_i = 0 gives nan.
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_ratio = float(np.float64(sigma_j) / np.float64(sigma_i))

    if equation == 1:
        rho = mu / np.hypot(mu, gam)
    elif equation == 2:
        term = 1.0 - (sigma_ratio * gam) ** 2
        rho = np.sign(mu) * np.sqrt(np.maximum(term, 0.0))
    elif equation == 3:
        rho = mu * sigma_ratio
    else:
        raise ValueError(f"equation must be 1, 2 or 3, got {equation!r}.")

    return float(np.clip(rho, -1.0, 1.0))
