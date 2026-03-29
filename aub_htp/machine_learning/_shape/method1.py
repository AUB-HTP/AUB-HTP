import numpy as np
from typing import Literal
from aub_htp.statistics import alpha_location, alpha_power
from .utils import estimate_marginals, compute_feature_wise_location

CoorolationEquationLiteral = Literal["5.12", "5.13", "5.14"]


def estimate_shape_method1(
    data: np.ndarray, 
    alpha: float, 
    *,
    coorolation_equation: CoorolationEquationLiteral = "5.12"
):
    data = np.asarray(data)
    data_is_one_dimensional = data.ndim == 1
    if data_is_one_dimensional:
        data = data.reshape(-1, 1)
    _n_samples, _n_features = data.shape

    small_sigma = estimate_marginals(data, alpha_kernel=alpha, alpha_data=alpha)
    location = compute_feature_wise_location(data, alpha)

    data_centered = data - location
    
    coorolation_matrix = estimate_correlation_matrix(data_centered, small_sigma, coorolation_equation=coorolation_equation)

    diagonal_matrix = np.diag(small_sigma)
    shape_matrix = diagonal_matrix @ coorolation_matrix @ diagonal_matrix
    return shape_matrix


def estimate_correlation_matrix(
    data: np.ndarray, 
    small_sigma: np.ndarray,
    *, 
    coorolation_equation: CoorolationEquationLiteral = "5.12",
    min_denominator: float = 0.0,
    min_keep_frac: float = 0.90,
) -> np.ndarray:
    """
    Estimate correlation matrix R using the ratio trick and Eq. (5.12–5.14):

        z_ij = X_i / X_j  (assuming centered data)
        Fit Cauchy(z_ij) → (μ_ij, γ_ij)
        Convert to ρ_ij via selected equation.
    Returns
    -------
    coorolation_matrix: (d, d) correlation matrix.
    """
    n, d = data.shape
    small_sigma = np.asarray(small_sigma, dtype=float).reshape(-1)
    coorolation_matrix = np.eye(d, dtype=float)
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
                rho = compute_correlation_element(mu_ij, gam_ij, small_sigma[i], small_sigma[j], coorolation_equation)
            coorolation_matrix[i, j] = coorolation_matrix[j, i] = float(np.clip(rho, -1.0, 1.0))

    # Symmetrize & normalize
    coorolation_matrix = 0.5 * (coorolation_matrix + coorolation_matrix.T)
    np.fill_diagonal(coorolation_matrix, 1.0)
    return coorolation_matrix


def fit_cauchy_1d(data: np.ndarray) -> tuple[float, float]:
    """
    Fit a Cauchy(μ, γ) to 1-D data using our α=1 location/power solver.

    For α=1, the α-power P* equals the Cauchy scale γ at optimum μ*.
    Returns (μ̂, γ̂).
    """
    mu_hat = alpha_location(data, alpha=1.0)
    gamma_hat = alpha_power(data, alpha=1.0)
    return float(np.squeeze(mu_hat)), gamma_hat


def compute_correlation_element(
    mu_ij: float,
    gamma_ij: float,
    sigma_i: float,
    sigma_j: float,
    coorolation_equation: CoorolationEquationLiteral,
) -> float:
    """
    Compute correlation ρ_ij using one of Eqs. (5.12)–(5.14):

      (5.12)  ρ_ij = μ_ij / √(μ_ij² + γ_ij²)
      (5.13)  ρ_ij = sgn(μ_ij) * √(1 − (σ_i² / σ_j²) * γ_ij²)
      (5.14)  ρ_ij = μ_ij * (σ_j / σ_i)

    Note: σ_i, σ_j are marginal Gaussian scales with σ = √2·γ.
    """
    mu = float(mu_ij)
    gam = float(max(gamma_ij, np.finfo(float).eps))
    si  = float(max(sigma_i, np.finfo(float).eps))
    sj  = float(max(sigma_j, np.finfo(float).eps))

    if coorolation_equation == "5.12":
        rho = mu / np.hypot(mu, gam)
    elif coorolation_equation == "5.13":
        term = 1.0 - (si * si / (sj * sj)) * (gam * gam)
        term = max(term, 0.0)
        rho = np.sign(mu) * np.sqrt(term)
    elif coorolation_equation == "5.14":
        rho = mu * (sj / si)
    else:
        raise ValueError("equation must be 5.12, 5.13, or 5.14")

    return float(np.clip(rho, -1.0, 1.0))
