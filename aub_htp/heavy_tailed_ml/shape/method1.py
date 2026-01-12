# shape/method1.py
"""
Method 1 (ratio-based) shape estimation with equation selector (5.12–5.14).
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, Optional, Any

from ..location import location_L
from ..power import P_alpha
from .utilities import estimate_marginals_sigma

Array = np.ndarray


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _as_2d_samples(X: Array) -> Array:
    """Ensure data is (n, d)."""
    X = np.asarray(X, dtype=float)
    return X[:, None] if X.ndim == 1 else X


def fit_cauchy_1d(z: Array) -> Tuple[float, float]:
    """
    Fit a Cauchy(μ, γ) to 1-D data using our α=1 location/power solver.

    For α=1, the α-power P* equals the Cauchy scale γ at optimum μ*.
    
    Parameters
    ----------
    z : ndarray
        1-D data to fit.
    
    Returns
    -------
    mu_hat : float
        Fitted location parameter.
    gamma_hat : float
        Fitted scale parameter.
    """
    z = _as_2d_samples(np.asarray(z).reshape(-1))
    mu_hat, P_star = location_L(z, alpha=1.0, h_lookup=None, model="multivariate")
    gamma_hat = float(P_star)
    return float(np.squeeze(mu_hat)), gamma_hat


def _rho_from_equation(
    mu_ij: float,
    gamma_ij: float,
    sigma_i: float,
    sigma_j: float,
    equation: int,
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
    si = float(max(sigma_i, np.finfo(float).eps))
    sj = float(max(sigma_j, np.finfo(float).eps))

    if equation == 1:
        rho = mu / np.hypot(mu, gam)
    elif equation == 2:
        term = 1.0 - (si * si / (sj * sj)) * (gam * gam)
        term = max(term, 0.0)
        rho = np.sign(mu) * np.sqrt(term)
    elif equation == 3:
        rho = mu * (sj / si)
    else:
        raise ValueError("equation must be 1, 2, or 3")

    return float(np.clip(rho, -1.0, 1.0))


# ---------------------------------------------------------------------
# Correlation estimation
# ---------------------------------------------------------------------

def estimate_correlation_by_ratios(
    X: Array,
    *,
    sigma: Array,
    equation: int = 1,
    min_denominator: float = 0.0,
    min_keep_frac: float = 0.90,
) -> Array:
    """
    Estimate correlation matrix R using the ratio trick and Eq. (5.12–5.14).

    For each pair (i,j):
        z_ij = X_i / X_j  (assuming centered data)
        Fit Cauchy(z_ij) → (μ_ij, γ_ij)
        Convert to ρ_ij via selected equation.

    Parameters
    ----------
    X : ndarray of shape (n, d)
        Centered observations.
    sigma : ndarray of shape (d,)
        Marginal Gaussian-equivalent scales.
    equation : {1, 2, 3}, default=1
        Which equation to use for ρ_ij.
    min_denominator : float, default=0.0
        Minimum |denominator| threshold to include a ratio.
    min_keep_frac : float, default=0.90
        Fraction of samples that must remain to fit a ratio.

    Returns
    -------
    R : ndarray of shape (d, d)
        Correlation matrix.
    """
    X = _as_2d_samples(X)
    n, d = X.shape
    sigma = np.asarray(sigma, dtype=float).reshape(-1)
    R = np.eye(d, dtype=float)
    eps = np.finfo(float).eps

    def _mask_denom(den: Array) -> Array:
        thr = max(min_denominator, eps)
        return np.abs(den) > thr

    for i in range(d):
        num_i = X[:, i]
        for j in range(i + 1, d):
            den_j = X[:, j]
            mask = _mask_denom(den_j)
            if mask.sum() < max(5, int(min_keep_frac * n)):
                rho = 0.0
            else:
                z = num_i[mask] / den_j[mask]
                mu_ij, gam_ij = fit_cauchy_1d(z)
                rho = _rho_from_equation(mu_ij, gam_ij, sigma[i], sigma[j], equation)
            R[i, j] = R[j, i] = float(np.clip(rho, -1.0, 1.0))

    # Symmetrize & normalize
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)
    return R


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def estimate_shape_method1(
    X: Array,
    *,
    alpha: float,
    h_lookup: Dict[float, float] | None = None,
    equation: int = 1,
    location_kwargs: Optional[Dict[str, Any]] = None,
    min_denominator: float = 0.0,
    min_keep_frac: float = 0.90,
) -> dict:
    """
    End-to-end Method 1 (ratio-based) estimation.

    Steps:
      1. Estimate marginal α-power scales P and Gaussian-equivalent σ.
      2. Estimate correlation matrix R via the ratio trick.
      3. Assemble shape Σ̂ = diag(P) @ R @ diag(P).

    Parameters
    ----------
    X : ndarray of shape (n, d)
        Data matrix.
    alpha : float
        Stability parameter.
    h_lookup : dict, optional
        Pre-computed entropies for α ≠ 1.
    equation : {1, 2, 3}, default=1
        Which equation for ρ_ij computation.
    location_kwargs : dict, optional
        Additional kwargs for location estimation.
    min_denominator : float, default=0.0
        Minimum denominator for ratio computation.
    min_keep_frac : float, default=0.90
        Minimum fraction of samples to keep.

    Returns
    -------
    dict
        {
            "mu": robust locations (d,),
            "P": marginal α-power scales (d,),
            "sigma": Gaussian-equivalent scales (d,),
            "R": correlation matrix (d, d),
            "Sigma": shape matrix = diag(σ) @ R @ diag(σ)
        }
    """
    X = _as_2d_samples(X)

    # (1) Marginals
    mu, P, sigma = estimate_marginals_sigma(
        X, alpha_kernel=alpha, h_lookup=h_lookup, location_kwargs=location_kwargs
    )

    # (2) Center data by removing estimated locations
    Xc = X - mu[None, :]

    # (3) Correlations
    R = estimate_correlation_by_ratios(
        Xc,
        sigma=sigma,
        equation=equation,
        min_denominator=min_denominator,
        min_keep_frac=min_keep_frac,
    )

    # (4) Shape matrix
    D = np.diag(sigma)
    Sigma = D @ R @ D

    return {
        "mu": mu,
        "P": P,
        "sigma": sigma,
        "R": R,
        "Sigma": Sigma,
    }
