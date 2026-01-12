# shape/utilities.py
"""
Utility functions for shape estimation.
"""
from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import numpy as np

from ..location import location_L
from ..power import P_alpha

Array = np.ndarray


# =========================
# Basic helpers
# =========================

def as_2d(X: Array) -> Array:
    """(n,) -> (n,1); (n,d) stays. Float64 enforced."""
    X = np.asarray(X, dtype=float)
    return X[:, None] if X.ndim == 1 else X


def assert_finite(X: Array) -> None:
    """Raise ValueError if X contains NaN or inf."""
    if not np.isfinite(X).all():
        raise ValueError("Input contains NaN or inf.")


def symmetrize_clip(M: Array, lo: float = -1.0, hi: float = 1.0, diag: float | None = 1.0) -> Array:
    """Symmetrize matrix and clip values."""
    M = 0.5 * (M + M.T)
    M = np.clip(M, lo, hi)
    if diag is not None:
        np.fill_diagonal(M, diag)
    return M


# =========================================================
# α-scale factor
# =========================================================

def alpha_scale_factor(alpha_data: float, alpha_kernel: float) -> float:
    """
    Scale factor linking α-power (P) to Gaussian-equivalent σ:

        σ = f(α) · P ,  where f(α) = √2 / α^(1/α)

    This relation arises from:
        σ² = 2·γ²   and   P = α^(1/α)·γ  ⇒  σ = (√2 / α^(1/α))·P

    Parameters
    ----------
    alpha_data : float
        Stability parameter of the data distribution.
    alpha_kernel : float
        Stability parameter used by the kernel (in P_α and location_L).

    Returns
    -------
    scale : float
        Multiplicative factor converting P → σ.
    """
    if not np.isclose(alpha_data, alpha_kernel):
        alpha_kernel = alpha_data

    return float(np.sqrt(2.0) / (alpha_kernel ** (1.0 / alpha_kernel)))


def _compute_P_at_location(
    x: np.ndarray,
    mu_j: float,
    *,
    alpha_kernel: float,
    h_lookup: Optional[Dict[float, float]],
) -> float:
    """
    Compute marginal α-power Pα for a single column x_j around a known location mu_j.
    """
    r = x - mu_j
    P = P_alpha(r, alpha=alpha_kernel, h_lookup=h_lookup)
    return float(max(P, np.finfo(float).eps))


# =========================================================
# Marginals estimator
# =========================================================

def estimate_marginals_sigma(
    X: Array,
    *,
    alpha_kernel: float = 1.0,
    alpha_data: float | None = None,
    h_lookup: Optional[Dict[float, float]] = None,
    location_kwargs: Optional[Dict[str, Any]] = None,
    mu_known: Optional[Array] = None,
) -> Tuple[Array, Array, Array]:
    """
    Per-dimension α-power and Gaussian-equivalent σ.

    If mu_known is provided (scalar or (d,)), we compute Pα on (X - mu_known)
    and *do not* run the location solver for that column.
    Otherwise, we estimate (μ*, P) via location_L and compute σ.

    Parameters
    ----------
    X : ndarray of shape (n,) or (n, d)
        Data array.
    alpha_kernel : float, default=1.0
        Stability parameter for the kernel.
    alpha_data : float, optional
        Stability parameter of the data. Defaults to alpha_kernel.
    h_lookup : dict, optional
        Pre-computed entropies for α ≠ 1.
    location_kwargs : dict, optional
        Additional kwargs for location_L.
    mu_known : ndarray, optional
        Known location to use instead of estimating.

    Returns
    -------
    mu_used : ndarray of shape (d,)
        Location actually used (mu_known or estimated μ*).
    P : ndarray of shape (d,)
        Marginal α-power at mu_used.
    sigma : ndarray of shape (d,)
        Gaussian-equivalent stds via σ = (√2 / α^(1/α)) · P.
    """
    if alpha_data is None:
        alpha_data = alpha_kernel

    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise ValueError("X must be (n,) or (n,d).")

    n, d = X.shape
    mu_used = np.empty(d, dtype=float)
    P = np.empty(d, dtype=float)
    sigma = np.empty(d, dtype=float)
    lk = dict(location_kwargs or {})

    # Normalize mu_known to a vector
    if mu_known is None:
        mu_vec = None
    else:
        mk = np.asarray(mu_known, dtype=float)
        if mk.ndim == 0:
            mu_vec = np.full(d, float(mk))
        elif mk.ndim == 1 and mk.size == d:
            mu_vec = mk
        else:
            raise ValueError(f"mu_known must be a scalar or shape (d,), got {mk.shape}")

    for j in range(d):
        xj = X[:, j]
        if mu_vec is not None:
            mu_j = float(mu_vec[j])
            P[j] = _compute_P_at_location(xj, mu_j, alpha_kernel=alpha_kernel, h_lookup=h_lookup)
            mu_used[j] = mu_j
        else:
            mu_star, P_star = location_L(
                xj[:, None],
                alpha=alpha_kernel,
                h_lookup=h_lookup,
                **lk
            )
            mu_used[j] = float(np.squeeze(mu_star))
            P[j] = float(max(P_star, np.finfo(float).eps))

    scale = alpha_scale_factor(alpha_data=alpha_data, alpha_kernel=alpha_kernel)
    sigma[:] = scale * P

    return mu_used, P, sigma
