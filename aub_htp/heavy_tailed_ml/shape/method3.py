# shape/method3.py
"""
Method 3 (LLN demixing) shape estimation.
"""
from __future__ import annotations

from typing import Dict, Any, Optional, Literal
import numpy as np

from .utilities import estimate_marginals_sigma, symmetrize_clip

Array = np.ndarray
TraceCorrection = Literal["none", "d-1_over_d"]


def normalize_mean(X: Array, mu: Array | None = None) -> Array:
    """
    Placeholder: do NOT subtract location yet (location != mean; theory pending).
    """
    return np.asarray(X, dtype=float)


def _estimate_A_by_lln(
    X: Array,
    trace_Sigma_hat: float,
    *,
    d: int,
    trace_correction: TraceCorrection,
) -> Array:
    """
    Estimate row-wise mixing variable A via LLN (eq. 5.18).
    
    Â_i = (sum_j X_{ij}^2) / (t̂ * c), with t̂ := tr(Σ̂) = sum_j sigma_j^2.
    
    Parameters
    ----------
    X : ndarray of shape (n, d)
        Data matrix.
    trace_Sigma_hat : float
        Trace of estimated shape matrix.
    d : int
        Number of dimensions.
    trace_correction : str
        "none" for c=1, "d-1_over_d" for c=(d-1)/d.
    
    Returns
    -------
    A_hat : ndarray of shape (n,)
        Estimated mixing variable per sample.
    """
    X = np.asarray(X, dtype=float)
    s2 = np.sum(X * X, axis=1)
    tiny = np.finfo(float).tiny

    if trace_correction == "d-1_over_d":
        c = (d - 1.0) / max(float(d), 1.0)
    else:
        c = 1.0

    denom = max(float(trace_Sigma_hat) * c, tiny)
    A_hat = s2 / denom
    return np.maximum(A_hat, tiny)


def normalize_row_by_A(X: Array, A_hat: Array) -> Array:
    """
    Divide each row i by sqrt(Â_i).
    
    Parameters
    ----------
    X : ndarray of shape (n, d)
        Data matrix.
    A_hat : ndarray of shape (n,)
        Estimated mixing variable per sample.
    
    Returns
    -------
    X_normalized : ndarray of shape (n, d)
        Row-normalized data.
    """
    X = np.asarray(X, dtype=float)
    A_hat = np.asarray(A_hat, dtype=float)
    scale = np.sqrt(np.maximum(A_hat, np.finfo(float).tiny))
    return X / scale[:, None]


def estimate_shape_method3(
    X: Array,
    *,
    alpha_kernel: float = 1.0,
    h_lookup: Optional[Dict[float, float]] = None,
    location_kwargs: Optional[Dict[str, Any]] = None,
    trace_correction: TraceCorrection = "none",
) -> Dict[str, Any]:
    """
    Heavy-tailed PCA – Method 3 (Law of Large Numbers demixing).

    Pipeline
    --------
    1) Marginals (1-D): (mu_j, P_j, sigma_j) with sigma_j = √2·P_j.
    2) t̂ = tr(Σ̂) = Σ_j sigma_j^2.
    3) Â_i = (Σ_j X_{ij}^2) / (t̂ * c), with optional c = (d-1)/d.
    4) X_G = X / sqrt(Â_i).
    5) Σ̂ = (1/n) X_G^T X_G; R̂ = cor(Σ̂).
    
    Parameters
    ----------
    X : ndarray of shape (n, d)
        Data matrix.
    alpha_kernel : float, default=1.0
        Stability parameter for marginal estimation.
    h_lookup : dict, optional
        Entropy lookup for α ≠ 1.
    location_kwargs : dict, optional
        Additional kwargs for location estimation.
    trace_correction : str, default="none"
        "none" or "d-1_over_d" for bias correction.

    Returns
    -------
    dict
        Shape estimation results with diagnostics.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be (n, d).")
    n, d = X.shape
    if n < 2 or d < 1:
        raise ValueError("Not enough data.")

    # 1) Unified marginals
    mu, P, sigma = estimate_marginals_sigma(
        X, alpha_kernel=alpha_kernel, h_lookup=h_lookup, location_kwargs=location_kwargs
    )

    # 2) Trace estimate from marginals
    trace_Sigma_hat = float(np.sum(sigma * sigma))

    # 3) LLN A-hat (raw rows; optional legacy correction)
    Xm = normalize_mean(X, mu=None)
    A_hat = _estimate_A_by_lln(
        Xm, trace_Sigma_hat, d=d, trace_correction=trace_correction
    )

    # 4) Demix rows
    X_G = normalize_row_by_A(Xm, A_hat)

    # 5) Gaussian covariance + correlation
    Sigma = (X_G.T @ X_G) / float(n)
    std_g = np.sqrt(np.maximum(np.diag(Sigma), np.finfo(float).tiny))
    R = Sigma / (std_g[:, None] * std_g[None, :])
    R = symmetrize_clip(R, lo=-1.0, hi=1.0, diag=1.0)

    # Diagnostics
    corr_factor = (d - 1.0) / float(d) if trace_correction == "d-1_over_d" else 1.0
    diagnostics = {
        "n": int(n),
        "d": int(d),
        "alpha_kernel": float(alpha_kernel),
        "trace_Sigma_hat": float(trace_Sigma_hat),
        "trace_correction": str(trace_correction),
        "trace_correction_factor": float(corr_factor),
        "A_hat_stats": {
            "min": float(np.min(A_hat)),
            "max": float(np.max(A_hat)),
            "mean": float(np.mean(A_hat)),
            "median": float(np.median(A_hat)),
        },
        "mean_handler": "pass_through",
        "notes": "Σ̂ = (1/n) X_G^T X_G with X_G = X / sqrt(Â_i). No centering.",
    }

    return {
        "mu": mu,
        "P": P,
        "sigma": sigma,
        "A_hat": A_hat,
        "R": R,
        "Sigma": Sigma,
        "diagnostics": diagnostics,
    }
