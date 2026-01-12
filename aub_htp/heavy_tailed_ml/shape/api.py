# shape/api.py
"""
Public API for heavy-tailed shape estimation.
"""
from __future__ import annotations

from typing import Literal, Any, Dict
import numpy as np

ResultDict = Dict[str, Any]
MethodLiteral = Literal["method1", "method2", "method3"]


def estimate_shape(
    X: np.ndarray,
    method: MethodLiteral = "method1",
    alpha: float = 1.0,
    **kwargs: Any,
) -> ResultDict:
    """
    Public entrypoint to estimate the heavy-tailed 'shape' (robust covariance surrogate).

    This estimates a covariance-like matrix that can be used for PCA on heavy-tailed
    data where the standard covariance matrix fails.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data matrix. No NaNs/inf allowed.
    method : {"method1", "method2", "method3"}, default="method1"
        Which method to use:
        - "method1": Ratio-based correlation estimation (most robust)
        - "method2": Log-correlation method
        - "method3": LLN demixing method
    alpha : float, default=1.0
        Stability parameter (α=1 is Cauchy). Some methods may require
        auxiliary lookup (e.g., h_lookup) when alpha != 1.
    **kwargs : Any
        Method-specific parameters:
        
        For method1:
          - equation : int in {1, 2, 3}
          - h_lookup : dict (required for α ≠ 1)
          - location_kwargs : dict
          - min_denominator : float
          - min_keep_frac : float
        
        For method2:
          - lookup_rho_to_f : dict (required)
          - alpha_data : float
          - h_lookup : dict
          - E_logA, E_logA2 : float
          - log_eps : float
          - mu_known : ndarray
        
        For method3:
          - h_lookup : dict
          - location_kwargs : dict
          - trace_correction : "none" or "d-1_over_d"

    Returns
    -------
    dict
        {
          "mu": (n_features,) robust location vector,
          "P": (n_features,) marginal alpha-powers (scales),
          "R": (n_features, n_features) correlation-like matrix,
          "Sigma": (n_features, n_features) shape matrix = diag(P) @ R @ diag(P),
          "diagnostics": {...} (method-specific)
        }

    Notes
    -----
    - This is a *robust covariance surrogate*, not the classical covariance.
    - Overall scale of Sigma is typically irrelevant for PCA directions.

    Examples
    --------
    >>> from aub_htp import estimate_shape, load_entropy_lookup
    >>> h_lookup = load_entropy_lookup()
    >>> result = estimate_shape(X, method="method1", alpha=1.0)
    >>> eigvals, eigvecs = np.linalg.eigh(result["Sigma"])
    """
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or inf.")

    if method == "method1":
        from .method1 import estimate_shape_method1
        return estimate_shape_method1(X, alpha=alpha, **kwargs)

    elif method == "method2":
        from .method2 import estimate_shape_method2
        if "lookup_rho_to_f" not in kwargs:
            raise ValueError("Method 2 requires 'lookup_rho_to_f'.")
        return estimate_shape_method2(X, alpha_kernel=alpha, **kwargs)

    elif method == "method3":
        from .method3 import estimate_shape_method3
        return estimate_shape_method3(X, alpha_kernel=alpha, **kwargs)

    else:
        raise ValueError("Unknown method. Expected one of {'method1', 'method2', 'method3'}.")
