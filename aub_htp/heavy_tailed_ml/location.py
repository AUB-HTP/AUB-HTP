# location.py
"""
Robust location estimation for heavy-tailed distributions.

The α-location L_X = argmin_μ P_α(X - μ) provides a robust center estimate
that works even when the mean doesn't exist (α ≤ 1).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .power import P_alpha


def location_L(
    X: np.ndarray,
    *,
    alpha: float,
    h_lookup: dict | None = None,
    model: str = "multivariate",
    mu0: np.ndarray | float | None = None,
    P0: float = 1.0,
    optimizer: str = "Powell",
    maxiter: int = 2000,
) -> tuple[np.ndarray | float, float]:
    """
    Unified location estimator L_X = argmin_μ P_α(X - μ) for 1-D and d-D.
    
    The α-location is a robust center estimate that generalizes the median
    to arbitrary stability indices α.
    
    Parameters
    ----------
    X : ndarray
        Data array. Shape (n,) for 1-D or (n, d) for d-D.
    alpha : float
        Stability parameter in (0, 2].
    h_lookup : dict, optional
        Pre-computed entropies keyed by α. Required for α ≠ 1.
    model : str, default="multivariate"
        For d-D: "iid" or "multivariate". Ignored for 1-D.
    mu0 : ndarray or float, optional
        Initial location guess. Defaults to median.
    P0 : float, default=1.0
        Initial α-power guess for warm-starting inner solver.
    optimizer : str, default="Powell"
        Scipy optimizer method.
    maxiter : int, default=2000
        Maximum iterations for optimizer.
    
    Returns
    -------
    mu_star : ndarray or float
        Optimal location estimate.
    P_star : float
        α-power at the optimal location.
    """
    A = np.asarray(X, float)
    
    if A.ndim == 1:
        # ---------- 1-D ----------
        x = A.ravel()
        if mu0 is None:
            mu0 = float(np.median(x))
        mu0 = float(mu0)

        def objective(mu_vec):
            mu = float(mu_vec[0])
            r = x - mu
            return P_alpha(r, alpha=alpha, h_lookup=h_lookup, P0=P0)

        res = minimize(objective, x0=np.array([mu0], float), method=optimizer, options=dict(maxiter=int(maxiter)))
        mu_star = float(res.x[0])
        P_star = float(res.fun)
        return mu_star, P_star

    # ---------- d-D ----------
    if A.ndim != 2:
        raise ValueError("X must be shape (n,) or (n,d)")

    n, d = A.shape
    if mu0 is None:
        mu0 = np.median(A, axis=0)
    mu0 = np.asarray(mu0, float).ravel()
    if mu0.shape != (d,):
        raise ValueError("mu0 must have shape (d,) for d-D data")

    state = {"P0": float(P0)}

    def objective(mu_vec: np.ndarray) -> float:
        mu = mu_vec.ravel()
        R = A - mu
        P_star = P_alpha(R, alpha=alpha, h_lookup=h_lookup, P0=state["P0"], model=model)
        state["P0"] = float(max(P_star, 1e-12))
        return float(P_star)

    res = minimize(objective, x0=mu0, method=optimizer, options=dict(maxiter=int(maxiter)))
    mu_star = np.array(res.x, float)
    P_star = float(res.fun)
    return mu_star, P_star
