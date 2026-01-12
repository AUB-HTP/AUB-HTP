# regression.py
"""
Heavy-tailed linear regression.

Minimizes α-power of residuals instead of squared error,
providing robust regression for heavy-tailed data.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .power import P_alpha


def _as_2d_samples(A: np.ndarray) -> np.ndarray:
    """
    Enforce 'samples as rows':
      - (n,) -> (n, 1)
      - (n, d) -> (n, d)
    """
    A = np.asarray(A, dtype=float)
    return A[:, None] if A.ndim == 1 else A


def robust_init_nd(X: np.ndarray, Y: np.ndarray, rng_seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple robust initializer for regression parameters.
    
    Parameters
    ----------
    X : ndarray of shape (n, d_in)
        Input features.
    Y : ndarray of shape (n, d_out)
        Target values.
    rng_seed : int, default=0
        Random seed (unused, kept for API compatibility).
    
    Returns
    -------
    A0 : ndarray of shape (d_out, d_in)
        Initial coefficient matrix (zeros).
    B0 : ndarray of shape (d_out,)
        Initial intercept (median of Y per dimension).
    """
    X = _as_2d_samples(X)
    Y = _as_2d_samples(Y)
    n, d_in = X.shape
    n2, d_out = Y.shape
    if n != n2:
        raise ValueError(f"X and Y must have same number of samples; got {n} vs {n2}.")
    A0 = np.zeros((d_out, d_in), dtype=float)
    B0 = np.median(Y, axis=0).astype(float)
    return A0, B0


def ht_linear_regression_fit_nd(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    alpha: float = 1.0,
    h_lookup: dict | None = None,
    theta0: tuple[np.ndarray, np.ndarray] | None = None,
    P0: float = 1.0,
    optimizer: str = "Powell",
    maxiter: int = 3000,
) -> dict:
    """
    Heavy-tailed linear regression for d-D inputs and outputs.
    
    Minimizes:
        P_α(Y - (X @ A.T + B))
    
    Parameters
    ----------
    X : ndarray of shape (n, d_in)
        Input features.
    Y : ndarray of shape (n, d_out)
        Target values.
    alpha : float, default=1.0
        Stability parameter. α=1 is Cauchy (no h_lookup needed).
    h_lookup : dict, optional
        Pre-computed entropies. Required for α ≠ 1.
    theta0 : tuple of (A0, B0), optional
        Initial parameters. Defaults to robust initialization.
    P0 : float, default=1.0
        Warm-start for inner α-power solver.
    optimizer : str, default="Powell"
        Scipy optimizer method.
    maxiter : int, default=3000
        Maximum iterations.
    
    Returns
    -------
    dict
        {
            "A": coefficient matrix (d_out, d_in),
            "B": intercept vector (d_out,),
            "P_star": final α-power,
            "status": optimization status dict,
            "objective_calls": number of objective evaluations
        }
    """
    X = _as_2d_samples(X)
    Y = _as_2d_samples(Y)
    n, d_in = X.shape
    n2, d_out = Y.shape
    if n != n2:
        raise ValueError(f"X and Y must have the same number of samples; got {n} vs {n2}.")

    # Initialize parameters
    if theta0 is None:
        A0, B0 = robust_init_nd(X, Y)
    else:
        A0, B0 = theta0
        A0 = np.asarray(A0, dtype=float).reshape(d_out, d_in)
        B0 = np.asarray(B0, dtype=float).ravel()
        if B0.shape != (d_out,):
            raise ValueError(f"B0 must have shape {(d_out,)}, got {B0.shape}.")

    # Flatten θ = [vec(A), B]
    theta0_vec = np.concatenate([A0.ravel(), B0])

    state = {"P0": float(P0), "calls": 0}

    def objective(theta_vec: np.ndarray) -> float:
        A = theta_vec[: d_out * d_in].reshape(d_out, d_in)
        B = theta_vec[d_out * d_in:]
        R = Y - (X @ A.T + B)
        state["calls"] += 1
        P_star = P_alpha(R, alpha=alpha, h_lookup=h_lookup, P0=state["P0"])
        state["P0"] = float(max(P_star, 1e-12))
        return float(P_star)

    res = minimize(
        fun=objective,
        x0=theta0_vec,
        method=optimizer,
        options=dict(maxiter=int(maxiter)),
    )

    A_star = res.x[: d_out * d_in].reshape(d_out, d_in)
    B_star = res.x[d_out * d_in:]
    P_star = float(res.fun)

    return {
        "A": A_star,
        "B": B_star,
        "P_star": P_star,
        "status": {
            "success": bool(res.success),
            "message": res.message,
            "nfev": getattr(res, "nfev", None),
            "nit": getattr(res, "nit", None),
        },
        "objective_calls": state["calls"],
    }


def ht_predict_nd(X: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Predict Ŷ = X @ A.T + B.
    
    Parameters
    ----------
    X : ndarray of shape (n,) or (n, d_in)
        Input features.
    A : ndarray of shape (d_out, d_in)
        Coefficient matrix.
    B : ndarray of shape (d_out,)
        Intercept vector.
    
    Returns
    -------
    ndarray of shape (n, d_out)
        Predictions.
    """
    X = _as_2d_samples(X)
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float).ravel()
    return X @ A.T + B


class HeavyTailedLinearRegressionND:
    """
    sklearn-style multi-output heavy-tailed linear regression.
    
    Parameters
    ----------
    alpha : float, default=1.0
        Stability parameter.
    h_lookup : dict, optional
        Pre-computed entropies for α ≠ 1.
    optimizer : str, default="Powell"
        Scipy optimizer method.
    maxiter : int, default=3000
        Maximum iterations.
    P0 : float, default=1.0
        Initial α-power guess.
    
    Attributes
    ----------
    A_ : ndarray
        Fitted coefficient matrix.
    B_ : ndarray
        Fitted intercept.
    P_star_ : float
        Final α-power.
    
    Examples
    --------
    >>> reg = HeavyTailedLinearRegressionND(alpha=1.5, h_lookup=lookup)
    >>> reg.fit(X, Y)
    >>> Y_pred = reg.predict(X_new)
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        h_lookup: dict | None = None,
        optimizer: str = "Powell",
        maxiter: int = 3000,
        P0: float = 1.0,
    ):
        self.alpha = float(alpha)
        self.h_lookup = h_lookup
        self.optimizer = optimizer
        self.maxiter = int(maxiter)
        self.P0 = float(P0)

        self.A_: np.ndarray | None = None
        self.B_: np.ndarray | None = None
        self.P_star_: float | None = None
        self.status_: dict | None = None
        self.objective_calls_: int | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray, theta0: tuple[np.ndarray, np.ndarray] | None = None):
        """Fit the model."""
        res = ht_linear_regression_fit_nd(
            X, Y,
            alpha=self.alpha,
            h_lookup=self.h_lookup,
            theta0=theta0,
            P0=self.P0,
            optimizer=self.optimizer,
            maxiter=self.maxiter,
        )
        self.A_ = res["A"]
        self.B_ = res["B"]
        self.P_star_ = res["P_star"]
        self.status_ = res["status"]
        self.objective_calls_ = res["objective_calls"]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the fitted model."""
        if self.A_ is None or self.B_ is None:
            raise RuntimeError("Call fit() before predict().")
        return ht_predict_nd(X, self.A_, self.B_)
