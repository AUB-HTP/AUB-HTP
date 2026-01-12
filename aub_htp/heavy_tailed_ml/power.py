# power.py
"""
α-power computation for heavy-tailed distributions.

The α-power P_α is a robust scale measure that replaces variance for 
heavy-tailed data where variance may be infinite.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import fsolve, brentq
from scipy import special

from ..alpha_stable_pdf import generate_alpha_stable_pdf


# =========================
# Entropy helper (1-D table)
# =========================

def h_Z_tilde_1d(alpha: float, gamma_ref: float, h_lookup: dict[float, float]) -> float:
    """Compute entropy for 1-D α-stable with given reference scale."""
    return float(h_lookup[alpha]) + np.log(gamma_ref)


# =========================
# Core kernels
# =========================

def _neglogpdf_alpha_1d(z: np.ndarray, alpha: float, gamma_ref: float) -> np.ndarray:
    """
    Per-coordinate 1-D kernel using logpdf (avoids underflow and preserves scaling).
    Vectorized over z with any shape.
    """
    # Use alpha_stable_pdf: generate_alpha_stable_pdf(X, alpha, beta, gamma, delta)
    # For symmetric stable (beta=0), centered at 0 (delta=0), with scale gamma_ref
    pdf_vals = generate_alpha_stable_pdf(z, alpha, beta=0, gamma=gamma_ref, delta=0)
    # Clamp to avoid log(0) for numerical stability
    pdf_vals = np.maximum(pdf_vals, np.finfo(float).tiny)
    return -np.log(pdf_vals) 


def _cauchy_isotropic_constant(d: int) -> float:
    """Compute the isotropic Cauchy constant for d dimensions."""
    return float(special.digamma((d + 1) / 2) + np.log(4.0) - special.digamma(1.0))


def _iid(
    residuals: ArrayLike,
    *,
    alpha: float,
    h_lookup: dict[float, float] | None = None,
    P0: float = 1.0,
    solver: str = "brentq",
    P_min: float = 1e-8,
    P_max: float = 1e8,
) -> float:
    """
    Compute α-power for iid model (each dimension treated independently).
    """
    if not (0 < alpha <= 2):
        raise ValueError("alpha must be in (0, 2].")
    
    R = np.asarray(residuals, dtype=float)

    if alpha == 1.0:
        def g(P: float) -> float:
            Z = R / P
            return np.mean(np.log(1.0 + Z**2)) - np.log(4.0)  #TODO verify Z**2
        return float(_bracket_and_solve(g, P0=P0, solver=solver, P_min=P_min, P_max=P_max))
    else:
        if h_lookup is None:
            raise ValueError("h_lookup (unit-scale 1-D entropies) is required for α≠1 with model='iid'.")
        gamma_ref = (1.0 / alpha) ** (1.0 / alpha)
        h_ref = h_Z_tilde_1d(alpha, gamma_ref, h_lookup)
        def f(P: float) -> float:
            Z = R / P
            k = _neglogpdf_alpha_1d(Z, alpha, gamma_ref)
            return (np.mean(k) - h_ref)
        return float(_bracket_and_solve(f, P0=P0, solver=solver, P_min=P_min, P_max=P_max))


# =========================
# Unified α-power
# =========================

def P_alpha(
    residuals: ArrayLike,
    *,
    alpha: float,
    h_lookup: dict[float, float] | None = None,
    P0: float = 1.0,
    solver: str = "brentq",
    model: str = "multivariate",
    P_min: float = 1e-8,
    P_max: float = 1e8,
) -> float:
    """
    Compute the α-power of residuals.
    
    The α-power is a robust scale measure that generalizes variance to 
    heavy-tailed distributions where variance may be infinite.
    
    Parameters
    ----------
    residuals : array-like
        Data residuals. Shape (n,) for 1-D or (n, d) for d-D.
    alpha : float
        Stability parameter in (0, 2]. α=1 is Cauchy, α=2 is Gaussian.
    h_lookup : dict, optional
        Pre-computed entropies keyed by α. Required for α ≠ 1.
    P0 : float, default=1.0
        Initial guess for the solver.
    solver : str, default="brentq"
        Root-finding method.
    model : str, default="multivariate"
        For d-D data: "iid" treats dimensions independently,
        "multivariate" uses Euclidean norms.
    P_min, P_max : float
        Bracket bounds for the solver.
    
    Returns
    -------
    float
        The α-power (robust scale measure).
    """
    if not (0 < alpha <= 2):
        raise ValueError("alpha must be in (0, 2].")

    R = np.asarray(residuals, dtype=float)

    # ---------- 1-D ----------
    if R.ndim == 1:
        if alpha == 1.0:
            h_cauchy = np.log(4.0)
            def g(P: float) -> float:
                z = R / P
                return np.mean(np.log(1.0 + z**2)) - h_cauchy
            return float(_bracket_and_solve(g, P0=P0, solver=solver, P_min=P_min, P_max=P_max))
        else:
            if h_lookup is None:
                raise ValueError("h_lookup (unit-scale 1-D entropies) is required for α≠1.")
            gamma_ref = (1.0 / alpha) ** (1.0 / alpha)
            h_ref = h_Z_tilde_1d(alpha, gamma_ref, h_lookup)
            def f(P: float) -> float:
                z = R / P
                return (np.mean(_neglogpdf_alpha_1d(z, alpha, gamma_ref)) - h_ref)
            return float(_bracket_and_solve(f, P0=P0, solver=solver, P_min=P_min, P_max=P_max))

    # ---------- d-D ----------
    if R.ndim != 2:
        raise ValueError("residuals must be 1-D (n,) or 2-D (n,d).")
    n, d = R.shape
    if d == 1:
        return P_alpha(R.ravel(), alpha=alpha, h_lookup=h_lookup, P0=P0, solver=solver, model=model, P_min=P_min, P_max=P_max)

    if model == "iid":
        return _iid(
            residuals=residuals,
            alpha=alpha,
            h_lookup=h_lookup,
            P0=P0,
            solver=solver,
            P_min=P_min,
            P_max=P_max,
        )

    elif model == "multivariate":
        if alpha != 1.0:
            raise NotImplementedError("α≠1 not implemented (no simple multivariate SαS pdf).")
        C_d = _cauchy_isotropic_constant(d)
        norms = np.linalg.norm(R, axis=1)
        def g(P: float) -> float:
            z = norms / P
            return np.mean(np.log(1.0 + z**2)) - C_d
        return float(_bracket_and_solve(g, P0=P0, solver=solver, P_min=P_min, P_max=P_max))

    else:
        raise ValueError("model must be 'iid' or 'multivariate'.")


# =========================
# Internal robust solvers
# =========================

def _bracket_and_solve(
    fun,
    *,
    P0: float,
    solver: str,
    P_min: float,
    P_max: float,
) -> float:
    """
    Solve fun(P) = 0 for P > 0 robustly using a known bracket.
    If no sign change is detected, fall back to fsolve in log-space.
    """
    # Step 1: Try using P_min and P_max directly as the bracket for brentq
    a, b = P_min, P_max
    fa, fb = fun(a), fun(b)

    # If we have a valid sign change, use brentq
    if np.sign(fa) != np.sign(fb):
        return brentq(lambda p: fun(float(p)), a=a, b=b)

    # Step 2: Fall back to fsolve (log-space)
    def log_fun(s: float) -> float:
        return fun(np.exp(s))

    s0 = np.log(P0)
    s_sol = fsolve(log_fun, s0)[0]
    return np.exp(s_sol)
