# shape/method2.py
"""
Method 2 (log-correlation) shape estimation.
"""
from __future__ import annotations

from typing import Dict, Any, Optional, Callable, Tuple
import numpy as np

try:
    from scipy.integrate import quad
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

from ...alpha_stable_pdf import generate_alpha_stable_pdf

from .utilities import estimate_marginals_sigma

Array = np.ndarray

# Gaussian constant: m_Z = E[log|Z|] for Z ~ N(0,1)
E_LOG_ABS_Z: float = -(np.euler_gamma + np.log(2.0)) / 2.0


# ---------------------------------------------------------------------
# A-log moments E[log A], E[(log A)²]
# ---------------------------------------------------------------------

def _stable_scale_for_mixture(alpha: float) -> float:
    """Compute scale for stable mixture variable."""
    return float(np.cos(np.pi * alpha / 4.0) ** (2.0 / alpha))


_LOGA_CACHE: Dict[float, Tuple[float, float]] = {}


def compute_logA_moments(alpha: float) -> Tuple[float, float]:
    """
    Compute E[log A] and E[(log A)²] for the stable mixing variable.
    
    Parameters
    ----------
    alpha : float
        Stability parameter.
    
    Returns
    -------
    E_logA : float
        Expected value of log A.
    E_logA2 : float
        Expected value of (log A)².
    """
    if alpha in _LOGA_CACHE:
        return _LOGA_CACHE[alpha]

    if not _HAVE_SCIPY:
        raise RuntimeError("Pass E_logA and E_logA2 explicitly, or install SciPy (for scipy.integrate.quad).")
    
    a2 = alpha / 2.0
    scale = _stable_scale_for_mixture(alpha)

    def integrand_log(x):
        # Use alpha_stable_pdf: generate_alpha_stable_pdf(X, alpha, beta, gamma, delta)
        # For fully skewed positive stable (beta=1), loc=0 (delta=0), with computed scale (gamma)
        pdf_val = generate_alpha_stable_pdf(np.atleast_1d(x), a2, beta=1, gamma=scale, delta=0.0)
        return np.log(x) * pdf_val[0]

    def integrand_log2(x):
        lx = np.log(x)
        pdf_val = generate_alpha_stable_pdf(np.atleast_1d(x), a2, beta=1, gamma=scale, delta=0.0)
        return lx * lx * pdf_val[0]

    E_logA, _ = quad(integrand_log, 0.0, np.inf, limit=200)
    E_logA2, _ = quad(integrand_log2, 0.0, np.inf, limit=200)
    _LOGA_CACHE[alpha] = (float(E_logA), float(E_logA2))
    return float(E_logA), float(E_logA2)


# ---------------------------------------------------------------------
# Inverse lookup f(ρ) → ρ
# ---------------------------------------------------------------------

def make_f_inverse(lookup_rho_to_f: Dict[float, float]) -> Tuple[Callable[[Array], Array], Tuple[float, float]]:
    """
    Create inverse lookup function from ρ → f(ρ) mapping.
    
    Parameters
    ----------
    lookup_rho_to_f : dict
        Mapping from ρ values to f(ρ) values.
    
    Returns
    -------
    f_to_rho : callable
        Function mapping f values back to ρ.
    bounds : tuple of (f_min, f_max)
        Valid range for f values.
    """
    items = sorted(((float(r), float(f)) for r, f in lookup_rho_to_f.items()),
                   key=lambda t: t[1])
    rhos = np.array([t[0] for t in items], dtype=float)
    fvals = np.array([t[1] for t in items], dtype=float)
    if fvals.size < 5:
        raise ValueError("lookup_rho_to_f must have enough samples.")

    f_min, f_max = fvals.min(), fvals.max()

    def f_to_rho(f_in: Array) -> Array:
        f_in = np.asarray(f_in, dtype=float)
        clipped = np.clip(f_in, f_min, f_max)
        return np.interp(clipped, fvals, rhos)

    return f_to_rho, (float(f_min), float(f_max))


# ---------------------------------------------------------------------
# Main: Method 2 — Log-Correlation
# ---------------------------------------------------------------------

def estimate_shape_method2(
    X: Array,
    *,
    alpha_data: float = 1.0,
    alpha_kernel: float = 1.0,
    lookup_rho_to_f: Dict[float, float],
    E_logA: Optional[float] = None,
    E_logA2: Optional[float] = None,
    h_lookup: Optional[Dict[float, float]] = None,
    location_kwargs: Optional[Dict[str, Any]] = None,
    log_eps: float = 0.0,
    mu_known: Optional[Array] = None,
) -> Dict[str, Any]:
    """
    Heavy-tailed shape estimation via the Log-Correlation method (Eq. 5.15–5.16).

    Returns Σ̂ = diag(P) R diag(P) with R = |ρ|.
    No sign heuristic is applied.
    
    Parameters
    ----------
    X : ndarray of shape (n, d)
        Data matrix.
    alpha_data : float, default=1.0
        Stability parameter of the data.
    alpha_kernel : float, default=1.0
        Stability parameter for kernel computations.
    lookup_rho_to_f : dict
        Mapping from ρ to f(ρ) for inverse lookup.
    E_logA, E_logA2 : float, optional
        Pre-computed log moments. Computed if not provided.
    h_lookup : dict, optional
        Entropy lookup for α ≠ 1.
    location_kwargs : dict, optional
        Additional kwargs for location estimation.
    log_eps : float, default=0.0
        Small value added before log to avoid log(0).
    mu_known : ndarray, optional
        Known location values.

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

    # 0) f→ρ inverse from lookup table
    f_to_rho, (f_min, f_max) = make_f_inverse(lookup_rho_to_f)

    # 1) Marginals: Pα and σ = √2 Pα
    mu, P, sigma = estimate_marginals_sigma(
        X, alpha_kernel=alpha_kernel, alpha_data=alpha_data,
        h_lookup=h_lookup, location_kwargs=location_kwargs, mu_known=mu_known
    )

    # 2.1) Subtract by mu
    X = X - mu

    # 2.2) Standardize by σ
    X_std = X / np.maximum(sigma, np.finfo(float).tiny)[None, :]

    # 3) Empirical log-moment matrix
    if log_eps > 0.0:
        L = np.log(np.abs(X_std) + log_eps)
    else:
        L = np.log(np.abs(X_std))
    C_S = (L.T @ L) / float(n)

    # 4) Mixing constants
    computed_constants = False
    if (E_logA is None) or (E_logA2 is None):
        E_logA, E_logA2 = compute_logA_moments(alpha=alpha_data)
        computed_constants = True

    c0 = 0.25 * E_logA2
    c1 = 0.5 * E_logA

    # 5) Subtract A-contribution → Gaussian log-corr C_G
    C_G = C_S - c0 - (2.0 * c1 * E_LOG_ABS_Z)

    # 6) Map C_G → |ρ| (off-diagonal only)
    R = np.eye(d, dtype=float)
    iu = np.triu_indices(d, 1)
    F_off = C_G[iu]

    pre_clip_below = int(np.sum(F_off < f_min))
    pre_clip_above = int(np.sum(F_off > f_max))
    
    clipped_below_vals = F_off[F_off < f_min]
    clipped_above_vals = F_off[F_off > f_max]

    clip_info = {
        "count_below": pre_clip_below,
        "count_above": pre_clip_above,
        "min_val": float(F_off.min()) if F_off.size else None,
        "max_val": float(F_off.max()) if F_off.size else None,
        "below_vals": clipped_below_vals.tolist()[:10],
        "above_vals": clipped_above_vals.tolist()[:10],
        "below_diff": (clipped_below_vals - f_min).tolist()[:10],
        "above_diff": (clipped_above_vals - f_max).tolist()[:10],
    }

    rho_vals = f_to_rho(F_off)
    R[iu] = rho_vals
    R[(iu[1], iu[0])] = rho_vals

    R = np.clip(0.5 * (R + R.T), 0.0, 1.0)
    np.fill_diagonal(R, 1.0)

    # 7) Assemble Σ̂
    D = np.diag(P)
    Sigma = D @ R @ D

    # Diagnostics
    total_entries = int(np.prod(X_std.shape))
    zero_count = int(np.sum(X_std == 0.0)) if log_eps > 0.0 else 0

    diagnostics = {
        "n": int(n),
        "d": int(d),
        "alpha_kernel": float(alpha_kernel),
        "alpha_data": float(alpha_data),
        "E_LOG_ABS_Z": float(E_LOG_ABS_Z),
        "E_logA": float(E_logA),
        "E_logA2": float(E_logA2),
        "constants_source": "computed" if computed_constants else "provided",
        "lookup_f_min": float(f_min),
        "lookup_f_max": float(f_max),
        "lookup_clip_below": pre_clip_below,
        "lookup_clip_above": pre_clip_above,
        "clip_info": clip_info,
        "log_eps": float(log_eps),
        "zero_fraction_after_std": zero_count / total_entries if total_entries > 0 else 0.0,
        "location_mode": "known" if mu_known is not None else "estimated",
        "mean_handler": "pass_through",
        "std_handler": "σ = f(α) Pα",
        "notes": "No sign resolution; R contains |ρ| only (Eq. 5.16).",
    }

    return {
        "mu": mu,
        "P": P,
        "sigma": sigma,
        "R": R,
        "Sigma": Sigma,
        "diagnostics": diagnostics,
    }
