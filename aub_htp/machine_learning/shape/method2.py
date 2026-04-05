import numpy as np
from scipy.integrate import quad
from functools import lru_cache
from pathlib import Path
from scipy.interpolate import interp1d
import aub_htp as ht

from .utils import estimate_marginals, compute_feature_wise_location, compute_feature_wise_power

# Gaussian constant:  m_Z = E[log|Z|] for Z ~ N(0,1)
E_LOG_ABS_Z: float = - (np.euler_gamma + np.log(2.0)) / 2.0


def estimate_shape_method2(
    data: np.ndarray, 
    alpha_kernel: float, 
    alpha_data: float,
    *,
    log_eps: float = 0.0,
) -> np.ndarray:
    data = np.asarray(data)
    data_is_one_dimensional = data.ndim == 1
    if data_is_one_dimensional:
        data = data.reshape(-1, 1)
    n_samples, n_features = data.shape

    small_sigma = estimate_marginals(data, alpha_kernel=alpha_kernel, alpha_data=alpha_data)
    
    data_centered = data - compute_feature_wise_location(data, alpha_kernel)
    data_standardized = ( data_centered
                        / np.maximum(small_sigma, np.finfo(float).tiny)[None, :])
    
    L = np.log(np.abs(data_standardized) + max(log_eps, 0))
    C_S = (L.T @ L) / float(n_samples)

    E_logA, E_logA2 = compute_logA_moments(alpha=alpha_data)
    c0 = 0.25 * E_logA2
    c1 = 0.5  * E_logA

    C_G = C_S - c0 - (2.0 * c1 * E_LOG_ABS_Z)

    coorolation_matrix = np.eye(n_features, dtype=float)
    iu = np.triu_indices(n_features, 1)
    F_off = C_G[iu]
    
    rho_vals = get_f_to_rho_interpolator()(F_off)
    coorolation_matrix[iu] = rho_vals
    coorolation_matrix[(iu[1], iu[0])] = rho_vals

    coorolation_matrix = np.clip(0.5 * (coorolation_matrix + coorolation_matrix.T), 0.0, 1.0)
    np.fill_diagonal(coorolation_matrix, 1.0)

    diagonal_matrix = np.diag(compute_feature_wise_power(data_centered, alpha_kernel))
    shape_matrix = diagonal_matrix @ coorolation_matrix @ diagonal_matrix
    if data_is_one_dimensional:
        return shape_matrix.item()
    return shape_matrix

@lru_cache(maxsize=None)
def get_f_to_rho_interpolator() -> interp1d:
    data = np.load(Path(__file__).parent / "data" / "lookup_table_rho.npz")
    xs = np.asarray(data["xs"])
    ys = np.asarray(data["ys"])
    return interp1d(xs, ys, kind="cubic", bounds_error=False, fill_value=0.0)
    

@lru_cache(maxsize=None)
def compute_logA_moments(alpha: float) -> tuple[float, float]:
    a2 = alpha / 2.0
    scale = float(np.cos(np.pi * alpha / 4.0) ** (2.0 / alpha))

    def integrand_log(x):
        return np.log(x) * ht.alpha_stable.pdf(x, a2, 1, loc=0.0, scale=scale)

    def integrand_log2(x):
        lx = np.log(x)
        return lx * lx * ht.alpha_stable.pdf(x, a2, 1, loc=0.0, scale=scale)

    E_logA,  _ = quad(integrand_log,  0.0, np.inf, limit=200)
    E_logA2, _ = quad(integrand_log2, 0.0, np.inf, limit=200)
    return float(E_logA), float(E_logA2)
