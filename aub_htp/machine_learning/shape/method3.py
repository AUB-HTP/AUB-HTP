import numpy as np
from .utils import estimate_marginals

epsilon = np.finfo(float).tiny

def estimate_shape_method3(data: np.ndarray, alpha_kernel: float, alpha_data: float) -> np.ndarray:
    data = np.asarray(data)
    data_is_one_dimensional = data.ndim == 1
    if data_is_one_dimensional:
        data = data.reshape(-1, 1)
    n_samples, _n_features = data.shape

    small_sigma = estimate_marginals(data, alpha_kernel=alpha_kernel, alpha_data=alpha_data)
    A_hat = estimate_A_hat_by_lln(data, small_sigma)
    data_normalized = data / np.maximum(np.sqrt(A_hat)[:, None], epsilon)
    shape_matrix = data_normalized.T @ data_normalized / n_samples
    if data_is_one_dimensional:
        return shape_matrix.item()
    return shape_matrix


def estimate_A_hat_by_lln(data: np.ndarray, small_sigma: np.ndarray) -> float:
    """
    Â_i = (sum_j X_{ij}^2) / (t̂ * c), with t̂ := tr(Σ̂) = sum_j sigma_j^2.
    with correction c = 1
    """
    trace_sigma_squared = float(np.sum(small_sigma * small_sigma))
    squared_norm = np.sum(data * data, axis=1)
    c = 1 # (d - 1.0) / max(float(d), 1.0)
    return squared_norm / np.maximum(trace_sigma_squared * c, epsilon)