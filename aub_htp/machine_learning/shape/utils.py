import numpy as np
from aub_htp.statistics import alpha_location, alpha_power

def estimate_marginals(data: np.ndarray, alpha_kernel: float, alpha_data: float) -> np.ndarray:
    assert data.ndim == 2
    location = compute_feature_wise_location(data, alpha_kernel)
    power = compute_feature_wise_power(data - location, alpha_kernel)
    small_sigma = float(np.sqrt(2.0) / (alpha_data ** (1.0 / alpha_data))) * power
    return small_sigma 


def compute_feature_wise_location(data: np.ndarray, alpha: float) -> np.ndarray:
    assert data.ndim == 2
    _, n_features = data.shape
    location = np.empty(n_features, dtype=float)
    for feature in range(n_features):
        data_feature = data[:, feature]
        location[feature] = alpha_location(data_feature, alpha)
    return location

def compute_feature_wise_power(data: np.ndarray, alpha: float) -> np.ndarray:
    assert data.ndim == 2
    _, n_features = data.shape
    power = np.empty(n_features, dtype=float)
    for feature in range(n_features):
        data_feature = data[:, feature]
        power[feature] = alpha_power(data_feature, alpha)
    return power