import warnings
import numpy as np
from aub_htp.statistics import alpha_location, alpha_power


def fit_location_and_power(data: np.ndarray, alpha: float) -> tuple[float, float]:
    """
    alpha-location of 1-D data, and the alpha-power of the data centered at it.
    """
    power_tol, max_iter = 2.0 ** -40, 100
    location = float(alpha_location(data, alpha, tol=2.0 ** -20, power_tol=power_tol, max_iter=max_iter))
    power = alpha_power(data - location, alpha, tol=power_tol, max_iter=max_iter)
    return location, power


def compute_feature_wise_power(data: np.ndarray, alpha: float) -> np.ndarray:
    if data.ndim != 2:
        raise ValueError(f"data must be a 2D array, got {data.ndim}D")
    _, n_features = data.shape
    power = np.empty(n_features, dtype=float)
    for feature in range(n_features):
        data_feature = data[:, feature]
        power[feature] = alpha_power(data_feature, alpha)
    return power


def divide_by_sigma(data: np.ndarray, small_sigma: np.ndarray) -> np.ndarray:
    """Divide each feature by its sigma_i. A feature with sigma_i = 0 is all zeros, so it is left as it is."""
    small_sigma = np.asarray(small_sigma, dtype=float).reshape(-1)
    return data / np.where(small_sigma > 0, small_sigma, 1.0)


def warn_pairs_without_samples() -> None:
    """Warn that some rho_ij could not be estimated and were set to 0."""
    warnings.warn(
        "Some pairs of features have fewer than 2 samples where both are non-zero. "
        "Their correlation was set to 0.",
        UserWarning,
        stacklevel=4,  # the caller of fit
    )


def repair_correlation_matrix(correlation_matrix: np.ndarray, *, eps: float = 2.0 ** -20) -> np.ndarray:
    """
    Make a correlation matrix positive semi-definite.

    Pairwise estimates of rho_ij need not form a PSD matrix. If R = Q diag(lambda) Q^T
    has a negative eigenvalue, the negative ones are set to eps, R is rebuilt, and it
    is rescaled so its diagonal is 1 again. A warning is raised when this happens.

    Raises ValueError if R has NaN or infinite entries.
    """
    if not np.all(np.isfinite(correlation_matrix)):
        raise ValueError("The estimated correlation matrix has NaN or infinite entries.")

    n_features = correlation_matrix.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
    # The error of eigh grows with the dimension d and the largest eigenvalue, which is at most d here.
    if eigenvalues.min() >= -n_features * np.finfo(float).eps:
        return correlation_matrix

    warnings.warn(
        "The estimated correlation matrix is not positive semi-definite "
        f"(smallest eigenvalue {eigenvalues.min():.3g}). Its negative eigenvalues were clipped.",
        UserWarning,
        stacklevel=3,
    )
    eigenvalues = np.where(eigenvalues < 0, eps, eigenvalues)
    repaired = (eigenvectors * eigenvalues) @ eigenvectors.T
    # D^(-1/2) R D^(-1/2), with D = diag(R)
    inverse_sqrt_diagonal = 1.0 / np.sqrt(np.diag(repaired))
    repaired = inverse_sqrt_diagonal[:, None] * repaired * inverse_sqrt_diagonal[None, :]
    repaired = 0.5 * (repaired + repaired.T)
    np.fill_diagonal(repaired, 1.0)
    return repaired
