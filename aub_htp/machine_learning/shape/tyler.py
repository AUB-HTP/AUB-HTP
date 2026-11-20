import warnings
import numpy as np
from scipy.linalg import solve_triangular
from sklearn.exceptions import ConvergenceWarning


def estimate_correlation_tyler(
    data: np.ndarray,
    *,
    eps: float = 2.0 ** -20,
    iterations: int = 100,
) -> np.ndarray:
    """
    Estimate the correlation matrix R of centered data with Tyler's M-estimator.

    Each row is sqrt(A) G, so its direction u = x / ||x|| = G / ||G|| does not depend
    on A. Tyler's estimator (Tyler 1987) is the MLE of the shape matrix from these
    directions, up to a constant. Starting from S = I, it repeats

        S <- d / n * sum_k u_k u_k^T / (u_k^T S^-1 u_k),    S <- S / (tr(S) / d)

    until the largest change of an entry is below eps, or `iterations` is reached.
    Then R = D^(-1/2) S D^(-1/2) with D = diag(S), which is always PSD.

    Rows equal to 0 are dropped. It needs more rows than features.
    """
    _, n_features = data.shape

    # Divide each row by its largest entry first, so huge rows don't overflow the norm.
    largest_entry = np.max(np.abs(data), axis=1)
    data = data[largest_entry > 0] / largest_entry[largest_entry > 0, None]
    directions = data / np.linalg.norm(data, axis=1, keepdims=True)

    n_samples = directions.shape[0]
    if n_samples <= n_features:
        raise ValueError(
            "The tyler method needs more non-zero samples than features, "
            f"got {n_samples} samples and {n_features} features."
        )

    S = np.eye(n_features, dtype=float)
    for _ in range(iterations):
        # q_k = u_k^T S^-1 u_k = ||L^-1 u_k||^2, with S = L L^T
        L = np.linalg.cholesky(S)
        q = np.sum(solve_triangular(L, directions.T, lower=True) ** 2, axis=0)

        S_new = n_features / n_samples * directions.T @ (directions / q[:, None])
        S_new /= np.trace(S_new) / n_features

        change = np.max(np.abs(S_new - S))
        S = S_new
        if change < eps:
            break
    else:
        warnings.warn(
            f"The tyler method did not converge in {iterations} iterations.",
            ConvergenceWarning,
            stacklevel=3,
        )

    # D^(-1/2) S D^(-1/2), with D = diag(S)
    inverse_sqrt_diagonal = 1.0 / np.sqrt(np.diag(S))
    correlation_matrix = inverse_sqrt_diagonal[:, None] * S * inverse_sqrt_diagonal[None, :]
    correlation_matrix = 0.5 * (correlation_matrix + correlation_matrix.T)
    np.fill_diagonal(correlation_matrix, 1.0)
    return correlation_matrix
