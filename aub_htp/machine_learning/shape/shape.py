from typing import get_args, Literal
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import validate_data

from aub_htp.statistics import alpha_location, alpha_power

MethodLiteral = Literal["ratio", "log-difference"]

from .ratio import estimate_correlation_ratio
from .log_difference import estimate_correlation_log_difference
from .utils import repair_correlation_matrix


class AlphaStableShape(BaseEstimator):
    def __init__(self, alpha: float = 1.0, method: MethodLiteral = "ratio"):
        self.alpha = alpha
        self.method = method

    def fit(self, X, y=None):
        X = validate_data(self, X, ensure_min_samples=2)
        X = np.asarray(X, dtype=float)

        alpha = self.alpha
        if alpha is None or not (0 < alpha <= 2):
            raise ValueError(f"alpha must be in (0, 2], got {alpha!r}.")
        if self.method not in get_args(MethodLiteral):
            raise ValueError(
                f"method must be one of {get_args(MethodLiteral)}, got {self.method!r}."
            )

        location, small_sigma = self._fit_marginals(X, alpha)
        X_centered = X - location

        if self.method == "ratio":
            correlation_matrix = estimate_correlation_ratio(X_centered)
        else:
            correlation_matrix = estimate_correlation_log_difference(X_centered)
        # Both methods estimate each rho_ij on its own, so R may not be PSD.
        correlation_matrix = repair_correlation_matrix(correlation_matrix)

        # diag(sigma) R diag(sigma)
        self.shape_ = small_sigma[:, None] * correlation_matrix * small_sigma[None, :]
        self.location_ = location
        return self

    @staticmethod
    def _fit_marginals(X: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Location and sigma_i = sqrt(shape_ii) of each feature.

        Each feature is symmetric alpha-stable with scale gamma_i = sigma_i / sqrt(2),
        and its alpha-power is alpha^(1/alpha) gamma_i, so

            sigma_i = sqrt(2) / alpha^(1/alpha) * power_i
        """
        _, n_features = X.shape
        power_to_sigma = float(np.sqrt(2.0) / alpha ** (1.0 / alpha))

        location = np.empty(n_features, dtype=float)
        small_sigma = np.empty(n_features, dtype=float)
        for feature in range(n_features):
            x = X[:, feature]
            location[feature] = alpha_location(x, alpha)
            small_sigma[feature] = power_to_sigma * alpha_power(x - location[feature], alpha)
        return location, small_sigma
