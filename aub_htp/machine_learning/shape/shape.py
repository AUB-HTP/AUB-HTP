from typing import get_args, Literal
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import validate_data

from .utils import compute_feature_wise_location

MethodLiteral = Literal["ratio", "log-difference"]

from .ratio import estimate_shape_ratio
from .log_difference import estimate_shape_log_difference


class AlphaStableShape(BaseEstimator):
    def __init__(self, alpha: float = 1.0, method: MethodLiteral = "ratio"):
        self.alpha = alpha
        self.method = method

    def fit(self, X, y=None):
        X = validate_data(self, X)
        X = np.asarray(X, dtype=float)

        alpha = self.alpha
        if alpha is None or not (0 < alpha <= 2):
            raise ValueError(f"alpha must be in (0, 2], got {alpha!r}.")
        if self.method not in get_args(MethodLiteral):
            raise ValueError(
                f"method must be one of {get_args(MethodLiteral)}, got {self.method!r}."
            )

        if self.method == "ratio":
            shape_matrix = estimate_shape_ratio(X, alpha)
        else:
            shape_matrix = estimate_shape_log_difference(X, alpha)
        location = compute_feature_wise_location(X, alpha)

        self.shape_ = shape_matrix
        self.location_ = location
        return self
