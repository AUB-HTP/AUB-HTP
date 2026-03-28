from typing import get_args
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import validate_data

from ._shape import (
    MethodLiteral,
    estimate_shape_method3,
)

class HeavyTailedCovariance(BaseEstimator):
    def __init__(self, alpha: float = 1.0, method: MethodLiteral = "method1"):
        self.alpha = alpha
        self.method = method

    def fit(self, X, y=None):
        X = validate_data(self, X)
        X = np.asarray(X, dtype=float)

        alpha = float(self.alpha)
        if not (0 < alpha <= 2):
            raise ValueError(
                f"alpha must be in (0, 2], got {self.alpha!r}."
            )

        if self.method == "method1":
            result = estimate_shape_method1(X, alpha=alpha)
        elif self.method == "method2":
            result = estimate_shape_method2(X, alpha=alpha)
        elif self.method == "method3":
            result = estimate_shape_method3(X, alpha=alpha)
        else:
            raise ValueError(
                f"method must be one of {get_args(MethodLiteral)}, got {self.method!r}."
            )

        self.location_ = result["location"]
        self.scale_ = result["scale"]
        self.correlation_ = result["correlation"]
        self.covariance_ = result["shape"]

        return self

    def score(self, X, y=None):
        raise NotImplementedError()
