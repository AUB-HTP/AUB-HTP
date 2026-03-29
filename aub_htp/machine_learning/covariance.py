from typing import get_args
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import validate_data

from ._shape import (
    MethodLiteral,
    estimate_shape_method1,
    estimate_shape_method2,
    estimate_shape_method3,
)

class HeavyTailedCovariance(BaseEstimator):
    def __init__(self, alpha: float = 1.0, alpha_kernel: float = None, alpha_data: float = None, method: MethodLiteral = "method1"):
        self.alpha = alpha
        self.alpha_kernel = alpha_kernel
        self.alpha_data = alpha_data
        self.method = method

    def fit(self, X, y=None):
        X = validate_data(self, X)
        X = np.asarray(X, dtype=float)

        alpha_kernel, alpha_data = self._resolve_alpha_kernel_and_alpha_data()


        if self.method == "method1":
            shape_matrix = estimate_shape_method1(X, alpha=alpha_kernel)
        elif self.method == "method2":
            shape_matrix = estimate_shape_method2(X, alpha_kernel=alpha_kernel, alpha_data=alpha_data)
        elif self.method == "method3":
            shape_matrix = estimate_shape_method3(X, alpha_kernel=alpha_kernel, alpha_data=alpha_data)
        else:
            raise ValueError(
                f"method must be one of {get_args(MethodLiteral)}, got {self.method!r}."
            )

        self.covariance_ = shape_matrix

        return self

    def score(self, X, y=None):
        # TODO: ask profs about a scoring mechanism
        raise NotImplementedError()

    def _resolve_alpha_kernel_and_alpha_data(self) -> tuple[float, float]:
        ak, ad, a = self.alpha_kernel, self.alpha_data, self.alpha
        if ak is not None and ad is not None:
            if not (0 < ak <= 2) or not (0 < ad <= 2):
                raise ValueError(f"alpha_kernel and alpha_data must be in (0, 2], got {ak!r}, {ad!r}.")
            return ak, ad
        if a is not None:
            if not (0 < a <= 2):
                raise ValueError(f"alpha must be in (0, 2], got {a!r}.")
            return a, a
        raise ValueError("Either alpha or both alpha_kernel and alpha_data must be provided.")