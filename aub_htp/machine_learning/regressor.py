import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import validate_data, check_is_fitted

from scipy.optimize import minimize

from ..statistics import alpha_power

def heavy_tailed_loss(y, y_pred, *, alpha: float):
    return alpha_power(y - y_pred, alpha)

#TODO: test everything in the class
class AlphaStableLinearRegressor(RegressorMixin, BaseEstimator):
    """
    Alpha-stable linear regression:
        min_{w,b} alpha_power(y - (x @ w + b))
    """

    def __init__(
        self,
        alpha: float = 1.0,
        *,
        max_iter: int = 2000,
        tol: float = 1e-6,
        optimizer: str = "Powell",
    ):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.optimizer = optimizer


    def fit(self, X, y):
        X, y, alpha, y_is_one_dimensional = self._validate_and_reshape(X, y)
        self._y_is_one_dimensional = y_is_one_dimensional

        _, n_features = X.shape
        _, n_targets = y.shape

        def objective(weights: np.ndarray) -> float:
            weights = np.asarray(weights, dtype=float)
            w = weights[:n_features * n_targets].reshape(n_features, n_targets)
            b = weights[n_features * n_targets:].reshape(1, n_targets)
            y_pred = X @ w + b
            return float(alpha_power(y - y_pred, alpha))

        weights0 = np.random.randn(size=(n_features * n_targets + n_targets), dtype=float)
        res = minimize(
            objective,
            x0=weights0,
            method=self.optimizer,
            options={
                "maxiter": int(self.max_iter),
                "xtol": float(self.tol),
                "ftol": float(self.tol),
            },
        )
        weights = np.asarray(res.x, dtype=float)

        self.coef_ = weights[:n_features * n_targets].reshape(n_features, n_targets)
        self.intercept_ = weights[n_features * n_targets:].reshape(1, n_targets)

        return self
        
    def predict(self, X):
        check_is_fitted(self, attributes=["coef_", "intercept_"])
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X = validate_data(self, X, reset=False)
        X = np.asarray(X, dtype=float)
        y_pred = X @ self.coef_ + self.intercept_
        if self._y_is_one_dimensional:
            y_pred = y_pred.ravel()
        return y_pred


    def _validate_and_reshape(self, X, y):
        X, y = validate_data(self, X, y, y_numeric=True)

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        y_is_one_dimensional = y.ndim == 1
        if y_is_one_dimensional:
            y = y.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError(f"Expected X to be 1D or 2D; got {X.ndim = }")

        if y.ndim != 2:
            raise ValueError(f"Expected y to be 1D or 2D; got {y.ndim = }")


        alpha = float(self.alpha)
        if not (0 < alpha <= 2):
            raise ValueError(f"{self.alpha = } must be in (0, 2]")

        return X, y, alpha, y_is_one_dimensional
