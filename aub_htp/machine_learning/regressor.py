import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import validate_data, check_is_fitted

from scipy.optimize import minimize

from ..statistics import alpha_power, alpha_location


def l_alpha_loss(y, y_pred, *, alpha: float, tol: float = 2.0 ** -20, max_iter: int = 100):
    return alpha_power(y - y_pred, alpha, tol=tol, max_iter=max_iter) ** 2


def r_alpha_score(y, y_pred, *, alpha: float) -> float:
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y.size == 0:
        raise ValueError(f"y must not be empty, got shape {y.shape}.")
    if y_pred.shape != y.shape:
        if y_pred.size != y.size:
            raise ValueError(f"y and y_pred must have the same size, got shapes {y.shape} and {y_pred.shape}.")
        # (n,) against (n, 1): the same values, which would broadcast to (n, n).
        y_pred = y_pred.reshape(y.shape)
    if np.all(y == y[0]):
        # A constant y has no power around its location, so the ratio would divide by 0.
        # As in scikit-learn's r2_score: 1 if the prediction is exact, 0 otherwise.
        return 1.0 if np.array_equal(y, y_pred) else 0.0
    y_location = np.broadcast_to(alpha_location(y, alpha), y.shape)
    return (1 - l_alpha_loss(y, y_pred, alpha=alpha)
              / l_alpha_loss(y, y_location, alpha=alpha))


class AlphaStableLinearRegressor(RegressorMixin, BaseEstimator):
    r"""
    Scikit-learn compatible Alpha-stable linear regression.

    Given :math:`\textbf{x}` and :math:`\textbf{y}` as training data, where :math:`\textbf{x}` 
    is a matrix of shape (n_samples, n_features) and :math:`\textbf{y}` is a matrix 
    of shape (n_samples, n_targets), the objective is to find the weights 
    :math:`\textbf{w}` and bias :math:`b` that minimizes the loss function:

    .. math::
        \mathrm{arg\,min}_{\mathbf{w}, b} P_\alpha(y - (\mathbf{x}\mathbf{w}^T + b))^\alpha
    """

    def __init__(
        self,
        alpha: float = 1.0,
        *,
        max_iter: int = 5000,
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

        # The weights are wanted to within tol. Near the minimum, moving the weights by tol
        # changes the loss by only about tol^2, so the power is computed to tol^2.
        power_tol = max(float(self.tol) ** 2, 4 * np.finfo(float).eps)

        def objective(weights: np.ndarray) -> float:
            weights = np.asarray(weights, dtype=float)
            w = weights[:n_features * n_targets].reshape(n_targets, n_features)
            b = weights[n_features * n_targets:].reshape(n_targets)
            y_pred = X @ w.T + b
            return l_alpha_loss(y, y_pred, alpha = alpha, tol = power_tol, max_iter = 100)

        weights0 = np.random.randn(n_features * n_targets + n_targets)
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

        self.coef_ = weights[:n_features * n_targets].reshape(n_targets, n_features)
        self.intercept_ = weights[n_features * n_targets:].reshape(n_targets)

        return self
        

    def predict(self, X):
        check_is_fitted(self, attributes=["coef_", "intercept_"])
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X = validate_data(self, X, reset=False)
        X = np.asarray(X, dtype=float)
        y_pred = X @ self.coef_.T + self.intercept_
        if self._y_is_one_dimensional:
            y_pred = y_pred.ravel()
        return y_pred


    def _validate_and_reshape(self, X, y):
        X, y = validate_data(self, X, y, y_numeric=True, multi_output=True)

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


    #TODO: figure out what sample_weight is and how to use it
    def score(self, X, y, sample_weight=None):
        return r_alpha_score(y, self.predict(X), alpha=self.alpha)