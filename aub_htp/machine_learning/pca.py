import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import validate_data, check_is_fitted

from .covariance import HeavyTailedCovariance
from ._shape import MethodLiteral


class HeavyTailedPCA(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        n_components: int | None = None,
        alpha: float = 1.0,
        method: MethodLiteral = "method1",
    ):
        self.n_components = n_components
        self.alpha = alpha
        self.method = method


    def fit(self, X, y=None):
        X = validate_data(self, X)
        X = np.asarray(X, dtype=float)

        _, n_features = X.shape
        n_components = self._resolve_n_components(n_features)

        # Fit the shape matrix via the chosen method
        self.covariance_estimator_ = HeavyTailedCovariance(
            alpha=self.alpha,
            method=self.method,
        ).fit(X)

        # Eigendecomposition of the shape matrix
        # eigh guarantees real eigenvalues and sorted ascending for symmetric matrices
        eigenvalues, eigenvectors = np.linalg.eigh(
            self.covariance_estimator_.covariance_
        )

        # Sort descending: largest eigenvalue = most important direction
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        self.components_ = eigenvectors[:, :n_components].T   # (n_components, n_features)
        self.explained_variance_ = eigenvalues[:n_components]

        total_variance = float(eigenvalues.sum())
        if total_variance > 0:
            self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        else:
            self.explained_variance_ratio_ = np.zeros(n_components)

        return self


    def transform(self, X):
        check_is_fitted(self, attributes=["components_", "covariance_estimator_"])
        X = validate_data(self, X, reset=False)
        X = np.asarray(X, dtype=float)

        X_centered = X - self.covariance_estimator_.location_
        return X_centered @ self.components_.T

    def _resolve_n_components(self, n_features: int) -> int:
        if self.n_components is None:
            return n_features
        if not isinstance(self.n_components, int) or self.n_components < 1:
            raise ValueError(
                f"n_components must be a positive integer or None, "
                f"got {self.n_components!r}."
            )
        if self.n_components > n_features:
            raise ValueError(
                f"n_components ({self.n_components}) cannot exceed "
                f"n_features ({n_features})."
            )
        return self.n_components
