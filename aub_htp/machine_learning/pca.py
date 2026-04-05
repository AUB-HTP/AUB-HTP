import numpy as np
from .shape.utils import compute_feature_wise_location, compute_feature_wise_power
from .shape import AlphaStableShape, MethodLiteral
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import validate_data

def get_principal_components(X: np.ndarray, alpha: float, shape_estimation_method: MethodLiteral = "method1") -> np.ndarray:
    X = np.asarray(X, dtype=float)
    assert X.ndim == 2
    shape = AlphaStableShape(alpha=alpha, method=shape_estimation_method).fit(X).shape_
    eigenvalues, eigenvectors = np.linalg.eigh(shape)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    return shape, sorted_eigenvectors


class AlphaStablePCA(TransformerMixin, BaseEstimator):
    components_ : np.ndarray
    shape_: np.ndarray
    location_: np.ndarray
    power_: np.ndarray

    def __init__(self,
        n_components: int = None,
        alpha: float = 1.0,
        *,
        shape_estimation_method: MethodLiteral = "method1",
    ):
        self.n_components = n_components
        self.alpha = alpha
        self.shape_estimation_method = shape_estimation_method

    def fit(self, X, y=None):
        X = self._validate_data(X)
        shape, eigenvectors = get_principal_components(X, self.alpha, self.shape_estimation_method)

        self.location_ = compute_feature_wise_location(X, self.alpha)
        self.power_ = compute_feature_wise_power(X - self.location_, self.alpha)
        self.shape_ = shape
        self.components_ = eigenvectors.T[:self.n_components, :]
        return self

    def transform(self, X):
        X = self._validate_data(X)
        return X @ self.components_.T

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
        
    def score(self, X, y=None):
        raise NotImplementedError("AlphaStablePCA is not implemented yet.")
    
    def _validate_data(self, X):
        X = validate_data(self, X)
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X
