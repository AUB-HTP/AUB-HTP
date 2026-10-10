import numpy as np
from aub_htp.statistics import alpha_location, alpha_power
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import validate_data, check_is_fitted


def initialize_cluster_centers(X: np.ndarray, n_clusters: int) -> np.ndarray:
    percentiles = 100.0 * (2 * np.arange(n_clusters) + 1) / (2 * n_clusters)
    cluster_centers = np.percentile(X, percentiles, axis=0)
    return cluster_centers

def compute_inertia(X: np.ndarray, cluster_centers: np.ndarray, labels: np.ndarray, alpha: float) -> float:
    return alpha_power(X - cluster_centers[labels], alpha)

def compute_labels(X: np.ndarray, cluster_centers: np.ndarray) -> np.ndarray:
    return np.argmin(np.linalg.norm(X[:, None, :] - cluster_centers[None, :, :], axis=2), axis=1)

def update_cluster_centers(X: np.ndarray, cluster_centers: np.ndarray, labels: np.ndarray, alpha: float):
    n_clusters, _ = cluster_centers.shape
    updated_cluster_centers = np.zeros_like(cluster_centers)
    for i in range(n_clusters):
        mask = labels == i
        if mask.sum() > 0:
            updated_cluster_centers[i] = alpha_location(X[mask], alpha)
        else:
            updated_cluster_centers[i] = cluster_centers[i]
    return updated_cluster_centers

class AlphaStableKMeans(ClusterMixin, BaseEstimator):

    def __init__(self, n_clusters: int = 8, alpha: float = 1.0, *, max_iter: int = 100, tol: float = 1e-6):
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
    
    def fit(self, X, y = None):
        X, _ = self._validate_data(X)
        
        cluster_centers = initialize_cluster_centers(X, self.n_clusters)
        labels = compute_labels(X, cluster_centers)
        inertia = 0

        for _ in range(self.max_iter):
            previous_inertia = inertia
            inertia = compute_inertia(X, cluster_centers, labels, self.alpha)
            if abs((previous_inertia - inertia) / max(previous_inertia, 1e-12)) < self.tol:
                break
            cluster_centers = update_cluster_centers(X, cluster_centers, labels, self.alpha)
            labels = compute_labels(X, cluster_centers)
        
        self._n_features = X.shape[1]
        self.cluster_centers_ = cluster_centers
        self.labels_ = labels
        self.inertia_ = inertia
        return self
        
    def predict(self, X):
        check_is_fitted(self, attributes=["cluster_centers_"])
        X, X_is_one_dimensional = self._validate_data(X, n_features=self._n_features)
        labels = compute_labels(X, self.cluster_centers_)
        if X_is_one_dimensional:
            labels = labels.ravel()
        return labels
    
    # TODO: check what is sample_weight and how to use it.
    def score(self, X, y=None, sample_weight=None):
        return - compute_inertia(X, self.cluster_centers_, self.predict(X), self.alpha)

    def _validate_data(self, X, n_features=None):
        X = validate_data(self, X)
        X = np.asarray(X, dtype=float)
        
        X_is_one_dimensional = X.ndim == 1
        if X_is_one_dimensional:
            X = X.reshape(-1, 1)
        
        if X.ndim != 2:
            raise ValueError(f"Expected X.shape to be 1D or 2D; got {X.ndim = }")
        
        if n_features is not None and X.shape[1] != n_features:
            raise ValueError(f"Expected X.shape[1] to be {n_features}; got {X.shape[1] = }")
        
        return X, X_is_one_dimensional