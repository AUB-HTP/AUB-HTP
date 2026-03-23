import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import validate_data, check_is_fitted

from ..statistics import alpha_location, alpha_power


def heavy_tailed_inertia(
    X: np.ndarray,
    centers: np.ndarray,
    labels: np.ndarray,
    *,
    alpha: float,
) -> float:
    """
    Compute the global heavy-tailed clustering objective.

    This plays the role of "inertia" in standard K-means, but instead of
    summing squared Euclidean distances, it computes the alpha-power of the
    residuals between each sample and its assigned cluster center.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input data.
    centers : np.ndarray of shape (n_clusters, n_features)
        Current cluster centers.
    labels : np.ndarray of shape (n_samples,)
        Cluster assignment for each sample.
    alpha : float
        Stability parameter used by alpha_power.

    Returns
    -------
    float
        Heavy-tailed inertia value. Smaller values indicate tighter clustering.
    """
    # Residual for each sample relative to its assigned center.
    residuals = X - centers[labels]

    # alpha_power acts as a heavy-tailed scale / dispersion measure.
    return float(alpha_power(residuals, alpha))


class AlphaStableKMeans(ClusterMixin, BaseEstimator):
    """
    Heavy-tailed K-means clustering using alpha_location and alpha_power.

    This estimator is a robust alternative to standard K-means. Standard
    K-means updates cluster centers with means and minimizes squared Euclidean
    distances. Here, cluster centers are updated using alpha_location and the
    clustering objective is measured with alpha_power.

    Objective
    ---------
    The algorithm alternates between:
    1. assigning each point to the nearest center in Euclidean distance
    2. updating each cluster center with alpha_location
    3. evaluating clustering quality with alpha_power

    Conceptually, it minimizes a heavy-tailed analogue of the K-means objective:

        minimize alpha_power(X - mu_cluster, alpha)

    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters.
    alpha : float, default=1.0
        Stability parameter in (0, 2].

        With the current statistics.py implementation:
        - 1D data supports general alpha in (0, 2]
        - multivariate data (n_features > 1) only supports alpha = 1
    max_iter : int, default=100
        Maximum number of Lloyd-style iterations.
    tol : float, default=1e-6
        Relative tolerance on the inertia for convergence.
    init : {"percentile", "random"}, default="percentile"
        Strategy used to initialize cluster centers.
    random_state : int or None, default=None
        Seed used when init="random".

    Attributes
    ----------
    cluster_centers_ : np.ndarray of shape (n_clusters, n_features)
        Final cluster centers.
    labels_ : np.ndarray of shape (n_samples,)
        Final cluster assignments for the training data.
    cluster_power_ : np.ndarray of shape (n_clusters,)
        Alpha-power of each cluster after the last update step.
    inertia_ : float
        Final global heavy-tailed inertia.
    inertia_history_ : np.ndarray
        Inertia values across iterations.
    n_iter_ : int
        Number of iterations performed.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        *,
        alpha: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-6,
        init: str = "percentile",
        random_state: int | None = None,
    ):
        """
        Initialize the estimator with hyperparameters only.

        Following scikit-learn conventions, __init__ should store parameters
        without performing validation or computation.
        """
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Fit the clustering model on X.

        The algorithm follows a Lloyd-style loop:
        1. initialize centers
        2. assign labels
        3. update centers with alpha_location
        4. compute heavy-tailed inertia
        5. stop when inertia stabilizes or labels stop changing

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None, default=None
            Ignored. Present for scikit-learn API consistency.

        Returns
        -------
        self : AlphaStableKMeans
            Fitted estimator.
        """
        # Validate input using sklearn utilities and store n_features_in_.
        X = validate_data(self, X, reset=True)
        X = np.asarray(X, dtype=float)

        # Ensure X is always 2D.
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape

        # Basic hyperparameter checks.
        if self.n_clusters < 1:
            raise ValueError("n_clusters must be at least 1")
        if self.n_clusters > n_samples:
            raise ValueError("n_clusters cannot exceed number of samples")

        alpha = float(self.alpha)
        if not (0 < alpha <= 2):
            raise ValueError("alpha must be in (0, 2]")

        # Current statistics.py only supports multivariate alpha=1.
        if n_features > 1 and alpha != 1:
            raise ValueError(
                "With the current statistics.py, multivariate clustering only supports alpha = 1"
            )

        # Random generator is only used for random initialization.
        rng = np.random.default_rng(self.random_state)

        # Step 1: initialize centers, then assign each point to its nearest one.
        centers = self._initialize_centers(X, rng)
        labels = self._assign_labels(X, centers)

        # Keep track of inertia values for convergence monitoring.
        inertia_history = []

        for it in range(self.max_iter):
            # Step 2: update cluster centers and compute per-cluster dispersion.
            centers, cluster_power = self._update_centers_and_cluster_power(
                X, labels, centers, alpha
            )

            # Step 3: evaluate global clustering objective.
            inertia = heavy_tailed_inertia(X, centers, labels, alpha=alpha)
            inertia_history.append(inertia)

            # Step 4: reassign labels based on updated centers.
            new_labels = self._assign_labels(X, centers)

            # Convergence check based on relative inertia change.
            if it >= 1:
                rel_change = abs(inertia_history[-1] - inertia_history[-2]) / max(
                    inertia_history[-2], 1e-12
                )
                if rel_change <= self.tol:
                    labels = new_labels
                    break

            # Secondary convergence check: if assignments stop changing.
            if np.array_equal(new_labels, labels):
                labels = new_labels
                break

            labels = new_labels

        # Save learned quantities using sklearn's trailing-underscore convention.
        self.cluster_centers_ = centers
        self.labels_ = labels
        self.cluster_power_ = cluster_power
        self.inertia_ = float(
            heavy_tailed_inertia(X, self.cluster_centers_, self.labels_, alpha=alpha)
        )
        self.inertia_history_ = np.asarray(inertia_history, dtype=float)
        self.n_iter_ = it + 1

        return self

    def predict(self, X):
        """
        Assign each sample in X to the nearest learned cluster center.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Predicted cluster indices.
        """
        # Make sure fit has already been called.
        check_is_fitted(self, attributes=["cluster_centers_", "labels_", "inertia_"])

        # Validate against training-time feature count.
        X = validate_data(self, X, reset=False)
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return self._assign_labels(X, self.cluster_centers_)

    def fit_predict(self, X, y=None):
        """
        Fit the model on X and return cluster assignments.

        This is a convenience method commonly provided by sklearn clustering
        estimators.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None, default=None
            Ignored.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Cluster labels for each sample.
        """
        return self.fit(X, y).labels_

    def transform(self, X):
        """
        Compute distances from each sample to each learned cluster center.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        distances : np.ndarray of shape (n_samples, n_clusters)
            Euclidean distance from each sample to each cluster center.

        Notes
        -----
        This is optional for clustering itself, but useful for inspection,
        feature engineering, and compatibility with sklearn-style APIs.
        """
        check_is_fitted(self, attributes=["cluster_centers_"])
        X = validate_data(self, X, reset=False)
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return self._pairwise_distances(X, self.cluster_centers_)

    def score(self, X, y=None):
        """
        Return a clustering score on X.

        This method returns the negative heavy-tailed inertia so that larger
        values correspond to better clusterings, matching sklearn's convention
        that higher scores are better.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : None, default=None
            Ignored.

        Returns
        -------
        float
            Negative heavy-tailed inertia.
        """
        X = validate_data(self, X, reset=False)
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        labels = self.predict(X)
        return -heavy_tailed_inertia(
            X, self.cluster_centers_, labels, alpha=float(self.alpha)
        )

    def _initialize_centers(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Initialize cluster centers according to the selected strategy.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        rng : np.random.Generator
            Random number generator used when init="random".

        Returns
        -------
        centers : np.ndarray of shape (n_clusters, n_features)
            Initial cluster centers.
        """
        if self.init == "percentile":
            return self._init_locations_percentiles(X, self.n_clusters)
        if self.init == "random":
            # Choose distinct samples as initial centers.
            indices = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
            return X[indices].copy()
        raise ValueError("init must be 'percentile' or 'random'")

    @staticmethod
    def _init_locations_percentiles(X: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        Initialize centers using evenly spaced percentiles of the data.

        In 1D, this places centers along the empirical distribution.
        In multiple dimensions, percentiles are computed independently per
        feature, producing a simple deterministic initialization.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.
        n_clusters : int
            Number of centers to initialize.

        Returns
        -------
        centers : np.ndarray of shape (n_clusters, n_features)
            Percentile-based initial centers.
        """
        X = np.asarray(X, dtype=float)
        _, n_features = X.shape

        if n_features == 1:
            centers = np.array(
                [
                    np.percentile(
                        X[:, 0],
                        100.0 * (2 * i + 1) / (2 * n_clusters),
                    )
                    for i in range(n_clusters)
                ],
                dtype=float,
            ).reshape(n_clusters, 1)
            return centers

        centers = np.zeros((n_clusters, n_features), dtype=float)
        for dim in range(n_features):
            for i in range(n_clusters):
                centers[i, dim] = np.percentile(
                    X[:, dim],
                    100.0 * (2 * i + 1) / (2 * n_clusters),
                )
        return centers

    @staticmethod
    def _pairwise_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distances from each sample to each cluster center.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.
        centers : np.ndarray of shape (n_clusters, n_features)
            Cluster centers.

        Returns
        -------
        np.ndarray of shape (n_samples, n_clusters)
            Distance matrix.
        """
        return np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)

    def _assign_labels(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        Assign each sample to the nearest center.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.
        centers : np.ndarray of shape (n_clusters, n_features)
            Current cluster centers.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Index of the closest center for each sample.
        """
        dists = self._pairwise_distances(X, centers)
        return np.argmin(dists, axis=1)

    def _update_centers_and_cluster_power(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        previous_centers: np.ndarray,
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Update cluster centers and compute per-cluster alpha-power.

        For each cluster:
        - collect its assigned points
        - estimate a robust center with alpha_location
        - measure within-cluster dispersion with alpha_power

        Empty clusters are left unchanged.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.
        labels : np.ndarray of shape (n_samples,)
            Current cluster assignments.
        previous_centers : np.ndarray of shape (n_clusters, n_features)
            Centers from the previous iteration.
        alpha : float
            Stability parameter.

        Returns
        -------
        centers : np.ndarray of shape (n_clusters, n_features)
            Updated cluster centers.
        cluster_power : np.ndarray of shape (n_clusters,)
            Alpha-power for each cluster.
        """
        n_features = X.shape[1]
        centers = previous_centers.copy()
        cluster_power = np.zeros(self.n_clusters, dtype=float)

        for j in range(self.n_clusters):
            # Points currently assigned to cluster j.
            Xj = X[labels == j]

            # If a cluster is empty, keep its old center.
            if Xj.shape[0] == 0:
                continue

            # Robust cluster center under the alpha-stable objective.
            mu_j = np.asarray(alpha_location(Xj, alpha), dtype=float).reshape(-1)

            # Defensive check to ensure dimensional consistency.
            if mu_j.size != n_features:
                raise ValueError(
                    f"alpha_location returned shape {mu_j.shape}, expected {(n_features,)}"
                )

            centers[j] = mu_j

            # Store per-cluster dispersion for diagnostics / analysis.
            cluster_power[j] = float(alpha_power(Xj - mu_j, alpha))

        return centers, cluster_power