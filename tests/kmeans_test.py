import numpy as np
import pytest
from scipy.stats import cauchy, multivariate_t
from sklearn.exceptions import NotFittedError

from aub_htp.machine_learning.kmeans import (
    initialize_cluster_centers,
    compute_inertia,
    compute_labels,
    update_cluster_centers,
    AlphaStableKMeans,
)


def generate_univariate_data(alpha: float, n_samples: int, loc: float = 0.0, scale: float = 1.0) -> np.ndarray:
    """Generate univariate data matched to alpha."""
    if np.isclose(alpha, 2.0):
        return np.random.normal(loc, scale, n_samples).reshape(-1, 1)
    if np.isclose(alpha, 1.0):
        return cauchy.rvs(loc=loc, scale=scale, size=n_samples).reshape(-1, 1)
    raise ValueError("This test helper only supports alpha = 1 or alpha = 2.")


def generate_multivariate_data(
    alpha: float,
    n_samples: int,
    mean: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """Generate multivariate data matched to alpha."""
    mean = np.asarray(mean, dtype=float)
    d = mean.size

    if np.isclose(alpha, 2.0):
        return np.random.multivariate_normal(mean, (scale**2) * np.eye(d), size=n_samples)
    if np.isclose(alpha, 1.0):
        return multivariate_t.rvs(loc=mean, shape=(scale**2) * np.eye(d), df=1, size=n_samples)
    raise ValueError("This test helper only supports alpha = 1 or alpha = 2.")


class TestInitializeClusterCenters:
    def test_shape_univariate(self):
        """initialize_cluster_centers should return shape (n_clusters, 1) for univariate feature data."""
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        centers = initialize_cluster_centers(X, 2)
        assert centers.shape == (2, 1)

    def test_shape_multivariate(self):
        """initialize_cluster_centers should return shape (n_clusters, n_features) for multivariate data."""
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        centers = initialize_cluster_centers(X, 2)
        assert centers.shape == (2, 2)

    def test_percentile_positions(self):
        """initialize_cluster_centers should match percentile-based initialization."""
        X = np.array([[0.0], [10.0], [20.0], [30.0], [40.0]])
        centers = initialize_cluster_centers(X, 2)
        expected = np.percentile(X, [25, 75], axis=0)
        assert np.allclose(centers, expected)


class TestComputeLabels:
    def test_nearest_center_univariate(self):
        """compute_labels should assign each 1D point to the nearest center."""
        X = np.array([[0.0], [1.0], [9.0], [10.0]])
        centers = np.array([[0.0], [10.0]])
        labels = compute_labels(X, centers)
        assert np.array_equal(labels, [0, 0, 1, 1])

    def test_nearest_center_multivariate(self):
        """compute_labels should assign each 2D point to the nearest center."""
        X = np.array([[0.0, 0.0], [1.0, 1.0], [9.0, 9.0], [10.0, 10.0]])
        centers = np.array([[0.0, 0.0], [10.0, 10.0]])
        labels = compute_labels(X, centers)
        assert np.array_equal(labels, [0, 0, 1, 1])


class TestComputeInertia:
    @pytest.mark.parametrize("alpha", [1.0, 2.0])
    def test_returns_scalar_univariate(self, alpha):
        """compute_inertia should return a scalar for univariate data."""
        np.random.seed(42)
        X = generate_univariate_data(alpha, n_samples=40, loc=0.0, scale=1.0)
        centers = np.array([[-1.0], [1.0]])
        labels = compute_labels(X, centers)

        inertia = compute_inertia(X, centers, labels, alpha)

        assert np.isscalar(inertia)

    @pytest.mark.parametrize("alpha", [1.0, 2.0])
    def test_returns_scalar_multivariate(self, alpha):
        """compute_inertia should return a scalar for multivariate data."""
        np.random.seed(42)
        X = generate_multivariate_data(alpha, n_samples=40, mean=np.array([0.0, 0.0]), scale=1.0)
        centers = np.array([[-1.0, -1.0], [1.0, 1.0]])
        labels = compute_labels(X, centers)

        inertia = compute_inertia(X, centers, labels, alpha)

        assert np.isscalar(inertia)

    def test_inertia_translation_invariance(self):
        """Inertia should not change if both data and centers are shifted by the same amount."""
        X = np.array([[0.0], [1.0], [9.0], [10.0]])
        centers = np.array([[0.0], [10.0]])
        labels = np.array([0, 0, 1, 1])

        shift = 5.0

        inertia_original = compute_inertia(X, centers, labels, alpha=2.0)
        inertia_shifted = compute_inertia(X + shift, centers + shift, labels, alpha=2.0)

        assert np.isclose(inertia_original, inertia_shifted, rtol=0.05)


class TestUpdateClusterCenters:
    @pytest.mark.parametrize("alpha", [1.0, 2.0])
    def test_shape_preserved_univariate(self, alpha):
        """update_cluster_centers should preserve center shape for univariate data."""
        np.random.seed(42)
        X = generate_univariate_data(alpha, n_samples=20, loc=0.0, scale=1.0)
        centers = np.array([[-1.0], [1.0]])
        labels = compute_labels(X, centers)

        updated = update_cluster_centers(X, centers, labels, alpha)

        assert updated.shape == centers.shape

    @pytest.mark.parametrize("alpha", [1.0, 2.0])
    def test_shape_preserved_multivariate(self, alpha):
        """update_cluster_centers should preserve center shape for multivariate data."""
        np.random.seed(42)
        X = generate_multivariate_data(alpha, n_samples=20, mean=np.array([0.0, 0.0]), scale=1.0)
        centers = np.array([[-1.0, -1.0], [1.0, 1.0]])
        labels = compute_labels(X, centers)

        updated = update_cluster_centers(X, centers, labels, alpha)

        assert updated.shape == centers.shape

    def test_alpha_2_matches_cluster_means_univariate(self):
        """
        For alpha=2 (Gaussian case), alpha_location reduces to the mean.
        This test verifies that updated cluster centers match the sample means
        of each cluster using generated univariate normal data.
        """
        np.random.seed(42)

        # Generate two clusters
        cluster1 = generate_univariate_data(alpha=2.0,loc=2,n_samples=1000)
        cluster2 = generate_univariate_data(alpha=2.0,loc=10 ,n_samples=1000)

        X = np.concatenate([cluster1, cluster2]).reshape(-1, 1)
        labels = np.array([0]*len(cluster1) + [1]*len(cluster2))

        # Initial centers (not important for this test)
        centers = np.array([[0.0], [12.0]])

        updated = update_cluster_centers(X, centers, labels, alpha=2.0)

        # Compute true means
        expected = np.array([
            [np.mean(X[labels == 0])],
            [np.mean(X[labels == 1])]
        ])

        assert np.allclose(updated, expected, rtol=0.05, atol=0.2)

    def test_alpha_2_matches_cluster_means_multivariate(self):
        """
        For alpha=2, alpha_location should match the mean vector
        for each cluster in the multivariate Gaussian case.
        """
        np.random.seed(42)

        d = 2
        # Generate two clusters
        cluster1 = generate_multivariate_data(alpha=2.0, mean=np.array([2,-3]), n_samples=1500) 
        cluster2 = generate_multivariate_data(alpha=2.0, mean=np.array([10,5]), n_samples=1500)

        X = np.vstack([cluster1, cluster2])
        labels = np.array([0]*len(cluster1) + [1]*len(cluster2))

        centers = np.array([[0.0, 0.0], [12.0, 6.0]])

        updated = update_cluster_centers(X, centers, labels, alpha=2.0)

        # Compute true means per cluster
        expected = np.vstack([
            np.mean(X[labels == 0], axis=0),
            np.mean(X[labels == 1], axis=0)
        ])

        assert np.allclose(updated, expected, rtol=0.05, atol=0.3)
    @pytest.mark.parametrize("alpha", [1.0, 2.0])
    def test_empty_cluster_unchanged(self, alpha):
        """update_cluster_centers should leave an empty cluster center unchanged."""
        np.random.seed(42)
        X = generate_univariate_data(alpha, n_samples=10, loc=0.0, scale=1.0)
        centers = np.array([[0.0], [10.0]])
        labels = np.zeros(X.shape[0], dtype=int)

        updated = update_cluster_centers(X, centers, labels, alpha)

        assert np.allclose(updated[1], centers[1])


class TestAlphaStableKMeansValidation:
    def test_validate_data_reshapes_univariate_input(self):
        """_validate_data should reshape 1D input to shape (n_samples, 1)."""
        model = AlphaStableKMeans(n_clusters=2, alpha=2.0)
        X = np.array([1.0, 2.0, 3.0, 4.0]).reshape(-1, 1)
        validated_X, X_is_one_dimensional = model._validate_data(X)
        assert validated_X.shape == (4, 1)
   

    def test_validate_data_preserves_multivariate_input_shape(self):
        """_validate_data should preserve the shape of 2D input."""
        model = AlphaStableKMeans(n_clusters=2, alpha=2.0)
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

        validated_X, X_is_one_dimensional = model._validate_data(X)

        assert validated_X.shape == X.shape
        assert X_is_one_dimensional is False

    def test_validate_data_raises_for_wrong_number_of_features(self):
        """_validate_data should raise when the number of features does not match."""
        model = AlphaStableKMeans(n_clusters=2, alpha=2.0)
        X = np.array([[1.0, 10.0], [2.0, 20.0]])

        with pytest.raises(ValueError):
            model._validate_data(X, n_features=1)


class TestAlphaStableKMeansEstimator:
    @pytest.mark.parametrize("alpha", [1.0, 2.0])
    def test_fit_sets_attributes_univariate(self, alpha):
        """fit should set fitted attributes for 1D data."""
        np.random.seed(42)
        X = generate_univariate_data(alpha, n_samples=30, loc=0.0, scale=1.0)

        model = AlphaStableKMeans(n_clusters=2, alpha=alpha, max_iter=20)
        model.fit(X)

        assert hasattr(model, "cluster_centers_")
        assert hasattr(model, "labels_")
        assert hasattr(model, "inertia_")
        assert hasattr(model, "_n_features")

    @pytest.mark.parametrize("alpha", [1.0, 2.0])
    def test_fit_sets_attributes_multivariate(self, alpha):
        """fit should set fitted attributes for multivariate data."""
        np.random.seed(42)
        X = generate_multivariate_data(alpha, n_samples=30, mean=np.array([0.0, 0.0]), scale=1.0)

        model = AlphaStableKMeans(n_clusters=2, alpha=alpha, max_iter=20)
        model.fit(X)

        assert hasattr(model, "cluster_centers_")
        assert hasattr(model, "labels_")
        assert hasattr(model, "inertia_")
        assert hasattr(model, "_n_features")

    @pytest.mark.parametrize("alpha", [1.0, 2.0])
    def test_predict_returns_one_label_per_sample_univariate(self, alpha):
        """predict should return one label per sample for univariate data."""
        np.random.seed(42)
        X = generate_univariate_data(alpha, n_samples=30, loc=0.0, scale=1.0)

        model = AlphaStableKMeans(n_clusters=2, alpha=alpha, max_iter=20)
        model.fit(X)
        labels = model.predict(X)

        assert labels.shape == (X.shape[0],)

    @pytest.mark.parametrize("alpha", [1.0, 2.0])
    def test_predict_returns_one_label_per_sample_multivariate(self, alpha):
        """predict should return one label per sample for multivariate data."""
        np.random.seed(42)
        X = generate_multivariate_data(alpha, n_samples=30, mean=np.array([0.0, 0.0]), scale=1.0)

        model = AlphaStableKMeans(n_clusters=2, alpha=alpha, max_iter=20)
        model.fit(X)
        labels = model.predict(X)

        assert labels.shape == (X.shape[0],)

    @pytest.mark.parametrize("alpha", [1.0, 2.0])
    def test_score_matches_negative_inertia_univariate(self, alpha):
        """score should equal negative inertia on univariate data."""
        np.random.seed(42)
        X = generate_univariate_data(alpha, n_samples=30, loc=0.0, scale=1.0)

        model = AlphaStableKMeans(n_clusters=2, alpha=alpha, max_iter=20)
        model.fit(X)

        expected = -compute_inertia(X, model.cluster_centers_, model.predict(X), model.alpha)
        actual = model.score(X)

        assert np.isclose(actual, expected)

    @pytest.mark.parametrize("alpha", [1.0, 2.0])
    def test_score_matches_negative_inertia_multivariate(self, alpha):
        """score should equal negative inertia on multivariate data."""
        np.random.seed(42)
        X = generate_multivariate_data(alpha, n_samples=30, mean=np.array([0.0, 0.0]), scale=1.0)

        model = AlphaStableKMeans(n_clusters=2, alpha=alpha, max_iter=20)
        model.fit(X)

        expected = -compute_inertia(X, model.cluster_centers_, model.predict(X), model.alpha)
        actual = model.score(X)

        assert np.isclose(actual, expected)


class Alpha_location_properties:
    def test_alpha_2_matches_mean_for_single_cluster(self):
        """For alpha=2 and one cluster, the fitted center should match the sample mean."""
        np.random.seed(42)
        X = generate_univariate_data(alpha=2.0, n_samples=2000, loc=5.0, scale=2.0)

        model = AlphaStableKMeans(n_clusters=1, alpha=2.0, max_iter=50)
        model.fit(X)

        assert np.isclose(model.cluster_centers_[0, 0], np.mean(X), atol=0.2)

    def test_alpha_2_matches_mean_for_single_cluster_multivariate(self):
        """For alpha=2 and one cluster, the fitted multivariate center should match the sample mean."""
        np.random.seed(42)
        X = generate_multivariate_data(alpha=2.0, n_samples=2000, mean=np.array([5.0, -3.0]), scale=2.0)

        model = AlphaStableKMeans(n_clusters=1, alpha=2.0, max_iter=50)
        model.fit(X)

        assert np.allclose(model.cluster_centers_[0], np.mean(X, axis=0), atol=0.25)

    def test_alpha_1_more_robust_than_alpha_2_to_univariate_outliers(self):
        """With strong 1D outliers, alpha=1 should produce a center closer to the clean-data center than alpha=2."""
        np.random.seed(42)
        clean = generate_univariate_data(alpha=2.0, n_samples=1000, loc=0.0, scale=1.0)
        outliers = np.array([[1000.0], [-1000.0]])
        contaminated = np.vstack([clean, outliers])

        model_alpha_2 = AlphaStableKMeans(n_clusters=1, alpha=2.0, max_iter=50)
        model_alpha_1 = AlphaStableKMeans(n_clusters=1, alpha=1.0, max_iter=50)

        model_alpha_2.fit(contaminated)
        model_alpha_1.fit(contaminated)

        center_alpha_2 = model_alpha_2.cluster_centers_[0, 0]
        center_alpha_1 = model_alpha_1.cluster_centers_[0, 0]

        assert abs(center_alpha_1) < abs(center_alpha_2)

    def test_alpha_1_more_robust_than_alpha_2_to_multivariate_outliers(self):
        """With strong 2D outliers, alpha=1 should produce a center closer to the clean-data center than alpha=2."""
        np.random.seed(42)
        clean = generate_multivariate_data(alpha=2.0, n_samples=1000, mean=np.array([0.0, 0.0]), scale=1.0)
        outliers = np.array([[1000.0, 1000.0], [-1000.0, -1000.0]])
        contaminated = np.vstack([clean, outliers])

        model_alpha_2 = AlphaStableKMeans(n_clusters=1, alpha=2.0, max_iter=50)
        model_alpha_1 = AlphaStableKMeans(n_clusters=1, alpha=1.0, max_iter=50)

        model_alpha_2.fit(contaminated)
        model_alpha_1.fit(contaminated)

        norm_alpha_2 = np.linalg.norm(model_alpha_2.cluster_centers_[0])
        norm_alpha_1 = np.linalg.norm(model_alpha_1.cluster_centers_[0])

        assert norm_alpha_1 < norm_alpha_2

    def test_fit_separates_two_clear_gaussian_clusters(self):
        """For alpha=2, the model should identify two clearly separated 1D Gaussian clusters."""
        np.random.seed(42)
        cluster_1 = generate_univariate_data(alpha=2.0, n_samples=50, loc=0.0, scale=0.2)
        cluster_2 = generate_univariate_data(alpha=2.0, n_samples=50, loc=10.0, scale=0.2)
        X = np.vstack([cluster_1, cluster_2])

        model = AlphaStableKMeans(n_clusters=2, alpha=2.0, max_iter=50)
        model.fit(X)

        assert len(np.unique(model.labels_)) == 2

    def test_fit_separates_two_clear_cauchy_clusters(self):
        """For alpha=1, the model should identify two clearly separated 1D Cauchy clusters."""
        np.random.seed(42)
        cluster_1 = generate_univariate_data(alpha=1.0, n_samples=100, loc=0.0, scale=0.2)
        cluster_2 = generate_univariate_data(alpha=1.0, n_samples=100, loc=10.0, scale=0.2)
        X = np.vstack([cluster_1, cluster_2])

        model = AlphaStableKMeans(n_clusters=2, alpha=1.0, max_iter=50)
        model.fit(X)

        assert len(np.unique(model.labels_)) == 2