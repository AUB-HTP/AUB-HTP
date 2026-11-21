import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from aub_htp.machine_learning import AlphaStableShape
from aub_htp.machine_learning.shape import shape as shape_module
from aub_htp.machine_learning.shape.log_difference import estimate_correlation_log_difference
from aub_htp.machine_learning.shape.ratio import compute_correlation_element, estimate_correlation_ratio
from aub_htp.machine_learning.shape.sheppard import estimate_correlation_sheppard
from aub_htp.machine_learning.shape.tyler import estimate_correlation_tyler
from aub_htp.machine_learning.shape.utils import repair_correlation_matrix

RATIO_METHODS = ["ratio-1", "ratio-2", "ratio-3"]
METHODS = RATIO_METHODS + ["log-difference", "sheppard", "tyler"]
# These estimate each rho_ij on its own, so their R may not be PSD.
PAIRWISE_METHODS = RATIO_METHODS + ["log-difference", "sheppard"]
ALPHAS = [1.0, 1.5, 2.0]
RHOS = [-0.9, -0.5, 0.0, 0.5, 0.9]
SCALES = [1e6]

N_SAMPLES = 4000
SIGMA = np.array([1.0, 2.0])
LOCATION = np.array([3.0, -1.0])


def sample_sub_gaussian(n_samples, alpha, shape_matrix, location, seed):
    """Rows are location + sqrt(A) G, with G ~ N(0, shape_matrix) and
    A ~ S(alpha/2, 1, cos(pi alpha/4)^(2/alpha), 0), sampled with Kanter's formula."""
    rng = np.random.default_rng(seed)
    n_features = shape_matrix.shape[0]
    G = rng.multivariate_normal(np.zeros(n_features), shape_matrix, size=n_samples)
    if alpha == 2.0:
        A = np.ones(n_samples)
    else:
        a = alpha / 2.0
        U = rng.uniform(0.0, np.pi, n_samples)
        E = rng.exponential(1.0, n_samples)
        A = (
            np.sin(a * U) / np.sin(U) ** (1.0 / a)
            * (np.sin((1.0 - a) * U) / E) ** ((1.0 - a) / a)
        )
    return location + np.sqrt(A)[:, None] * G


def make_shape_matrix(sigma, correlation_matrix):
    sigma = np.asarray(sigma, dtype=float)
    return sigma[:, None] * np.asarray(correlation_matrix, dtype=float) * sigma[None, :]


def to_sigma_and_correlation(shape_matrix):
    sigma = np.sqrt(np.diag(shape_matrix))
    return sigma, shape_matrix / np.outer(sigma, sigma)


def sigma_error(estimate, shape_matrix):
    """Largest relative error of sigma_i = sqrt(shape_ii)."""
    sigma_estimate, _ = to_sigma_and_correlation(estimate)
    sigma, _ = to_sigma_and_correlation(shape_matrix)
    return np.max(np.abs(sigma_estimate / sigma - 1.0))


def correlation_error(estimate, shape_matrix):
    """Largest absolute error of rho_ij = shape_ij / (sigma_i sigma_j)."""
    _, correlation_estimate = to_sigma_and_correlation(estimate)
    _, correlation = to_sigma_and_correlation(shape_matrix)
    return np.max(np.abs(correlation_estimate - correlation))


def correlation_tolerance(method, rho):
    # log-difference only sees |rho|: noisy at moderate rho, and biased away from 0 at rho = 0.
    if method == "log-difference":
        if rho == 0.0:
            return 0.5
        if abs(rho) == 0.5:
            return 0.25
    # ratio-2 takes |rho| from sqrt(1 - ...), which is biased away from 0 at rho = 0.
    if method == "ratio-2" and rho == 0.0:
        return 0.3
    # A strong correlation is estimated more precisely.
    if abs(rho) == 0.9:
        return 0.05
    return 0.1


def fit_two_features(method, alpha, rho):
    shape_matrix = make_shape_matrix(SIGMA, [[1.0, rho], [rho, 1.0]])
    X = sample_sub_gaussian(N_SAMPLES, alpha, shape_matrix, LOCATION, seed=0)
    return AlphaStableShape(alpha=alpha, method=method).fit(X), shape_matrix


class TestAlphaStableShapeCorrectness:

    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("alpha", ALPHAS)
    @pytest.mark.parametrize("rho", RHOS)
    def test_recovers_sigma(self, method, alpha, rho):
        """sqrt(shape_ii) should be close to the true sigma_i."""
        estimator, shape_matrix = fit_two_features(method, alpha, rho)
        assert sigma_error(estimator.shape_, shape_matrix) < 0.1

    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("alpha", ALPHAS)
    @pytest.mark.parametrize("rho", RHOS)
    def test_recovers_correlation(self, method, alpha, rho):
        """shape_ij / (sigma_i sigma_j) should be close to the true rho, sign included."""
        estimator, shape_matrix = fit_two_features(method, alpha, rho)
        tolerance = correlation_tolerance(method, rho)
        assert correlation_error(estimator.shape_, shape_matrix) < tolerance

    @pytest.mark.parametrize("method", METHODS)
    def test_recovers_shape_matrix_three_features(self, method):
        """shape_ should be close to the true shape matrix with mixed-sign correlations."""
        correlation_matrix = [[1.0, 0.8, -0.5], [0.8, 1.0, -0.3], [-0.5, -0.3, 1.0]]
        shape_matrix = make_shape_matrix([1.0, 0.5, 3.0], correlation_matrix)
        X = sample_sub_gaussian(N_SAMPLES, 1.0, shape_matrix, np.zeros(3), seed=1)

        estimator = AlphaStableShape(alpha=1.0, method=method).fit(X)

        assert sigma_error(estimator.shape_, shape_matrix) < 0.1
        assert correlation_error(estimator.shape_, shape_matrix) < 0.1

    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("alpha", ALPHAS)
    def test_recovers_location(self, method, alpha):
        """location_ should be close to the true location."""
        shape_matrix = make_shape_matrix(SIGMA, [[1.0, 0.5], [0.5, 1.0]])
        X = sample_sub_gaussian(N_SAMPLES, alpha, shape_matrix, LOCATION, seed=2)

        estimator = AlphaStableShape(alpha=alpha, method=method).fit(X)

        assert np.all(np.abs(estimator.location_ - LOCATION) < 0.15 * SIGMA)


class TestRatioZeroHandling:

    @pytest.mark.parametrize("equation", [1, 2, 3])
    def test_rows_with_zero_denominator_are_left_out(self, equation):
        """Exact zeros in x_j should be skipped, not turn rho into a guess of 0."""
        shape_matrix = make_shape_matrix(SIGMA, [[1.0, 0.5], [0.5, 1.0]])
        X = sample_sub_gaussian(N_SAMPLES, 1.0, shape_matrix, np.zeros(2), seed=10)
        X[::5, 1] = 0.0

        correlation_matrix = estimate_correlation_ratio(X, SIGMA, equation=equation)

        assert abs(correlation_matrix[0, 1] - 0.5) < 0.1

    @pytest.mark.parametrize("equation", [1, 2, 3])
    def test_tiny_scale_data_is_not_masked(self, equation):
        """The ratio x_i / x_j has no scale, so tiny data should give the same rho."""
        shape_matrix = make_shape_matrix(SIGMA, [[1.0, 0.5], [0.5, 1.0]])
        X = sample_sub_gaussian(N_SAMPLES, 1.0, shape_matrix, np.zeros(2), seed=11)

        correlation_matrix = estimate_correlation_ratio(X, SIGMA, equation=equation)
        correlation_matrix_of_tiny = estimate_correlation_ratio(1e-20 * X, 1e-20 * SIGMA, equation=equation)

        assert np.allclose(correlation_matrix_of_tiny, correlation_matrix, rtol=0.0, atol=1e-2)


class TestRatioEquations:

    # A Cauchy fit of z = x_i / x_j for sigma_i = 1, sigma_j = 2 and rho = 0.6:
    # mu = rho sigma_i / sigma_j = 0.3, gamma = (sigma_i / sigma_j) sqrt(1 - rho^2) = 0.4
    @pytest.mark.parametrize("equation", [1, 2, 3])
    @pytest.mark.parametrize("sign", [1.0, -1.0])
    def test_each_equation_recovers_rho_from_an_exact_fit(self, equation, sign):
        rho = compute_correlation_element(sign * 0.3, 0.4, 1.0, 2.0, equation)
        assert np.isclose(rho, sign * 0.6, rtol=1e-12, atol=0.0)

    def test_equation_2_clips_a_negative_square_root_to_zero(self):
        """gamma too large for the sigmas gives 1 - ... < 0, which is clipped to rho = 0."""
        assert compute_correlation_element(0.3, 0.6, 1.0, 2.0, 2) == 0.0

    def test_equation_3_clips_rho_to_one(self):
        assert compute_correlation_element(0.6, 0.4, 1.0, 2.0, 3) == 1.0

    def test_rejects_unknown_equation(self):
        with pytest.raises(ValueError):
            compute_correlation_element(0.3, 0.4, 1.0, 2.0, 4)


class TestLogDifference:

    @pytest.mark.parametrize("scale", [1e-20, 1e20])
    def test_scaled_data_gives_the_same_correlation(self, scale):
        """The floor on |x| is relative to sigma, so the scale of the data should not matter."""
        shape_matrix = make_shape_matrix(SIGMA, [[1.0, 0.5], [0.5, 1.0]])
        X = sample_sub_gaussian(N_SAMPLES, 1.0, shape_matrix, np.zeros(2), seed=18)

        correlation_matrix = estimate_correlation_log_difference(X, SIGMA)
        correlation_matrix_of_scaled = estimate_correlation_log_difference(scale * X, scale * SIGMA)

        assert np.allclose(correlation_matrix_of_scaled, correlation_matrix, rtol=0.0, atol=1e-9)


class TestSheppard:

    def test_rows_with_a_zero_product_are_left_out(self):
        """Exact zeros should not count as negative products."""
        shape_matrix = make_shape_matrix(SIGMA, [[1.0, 0.5], [0.5, 1.0]])
        X = sample_sub_gaussian(N_SAMPLES, 1.0, shape_matrix, np.zeros(2), seed=13)
        X_with_zeros = np.vstack([X, np.zeros((N_SAMPLES, 2))])

        correlation_matrix = estimate_correlation_sheppard(X)
        correlation_matrix_with_zeros = estimate_correlation_sheppard(X_with_zeros)

        assert np.array_equal(correlation_matrix_with_zeros, correlation_matrix)

    def test_known_sign_counts(self):
        """3 positive products out of 4 give p = 3/4, so rho = sin(pi / 4)."""
        X = np.array([[1.0, 2.0], [-1.0, -3.0], [2.0, 1.0], [1.0, -1.0], [0.0, 5.0]])
        correlation_matrix = estimate_correlation_sheppard(X)
        assert np.isclose(correlation_matrix[0, 1], np.sin(np.pi / 4.0), rtol=1e-12, atol=0.0)


class TestTyler:

    def test_rejects_too_few_samples(self):
        """Tyler needs more samples than features."""
        with pytest.raises(ValueError):
            estimate_correlation_tyler(np.array([[1.0, 2.0], [3.0, -1.0]]))

    def test_zero_rows_do_not_count_as_samples(self):
        """Rows equal to 0 are dropped before counting samples."""
        X = np.array([[1.0, 2.0], [3.0, -1.0], [0.0, 0.0], [0.0, 0.0]])
        with pytest.raises(ValueError):
            estimate_correlation_tyler(X)

    def test_fit_rejects_too_few_samples(self):
        with pytest.raises(ValueError):
            AlphaStableShape(alpha=1.0, method="tyler").fit(np.array([[1.0, 2.0], [3.0, -1.0]]))

    def test_rejects_linearly_dependent_features(self):
        """A repeated feature makes the estimate singular, which should be a ValueError."""
        shape_matrix = make_shape_matrix(SIGMA, [[1.0, 0.5], [0.5, 1.0]])
        X = sample_sub_gaussian(500, 1.0, shape_matrix, np.zeros(2), seed=19)
        X = np.column_stack([X, X[:, 0]])
        with pytest.raises(ValueError, match="became singular"):
            estimate_correlation_tyler(X)

    def test_zero_rows_are_dropped(self):
        """Adding rows equal to 0 should not change the result."""
        shape_matrix = make_shape_matrix(SIGMA, [[1.0, 0.5], [0.5, 1.0]])
        X = sample_sub_gaussian(500, 1.0, shape_matrix, np.zeros(2), seed=14)
        X_with_zeros = np.vstack([X, np.zeros((100, 2))])

        correlation_matrix = estimate_correlation_tyler(X)
        correlation_matrix_with_zeros = estimate_correlation_tyler(X_with_zeros)

        assert np.allclose(correlation_matrix_with_zeros, correlation_matrix, rtol=0.0, atol=1e-12)

    def test_warns_when_it_does_not_converge(self):
        shape_matrix = make_shape_matrix(SIGMA, [[1.0, 0.9], [0.9, 1.0]])
        X = sample_sub_gaussian(500, 1.0, shape_matrix, np.zeros(2), seed=15)
        with pytest.warns(ConvergenceWarning):
            estimate_correlation_tyler(X, iterations=1)

    def test_converges_without_warning(self):
        shape_matrix = make_shape_matrix(SIGMA, [[1.0, 0.9], [0.9, 1.0]])
        X = sample_sub_gaussian(500, 1.0, shape_matrix, np.zeros(2), seed=15)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            estimate_correlation_tyler(X)

    def test_huge_rows_do_not_overflow(self):
        """Rows are normalized, so entries near the largest float should be fine."""
        shape_matrix = make_shape_matrix(SIGMA, [[1.0, 0.5], [0.5, 1.0]])
        X = sample_sub_gaussian(500, 1.0, shape_matrix, np.zeros(2), seed=16)

        correlation_matrix = estimate_correlation_tyler(X)
        correlation_matrix_of_huge = estimate_correlation_tyler(1e300 * X)

        assert np.allclose(correlation_matrix_of_huge, correlation_matrix, rtol=0.0, atol=1e-9)

    def test_result_is_psd(self):
        correlation_matrix = [[1.0, 0.8, -0.5], [0.8, 1.0, -0.3], [-0.5, -0.3, 1.0]]
        shape_matrix = make_shape_matrix([1.0, 0.5, 3.0], correlation_matrix)
        X = sample_sub_gaussian(500, 1.0, shape_matrix, np.zeros(3), seed=17)

        estimate = estimate_correlation_tyler(X)

        assert np.linalg.eigvalsh(estimate).min() > 0
        assert np.allclose(np.diag(estimate), 1.0, rtol=0.0, atol=1e-12)


# Each pair is a valid correlation, but together they are impossible:
# its smallest eigenvalue is -0.8.
NOT_PSD = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])


class TestPositiveSemiDefiniteRepair:

    def test_repairs_a_matrix_that_is_not_psd(self):
        """The result should be a PSD correlation matrix, and a warning should be raised."""
        with pytest.warns(UserWarning, match="not positive semi-definite"):
            repaired = repair_correlation_matrix(NOT_PSD)

        assert np.linalg.eigvalsh(repaired).min() > 0
        assert np.allclose(np.diag(repaired), 1.0, rtol=0.0, atol=1e-12)
        assert np.allclose(repaired, repaired.T, rtol=0.0, atol=1e-12)
        assert np.all(np.abs(repaired) <= 1.0 + 1e-12)

    @pytest.mark.parametrize("rho", [0.0, 0.5, -0.9, 1.0])
    def test_leaves_a_psd_matrix_unchanged(self, rho):
        """A PSD matrix should come back as it is, with no warning."""
        correlation_matrix = np.array([[1.0, rho], [rho, 1.0]])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            repaired = repair_correlation_matrix(correlation_matrix)

        assert np.array_equal(repaired, correlation_matrix)

    @pytest.mark.parametrize("method", PAIRWISE_METHODS)
    def test_fit_returns_a_psd_shape(self, method, monkeypatch):
        """fit should repair R before building shape_."""
        name = "ratio" if method in RATIO_METHODS else method.replace("-", "_")
        monkeypatch.setattr(shape_module, "estimate_correlation_" + name, lambda *args, **kwargs: NOT_PSD.copy())
        shape_matrix = make_shape_matrix([1.0, 2.0, 0.5], np.eye(3))
        X = sample_sub_gaussian(500, 1.0, shape_matrix, np.zeros(3), seed=12)

        with pytest.warns(UserWarning, match="not positive semi-definite"):
            estimator = AlphaStableShape(alpha=1.0, method=method).fit(X)

        assert np.linalg.eigvalsh(estimator.shape_).min() > 0


class TestAlphaStableShapeScaling:

    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("scale", SCALES)
    def test_scaling_data_by_c_scales_shape_by_c_squared(self, method, scale):
        """Fitting c X should give c^2 times the shape matrix of X."""
        shape_matrix = make_shape_matrix(SIGMA, [[1.0, 0.5], [0.5, 1.0]])
        X = sample_sub_gaussian(N_SAMPLES, 1.0, shape_matrix, np.zeros(2), seed=3)

        shape = AlphaStableShape(alpha=1.0, method=method).fit(X).shape_
        shape_of_scaled = AlphaStableShape(alpha=1.0, method=method).fit(scale * X).shape_

        sigma, correlation = to_sigma_and_correlation(shape)
        sigma_of_scaled, correlation_of_scaled = to_sigma_and_correlation(shape_of_scaled)
        assert np.allclose(sigma_of_scaled / scale, sigma, rtol=1e-3, atol=0.0)
        # alpha_location is only accurate to about 1e-3 sigma, and log-difference is
        # sensitive to the location through the samples nearest to it.
        tolerance = 0.1 if method == "log-difference" else 1e-2
        assert np.allclose(correlation_of_scaled, correlation, rtol=0.0, atol=tolerance)

    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("scale", SCALES)
    def test_scaling_data_by_c_scales_location_by_c(self, method, scale):
        """Fitting c X should give c times the location of X."""
        shape_matrix = make_shape_matrix(SIGMA, [[1.0, 0.5], [0.5, 1.0]])
        X = sample_sub_gaussian(N_SAMPLES, 1.0, shape_matrix, LOCATION, seed=4)

        location = AlphaStableShape(alpha=1.0, method=method).fit(X).location_
        location_of_scaled = AlphaStableShape(alpha=1.0, method=method).fit(scale * X).location_

        assert np.all(np.abs(location_of_scaled / scale - location) < 1e-2 * SIGMA)


class TestAlphaStableShapeOutput:

    @pytest.mark.parametrize("method", METHODS)
    def test_output_shapes_and_symmetry(self, method):
        """shape_ should be a symmetric (d, d) matrix and location_ a (d,) vector."""
        shape_matrix = make_shape_matrix([1.0, 2.0, 0.5], np.eye(3))
        X = sample_sub_gaussian(500, 1.0, shape_matrix, np.zeros(3), seed=5)

        estimator = AlphaStableShape(alpha=1.0, method=method).fit(X)

        assert estimator.shape_.shape == (3, 3)
        assert estimator.location_.shape == (3,)
        assert np.allclose(estimator.shape_, estimator.shape_.T, rtol=1e-12, atol=0.0)

    @pytest.mark.parametrize("method", METHODS)
    def test_fit_returns_self(self, method):
        """fit should return the estimator, as scikit-learn expects."""
        X = sample_sub_gaussian(200, 1.0, np.eye(2), np.zeros(2), seed=6)
        estimator = AlphaStableShape(alpha=1.0, method=method)
        assert estimator.fit(X) is estimator


class TestAlphaStableShapeValidation:

    @pytest.mark.parametrize("method", METHODS)
    def test_rejects_one_dimensional_input(self, method):
        """1-D input should be rejected, as scikit-learn does."""
        X = sample_sub_gaussian(200, 1.0, np.eye(1), np.zeros(1), seed=7).ravel()
        with pytest.raises(ValueError):
            AlphaStableShape(alpha=1.0, method=method).fit(X)

    @pytest.mark.parametrize("method", METHODS)
    def test_rejects_fewer_than_two_samples(self, method):
        """A single sample has no spread, so fit should reject it."""
        with pytest.raises(ValueError):
            AlphaStableShape(alpha=1.0, method=method).fit(np.array([[1.0, 2.0]]))

    @pytest.mark.parametrize("method", ["unknown", "method1", "method2", "method3", "ratio"])
    def test_rejects_unknown_and_old_method_names(self, method):
        """The old names were removed without aliases."""
        X = sample_sub_gaussian(200, 1.0, np.eye(2), np.zeros(2), seed=8)
        with pytest.raises(ValueError):
            AlphaStableShape(alpha=1.0, method=method).fit(X)

    @pytest.mark.parametrize("argument", ["alpha_kernel", "alpha_data"])
    def test_rejects_removed_alpha_arguments(self, argument):
        """alpha_kernel and alpha_data were replaced by the single alpha."""
        with pytest.raises(TypeError):
            AlphaStableShape(**{argument: 1.0})

    def test_has_no_score(self):
        """score was never implemented, so it was removed."""
        assert not hasattr(AlphaStableShape(), "score")

    @pytest.mark.parametrize("alpha", [0.0, -1.0, 2.5])
    def test_rejects_alpha_outside_range(self, alpha):
        X = sample_sub_gaussian(200, 1.0, np.eye(2), np.zeros(2), seed=9)
        with pytest.raises(ValueError):
            AlphaStableShape(alpha=alpha).fit(X)
