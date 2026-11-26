import pytest
import numpy as np
from scipy.ndimage import shift
from scipy.stats import alpha, cauchy
from aub_htp.statistics import alpha_power, alpha_location, isotropic_entropy, isotropic_pdf
from scipy.stats import multivariate_t


class TestAlphaPowerCorrectness:
    def test_alpha_power_alpha_2_matches_square_root_of_expected_value_of_X_squared(self):
        """For alpha=2, alpha_power should equal sqrt(E[X^2]) in the univariate case."""
        np.random.seed(42)
        data = np.random.normal(10, 100, 1000)
        expected_value_of_X_squared = np.mean(data ** 2)**.5
        alpha_power_of_data = alpha_power(data, 2)
        assert np.isclose(alpha_power_of_data, expected_value_of_X_squared)

    def test_alpha_power_alpha_2_matches_square_root_of_expected_squared_norm_multivariate(self):
        """For alpha=2, alpha_power should equal sqrt(E[||X||^2]/d) in the multivariate case."""
        np.random.seed(42)
        data = np.random.multivariate_normal(
            mean=[0, 0],
            cov=[[1, 0], [0, 1]],
            size=1000
        )

        d = data.shape[1]
        expected_value_of_X_squared = (np.mean(np.sum(data ** 2, axis=1)) / d) ** 0.5
        alpha_power_of_data = alpha_power(data, 2)

        assert np.allclose(alpha_power_of_data, expected_value_of_X_squared)

    def test_alpha_power_alpha_2_matches_normal_scale(self):
        """For alpha=2, alpha_power should recover the standard deviation of a normal distribution."""
        np.random.seed(42)
        std = 100
        data = np.random.normal(0, std, 1000)
        alpha_power_of_data = alpha_power(data, 2)
        assert np.isclose(alpha_power_of_data, std, rtol=0.05)

    def test_alpha_power_alpha_2_matches_normal_scale_multivariate(self):
        """For alpha=2, alpha_power should recover the scale of an isotropic multivariate normal."""
        np.random.seed(42)
        std = 100
        d = 2
        data = np.random.multivariate_normal(
            mean=np.zeros(d),
            cov=(std**2) * np.eye(d),
            size=1000
        )
        alpha_power_of_data = alpha_power(data, 2)
        assert np.isclose(alpha_power_of_data, std, rtol=0.05)

    def test_alpha_power_alpha_1_matches_cauchy_scale(self):
        """For alpha=1, alpha_power should recover the scale of a Cauchy distribution."""
        np.random.seed(42)
        scale = 100
        data = cauchy.rvs(loc=0, scale=scale, size=1000)
        alpha_power_of_data = alpha_power(data, 1)
        assert np.isclose(alpha_power_of_data, scale, rtol=0.05)

    def test_alpha_power_alpha_1_matches_cauchy_scale_multivariate(self):
        """For alpha=1, alpha_power should recover the scale of an isotropic multivariate Cauchy."""
        np.random.seed(42)
        scale = 100
        d = 2
        data = multivariate_t.rvs(
            loc=np.zeros(d),
            shape=(scale**2) * np.eye(d),
            df=1,
            size=2000
        )
        alpha_power_of_data = alpha_power(data, 1)
        assert np.isclose(alpha_power_of_data, scale, rtol=0.05)

    @pytest.mark.parametrize("alpha", [0.2, 0.3, 1.2, 1.5, 0.9, 1.1, 2])
    def test_alpha_power_is_always_positive(self, alpha: float):
        """alpha_power should always be strictly positive for non-degenerate data."""
        data = np.random.normal(0, 1, 1000)
        alpha_power_of_data = alpha_power(data, alpha)
        assert alpha_power_of_data > 0

    @pytest.mark.parametrize(
        ("alpha", "scale"),
        [(0.2, -0.4), (0.5, 2), (1.0, 4), (1.2, 10.), (1.5, -20.), (1.5, 20.), (2., 40.)]
    )
    def test_alpha_power_matches_data_scaling(self, alpha: float, scale: float):
        """alpha_power should scale linearly with the magnitude of the data (scale equivariance)."""
        data = np.random.normal(0, 1, 1000)
        alpha_power_of_data = alpha_power(data, alpha)
        alpha_power_of_scaled_data = alpha_power(data * scale, alpha)

        assert np.isclose(
            alpha_power_of_data * abs(scale),
            alpha_power_of_scaled_data,
            rtol=0.05
        )

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5, 2.0])
    def test_alpha_power_is_sign_invariant(self, alpha):
        """alpha_power should be invariant to sign flips: P_alpha(-X) = P_alpha(X)."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)

        assert np.isclose(
            alpha_power(data, alpha),
            alpha_power(-data, alpha),
            rtol=0.05
        )
  

class TestAlphaLocationCorrectness:

    def test_alpha_location_alpha_2_matches_mean_univariate(self):
        """For alpha=2, alpha_location should match the sample mean in the univariate case."""
        np.random.seed(42)
        mean = 10
        std = 3
        data = np.random.normal(mean, std, 2000)

        estimated_location = alpha_location(data, 2.0)
        assert np.isclose(estimated_location, np.mean(data), rtol=0.05, atol=0.2)

    def test_alpha_location_alpha_2_matches_mean_multivariate(self):
        """For alpha=2, alpha_location should match the sample mean in the multivariate case."""
        np.random.seed(42)
        mean = np.array([10.0, -5.0])
        cov = np.array([[9.0, 0.0], [0.0, 4.0]])
        data = np.random.multivariate_normal(mean, cov, size=3000)

        estimated_location = alpha_location(data, 2.0)
        sample_mean = np.mean(data, axis=0)

        assert np.allclose(estimated_location, sample_mean, rtol=0.05, atol=0.25)

    @pytest.mark.parametrize(
        ("alpha", "a", "b"),
        [
            (0.5, 2.0, 3.0),
            (1.0, -4.0, 10.0),
            (1.5, 0.5, -7.0),
            (2.0, -3.0, 1.0),
        ]
    )
    def test_alpha_location_affine_equivariance_univariate(self, alpha, a, b):
        """alpha_location should satisfy affine equivariance: L(aX + b) = aL(X) + b."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1500)

        original_location = alpha_location(data, alpha)
        transformed_location = alpha_location(a * data + b, alpha)

        assert np.isclose(transformed_location, a * original_location + b, rtol=0.05, atol=0.3)

    @pytest.mark.parametrize("alpha", [1.0, 2.0])
    def test_alpha_location_translation_equivariance_multivariate(self, alpha):
        """alpha_location should be translation equivariant: L(X + b) = L(X) + b."""
        np.random.seed(42)
        shift = np.array([20.0, -15.0])
        data = np.random.multivariate_normal([0.0, 0.0], np.eye(2), size=2000)

        location_original = alpha_location(data, alpha)
        location_shifted = alpha_location(data + shift, alpha)

        assert np.allclose(location_shifted, location_original + shift, rtol=0.05, atol=0.4)

    def test_alpha_location_rotationally_symmetric_gaussian_center(self):
        """For isotropic Gaussian data centered at mu, alpha_location should recover the true center mu."""
        np.random.seed(42)
        mu = np.array([8.0, -6.0])
        std = 5.0
        data = np.random.multivariate_normal(mu, (std**2) * np.eye(2), size=3000)

        estimated_location = alpha_location(data, 2.0)
        assert np.allclose(estimated_location, mu, rtol=0.05, atol=0.35)

    def test_alpha_location_alpha_1_matches_cauchy_center_multivariate(self):
        """For alpha=1, alpha_location should recover the center of an isotropic multivariate Cauchy distribution."""
        np.random.seed(42)
        mu = np.array([12.0, -9.0])
        scale = 4.0

        data = multivariate_t.rvs(
            loc=mu,
            shape=(scale**2) * np.eye(2),
            df=1,
            size=4000
        )

        estimated_location = alpha_location(data, 1.0)
        assert np.allclose(estimated_location, mu, rtol=0.1, atol=0.6)

    def test_alpha_location_additivity_for_independent_symmetric_univariate(self):
        """alpha_location should be additive for independent symmetric variables: L(X + Y) = L(X) + L(Y)."""
        np.random.seed(42)
        x = np.random.normal(5.0, 2.0, 2000)
        y = np.random.normal(-3.0, 1.5, 2000)

        location_x = alpha_location(x, 2.0)
        location_y = alpha_location(y, 2.0)
        location_sum = alpha_location(x + y, 2.0)

        assert np.isclose(location_sum, location_x + location_y, rtol=0.05, atol=0.25)

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5, 2.0])
    def test_alpha_location_sign_equivariance(self, alpha):
        """alpha_location should be sign equivariant: L(-X) = -L(X)."""
        np.random.seed(42)
        data = np.random.normal(7.0, 2.0, 2000)

        location = alpha_location(data, alpha)
        negated_location = alpha_location(-data, alpha)

        assert np.isclose(negated_location, -location, rtol=0.05, atol=0.3)

class TestDimensionsAndShapes:

    @pytest.mark.parametrize("alpha", [1.0, 2.0])
    def test_alpha_location_shape_matches_single_feature_vector_shape(self, alpha):
        """
        alpha_location should return a vector of shape (1,) when the input data
        consists of 1-dimensional vectors (shape: (n_samples, 1)).
        """
        data = np.array([[1.0], [2.0], [3.0], [4.0]])

        location = alpha_location(data, alpha)

        assert location.shape == data[0].shape

    @pytest.mark.parametrize("alpha", [1.0, 2.0])
    def test_alpha_location_shape_matches_multivariate_vector_shape(self, alpha):
        """
        alpha_location should return a vector of shape (d,) when the input data
        consists of d-dimensional vectors (shape: (n_samples, d)).
        """
        data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])

        location = alpha_location(data, alpha)

        assert location.shape == data[0].shape

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5, 2.0])
    def test_alpha_location_returns_scalar_for_univariate_float_array(self, alpha):
        """
        alpha_location should return a scalar when the input data is a 1D array
        of floats (shape: (n_samples,)).
        """
        data = np.array([1.0, 2.0, 3.0, 4.0])

        location = alpha_location(data, alpha)

        assert np.isscalar(location)

    @pytest.mark.parametrize(
        ("alpha", "data"),
        [
            (0.5, np.array([1.0, 2.0, 3.0, 4.0])),
            (1.0, np.array([[1.0], [2.0], [3.0], [4.0]])),
            (1.0, np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])),
            (1.5, np.array([5.0, 6.0, 7.0, 8.0])),
            (2.0, np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])),
        ]
    )
    def test_alpha_power_always_returns_scalar(self, alpha, data):
        """
        alpha_power should always return a scalar value regardless of:
        - the dimensionality of the input data
        - the value of alpha
        """
        power = alpha_power(data, alpha)

        assert np.isscalar(power)


class TestAllZeroData:

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5, 2.0])
    @pytest.mark.parametrize("shape", [(10,), (10, 1)])
    def test_alpha_power_of_all_zero_data_is_zero(self, alpha, shape):
        """All-zero data has no root in P, so its power is 0."""
        assert alpha_power(np.zeros(shape), alpha) == 0

    @pytest.mark.parametrize("alpha", [1.0, 2.0])
    def test_alpha_power_of_all_zero_multivariate_data_is_zero(self, alpha):
        assert alpha_power(np.zeros((10, 3)), alpha) == 0


class TestAlphaCloseToOneOrTwo:

    @pytest.mark.parametrize("alpha", [1.0, 2.0])
    @pytest.mark.parametrize("offset", [-1e-9, 1e-9])
    def test_isotropic_entropy_takes_the_closed_form_branch(self, alpha, offset):
        """An alpha close to 1 or 2 gives the same entropy as exactly 1 or 2, in 1-D too."""
        assert isotropic_entropy(1, alpha + offset) == isotropic_entropy(1, alpha)
        assert isotropic_entropy(3, alpha + offset) == isotropic_entropy(3, alpha)

    @pytest.mark.parametrize("alpha", [1.0, 2.0])
    @pytest.mark.parametrize("offset", [-1e-9, 1e-9])
    def test_isotropic_pdf_takes_the_closed_form_branch(self, alpha, offset):
        data = np.linspace(-3, 3, 7)
        assert np.array_equal(isotropic_pdf(data, alpha + offset), isotropic_pdf(data, alpha))

    @pytest.mark.parametrize("alpha", [1.0, 2.0])
    def test_alpha_power_accepts_alpha_close_to_one_or_two(self, alpha):
        data = np.random.default_rng(0).normal(0, 1, 200)
        assert np.isclose(alpha_power(data, alpha + 1e-9), alpha_power(data, alpha))

    def test_close_alphas_share_one_cache_entry(self):
        from aub_htp.statistics.statistics import _isotropic_entropy
        isotropic_entropy(1, 1.0)
        misses = _isotropic_entropy.cache_info().misses
        isotropic_entropy(1, 1.0 + 1e-9)
        isotropic_entropy(1, 1.0 - 1e-9)
        assert _isotropic_entropy.cache_info().misses == misses

    @pytest.mark.parametrize("alpha", [1.0, 1.5, 2.0])
    def test_isotropic_entropy_accepts_zero_dimensional_array_alpha(self, alpha):
        """A 0-d array is unhashable, so it must be converted before the cache sees it."""
        assert isotropic_entropy(1, np.array(alpha)) == isotropic_entropy(1, alpha)

    def test_isotropic_entropy_matches_known_values_in_one_dimension(self):
        assert np.isclose(isotropic_entropy(1, 2), 0.5 * np.log(2 * np.pi * np.e))
        assert np.isclose(isotropic_entropy(1, 1), np.log(4 * np.pi))


class TestInvalidInput:

    @pytest.mark.parametrize("alpha", [0.0, -1.0, 3.0, 2.5, np.nan])
    def test_invalid_alpha_raises_value_error(self, alpha):
        data = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="alpha"):
            alpha_power(data, alpha)
        with pytest.raises(ValueError, match="alpha"):
            alpha_location(data, alpha)
        with pytest.raises(ValueError, match="alpha"):
            isotropic_pdf(data, alpha)
        with pytest.raises(ValueError, match="alpha"):
            isotropic_entropy(1, alpha)

    @pytest.mark.parametrize("function", [alpha_power, alpha_location])
    @pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
    @pytest.mark.parametrize("alpha", [1.0, 1.5, 2.0])
    def test_nan_or_infinite_data_raises_value_error(self, function, bad_value, alpha):
        with pytest.raises(ValueError, match="NaN or infinite"):
            function(np.array([1.0, bad_value, 3.0]), alpha)
        with pytest.raises(ValueError, match="NaN or infinite"):
            function(np.array([[1.0, 2.0], [bad_value, 4.0]]), alpha)

    @pytest.mark.parametrize("function", [alpha_power, alpha_location])
    @pytest.mark.parametrize("shape", [(0,), (0, 2), (3, 0)])
    def test_empty_data_raises_value_error(self, function, shape):
        with pytest.raises(ValueError, match="empty"):
            function(np.empty(shape), 1.0)

    @pytest.mark.parametrize("function", [alpha_power, alpha_location])
    @pytest.mark.parametrize("data", [np.array(1.0), np.ones((2, 2, 2))])
    def test_data_with_wrong_number_of_dimensions_raises_value_error(self, function, data):
        with pytest.raises(ValueError, match=r"\(n_samples,\) or \(n_samples, n_features\)"):
            function(data, 1.0)


class TestIsotropicPdfShape:

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5, 2.0])
    def test_isotropic_pdf_returns_one_value_per_sample_in_one_dimension(self, alpha):
        data = np.linspace(-3, 3, 7).reshape(-1, 1)
        assert isotropic_pdf(data, alpha).shape == (7,)

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5, 2.0])
    def test_isotropic_pdf_takes_one_dimensional_data_as_one_feature(self, alpha):
        data = np.linspace(-3, 3, 7)
        pdf = isotropic_pdf(data, alpha)
        assert pdf.shape == (7,)
        assert np.array_equal(pdf, isotropic_pdf(data.reshape(-1, 1), alpha))

    @pytest.mark.parametrize("alpha", [1.0, 2.0])
    def test_isotropic_pdf_returns_one_value_per_sample_in_many_dimensions(self, alpha):
        assert isotropic_pdf(np.ones((7, 3)), alpha).shape == (7,)

    def test_isotropic_pdf_matches_scipy_in_one_dimension(self):
        from scipy.stats import norm
        data = np.linspace(-3, 3, 7)
        assert np.allclose(isotropic_pdf(data, 2.0), norm.pdf(data))
        assert np.allclose(isotropic_pdf(data, 1.0), cauchy.pdf(data))

    @pytest.mark.parametrize("data", [np.array(1.0), np.ones((2, 2, 2))])
    def test_isotropic_pdf_with_wrong_number_of_dimensions_raises_value_error(self, data):
        with pytest.raises(ValueError, match=r"\(n_samples,\) or \(n_samples, n_features\)"):
            isotropic_pdf(data, 1.0)


def test_statistics_all_lists_the_public_functions():
    import aub_htp.statistics
    assert set(aub_htp.statistics.__all__) == {
        "alpha_power", "alpha_location", "isotropic_pdf", "isotropic_entropy",
    }