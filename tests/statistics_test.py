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


class TestMultivariateIsotropicPdfAndEntropy:
    """isotropic_pdf and isotropic_entropy in d >= 2 dimensions for alpha != 1, 2."""

    @staticmethod
    def points_at_radius(radius, dimensions):
        data = np.zeros((len(radius), dimensions))
        data[:, 0] = radius
        return data

    @pytest.mark.parametrize("alpha", [0.3, 0.5, 0.8, 1.2, 1.5, 1.9])
    @pytest.mark.parametrize("dimensions", [2, 3, 5])
    def test_isotropic_pdf_integrates_to_one(self, alpha, dimensions):
        """int |S^(d-1)| r^(d-1) f_d(r) dr = 1, integrated over s = ln r."""
        from scipy.integrate import simpson
        from scipy.special import gamma
        # The mass beyond r decays like r^(-alpha).
        s = np.linspace(-25, 35 / alpha, 40001)
        pdf = isotropic_pdf(self.points_at_radius(np.exp(s), dimensions), alpha)
        sphere_area = 2 * np.pi ** (dimensions / 2) / gamma(dimensions / 2)

        assert pdf.shape == s.shape
        assert np.isclose(simpson(sphere_area * np.exp(dimensions * s) * pdf, x=s), 1, atol=1e-4)

    @pytest.mark.parametrize("alpha", [0.5, 1.5])
    @pytest.mark.parametrize("dimensions", [2, 3])
    def test_isotropic_pdf_matches_radial_quadrature(self, alpha, dimensions):
        from aub_htp.pdf.multivariate import _isotropic_radial_pdf
        radius = np.concatenate(([0.0], np.geomspace(1e-5, 20, 41)))
        expected = _isotropic_radial_pdf(radius, alpha, dimensions, (1 / alpha) ** (1 / alpha))

        pdf = isotropic_pdf(self.points_at_radius(radius, dimensions), alpha)

        assert np.allclose(pdf, expected, rtol=1e-4, atol=0)

    @pytest.mark.parametrize("alpha", [0.5, 1.5])
    def test_isotropic_pdf_depends_only_on_the_norm(self, alpha):
        rng = np.random.default_rng(0)
        directions = rng.normal(size=(20, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)

        pdf = isotropic_pdf(2.5 * directions, alpha)

        assert np.allclose(pdf, pdf[0], rtol=1e-12)

    @pytest.mark.parametrize("alpha", [0.3, 0.5, 0.8, 1.2, 1.5, 1.9])
    @pytest.mark.parametrize("dimensions", [2, 5])
    def test_isotropic_pdf_decreases_with_the_norm(self, alpha, dimensions):
        """alpha_power relies on this."""
        radius = np.concatenate(([0.0], np.geomspace(1e-8, 1e8, 2001)))

        pdf = isotropic_pdf(self.points_at_radius(radius, dimensions), alpha)

        assert np.all(pdf > 0)
        assert np.all(np.diff(pdf) <= 0)

    @pytest.mark.parametrize("dimensions", [2, 3, 5])
    def test_isotropic_pdf_is_continuous_in_alpha_at_one_and_two(self, dimensions):
        data = self.points_at_radius(np.geomspace(1e-3, 50, 30), dimensions)

        assert np.allclose(isotropic_pdf(data, 1 - 1e-4), isotropic_pdf(data, 1), rtol=5e-3)
        assert np.allclose(isotropic_pdf(data, 1 + 1e-4), isotropic_pdf(data, 1), rtol=5e-3)
        # The tail of alpha < 2 is a power law, so compare with the normal pdf only in its bulk.
        bulk = self.points_at_radius(np.geomspace(1e-3, 3, 30), dimensions)
        assert np.allclose(isotropic_pdf(bulk, 2 - 1e-4), isotropic_pdf(bulk, 2), rtol=5e-3)

    @pytest.mark.parametrize("dimensions", [2, 3, 5])
    def test_isotropic_entropy_is_continuous_in_alpha_at_one_and_two(self, dimensions):
        assert np.isclose(isotropic_entropy(dimensions, 1 - 1e-4), isotropic_entropy(dimensions, 1), atol=5e-3)
        assert np.isclose(isotropic_entropy(dimensions, 1 + 1e-4), isotropic_entropy(dimensions, 1), atol=5e-3)
        assert np.isclose(isotropic_entropy(dimensions, 2 - 1e-4), isotropic_entropy(dimensions, 2), atol=5e-3)

    @pytest.mark.parametrize("dimensions", [2, 3])
    def test_isotropic_entropy_decreases_with_alpha(self, dimensions):
        alphas = [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]
        entropies = [isotropic_entropy(dimensions, alpha) for alpha in alphas]

        assert np.all(np.diff(entropies) < 0)

    @pytest.mark.parametrize("alpha", [0.5, 1.5])
    def test_isotropic_entropy_matches_monte_carlo(self, alpha):
        """h = E[-ln p(X)], with X = sqrt(A) G for A positive (alpha / 2)-stable and G normal."""
        from scipy.stats import levy_stable
        n, dimensions = 200_000, 2
        rng = np.random.default_rng(0)
        # X has characteristic function exp(-|t|^alpha / alpha).
        mixing = levy_stable.rvs(
            alpha / 2, 1, loc=0, scale=np.cos(np.pi * alpha / 4) ** (2 / alpha), size=n, random_state=rng,
        )
        normal = rng.normal(scale=np.sqrt(2) * (1 / alpha) ** (1 / alpha), size=(n, dimensions))
        samples = np.sqrt(mixing)[:, None] * normal

        negative_log_pdf = -np.log(isotropic_pdf(samples, alpha))
        standard_error = np.std(negative_log_pdf) / np.sqrt(n)

        assert abs(np.mean(negative_log_pdf) - isotropic_entropy(dimensions, alpha)) < 5 * standard_error

    @pytest.mark.parametrize(("alpha", "dimensions"), [(1.5, 100), (0.5, 100), (0.2, 20), (1.9, 300)])
    def test_many_dimensions_and_small_alpha_work(self, alpha, dimensions):
        entropy = isotropic_entropy(dimensions, alpha)
        pdf = isotropic_pdf(np.ones((3, dimensions)), alpha)

        assert np.isfinite(entropy)
        # The entropy decreases with alpha, down to the normal one.
        assert entropy > isotropic_entropy(dimensions, 2)
        assert np.all(pdf > 0)

    def test_too_small_alpha_raises_value_error(self):
        """The radii of the table underflow."""
        with pytest.raises(ValueError, match="cannot be computed accurately"):
            isotropic_entropy(2, 0.005)
        with pytest.raises(ValueError, match="cannot be computed accurately"):
            isotropic_pdf(np.ones((3, 2)), 0.005)

    @pytest.mark.parametrize(
        ("alpha", "dimensions"), [(0.05, 2), (0.1, 2), (0.1, 5), (0.2, 3), (0.5, 2), (1.5, 3)]
    )
    def test_isotropic_entropy_matches_direct_integration(self, alpha, dimensions):
        """
        h = -int |S^(d-1)| r^(d-1) f_d ln f_d dr from the exact log pdf, far beyond the table.

        For a small alpha much of the mass is beyond the table, where the entropy uses the power law.
        """
        from scipy.integrate import simpson
        from scipy.special import gammaln
        from aub_htp.pdf.multivariate import _isotropic_radial_log_pdf
        gamma_scale = (1 / alpha) ** (1 / alpha)
        # Not beyond ln r = 700, where r overflows.
        s = np.linspace(-45, min(np.log(2 * gamma_scale) + 60 / alpha, 700.0), 8001)
        log_pdf = _isotropic_radial_log_pdf(np.exp(s), alpha, dimensions, gamma_scale)
        log_sphere_area = np.log(2) + dimensions / 2 * np.log(np.pi) - gammaln(dimensions / 2)
        mass_density = np.exp(log_sphere_area + dimensions * s + log_pdf)

        assert np.isclose(simpson(mass_density, x=s), 1, atol=1e-6)
        assert np.isclose(isotropic_entropy(dimensions, alpha), -simpson(mass_density * log_pdf, x=s), rtol=1e-4)


def isotropic_samples(n, dimensions, alpha, seed=0):
    """Samples of the isotropic alpha-stable law with characteristic function exp(-|t|^alpha / alpha)."""
    from scipy.stats import levy_stable
    rng = np.random.default_rng(seed)
    if alpha == 2:
        return rng.normal(size=(n, dimensions))
    # X = sqrt(A) G for A positive (alpha / 2)-stable and G normal.
    mixing = levy_stable.rvs(
        alpha / 2, 1, loc=0, scale=np.cos(np.pi * alpha / 4) ** (2 / alpha), size=n, random_state=rng,
    )
    normal = rng.normal(scale=np.sqrt(2) * (1 / alpha) ** (1 / alpha), size=(n, dimensions))
    return np.sqrt(mixing)[:, None] * normal


class TestAlphaPowerBinarySearch:

    @staticmethod
    def residual(data, power, alpha):
        """g(P) = -mean ln p(x / P) - h"""
        data = np.asarray(data, dtype=float).reshape(len(data), -1)
        return -np.mean(np.log(isotropic_pdf(data / power, alpha))) - isotropic_entropy(data.shape[1], alpha)

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5])
    @pytest.mark.parametrize("dimensions", [1, 2, 3])
    def test_alpha_power_is_the_root(self, alpha, dimensions):
        data = 3.0 * isotropic_samples(500, dimensions, alpha)

        power = alpha_power(data, alpha)

        assert abs(self.residual(data, power, alpha)) < 1e-4
        # g decreases with P.
        assert self.residual(data, 0.9 * power, alpha) > 0 > self.residual(data, 1.1 * power, alpha)

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5, 2.0])
    @pytest.mark.parametrize("dimensions", [1, 2, 5])
    def test_alpha_power_recovers_the_scale(self, alpha, dimensions):
        """The power of c X is c for X the isotropic law itself."""
        data = 7.0 * isotropic_samples(20_000, dimensions, alpha)

        assert np.isclose(alpha_power(data, alpha), 7.0, rtol=0.05)

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5])
    @pytest.mark.parametrize("scale", [1e-6, 1e-3, 1e3, 1e6])
    def test_alpha_power_scales_with_the_data(self, alpha, scale):
        """P(c X) = c P(X) at any scale, with tol set to the scale of the data."""
        data = isotropic_samples(300, 2, alpha)

        power = alpha_power(data, alpha, tol=1e-9)
        scaled_power = alpha_power(scale * data, alpha, tol=1e-9 * scale)

        assert np.isclose(scaled_power, scale * power, rtol=1e-6)

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5])
    def test_alpha_power_does_not_depend_on_the_starting_scale(self, alpha):
        """Powell started from P = 1 whatever the data's scale."""
        data = 1e5 * isotropic_samples(2000, 1, alpha)

        assert np.isclose(alpha_power(data, alpha), 1e5, rtol=0.1)

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5])
    def test_alpha_power_is_in_the_bracket(self, alpha):
        from aub_htp.statistics.statistics import _entropy_radius
        data = isotropic_samples(50, 2, alpha)
        radius = np.linalg.norm(data, axis=1)
        entropy_radius = _entropy_radius(2, float(alpha))

        power = alpha_power(data, alpha)

        assert radius.min() / entropy_radius <= power <= radius.max() / entropy_radius

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5])
    @pytest.mark.parametrize("dimensions", [1, 2, 3])
    def test_entropy_radius_is_where_negative_log_pdf_is_the_entropy(self, alpha, dimensions):
        from aub_htp.statistics.statistics import _entropy_radius
        entropy_radius = _entropy_radius(dimensions, float(alpha))
        point = np.zeros((1, dimensions))
        point[0, 0] = entropy_radius

        assert entropy_radius > 0
        assert np.isclose(-np.log(isotropic_pdf(point, alpha))[0], isotropic_entropy(dimensions, alpha), atol=1e-6)

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5])
    @pytest.mark.parametrize("dimensions", [1, 3])
    def test_alpha_power_of_one_sample_is_its_norm_over_the_entropy_radius(self, alpha, dimensions):
        """Then a = b, and g(a) = -ln p(x_h) - h = 0."""
        from aub_htp.statistics.statistics import _entropy_radius
        data = np.full((1, dimensions), 2.0)

        power = alpha_power(data, alpha)

        assert np.isclose(power, np.linalg.norm(data) / _entropy_radius(dimensions, float(alpha)))

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5])
    def test_alpha_power_with_a_zero_sample(self, alpha):
        """Then a = 0, where g is not defined. The search never evaluates g there."""
        data = np.vstack([np.zeros((1, 2)), isotropic_samples(200, 2, alpha)])

        power = alpha_power(data, alpha)

        assert np.isfinite(power) and power > 0
        assert abs(self.residual(data, power, alpha)) < 1e-4

    def test_entropy_radius_is_cached_on_the_settled_alpha(self):
        from aub_htp.statistics.statistics import _entropy_radius
        data = isotropic_samples(100, 1, 1.0)
        alpha_power(data, 1.0)
        misses = _entropy_radius.cache_info().misses

        alpha_power(data, 1.0 + 1e-9)
        alpha_power(data, np.array(1.0 - 1e-9))

        assert _entropy_radius.cache_info().misses == misses

    def test_tol_and_max_iter_are_keyword_only(self):
        data = isotropic_samples(100, 1, 1.5)

        with pytest.raises(TypeError):
            alpha_power(data, 1.5, 1e-3)

    def test_max_iter_zero_returns_the_middle_of_the_bracket(self):
        from aub_htp.statistics.statistics import _entropy_radius
        data = isotropic_samples(100, 2, 1.5)
        radius = np.linalg.norm(data, axis=1)

        power = alpha_power(data, 1.5, max_iter=0)

        assert np.isclose(power, (radius.min() + radius.max()) / 2 / _entropy_radius(2, 1.5))

    def test_smaller_tol_is_more_accurate(self):
        data = isotropic_samples(500, 2, 1.5)
        exact = alpha_power(data, 1.5, tol=1e-12, max_iter=200)

        loose = alpha_power(data, 1.5, tol=1e-1, max_iter=200)
        tight = alpha_power(data, 1.5, tol=1e-6, max_iter=200)

        assert abs(loose - exact) <= 1e-1
        assert abs(tight - exact) <= 1e-6

    def test_alpha_two_is_the_root_mean_square(self):
        data = isotropic_samples(1000, 3, 2.0)

        assert alpha_power(data, 2.0) == np.sqrt(np.mean(np.sum(data ** 2, axis=1)) / 3)
        assert alpha_power(data, 2.0 - 1e-9) == alpha_power(data, 2.0)


class TestAlphaLocationSearch:

    # Tolerances for data of scale about 1: the location is good to about sqrt(power_tol).
    TIGHT = {"tol": 1e-8, "power_tol": 1e-14, "max_iter": 100}

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5])
    @pytest.mark.parametrize("dimensions", [1, 2, 3])
    def test_alpha_location_recovers_the_center(self, alpha, dimensions):
        center = np.array([12.0, -9.0, 4.0])[:dimensions]
        data = center + 2.0 * isotropic_samples(4000, dimensions, alpha, seed=1)

        location = alpha_location(data, alpha)

        assert np.allclose(location, center, atol=0.3)

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5])
    @pytest.mark.parametrize("dimensions", [1, 2])
    def test_alpha_location_minimizes_the_power(self, alpha, dimensions):
        data = 5.0 + isotropic_samples(500, dimensions, alpha, seed=2)
        tolerances = {"tol": self.TIGHT["power_tol"], "max_iter": 100}

        location = np.atleast_1d(alpha_location(data, alpha, **self.TIGHT))
        power = alpha_power(data - location, alpha, **tolerances)

        for feature in range(dimensions):
            for step in (-1e-2, 1e-2):
                moved = location.copy()
                moved[feature] += step
                assert alpha_power(data - moved, alpha, **tolerances) > power

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5])
    def test_alpha_location_is_translation_equivariant(self, alpha):
        data = isotropic_samples(500, 2, alpha, seed=3)
        shift = np.array([200.0, -150.0])

        location = alpha_location(data, alpha, **self.TIGHT)
        shifted_location = alpha_location(data + shift, alpha, **self.TIGHT)

        assert np.allclose(shifted_location, location + shift, atol=1e-5)

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5])
    @pytest.mark.parametrize("scale", [1e-6, 1e6])
    def test_alpha_location_scales_with_the_data(self, alpha, scale):
        """L(c X) = c L(X) at any scale, with the tolerances set to the scale of the data."""
        data = 3.0 + isotropic_samples(500, 1, alpha, seed=4).ravel()
        scaled_tolerances = {"tol": 1e-8 * scale, "power_tol": 1e-14 * scale, "max_iter": 100}

        location = alpha_location(data, alpha, **self.TIGHT)
        scaled_location = alpha_location(scale * data, alpha, **scaled_tolerances)

        assert np.isclose(scaled_location / scale, location, atol=1e-5)

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5])
    def test_alpha_location_is_in_the_box_of_the_data(self, alpha):
        data = isotropic_samples(30, 3, alpha, seed=5)

        location = alpha_location(data, alpha)

        assert np.all(location >= data.min(axis=0)) and np.all(location <= data.max(axis=0))

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5, 2.0])
    def test_constant_feature_gets_its_value(self, alpha):
        data = isotropic_samples(200, 3, alpha, seed=6)
        data[:, 1] = 7.25

        location = alpha_location(data, alpha)

        assert location[1] == 7.25
        assert np.all(np.isfinite(location))

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5, 2.0])
    @pytest.mark.parametrize("n_samples", [1, 5])
    def test_constant_data_returns_that_point_without_searching(self, alpha, n_samples, monkeypatch):
        from aub_htp.statistics import statistics as statistics_module

        def fail(*args, **kwargs):
            raise AssertionError("Powell should not be called")

        monkeypatch.setattr(statistics_module, "minimize", fail)
        point = np.array([1.5, -2.0])

        assert np.array_equal(alpha_location(np.tile(point, (n_samples, 1)), alpha), point)
        assert alpha_location(np.full(n_samples, 3.0), alpha) == 3.0

    def test_alpha_two_is_the_mean(self):
        data = isotropic_samples(300, 3, 2.0, seed=7) + 4.0

        assert np.array_equal(alpha_location(data, 2.0), data.mean(axis=0))
        assert np.array_equal(alpha_location(data, 2.0 - 1e-9), data.mean(axis=0))

    def test_tolerances_are_keyword_only(self):
        with pytest.raises(TypeError):
            alpha_location(np.array([1.0, 2.0, 3.0]), 1.0, 1e-3)

    def test_smaller_power_tol_is_more_accurate(self):
        data = 5.0 + isotropic_samples(500, 1, 1.0, seed=8).ravel()
        exact = alpha_location(data, 1.0, tol=1e-10, power_tol=1e-15, max_iter=100)

        default = alpha_location(data, 1.0)
        tight = alpha_location(data, 1.0, **self.TIGHT)

        assert abs(default - exact) < 1e-2
        assert abs(tight - exact) < 1e-5

    def test_deprecated_alpha_location_is_removed(self):
        from aub_htp.statistics import statistics as statistics_module

        assert not hasattr(statistics_module, "deprecated_alpha_location")


def test_statistics_all_lists_the_public_functions():
    import aub_htp.statistics
    assert set(aub_htp.statistics.__all__) == {
        "alpha_power", "alpha_location", "isotropic_pdf", "isotropic_entropy",
    }