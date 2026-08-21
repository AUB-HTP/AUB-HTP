import numpy as np
import pytest
from numpy.testing import assert_allclose

from aub_htp import multivariate_alpha_stable
from aub_htp.random import EllipticSampler, IsotropicSampler


def _elliptic_sampler(dimensions=2):
    sigma = np.diag(np.arange(1, dimensions + 1))
    return EllipticSampler(dimensions, 1.5, sigma, mass=1)


def test_elliptical_default_size_returns_one_vector():
    sample = multivariate_alpha_stable.rvs(
        1.5,
        _elliptic_sampler(),
        random_state=123,
    )

    assert sample.shape == (2,)


def test_elliptical_shift_is_applied():
    shift = np.array([10.0, -4.0])
    unshifted = multivariate_alpha_stable.rvs(
        1.5,
        _elliptic_sampler(),
        size=5,
        random_state=123,
    )
    shifted = multivariate_alpha_stable.rvs(
        1.5,
        _elliptic_sampler(),
        shift=shift,
        size=5,
        random_state=123,
    )

    assert_allclose(shifted, unshifted + shift)


class TestEllipticalDimensions:
    @pytest.mark.parametrize("dimensions", [1, 2, 3, 5])
    def test_default_size_returns_vector_matching_dimensions(self, dimensions):
        sample = multivariate_alpha_stable.rvs(
            1.5,
            _elliptic_sampler(dimensions),
            random_state=123,
        )

        assert sample.shape == (dimensions,)

    @pytest.mark.parametrize("dimensions", [1, 2, 3, 5])
    @pytest.mark.parametrize("size", [1, 2, 7])
    def test_batch_shape_matches_size_and_dimensions(self, dimensions, size):
        samples = multivariate_alpha_stable.rvs(
            1.5,
            _elliptic_sampler(dimensions),
            size=size,
            random_state=123,
        )

        assert samples.shape == (size, dimensions)

    @pytest.mark.parametrize("dimensions", [1, 2, 3, 5])
    def test_vector_shift_preserves_batch_dimensions(self, dimensions):
        shift = np.arange(dimensions, dtype=float)
        samples = multivariate_alpha_stable.rvs(
            1.5,
            _elliptic_sampler(dimensions),
            shift=shift,
            size=4,
            random_state=123,
        )

        assert samples.shape == (4, dimensions)

    def test_incompatible_shift_dimensions_raise_value_error(self):
        with pytest.raises(ValueError):
            multivariate_alpha_stable.rvs(
                1.5,
                _elliptic_sampler(3),
                shift=np.zeros(2),
                size=4,
                random_state=123,
            )


class TestIsotropicAsElliptical:
    @pytest.mark.parametrize("dimensions", [1, 2, 3, 5])
    @pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0])
    def test_matches_spherical_elliptical_sampler(self, dimensions, gamma):
        isotropic = IsotropicSampler(dimensions, alpha=1.5, gamma=gamma)
        elliptical = EllipticSampler(
            dimensions,
            alpha=1.5,
            sigma=2 * gamma**2 * np.eye(dimensions),
            mass=1,
        )

        isotropic_samples = multivariate_alpha_stable.rvs(
            1.5,
            isotropic,
            size=7,
            random_state=123,
        )
        elliptical_samples = multivariate_alpha_stable.rvs(
            1.5,
            elliptical,
            size=7,
            random_state=123,
        )

        assert_allclose(isotropic_samples, elliptical_samples)

    @pytest.mark.parametrize("dimensions", [1, 2, 3, 5])
    def test_default_size_returns_vector_matching_dimensions(self, dimensions):
        sample = multivariate_alpha_stable.rvs(
            1.5,
            IsotropicSampler(dimensions, alpha=1.5, gamma=1),
            random_state=123,
        )

        assert sample.shape == (dimensions,)

    @pytest.mark.parametrize("dimensions", [1, 2, 3, 5])
    @pytest.mark.parametrize("size", [1, 2, 7])
    def test_batch_shape_matches_size_and_dimensions(self, dimensions, size):
        samples = multivariate_alpha_stable.rvs(
            1.5,
            IsotropicSampler(dimensions, alpha=1.5, gamma=1),
            size=size,
            random_state=123,
        )

        assert samples.shape == (size, dimensions)

    def test_shift_matches_equivalent_spherical_elliptical_sampler(self):
        shift = np.array([10.0, -4.0, 2.0])
        isotropic_samples = multivariate_alpha_stable.rvs(
            1.5,
            IsotropicSampler(3, alpha=1.5, gamma=2),
            shift=shift,
            size=5,
            random_state=123,
        )
        elliptical_samples = multivariate_alpha_stable.rvs(
            1.5,
            EllipticSampler(3, alpha=1.5, sigma=8 * np.eye(3), mass=1),
            shift=shift,
            size=5,
            random_state=123,
        )

        assert_allclose(isotropic_samples, elliptical_samples)
