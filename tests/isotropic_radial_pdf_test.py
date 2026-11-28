import numpy as np
import pytest
from scipy.special import gamma

from aub_htp.pdf.multivariate import (
    _isotropic_radial_log_mixture,
    _isotropic_radial_log_pdf,
    _isotropic_radial_log_series,
    _isotropic_radial_pdf,
    _isotropic_series_radius,
)


def cauchy_pdf(radius, d, gamma_scale):
    coefficient = gamma((d + 1) / 2) / (np.pi ** ((d + 1) / 2) * gamma_scale ** d)
    return coefficient * (1 + (radius / gamma_scale) ** 2) ** (-(d + 1) / 2)


def normal_pdf(radius, d, gamma_scale):
    # exp(-gamma^2 |t|^2) is N(0, 2 gamma^2 I).
    variance = 2 * gamma_scale ** 2
    return (2 * np.pi * variance) ** (-d / 2) * np.exp(-radius ** 2 / (2 * variance))


RADIUS = np.concatenate(([0.0], np.geomspace(1e-6, 1e8, 57)))


class TestClosedForms:

    @pytest.mark.parametrize("d", [1, 2, 3, 10])
    @pytest.mark.parametrize("gamma_scale", [0.5, 1.0, 3.0])
    def test_alpha_one_is_cauchy(self, d, gamma_scale):
        pdf = _isotropic_radial_pdf(RADIUS, 1.0, d, gamma_scale)

        assert np.allclose(pdf, cauchy_pdf(RADIUS, d, gamma_scale), rtol=1e-12, atol=0)

    @pytest.mark.parametrize("d", [1, 2, 3, 10])
    @pytest.mark.parametrize("gamma_scale", [0.5, 1.0, 3.0])
    def test_alpha_two_is_normal(self, d, gamma_scale):
        pdf = _isotropic_radial_pdf(RADIUS, 2.0, d, gamma_scale)

        assert np.allclose(pdf, normal_pdf(RADIUS, d, gamma_scale), rtol=1e-12, atol=0)

    @pytest.mark.parametrize("d", [2, 3, 5])
    def test_alpha_close_to_one_is_close_to_cauchy(self, d):
        """The transform and the series, which alpha = 1 itself does not use, against the formula."""
        for alpha in (1 - 1e-5, 1 + 1e-5):
            pdf = _isotropic_radial_pdf(RADIUS, alpha, d, 1.0)

            assert np.allclose(pdf, cauchy_pdf(RADIUS, d, 1.0), rtol=1e-3, atol=0)


class TestLargeRadius:

    @pytest.mark.parametrize("alpha", [0.5, 0.8, 1.5, 1.9])
    @pytest.mark.parametrize("d", [2, 3])
    def test_large_radius_is_positive_and_decreasing(self, alpha, d):
        radius = np.geomspace(1.0, 1e12, 200)

        pdf = _isotropic_radial_pdf(radius, alpha, d, 1.0)

        assert np.all(pdf > 0)
        assert np.all(np.diff(pdf) < 0)

    @pytest.mark.parametrize("alpha", [0.5, 0.8, 1.5, 1.9])
    @pytest.mark.parametrize("d", [2, 3])
    def test_tail_is_the_power_law(self, alpha, d):
        """f_d(r) r^(d + alpha) tends to the constant of the first term of the series."""
        from scipy.special import gamma as gamma_function
        constant = (
            2 ** alpha * np.sin(np.pi * alpha / 2) * gamma_function(alpha / 2 + 1)
            * gamma_function((alpha + d) / 2) / np.pi ** (d / 2 + 1)
        )

        pdf = _isotropic_radial_pdf(np.array([1e10]), alpha, d, 1.0)

        assert np.isclose(pdf[0] * 1e10 ** (d + alpha), constant, rtol=1e-3)

    @pytest.mark.parametrize("alpha", [0.5, 0.8, 1.5, 1.9])
    @pytest.mark.parametrize("d", [2, 3])
    def test_series_and_transform_agree_where_they_meet(self, alpha, d):
        series_radius = _isotropic_series_radius(alpha, d, 1.0)
        radius = series_radius * np.array([0.9, 0.99, 1.01, 1.1])

        pdf = _isotropic_radial_pdf(radius, alpha, d, 1.0)
        log_series, _ = _isotropic_radial_log_series(radius, alpha, d, 1.0)

        assert np.allclose(pdf[2:], np.exp(log_series[2:]), rtol=1e-12)
        # The pdf is smooth across the switch.
        slopes = np.diff(np.log(pdf)) / np.diff(np.log(radius))
        assert np.allclose(slopes, slopes[0], rtol=0.2)

    @pytest.mark.parametrize("alpha", [1.95, 1.99, 1.999, 1.9999])
    @pytest.mark.parametrize("d", [1, 2, 3])
    def test_mixture_matches_the_series_for_alpha_close_to_two(self, alpha, d):
        """Past the normal core the density is all tail, which comes from a large A."""
        radius = _isotropic_series_radius(alpha, d, 1.0) * np.array([1.0, 1.1, 1.3])

        log_mixture = _isotropic_radial_log_mixture(radius, alpha, d, 1.0)
        log_series, _ = _isotropic_radial_log_series(radius, alpha, d, 1.0)

        assert np.allclose(log_mixture, log_series, rtol=0, atol=1e-7)

    def test_scale(self):
        """f(r; gamma) = f(r / gamma; 1) / gamma^d"""
        radius = np.geomspace(1e-3, 1e6, 40)

        pdf = _isotropic_radial_pdf(radius, 1.5, 3, 2.5)
        unit_pdf = _isotropic_radial_pdf(radius / 2.5, 1.5, 3, 1.0)

        assert np.allclose(pdf, unit_pdf / 2.5 ** 3, rtol=1e-6)


class TestRobustness:

    def test_origin_does_not_overflow_in_many_dimensions(self):
        pdf = _isotropic_radial_pdf(0.0, 0.5, 200, 1.0)

        assert np.isfinite(pdf[0]) or pdf[0] == np.inf
        assert not np.isnan(pdf[0])

    def test_small_alpha_uses_the_mixture(self):
        """The transform would need too many panels here. It used to run out of memory."""
        alpha = 0.1
        gamma_scale = (1 / alpha) ** (1 / alpha)
        radius = _isotropic_series_radius(alpha, 2, gamma_scale) * np.array([0.5, 2.0])

        log_pdf = _isotropic_radial_log_pdf(radius, alpha, 2, gamma_scale)
        log_series, _ = _isotropic_radial_log_series(radius, alpha, 2, gamma_scale)

        # At half the radius of the series, the series is still good to about 1e-3.
        assert np.allclose(log_pdf, log_series, atol=1e-2)

    @pytest.mark.parametrize("d", [2, 10, 50, 200])
    def test_mixture_is_cauchy_for_alpha_one(self, d):
        """The mixture of normals against the formula, in few and in many dimensions."""
        radius = np.concatenate(([0.0], np.geomspace(1e-3, 50, 20)))

        log_pdf = _isotropic_radial_log_mixture(radius, 1.0, d, 1.0)
        # alpha = 1 uses the Cauchy formula. In logs, since the pdf underflows in 200 dimensions.
        expected = _isotropic_radial_log_pdf(radius, 1.0, d, 1.0)

        assert np.allclose(log_pdf, expected, atol=1e-4)

    @pytest.mark.parametrize("alpha", [0.5, 1.5])
    def test_mixture_matches_the_univariate_pdf_in_one_dimension(self, alpha):
        import aub_htp as ht
        radius = np.geomspace(1e-3, 20, 12)

        log_mixture = _isotropic_radial_log_mixture(radius, alpha, 1, 1.0)
        expected = ht.alpha_stable.pdf(radius, alpha, 0, loc=0, scale=1.0)

        assert np.allclose(log_mixture, np.log(expected), atol=2e-3)

    def test_keeps_the_shape_and_order_of_the_input(self):
        radius = np.array([5.0, 0.0, 1e9, 0.3, 5.0])

        pdf = _isotropic_radial_pdf(radius, 1.5, 2, 1.0)

        assert pdf.shape == radius.shape
        assert pdf[0] == pdf[4]
        assert pdf[1] > pdf[3] > pdf[0] > pdf[2] > 0
