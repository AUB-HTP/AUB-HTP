"""
Unit tests for the hybrid alpha-stable CDF wrapper.

The wrapper dispatches each x to whichever of three methods the offline
method-selection search found most accurate for that (alpha, beta, x):

- 'Z' Zolotarev closed form
- 'I' numerical integration of the pdf wrapper
- 'S' Skorohod tail series with optimal truncation

Covered here:
- dimension handling (scalar, 1D, list, empty)
- the distribution-function properties: range, monotonicity, limits, centre
- the exact identities: symmetry at beta=0, beta-reflection
- tail decay exponent, which should recover -alpha
- agreement with scipy where scipy is itself reliable
- the switching table loads and covers the positive axis
"""

import pytest
import numpy as np
from scipy.stats import levy_stable

from aub_htp import alpha_stable
from aub_htp.cdf import generate_alpha_stable_cdf
from aub_htp.cdf.cdf import describe_intervals, generate_switching_intervals


# (alpha, beta) pairs spanning both sides of alpha = 1 and a range of skew.
CONFIGURATIONS = [
    (0.20, 0.50),
    (0.35, -0.80),
    (0.50, 0.50),
    (0.80, -0.50),
    (1.20, 0.50),
    (1.50, 0.00),
    (1.80, 0.80),
]

# Worst observed disagreement with scipy over these configurations is 6.8e-5,
# in the numerically-integrated body for alpha < 1; the median is machine
# precision. 2e-4 leaves headroom without making the check vacuous.
SCIPY_TOLERANCE = 2e-4


class TestCdfDimensions:
    """Test dimension handling."""

    def test_scalar_x(self):
        """Scalar x -> scalar output."""
        result = generate_alpha_stable_cdf(0.0, 1.5, 0.0)
        assert np.isscalar(result) or np.asarray(result).shape == ()

    def test_1d_x(self):
        """1D x array -> 1D output with same length."""
        x = np.linspace(-2, 2, 10)
        result = generate_alpha_stable_cdf(x, 1.5, 0.0)
        assert result.shape == (10,)

    def test_list_x(self):
        """List x (auto-converted) -> 1D output."""
        result = generate_alpha_stable_cdf([-1.0, 0.0, 1.0], 1.5, 0.0)
        assert result.shape == (3,)

    def test_single_element_x(self):
        """1D x with a single element -> shape (1,)."""
        result = generate_alpha_stable_cdf(np.array([0.0]), 1.5, 0.0)
        assert result.shape == (1,)

    def test_empty_x(self):
        """Empty x array -> empty output."""
        result = generate_alpha_stable_cdf(np.array([]), 1.5, 0.0)
        assert result.shape == (0,)

    def test_output_dtype_is_float(self):
        """Output dtype should be floating point."""
        result = generate_alpha_stable_cdf(np.linspace(-1, 1, 5), 1.5, 0.0)
        assert np.issubdtype(result.dtype, np.floating)


class TestCdfRange:
    """The CDF must stay inside [0, 1] and be finite."""

    @pytest.mark.parametrize("alpha,beta", CONFIGURATIONS)
    def test_within_unit_interval(self, alpha, beta):
        """0 <= F(x) <= 1 across a wide grid."""
        x = np.linspace(-100, 100, 300)
        F = generate_alpha_stable_cdf(x, alpha, beta)
        assert np.all(F >= 0.0) and np.all(F <= 1.0)

    @pytest.mark.parametrize("alpha,beta", CONFIGURATIONS)
    def test_finite(self, alpha, beta):
        """No nan or inf leaks out of the method dispatch."""
        x = np.linspace(-100, 100, 300)
        F = generate_alpha_stable_cdf(x, alpha, beta)
        assert np.all(np.isfinite(F))


class TestCdfMonotonicity:
    """A CDF is non-decreasing."""

    @pytest.mark.parametrize("alpha,beta", CONFIGURATIONS)
    def test_non_decreasing(self, alpha, beta):
        """F is non-decreasing across the switching points."""
        x = np.linspace(-100, 100, 300)
        F = generate_alpha_stable_cdf(x, alpha, beta)
        assert np.all(np.diff(F) >= -1e-10)

    @pytest.mark.parametrize("alpha,beta", CONFIGURATIONS)
    def test_non_decreasing_without_repair(self, alpha, beta):
        """
        The raw hybrid is already monotone, so the repair is not doing the work.

        This is the check that keeps enforce_monotone honest: with the repair
        switched off, a genuine discontinuity at a switch point would show up
        here as a negative difference.
        """
        x = np.linspace(-100, 100, 300)
        F = generate_alpha_stable_cdf(
            x, alpha, beta, clip=False, enforce_monotone=False
        )
        assert np.all(np.diff(F) >= -1e-10)


class TestCdfLimits:
    """F -> 0 and F -> 1 in the tails."""

    @pytest.mark.parametrize("alpha,beta", CONFIGURATIONS)
    def test_limits(self, alpha, beta):
        """
        F(-L) ~ 0 and F(L) ~ 1 at an alpha-appropriate probe point.

        The probe has to scale with alpha: a stable tail decays like x**-alpha,
        so at alpha=0.2 the true tail mass at x=1e6 is still 0.042 and a fixed
        probe would be asserting a wrong value rather than a limit. 10**(8/alpha)
        puts the residual tail mass near 1e-8 for every alpha.
        """
        limit = 10.0 ** (8.0 / alpha)
        F = generate_alpha_stable_cdf(np.array([-limit, limit]), alpha, beta)
        assert F[0] < 1e-5
        assert F[1] > 1 - 1e-5


class TestCdfCentre:
    """Known value at the centre of the symmetric case."""

    @pytest.mark.parametrize("alpha", [0.5, 1.2, 1.5, 1.8])
    def test_symmetric_centre_is_one_half(self, alpha):
        """F(0) = 1/2 exactly when beta = 0."""
        F = generate_alpha_stable_cdf(np.array([0.0]), alpha, 0.0)
        assert abs(F[0] - 0.5) < 1e-8


class TestCdfIdentities:
    """Exact identities the wrapper is built to preserve."""

    @pytest.mark.parametrize("alpha", [0.5, 0.8, 1.2, 1.5, 1.8])
    def test_symmetry_at_zero_skew(self, alpha):
        """F(-x) = 1 - F(x) when beta = 0."""
        x = np.linspace(0, 20, 121)
        F_positive = generate_alpha_stable_cdf(x, alpha, 0.0)
        F_negative = generate_alpha_stable_cdf(-x, alpha, 0.0)
        assert np.max(np.abs(F_negative - (1.0 - F_positive))) < 1e-6

    @pytest.mark.parametrize("alpha,beta", CONFIGURATIONS)
    def test_beta_reflection(self, alpha, beta):
        """
        F(x; alpha, beta) = 1 - F(-x; alpha, -beta).

        The wrapper evaluates only x >= 0 and reflects, so this should hold to
        machine precision rather than approximately.
        """
        x = np.linspace(-20, 20, 121)
        F1 = generate_alpha_stable_cdf(x, alpha, beta)
        F2 = generate_alpha_stable_cdf(-x, alpha, -beta)
        assert np.max(np.abs(F1 - (1.0 - F2))) < 1e-6


class TestCdfTail:
    """The tail must decay with the right power."""

    @pytest.mark.parametrize("alpha,beta", CONFIGURATIONS)
    def test_tail_exponent(self, alpha, beta):
        """
        1 - F(x) ~ x**-alpha, so the log-log slope recovers -alpha.

        This is the check that fails if the tail method saturates to 1.0: the
        survival function collapses and the fitted slope runs away.
        """
        x = np.logspace(2, 6, 120)
        survival = np.maximum(1.0 - generate_alpha_stable_cdf(x, alpha, beta), 1e-300)
        slope = np.polyfit(np.log(x), np.log(survival), 1)[0]
        assert abs(slope + alpha) < 0.1


class TestCdfAgainstScipy:
    """Agreement with scipy over the range where scipy is itself reliable."""

    @pytest.mark.parametrize("alpha,beta", CONFIGURATIONS)
    def test_matches_scipy(self, alpha, beta):
        """|ours - scipy| stays within tolerance on a log-spaced grid."""
        x = np.concatenate([
            -np.logspace(-2, 2, 40)[::-1],
            [0.0],
            np.logspace(-2, 2, 40),
        ])
        ours = generate_alpha_stable_cdf(x, alpha, beta)
        reference = levy_stable.cdf(x, alpha, beta)
        assert np.max(np.abs(ours - reference)) < SCIPY_TOLERANCE

    def test_gaussian_limit(self):
        """alpha = 2 is the normal law with scale sqrt(2)."""
        from scipy.stats import norm

        x = np.linspace(-10, 10, 121)
        ours = generate_alpha_stable_cdf(x, 2.0, 0.0)
        assert np.max(np.abs(ours - norm.cdf(x, scale=np.sqrt(2)))) < 1e-6

    def test_cauchy_limit(self):
        """alpha = 1, beta = 0 is the Cauchy law."""
        from scipy.stats import cauchy

        x = np.linspace(-10, 10, 121)
        ours = generate_alpha_stable_cdf(x, 1.0, 0.0)
        assert np.max(np.abs(ours - cauchy.cdf(x))) < 1e-6


class TestCdfLocationScale:
    """gamma and delta shift and scale the argument, not the value."""

    @pytest.mark.parametrize("alpha,beta", [(0.8, -0.5), (1.5, 0.0), (1.8, 0.8)])
    def test_matches_scipy_with_location_and_scale(self, alpha, beta):
        """F(x; gamma, delta) equals scipy's loc/scale form."""
        x = np.linspace(-20, 20, 61)
        ours = generate_alpha_stable_cdf(x, alpha, beta, 2.0, 1.0)
        reference = levy_stable.cdf(x, alpha, beta, loc=1.0, scale=2.0)
        assert np.max(np.abs(ours - reference)) < SCIPY_TOLERANCE


class TestSwitchingTable:
    """The tabulated method selection must load and cover the axis."""

    @pytest.mark.parametrize("alpha,beta", CONFIGURATIONS)
    def test_intervals_cover_positive_axis(self, alpha, beta):
        """Intervals are contiguous, ordered, and reach +inf."""
        intervals = generate_switching_intervals(alpha, beta)
        assert len(intervals) >= 1
        assert intervals[-1][1] == np.inf
        for (_, previous_hi, _), (next_lo, _, _) in zip(intervals, intervals[1:]):
            assert previous_hi == next_lo

    @pytest.mark.parametrize("alpha,beta", CONFIGURATIONS)
    def test_methods_are_known(self, alpha, beta):
        """Every interval names one of the three implemented methods."""
        for _, _, method in generate_switching_intervals(alpha, beta):
            assert method in (0, 1, 2)

    def test_tail_interval_uses_series(self):
        """The unbounded interval is always the asymptotic tail series."""
        for alpha, beta in CONFIGURATIONS:
            assert generate_switching_intervals(alpha, beta)[-1][2] == 2

    def test_describe_is_readable(self):
        """describe_intervals returns a non-empty summary string."""
        assert "->" in describe_intervals(1.5, 0.0)


class TestCdfParameterValidation:
    """Out-of-domain parameters are rejected."""

    @pytest.mark.parametrize("alpha", [-0.5, 0.0, 2.5])
    def test_invalid_alpha_raises(self, alpha):
        """alpha must satisfy 0 < alpha <= 2."""
        with pytest.raises(Exception):
            generate_alpha_stable_cdf(np.array([0.0]), alpha, 0.0)

    @pytest.mark.parametrize("beta", [-1.5, 1.5])
    def test_invalid_beta_raises(self, beta):
        """beta must satisfy -1 <= beta <= 1."""
        with pytest.raises(Exception):
            generate_alpha_stable_cdf(np.array([0.0]), 1.5, beta)


@pytest.fixture(autouse=True)
def _restore_parameterization():
    """
    alpha_stable is a module-level singleton and with_parametrization mutates it,
    so a test that switches to S0 would otherwise leak into every test after it.
    """
    yield
    alpha_stable.parameterization = "S1"


class TestFrontendCdfDimensions:
    """alpha_stable.cdf follows the same broadcasting conventions as .pdf."""

    def test_scalar_x(self):
        """Scalar x -> scalar output."""
        result = alpha_stable.cdf(0.0, alpha=1.5, beta=0.0)
        assert np.isscalar(result) or np.asarray(result).shape == ()

    def test_1d_x(self):
        """1D x -> 1D output of the same length."""
        assert alpha_stable.cdf(np.linspace(-2, 2, 10), 1.5, 0.0).shape == (10,)

    def test_list_x(self):
        """List x is accepted."""
        assert alpha_stable.cdf([-1.0, 0.0, 1.0], 1.5, 0.0).shape == (3,)

    def test_empty_x(self):
        """Empty x -> empty output."""
        assert alpha_stable.cdf(np.array([]), 1.5, 0.0).shape == (0,)

    def test_array_alpha(self):
        """Array alpha broadcasts."""
        assert alpha_stable.cdf(0.0, alpha=np.array([1.2, 1.5, 1.8]), beta=0.0).shape == (3,)

    def test_array_beta(self):
        """Array beta broadcasts."""
        assert alpha_stable.cdf(0.0, alpha=1.5, beta=np.array([-0.5, 0.0, 0.5])).shape == (3,)

    def test_array_loc_scale(self):
        """Array loc and scale broadcast."""
        result = alpha_stable.cdf(
            np.array([0.0, 1.0]), 1.5, 0.0,
            loc=np.array([0.0, 1.0]), scale=np.array([1.0, 2.0]),
        )
        assert result.shape == (2,)


class TestFrontendCdfValues:
    """The frontend must agree with the wrapper it delegates to."""

    @pytest.mark.parametrize("alpha,beta", CONFIGURATIONS)
    def test_matches_wrapper_at_unit_scale(self, alpha, beta):
        """With loc=0, scale=1 the frontend equals generate_alpha_stable_cdf."""
        x = np.linspace(-20, 20, 41)
        assert np.max(np.abs(
            alpha_stable.cdf(x, alpha, beta) - generate_alpha_stable_cdf(x, alpha, beta)
        )) < 1e-12

    @pytest.mark.parametrize("alpha,beta", [(0.8, -0.5), (1.5, 0.0), (1.8, 0.8)])
    def test_matches_scipy_with_loc_scale(self, alpha, beta):
        """loc/scale behave as scipy's do."""
        x = np.linspace(-20, 20, 41)
        ours = alpha_stable.cdf(x, alpha, beta, loc=1.0, scale=2.0)
        reference = levy_stable.cdf(x, alpha, beta, loc=1.0, scale=2.0)
        assert np.max(np.abs(ours - reference)) < SCIPY_TOLERANCE

    def test_survival_is_one_minus_cdf(self):
        """sf now routes through _cdf and must stay consistent with it."""
        x = np.linspace(-10, 10, 21)
        assert np.max(np.abs(
            alpha_stable.sf(x, 1.5, 0.0) - (1.0 - alpha_stable.cdf(x, 1.5, 0.0))
        )) < 1e-12

    @pytest.mark.parametrize("q", [0.1, 0.25, 0.5, 0.75, 0.9])
    def test_quantile_round_trip(self, q):
        """ppf is inverted from _cdf, so cdf(ppf(q)) must return q."""
        assert abs(alpha_stable.cdf(alpha_stable.ppf(q, 1.5, 0.0), 1.5, 0.0) - q) < 1e-6


class TestFrontendCdfParameterization:
    """S0 and S1 must be handled the way .pdf handles them."""

    @pytest.mark.parametrize("alpha,beta", [(1.5, 0.5), (0.5, 0.5), (1.8, -0.8)])
    def test_s0_is_s1_shifted(self, alpha, beta):
        """
        For alpha != 1 the two conventions differ by beta*tan(pi*alpha/2):
        F_S0(x) = F_S1(x + beta*tan(pi*alpha/2)).
        """
        x = np.linspace(-10, 10, 21)
        shift = beta * np.tan(np.pi * alpha / 2.0)

        s0 = alpha_stable.with_parametrization("S0").cdf(x, alpha, beta)
        s1 = alpha_stable.with_parametrization("S1").cdf(x + shift, alpha, beta)

        assert np.max(np.abs(s0 - s1)) < 1e-12

    @pytest.mark.parametrize("alpha", [0.5, 1.5, 1.8])
    def test_conventions_agree_at_zero_skew(self, alpha):
        """At beta = 0 there is no shift, so S0 and S1 coincide."""
        x = np.linspace(-10, 10, 21)
        s0 = alpha_stable.with_parametrization("S0").cdf(x, alpha, 0.0)
        s1 = alpha_stable.with_parametrization("S1").cdf(x, alpha, 0.0)
        assert np.max(np.abs(s0 - s1)) < 1e-12

    def test_invalid_parameterization_raises(self):
        """Only S0 and S1 are accepted."""
        with pytest.raises(ValueError):
            alpha_stable.with_parametrization("S2")
