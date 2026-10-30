# tests/test_zolotarev_cdf.py

import numpy as np
import pytest
from scipy.special import ndtr

from aub_htp.cdf.zolotarev import generate_cdf, generate_cdf_one_point


@pytest.mark.parametrize(
    # Reference values are from scipy.stats.levy_stable.cdf(x, alpha, beta)

    "alpha,beta,x,expected",
    [
        # Incorrect additive constant in the original alpha < 1 branch.
        # Reference: levy_stable.cdf(1.0, 0.5, 0.5)
        (0.5, 0.5, 1.0, 0.5399839777834092),

        # Incorrect exponential factor in the original alpha = 1 branch.
        # Reference: levy_stable.cdf(x, 1.0, 0.5)
        (1.0, 0.5, 0.0, 0.4375114838590879),
        (1.0, 0.5, 1.0, 0.6635450982516822),

        # Zero raised to a negative power in the original implementation.
        # Reference: closed form F(0) = 1/2 - theta0/pi
        (0.5, 0.5, 0.0, 0.20483276469913347),

        # Overflow in the original alpha = 1 kernel.
        # Reference: levy_stable.cdf(10.0, 1.0, 0.01)
        (1.0, 0.01, 10.0, 0.9679210693916145),
    ],
)
def test_pointwise_values(alpha, beta, x, expected):
    actual = generate_cdf_one_point(x, alpha, beta)

    assert np.isfinite(actual)
    assert actual == pytest.approx(expected, abs=1e-8)


def test_gaussian_near_zero():
    # S1(alpha=2, scale=1) is Normal(0, variance=2).
    x = 1e-4
    expected = ndtr(x / np.sqrt(2))

    assert generate_cdf_one_point(x, 2.0, 0.0) == pytest.approx(
        expected, abs=1e-12
    )


def test_cauchy_not_implemented():
    # alpha = 1, beta = 0 is handled by the other methods; the switching
    # table must never route it to Zolotarev.
    with pytest.raises(NotImplementedError):
        generate_cdf_one_point(0.0, 1.0, 0.0)



def test_one_sided_support():
    assert generate_cdf_one_point(1.0, 0.5, -1.0) == 1.0
    assert generate_cdf_one_point(-1.0, 0.5, 1.0) == 0.0


@pytest.mark.parametrize("alpha,beta", [(0.5, 0.5), (1.0, 0.5), (1.8, 0.8)])
def test_reflection(alpha, beta):
    positive = generate_cdf_one_point(1.0, alpha, beta)
    reflected = generate_cdf_one_point(-1.0, alpha, -beta)

    assert positive + reflected == pytest.approx(1.0, abs=1e-12)


# Inputs where the CDF came back NaN or raised OverflowError instead of a
# probability.
NAN_CASES = [
    # alpha < 1, small x: x ** (alpha / (alpha - 1)) overflows.
    (0.99, 0.0, 1e-8),
    (0.99, 0.5, 1e-4),

    # alpha < 1, huge x: NaN.
    (0.99, -0.5, 1e5),
    (0.99, -0.5, 1e8),

    # alpha > 1, huge x: x ** (alpha / (alpha - 1)) overflows.
    (1.01, 0.5, 1e5),
    (1.01, 0.0, 1e8),
    (1.01, -1.0, 1e5),

    # alpha = 1: the log-space kernel must stay finite for extreme x / beta.
    (1.0, 0.01, 1e3),
    (1.0, 0.5, -1e3),
    (1.0, 1.0, 1e5),
]


@pytest.mark.parametrize("alpha,beta,x", NAN_CASES)
def test_finite_probability(alpha, beta, x):
    actual = generate_cdf_one_point(x, alpha, beta)

    assert np.isfinite(actual)
    assert -1e-8 <= actual <= 1 + 1e-8


@pytest.mark.parametrize("alpha,beta,x", NAN_CASES)
def test_finite_probability_vectorized(alpha, beta, x):
    # generate_cdf passes np.float64 values, which overflow to inf instead of
    # raising OverflowError.
    actual = generate_cdf([x], alpha, beta)

    assert np.all(np.isfinite(actual))
    assert np.all((actual >= -1e-8) & (actual <= 1 + 1e-8))

@pytest.mark.parametrize(
    # Reference values are from scipy.stats.levy_stable.cdf(x, alpha, beta).
    # alpha stays away from 1: levy_stable.cdf is flat in x at alpha = 0.99
    # and 1.01. Every value below agrees with this implementation to 1e-11.

    "alpha,beta,x,expected",
    [
        # alpha < 1
        (0.3, 0.5, -2.0, 0.120808304614762),
        (0.3, -0.7, 0.5, 0.9089782995185993),
        (0.3, 0.0, 5.0, 0.7946366433555807),
        (0.5, 0.5, -0.5, 0.1416617160843262),
        (0.5, -0.9, 2.0, 0.9822659063232718),
        (0.5, 1.0, 0.2, 0.02534731867746834),
        (0.8, 0.0, 1.0, 0.744140237907118),
        (0.8, 0.5, -3.0, 0.04590991393116284),
        (0.8, -0.7, 0.3, 0.956892773573848),
        (0.8, 0.0, 50.0, 0.984748872480256),

        # alpha > 1
        (1.3, 0.0, -1.0, 0.24548475760060118),
        (1.3, 0.5, 2.0, 0.8912732599053111),
        (1.3, -1.0, 0.5, 0.29268931390905273),
        (1.5, 0.8, -2.0, 0.1383998211603702),
        (1.5, 0.0, 4.0, 0.9694245437885906),
        (1.5, -0.3, 0.1, 0.46585815853186174),
        (1.5, 0.0, -30.0, 0.001225830277318929),
        (1.7, 0.5, -1.0, 0.27378568247588264),
        (1.7, 0.0, 0.5, 0.638497071881589),
        (1.7, -0.9, 3.0, 0.9814812056424024),
        (1.9, 0.5, 1.0, 0.766615015713188),
        (1.9, 0.0, -2.0, 0.08296397263312405),
        (1.9, 1.0, 0.3, 0.6084079199026384),
    ],
)
def test_scipy_values(alpha, beta, x, expected):
    actual = generate_cdf_one_point(x, alpha, beta)

    assert np.isfinite(actual)
    assert actual == pytest.approx(expected, abs=1e-8)
