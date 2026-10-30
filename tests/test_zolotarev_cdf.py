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