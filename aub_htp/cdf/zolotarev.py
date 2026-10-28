import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from aub_htp.pdf.zolotarev import theta0_stable, calculate_V

def transition_points(log_g, lower, upper):
    """
    Breakpoints for quad where the integrand exp(-g) falls between 1 and 0.
    - g is monotone on (lower, upper), so each level of log g is crossed once.
    - Levels -36, -4, 0, 4 take exp(-g) from 1 - 2e-16 down to 2e-24. Without them
      quad can step over a narrow transition and report a tiny error.
    - The bracket stays 1e-12 inside the interval: closer than that the
      kernel is rounding noise, and the points only need to be approximate.
    """
    if upper - lower < 1e-11:
        return None

    a, b = lower + 1e-12, upper - 1e-12

    with np.errstate(all="ignore"):
        log_g_a, log_g_b = log_g(a), log_g(b)

        if not (np.isfinite(log_g_a) and np.isfinite(log_g_b)):
            return None

        points = [
            brentq(lambda theta: log_g(theta) - level, a, b)
            for level in (-36.0, -4.0, 0.0, 4.0)
            if (log_g_a - level) * (log_g_b - level) < 0
        ]

    return points or None


def generate_cdf_one_point(x, alpha, beta):
    """
    Zolotarev CDF (Chapter 4.5).

    Implements equations

        (4.5.2)  alpha < 1
        (4.5.3)  alpha > 1
        (4.5.4)  alpha = 1

    from Zolotarev.
    """

    # ---------------------------------------------
    # α = 1
    # ---------------------------------------------
    if alpha == 1:

        if beta == 0:
            raise NotImplementedError("α = 1, β = 0 (Cauchy) not implemented.")

        # Reflection identity
        if beta < 0:
            return 1.0 - generate_cdf_one_point(-x, 1, -beta)

        theta0 = np.pi / 2
        U = calculate_V(1, beta, theta0)

        integrand = lambda theta: np.exp(
            -np.exp(-x / beta) * (np.pi / 2) * U(theta)
        )

        with np.errstate(over="ignore"):
            value, _ = quad(
                integrand,
                -np.pi / 2,
                np.pi / 2,
                points=transition_points(log_g, -np.pi / 2, np.pi / 2),   # BREAKPOINTS
                epsabs=1e-8
            )

        return value / np.pi

    # ---------------------------------------------
    # α ≠ 1
    # ---------------------------------------------
    if x < 0:
        return 1.0 - generate_cdf_one_point(-x, alpha, -beta)

    theta0 = theta0_stable(alpha, beta)
    U = calculate_V(alpha, beta, theta0)

    exponent = alpha / (alpha - 1)

    integrand = lambda theta: np.exp(
        -(x ** exponent) * U(theta)
    )

    log_g = lambda theta: exponent * np.log(x) + np.log(U(theta))   # BREAKPOINTS

    value, _ = quad(
        integrand,
        -theta0,
        np.pi / 2,
        points=transition_points(log_g, -theta0, np.pi / 2),        # BREAKPOINTS
        epsabs=1e-8
    )

    if alpha < 1:
        return (1 - beta) / 2 + value / np.pi
    else:
        return 1 - value / np.pi


def generate_cdf(X, alpha, beta):
    """
    Vectorized wrapper.
    """
    X = np.asarray(X, dtype=float)
    return np.array(
        [generate_cdf_one_point(x, alpha, beta) for x in X]
    )