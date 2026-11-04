import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from aub_htp.pdf.zolotarev import theta0_stable

def calculate_log_V(alpha, theta0):
    """
    log V(θ) for alpha != 1, term by term.
    V itself overflows or underflows when alpha is close to 1, because its
    last factor is raised to alpha / (alpha - 1).
    """
    exponent = alpha / (alpha - 1)
    c = np.log(np.cos(alpha * theta0)) / (alpha - 1)
    return lambda theta: (
        c
        + np.log(np.cos(alpha * theta0 + (alpha - 1) * theta) / np.cos(theta))
        + exponent * np.log(np.cos(theta) / np.sin(alpha * (theta + theta0)))
    )


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

        log_g = lambda theta: (
            np.log(2 / np.pi)
            + np.log((np.pi / 2 + beta * theta) / np.cos(theta))
            + (np.pi / 2 + beta * theta) * np.tan(theta) / beta
            + (-x / beta) * (np.pi / 2)
        )
        
        integrand = lambda theta: np.exp(-np.exp(log_g(theta)))

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
    if x == 0:
            return 0.5 - theta0 / np.pi
    log_V = calculate_log_V(alpha, theta0)

    exponent = alpha / (alpha - 1)

    log_g = lambda theta: exponent * np.log(x) + log_V(theta)

    integrand = lambda theta: np.exp(-np.exp(log_g(theta)))

    with np.errstate(over="ignore"):
        value, _ = quad(
            integrand,
            -theta0,
            np.pi / 2,
            points=transition_points(log_g, -theta0, np.pi / 2),    # BREAKPOINTS
            epsabs=1e-8
        )

    if alpha < 1:
        return 0.5 - theta0 / np.pi + value / np.pi
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