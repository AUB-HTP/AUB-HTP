import numpy as np
from scipy.integrate import quad

from aub_htp.pdf.zolotarev import theta0_stable, calculate_V


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

        value, _ = quad(
            integrand,
            -np.pi / 2,
            np.pi / 2,
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

    value, _ = quad(
        integrand,
        -theta0,
        np.pi / 2,
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