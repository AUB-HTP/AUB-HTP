import numpy as np
from scipy.integrate import quad

from aub_htp.pdf import generate_alpha_stable_pdf


def generate_cdf_one_point(x, alpha, beta, gamma=1.0, delta=0.0, epsabs=1e-8):
    """
    Numerically compute the CDF by integrating the AUB HTP PDF.

        F(x) = ∫_{-∞}^{x} f(t) dt
    """

    pdf = lambda t: generate_alpha_stable_pdf(np.array([t]), alpha, beta, gamma, delta)[0]

    # value, _ = quad(pdf, -np.inf, x, epsabs=epsabs)
    if x > 0:
        left, _ = quad(
            pdf,
            -np.inf,
            0,
            epsabs=epsabs
        )

        right, _ = quad(
            pdf,
            0,
            x,
            epsabs=epsabs
        )

        value = left + right

    else:
        value, _ = quad(
            pdf,
            -np.inf,
            x,
            epsabs=epsabs
        )

    return value


def generate_cdf(X, alpha, beta, gamma=1.0, delta=0.0, epsabs=1e-8):
    """
    Vectorized wrapper for the numerical CDF.
    """

    X = np.asarray(X, dtype=float)

    return np.array([generate_cdf_one_point(x, alpha, beta, gamma, delta, epsabs) for x in X])