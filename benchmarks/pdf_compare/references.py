"""Independent ground-truth references for the multivariate stable density.

None of these use either method under test, so agreement is meaningful:

* :func:`gaussian_pdf` -- exact closed form at ``alpha == 2``.  The symmetric
  stable log-CF is ``-integral |<t,s>|^alpha Gamma(ds)``; at ``alpha == 2`` this
  is ``-t^T M t`` with ``M = integral s s^T Gamma(ds)``, i.e. a Gaussian with
  covariance ``Sigma = 2 M``.  Estimated from spectral samples as
  ``2 * mass * mean(V V^T)``.
* :func:`axis_marginal_pdf` -- any one-dimensional projection of a stable vector
  is univariate stable.  The axis marginal has scale
  ``sigma_k = (mass * mean(|V_k|^alpha))^{1/alpha}`` and skewness
  ``beta_k``; its density comes from the package's independent 1D
  ``alpha_stable`` implementation.
* :func:`monte_carlo_marginal` -- a histogram of a coordinate of vectors drawn
  by ``sample_alpha_stable_vector`` (noisy, LePage-series based).
"""

from __future__ import annotations

import numpy as np

from aub_htp import alpha_stable, sample_alpha_stable_vector
from aub_htp.random.spectral_measure_sampler import BaseSpectralMeasureSampler


def spectral_samples(sampler: BaseSpectralMeasureSampler, n, random_state=None):
    """Draw ``n`` spectral-measure samples, shape ``(n, d)``."""
    return sampler.sample(n, random_state)


def gaussian_covariance(sampler, samples):
    """Covariance of the ``alpha == 2`` Gaussian limit from spectral samples.

    ``Sigma = 2 * mass * mean(V V^T)``.
    """
    mass = sampler.mass()
    return 2.0 * mass * (samples.T @ samples) / samples.shape[0]


def gaussian_pdf(points, cov):
    """Exact multivariate-normal density at ``points`` (the ``alpha==2`` law)."""
    from scipy.stats import multivariate_normal
    points = np.asarray(points, dtype=np.float64)
    rv = multivariate_normal(mean=np.zeros(cov.shape[0]), cov=cov)
    return rv.pdf(points)


def axis_marginal_params(sampler, samples, alpha, axis=0):
    """Scale and skewness of the marginal along a coordinate ``axis``.

    Returns
    -------
    beta : float
        Projection skewness (S1), clipped to ``[-1, 1]``.
    scale : float
        Projection scale ``sigma_k``.
    """
    mass = sampler.mass()
    v = samples[:, axis]
    av = np.abs(v)
    sigma_alpha = mass * np.mean(av ** alpha)
    scale = sigma_alpha ** (1.0 / alpha)
    if sigma_alpha > 0.0:
        beta = mass * np.mean(np.sign(v) * av ** alpha) / sigma_alpha
    else:
        beta = 0.0
    return float(np.clip(beta, -1.0, 1.0)), float(scale)


def axis_marginal_pdf(x, alpha, beta, scale, loc=0.0):
    """1D stable marginal density via the package's own ``alpha_stable``.

    For ``alpha != 1`` a strictly-stable vector has centred marginals
    (``loc == 0``); a location vector shifts the marginal by its component.
    """
    return alpha_stable.pdf(np.asarray(x, dtype=np.float64),
                            alpha, beta, loc=loc, scale=scale)


def monte_carlo_marginal(alpha, sampler, *, axis=0, number_of_samples=200_000,
                         shift=0.0, random_state=0, bins=61,
                         lo=-12.0, hi=12.0):
    """Histogram density of one coordinate from sampled vectors.

    Returns ``(centers, density)`` over ``[lo, hi]``.
    """
    x = sample_alpha_stable_vector(
        alpha, sampler,
        number_of_samples=number_of_samples,
        shift_vector=shift,
        random_state=random_state,
    )
    coord = x[:, axis]
    density, edges = np.histogram(coord, bins=bins, range=(lo, hi), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, density
