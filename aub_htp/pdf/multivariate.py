"""Isotropic alpha-stable density as a function of the radius.

A large radius uses a series in ``1 / r``. Any other radius uses a mixture of
normals.
"""

import math
from functools import lru_cache

import numpy as np
from scipy.special import gammaln, logsumexp


def _isotropic_radial_log_series(radius, alpha, d, gamma_scale, *, terms=200):
    """ln of the isotropic density from its series in powers of ``1 / r``.

    ::

        f_d(r) = 1 / (pi^(d/2 + 1) r^d) * sum_{k >= 1} (-1)^(k+1) / k!
                 * Gamma(k alpha / 2 + 1) Gamma((k alpha + d) / 2)
                 * sin(k alpha pi / 2) (2 gamma / r)^(k alpha)

    The series converges for ``alpha < 1`` and is asymptotic in ``r`` for
    ``alpha > 1``, so the sum stops at its smallest term, or after ``terms``.

    Returns ``(log_pdf, relative_error)``, where ``relative_error`` estimates
    the relative error of ``f_d(r)``.
    """
    radius = np.atleast_1d(np.asarray(radius, dtype=np.float64))
    k = np.arange(1, terms + 1)[:, None]
    # ln of each term, relative to the first one so that nothing underflows.
    log_first = (gammaln(alpha / 2.0 + 1.0) + gammaln((alpha + d) / 2.0)
                 + alpha * (np.log(2.0 * gamma_scale) - np.log(radius)))
    log_term = (
        gammaln(k * alpha / 2.0 + 1.0) + gammaln((k * alpha + d) / 2.0)
        - gammaln(k + 1.0)
        + k * alpha * (np.log(2.0 * gamma_scale) - np.log(radius)[None, :])
        - log_first[None, :]
    )
    sign = (-1.0) ** (k + 1) * np.sin(k * alpha * np.pi / 2.0)
    kept = k - 1 <= np.argmin(log_term, axis=0)[None, :]
    term = np.where(kept, sign * np.exp(np.where(kept, log_term, 0.0)), 0.0)
    total = np.sum(term, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_pdf = (np.log(total) + log_first
                   - (d / 2.0 + 1.0) * np.log(np.pi) - d * np.log(radius))
        # Truncation: the smallest term. Rounding: the largest term times eps.
        relative_error = (
            np.exp(np.min(log_term, axis=0))
            + np.finfo(np.float64).eps * np.max(np.abs(term), axis=0)
        ) / total
    return log_pdf, np.where(total > 0.0, relative_error, np.inf)


@lru_cache(maxsize=128)
def _isotropic_series_radius(alpha, d, gamma_scale):
    """Smallest radius from which the series in ``1 / r`` is accurate to 1e-10.

    ``inf`` if it is accurate nowhere.
    """
    with np.errstate(over="ignore"):
        candidates = (2.0 * gamma_scale
                      / np.geomspace(1e3, 1e-8, 600) ** (1.0 / alpha))
    # A small alpha overflows the largest ones.
    candidates = candidates[np.isfinite(candidates)]
    _, relative_error = _isotropic_radial_log_series(
        candidates, alpha, d, gamma_scale)
    inaccurate = np.flatnonzero(~(relative_error <= 1e-10))
    if inaccurate.size == 0:
        return float(candidates[0])
    if inaccurate[-1] + 1 == candidates.size:
        return math.inf
    return float(candidates[inaccurate[-1] + 1])


@lru_cache(maxsize=8)
def _positive_stable_nodes(alpha, d):
    """Quadrature nodes (given as ``ln A``) and log-weights for ``A`` positive ``alpha / 2``-stable.

    Kanter's formula, with ``a = alpha / 2``, ``U`` uniform on ``(0, pi)`` and
    ``E`` exponential::

        A = sin(a U) / sin(U)^(1 / a) * (sin((1 - a) U) / E)^((1 - a) / a)

    Two Gauss--Legendre grids: over ``s``, with ``U = pi (1 - exp(-s))``, and
    over ``ln E``, finer and wider in many dimensions. The nodes cluster at
    ``U = pi``, where ``A`` is large for ``alpha`` near 2.
    """
    a = alpha / 2.0
    x_s, w_s = np.polynomial.legendre.leggauss(200)
    # Beyond s = 45, U is closer to pi than any A that matters needs.
    s = 0.5 * 45.0 * (x_s + 1.0)
    u = np.pi * -np.expm1(-s)
    # sin(U) = sin(pi - U), which does not lose digits near pi.
    sin_u = np.sin(np.pi * np.exp(-s))
    w_u = 0.5 * 45.0 * w_s * np.exp(-s)  # du / pi
    peak = 1.0 + d / 2.0 * (1.0 - a) / a
    # In ln E the mass has width 1 / sqrt(peak).
    x_t, w_t = np.polynomial.legendre.leggauss(
        int(np.clip(160.0 * np.sqrt(peak), 300, 6000)))
    # ln E. Outside this range exp(-E) E^(1 + m) is negligible.
    low, high = -45.0, np.log(745.0 + peak + 40.0 * np.sqrt(peak))
    t = low + 0.5 * (high - low) * (x_t + 1.0)
    w_t = 0.5 * (high - low) * w_t
    u, t = np.meshgrid(u, t, indexing="ij")
    log_a = (np.log(np.sin(a * u)) - np.log(sin_u)[:, None] / a
             + (1.0 - a) / a * (np.log(np.sin((1.0 - a) * u)) - t))
    # dE = E d(ln E)
    log_weight = np.log(w_u)[:, None] + np.log(w_t)[None, :] - np.exp(t) + t
    return log_a.ravel(), log_weight.ravel()


def _isotropic_radial_log_mixture(radius, alpha, d, gamma_scale):
    """ln of the isotropic density as a mixture of normals.

    With ``A`` positive ``alpha / 2``-stable::

        f_d(r) = E[(4 pi gamma^2 A)^(-d / 2) exp(-r^2 / (4 gamma^2 A))]

    Not accurate at a very large radius.
    """
    radius = np.atleast_1d(np.asarray(radius, dtype=np.float64))
    log_a, log_weight = _positive_stable_nodes(float(alpha), int(d))
    base = log_weight - d / 2.0 * (np.log(4.0 * np.pi) + 2.0 * np.log(gamma_scale) + log_a)
    # 1 / (2 gamma sqrt(A)). Squared after the product with r, so that a tiny r does not underflow.
    inverse_deviation = np.exp(-0.5 * log_a - np.log(2.0 * gamma_scale))
    out = np.empty_like(radius)
    # About 4e6 values in memory at once.
    rows = max(1, 4_000_000 // log_a.size)
    for start in range(0, radius.size, rows):
        block = radius[start:start + rows]
        out[start:start + rows] = logsumexp(
            base[None, :] - (block[:, None] * inverse_deviation[None, :]) ** 2,
            axis=1)
    return out


def _isotropic_radial_log_pdf(radius, alpha, d, gamma_scale):
    """ln of the isotropic density with characteristic function ``exp(-(gamma |t|)^alpha)``.

    * ``alpha = 1`` and ``alpha = 2`` use the Cauchy and normal formulas.
    * A large radius uses the series in ``1 / r``.
    * Any other radius uses the mixture of normals.
    """
    radius = np.atleast_1d(np.asarray(radius, dtype=np.float64))
    alpha = float(alpha)
    gamma_scale = float(gamma_scale)

    if alpha == 2.0:
        # exp(-gamma^2 |t|^2) is N(0, 2 gamma^2 I).
        return (-d / 2.0 * np.log(4.0 * np.pi * gamma_scale ** 2)
                - radius ** 2 / (4.0 * gamma_scale ** 2))
    if alpha == 1.0:
        return (gammaln((d + 1) / 2.0) - (d + 1) / 2.0 * np.log(np.pi)
                - d * np.log(gamma_scale)
                - (d + 1) / 2.0 * np.log1p((radius / gamma_scale) ** 2))

    out = np.empty_like(radius)
    in_tail = radius >= _isotropic_series_radius(alpha, d, gamma_scale)
    if np.any(in_tail):
        out[in_tail], _ = _isotropic_radial_log_series(
            radius[in_tail], alpha, d, gamma_scale)
    if not np.all(in_tail):
        out[~in_tail] = _isotropic_radial_log_mixture(
            radius[~in_tail], alpha, d, gamma_scale)
    return out


def _isotropic_radial_pdf(radius, alpha, d, gamma_scale):
    """Isotropic density with characteristic function ``exp(-(gamma |t|)^alpha)``.

    See ``_isotropic_radial_log_pdf``, which does not underflow.
    """
    return np.exp(_isotropic_radial_log_pdf(radius, alpha, d, gamma_scale))
