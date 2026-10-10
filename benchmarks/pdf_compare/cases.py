"""Standard comparison cases.

Three families:

* ``CROSS_METHOD_CASES`` -- symmetric, centred laws.
* ``GAUSSIAN_CASES`` -- ``alpha == 2`` isotropic laws with an exact Gaussian
  reference.  FFT only: the projection kernel is singular at ``alpha == 2``
  (``tan(pi alpha / 2)`` diverges) and the model refuses that value.
* ``SKEWED_CASES`` -- skewed and/or shifted laws.

Both methods handle every family except the Gaussian limit: the FFT estimator
carries skewness in the imaginary part of the characteristic function and the
location as an ``exp(i <t, mu>)`` phase, so ``both_methods`` is ``False`` only
for ``alpha == 2``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from aub_htp.random.spectral_measure_sampler import (
    DiscreteSampler,
    EllipticSampler,
    IsotropicSampler,
)


@dataclass
class Case:
    """One comparison scenario.

    Attributes
    ----------
    name : str
        Unique identifier.
    alpha : float
        Stability index.
    make_sampler : Callable[[float], BaseSpectralMeasureSampler]
        Factory taking ``alpha`` and returning a fresh spectral sampler.
    d : int
        Dimension.
    symmetric : bool
        Whether the law is symmetric (both methods valid) or not.
    shift : np.ndarray
        Location vector.
    both_methods : bool
        True when the FFT method is applicable (symmetric, any shift).
    """

    name: str
    alpha: float
    make_sampler: Callable
    d: int
    symmetric: bool = True
    shift: np.ndarray = field(default_factory=lambda: 0.0)
    both_methods: bool = True

    def sampler(self):
        return self.make_sampler(self.alpha)


# --------------------------------------------------------------------------- #
# Symmetric, centred -- both methods comparable head-to-head.
# --------------------------------------------------------------------------- #
CROSS_METHOD_CASES = [
    Case("iso2d_a1.2", 1.2, lambda a: IsotropicSampler(2, a, 1.0), d=2),
    Case("iso2d_a1.5", 1.5, lambda a: IsotropicSampler(2, a, 1.0), d=2),
    Case("iso2d_a1.8", 1.8, lambda a: IsotropicSampler(2, a, 1.0), d=2),
    Case("iso2d_a0.8", 0.8, lambda a: IsotropicSampler(2, a, 1.0), d=2),
    Case("elliptic2d_a1.5", 1.5,
         lambda a: EllipticSampler(2, a, sigma=[[1.2, 0.35], [0.35, 1.8]],
                                   mass=1.0), d=2),
    Case("iso3d_a1.5", 1.5, lambda a: IsotropicSampler(3, a, 1.0), d=3),
]


# --------------------------------------------------------------------------- #
# Exact Gaussian limit -- FFT vs closed form.
# --------------------------------------------------------------------------- #
GAUSSIAN_CASES = [
    Case("gauss2d", 2.0, lambda a: IsotropicSampler(2, a, 1.0), d=2,
         both_methods=False),
    Case("gauss3d", 2.0, lambda a: IsotropicSampler(3, a, 1.0), d=3,
         both_methods=False),
]


# --------------------------------------------------------------------------- #
# Skewed / shifted -- projection method only (FFT cannot represent skew).
# --------------------------------------------------------------------------- #
def _skewed_discrete(alpha):
    # A "Mercedes-star" of three unit directions.  The weighted mean is exactly
    # zero (so the alpha >= 1 spectral constraint holds), yet the measure is not
    # symmetric under reflection of the second axis, so the y-marginal is
    # genuinely skewed (beta ~ 0.17 at alpha = 1.5) -- a light-tailed skew case
    # the point-wise method can be validated on cheaply.
    s = np.sqrt(3.0) / 2.0
    positions = np.array([[0.0, 1.0], [-s, -0.5], [s, -0.5]])
    weights = np.array([1.0, 1.0, 1.0])
    return DiscreteSampler(alpha, positions, weights)


SKEWED_CASES = [
    Case("skew_star2d_a1.5", 1.5, _skewed_discrete, d=2, symmetric=False),
    Case("skew_star2d_a0.8", 0.8, _skewed_discrete, d=2, symmetric=False),
    Case("shifted_iso2d_a1.5", 1.5, lambda a: IsotropicSampler(2, a, 1.0),
         d=2, symmetric=True, shift=np.array([1.0, -2.0])),
]

# Backwards-compatible alias: these used to be projection-only.
PROJECTION_ONLY_CASES = SKEWED_CASES

ALL_CASES = CROSS_METHOD_CASES + GAUSSIAN_CASES + SKEWED_CASES


def get_case(name):
    """Look up a case by name."""
    for case in ALL_CASES:
        if case.name == name:
            return case
    raise KeyError(f"unknown case {name!r}; known: {[c.name for c in ALL_CASES]}")
