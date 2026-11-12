"""
Tools for working with Alpha Stable (Levy-Stable) distributions.
"""
from .pdf import *
from .random import *

from ._alpha_stable import alpha_stable_gen, multivariate_alpha_stable_gen

#: Univariate Alpha Stable distribution, an instance of :class:`alpha_stable_gen`.
alpha_stable = alpha_stable_gen("alpha_stable")
#: Multivariate Alpha Stable distribution, an instance of :class:`multivariate_alpha_stable_gen`.
multivariate_alpha_stable = multivariate_alpha_stable_gen()

__version__ = "1.0.11"

__all__ = [
    # Alpha Stable Scipy-Compatible frontend
    "alpha_stable",
    "alpha_stable_gen",
    "multivariate_alpha_stable",
    "multivariate_alpha_stable_gen",

    # PDF
    "generate_alpha_stable_pdf",

    # Random
    "BaseSpectralMeasureSampler",
    "IsotropicSampler",
    "EllipticSampler",
    "DiscreteSampler",
    "MixedSampler",
    "UnivariateSampler",
    "sample_alpha_stable_vector",
]