"""
API Reference
=============
.. autosummary::
    :toctree: generated

    alpha_stable
    multivariate_alpha_stable
    random
    statistics
    machine_learning
"""

from ._alpha_stable import alpha_stable_gen, multivariate_alpha_stable_gen
from .random import (
    BaseSpectralMeasureSampler,
    DiscreteSampler,
    EllipticSampler,
    IsotropicSampler,
    MixedSampler,
    UnivariateSampler,
    get_random_state_generator,
    sample_alpha_stable_vector,
)
from .statistics import (
    alpha_location,
    alpha_power,
)
from .machine_learning import (
    AlphaStableLinearRegressor,
    AlphaStableKMeans,
    l_alpha_loss,
    r_alpha_score,
)

alpha_stable = alpha_stable_gen("alpha_stable")
multivariate_alpha_stable = multivariate_alpha_stable_gen()



__version__ = "1.0.7"

__all__ = [
    # Alpha Stable Scipy-Compatible frontend
    "alpha_stable",
    "alpha_stable_gen",
    "multivariate_alpha_stable",
    "multivariate_alpha_stable_gen",

    # Random
    "BaseSpectralMeasureSampler",
    "IsotropicSampler",
    "EllipticSampler",
    "DiscreteSampler",
    "MixedSampler",
    "UnivariateSampler",
    "sample_alpha_stable_vector",
    "get_random_state_generator",

    # Statistics
    "alpha_location",
    "alpha_power",

    # Machine Learning
    "AlphaStableLinearRegressor",
    "AlphaStableKMeans",
    "l_alpha_loss",
    "r_alpha_score",

    # Versioning
    "__version__",
]