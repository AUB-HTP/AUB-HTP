import numpy as np
from scipy.special import gamma
from .spectral_measure_sampler import BaseSpectralMeasureSampler,IsotropicSampler

def sample_alpha_stable_vector(
    alpha: float,
    spectral_measure: BaseSpectralMeasureSampler,
    number_of_samples: int,
    number_of_convergence_terms: int | None = None,
):
    number_of_convergence_terms = number_of_convergence_terms or 100#TODO: Document the error better
    d = spectral_measure.dimensions()
    x = np.zeros((number_of_samples, d))

    cumulative_exponential = np.zeros(number_of_samples)

    for _ in range(number_of_convergence_terms):
        cumulative_exponential += np.random.exponential(scale=1.0, size=number_of_samples)

        spectral_measure_samples = spectral_measure.sample(number_of_samples)
        weights = cumulative_exponential ** (-1.0 / alpha)

        x += spectral_measure_samples * weights[:, None]

    x *= _c(alpha)
    return x


def _c(alpha):
    #TODO:
    # 1. alpha = 2 is undefined
    # 2. add mass
    return _kappa(alpha) ** (-1 / alpha)

def _kappa(alpha):
    if abs(alpha - 1.0) < 1e-12:
        return np.pi / 2
    return gamma(2 - alpha) * np.cos(np.pi * alpha / 2) / (1 - alpha)
