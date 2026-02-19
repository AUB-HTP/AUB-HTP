import numpy as np
from scipy.special import gamma
from .spectral_measure_sampler import BaseSpectralMeasureSampler
import logging

def sample_alpha_stable_vector(
    alpha: float,
    spectral_measure: BaseSpectralMeasureSampler,
    number_of_samples: int = 1,
    shift_vector: np.ndarray = 0,
    number_of_convergence_terms: int | None = None,
):
    dimensions = spectral_measure.dimensions()
    shift_vector = np.broadcast_to(shift_vector, dimensions)

    number_of_convergence_terms = number_of_convergence_terms or estimate_number_of_convergence_terms(alpha)

    x = np.zeros((number_of_samples, dimensions))

    cumulative_exponential = np.zeros(number_of_samples)

    for _ in range(number_of_convergence_terms):
        cumulative_exponential += np.random.exponential(scale=1.0, size=number_of_samples)

        spectral_measure_samples = spectral_measure.sample(number_of_samples)
        weights = cumulative_exponential ** (-1.0 / alpha)

        x += spectral_measure_samples * weights[:, None]

    x *= _c(alpha, spectral_measure.mass())
    x += shift_vector

    if dimensions == 1:
        x = x.ravel()
    return x


def _c(alpha: float, mass: float):
    return (_kappa(alpha)/mass) ** (-1 / alpha)

def _kappa(alpha: float):
    if abs(alpha - 1.0) < 1e-12:
        return np.pi / 2
    return gamma(2 - alpha) * np.cos(np.pi * alpha / 2) / (1 - alpha)

def estimate_number_of_convergence_terms(alpha: float):
    # Future works: this a function of the error such that the error is less than p (default p=0.01)
    logging.warning("for large alpha, the number of convergence terms needs to be large for more accuracy.")

    if alpha < 0.5:
        return 100
    elif 0.5 < alpha <= 1:
        return 1000
    elif 1 < alpha < 2:
        return 2000
    else:
        raise ValueError("alpha must be in (0,2)")