import numpy as np
from scipy.special import gamma
from .spectral_measure_sampler import BaseSpectralMeasureSampler
from .util import get_random_state_generator
import logging

def sample_alpha_stable_vector(
    alpha: float,
    spectral_measure: BaseSpectralMeasureSampler,
    number_of_samples: int = 1,
    shift_vector: np.ndarray = 0,
    max_number_of_convergence_terms: int = 50000,
    error: float = 0.01,
    random_state: None | int | np.random.RandomState | np.random.Generator = None
):
    random_state = get_random_state_generator(random_state)
    dimensions = spectral_measure.dimensions()
    mass = spectral_measure.mass()
    shift_vector = np.broadcast_to(shift_vector, dimensions)

    number_of_convergence_terms = int(estimate_number_of_convergence_terms(error, alpha, mass))
    if number_of_convergence_terms > max_number_of_convergence_terms:
        logging.warning(f"Estimated number of convergence terms {number_of_convergence_terms} exceeded the maximum number of convergence terms {max_number_of_convergence_terms}. Using {max_number_of_convergence_terms} convergence terms.")
        number_of_convergence_terms = max_number_of_convergence_terms

    x = np.zeros((number_of_samples, dimensions))

    cumulative_exponential = np.zeros(number_of_samples)

    for _ in range(number_of_convergence_terms):
        cumulative_exponential += random_state.exponential(scale=1.0, size=number_of_samples)

        spectral_measure_samples = spectral_measure.sample(number_of_samples, random_state)
        weights = cumulative_exponential ** (-1.0 / alpha)

        x += spectral_measure_samples * weights[:, None]

    x *= _c(alpha, mass)
    x += shift_vector

    if dimensions == 1:
        x = x.ravel()
    return x


def _c(alpha: float, mass: float):
    return (_kappa(alpha)/mass)**(-1 / alpha)

def _kappa(alpha: float):
    if abs(alpha - 1.0) < 1e-12:
        return np.pi / 2
    return gamma(2 - alpha) * np.cos(np.pi * alpha / 2) / (1 - alpha)

def estimate_number_of_convergence_terms(error: float, alpha: float, mass: float) -> float:
    if 0 < alpha < 1:
        return estimate_number_of_convergence_terms_alpha_0_to_1(error, alpha, mass)
    elif 1 <= alpha < 2:
        return estimate_number_of_convergence_terms_alpha_1_to_2(error, alpha, mass)
    else:
        raise ValueError(f"{alpha = } must be in (0,2)")


def estimate_number_of_convergence_terms_alpha_0_to_1(mse_error: float, alpha: float, mass: float) -> float:
    assert 0 < alpha < 1 and mse_error > 0, "alpha must satisfy 0 < alpha < 1, and mse_error > 0"
    c_alpha = _c(alpha, mass)
    argument = (((1 - alpha) ** 2) * mse_error) / (2 * (alpha ** 2) * (c_alpha ** 2))
    logn = (alpha / (2 * alpha - 2)) * np.log(argument) + 1
    return np.exp(logn)


def estimate_number_of_convergence_terms_alpha_1_to_2(mse_error: float, alpha: float, mass: float) -> float:
    assert 1 <= alpha < 2 and mse_error > 0, "alpha must satisfy 1 <= alpha < 2, and mse_error > 0"
    c_alpha = _c(alpha, mass)
    log_arg = np.log(2 - alpha) + np.log(mse_error) - np.log(2 * alpha * c_alpha ** 2)
    logn = (alpha / (alpha - 2)) * log_arg
    return np.exp(logn) + 1