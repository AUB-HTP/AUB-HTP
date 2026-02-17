import numpy as np
from scipy.special import gamma
from spectral_measure_sampler import BaseSpectralMeasureSampler

def sample_alpha_stable_vector(
    alpha: float,
    spectral_measure: BaseSpectralMeasureSampler,
    number_of_samples: int,
    shift_vector,
    number_of_convergence_terms: int | None = None,
):
    if number_of_convergence_terms is None:
        if alpha < 0.5:
            number_of_convergence_terms = 100
        elif 0.5 < alpha <= 1:
            number_of_convergence_terms = 1000
        elif 1 < alpha < 2:
            number_of_convergence_terms = 2000
        else:
            raise ValueError("alpha must be in (0,2)")
    d = spectral_measure.dimensions()
    x = np.zeros((number_of_samples, d))

    cumulative_exponential = np.zeros(number_of_samples)

    for _ in range(number_of_convergence_terms):
        cumulative_exponential += np.random.exponential(scale=1.0, size=number_of_samples)

        spectral_measure_samples = spectral_measure.sample(number_of_samples)
        weights = cumulative_exponential ** (-1.0 / alpha)

        x += spectral_measure_samples * weights[:, None]
    p=spectral_measure.mass()
    print(p)
    x *= _c(alpha, spectral_measure.mass())
    if d == 1:
        if shift_vector.size != 1:
            raise ValueError("Shift vector must have size 1 in 1D case.")
        x = x[:, 0] + shift_vector.item()   # return (n,) not (n,1)
    else:
        if shift_vector.shape != (d,):
            raise ValueError(f"Shift vector must have shape ({d},)")
        x = x + shift_vector
    return x


def _c(alpha: float, mass: float):
    #TODO: alpha = 2 is undefined #dont have to worry about alpha=2, it is the well known guassian case, and the LePage series 
    # theorem doesn't apply to the guassian case ~ Wael 
    return (_kappa(alpha)/mass) ** (-1 / alpha)

def _kappa(alpha: float):
    if abs(alpha - 1.0) < 1e-12:
        return np.pi / 2
    return gamma(2 - alpha) * np.cos(np.pi * alpha / 2) / (1 - alpha)
