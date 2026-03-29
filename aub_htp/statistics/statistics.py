from functools import lru_cache
from scipy.optimize import minimize
import aub_htp as ht
from scipy import special
from scipy.stats import levy_stable
import numpy as np


def isotropic_pdf(data: np.ndarray, alpha: float) -> np.ndarray:
    assert data.ndim == 2
    _, dimensions = data.shape
    if dimensions == 1:
        return ht.alpha_stable.pdf(data.ravel(), alpha, 0, loc = 0, scale = (1 / alpha) ** (1 / alpha))
    else:
        squared_norm = np.sum(data ** 2, axis=1)
        if alpha == 2:
            # Normal distribution: N(0, I)
            return (2 * np.pi) ** (-dimensions / 2) * np.exp(-squared_norm / 2)
        elif alpha == 1:
            # Cauchy distribution
            coefficient = special.gamma((dimensions + 1) / 2) / (np.pi ** ((dimensions + 1) / 2))
            return coefficient * (1 + squared_norm) ** (-(dimensions + 1) / 2)
        else:
            raise ValueError("Multidimensional isotropic distribution is only defined for alpha = 1 (Cauchy) or alpha = 2 (Normal)")


@lru_cache(maxsize=None)
def isotropic_entropy(dimensions: int, alpha: float) -> float:
    if dimensions == 1:
        return levy_stable.entropy(alpha, 0, loc = 0, scale = (1 / alpha) ** (1 / alpha))
    else:
        if alpha == 2:
            # Normal distribution: N(0, I)
            return 0.5 * dimensions * np.log(2 * np.pi * np.e)
        elif alpha == 1:
            # Cauchy distribution
            return (
                np.log(np.pi ** ((dimensions + 1) / 2) / special.gamma((dimensions + 1) / 2))
                + ((dimensions + 1) / 2)
                * (special.digamma((dimensions + 1) / 2) - special.digamma(0.5))
            )
        else:
            raise ValueError("Multidimensional isotropic distribution is only defined for alpha = 1 (Cauchy) or alpha = 2 (Normal)")


def alpha_power(data: np.ndarray, alpha: float) -> float:
    data = np.asarray(data) 
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    _, dimensions = data.shape
    entropy = isotropic_entropy(dimensions, alpha)

    def objective_function(power: float) -> float:
        log_pdf = -np.log(isotropic_pdf(data / power, alpha))  
        expected_log_pdf = np.mean(log_pdf)
        return (expected_log_pdf - entropy)**2

    return minimize(
        objective_function, 
        x0 = 1.0,
        method = "Powell",
        options = {
            "maxiter": int(5000),
            "xtol": float(1e-8),
            "ftol": float(1e-8),
        }
    ).x.item()



def deprecated_alpha_location(data: np.ndarray, alpha: float) -> np.ndarray:
    data = np.asarray(data)
    data_is_one_dimensional = data.ndim == 1
    if data_is_one_dimensional:
        data = data.reshape(-1, 1)
    optimal_location = minimize(
        lambda location: alpha_power(data - location, alpha), 
        x0 = np.median(data, axis = 0),
        method = "Powell",
        options = {
            "maxiter": int(5000),
            "xtol": float(1e-8),
            "ftol": float(1e-8),
        },
    ).x
    if data_is_one_dimensional:
        return optimal_location.item()
    else:
        return optimal_location


def alpha_location(data: np.ndarray, alpha: float) -> np.ndarray:
    data = np.asarray(data)
    data_is_one_dimensional = data.ndim == 1
    if data_is_one_dimensional:
        data = data.reshape(-1, 1)
    
    _, dimensions = data.shape

    entropy = isotropic_entropy(dimensions, alpha)
    penalty = 1e6

    def objective_function(log_power_and_location: np.ndarray) -> float: #TODO: ask profs about this objective function
        power = log_power_and_location[0]
        location = log_power_and_location[1:]
        if power < 0:
            return np.inf

        pdf = isotropic_pdf((data - location) / power, alpha)
        if np.any(pdf <= 0):
            print(((data - location) / power)[pdf<=0])

        log_pdf = -np.log(pdf)  
        residual = np.mean(log_pdf) - entropy

        return power + penalty * residual**2

    optimal_log_power_and_location = minimize(
        objective_function, 
        x0 = np.concatenate(([1.0], np.median(data, axis = 0))),
        method = "Powell",
        options = {
            "maxiter": int(5000),
            "xtol": float(1e-8),
            "ftol": float(1e-8),
        },
    ).x

    optimal_location = optimal_log_power_and_location[1:]
    if data_is_one_dimensional:
        return optimal_location.item()
    else:
        return optimal_location
