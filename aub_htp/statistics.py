from functools import lru_cache
from scipy.optimize import minimize
from scipy.stats import levy_stable as alpha_stable
from scipy import special
import numpy as np


def _isotropic_pdf(data: np.ndarray, alpha: float) -> np.ndarray:
    assert data.ndim == 2
    _, dimensions = data.shape
    if dimensions == 1:
        return alpha_stable.pdf(data.ravel(), alpha, 0, loc = 0, scale = (1 / alpha) ** (1 / alpha))
    else:
        #TODO: validate this case theoretically
        if alpha != 1:
            raise ValueError("Isotropic PDF is only defined for alpha = 1")
        squared_norm = np.sum(data ** 2, axis=1)
        coefficient = special.gamma((dimensions + 1) / 2) / (np.pi ** ((dimensions + 1) / 2))
        return coefficient * (1 + squared_norm) ** (-(dimensions + 1) / 2)


@lru_cache(maxsize=None)
def _isotropic_entropy(dimensions: int, alpha: float) -> float:
    if dimensions == 1:
        return alpha_stable.entropy(alpha, 0, loc = 0, scale = (1 / alpha) ** (1 / alpha))
    else:
        #TODO: validate this case theoretically
        if alpha != 1:
            raise ValueError("Isotropic entropy is only defined for alpha = 1")
        
        return (
            np.log(np.pi ** ((dimensions + 1) / 2) / special.gamma((dimensions + 1) / 2))
            + ((dimensions + 1) / 2)
            * (
                special.digamma((dimensions + 1) / 2)
                - special.digamma(0.5)
            )
        )


def alpha_power(data: np.ndarray, alpha: float) -> float:
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    _, dimensions = data.shape
    entropy = _isotropic_entropy(dimensions, alpha)

    def objective_function(power: float) -> float:
        log_pdf = -np.log(_isotropic_pdf(data / power, alpha))
        expected_log_pdf = np.mean(log_pdf)
        return (expected_log_pdf - entropy)**2

    return minimize(objective_function, x0 = 1.0).x


def alpha_location(data: np.ndarray, alpha: float) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return minimize(
        lambda location: alpha_power(data - location, alpha), 
        x0 = np.median(data, axis = 0)
    ).x