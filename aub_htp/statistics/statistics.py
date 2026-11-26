from functools import lru_cache
from scipy.optimize import minimize, brentq
import aub_htp as ht
from scipy import special
from scipy.stats import levy_stable
import numpy as np


def _settle_alpha(alpha: float) -> float:
    """
    Check alpha and return it as a float, set to exactly 1 or 2 when close to them.

    Every function here calls this, so they all take the same branch for the same alpha.
    """
    alpha = float(alpha)
    if np.isclose(alpha, 1):
        return 1.0
    if np.isclose(alpha, 2):
        return 2.0
    if not (0 < alpha <= 2):
        raise ValueError(f"alpha must be in (0, 2], got {alpha!r}.")
    return alpha


def _validate_data(data: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return the data as (n_samples, n_features), and whether it was given as 1-D."""
    data = np.asarray(data, dtype=float)
    if data.ndim not in (1, 2):
        raise ValueError(
            "data must have shape (n_samples,) or (n_samples, n_features), "
            f"got {data.ndim}D data of shape {data.shape}."
        )
    if data.size == 0:
        raise ValueError(f"data must not be empty, got shape {data.shape}.")
    if not np.all(np.isfinite(data)):
        raise ValueError("data must not have NaN or infinite values.")
    data_is_one_dimensional = data.ndim == 1
    if data_is_one_dimensional:
        data = data.reshape(-1, 1)
    return data, data_is_one_dimensional


def isotropic_pdf(data: np.ndarray, alpha: float) -> np.ndarray:
    """
    Isotropic PDF for alpha-stable distribution, one value per sample.

    data has shape (n_samples,) for one feature, or (n_samples, n_features).

    Note for maintainers: This function uses the excplicit formulas for the Cauchy and Normal distributions for alpha = 1 and 2.
    This is for better performance.
    """
    alpha = _settle_alpha(alpha)
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.ndim != 2:
        raise ValueError(
            "data must have shape (n_samples,) or (n_samples, n_features), "
            f"got {data.ndim}D data of shape {data.shape}."
        )
    _, dimensions = data.shape
    if alpha == 2:
        # Normal distribution: N(0, I)
        return (2 * np.pi) ** (-dimensions / 2) * np.exp(-np.sum(data ** 2, axis=1) / 2)
    elif alpha == 1:
        # Cauchy distribution
        coefficient = special.gamma((dimensions + 1) / 2) / (np.pi ** ((dimensions + 1) / 2))
        return coefficient * (1 + np.sum(data ** 2, axis=1)) ** (-(dimensions + 1) / 2)
    elif dimensions == 1:
        scale = (1 / alpha) ** (1 / alpha)
        return ht.alpha_stable.pdf(data.ravel(), alpha, 0, loc = 0, scale = scale)
    else:
        raise ValueError("Multidimensional isotropic distribution is only defined for alpha = 1 (Cauchy) or alpha = 2 (Normal)")


def isotropic_entropy(dimensions: int, alpha: float) -> float:
    """Differential entropy of the isotropic alpha-stable distribution."""
    return _isotropic_entropy(int(dimensions), _settle_alpha(alpha))


@lru_cache(maxsize=None)
def _isotropic_entropy(dimensions: int, alpha: float) -> float:
    # alpha comes from _settle_alpha, so values close to 1 or 2 share one cache entry.
    if alpha == 2:
        # Normal distribution: N(0, I)
        return float(0.5 * dimensions * np.log(2 * np.pi * np.e))
    elif alpha == 1:
        # Cauchy distribution
        return float(
            np.log(np.pi ** ((dimensions + 1) / 2) / special.gamma((dimensions + 1) / 2))
            + ((dimensions + 1) / 2)
            * (special.digamma((dimensions + 1) / 2) - special.digamma(0.5))
        )
    elif dimensions == 1:
        return float(ht.alpha_stable.entropy(alpha, 0, loc = 0, scale = (1 / alpha) ** (1 / alpha)))
    else:
        raise ValueError("Multidimensional isotropic distribution is only defined for alpha = 1 (Cauchy) or alpha = 2 (Normal)")


def alpha_power(data: np.ndarray, alpha: float) -> float:
    alpha = _settle_alpha(alpha)
    data, _ = _validate_data(data)

    # For all-zero data, -mean(ln p(x / P)) = -ln p(0) < entropy for every P.
    # There is no root, so the power is 0.
    if not np.any(data):
        return 0.0

    _, dimensions = data.shape
    entropy = isotropic_entropy(dimensions, alpha)
    epsilon = np.finfo(float).eps
    def objective_function(power: float) -> float:
        if power < 0:
            return np.inf
        pdf = np.maximum(isotropic_pdf(data / power, alpha), epsilon)
        log_pdf = np.log(pdf)
        expected_log_pdf = np.mean(log_pdf)
        return (expected_log_pdf + entropy)**2

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
    alpha = _settle_alpha(alpha)
    data, data_is_one_dimensional = _validate_data(data)

    _, dimensions = data.shape

    entropy = isotropic_entropy(dimensions, alpha)
    epsilon = np.finfo(float).eps
    penalty = 1e6

    def objective_function(log_power_and_location: np.ndarray) -> float: #TODO: ask profs about this objective function
        power = log_power_and_location[0]
        location = log_power_and_location[1:]
        if power < 0:
            return np.inf

        pdf = np.maximum(isotropic_pdf((data - location) / power, alpha), epsilon)
        log_pdf = np.log(pdf)
        residual = np.mean(log_pdf) + entropy

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
