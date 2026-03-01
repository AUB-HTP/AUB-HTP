import numpy as np
from .util import get_random_state_generator

def sample_cms(alpha : float, beta : float, size : int = 1, random_state : None | int | np.random.RandomState | np.random.Generator = None) -> np.ndarray:
    """
    Generate alpha-stable random variables in Nolan Type 1 parametrization
    with gamma = 1 and delta = 0.

    Parameters
    ----------
    alpha : float in (0,2]
    beta  : float in [-1,1]
    size  : int
    random_state : int or None

    Returns
    -------
    samples : ndarray
        Samples from S(alpha, beta, 1, 0; 1)
    """

    if not (0 < alpha <= 2):
        raise ValueError("alpha must be in (0,2]")
    if not (-1 <= beta <= 1):
        raise ValueError("beta must be in [-1,1]")
    random_state = get_random_state_generator(random_state)

    # CMS base variables
    U = random_state.uniform(-np.pi/2, np.pi/2, size)
    W = random_state.exponential(1.0, size)

    # Small tolerance for alpha=1 detection
    tol = 1e-12

    if abs(alpha - 1.0) > tol:
        # -------- alpha != 1 case --------

        # theta0 = (1/alpha) arctan(beta tan(pi alpha / 2))
        theta0 = (1.0 / alpha) * np.arctan(beta * np.tan(np.pi * alpha / 2))

        # Precompute trig terms
        alphaU = alpha * U
        cosU = np.cos(U)

        # Prevent division by extremely small cos(U)
        #cosU = np.where(np.abs(cosU) < 1e-14, 1e-14, cosU)

        term1 = (
            (np.sin(alphaU) + np.tan(alpha * theta0) * np.cos(alphaU))
            / cosU
        )

        inner = (
            np.cos((alpha - 1) * U)
            - np.tan(alpha * theta0) * np.sin((alpha - 1) * U)
        ) / (W * cosU)

        # Use log-exp form for numerical stability
        exponent = (1.0 - alpha) / alpha
        X = term1 * np.exp(exponent * np.log(inner))

    else:
        # -------- alpha = 1 case --------

        # Stable limit formula
        betaU = beta * U
        part1 = (np.pi / 2 + betaU) * np.tan(U)
        part2 = beta * np.log((np.pi / 2 * W * np.cos(U)) / (np.pi / 2 + betaU))

        X = (2 / np.pi) * (part1 - part2)

    return X
