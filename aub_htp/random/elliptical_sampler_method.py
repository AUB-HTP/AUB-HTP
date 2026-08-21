import numpy as np
from .cms_univariate_sampler import sample_cms
from .util import get_random_state_generator

def sample_elliptical(alpha, Sigma, n_samples, random_state: None | int | np.random.RandomState | np.random.Generator = None):
    """
    Generate subgaussian stable random vectors using the definition.
    
    X = √A · G where:
    - A ~ S_{α/2}((cos(πα/4))^(2/α), 1, 0)
    - G ~ N(0, Sigma)
    
    Parameters:
    -----------
    alpha : float
        Stability parameter (0 < alpha < 2)
    Sigma : ndarray
        Shape matrix (d x d covariance matrix)
    n_samples : int
        Number of samples to generate
    
    Returns:
    --------
    X_samples : ndarray
        (n_samples, d) array of subgaussian stable samples
    """
    random_state = get_random_state_generator(random_state)
    
    # Generate amplitude A ~ S_{α/2}
    alpha_A = alpha / 2
    beta_A = 1
    scale_A = (np.cos(np.pi * alpha / 4)) ** (2 / alpha)
    loc_A = 0
    
    A_samples = sample_cms(
        alpha=alpha_A,
        beta=beta_A,
        size=n_samples,
        random_state = random_state,
    ) * scale_A + loc_A
    
    # Generate Gaussian samples G ~ N(0, Sigma)
    d = Sigma.shape[0]
    G_samples = random_state.multivariate_normal(
        mean=np.zeros(d),
        cov=Sigma,
        size=n_samples
    )
    
    # Element-wise multiplication: √A · G
    sqrt_A = np.sqrt(A_samples)
    X_samples = sqrt_A[:, None] * G_samples
    
    return X_samples