import numpy as np
import math
import matplotlib.pyplot as plt
import sympy as s
from scipy.stats import levy_stable
from scipy.special import psi, gammaln

''' RNG for Univariate Alpha Stable '''


def Sample_Univariate_LePage(alpha, beta, gamma, delta, k, N, M):
    assert not (alpha > 1 and abs(beta) > 1e-12), \
        "For alpha > 1, skewness beta must be 0."
    '''
    This samples a univariate alpha stable distribution using LePage Series

    :param alpha: stability index 
    :param beta: skewness parameter
    :param gamma: scale parameter
    :param delta: location parameter
    :param k: parametrization type
    :param N: number of terms taken from the LePage series
    :param M: number of samples
    '''

    p_plus = (1 + beta) / 2
    c = (kappa(alpha)) ** (-1.0 / alpha)
    X = np.empty(M)
    for m in range(M):
        E = np.random.exponential(1.0, size=N)
        W = np.cumsum(E) ** (-1.0 / alpha)
        S = np.where(np.random.rand(N) <= p_plus, 1.0, -1.0)
        X[m] = c * np.sum(S * W)
        # Type 0 Parametrization
    if k == 0:
        if abs(alpha - 1) <= 1e-12:
            return (gamma * X) + delta
        else:
            return gamma * (X - (beta * np.tan((np.pi * alpha) / 2))) + delta
            return (gamma * X) + delta - (beta * np.tan((np.pi * alpha) / 2))
    else:
        # Type 1 Parametrization
        if k == 1:
            if abs(alpha - 1) <= 1e-12:
                return (gamma * X) + delta + (beta * (2 / np.pi) * gamma * np.log(gamma))
            else:
                return (gamma * X) + delta
    return X


'''RNG for Multivariate Alpha Stable'''


class SpectralMeasure:
    '''
    This class defines a spectral measure as an object with dimension d,
    a sampler function, and the total mass of the unit sphere.
    '''

    def __init__(self, d, sampler, sphere_mass):
        self.d = d
        self.sampler = sampler
        self.sphere_mass = float(sphere_mass)


def Estimate_Spectral_Measure_Subguassian(alpha, d, M, sigma):
    '''
    This estimates the sepctral measure of the unit sphere for the Subguassian Random Vector
    :param alpha: stability index
    :param N: number of terms taken from the LePage series
    :param d: dimension
    :param M: number of vectors sampled
    :param sigma: shape matrix of the subguassian vector
    '''
    U = np.random.normal(size=(M, d))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    L = np.linalg.cholesky(sigma)
    norms = np.linalg.norm(U @ L.T, axis=1) ** alpha
    return np.mean(norms)


def kappa(alpha):
    '''
    this function computes a constant that is needed for the LePage series
    :param alpha: stability index
    '''
    if abs(alpha - 1.0) < 1e-12:
        return math.pi / 2
    return math.gamma(2 - alpha) * math.cos(math.pi * alpha / 2) / (1 - alpha)


def weights(alpha, N):
    '''
    Docstring for weights
    This function computes the weights of the LePage Series

    :param alpha: the stability index
    :param N: number of terms taken from the LePage series
    '''
    return np.cumsum(np.random.exponential(1.0, size=N)) ** (-1.0 / alpha)


def isotropic_scale_correction(d, alpha, gamma_scale):
    '''
    This function computes the constant which rescales the sampled Isotropic random vector to have scale gamma
    :param d: the dimension
    :param alpha:the stability index
    :param gamma_scale: desired scale
    '''
    m_d_alpha = (
            math.gamma((alpha + 1) / 2)
            * math.gamma(d / 2)
            / (math.sqrt(math.pi) * math.gamma((d + alpha) / 2))
    )
    return gamma_scale * (m_d_alpha ** (-1.0 / alpha))


def Isotropic_Spectral_Measure(d, N):
    '''
    This function returns vectors that are uniformly distributed on the unit sphere
    :param d: dimension
    :param N: number of terms taken from the LePage series
    '''
    X = np.random.normal(size=(N, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def Subguassian_Spectral_Measure(N, d, sigma):
    '''
    This estimates the sepctral measure of the unit sphere for the Subguassian Random Vector
    :param alpha: stability index
    :param N: number of terms taken from the LePage series
    :param d: dimension
    :param M: number of vectors sampled
    :param sigma: shape matrix of the subguassian vector
    '''
    U = np.random.normal(size=(N, d))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    L = np.linalg.cholesky(sigma)
    return U @ L.T


def Sample_Lepage_One_Point_Isotropic(alpha, N, d, gamma_scale):
    '''
    this function samples one vector from the isotropic spectral measure with unit scale

    :param alpha: stability index
    :param N: number of terms taken from the LePage series
    :param d: dimension
    :param gamma_scale: desired scale
    :total_measure_of_unit_sphere: total spectral measure of the unit sphere
    '''
    W = weights(alpha, N)
    V = Isotropic_Spectral_Measure(d, N)
    c = (kappa(alpha)) ** (-1.0 / alpha)
    scale_corr = isotropic_scale_correction(d, alpha, gamma_scale)
    return scale_corr * c * np.sum(W[:, None] * V, axis=0)


def Sample_LePage_Isotropic(alpha, N, d, M, gamma_scale=1):
    '''
    this function samples many vectors from the isotropic spectral measure with unit scale

    :param alpha: stability index
    :param N: number of terms taken from the LePage series
    :param d: dimension
    :param M: number of vectors sampled
    :param gamma_scale: desired scale
    :param total_measure_of_unit_sphere: total spectral measure of the unit sphere
    '''
    samples = np.empty((M, d))
    for i in range(M):
        samples[i] = Sample_Lepage_One_Point_Isotropic(alpha, N, d, gamma_scale)
    return samples


def Sample_LePage_Subguassian(alpha, N, d, M, sigma):
    '''
    this function samples one vector from the subguassian spectral measure

    :param alpha: stability index
    :param N: number of terms taken from the LePage series
    :param d: dimension
    :param M: number of vectors sampled
    :param sigma: shape matrix of the subguassian vector
    '''
    Z = Sample_LePage_Isotropic(alpha, N, d, M)
    L = np.linalg.cholesky(sigma)
    return Z @ L.T


def Sample_LePage_One_Point_Discrete(N, masses, weights_of_masses):
    '''
    this function samples one vector from an atomic spectral measure

    :param N: number of terms taken from the LePage series
    :param masses: locations of the vector masses
    :param weights_of_masses: spectral measure of each mass
    '''
    k = len(masses)
    probs = np.zeros(k)
    s = np.sum(weights_of_masses)
    probs = [weights_of_masses[i] / s for i in range(k)]
    indices = np.random.choice(k, size=N, p=probs)
    return masses[indices]


def Sample_LePage_Discrete(alpha, N, d, M, masses, weights_of_masses):
    '''
    this function samples M multivariate stable vectors using LePage series with a discrete spectral measure

    :param alpha: stability index
    :param N: number of terms taken from the LePage series
    :param M: number of vectors sampled
    :param d: dimension
    :param masses: locations of the vector masses
    :param weights_of_masses: spectral measure of each mass
    '''
    samples = np.empty((M, d))
    c = (kappa(alpha) / np.sum(weights_of_masses)) ** (-1 / alpha)
    for i in range(M):
        W = weights(alpha, N)
        V = Sample_LePage_One_Point_Discrete(N, masses, weights_of_masses)
        samples[i] = c * np.sum(W[:, None] * V, axis=0)
    return samples


def categorical_choice(weights, size):
    '''
    This function draws indicies of spectral measures randomly according
    to their normalized weights so that they become probabilities

    :param weights: array of weights for spectral measures
    :param size: number of sampled vectors
    '''
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    return np.random.choice(len(weights), size=size, p=weights)


def Mixed_Spectral_Measure(N, d, spectral_measures, weights):
    '''
    This function samples a vector v according to a mixed spectral measure.

    :param spectral_samplers: array of spectral measures
    :param weights: array of weights for spectral measures
    :param N: number of terms taken from the LePage series
    :param d: dimension
    '''
    K = len(spectral_measures)
    assert len(weights) == K
    choices = categorical_choice(weights, N)
    V = np.empty((N, d))
    for k in range(K):
        idx = np.where(choices == k)[0]
        if len(idx) == 0:
            continue
        V[idx] = spectral_measures[k].sampler(len(idx))
    return V


def Sample_LePage_Mixed(alpha, N, d, M, spectral_measures, weights):
    '''
    This function samples a multivariate stable vector according to a mixed spectral measure.

    :param alpha: stability index
    :param N: number of terms taken from the LePage series
    :param d: dimension
    :param M: number of vectors sampled
    :param spectral_measures: array of spectral measures
    :param weights: array of weights for spectral measures
    '''
    samples = np.empty((M, d))
    masses_of_sphere = np.array([spectral_measures[i].get_mass() for i in range(len(weights))])
    weights = np.asarray(weights, dtype=float)
    total_mass_of_sphere = np.sum(weights * masses_of_sphere)
    c = (kappa(alpha) / total_mass_of_sphere) ** (-1.0 / alpha)

    for m in range(M):
        E = np.random.exponential(1.0, size=N)
        W = np.cumsum(E) ** (-1.0 / alpha)
        V = Mixed_Spectral_Measure(N, d, spectral_measures, weights)
        samples[m] = c * np.sum(W[:, None] * V, axis=0)
    return samples


def Sample_Multivariate_LePage(alpha, N, d, M, spectral_measure, delta=None, gamma=1, sigma=None, masses=None,
                               weights_of_masses=None, spectral_measures=None, weights_of_spectral_measures=None):
    '''
    Docstring for Sample_Multivariate_LePage

    :param alpha: stability index
    :param N: number of terms taken from the LePage series
    :param d: dimension
    :param M: number of sampled vectors
    :param spectral_measure: type of spectral measure
    :param gamma: scale parameter for Isotropic spectral measures
    :param sigma: shape matrix for Subguassian random vectors
    :param masses: locations of the vector masses
    :param weights_of_masses: spectral measure of each mass
    :param spectral_measures: array of spectral measures
    :param weights_of_spectral_measures: weights of spectral measures
    '''
    spectral_measure = spectral_measure.lower()

    if spectral_measure == "isotropic":
        if delta is not None:
            return delta + Sample_LePage_Isotropic(alpha, N, d, M, gamma)
        else:
            return Sample_LePage_Isotropic(alpha, N, d, M, gamma)
    elif spectral_measure == "elliptic":
        if delta is not None:
            return delta + Sample_LePage_Subguassian(alpha, N, d, M, sigma)
        else:
            return Sample_LePage_Subguassian(alpha, N, d, M, sigma)
    elif spectral_measure == "atomic":
        if delta is not None:
            return delta + Sample_LePage_Discrete(alpha, N, d, M, masses, weights_of_masses)
        else:
            return Sample_LePage_Discrete(alpha, N, d, M, masses, weights_of_masses)
    elif spectral_measure == "mixed":
        if delta is not None:
            return delta + Sample_LePage_Mixed(alpha, N, d, M, spectral_measures, weights_of_spectral_measures)
        else:
            return Sample_LePage_Mixed(alpha, N, d, M, spectral_measures, weights_of_spectral_measures)
    else:
        raise ValueError(f"Unknown spectral measure type: {spectral_measure}")


def pdf_cauchy_1d(x, gamma):
    return 1.0 / (math.pi * gamma * (1.0 + (x / gamma) ** 2))


def pdf_independent_cauchy_2d(x, gamma1, gamma2):
    x = np.atleast_2d(x)
    return pdf_cauchy_1d(x[:, 0], gamma1) * pdf_cauchy_1d(x[:, 1], gamma2)


def pdf_cauchy_nd(x, d, gamma):
    x = np.atleast_2d(x)
    r2 = np.sum(x ** 2, axis=1)
    const = math.gamma((d + 1) / 2.0) / (
            (math.pi ** ((d + 1) / 2.0)) * (gamma ** d)
    )
    denom = (1 + r2 / (gamma ** 2)) ** ((d + 1) / 2.0)
    return const / denom


def closed_entropy_isotropic_cauchy(d, gamma):
    term1 = (
            (d + 1) / 2.0
            * (math.log(4 * math.pi) + s.EulerGamma.evalf() + psi((d + 1) / 2.0))
    )
    term2 = -gammaln((d + 1) / 2.0)
    return term1 + term2 + d * math.log(gamma)


Sigma1 = np.array([[2, 0.8],
                   [0.8, 1.5]])  # horizontal major axis

Sigma2 = np.array([[2, -1],
                   [-1, 2]])  # vertical major axis

alpha = 1
N = 500
M = 500000
d = 2
masses1 = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
masses2 = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])
weight = np.array([0.25, 0.25, 0.25, 0.25])


# X = Sample_Multivariate_LePage(alpha=alpha,N=N,gamma=2,M=M,d=2,spectral_measure="isotropic",delta=np.array([1000 for i in range(d)]))
# X = Sample_Multivariate_LePage(alpha,N,d,M,spectral_measure="elliptic",delta=np.array([1000 for i in range(d)]),sigma=Sigma2,)
# X = Sample_Multivariate_LePage(alpha,N,d,M,spectral_measure="atomic",masses=masses,weights_of_masses=weight,delta=np.array([1000 for i in range(d)]))

def sampler1(k):
    return Isotropic_Spectral_Measure(d, k)


def sampler2(k):
    return Sample_LePage_One_Point_Discrete(k, masses=masses1, weights_of_masses=weight)


def sampler3(k):
    return Subguassian_Spectral_Measure(k, d, Sigma1)


def sampler4(k):
    return Subguassian_Spectral_Measure(k, d, Sigma2)


def sampler5(k):
    return Sample_LePage_One_Point_Discrete(k, masses2, weights_of_masses=weight)


SM1 = SpectralMeasure(d, sampler1, 1)
SM2 = SpectralMeasure(d, sampler2, 1)
SM3 = SpectralMeasure(d, sampler3, Estimate_Spectral_Measure_Subguassian(alpha, d, 200000, Sigma1))
SM4 = SpectralMeasure(d, sampler4, Estimate_Spectral_Measure_Subguassian(alpha, d, 200000, Sigma2))
SM5 = SpectralMeasure(d, sampler5, 1)
spec_measures = [SM1, SM2, SM4]
X = Sample_Multivariate_LePage(alpha, N, d, M, "mixed", delta=np.array([1000 for i in range(d)]),
                               spectral_measures=spec_measures, weights_of_spectral_measures=np.array([1, 1, 1]))
plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], s=0.1, alpha=0.3, color='blue')
plt.grid(True)
plt.axis("equal")
plt.xlim((-400 + 1000, 400 + 1000))
plt.ylim((-400 + 1000, 400 + 1000))
plt.title("Isotropic 2D Cauchy (LePage)")
plt.show()
