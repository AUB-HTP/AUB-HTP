import numpy as np
import math
import matplotlib.pyplot as plt
import sympy as s
from scipy.stats import levy_stable
from scipy.special import psi, gammaln
from aub_htp.random import BaseSpectralMeasureSampler, IsotropicSampler, EllipticSampler, DiscreteSampler, MixedSampler, UnivariateSampler
from aub_htp import sample_alpha_stable_vector
import logging

def pdf_cauchy_1d(x, gamma):
    return 1.0 / (math.pi * gamma * (1.0 + (x / gamma) ** 2))

class CustomSampler(BaseSpectralMeasureSampler):

    def __init__(self,
                 alpha: float,
                 dimension: int,
                 given_sampler,
                 total_mass_of_sphere: float | None = None):

        self._dimension = dimension
        self._given_sampler = given_sampler
        self._alpha = alpha
        if self._alpha >= 1:
            logging.warning(
                "α ≥ 1, If ∫_{S^{d−1}} s Λ(ds) != 0, then the resulting computations are not correct."
            )
        if total_mass_of_sphere is None:
            logging.warning(
                "The total mass of the unit sphere was not provided, so the computations are carried out assuming the total mass of the sphere is 1"
            )
            self._mass = 1.0
        else:
            self._mass = float(total_mass_of_sphere)

    def sample(self, number_of_samples: int) -> np.ndarray:
        samples = self._given_sampler(number_of_samples)

        if samples.shape != (number_of_samples, self._dimension):
            raise ValueError(
                f"Sampler must return shape "
                f"({number_of_samples}, {self._dimension})"
            )

        return samples

    def dimensions(self) -> int:
        return self._dimension

    def mass(self) -> float:
        return self._mass

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

sigma3=np.array([[4.36,-0.35,3.96]
                 ,[-0.35,0.29,-0.2],
                 [3.96, -0.2, 5.6]])
alpha = 0.5
N = 10
M = 1000000
d = 2
gamma=1
masses1 = np.array([[0, 1], [0, -1], [1,0], [-1,0]])
masses2 = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])
weight = np.array([1,1,0.1,0.1])


'''
SP1=IsotropicSampler(d,alpha,gamma=gamma)
SP2=EllipticSampler(d,alpha,Sigma1)
SP3=EllipticSampler(d,alpha,Sigma2)
SP7=EllipticSampler(d,alpha,sigma3)
SP4=DiscreteSampler(masses1,weight)
SP5=DiscreteSampler(masses2,weight)
spec_measures = [SP4,SP5]
SP6=MixedSampler(spec_measures,[0.5,0.5])

def sample_two_arc_points(N):
    """
    Returns an (N,2) numpy array of points sampled as follows:
    - Flip a fair coin (p <= 0.5).
    - If true:  sample theta ~ Uniform[-pi/4, pi/4]
    - Else:     sample theta ~ Uniform[pi - pi/4, pi + pi/4]
    - Return (cos(theta), sin(theta))
    """
    # Fair coin for each sample
    p = np.random.rand(N)

    theta = np.empty(N)

    mask = p <= 0.5

    # First arc
    theta[mask] = np.random.uniform(
        -np.pi/4,
        np.pi/4,
        size=mask.sum()
    )

    # Second arc
    theta[~mask] = np.random.uniform(
        np.pi - np.pi/4,
        np.pi + np.pi/4,
        size=(~mask).sum()
    )

    x = np.cos(theta)
    y = np.sin(theta)

    return np.column_stack((x, y))

SP7=CustomSampler(alpha,2,sample_two_arc_points,4)
SP8=DiscreteSampler(masses1,weight)
SP9=MixedSampler([SP7,SP8],np.array([1,1]))
X =sample_alpha_stable_vector(alpha,SP9,M,N) 

h_MC=-np.mean(np.log(pdf_cauchy_nd(X,d,gamma=gamma)))
h_true=closed_entropy_isotropic_cauchy(d,gamma=gamma)

print("h_MC :", h_MC)
print("h_true :", h_true)
print(abs(h_MC-h_true))

plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], s=0.1, alpha=0.3, color='blue')
plt.grid(True)
plt.axis("equal")
plt.xlim((-400 , 400))
plt.ylim((-400 , 400 ))
plt.title("Isotropic 2D Cauchy (LePage)")
plt.show()
'''

alpha=1.5
beta=0
gamma=3
delta=np.zeros(1)
delta[0]=5000
m=500000
Spectral_Measure=UnivariateSampler(alpha=alpha,beta=beta,gamma=gamma)
X=sample_alpha_stable_vector(alpha,Spectral_Measure,m,delta)



# --- Truncate range ---
xmin=-10000+delta[0]
xmax =10000+delta[0]
X_trunc = X[(X >= xmin) & (X <= xmax)]

# --- Histogram ---
counts, bins, _=plt.hist(X_trunc, bins=2000, density=True, color="#00e5ff",edgecolor="black")
# --- Grid for pdf ---
bin_centers=(bins[:-1]+bins[1:])/2

# SciPy parameterization:
# levy_stable(alpha, beta, loc=delta, scale=gamma)
pdf_vals = levy_stable.pdf(bin_centers, alpha, beta, loc=delta, scale=gamma)

# --- Renormalization for truncation ---
F_xmax = levy_stable.cdf(xmax, alpha, beta, loc=delta, scale=gamma)
F_xmin = levy_stable.cdf(xmin, alpha, beta, loc=delta, scale=gamma)

normalization = F_xmax - F_xmin
pdf_truncated = pdf_vals / normalization

# --- Overlay ---
plt.plot(bin_centers, pdf_truncated, 'black', lw=2, )
plt.xlim(xmin, xmax)
plt.legend()
plt.show()