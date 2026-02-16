import numpy as np
import math
import matplotlib.pyplot as plt
import sympy as s
from scipy.stats import levy_stable
from scipy.special import psi, gammaln
from spectral_measure_sampler import BaseSpectralMeasureSampler,IsotropicSampler, EllipticSampler, DiscreteSampler, MixedSampler
from alpha_stable_sampler import sample_alpha_stable_vector
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
gamma=3
masses1 = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
masses2 = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])
weight = np.array([0.25, 0.25, 0.25, 0.25])

SP1=IsotropicSampler(d,alpha,gamma=gamma)

SP2=EllipticSampler(d,alpha,Sigma1)
SP3=EllipticSampler(d,alpha,Sigma2)
SP4=DiscreteSampler(masses1,weight)
SP5=DiscreteSampler(masses2,weight)
spec_measures = [SP4,SP5]
SP6=MixedSampler(spec_measures,[0.5,0.5])
X =sample_alpha_stable_vector(alpha,SP2,M,N) 
'''
h_MC=-np.mean(np.log(pdf_cauchy_nd(X,d,gamma=gamma)))
h_true=closed_entropy_isotropic_cauchy(d,gamma=gamma)

print("h_MC :", h_MC)
print("h_true :", h_true)
print(abs(h_MC-h_true))
'''
plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], s=0.1, alpha=0.3, color='blue')
plt.grid(True)
plt.axis("equal")
plt.xlim((-400 , 400))
plt.ylim((-400 , 400 ))
plt.title("Isotropic 2D Cauchy (LePage)")
plt.show()

