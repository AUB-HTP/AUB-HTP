import aub_htp as ht
from aub_htp.random import IsotropicSampler, EllipticSampler, DiscreteSampler, MixedSampler

alpha = 1.2

isotropic = IsotropicSampler(number_of_dimensions = 2 , alpha = alpha, gamma = 2)
elliptic = EllipticSampler(number_of_dimensions = 2, alpha = alpha, sigma = [[10, 2], [2, 50]])
discrete = DiscreteSampler(positions = [[0, 1], [-1, 0]],  weights = [0.2, 0.8])

mixed = MixedSampler(spectral_measures = [isotropic, elliptic, discrete], weights = [0.4, 0.4, 0.2])

samples = ht.multivariate_alpha_stable.rvs(alpha = 1.2, spectral_measure_sampler = mixed, shift = 0, size = 10)

print(samples)