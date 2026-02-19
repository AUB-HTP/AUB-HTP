from aub_htp import alpha_stable, multivariate_alpha_stable, BaseSpectralMeasureSampler
from aub_htp.random import IsotropicSampler
from scipy.stats import levy_stable


alpha = 0.6
beta = 0.3

#ok = alpha_stable.rvs([alpha, 0.3], [beta, 0.7], loc = [2,3], scale = 3, size = (10,2))

#print(ok)
import numpy as np

np.random.seed(0)

#sampler = IsotropicSampler(4, 0.4, 5)
samples = multivariate_alpha_stable.rvs(0.3, 1, "standard_isotropic_2d")

print(samples)