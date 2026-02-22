import numpy as np
from aub_htp.random import IsotropicSampler, DiscreteSampler, sample_alpha_stable_vector

positions = np.linspace(-100, 100, 21)
positions = np.vstack([positions, np.zeros_like(positions)]).T
weights = np.ones(len(positions))/len(positions)

sampler_discrete = DiscreteSampler(positions, weights)

print(positions)
print(weights)
print(sampler_discrete.sample(4))

sampler_isotropic = IsotropicSampler(2, 1)
print(sampler_isotropic.sample(4))

samples = sample_alpha_stable_vector(1.2, sampler_discrete, 1)

print(samples)
