import aub_htp as ht
from aub_htp import DiscreteSampler
from aub_htp.random import BaseSpectralMeasureSampler, IsotropicSampler, EllipticSampler, MixedSampler


isotropic = IsotropicSampler(2, 1.2, 2)
elliptic = EllipticSampler(2, 1,[[10, 2], [2, 50]])
discrete = DiscreteSampler([1, 3], [0.2, 0.8])
mixed = MixedSampler([isotropic, elliptic], [0.2, 0.8])


samples = ht.multivariate_alpha_stable.rvs(1.2, 0, "coin_flip_discrete", 10)

print(samples)