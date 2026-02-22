import aub_htp as ht
from aub_htp.random import BaseSpectralMeasureSampler
import numpy as np

class ButterflySampler(BaseSpectralMeasureSampler):
    def sample(self, number_of_samples: int, random_state = None):
        p = np.random.rand(number_of_samples)
        theta = np.empty(number_of_samples)

        mask = p <= 0.5
        theta[mask] = np.random.uniform(-np.pi / 4, np.pi / 4, size=mask.sum())
        theta[~mask] = np.random.uniform(
            3 * np.pi / 4,
            5 * np.pi / 4,
            size=(~mask).sum()
        )

        x = np.cos(theta)
        y = np.sin(theta)

        return np.column_stack((x, y))

    def dimensions(self) -> int:
        return 2

    def mass(self) -> float:
        return 1.0

samples = ht.multivariate_alpha_stable.rvs(alpha = 0.8, spectral_measure_sampler=ButterflySampler(), size = 10000)

import matplotlib.pyplot as plt
plt.xlim(-100, 100)
plt.ylim(-100, 100)
plt.scatter(samples[:, 0], samples[:, 1])
plt.savefig("alpha_stable.png")