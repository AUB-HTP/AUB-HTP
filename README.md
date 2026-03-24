# AUB-HTP

American University of Beirut's Heavy Tails Package (AUB-HTP) aims to provide a modern python toolkit for analyzing Alpha Stable Distributions (also called Levy-Stable distributions)

As of date, this repository encompasses a **scipy-compatible** frontend to **generate the PDF** as well as **sample random numbers** from a univariate Alpha Stable distribution. Moreover, the package supports sampling from a **multivariate** Alpha Stable distribution.

### Installation

```commandline
pip install aub-htp
```

For alternative installation methods (uv, conda, poetry), see [docs/install](docs/source/install.rst).

### Univariate Alpha Stable Distributions

#### Probability Density Function (PDF) Generation
```python
import aub_htp as ht
import numpy as np

x = np.linspace(-10, 10, 100)
y = ht.alpha_stable.pdf(x, alpha = 0.4, beta = 0.8, loc = 2, scale = 3)
```

You can also switch between [parametrizations](https://en.wikipedia.org/wiki/Stable_distribution#Parametrizations) `S0` and `S1` (default).
```python
import aub_htp as ht
import numpy as np

x = np.linspace(-10, 10, 100)
ht.alpha_stable.with_parametrization("S0")
y = ht.alpha_stable.pdf(x, alpha = 0.4, beta = 0.8, loc = 2, scale = 3)
```

![docs/examples/plot_parametrizations.png](docs/examples/plot_parametrizations.png)

#### Random Variable Sampling (RVS)
```python
import aub_htp as ht

sample = ht.alpha_stable.rvs(alpha = 0.4, beta = 0.8, loc = 2, scale = 3)
print(sample)
# --------------------------------------------------------------------
# 32712.16830783209
```

```python
import aub_htp as ht

ht.alpha_stable.with_parametrization("S0")

samples = ht.alpha_stable.rvs(alpha = 0.4, beta = 0.8, loc = 2, scale = 3, size = (2,5), random_state = 38)
print(samples)
# --------------------------------------------------------------------
# [[  3.23516232   2.69285354   6.42379299   2.47262474   0.9449036 ]
#  [259.96755723   0.6227459  118.69956139   3.63961872   3.45083368]]
```

![docs/examples/plot_univariate_sampling.png](docs/examples/plot_univariate_sampling.png)

### Multivariate Alpha Stable Distributions

#### Random Variable Sampling (RVS)

Sampling from a multivariate alpha stable distribution requires knowledge of the underlying mathematical spectral measure you wish to sample against.
AUB-HTP's standard library of spectral measure samplers includes:
- Isotropic Spectral Measure Sampler
- Elliptic Spectral Measure Sampler
- Discrete Spectral Measure Sampler
- Mixed Spectral Measure Sampler
- Univariate Spectral Measure Sampler

Example 0: Sampling from the standard 2d isotropic spectral measure.
```python
import aub_htp as ht

samples = ht.multivariate_alpha_stable.rvs(alpha = 1.2, spectral_measure_sampler = "standard_isotropic_2d")
print(samples)
```


Example 1: Sampling from an isotropic spectral measure.
```python
import aub_htp as ht
from aub_htp.random import IsotropicSampler

alpha = 1.2
sampler = IsotropicSampler(number_of_dimensions= 2, alpha = alpha, gamma = 2)

samples = ht.multivariate_alpha_stable.rvs(alpha = alpha, spectral_measure_sampler = sampler, size = 10)
print(samples)
```

Example 2: Sampling from a mixed spectral measure.

```python
import aub_htp as ht
from aub_htp.random import IsotropicSampler, EllipticSampler, DiscreteSampler, MixedSampler

alpha = 0.6

isotropic = IsotropicSampler(number_of_dimensions = 2 , alpha = alpha, gamma = 2)
elliptic = EllipticSampler(number_of_dimensions = 2, alpha = alpha, sigma = [[10, 2], 
                                                                             [2, 50]])
discrete = DiscreteSampler(alpha = alpha, positions = [[0, 1], [-1, 0]],  weights = [0.2, 0.8])

mixed = MixedSampler(spectral_measures = [isotropic, elliptic, discrete], weights = [0.4, 0.4, 0.2])

samples = ht.multivariate_alpha_stable.rvs(alpha = alpha, spectral_measure_sampler = mixed, shift = [4, 2], size = 10)

print(samples)
```

#### Custom Spectral Measures

You can also extend the standard library collection by extending the base class `BaseSpectralMeasureSampler`. 

**Important Note**: When sampling an alpha stable vector against your custom defined spectral measure with $alpha \ge 1$, your custom spectral measure $\Lambda$ has to uphold the following property:
$$\int_{\mathbb{S}^{d-1}}s\Lambda(ds)=0$$

Example: Butterfly Spectral Measure Sampler
```python
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
```

![docs/examples/plot_multivariate_sampling.png](docs/examples/plot_multivariate_sampling.png)

### Papers and Further Readings