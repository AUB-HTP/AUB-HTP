import numpy as np
from scipy.special import gamma
from abc import abstractmethod, ABC

class BaseSpectralMeasureSampler(ABC):
    '''
    A Spectral Measure Sampler is as an interface which defines:
        - self.sample(): sampling algorithm (self.sample)
        - self.dimensions(): the number of dimensions of the spectral measure
    '''

    @abstractmethod
    def sample(self, number_of_samples: int) -> np.ndarray:
        pass

    @abstractmethod
    def dimensions(self) -> int:
        pass


class IsotropicSampler(BaseSpectralMeasureSampler):

    def __init__(self,
        number_of_dimensions: int,
        alpha: float,
        gamma: float = 1.0,
    ):
        self.number_of_dimensions = number_of_dimensions
        self.alpha = alpha
        self.gamma = gamma

    def sample(self, number_of_samples: int) -> np.ndarray:
        X = np.random.normal(size=(number_of_samples, self.number_of_dimensions))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        corr= self.__class__.isotropic_scale_correction(self.dimensions(), self.alpha, self.gamma)
        return corr * X  # I fixed the scaling issue :) ~ Wael

    def dimensions(self) -> int:
        return self.number_of_dimensions

    @staticmethod
    def isotropic_scale_correction(d, alpha, gamma_scale):
        m_d_alpha = (
                gamma((alpha + 1) / 2)
                * gamma(d / 2)
                / (np.sqrt(np.pi) * gamma((d + alpha) / 2))
        )
        return gamma_scale * (m_d_alpha ** (-1.0 / alpha))


class EllipticSampler(BaseSpectralMeasureSampler):

    def __init__(self,
        number_of_dimensions: int,
        alpha: float,
        sigma: np.ndarray
    ):
        self.alpha = alpha
        self.number_of_dimensions = number_of_dimensions
        self.sigma = np.asarray(sigma)

    def sample(self, number_of_samples: int) -> np.ndarray:
        X = np.random.normal(size=(number_of_samples, self.number_of_dimensions))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        corr= self.__class__.isotropic_scale_correction(self.dimensions(), self.alpha, gamma_scale=1)
        L = np.linalg.cholesky(self.sigma)
        return corr * X @ L.T

    def dimensions(self) -> int:
        return self.number_of_dimensions
    
    @staticmethod
    def isotropic_scale_correction(d, alpha, gamma_scale):
        m_d_alpha = (
                gamma((alpha + 1) / 2)
                * gamma(d / 2)
                / (np.sqrt(np.pi) * gamma((d + alpha) / 2))
        )
        return gamma_scale * (m_d_alpha ** (-1.0 / alpha))


class DiscreteSampler(BaseSpectralMeasureSampler):
    def __init__(self,
        positions: np.ndarray,
        weights: np.ndarray
    ):
        self.positions = np.asarray(positions)
        self.weights = np.asarray(weights)
        assert self.positions.shape[0] == self.weights.shape[0] and self.positions.shape[0] > 0
        self.number_of_dimensions = self.positions.shape[1]

    def sample(self, number_of_samples: int) -> np.ndarray:
        indices = np.random.choice(len(self.weights), size=number_of_samples, p=self.weights / self.weights.sum())
        return self.positions[indices]

    def dimensions(self) -> int:
        return self.number_of_dimensions


class MixedSampler(BaseSpectralMeasureSampler):
    def __init__(self,
        spectral_measures: list[BaseSpectralMeasureSampler],
        weights: np.ndarray,
    ):
        assert len(weights) == len(spectral_measures)
        assert len(spectral_measures) > 0
        assert all(sprectral_measure.dimensions() == spectral_measures[0].dimensions() for sprectral_measure in spectral_measures)

        self.number_of_dimensions = spectral_measures[0].dimensions()
        self.spectral_measures = spectral_measures
        self.weights = np.asarray(weights)

    def sample(self, number_of_samples: int) -> np.ndarray:
        weights = self.weights / self.weights.sum()
        indices = np.random.choice(len(weights), size=number_of_samples, p=weights)

        samples = []
        for i in range(len(weights)):
            count = np.sum(indices == i)
            if count > 0:
                samples.append(self.spectral_measures[i].sample(count))

        return np.vstack(samples)

    def dimensions(self) -> int:
        return self.number_of_dimensions


class UnivariateSampler(BaseSpectralMeasureSampler):

    def __init__(self,
        alpha: float,
        beta: float,
    ):
        self.alpha = alpha
        self.beta = beta

    def sample(self, number_of_samples: int) -> np.ndarray:
        p_plus = (1.0 + self.beta) / 2.0
        signs = np.where(
            np.random.rand(number_of_samples) <= p_plus,
            1.0,
            -1.0
        ).reshape(-1, 1) # reshape to (n, 1) since framework expects vectors

        return signs

    def dimensions(self) -> int:
        return 1
    

