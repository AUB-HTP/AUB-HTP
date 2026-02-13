import numpy as np
from scipy.special import gamma
from abc import abstractmethod, ABC

class BaseSpectralMeasureSampler(ABC):
    '''
    A Spectral Measure Sampler is as an interface which defines:
        - self.sample(): sampling algorithm (self.sample)
        - self.mass(): the mass of the spectral measure
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
        mass: float
    ):
        self.number_of_dimensions = number_of_dimensions
        self.mass = mass

    def sample(self, number_of_samples: int) -> np.ndarray:
        X = np.random.normal(size=(number_of_samples, self.number_of_dimensions))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X * np.sqrt(self.mass)

    def dimensions(self) -> int:
        return self.number_of_dimensions


class EllipticSampler(BaseSpectralMeasureSampler):

    def __init__(self,
        number_of_dimensions: int,
        sigma: np.ndarray
    ):
        self.number_of_dimensions = number_of_dimensions
        self.sigma = np.asarray(sigma)

    def sample(self, number_of_samples: int) -> np.ndarray:
        X = np.random.normal(size=(number_of_samples, self.number_of_dimensions))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        L = np.linalg.cholesky(self.sigma)
        return X @ L.T

    def dimensions(self) -> int:
        return self.number_of_dimensions


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