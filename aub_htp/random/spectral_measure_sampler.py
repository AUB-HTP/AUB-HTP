import numpy as np
from scipy.special import gamma
import logging
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

    @abstractmethod
    def mass(self) -> float:
        pass


class IsotropicSampler(BaseSpectralMeasureSampler):

    def __init__(self,
        number_of_dimensions: int,
        alpha: float,
        gamma: float,
    ):
        self.number_of_dimensions = number_of_dimensions
        self.alpha = alpha
        self.gamma = gamma
        self._mass = 1

    def sample(self, number_of_samples: int) -> np.ndarray:
        X = np.random.normal(size=(number_of_samples, self.number_of_dimensions))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        corr= self.__class__.isotropic_scale_correction(self.dimensions(), self.alpha, self.gamma)
        return corr * X  # I fixed the scaling issue :) ~ Wael
    
    @staticmethod
    def isotropic_scale_correction(d, alpha, gamma_scale):
        m_d_alpha = (
                gamma((alpha + 1) / 2)
                * gamma(d / 2)
                / (np.sqrt(np.pi) * gamma((d + alpha) / 2))
        )
        return gamma_scale * (m_d_alpha ** (-1.0 / alpha))

    def dimensions(self) -> int:
        return self.number_of_dimensions

    def mass(self) -> float:
        return self._mass

class EllipticSampler(BaseSpectralMeasureSampler):

    def __init__(self,
        number_of_dimensions: int,
        alpha: float,
        sigma: np.ndarray
    ):
        self.alpha = alpha
        self.number_of_dimensions = number_of_dimensions
        self.alpha = alpha
        self.sigma = np.asarray(sigma)
        self._mass = self._estimate_mass()

    def sample(self, number_of_samples: int) -> np.ndarray:
        X = np.random.normal(size=(number_of_samples, self.number_of_dimensions))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        corr= self.__class__.isotropic_scale_correction(self.dimensions(), self.alpha, gamma_scale=1)
        L = np.linalg.cholesky(self.sigma)
        return corr * X @ L.T

    def dimensions(self) -> int:
        return self.number_of_dimensions
    
    def mass(self) -> float:
        return self._mass

    @staticmethod
    def isotropic_scale_correction(d, alpha, gamma_scale):
        m_d_alpha = (
                gamma((alpha + 1) / 2)
                * gamma(d / 2)
                / (np.sqrt(np.pi) * gamma((d + alpha) / 2))
        )
        return gamma_scale * (m_d_alpha ** (-1.0 / alpha))

    def _estimate_mass(self, number_of_samples_taken_for_accuracy: int = 1000000):
        U = np.random.normal(size=(number_of_samples_taken_for_accuracy, self.dimensions()))
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        L = np.linalg.cholesky(self.sigma)
        norms = np.linalg.norm(U @ L.T, axis=1) ** self.alpha
        return np.mean(norms)


class DiscreteSampler(BaseSpectralMeasureSampler):

    def __init__(self,
        positions: np.ndarray,
        weights: np.ndarray
    ):
        self.positions = np.asarray(positions)
        self.weights = np.asarray(weights)
        assert self.positions.shape[0] == self.weights.shape[0] and self.positions.shape[0] > 0
        self.number_of_dimensions = self.positions.shape[1]
        self._mass = self.weights.sum()

    def sample(self, number_of_samples: int) -> np.ndarray:
        indices = np.random.choice(len(self.weights), size=number_of_samples, p=self.weights / self.weights.sum())
        return self.positions[indices]

    def dimensions(self) -> int:
        return self.number_of_dimensions

    def mass(self) -> float:
        return self._mass


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
        self._mass = self._calculate_mass()

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

    def mass(self) -> float:
        return float(self._mass)

    def _calculate_mass(self):
        return np.sum(
            spectral_measure.mass() * weight
                for spectral_measure, weight in zip(self.spectral_measures, self.weights)
        )
    
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
                "When α ≥ 1, If ∫_{S^{d−1}} s Λ(ds) != 0, then the resulting computations are not correct. "
            )
        if total_mass_of_sphere is None:
            logging.warning(
                "warning : the total mass of the unit sphere was not provided, so the computations are carried out assuming the total mass of the sphere is 1"
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
    
class UnivariateSampler(BaseSpectralMeasureSampler):
    def __init__(self,
        alpha: float,
        beta: float,
        gamma :float
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma= gamma 
        self._mass= self.mass()       

        if self.alpha >= 1 and self.beta != 0:
            logging.warning(
                "For alpha >= 1, beta should be 0, else the computations are not correct. "
            )

    def sample(self, number_of_samples: int) -> np.ndarray:
        p_plus = (1.0 + self.beta) / 2.0
        signs = np.where(
            np.random.rand(number_of_samples) <= p_plus,
            1.0,
            -1.0
        ).reshape(-1, 1) # reshape to (n, 1) since we expect vectors
        return signs

    def dimensions(self) -> int:
        return 1
    def mass(self) ->float:
        return self.gamma**(self.alpha)   

