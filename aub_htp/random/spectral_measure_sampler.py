import numpy as np
from scipy.special import gamma
import logging
from abc import abstractmethod, ABC
from .util import get_random_state_generator

class BaseSpectralMeasureSampler(ABC):
    '''
    Spectral Measure Sampler is an interface which defines:
        - self.sample(number_of_samples: int) sampling algorithm (self.sample)
        - self.dimensions() the number of dimensions of the spectral measure
        - self.mass() the number of dimensions of the spectral measure

    The underlying mathematical spectral measure being sampled against :math:`\Lambda` has to uphold the following property:
        :math:`\int_{\mathbb{S}^{d-1}}s\Lambda(ds)=0`
    '''

    @abstractmethod
    def sample(self, number_of_samples: int, random_state: None | int | np.random.RandomState | np.random.Generator = None) -> np.ndarray:
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

    def sample(self, number_of_samples: int, random_state: None | int | np.random.RandomState | np.random.Generator = None) -> np.ndarray:
        random_state = get_random_state_generator(random_state)
        X = random_state.normal(size=(number_of_samples, self.number_of_dimensions))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        corr = isotropic_scale_correction(self.dimensions(), self.alpha, self.gamma)
        return corr * X

    def dimensions(self) -> int:
        return self.number_of_dimensions

    def mass(self) -> float:
        return self._mass


class EllipticSampler(BaseSpectralMeasureSampler):

    def __init__(self,
        number_of_dimensions: int,
        alpha: float,
        sigma: np.ndarray,
        mass: float | None = None,
    ):
        self.alpha = alpha
        self.number_of_dimensions = number_of_dimensions
        self.alpha = alpha
        self.sigma = np.asarray(sigma)
        self._mass = mass or self._estimate_mass()

    def sample(self, number_of_samples: int, random_state: None | int | np.random.RandomState | np.random.Generator = None) -> np.ndarray:
        random_state = get_random_state_generator(random_state)
        X = random_state.normal(size=(number_of_samples, self.number_of_dimensions))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        corr = isotropic_scale_correction(self.dimensions(), self.alpha, gamma_scale=1)
        L = np.linalg.cholesky(self.sigma)
        return corr * X @ L.T

    def dimensions(self) -> int:
        return self.number_of_dimensions
    
    def mass(self) -> float:
        return float(self._mass)

    def _estimate_mass(self, number_of_samples_taken_for_accuracy: int = 1000000):
        logging.warning("EllipticSampler(... mass = None), mass was not set. Using estimated mass instead.")
        U = np.random.normal(size=(number_of_samples_taken_for_accuracy, self.dimensions()))
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        L = np.linalg.cholesky(self.sigma)
        norms = np.linalg.norm(U @ L.T, axis=1) ** self.alpha
        return np.mean(norms)


class DiscreteSampler(BaseSpectralMeasureSampler):

    def __init__(self,
        alpha: float,
        positions: np.ndarray,
        weights: np.ndarray
    ):
        self.positions = np.asarray(positions, dtype=float)
        self.weights = np.asarray(weights, dtype=float)

        if self.positions.ndim == 1:
            self.positions = self.positions.reshape(-1, 1)
        if self.positions.ndim != 2:
            raise ValueError("positions must be a 1D or 2D array")
        if self.weights.ndim != 1:
            raise ValueError("weights must be a 1D array")
        if len(self.weights) == 0 or len(self.positions) != len(self.weights):
            raise ValueError("positions and weights must be non-empty and have the same length")
        if not (np.all(np.isfinite(self.positions)) and np.all(np.isfinite(self.weights))):
            raise ValueError("positions and weights must be finite")
        if np.any(self.weights < 0):
            raise ValueError("weights must be non-negative")

        self._mass = self.weights.sum()
        if self._mass == 0:
            raise ValueError("weights must not all be zero")

        self.number_of_dimensions = self.positions.shape[1]

        if alpha >= 1:
            weighted = self.positions * self.weights[:, None]
            # Rounding error in a sum grows with the size of its terms,
            # so the tolerance is scaled by their total magnitude.
            tolerance = 1e-12 * np.abs(weighted).sum()
            if np.any(np.abs(weighted.sum(axis=0)) > tolerance):
                raise ValueError("when alpha >= 1, the weighted mean of the positions must be 0")

    def sample(self, number_of_samples: int, random_state: None | int | np.random.RandomState | np.random.Generator = None) -> np.ndarray:
        random_state = get_random_state_generator(random_state)
        indices = random_state.choice(len(self.weights), size=number_of_samples, p=self.weights / self._mass)
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
        if len(spectral_measures) == 0:
            raise ValueError("spectral_measures must be non-empty")
        if len(weights) != len(spectral_measures):
            raise ValueError("weights must have one weight per spectral measure")
        if any(m.dimensions() != spectral_measures[0].dimensions() for m in spectral_measures):
            raise ValueError("all spectral measures must have the same number of dimensions")

        self.number_of_dimensions = spectral_measures[0].dimensions()
        self.spectral_measures = spectral_measures
        self.weights = np.asarray(weights)
        self._mass = self._calculate_mass()

    def sample(self, number_of_samples: int, random_state: None | int | np.random.RandomState | np.random.Generator = None) -> np.ndarray:
        random_state = get_random_state_generator(random_state)
        weights = self.weights / self.weights.sum()
        indices = random_state.choice(len(weights), size=number_of_samples, p=weights)
        samples = []
        for i in range(len(weights)):
            count = np.sum(indices == i)
            if count > 0:
                samples.append(self.spectral_measures[i].sample(count, random_state))
        return np.vstack(samples)

    def dimensions(self) -> int:
        return self.number_of_dimensions

    def mass(self) -> float:
        return float(self._mass)

    def _calculate_mass(self):
        return sum(
            spectral_measure.mass() * weight
                for spectral_measure, weight in zip(self.spectral_measures, self.weights)
        )

    
class UnivariateSampler(BaseSpectralMeasureSampler):
    def __init__(self,
        alpha: float,
        beta: float,
        gamma :float
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        if self.alpha >= 1 and self.beta != 0:
            raise ValueError("when alpha >= 1, beta must be 0")

    def sample(self, number_of_samples: int, random_state: None | int | np.random.RandomState | np.random.Generator = None) -> np.ndarray:
        random_state = get_random_state_generator(random_state)
        p_plus = (1.0 + self.beta) / 2.0
        signs = np.where(
            random_state.random(number_of_samples) <= p_plus,
            1.0,
            -1.0
        ).reshape(-1, 1) # reshape to (n, 1) since we expect vectors
        return signs

    def dimensions(self) -> int:
        return 1

    def mass(self) -> float:
        return self.gamma**(self.alpha)   


def isotropic_scale_correction(dimentions: int, alpha: float, gamma_scale: float):
    m_d_alpha = (
            gamma((alpha + 1) / 2)
            * gamma(dimentions / 2)
            / (np.sqrt(np.pi) * gamma((dimentions + alpha) / 2))
    )
    return gamma_scale * (m_d_alpha ** (-1.0 / alpha))

