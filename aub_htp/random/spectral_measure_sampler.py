import numpy as np
from scipy.special import gamma
import logging
from abc import abstractmethod, ABC
from .util import get_random_state_generator

class BaseSpectralMeasureSampler(ABC):
    '''
    Spectral Measure Sampler is an interface to define the sampling algorithm for a spectral measure.
    The underlying mathematical spectral measure :math:`\Lambda` has to uphold the following property:
    
    .. math::
        \int_{\mathbb{S}^{d-1}}s\Lambda(ds)=0

    To create a custom spectral measure sampler, you need to inherit from this class and implement the following methods:  
    
    - :meth:`sample`  
    
    - :meth:`dimensions`  
    
    - :meth:`mass`  

    Example:
    ::
        class CustomSpectralMeasureSampler(BaseSpectralMeasureSampler):
            def sample(self, number_of_samples: int, random_state: None | int | np.random.RandomState | np.random.Generator = None) -> np.ndarray:
                random_state = get_random_state_generator(random_state)
                return random_state.rand(number_of_samples, self.dimensions())
            def dimensions(self) -> int:
                return 2
            def mass(self) -> float:
                return 1.0
    '''

    @abstractmethod
    def sample(self, number_of_samples: int, random_state: None | int | np.random.RandomState | np.random.Generator = None) -> np.ndarray:
        """Sampling algorithm for the spectral measure sampler.

        Parameters
        ----------
        number_of_samples : int
            The number of samples to draw from the spectral measure.
        random_state : None | int | np.random.RandomState | np.random.Generator, optional
            The random state to use for the sampling. You are encouraged to use :func:`aub_htp.random.get_random_state_generator` 
            to get a random state generator as such:
            :code:`random_state = get_random_state_generator(random_state)`
        
        Returns
        -------
        samples : np.ndarray
            The samples from the spectral measure.
        """

    @abstractmethod
    def dimensions(self) -> int:
        """
        Number of dimensions of the vector space of the spectral measure.

        Returns
        -------
        dimensions : int
        """

    @abstractmethod
    def mass(self) -> float:
        """
        Mass of the spectral measure.

        Returns
        -------
        mass : float
        """


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

    def _estimate_mass(self, number_of_samples_taken_for_accuracy: int = 100000):
        # TODO: include random_state
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
        self.positions = np.asarray(positions)
        self.weights = np.asarray(weights)
        assert self.positions.shape[0] == self.weights.shape[0] and self.positions.shape[0] > 0
        if len(self.positions.shape) < 2:
            self.positions = self.positions.reshape(-1, 1)
        self.number_of_dimensions = self.positions.shape[1]
        self._mass = self.weights.sum()
        if alpha >= 1:
            assert np.all((self.positions*self.weights[:, None]).sum(axis = 0) == 0), "when alpha >= 1, the weighted mean of the positions should be 0."

    def sample(self, number_of_samples: int, random_state: None | int | np.random.RandomState | np.random.Generator = None) -> np.ndarray:
        random_state = get_random_state_generator(random_state)
        indices = random_state.choice(len(self.weights), size=number_of_samples, p=self.weights / self.weights.sum())
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
        return np.sum(
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

        if self.alpha >= 1:
            assert self.beta == 0, "For alpha >= 1, beta is 0"

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

