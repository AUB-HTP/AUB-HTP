from ast import Not
import numpy as np
import numpy.typing as npt
from typing import Literal
from scipy.stats import rv_continuous
from scipy._lib.doccer import inherit_docstring_from
from scipy.stats._multivariate import multi_rv_generic
from scipy.stats._distn_infrastructure import _ShapeInfo

from .pdf import generate_alpha_stable_pdf
from .pdf.multivariate import get_pdf_model
from .random import sample_alpha_stable_vector, IsotropicSampler, UnivariateSampler, EllipticSampler, DiscreteSampler, BaseSpectralMeasureSampler
from .random.cms_univariate_sampler import sample_cms
from .random.alpha_stable_sampler import estimate_number_of_convergence_terms
from .random.elliptical_sampler_method import sample_elliptical
import logging

class alpha_stable_gen(rv_continuous):
    _parameterization: Literal["S1", "S0"] = "S1"

    @property
    def parameterization(self):
        return self._parameterization

    @parameterization.setter
    def parameterization(self, parametrization: Literal["S1", "S0"]):
        if parametrization not in ("S1", "S0"):
            raise ValueError(f"Parametrization '{parametrization}' not supported. Use 'S1' (default) or 'S0'.")
        self._parameterization = parametrization

    def with_parametrization(self, parametrization: Literal["S1", "S0"]):
        self.parameterization = parametrization
        return self

    def _shape_info(self):
        return [_ShapeInfo("alpha", False, (0, 2), (False, True)),
                _ShapeInfo("beta", False, (-1, 1), (True, True))]

    @inherit_docstring_from(rv_continuous)
    def pdf(self, x, *args, **kwds):
        # override base class version to correct location for S1 parameterization
        (alpha, beta), delta, gamma = self._parse_args(*args, **kwds)
        (x, alpha, beta, delta, gamma), size = _sanitize_array_args(x, alpha, beta, delta, gamma)
        if size == (0,):
            return np.array([])

        if np.all(alpha == alpha[0]) \
        and np.all(beta == beta[0]) \
        and np.all(delta == delta[0]) \
        and np.all(gamma == gamma[0]):
            alpha_, beta_, delta_, gamma_ = alpha[0], beta[0], delta[0], gamma[0]
            return super().pdf(
                x,
                alpha_,
                beta_,
                loc = self._get_shift_term(alpha_, beta_, gamma_, delta_, self.parameterization),
                scale = gamma_
            ).reshape(size)
        else:
            pdf = [
                super(self.__class__, self).pdf(
                    x_,
                    alpha_,
                    beta_,
                    loc = self._get_shift_term(alpha_, beta_, gamma_, delta_, self.parameterization),
                    scale = gamma_
                ) for x_, alpha_, beta_, delta_, gamma_ in zip(x, alpha, beta, delta, gamma)
            ]
            return np.asarray(pdf).reshape(size)

    def _pdf(self, x, alpha, beta):
        (x, alpha, beta), size = _sanitize_array_args(x, alpha, beta)
        if size == (0,):
            return np.array([])

        pdf: np.ndarray
        if np.all(alpha == alpha[0]) and np.all(beta == beta[0]):
            pdf = generate_alpha_stable_pdf(x, alpha[0], beta[0], 1, 0)
            
        else:
            pdf = np.array([
                generate_alpha_stable_pdf(xi, ai, bi, 1, 0)
                for xi, ai, bi in zip(x, alpha, beta)
            ])

        return pdf.reshape(size)

    def _argcheck(self, alpha, beta):
        return (0 < alpha) & (alpha <= 2) & (-1 <= beta) & (beta <= 1)

    @staticmethod
    def _vectorized_rvs(alpha, beta, gamma, delta, parametrization, size = None, random_state = None):
        if size is None:
            (alpha, beta), size = _sanitize_array_args(alpha, beta)
        else:
            alpha = np.broadcast_to(alpha, size).ravel()
            beta = np.broadcast_to(beta, size).ravel()
            gamma = np.broadcast_to(gamma, size).ravel()
            delta = np.broadcast_to(delta, size).ravel()

        shift: np.ndarray
        if parametrization == "S0":
            shift = np.where(
                alpha == 1,
                delta,
                delta - gamma * beta * np.tan(alpha * np.pi / 2)
            )
        elif parametrization == "S1":
            shift = np.where(
                alpha == 1,
                delta + beta * 2 / np.pi * gamma * np.log(gamma),
                delta
            )

        if size == (0,):
            return np.array([])

        number_of_samples = int(np.prod(size))

        params = np.column_stack((alpha, beta, gamma, shift))
        unique_params, inverse = np.unique(params, axis=0, return_inverse=True)

        out = np.empty(number_of_samples)

        for idx, (alpha_, beta_, gamma_, shift_) in enumerate(unique_params):
            mask = inverse == idx
            number_of_samples_for_this_pair = mask.sum()

            samples = alpha_stable_gen._single_valued_rvs(alpha_, beta_, gamma_, shift_, number_of_samples_for_this_pair, random_state)

            # Assign samples to all rows that share this (alpha, beta) pair
            out[(inverse == idx)] = samples

        return out.reshape(size)
    

    @staticmethod
    def _single_valued_rvs(alpha_, beta_, gamma_, shift_, size, random_state):
        # patch: always use CMS sampler for now
        if True or alpha_ >= 1 and beta_ != 0 or estimate_number_of_convergence_terms(0.01, alpha_, gamma_**alpha_) >= 50000:
            logging.debug(f"Using CMS sampler for alpha = {alpha_}, beta = {beta_}")
            samples = sample_cms(alpha_, beta_, size, random_state)
            return samples * gamma_ + shift_
        else:
            logging.debug(f"Using Univariate sampler for alpha = {alpha_}, beta = {beta_}")
            sampler = UnivariateSampler(alpha_, beta_, gamma=gamma_)
            samples = sample_alpha_stable_vector(
                alpha_,
                sampler,
                size,
                shift_,
                random_state = random_state,
            )
        return samples


    def _rvs(self, alpha, beta, size=None, random_state=None):
        return self._vectorized_rvs(alpha, beta, 1, 0, size = size, random_state = random_state)


    @inherit_docstring_from(rv_continuous)
    def rvs(self, *args, **kwds):

        kwds.pop("discrete", None)
        random_state = kwds.pop("random_state", None)
        (alpha, beta), delta, gamma, size = self._parse_args_rvs(*args, **kwds)

        return self._vectorized_rvs(alpha, beta, gamma, delta, self.parameterization, size=size, random_state=random_state)


    def _get_shift_term(self, alpha: float, beta: float, gamma: float, delta: float, parameterization: Literal["S0", "S1"] | None = None) -> float:
        parameterization = parameterization or self.parameterization
        if parameterization == "S0":
            if alpha != 1:
                return delta - gamma * beta * np.tan(alpha * np.pi / 2)
            else:
                return delta
        elif parameterization == "S1":
            if alpha != 1:
                return delta
            else:
                return delta + (2 / np.pi) * beta * gamma * np.log(gamma)
        else:
            raise AssertionError("Unknown parametrization type")


class multivariate_alpha_stable_gen(multi_rv_generic):
    """
    Multivariate alpha stable distribution generator.

    This class provides the SciPy-compatible `rvs` interface for sampling
    multivariate alpha-stable law, and a `pdf` wrapper that caches the last
    computed density model so repeated evaluations reuse expensive internal
    precomputations.
    """
    def __init__(self, seed=None):
        super().__init__(seed)

    def rvs(self,
        alpha: float,
        spectral_measure_sampler: BaseSpectralMeasureSampler | Literal["standard_isotropic_2d", "standard_isotropic_3d", "1x2_elliptic_2d", "1x2x4_elliptic_3d", "coin_flip_discrete"] = "standard_isotropic_2d",
        shift: np.ndarray = 0,
        size: int | None = None,
        error: float = 0.01,
        random_state: None | int | np.random.RandomState | np.random.Generator = None,
    ):
        """
        Sample from a multivariate alpha stable distribution.
        
        Parameters
        ----------
        alpha : float
            The alpha parameter of the alpha stable distribution.
        spectral_measure_sampler : BaseSpectralMeasureSampler or str
            The spectral measure sampler to use.
        shift : np.ndarray
            The shift vector to apply to the samples.
        size : int or tuple of ints, optional
            Defines the shape of the returned array. Default is 1.
        random_state : int or None, optional
            Random state to use for the random number generator.

        Returns
        -------
        samples : ndarray
            Samples from the multivariate alpha stable distribution.

        Raises
        ------
        ValueError
            If the spectral measure sampler is not supported.
        """
        spectral_measure_sampler = self._check_spectral_measure_sampler(alpha, spectral_measure_sampler)
        assert isinstance(spectral_measure_sampler, BaseSpectralMeasureSampler)
        samples: np.ndarray
        if isinstance(spectral_measure_sampler, (EllipticSampler, IsotropicSampler)):
            if isinstance(spectral_measure_sampler, IsotropicSampler):
                dimensions = spectral_measure_sampler.dimensions()
                sigma = 2 * spectral_measure_sampler.gamma**2 * np.eye(dimensions)
            else:
                sigma = spectral_measure_sampler.sigma
            samples = sample_elliptical(alpha, sigma, size or 1, random_state)
            samples += np.broadcast_to(shift, spectral_measure_sampler.dimensions())
        else:
            samples = sample_alpha_stable_vector(alpha, spectral_measure_sampler, size or 1, shift, error = error, random_state = random_state)
        if size is None:
            return samples[0]
        return samples

    def pdf(
        self,
        x: npt.ArrayLike,
        alpha: float,
        spectral_measure_sampler: BaseSpectralMeasureSampler | Literal[
            "standard_isotropic_2d", "standard_isotropic_3d",
            "1x2_elliptic_2d", "1x2x4_elliptic_3d", "coin_flip_discrete"
        ] = "standard_isotropic_2d",
        shift: npt.ArrayLike = 0.0,
        number_of_spectral_samples: int = 200_000,
        number_of_sphere_points: int | None = None,
        random_state: None | int | np.random.RandomState | np.random.Generator = None,
        exact: bool = False,
        sphere_method: str = "sobol",
        antipodal: bool = False,
    ) -> npt.NDArray:
        """Evaluate the multivariate alpha-stable density at point(s) x.

        The method reuses the last computed internal density model when the
        pdf parameters are identical to the previous call. This preserves the
        costly `MultivariateStableDensity` setup for repeated evaluations.
        """
        sampler = self._check_spectral_measure_sampler(
            alpha, spectral_measure_sampler)
        model = get_pdf_model(
            alpha,
            sampler,
            shift,
            number_of_spectral_samples,
            number_of_sphere_points,
            random_state,
            exact,
            sphere_method,
            antipodal,
        )
        return model.pdf(x)

    def _check_spectral_measure_sampler(self, alpha: float, spectral_measure_sampler: BaseSpectralMeasureSampler | str) -> BaseSpectralMeasureSampler:
        if isinstance(spectral_measure_sampler, BaseSpectralMeasureSampler):
            return spectral_measure_sampler

        if spectral_measure_sampler == "standard_isotropic_2d":
            return IsotropicSampler(2, alpha, 1)

        if spectral_measure_sampler == "standard_isotropic_3d":
            return IsotropicSampler(3, alpha, 1)

        if spectral_measure_sampler == "1x2_elliptic_2d":
            return EllipticSampler(2, alpha, sigma=[[1, 0], [0, 2]])

        if spectral_measure_sampler == "1x2x4_elliptic_3d":
            return EllipticSampler(3, alpha, sigma=[[1, 0, 0], [0, 2, 0], [0, 0, 4]])

        if spectral_measure_sampler == "coin_flip_discrete":
            return DiscreteSampler(alpha, [-1, 1], [0.5, 0.5])

        raise ValueError("Unknown spectral measure sampler")

def _sanitize_array_args(*args):
    assert len(args) > 0
    args = list(map(np.asarray, args))
    args = np.broadcast_arrays(*args)
    size = args[0].shape
    args = list(map(np.ravel, args))
    return args, size
