import numpy as np
import numpy.typing as npt
from typing import Literal
from scipy.stats import rv_continuous
from scipy._lib.doccer import inherit_docstring_from
from scipy.stats._multivariate import multi_rv_generic
from scipy.stats._distn_infrastructure import _ShapeInfo

from .pdf import generate_alpha_stable_pdf
from .random import sample_alpha_stable_vector, IsotropicSampler, UnivariateSampler, EllipticSampler, DiscreteSampler, BaseSpectralMeasureSampler


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
        and np.all(gamma == gamma[0]) \
        and np.all(np.diff(x) > 0):
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
        if np.all(alpha == alpha[0]) and np.all(beta == beta[0]) and np.all(np.diff(x)>0):
            # Note: -1/2 and 1/2 have been chosen as arbitrary "close" values to x for interpolation to work.
            padded_x = np.concatenate([[x[0] - 1/2], x, [x[-1] + 1/2]])
            padded_pdf = generate_alpha_stable_pdf(padded_x, alpha[0], beta[0], 1, 0)
            pdf = padded_pdf[1:-1]

        else:
            pdf = np.array([
                generate_alpha_stable_pdf([xi - 1/2, xi, xi + 1/2], ai, bi, 1, 0)[1]
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

            sampler = UnivariateSampler(alpha_, beta_, gamma=gamma_)
            samples = sample_alpha_stable_vector(
                alpha_,
                sampler,
                number_of_samples_for_this_pair,
                shift_,
                random_state = random_state,
            )

            # Assign samples to all rows that share this (alpha, beta) pair
            out[(inverse == idx)] = samples

        return out.reshape(size)

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
            return delta
        elif parameterization == "S1":
            if alpha != 1:
                return delta
            else:
                return delta + (2 / np.pi) * beta * gamma * np.log(gamma)
        else:
            raise AssertionError("Unknown parametrization type")


class multivariate_alpha_stable_gen(multi_rv_generic):
    def rvs(self,
        alpha: float,
        spectral_measure_sampler: BaseSpectralMeasureSampler | Literal["standard_isotropic_2d", "standard_isotropic_3d", "1x2_elliptic_2d", "1x2x4_elliptic_3d", "coin_flip_discrete"] = "standard_isotropic_2d",
        shift: np.ndarray = 0,
        size: int | None = None,
        random_state: None | int | np.random.RandomState | np.random.Generator = None,
    ):
        spectral_measure_sampler = self._check_spectral_measure_sampler(alpha, spectral_measure_sampler)
        assert isinstance(spectral_measure_sampler, BaseSpectralMeasureSampler)
        samples = sample_alpha_stable_vector(alpha, spectral_measure_sampler, size or 1, shift, random_state = random_state)
        if size is None:
            return samples[0]
        return samples

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
