import numpy as np
import numpy.typing as npt
from typing import Literal
from scipy.stats import rv_continuous
from scipy._lib.doccer import inherit_docstring_from
from scipy.stats._multivariate import multi_rv_generic
from scipy.stats._distn_infrastructure import _ShapeInfo

from .pdf import generate_alpha_stable_pdf
from .random import sample_alpha_stable_vector, IsotropicSampler, UnivariateSampler, BaseSpectralMeasureSampler


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
        # override base class version to correct
        # location for S1 parameterization
        (alpha, beta), delta, gamma = self._parse_args(*args, **kwds)

        if self.parameterization == "S0":
            return super().pdf(x, *args, **kwds)

        elif self.parameterization == "S1":
            if np.all(alpha != 1):
                _kwds = kwds.copy()
                _kwds.pop("loc", None)
                return super().pdf(x, *args,
                                   loc = self._get_shift_term(alpha, beta, gamma, delta, self.parameterization),
                                   **_kwds)
            else:
                raise NotImplementedError() #TODO: multiple but different alpha?
        else:
            raise AssertionError("Unknown parametrization type")

    def _pdf(self, x, alpha, beta): #TODO: broadcast alpha and beta to x.shape
        x = np.asarray(x).ravel()
        alpha = np.asarray(alpha).ravel()
        beta = np.asarray(beta).ravel()

        if np.all(alpha == alpha[0]) and np.all(beta == beta[0]) and np.all(np.diff(x)>0):

            # -1/2 and 1/2 have been chosen as arbitrary "close" values to x for interpolation to work.
            padded_x = np.concatenate([[x[0] - 1/2], x, [x[-1] + 1/2]])
            padded_pdf = generate_alpha_stable_pdf(padded_x, alpha[0], beta[0], 1, 0)
            return padded_pdf[1:-1]

        return np.array([
            generate_alpha_stable_pdf([xi - 1/2, xi, xi + 1/2], ai, bi, 1, 0)[1]
            for xi, ai, bi in zip(x, alpha, beta)
        ])

    def _argcheck(self, alpha, beta):
        return (0 < alpha) & (alpha <= 2) & (-1 <= beta) & (beta <= 1)


    def _rvs(self, alpha, beta, size=None, random_state=None):
        # size is shape
        #TODO:
        # 1. Integrate beta into sample_alpha_stable_vector
        # 2. Make sure `size` parameter is fine
        # 3. test
        alpha = np.broadcast_to(alpha, size)
        beta = np.broadcast_to(beta, size)
        if np.all(alpha == alpha[0]) and np.all(beta == beta[0]):
            sampler = UnivariateSampler(alpha[0], beta[0])
            samples = sample_alpha_stable_vector(alpha[0], sampler, int(np.prod(size)))
            samples = samples.reshape(size)
            return samples
        else:
            #TODO: Test
            alpha_ = alpha.ravel()
            beta_ = beta.ravel()
            size = alpha_.shape[0]
            return np.asarray([self._rvs(alpha_[i], beta_[i], size) for i in range(size)])


    #def rvs(self): #TODO: similar wrapper to .pdf
    #    pass

    def _get_shift_term(self, alpha: float, beta: float, gamma: float, delta: float, parameterization: Literal["S0", "S1"] | None = None):
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


class multi_variate_alpha_stable(multi_rv_generic):
    def rvs(self,
        alpha: float,
        spectral_measure_sampler: BaseSpectralMeasureSampler,
        size: int | None = None,
        random_state: None | int | np.random.RandomState | np.random.Generator = None,
    ):
        # TODO:
        # 1. figure out what to do with random_state
        # 2. test
        samples = sample_alpha_stable_vector(alpha, spectral_measure_sampler, size or 1)
        if size is None:
            samples[0]
        return samples