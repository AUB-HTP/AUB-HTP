import numpy as np
import pytest

from aub_htp import multivariate_alpha_stable
from aub_htp.pdf.multivariate import clear_pdf_model_cache, get_pdf_model
from aub_htp.random import IsotropicSampler


def test_empty_input_returns_empty():
    pts = np.empty((0, 2))
    vals = multivariate_alpha_stable.pdf(
        pts,
        alpha=1.5,
        spectral_measure_sampler="standard_isotropic_2d",
        number_of_spectral_samples=1000,
        random_state=0,
    )
    assert isinstance(vals, np.ndarray)
    assert vals.shape == (0,)


def test_pdf_non_negative_and_finite_multiple_points():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, -0.2]])
    vals = multivariate_alpha_stable.pdf(
        pts,
        alpha=1.5,
        spectral_measure_sampler="standard_isotropic_2d",
        number_of_spectral_samples=1000,
        random_state=0,
    )
    assert vals.shape == (3,)
    assert np.all(vals >= 0.0)
    assert np.all(np.isfinite(vals))


def test_unknown_spectral_sampler_raises():
    pts = np.array([0.0, 0.0])
    with pytest.raises(ValueError):
        multivariate_alpha_stable.pdf(
            pts,
            alpha=1.5,
            spectral_measure_sampler="this_sampler_does_not_exist",
            number_of_spectral_samples=1000,
            random_state=0,
        )


def test_wrong_dimension_point_raises():
    pt = np.array([0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        multivariate_alpha_stable.pdf(
            pt,
            alpha=1.5,
            spectral_measure_sampler="standard_isotropic_2d",
            number_of_spectral_samples=1000,
            random_state=0,
        )


def test_pdf_cache_reuse():
    clear_pdf_model_cache()
    kwargs = dict(
        alpha=1.5,
        number_of_spectral_samples=1000,
        random_state=0,
    )

    first = get_pdf_model(
        spectral_measure_sampler=IsotropicSampler(2, 1.5, 1.0), **kwargs)
    second = get_pdf_model(
        spectral_measure_sampler=IsotropicSampler(2, 1.5, 1.0), **kwargs)

    assert first is second
