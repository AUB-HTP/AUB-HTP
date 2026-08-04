import numpy as np
import pytest

from aub_htp import multivariate_alpha_stable


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
    # Force a clean cache
    multivariate_alpha_stable._last_pdf_model = None
    multivariate_alpha_stable._last_pdf_key = None

    pts = np.array([[0.0, 0.0], [0.1, 0.0]])
    kwargs = dict(
        alpha=1.5,
        spectral_measure_sampler="standard_isotropic_2d",
        number_of_spectral_samples=1000,
        random_state=0,
    )

    _ = multivariate_alpha_stable.pdf(pts, **kwargs)
    first_id = id(multivariate_alpha_stable._last_pdf_model)

    # same call should reuse cached model
    _ = multivariate_alpha_stable.pdf(pts, **kwargs)
    second_id = id(multivariate_alpha_stable._last_pdf_model)

    assert first_id == second_id
