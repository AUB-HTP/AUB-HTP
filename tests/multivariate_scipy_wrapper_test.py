import numpy as np

from aub_htp import multivariate_alpha_stable


def test_multivariate_alpha_stable_scipy_pdf_interface():
    pts = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    vals = multivariate_alpha_stable.pdf(
        pts,
        alpha=1.5,
        spectral_measure_sampler="standard_isotropic_2d",
        number_of_spectral_samples=1000,
        random_state=0,
    )

    assert vals.shape == (2,)
    assert np.all(vals >= 0.0)
    assert np.all(np.isfinite(vals))


def test_multivariate_alpha_stable_scipy_pdf_scalar_point():
    pt = np.array([0.0, 0.0], dtype=float)
    val = multivariate_alpha_stable.pdf(
        pt,
        alpha=1.5,
        spectral_measure_sampler="standard_isotropic_2d",
        number_of_spectral_samples=1000,
        random_state=0,
    )

    assert np.isscalar(val)
    assert val >= 0.0
    assert np.isfinite(val)
