import numpy as np
import pytest
from scipy.special import gamma as gamma_fn

from aub_htp import alpha_stable
from aub_htp.pdf.multivariate import (
    MultivariateStableDensity,
    _projection_parameters,
    _projection_parameters_batched,
    _projection_parameters_discrete,
    _projection_parameters_elliptical,
    _projection_parameters_isotropic,
    _sphere_directions,
)
from aub_htp.random.spectral_measure_sampler import (
    DiscreteSampler,
    EllipticSampler,
    IsotropicSampler,
)


@pytest.mark.parametrize("alpha", [0.8, 1.0, 1.5])
def test_batched_projection_parameters_match_full(alpha):
    rng = np.random.default_rng(1234)
    directions = rng.normal(size=(17, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    samples = rng.normal(size=(137, 3))

    full = _projection_parameters(directions, samples, alpha, mass=2.3)
    batched = _projection_parameters_batched(
        directions, samples, alpha, mass=2.3, batch_size=19)

    np.testing.assert_allclose(batched[0], full[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(batched[1], full[1], rtol=1e-12, atol=1e-12)
    if alpha == 1.0:
        np.testing.assert_allclose(batched[2], full[2], rtol=1e-12,
                                   atol=1e-12)
    else:
        assert batched[2] is full[2] is None


@pytest.mark.parametrize("alpha", [0.8, 1.0, 1.5])
def test_discrete_projection_parameters_match_repeated_atoms(alpha):
    rng = np.random.default_rng(7)
    directions = rng.normal(size=(11, 2))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    positions = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 2.0]])
    integer_weights = np.array([2, 3, 5])
    samples = np.repeat(positions, integer_weights, axis=0)

    exact = _projection_parameters_discrete(
        directions, positions, integer_weights, alpha)
    empirical = _projection_parameters(
        directions, samples, alpha, mass=integer_weights.sum())

    for got, expected in zip(exact, empirical):
        if got is None:
            assert expected is None
        else:
            np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_discrete_density_is_independent_of_sample_count_and_seed():
    sampler = DiscreteSampler(
        alpha=1.5,
        positions=np.array([[1.0, 0.0], [-1.0, 0.0],
                            [0.0, 1.0], [0.0, -1.0]]),
        weights=np.ones(4),
    )
    kwargs = dict(number_of_sphere_points=48, exact=True)
    small = MultivariateStableDensity(
        1.5, sampler, number_of_spectral_samples=1, random_state=1, **kwargs)
    large = MultivariateStableDensity(
        1.5, sampler, number_of_spectral_samples=1_000_000,
        random_state=999, **kwargs)

    np.testing.assert_array_equal(small.sigma, large.sigma)
    np.testing.assert_array_equal(small.beta, large.beta)
    np.testing.assert_array_equal(small.mu, large.mu)


def test_elliptical_mass_estimate_is_reproducible():
    sampler = EllipticSampler(
        number_of_dimensions=3,
        alpha=1.2,
        sigma=np.diag([1.0, 2.0, 4.0]),
        mass=1.0,
    )
    first = sampler._estimate_mass(10_000, random_state=123)
    second = sampler._estimate_mass(10_000, random_state=123)
    different = sampler._estimate_mass(10_000, random_state=124)

    assert first == second
    assert first != different


def test_explicit_elliptical_mass_skips_estimation():
    sampler = EllipticSampler(
        number_of_dimensions=2,
        alpha=1.5,
        sigma=np.eye(2),
        mass=0.0,
    )
    assert sampler.mass() == 0.0


@pytest.mark.parametrize("method", ["mc", "sobol"])
def test_high_dimensional_sphere_nodes_are_reproducible_and_unit_length(method):
    first, weight1 = _sphere_directions(5, 256, 42, method=method)
    second, weight2 = _sphere_directions(5, 256, 42, method=method)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0,
                               rtol=0.0, atol=2e-15)
    assert weight1 == weight2


def test_antipodal_sphere_nodes_are_exact_pairs():
    directions, _ = _sphere_directions(
        3, 128, 5, method="sobol", antipodal=True)
    np.testing.assert_array_equal(directions[:64], -directions[64:])


@pytest.mark.parametrize("alpha", [0.8, 1.0, 1.5])
def test_exact_isotropic_projection_parameters(alpha):
    directions, _ = _sphere_directions(5, 64, 3, method="sobol")
    sigma, beta, mu = _projection_parameters_isotropic(
        directions, gamma_scale=2.5, alpha=alpha)
    np.testing.assert_array_equal(sigma, np.full(64, 2.5))
    np.testing.assert_array_equal(beta, np.zeros(64))
    if alpha == 1.0:
        np.testing.assert_array_equal(mu, np.zeros(64))
    else:
        assert mu is None


@pytest.mark.parametrize("alpha", [0.8, 1.0, 1.5])
def test_exact_elliptical_projection_parameters(alpha):
    directions, _ = _sphere_directions(3, 64, 3, method="sobol")
    scatter = np.diag([1.0, 2.0, 4.0])
    sigma, beta, mu = _projection_parameters_elliptical(
        directions, scatter, mass=2.0, alpha=alpha)
    expected = 2.0 ** (1.0 / alpha) * np.sqrt(
        np.einsum("ij,jk,ik->i", directions, scatter, directions))
    np.testing.assert_allclose(sigma, expected, rtol=2e-15, atol=0.0)
    np.testing.assert_array_equal(beta, np.zeros(64))
    if alpha == 1.0:
        np.testing.assert_array_equal(mu, np.zeros(64))
    else:
        assert mu is None


def test_elliptical_scatter_obeys_linear_change_of_variables():
    alpha = 1.5
    scatter = np.diag([1.0, 3.0])
    transform = np.linalg.cholesky(scatter)
    isotropic = MultivariateStableDensity(
        alpha, IsotropicSampler(2, alpha, 1.0), number_of_sphere_points=720)
    elliptical = MultivariateStableDensity(
        alpha, EllipticSampler(2, alpha, scatter, mass=1.0),
        number_of_sphere_points=720)
    x = np.array([[0.0, 0.0], [1.0, -0.5], [3.0, 2.0]])
    transformed = np.linalg.solve(transform, x.T).T
    expected = isotropic.pdf(transformed) / np.linalg.det(transform)
    np.testing.assert_allclose(elliptical.pdf(x), expected, rtol=2e-8, atol=1e-12)


def test_pdf_accepts_one_d_dimensional_element():
    model = MultivariateStableDensity(1.5, IsotropicSampler(3, 1.5, 1.0))

    density = model.pdf(np.array([0.0, 1.0, 2.0]))

    assert np.ndim(density) == 0


def test_pdf_accepts_n_d_dimensional_elements():
    model = MultivariateStableDensity(1.5, IsotropicSampler(3, 1.5, 1.0))

    densities = model.pdf(np.array([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]))

    assert densities.shape == (2,)


def test_pdf_rejects_one_element_with_wrong_dimension():
    model = MultivariateStableDensity(1.5, IsotropicSampler(3, 1.5, 1.0))

    with pytest.raises(ValueError, match="x has dimension 2.*3-dimensional"):
        model.pdf(np.array([0.0, 1.0]))


def test_pdf_rejects_n_elements_with_wrong_dimension():
    model = MultivariateStableDensity(1.5, IsotropicSampler(3, 1.5, 1.0))

    with pytest.raises(ValueError, match="x has dimension 2.*3-dimensional"):
        model.pdf(np.array([[0.0, 1.0], [2.0, 3.0]]))


def test_pdf_rejects_arrays_that_are_not_one_or_two_dimensional():
    model = MultivariateStableDensity(1.5, IsotropicSampler(3, 1.5, 1.0))

    with pytest.raises(ValueError):
        model.pdf(np.zeros((2, 1, 3)))


@pytest.mark.parametrize("d", [2, 3, 5, 10])
@pytest.mark.parametrize("alpha", [0.8, 1.0, 1.5])
def test_isotropic_density_at_origin_matches_closed_form(d, alpha):
    model = MultivariateStableDensity(
        alpha,
        IsotropicSampler(d, alpha, gamma=1.0),
        number_of_spectral_samples=1,
        number_of_sphere_points=2048,
        random_state=2,
    )
    sphere_area = 2.0 * np.pi ** (d / 2.0) / gamma_fn(d / 2.0)
    expected = sphere_area * gamma_fn(d / alpha) / (
        alpha * (2.0 * np.pi) ** d)
    assert model.pdf(np.zeros(d)) == pytest.approx(expected, rel=2e-8)


@pytest.mark.parametrize("d", [3, 5, 10])
def test_independent_density_matches_product_of_marginals(d):
    alpha = 1.5
    positions = np.vstack([np.eye(d), -np.eye(d)])
    sampler = DiscreteSampler(alpha, positions, np.full(2 * d, 0.5))
    model = MultivariateStableDensity(
        alpha, sampler, number_of_sphere_points=4096, random_state=1)
    rng = np.random.default_rng(8)
    x = np.vstack([np.zeros(d), np.eye(d), rng.normal(size=(20, d))])
    expected = np.prod(
        alpha_stable.pdf(x, alpha, 0.0, loc=0.0, scale=1.0), axis=1)
    got = model.pdf(x)
    normalized_rmse = np.sqrt(np.mean((got - expected) ** 2)) / expected.max()
    assert normalized_rmse < 1e-12


def test_independent_s0_location_conversion_uses_internal_s1_shift():
    alpha = 0.8
    beta = np.array([1.0, -1.0])
    positions = np.vstack([np.eye(2), -np.eye(2)])
    weights = np.array([1.0, 0.0, 0.0, 1.0])
    sampler = DiscreteSampler(alpha, positions, weights)
    s1_shift = -beta * np.tan(np.pi * alpha / 2.0)
    model = MultivariateStableDensity(alpha, sampler, shift=s1_shift)
    x = np.array([[0.0, 0.0], [2.0, -2.0], [5.0, -5.0]])

    expected = np.prod(
        alpha_stable.pdf(x, alpha, beta, loc=s1_shift, scale=1.0), axis=1)
    np.testing.assert_allclose(model.pdf(x), expected, rtol=1e-13, atol=1e-15)
