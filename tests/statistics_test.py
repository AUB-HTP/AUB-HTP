import pytest
import numpy as np
from scipy.stats import levy_stable

from aub_htp.statistics import alpha_power, alpha_location


def _as_scalar(x) -> float:
    """Convert scalar-like optimizer output to plain float."""
    arr = np.asarray(x)
    return float(arr.reshape(-1)[0])


def _sample_multivariate_cauchy(
    rng: np.random.Generator,
    n: int,
    d: int,
    scale: float = 1.0,
    location: np.ndarray | None = None,
) -> np.ndarray:
    """
    Sample from the standard isotropic multivariate Cauchy distribution
    (multivariate t with df=1 and identity scatter), then apply scale/location.
    """
    z = rng.normal(size=(n, d))
    u = rng.chisquare(df=1, size=n)
    x = z / np.sqrt(u)[:, None]

    if location is None:
        location = np.zeros(d)
    return scale * x + np.asarray(location)


class TestAlphaPowerCorrectness:
    def test_alpha_power_alpha_2_matches_normal_scale(self):
        rng = np.random.default_rng(0)
        true_scale = 3.0
        data = rng.normal(loc=0.0, scale=true_scale, size=4000)

        estimated = _as_scalar(alpha_power(data, alpha=2.0))

        assert np.isfinite(estimated)
        assert estimated > 0
        assert estimated == pytest.approx(true_scale, rel=0.08)

    def test_alpha_power_alpha_1_matches_cauchy_scale(self):
        rng = np.random.default_rng(1)
        true_scale = 2.5
        data = true_scale * rng.standard_cauchy(size=6000)

        estimated = _as_scalar(alpha_power(data, alpha=1.0))

        assert np.isfinite(estimated)
        assert estimated > 0
        assert estimated == pytest.approx(true_scale, rel=0.10)

    def test_alpha_power_alpha_1_matches_cauchy_scale_for_multivariate_data(self):
        rng = np.random.default_rng(2)
        true_scale = 1.8
        data = _sample_multivariate_cauchy(rng, n=8000, d=3, scale=true_scale)

        estimated = _as_scalar(alpha_power(data, alpha=1.0))

        assert np.isfinite(estimated)
        assert estimated > 0
        assert estimated == pytest.approx(true_scale, rel=0.12)

    @pytest.mark.parametrize(
        ("alpha", "scale"),
        [
            (0.2, -0.4),
            (0.5, 2),
            (1.0, 4),
            (1.2, 10.0),
            (1.5, -20.0),
            (1.5, 20.0),
            (2.0, 40.0),
        ]
    )
    def test_scale_equivariance_of_alpha_power(self, alpha: float, scale: float):
        rng = np.random.default_rng(3)
        data = rng.normal(size=600)

        p1 = _as_scalar(alpha_power(data, alpha=alpha))
        p2 = _as_scalar(alpha_power(scale * data, alpha=alpha))

        assert p1 > 0
        assert p2 > 0
        assert p2 == pytest.approx(abs(scale) * p1, rel=5e-2)


class TestAlphaLocationCorrectness:
    def test_alpha_location_alpha_2_matches_normal_location(self):
        rng = np.random.default_rng(4)
        true_location = 4.5
        data = rng.normal(loc=true_location, scale=2.0, size=4000)

        estimated = alpha_location(data, alpha=2.0)

        assert estimated.shape == (1,)
        assert np.all(np.isfinite(estimated))
        assert estimated[0] == pytest.approx(true_location, abs=0.15)

    def test_alpha_location_alpha_1_matches_cauchy_location(self):
        rng = np.random.default_rng(5)
        true_location = -3.0
        data = 1.7 * rng.standard_cauchy(size=7000) + true_location

        estimated = alpha_location(data, alpha=1.0)

        assert estimated.shape == (1,)
        assert np.all(np.isfinite(estimated))
        assert estimated[0] == pytest.approx(true_location, abs=0.20)

    def test_alpha_location_alpha_1_matches_cauchy_location_for_multivariate_data(self):
        rng = np.random.default_rng(6)
        true_location = np.array([1.5, -2.0, 0.75])
        data = _sample_multivariate_cauchy(
            rng, n=10000, d=3, scale=2.0, location=true_location
        )

        estimated = alpha_location(data, alpha=1.0)

        assert estimated.shape == true_location.shape
        assert np.all(np.isfinite(estimated))
        assert estimated == pytest.approx(true_location, abs=0.25)

    @pytest.mark.parametrize("alpha", [0.2, 0.3, 1.2, 1.5])
    def test_alpha_power_alpha_not_1_fails_for_multivariate_data(self, alpha: float):
        rng = np.random.default_rng(7)
        data = rng.normal(size=(200, 3))

        with pytest.raises(ValueError):
            alpha_location(data, alpha=alpha)

'''
class TestDimensionsAndShapes:
    def test_alpha_power_1d_and_column_vector_give_same_result(self):
        rng = np.random.default_rng(8)
        data_1d = rng.normal(size=500)
        data_2d = data_1d.reshape(-1, 1)

        p1 = _as_scalar(alpha_power(data_1d, alpha=1.2))
        p2 = _as_scalar(alpha_power(data_2d, alpha=1.2))

        assert p1 == pytest.approx(p2, rel=1e-8, abs=1e-10)

    def test_alpha_location_1d_and_column_vector_give_same_result(self):
        rng = np.random.default_rng(9)
        data_1d = rng.normal(loc=2.0, scale=1.5, size=1500)
        data_2d = data_1d.reshape(-1, 1)

        loc1 = alpha_location(data_1d, alpha=2.0)
        loc2 = alpha_location(data_2d, alpha=2.0)

        assert loc1.shape == (1,)
        assert loc2.shape == (1,)
        assert loc1[0] == pytest.approx(loc2[0], rel=1e-8, abs=1e-10)

    def test_alpha_power_returns_scalar_like_output(self):
        rng = np.random.default_rng(10)
        data = rng.normal(size=300)

        out = alpha_power(data, alpha=2.0)
        scalar = _as_scalar(out)

        assert np.asarray(out).size == 1
        assert np.isfinite(scalar)
        assert scalar > 0

    def test_alpha_location_returns_vector_of_dimension_d(self):
        rng = np.random.default_rng(11)
        data = rng.normal(size=(400, 4))

        loc = alpha_location(data, alpha=2.0)

        assert isinstance(loc, np.ndarray)
        assert loc.shape == (4,)
        assert np.all(np.isfinite(loc))

    def test_alpha_power_alpha_not_1_fails_for_multivariate_data(self):
        rng = np.random.default_rng(12)
        data = rng.normal(size=(150, 2))

        with pytest.raises(ValueError):
            alpha_power(data, alpha=1.5)

    def test_alpha_power_3d_input_raises(self):
        data = np.zeros((5, 2, 2))
        with pytest.raises(AssertionError):
            alpha_power(data, alpha=1.0)

    def test_alpha_location_3d_input_raises(self):
        data = np.zeros((5, 2, 2))
        with pytest.raises(AssertionError):
            alpha_location(data, alpha=1.0)
            '''