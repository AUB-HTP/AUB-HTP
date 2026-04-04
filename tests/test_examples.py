import pytest
import numpy as np
from scipy.ndimage import shift
from scipy.stats import alpha, cauchy
from aub_htp.statistics import alpha_power, alpha_location
from scipy.stats import multivariate_t


class TestAlphaPowerCorrectness:
    def test_alpha_power_alpha_2_matches_square_root_of_expected_value_of_X_squared(self):
        np.random.seed(42)
        data = np.random.normal(10, 100, 1000)
        expected_value_of_X_squared = np.mean(data ** 2)**.5
        alpha_power_of_data = alpha_power(data, 2)
        assert np.isclose(alpha_power_of_data, expected_value_of_X_squared)

    def test_alpha_power_alpha_2_matches_square_root_of_expected_squared_norm_multivariate(self):
        np.random.seed(42)
        data = np.random.multivariate_normal(
            mean=[0, 0],
        cov=[[1, 0], [0, 1]],
        size=1000
     )

        d = data.shape[1]
        expected_value_of_X_squared= (np.mean(np.sum(data ** 2, axis=1)) / d) ** 0.5
        alpha_power_of_data = alpha_power(data, 2)
   

        assert np.allclose(alpha_power_of_data, expected_value_of_X_squared)

    def test_alpha_power_alpha_2_matches_normal_scale(self):
        np.random.seed(42)
        std = 100
        data = np.random.normal(0, std, 1000)
        alpha_power_of_data = alpha_power(data, 2)
        assert np.isclose(alpha_power_of_data, std, rtol=0.05)
   
    def test_alpha_power_alpha_2_matches_normal_scale_multivariate(self):
        np.random.seed(42)
        std = 100
        d = 2
        data = np.random.multivariate_normal(
        mean=np.zeros(d),
        cov=(std**2) * np.eye(d),
        size=1000
    )
        alpha_power_of_data = alpha_power(data, 2)
        assert np.isclose(alpha_power_of_data, std, rtol=0.05)
    def test_alpha_power_alpha_1_matches_cauchy_scale(self):
        np.random.seed(42)
        scale = 100
        data = cauchy.rvs(loc = 0, scale = scale, size = 1000)
        alpha_power_of_data = alpha_power(data, 1)
        assert np.isclose(alpha_power_of_data, scale, rtol=0.05)

    #Cauchy
    def test_alpha_power_alpha_1_matches_cauchy_scale_multivariate(self):
        np.random.seed(42)
        scale = 100
        d = 2
        data = multivariate_t.rvs(
        loc=np.zeros(d),
        shape=(scale**2) * np.eye(d),
        df=1,
        size=2000  # larger sample for stability
    )
        alpha_power_of_data = alpha_power(data, 1)
        assert np.isclose(alpha_power_of_data, scale, rtol=0.05)

    @pytest.mark.parametrize("alpha", [0.2, 0.3, 1.2, 1.5, 0.9, 1.1, 2])
    def test_alpha_power_is_always_positive(self, alpha: float):
        data = np.random.normal(0, 1, 1000)
        alpha_power_of_data = alpha_power(data, alpha)
        assert alpha_power_of_data > 0

 
    @pytest.mark.parametrize(
        ("alpha", "scale"), 
        [
            (0.2, -0.4),
            (0.5, 2),
            (1.0, 4),
            (1.2, 10.),
            (1.5, -20.),
            (1.5, 20.),
            (2., 40.),
        ]    
    )
    def test_scale_equivariance_of_alpha_power(self, alpha: float, scale: float):
        data = np.random.normal(0, 1, 1000)
        alpha_power_of_data = alpha_power(data, alpha)
        alpha_power_of_scaled_data = alpha_power(data * scale, alpha)
        assert np.isclose(alpha_power_of_data * abs(scale), alpha_power_of_scaled_data, rtol=0.05), f"{alpha_power_of_data}"

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5, 2.0])
    def test_alpha_power_is_sign_invariant(self, alpha):
    # Property: P_alpha(cX) = |c| P_alpha(X)
    # Taking c = -1 gives P_alpha(-X) = P_alpha(X).
    # Intuition: flipping the sign of all data points does not change
    # the “scale” or spread of the distribution.
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        assert np.isclose(
         alpha_power(data, alpha),
        alpha_power(-data, alpha),
        rtol=0.05
    )
  

class TestAlphaLocationCorrectness:
    # 1) For alpha = 2, the location should match the mean in the univariate case.
    def test_alpha_location_alpha_2_matches_mean_univariate(self):
        np.random.seed(42)
        mean = 10
        std = 3
        data = np.random.normal(mean, std, 2000)

        estimated_location = alpha_location(data, 2.0)

        assert np.isclose(estimated_location, np.mean(data), rtol=0.05, atol=0.2)


    # 2) For alpha = 2, the location should match the mean in the multivariate case.
    def test_alpha_location_alpha_2_matches_mean_multivariate(self):
        np.random.seed(42)
        mean = np.array([10.0, -5.0])
        cov = np.array([[9.0, 0.0], [0.0, 4.0]])
        data = np.random.multivariate_normal(mean, cov, size=3000)
        estimated_location = alpha_location(data, 2.0)
        sample_mean = np.mean(data, axis=0)
        assert np.allclose(estimated_location, sample_mean, rtol=0.05, atol=0.25)

    # 3) Affine equivariance in the univariate case:
#    L_{aX+b} = a L_X + b
    @pytest.mark.parametrize(
        ("alpha", "a", "b"),
    [
        (0.5, 2.0, 3.0),
        (1.0, -4.0, 10.0),
        (1.5, 0.5, -7.0),
        (2.0, -3.0, 1.0),
    ]
)
    def test_alpha_location_affine_equivariance_univariate(self, alpha, a, b):
        np.random.seed(42)
        data = np.random.normal(0, 1, 1500)

        original_location = alpha_location(data, alpha)
        transformed_location = alpha_location(a * data + b, alpha)

        assert np.isclose(transformed_location, a * original_location + b, rtol=0.05, atol=0.3)


# 4) Translation equivariance in the multivariate case:
#    L_{X+b} = L_X + b
    @pytest.mark.parametrize("alpha", [1.0,2.0])
    def test_alpha_location_translation_equivariance_multivariate(self, alpha):
        np.random.seed(42)
        shift = np.array([20.0, -15.0])
        data = np.random.multivariate_normal([0.0, 0.0], np.eye(2), size=2000)

        location_original = alpha_location(data, alpha)
        location_shifted = alpha_location(data + shift, alpha)

        assert np.allclose(location_shifted, location_original + shift, rtol=0.05, atol=0.4)
    # 6) For isotropic multivariate Cauchy centered at mu, alpha=1 location should recover mu.
    def test_alpha_location_alpha_1_matches_cauchy_center_multivariate(self):
        np.random.seed(42)
        mu = np.array([12.0, -9.0])
        scale = 4.0

        data = multivariate_t.rvs(
        loc=mu,
        shape=(scale**2) * np.eye(2),
        df=1,
        size=4000
    )

        estimated_location = alpha_location(data, 1.0)

        assert np.allclose(estimated_location, mu, rtol=0.1, atol=0.6)
    # 7) Additivity for independent symmetric variables:
    #    L_{X+Y} = L_X + L_Y
    def test_alpha_location_additivity_for_independent_symmetric_univariate(self):
        np.random.seed(42)
        x = np.random.normal(5.0, 2.0, 2000)
        y = np.random.normal(-3.0, 1.5, 2000)

        location_x = alpha_location(x, 2.0)
        location_y = alpha_location(y, 2.0)
        location_sum = alpha_location(x + y, 2.0)

        assert np.isclose(location_sum, location_x + location_y, rtol=0.05, atol=0.25)
    # 8) Sign equivariance is a special case of linearity:
    #    L_{-X} = -L_X
    @pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5, 2.0])
    def test_alpha_location_sign_equivariance(self, alpha):
        np.random.seed(42)
        data = np.random.normal(7.0, 2.0, 2000)

        location = alpha_location(data, alpha)
        negated_location = alpha_location(-data, alpha)

        assert np.isclose(negated_location, -location, rtol=0.05, atol=0.3)






        
    '''def test_alpha_location_alpha_2_matches_normal_location(self):
        ...

    def test_alpha_location_alpha_1_matches_cauchy_location(self):
        ...

    def test_alpha_location_alpha_1_matches_cauchy_location_for_multivariate_data(self):
        ...

    @pytest.mark.parametrize("alpha", [0.2, 0.3, 1.2, 1.5])
    def test_alpha_power_alpha_not_1_fails_for_multivariate_data(self, alpha: float):
        ...


class TestDimensionsAndShapes:
    ...
    '''