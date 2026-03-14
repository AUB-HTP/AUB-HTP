import pytest
import numpy as np
from aub_htp.statistics import alpha_power, alpha_location


class TestAlphaPowerCorrectness:
    def test_alpha_power_alpha_2_matches_normal_scale(self):
        ...

    def test_alpha_power_alpha_1_matches_cauchy_scale(self):
        ...
    

    def test_alpha_power_alpha_1_matches_cauchy_scale_for_multivariate_data(self):
        ...
    
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
        ...


class TestAlphaLocationCorrectness:
    def test_alpha_location_alpha_2_matches_normal_location(self):
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
