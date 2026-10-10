"""
Unit tests for alpha_stable.pdf() dimension handling.

Tests SciPy's implicit broadcasting conventions:
- x can be scalar, 1D, or ND array
- shape parameters (alpha, beta) can be scalar or array
- loc and scale can be scalar or array
- output shape follows NumPy broadcasting rules
"""

import pytest
import numpy as np
from numpy.testing import assert_array_equal
from aub_htp import alpha_stable


class TestPdfScalarInputs:
    """Test pdf with scalar inputs."""

    def test_scalar_x_scalar_params(self):
        """Scalar x, scalar alpha, scalar beta -> scalar output."""
        result = alpha_stable.pdf(0.0, alpha=1.5, beta=0.0)
        assert np.isscalar(result) or result.shape == ()

    def test_scalar_x_scalar_params_with_loc_scale(self):
        """Scalar x with loc and scale -> scalar output."""
        result = alpha_stable.pdf(0.0, alpha=1.5, beta=0.0, loc=1.0, scale=2.0)
        assert np.isscalar(result) or result.shape == ()


class TestPdf1DArrayX:
    """Test pdf with 1D array x."""

    def test_1d_x_scalar_params(self):
        """1D x array, scalar params -> 1D output with same length."""
        x = np.linspace(-2, 2, 10)
        result = alpha_stable.pdf(x, 1.5, 0.0)
        assert result.shape == (10,)

    def test_1d_x_scalar_params_with_loc_scale(self):
        """1D x with scalar loc/scale -> 1D output."""
        x = np.linspace(-2, 2, 10)
        result = alpha_stable.pdf(x, alpha=1.5, beta=0.0, loc=0.0, scale=1.0)
        assert result.shape == (10,)

    def test_1d_x_length_1(self):
        """1D x with single element -> shape (1,)."""
        x = np.array([0.0])
        result = alpha_stable.pdf(x, alpha=1.5, beta=0.0)
        assert result.shape == (1,)

    def test_list_x_scalar_params(self):
        """List x (auto-converted) -> 1D output."""
        x = [-1.0, 0.0, 1.0]
        result = alpha_stable.pdf(x, alpha=1.5, beta=0.0)
        assert result.shape == (3,)


class TestPdfArrayParams:
    """Test pdf with array-valued alpha and beta."""

    def test_scalar_x_1d_alpha(self):
        """Scalar x, array alpha -> output broadcasts to alpha shape."""
        alphas = np.array([1.2, 1.5, 1.8])
        result = alpha_stable.pdf(0.0, alpha=alphas, beta=0.0)
        assert result.shape == (3,)

    def test_scalar_x_1d_beta(self):
        """Scalar x, array beta -> output broadcasts to beta shape."""
        betas = np.array([-0.5, 0.0, 0.5])
        result = alpha_stable.pdf(0.0, alpha=1.5, beta=betas)
        assert result.shape == (3,)

    def test_scalar_x_1d_alpha_1d_beta_same_length(self):
        """Scalar x, array alpha, array beta (same length) -> 1D output."""
        alphas = np.array([1.2, 1.5, 1.8])
        betas = np.array([-0.5, 0.0, 0.5])
        result = alpha_stable.pdf(0.0, alpha=alphas, beta=betas)
        assert result.shape == (3,)

    def test_1d_x_1d_alpha_same_length(self):
        """1D x and 1D alpha with same length -> element-wise, 1D output."""
        x = np.array([0.0, 0.5, 1.0])
        alphas = np.array([1.2, 1.5, 1.8])
        result = alpha_stable.pdf(x, alpha=alphas, beta=0.0)
        assert result.shape == (3,)

    def test_1d_x_1d_alpha_1d_beta_same_length(self):
        """All 1D with same length -> element-wise pairing."""
        x = np.array([0.0, 0.5, 1.0])
        alphas = np.array([1.2, 1.5, 1.8])
        betas = np.array([-0.5, 0.0, 0.5])
        result = alpha_stable.pdf(x, alpha=alphas, beta=betas)
        assert result.shape == (3,)


class TestPdfArrayLocScale:
    """Test pdf with array-valued loc and scale."""

    def test_1d_x_array_loc(self):
        """1D x with array loc -> broadcasts."""
        x = np.array([0.0, 1.0, 2.0])
        locs = np.array([0.0, 0.5, 1.0])
        result = alpha_stable.pdf(x, alpha=1.5, beta=0.0, loc=locs)
        assert result.shape == (3,)

    def test_1d_x_array_scale(self):
        """1D x with array scale -> broadcasts."""
        x = np.array([0.0, 1.0, 2.0])
        scales = np.array([1.0, 2.0, 3.0])
        result = alpha_stable.pdf(x, alpha=1.5, beta=0.0, scale=scales)
        assert result.shape == (3,)

    def test_scalar_x_array_loc_scale(self):
        """Scalar x with array loc and scale."""
        locs = np.array([0.0, 1.0])
        scales = np.array([1.0, 2.0])
        result = alpha_stable.pdf(0.0, alpha=1.5, beta=0.0, loc=locs, scale=scales)
        assert result.shape == (2,)


class TestPdf2DArrayX:
    """Test pdf with 2D array x (SciPy flattens internally)."""

    def test_2d_x_scalar_params(self):
        """2D x array -> output shape matches input or is flattened."""
        x = np.array([[0.0, 1.0], [2.0, 3.0]])
        result = alpha_stable.pdf(x, alpha=1.5, beta=0.0)
        # SciPy typically preserves shape or flattens to 1D
        assert result.size == 4

    def test_2d_x_preserves_or_flattens(self):
        """Verify 2D behavior - either (2,2) or (4,)."""
        x = np.array([[0.0, 1.0], [2.0, 3.0]])
        result = alpha_stable.pdf(x, alpha=1.5, beta=0.0)
        assert result.shape == (2, 2) or result.shape == (4,)


class TestPdfOutputType:
    """Test that output types are correct."""

    def test_returns_numpy_array_or_scalar(self):
        """Output should be numpy array or scalar float."""
        result = alpha_stable.pdf(0.0, alpha=1.5, beta=0.0)
        assert isinstance(result, (np.ndarray, np.floating, float))

    def test_array_input_returns_array(self):
        """Array input should return array."""
        x = np.linspace(-1, 1, 5)
        result = alpha_stable.pdf(x, alpha=1.5, beta=0.0)
        assert isinstance(result, np.ndarray)

    def test_output_dtype_is_float(self):
        """Output dtype should be floating point."""
        x = np.linspace(-1, 1, 5)
        result = alpha_stable.pdf(x, alpha=1.5, beta=0.0)
        assert np.issubdtype(result.dtype, np.floating)


class TestPdfOutputValues:
    """Basic sanity checks on output values (not correctness)."""

    def test_pdf_non_negative(self):
        """PDF values should be non-negative."""
        x = np.linspace(-3, 3, 20)
        result = alpha_stable.pdf(x, alpha=1.5, beta=0.0)
        assert np.all(result >= 0)

    def test_pdf_finite(self):
        """PDF values should be finite for reasonable inputs."""
        x = np.linspace(-3, 3, 20)
        result = alpha_stable.pdf(x, alpha=1.5, beta=0.0)
        assert np.all(np.isfinite(result))

    def test_pdf_not_all_zero(self):
        """PDF should have non-zero values in the center."""
        x = np.linspace(-1, 1, 10)
        result = alpha_stable.pdf(x, alpha=1.5, beta=0.0)
        assert np.any(result > 0)


class TestPdfEdgeCases:
    """Edge cases for dimensions."""

    def test_empty_x_array(self):
        """Empty x array -> empty output."""
        x = np.array([])
        result = alpha_stable.pdf(x, alpha=1.5, beta=0.0)
        assert result.shape == (0,)

    def test_single_element_arrays(self):
        """Single element arrays for all inputs."""
        x = np.array([0.0])
        alpha = np.array([1.5])
        beta = np.array([0.0])
        result = alpha_stable.pdf(x, alpha=alpha, beta=beta)
        assert result.shape == (1,)

    def test_large_array(self):
        """Large array should work without error."""
        x = np.linspace(-10, 10, 1000)
        result = alpha_stable.pdf(x, alpha=1.5, beta=0.0)
        assert result.shape == (1000,)


class TestPdfParameterRanges:
    """Test dimension handling across parameter ranges."""

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5, 2.0])
    def test_various_alpha_values(self, alpha):
        """Test dimension consistency across alpha values."""
        x = np.linspace(-2, 2, 10)
        result = alpha_stable.pdf(x, alpha=alpha, beta=0.0)
        assert result.shape == (10,)

    @pytest.mark.parametrize("beta", [-1.0, -0.5, 0.0, 0.5, 1.0])
    def test_various_beta_values(self, beta):
        """Test dimension consistency across beta values."""
        x = np.linspace(-2, 2, 10)
        result = alpha_stable.pdf(x, alpha=1.5, beta=beta)
        assert result.shape == (10,)

    @pytest.mark.parametrize("alpha,beta", [
        (0.5, 0.0),
        (1.0, 0.5),
        (1.5, -0.5),
        (2.0, 0.0),
    ])
    def test_alpha_beta_combinations(self, alpha, beta):
        """Test dimensions across alpha-beta combinations."""
        x = np.linspace(-2, 2, 10)
        result = alpha_stable.pdf(x, alpha=alpha, beta=beta)
        assert result.shape == (10,)


class TestPdfS0Parameterization:
    """Test dimensions with S0 parameterization."""

    def test_s0_scalar_inputs(self):
        """S0 parameterization with scalar inputs."""
        dist = alpha_stable.with_parametrization("S0")
        result = dist.pdf(0.0, alpha=1.5, beta=0.0)
        assert np.isscalar(result) or result.shape == ()

    def test_s0_array_x(self):
        """S0 parameterization with array x."""
        dist = alpha_stable.with_parametrization("S0")
        x = np.linspace(-2, 2, 10)
        result = dist.pdf(x, alpha=1.5, beta=0.0)
        assert result.shape == (10,)

    def test_s1_scalar_inputs(self):
        """S1 parameterization (default) with scalar inputs."""
        dist = alpha_stable.with_parametrization("S1")
        result = dist.pdf(0.0, alpha=1.5, beta=0.0)
        assert np.isscalar(result) or result.shape == ()

    def test_s1_array_x(self):
        """S1 parameterization with array x."""
        dist = alpha_stable.with_parametrization("S1")
        x = np.linspace(-2, 2, 10)
        result = dist.pdf(x, alpha=1.5, beta=0.0)
        assert result.shape == (10,)


class TestPdfPositionalArgs:
    """Test dimension handling with positional arguments (SciPy style)."""

    def test_positional_alpha_beta(self):
        """Positional args: pdf(x, alpha, beta)."""
        x = np.linspace(-2, 2, 10)
        result = alpha_stable.pdf(x, 1.5, 0.0)
        assert result.shape == (10,)

    def test_positional_with_loc_scale(self):
        """Positional with keyword loc/scale."""
        x = np.linspace(-2, 2, 10)
        result = alpha_stable.pdf(x, 1.5, 0.0, loc=0.0, scale=1.0)
        assert result.shape == (10,)
