import numpy as np
import pytest

from aub_htp.pdf.multivariate import _g_general_direct, g_alpha_d


@pytest.mark.parametrize("alpha", [0.5, 0.7, 0.8, 0.9, 0.95, 0.99])
@pytest.mark.parametrize("beta", [0.0, 0.5, -0.9])
def test_kernel_small_v_single_and_batch(alpha, beta):
    """g_alpha_d returned 0 for small |v| when alpha < 1 (alpha = 0.8, v = 1e-4:
    0 instead of 0.0421), depending on which other v shared the call."""
    vs = np.logspace(-6, 2, 33)
    b = np.full_like(vs, beta)
    ref = _g_general_direct(vs, b, alpha, 2)
    scale = np.abs(ref).max()
    single = np.array([g_alpha_d(vs[i:i + 1], b[i:i + 1], alpha, 2)[0]
                       for i in range(vs.size)])
    np.testing.assert_allclose(single, ref, rtol=0, atol=1e-6 * scale)
    np.testing.assert_allclose(g_alpha_d(vs, b, alpha, 2), ref, rtol=0, atol=1e-6 * scale)