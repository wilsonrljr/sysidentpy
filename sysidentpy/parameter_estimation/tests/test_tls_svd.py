"""Compare TLS with its original full-SVD numerical definition."""

import warnings

import numpy as np
import pytest

from sysidentpy import config_context
from sysidentpy._lib._array_api import get_namespace
from sysidentpy.parameter_estimation import TotalLeastSquares
from sysidentpy.tests._array_api_asserts import assert_allclose


@pytest.mark.parametrize("backend_name", ["numpy", "array_api_strict", "torch"])
@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize("case", ["tall", "square", "wide", "near_rank"])
def test_tls_matches_full_svd(backend_name, dtype_name, case):
    backend = pytest.importorskip(backend_name)
    rows = {"tall": 40, "square": 5, "wide": 3, "near_rank": 40}[case]
    rng = np.random.default_rng(703)
    psi = rng.normal(size=(rows, 4)).astype(dtype_name)
    if case == "near_rank":
        psi[:, 1] = psi[:, 0] + 1e-3 * psi[:, 1]
    y = psi @ np.array([[0.2], [-0.1], [0.4], [0.3]], dtype=dtype_name)
    y += rng.normal(scale=0.01, size=y.shape).astype(dtype_name)
    psi = backend.asarray(psi, dtype=getattr(backend, dtype_name))
    y = backend.asarray(y, dtype=getattr(backend, dtype_name))
    with config_context(array_api_dispatch=True):
        xp = get_namespace(psi, y)
        augmented = xp.concat([psi, y], axis=1)
        _, _, vh = xp.linalg.svd(augmented, full_matrices=True)
        expected = -vh.T[:4, 4:] / vh.T[4:, 4:]
        with warnings.catch_warnings():
            # Wide psi has deficient column rank, as in the existing public path.
            warnings.filterwarnings("ignore", message="Psi matrix might have")
            result = TotalLeastSquares().optimize(psi, y)
    assert result.shape == (4, 1)
    tolerance = 1e-5 if dtype_name == "float32" else 1e-12
    assert_allclose(result, expected, rtol=tolerance, atol=tolerance)
