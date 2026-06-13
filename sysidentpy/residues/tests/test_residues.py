import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal

from sysidentpy import config_context
from sysidentpy._lib._array_api import _to_numpy
from sysidentpy.residues.residues_correlation import (
    _input_ccf,
    _normalized_correlation,
    calculate_residues,
    compute_cross_correlation,
    compute_residues_autocorrelation,
    get_unnormalized_e_acf,
)


def test_compute_residues_autocorrelation():
    a = np.array([1, 3, 2, 4])
    b = np.array([1, 2, 3, 4])
    ee, upper, lower = compute_residues_autocorrelation(a, b)
    assert_almost_equal(ee, np.array([1, -0.5, 0, 0]))
    assert_almost_equal(upper, 0.7408103670980853, decimal=7)
    assert_almost_equal(lower, -0.7408103670980853, decimal=7)


def test_calculate_residues():
    a = np.array([1, 3, 2, 4])
    b = np.array([1, 2, 3, 4])
    e = calculate_residues(a, b)
    assert_equal(e, np.array([0, 1, -1, 0]))


def test_get_unnormalized_e_acf():
    e = np.array([0, 1, -1, 0])
    unnormalized_e_acf = get_unnormalized_e_acf(e)
    assert_equal(unnormalized_e_acf, np.array([0, 0, -1, 2, -1, 0, 0]))


def test_compute_cross_correlation():
    a = np.array([1, 3, 2, 4])
    b = np.array([1, 2, 3, 4])
    c = np.array([10, 22, 11, 9])
    ccf, upper, lower = compute_cross_correlation(a, b, c)
    assert_almost_equal(ccf, np.array([0.74161985, -0.70466426]), decimal=7)
    assert_almost_equal(upper, 0.7408103670980853, decimal=7)
    assert_almost_equal(lower, -0.7408103670980853, decimal=7)


def test_input_ccf():
    e = np.array([0, 1, -1, 0])
    c = np.array([10, 22, 11, 9])
    n = 7
    ccf, upper, lower = _input_ccf(e, c, n)
    assert_almost_equal(ccf, np.array([0.74161985, -0.70466426]), decimal=7)
    assert_almost_equal(upper, 0.7408103670980853, decimal=7)
    assert_almost_equal(lower, -0.7408103670980853, decimal=7)


def test_normalized_correlation():
    e = np.array([0, 1, -1, 0])
    c = np.array([10, 22, 11, 9])
    ccf = _normalized_correlation(c, e)
    assert_almost_equal(ccf, np.array([0.74161985, -0.70466426]), decimal=7)


def test_compute_cross_correlation_accepts_array_api_strict():
    xp = pytest.importorskip("array_api_strict")
    y = np.array([1.0, 3.0, 2.0, 4.0])
    yhat = np.array([1.0, 2.0, 3.0, 4.0])
    arr = np.array([10.0, 22.0, 11.0, 9.0])

    with config_context(array_api_dispatch=True):
        ccf, upper, lower = compute_cross_correlation(
            xp.asarray(y), xp.asarray(yhat), xp.asarray(arr)
        )

    assert_allclose(_to_numpy(ccf), np.array([0.74161985, -0.70466426]), rtol=1e-7)
    assert_almost_equal(upper, 0.7408103670980853, decimal=7)
    assert_almost_equal(lower, -0.7408103670980853, decimal=7)


def test_residues_autocorrelation_preserves_array_api_namespace():
    xp = pytest.importorskip("array_api_strict")
    y = np.array([1.0, 3.0, 2.0, 4.0])
    yhat = np.array([1.0, 2.0, 3.0, 4.0])

    with config_context(array_api_dispatch=True):
        e_acf, upper, lower = compute_residues_autocorrelation(
            xp.asarray(y), xp.asarray(yhat)
        )

    assert hasattr(e_acf, "__array_namespace__")
    assert_allclose(_to_numpy(e_acf), np.array([1.0, -0.5, 0.0, 0.0]))
    assert_almost_equal(upper, 0.7408103670980853, decimal=7)
    assert_almost_equal(lower, -0.7408103670980853, decimal=7)
