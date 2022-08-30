import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

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
