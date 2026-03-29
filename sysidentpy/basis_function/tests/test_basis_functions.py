import pytest

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal

from sysidentpy import config_context
from sysidentpy._lib._array_api import _to_numpy
from sysidentpy.basis_function import Polynomial, Fourier
from sysidentpy.basis_function.basis_function_base import BaseBasisFunction


def test_fit_polynomial():
    basis_function = Polynomial(degree=2)
    data = np.array(([1, 1, 1], [2, 3, 4], [3, 3, 3]))
    max_lag = 1
    output = np.array([[4, 6, 8, 9, 12, 16], [9, 9, 9, 9, 9, 9]])

    r = basis_function.fit(data=data, max_lag=max_lag)

    assert_array_equal(output, r)


def test_fit_polynomial_predefined():
    basis_function = Polynomial(degree=2)
    data = np.array(([1, 1, 1], [2, 3, 4], [3, 3, 3]))
    max_lag = 1
    predefined_regressors = np.array([0, 2, 4])
    output = np.array([[4, 8, 12], [9, 9, 9]])

    r = basis_function.fit(
        data=data, max_lag=max_lag, predefined_regressors=predefined_regressors
    )

    assert_array_equal(output, r)


def test_fit_polynomial_predefined_accepts_array_api_inputs():
    xp = pytest.importorskip("array_api_strict")
    basis_function = Polynomial(degree=2)
    data = xp.asarray(np.array(([1.0, 1.0, 1.0], [2.0, 3.0, 4.0], [3.0, 3.0, 3.0])))
    predefined_regressors = xp.asarray(np.array([0, 2, 4]))
    output = np.array([[4.0, 8.0, 12.0], [9.0, 9.0, 9.0]])

    with config_context(array_api_dispatch=True):
        result = basis_function.fit(
            data=data,
            max_lag=1,
            predefined_regressors=predefined_regressors,
        )

    assert result.__array_namespace__().__name__ == xp.__name__
    assert_array_equal(_to_numpy(result), output)


def test_transform_polynomial():
    basis_function = Polynomial(degree=2)
    data = np.array(([1, 1, 1], [2, 3, 4], [3, 3, 3]))
    max_lag = 1
    output = np.array([[4, 6, 8, 9, 12, 16], [9, 9, 9, 9, 9, 9]])

    r = basis_function.transform(data=data, max_lag=max_lag)

    assert_array_equal(output, r)


def test_fit_fourier():
    basis_function = Fourier(n=5, ensemble=False)
    data = np.array(([1, 1, 1], [2, 3, 4], [3, 3, 3]))
    max_lag = 1
    output = np.array(
        [
            [
                -0.9899925,
                0.14112001,
                0.96017029,
                -0.2794155,
                -0.91113026,
                0.41211849,
                0.84385396,
                -0.53657292,
                -0.75968791,
                0.65028784,
                -0.65364362,
                -0.7568025,
                -0.14550003,
                0.98935825,
                0.84385396,
                -0.53657292,
                -0.95765948,
                -0.28790332,
                0.40808206,
                0.91294525,
            ],
            [
                -0.9899925,
                0.14112001,
                0.96017029,
                -0.2794155,
                -0.91113026,
                0.41211849,
                0.84385396,
                -0.53657292,
                -0.75968791,
                0.65028784,
                -0.9899925,
                0.14112001,
                0.96017029,
                -0.2794155,
                -0.91113026,
                0.41211849,
                0.84385396,
                -0.53657292,
                -0.75968791,
                0.65028784,
            ],
        ]
    )

    r = basis_function.fit(data=data, max_lag=max_lag)

    assert_almost_equal(output, r, decimal=7)


def test_fit_fourier_predefined():
    basis_function = Fourier(n=5, ensemble=False)
    data = np.array(([1, 1, 1], [2, 3, 4], [3, 3, 3]))
    max_lag = 1
    predefined_regressors = np.array([0, 2, 4])
    output = np.array(
        [[-0.9899925, 0.96017029, -0.91113026], [-0.9899925, 0.96017029, -0.91113026]]
    )

    r = basis_function.fit(
        data=data, max_lag=max_lag, predefined_regressors=predefined_regressors
    )

    assert_almost_equal(output, r, decimal=7)


def test_transform_fourier():
    basis_function = Fourier(n=5, ensemble=False)
    data = np.array(([1, 1, 1], [2, 3, 4], [3, 3, 3]))
    max_lag = 1
    output = np.array(
        [
            [
                -0.9899925,
                0.14112001,
                0.96017029,
                -0.2794155,
                -0.91113026,
                0.41211849,
                0.84385396,
                -0.53657292,
                -0.75968791,
                0.65028784,
                -0.65364362,
                -0.7568025,
                -0.14550003,
                0.98935825,
                0.84385396,
                -0.53657292,
                -0.95765948,
                -0.28790332,
                0.40808206,
                0.91294525,
            ],
            [
                -0.9899925,
                0.14112001,
                0.96017029,
                -0.2794155,
                -0.91113026,
                0.41211849,
                0.84385396,
                -0.53657292,
                -0.75968791,
                0.65028784,
                -0.9899925,
                0.14112001,
                0.96017029,
                -0.2794155,
                -0.91113026,
                0.41211849,
                0.84385396,
                -0.53657292,
                -0.75968791,
                0.65028784,
            ],
        ]
    )

    r = basis_function.transform(data=data, max_lag=max_lag)

    assert_almost_equal(output, r, decimal=7)


class _DummyBasis(BaseBasisFunction):
    def __init__(self, degree=3):
        super().__init__(degree=degree)

    def fit(
        self,
        data,
        max_lag=1,
        ylag=1,
        xlag=1,
        model_type="NARMAX",
        predefined_regressors=None,
    ):
        return data

    def transform(
        self,
        data,
        max_lag=1,
        ylag=1,
        xlag=1,
        model_type="NARMAX",
        predefined_regressors=None,
    ):
        return data


def test_base_basis_function_init_sets_degree():
    dummy = _DummyBasis(degree=5)
    assert dummy.degree == 5
    data = np.eye(2)
    assert_array_equal(dummy.transform(data), data)
