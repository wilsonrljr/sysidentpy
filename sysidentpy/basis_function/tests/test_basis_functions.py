import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from numpy.testing import assert_raises

from sysidentpy.basis_function import Polynomial, Fourier


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
