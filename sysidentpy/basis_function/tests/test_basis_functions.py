import pytest

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal

from sysidentpy import config_context
from sysidentpy.basis_function import (
    Bernstein,
    Bilinear,
    Fourier,
    Hermite,
    HermiteNormalized,
    Laguerre,
    Legendre,
    Polynomial,
)
from sysidentpy.basis_function.basis_function_base import BaseBasisFunction
from sysidentpy.tests._array_api_asserts import (
    assert_allclose as xp_assert_allclose,
    assert_array_equal as xp_assert_array_equal,
)


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
    xp_assert_array_equal(result, output)


def test_polynomial_include_bias_controls_constant_term_before_selection():
    data = np.array(
        [
            [1.0, 1.0, 2.0],
            [1.0, 3.0, 4.0],
            [1.0, 5.0, 6.0],
        ]
    )
    default = Polynomial(degree=2).fit(data, max_lag=1)
    explicit = Polynomial(degree=2, include_bias=True).fit(data, max_lag=1)
    without_bias = Polynomial(degree=2, include_bias=False).fit(data, max_lag=1)

    assert_array_equal(default, explicit)
    assert_array_equal(without_bias, explicit[:, 1:])
    assert not np.any(np.all(without_bias == 1, axis=0))

    selected = Polynomial(degree=2, include_bias=False).fit(
        data,
        max_lag=1,
        predefined_regressors=np.array([0, 2]),
    )
    assert_array_equal(selected, without_bias[:, [0, 2]])


def test_polynomial_include_bias_uses_cached_combinations_safely():
    data = np.column_stack(
        [np.ones(4), np.arange(4, dtype=float), np.arange(4, dtype=float) + 1]
    )
    basis_function = Polynomial(degree=2)
    with_bias = basis_function.fit(data, max_lag=1)

    basis_function.include_bias = False
    without_bias = basis_function.fit(data, max_lag=1)

    assert_array_equal(without_bias, with_bias[:, 1:])


def test_polynomial_without_bias_accepts_array_api_inputs():
    xp = pytest.importorskip("array_api_strict")
    data_np = np.array(
        [
            [1.0, 1.0, 2.0],
            [1.0, 3.0, 4.0],
            [1.0, 5.0, 6.0],
        ]
    )
    basis_function = Polynomial(degree=2, include_bias=False)
    expected = basis_function.fit(data_np, max_lag=1)

    with config_context(array_api_dispatch=True):
        result = basis_function.fit(xp.asarray(data_np), max_lag=1)

    assert result.__array_namespace__().__name__ == xp.__name__
    xp_assert_array_equal(result, expected)


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


def test_fit_fourier_accepts_array_api_inputs():
    xp = pytest.importorskip("array_api_strict")
    basis_function = Fourier(n=5, ensemble=False)
    data_np = np.array(([1.0, 1.0, 1.0], [2.0, 3.0, 4.0], [3.0, 3.0, 3.0]))
    expected = basis_function.fit(data=data_np, max_lag=1)

    with config_context(array_api_dispatch=True):
        result = basis_function.fit(data=xp.asarray(data_np), max_lag=1)

    assert result.__array_namespace__().__name__ == xp.__name__
    xp_assert_allclose(result, expected, rtol=0, atol=1.5e-7)


def test_fit_bilinear_accepts_array_api_inputs():
    xp = pytest.importorskip("array_api_strict")
    basis_function = Bilinear(degree=2)
    data_np = np.array(
        ([1.0, 1.0, 1.0], [2.0, 3.0, 4.0], [3.0, 3.0, 3.0], [4.0, 5.0, 6.0])
    )
    expected = basis_function.fit(data=data_np, max_lag=1, ylag=2, xlag=2)

    with config_context(array_api_dispatch=True):
        result = basis_function.fit(
            data=xp.asarray(data_np),
            max_lag=1,
            ylag=2,
            xlag=2,
        )

    assert result.__array_namespace__().__name__ == xp.__name__
    xp_assert_allclose(result, expected, rtol=0, atol=1.5e-12)


def test_fit_bilinear_predefined_accepts_array_api_inputs():
    xp = pytest.importorskip("array_api_strict")
    basis_function = Bilinear(degree=2)
    data_np = np.array(
        ([1.0, 1.0, 1.0], [2.0, 3.0, 4.0], [3.0, 3.0, 3.0], [4.0, 5.0, 6.0])
    )
    predefined_regressors = np.array([0, 2])
    expected = basis_function.fit(
        data=data_np,
        max_lag=1,
        ylag=2,
        xlag=2,
        predefined_regressors=predefined_regressors,
    )

    with config_context(array_api_dispatch=True):
        result = basis_function.fit(
            data=xp.asarray(data_np),
            max_lag=1,
            ylag=2,
            xlag=2,
            predefined_regressors=xp.asarray(predefined_regressors),
        )

    assert result.__array_namespace__().__name__ == xp.__name__
    xp_assert_allclose(result, expected, rtol=0, atol=1.5e-12)


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


@pytest.mark.parametrize(
    ("basis_cls", "basis_name"),
    [
        (Bernstein, "Bernstein"),
        (Legendre, "Legendre"),
        (Hermite, "Hermite"),
        (HermiteNormalized, "HermiteNormalized"),
        (Laguerre, "Laguerre"),
    ],
)
def test_scipy_basis_functions_reject_array_api_dispatch(basis_cls, basis_name):
    xp = pytest.importorskip("array_api_strict")
    data = xp.asarray(np.array(([1.0, 1.0, 1.0], [2.0, 3.0, 4.0], [3.0, 3.0, 3.0])))
    basis_function = basis_cls(degree=2)

    with config_context(array_api_dispatch=True):
        with pytest.raises(
            NotImplementedError,
            match=rf"{basis_name}.*requires NumPy inputs",
        ):
            basis_function.fit(data=data, max_lag=1)


@pytest.mark.parametrize(
    "basis_cls",
    [
        Polynomial,
        Bilinear,
        Legendre,
        Laguerre,
        Hermite,
        HermiteNormalized,
        Bernstein,
    ],
)
@pytest.mark.parametrize("invalid_value", [1, "False", np.bool_(True), None])
def test_include_bias_requires_a_boolean(basis_cls, invalid_value):
    with pytest.raises(TypeError, match="include_bias must be False or True"):
        basis_cls(include_bias=invalid_value)


def test_bernstein_validates_legacy_bias_before_applying_precedence():
    with pytest.raises(TypeError, match="include_bias must be False or True"):
        Bernstein(include_bias=True, bias="False")

    with pytest.raises(TypeError, match="include_bias must be False or True"):
        Bernstein(include_bias="False", bias=True)


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
