"""Integration tests for configurable basis-function bias handling."""

import pickle

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from sysidentpy.basis_function import Bilinear, Legendre, Polynomial
from sysidentpy.basis_function.basis_function_base import BaseBasisFunction
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.narmax_base import RegressorDictionary
from sysidentpy.parameter_estimation.estimators import LeastSquares
from sysidentpy.utils.information_matrix import build_input_output_matrix


class _CustomBasisWithoutBiasOption(BaseBasisFunction):
    """Historical custom expansion that predates feature-code hooks and bias."""

    def __init__(self, degree=1):
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
        _ = (ylag, xlag, model_type)
        raw_features = data[max_lag:, :]
        features = np.column_stack([raw_features, np.sin(raw_features)])
        if predefined_regressors is not None:
            features = features[:, predefined_regressors]
        return features

    def transform(
        self,
        data,
        max_lag=1,
        ylag=1,
        xlag=1,
        model_type="NARMAX",
        predefined_regressors=None,
    ):
        return self.fit(
            data,
            max_lag,
            ylag,
            xlag,
            model_type,
            predefined_regressors,
        )


class _TruncatedPolynomial(Polynomial):
    """Custom Polynomial subclass whose feature layout predates code hooks."""

    def fit(
        self,
        data,
        max_lag=1,
        ylag=1,
        xlag=1,
        model_type="NARMAX",
        predefined_regressors=None,
    ):
        features = super().fit(
            data,
            max_lag,
            ylag,
            xlag,
            model_type,
            predefined_regressors=None,
        )[:, :2]
        if predefined_regressors is not None:
            features = features[:, np.asarray(predefined_regressors, dtype=np.intp)]
        return features


class _TruncatedLegendre(Legendre):
    """Custom Legendre subclass whose feature layout predates code hooks."""

    def fit(
        self,
        data,
        max_lag=1,
        ylag=1,
        xlag=1,
        model_type="NARMAX",
        predefined_regressors=None,
    ):
        features = super().fit(
            data,
            max_lag,
            ylag,
            xlag,
            model_type,
            predefined_regressors=None,
        )[:, :2]
        if predefined_regressors is not None:
            features = features[:, np.asarray(predefined_regressors, dtype=np.intp)]
        return features


class _UnchangedLegendre(Legendre):
    """Legendre subclass that preserves its inherited feature layout."""


class _LegacyRegressorSpaceFROLS(FROLS):
    """Model subclass preserving the historical regressor-space signature."""

    def regressor_space(self, n_inputs):
        return super().regressor_space(n_inputs)


def _generate_bias_free_siso_data(n_samples=320):
    rng = np.random.default_rng(193)
    x = rng.uniform(-1.0, 1.0, size=(n_samples, 1))
    y = np.zeros((n_samples, 1))

    for k in range(2, n_samples):
        y[k, 0] = (
            0.2 * y[k - 1, 0] + 0.1 * y[k - 1, 0] * x[k - 1, 0] + 0.9 * x[k - 2, 0]
        )

    split = 240
    return x[:split], x[split:], y[:split], y[split:]


def _fit_frols(basis_function):
    x_train, _, y_train, _ = _generate_bias_free_siso_data()
    model = FROLS(
        ylag=2,
        xlag=2,
        n_terms=3,
        order_selection=False,
        estimator=LeastSquares(),
        basis_function=basis_function,
    )
    return model.fit(X=x_train, y=y_train)


def _matrix_from_codes(dictionary, lagged_data, codes, max_lag):
    x_codes, y_codes = dictionary.create_narmax_code(n_inputs=1)
    lagged_column_codes = np.concatenate(([0], y_codes, x_codes))
    column_by_code = {
        int(code): column for column, code in enumerate(lagged_column_codes)
    }
    column_indices = np.array(
        [[column_by_code[int(code)] for code in term] for term in codes]
    )
    return np.prod(lagged_data[max_lag:, column_indices], axis=2)


@pytest.mark.parametrize("basis_cls", [Polynomial, Bilinear])
def test_default_bias_matches_explicit_true_and_false_removes_only_constant(
    basis_cls,
):
    data = np.array(
        [
            [1.0, 0.1, -0.4],
            [1.0, 0.3, 0.8],
            [1.0, -0.6, 0.5],
            [1.0, 0.9, -0.2],
        ]
    )
    fit_kwargs = {"data": data, "max_lag": 1, "ylag": 1, "xlag": 1}

    default = basis_cls(degree=2).fit(**fit_kwargs)
    with_bias = basis_cls(degree=2, include_bias=True).fit(**fit_kwargs)
    without_bias = basis_cls(degree=2, include_bias=False).fit(**fit_kwargs)

    assert_array_equal(default, with_bias)
    bias_columns = np.flatnonzero(np.all(with_bias == 1.0, axis=0))
    assert_array_equal(bias_columns.shape, (1,))
    assert_array_equal(
        without_bias,
        np.delete(with_bias, bias_columns.item(), axis=1),
    )
    assert not np.any(np.all(without_bias == 1.0, axis=0))


@pytest.mark.parametrize("basis_cls", [Polynomial, Bilinear])
@pytest.mark.parametrize("include_bias", [True, False])
def test_regressor_matrix_and_codes_have_identical_order(basis_cls, include_bias):
    x_train, _, y_train, _ = _generate_bias_free_siso_data(n_samples=40)
    max_lag = 1
    lagged_data = build_input_output_matrix(x_train, y_train, xlag=1, ylag=1)
    basis_function = basis_cls(degree=2, include_bias=include_bias)
    regressor_matrix = basis_function.fit(
        lagged_data,
        max_lag=max_lag,
        ylag=1,
        xlag=1,
    )
    dictionary = RegressorDictionary(
        xlag=1,
        ylag=1,
        basis_function=basis_function,
    )
    regressor_code = dictionary.regressor_space(
        n_inputs=1,
        n_features=regressor_matrix.shape[1],
    )

    expected_matrix = _matrix_from_codes(
        dictionary,
        lagged_data,
        regressor_code,
        max_lag,
    )
    assert regressor_code.shape[0] == regressor_matrix.shape[1]
    assert_allclose(regressor_matrix, expected_matrix, rtol=0, atol=0)
    assert bool(np.any(np.all(regressor_code == 0, axis=1))) is include_bias


def test_frols_without_bias_aligns_selected_matrix_codes_and_parameters():
    x_train, _, y_train, _ = _generate_bias_free_siso_data()
    model = _fit_frols(Polynomial(degree=2, include_bias=False))
    selected_indices = model.pivv[: model.n_terms]
    lagged_data = build_input_output_matrix(
        x_train,
        y_train,
        xlag=model.xlag,
        ylag=model.ylag,
    )
    selected_matrix = model.basis_function.transform(
        lagged_data,
        model.max_lag,
        model.ylag,
        model.xlag,
        model.model_type,
        predefined_regressors=selected_indices,
    )
    expected_matrix = _matrix_from_codes(
        model,
        lagged_data,
        model.final_model,
        model.max_lag,
    )
    expected_theta, *_ = np.linalg.lstsq(
        selected_matrix,
        y_train[model.max_lag :],
        rcond=None,
    )

    assert not np.any(np.all(model.regressor_code == 0, axis=1))
    assert_array_equal(model.final_model, model.regressor_code[selected_indices])
    assert selected_matrix.shape[1] == model.final_model.shape[0]
    assert model.theta.shape[0] == model.final_model.shape[0]
    assert_allclose(selected_matrix, expected_matrix, rtol=0, atol=0)
    assert_allclose(model.theta, expected_theta, rtol=1e-12, atol=1e-12)


def test_frols_default_and_explicit_bias_true_are_equivalent():
    default = _fit_frols(Polynomial(degree=2))
    explicit = _fit_frols(Polynomial(degree=2, include_bias=True))
    _, x_valid, _, y_valid = _generate_bias_free_siso_data()

    assert_array_equal(default.regressor_code, explicit.regressor_code)
    assert_array_equal(default.pivv, explicit.pivv)
    assert_array_equal(default.final_model, explicit.final_model)
    assert_allclose(default.theta, explicit.theta, rtol=0, atol=0)
    assert_allclose(
        default.predict(X=x_valid, y=y_valid),
        explicit.predict(X=x_valid, y=y_valid),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("include_bias", [True, False])
@pytest.mark.parametrize("steps_ahead", [None, 1, 3])
def test_frols_fit_and_prediction_paths_work_in_both_bias_modes(
    include_bias,
    steps_ahead,
):
    model = _fit_frols(Polynomial(degree=2, include_bias=include_bias))
    _, x_valid, _, y_valid = _generate_bias_free_siso_data()

    prediction = model.predict(
        X=x_valid,
        y=y_valid,
        steps_ahead=steps_ahead,
    )

    assert prediction.shape == y_valid.shape
    assert_allclose(prediction, y_valid, rtol=1e-9, atol=1e-10)


def test_fitted_model_pickle_preserves_include_bias_false():
    model = _fit_frols(Polynomial(degree=2, include_bias=False))
    restored = pickle.loads(pickle.dumps(model))
    _, x_valid, _, y_valid = _generate_bias_free_siso_data()

    assert restored.basis_function.include_bias is False
    assert_array_equal(restored.regressor_code, model.regressor_code)
    assert_array_equal(restored.final_model, model.final_model)
    assert_allclose(
        restored.predict(X=x_valid, y=y_valid),
        model.predict(X=x_valid, y=y_valid),
        rtol=0,
        atol=0,
    )


def test_legacy_pickle_without_include_bias_keeps_historical_behavior():
    model = _fit_frols(Polynomial(degree=2))
    _, x_valid, _, y_valid = _generate_bias_free_siso_data()
    expected = model.predict(X=x_valid, y=y_valid, steps_ahead=1)

    del model.basis_function.include_bias
    restored = pickle.loads(pickle.dumps(model))
    restored_codes = restored.regressor_space(
        n_inputs=1,
        n_features=restored.regressor_code.shape[0],
    )
    prediction = restored.predict(X=x_valid, y=y_valid, steps_ahead=1)

    assert_array_equal(restored_codes, restored.regressor_code)
    assert_allclose(prediction, expected, rtol=0, atol=0)


def test_custom_basis_without_include_bias_fits_predicts_and_keeps_code_space():
    x_train, x_valid, y_train, y_valid = _generate_bias_free_siso_data()
    model = FROLS(
        xlag=2,
        ylag=2,
        n_terms=3,
        order_selection=False,
        estimator=LeastSquares(),
        basis_function=_CustomBasisWithoutBiasOption(degree=1),
    ).fit(X=x_train, y=y_train)
    restored = pickle.loads(pickle.dumps(model))

    prediction = restored.predict(X=x_valid, y=y_valid, steps_ahead=1)

    assert_array_equal(
        restored.regressor_code,
        np.array(
            [
                [0],
                [1001],
                [1002],
                [2001],
                [2002],
                [0],
                [1001],
                [1002],
                [2001],
                [2002],
            ]
        ),
    )
    assert_array_equal(
        restored.final_model,
        restored.regressor_code[restored.pivv[: restored.n_terms]],
    )
    assert restored.theta.shape == (restored.n_terms, 1)
    assert prediction.shape == y_valid.shape
    assert np.all(np.isfinite(prediction))


@pytest.mark.parametrize(
    ("basis_cls", "expected_codes"),
    [
        (_TruncatedPolynomial, np.array([[1001, 0], [1002, 0]])),
        (_TruncatedLegendre, np.array([[1001, 0], [1001, 1001]])),
    ],
)
def test_custom_native_subclass_without_hook_uses_compatible_fallback(
    basis_cls,
    expected_codes,
):
    x_train, x_valid, y_train, y_valid = _generate_bias_free_siso_data()
    model = FROLS(
        xlag=2,
        ylag=2,
        n_terms=2,
        order_selection=False,
        estimator=LeastSquares(),
        basis_function=basis_cls(degree=2, include_bias=False),
    ).fit(X=x_train, y=y_train)

    prediction = model.predict(X=x_valid, y=y_valid, steps_ahead=1)

    assert_array_equal(model.regressor_code, expected_codes)
    assert not np.any(np.all(model.regressor_code == 0, axis=1))
    assert_array_equal(
        model.final_model,
        model.regressor_code[model.pivv[: model.n_terms]],
    )
    assert model.theta.shape == (model.n_terms, 1)
    assert prediction.shape == y_valid.shape
    assert np.all(np.isfinite(prediction))


def test_unchanged_native_subclass_keeps_inherited_canonical_hook():
    basis_function = _UnchangedLegendre(degree=2, include_bias=False)
    native_basis = Legendre(degree=2, include_bias=False)
    subclass_codes = RegressorDictionary(
        xlag=1,
        ylag=1,
        basis_function=basis_function,
    ).regressor_space(n_inputs=1, n_features=4)
    native_codes = RegressorDictionary(
        xlag=1,
        ylag=1,
        basis_function=native_basis,
    ).regressor_space(n_inputs=1, n_features=4)

    assert_array_equal(subclass_codes, native_codes)


def test_legacy_model_regressor_space_override_fits_and_predicts_without_bias():
    x_train, x_valid, y_train, y_valid = _generate_bias_free_siso_data()
    model = _LegacyRegressorSpaceFROLS(
        xlag=2,
        ylag=2,
        n_terms=3,
        order_selection=False,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=2, include_bias=False),
    ).fit(X=x_train, y=y_train)

    prediction = model.predict(X=x_valid, y=y_valid, steps_ahead=1)

    assert model.regressor_code.shape[0] == 14
    assert_array_equal(
        model.final_model,
        model.regressor_code[model.pivv[: model.n_terms]],
    )
    assert not np.any(np.all(model.regressor_code == 0, axis=1))
    assert prediction.shape == y_valid.shape
    assert np.all(np.isfinite(prediction))
