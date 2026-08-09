"""Cross-flow regressions for basis functions without an implicit bias term."""

from functools import partial
import inspect
from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.linear_model import LinearRegression  # type: ignore[reportMissingTypeStubs]

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
from sysidentpy.general_estimators import NARX
from sysidentpy.model_structure_selection import (
    AOLS,
    ER,
    FROLS,
    MetaMSS,
    OSF,
    RMSS,
    UOFR,
)
from sysidentpy.narmax_base import RegressorDictionary
from sysidentpy.parameter_estimation.estimators import LeastSquares
from sysidentpy.tests._array_api_asserts import assert_allclose as xp_assert_allclose
from sysidentpy.utils.information_matrix import build_input_output_matrix


def _generate_siso_data(n_samples=180, noise_scale=0.0):
    rng = np.random.default_rng(193)
    x = rng.uniform(-0.9, 0.9, size=(n_samples, 1))
    noise = rng.normal(scale=noise_scale, size=n_samples)
    y = np.zeros((n_samples, 1))

    for k in range(2, n_samples):
        y[k, 0] = (
            0.2 * y[k - 1, 0]
            + 0.1 * y[k - 1, 0] * x[k - 1, 0]
            + 0.9 * x[k - 2, 0]
            + noise[k]
        )

    split = int(0.75 * n_samples)
    return x[:split], x[split:], y[:split], y[split:]


def _native_matrix_and_codes(basis_function):
    x, _, y, _ = _generate_siso_data(n_samples=32)
    lagged_data = build_input_output_matrix(x, y, xlag=1, ylag=1)
    regressor_matrix = basis_function.fit(
        lagged_data,
        max_lag=1,
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
    return regressor_matrix, regressor_code


def _fit_frols_model(basis_function, x_train, y_train):
    return FROLS(
        ylag=2,
        xlag=2,
        n_terms=3,
        order_selection=False,
        estimator=LeastSquares(),
        basis_function=basis_function,
    ).fit(X=x_train, y=y_train)


def _generate_nar_data(n_samples=100):
    rng = np.random.default_rng(193)
    y = np.zeros((n_samples, 1))
    y[:2, 0] = [0.2, -0.1]

    for k in range(2, n_samples):
        y[k, 0] = 0.55 * y[k - 1, 0] - 0.1 * y[k - 2, 0] + 0.05 + 0.03 * rng.normal()

    return y[:-25], y[-25:]


def _fit_frols_nar(y_train, include_bias):
    return FROLS(
        ylag=2,
        xlag=2,
        model_type="NAR",
        n_terms=3,
        order_selection=False,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=2, include_bias=include_bias),
    ).fit(X=None, y=y_train)


def _fit_aols_nar(y_train, include_bias):
    return AOLS(
        ylag=2,
        xlag=2,
        model_type="NAR",
        k=3,
        L=1,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=2, include_bias=include_bias),
    ).fit(X=None, y=y_train)


def _fit_rmss_nar(y_train, include_bias):
    return RMSS(
        ylag=2,
        xlag=2,
        model_type="NAR",
        n_terms=3,
        order_selection=False,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=2, include_bias=include_bias),
    ).fit(X=None, y=y_train)


def _fit_frols_non_polynomial_nar(y_train, basis_function):
    return FROLS(
        ylag=2,
        xlag=2,
        model_type="NAR",
        n_terms=3,
        order_selection=False,
        estimator=LeastSquares(),
        basis_function=basis_function,
    ).fit(X=None, y=y_train)


def _fit_aols_non_polynomial_nar(y_train, basis_function):
    return AOLS(
        ylag=2,
        xlag=2,
        model_type="NAR",
        k=3,
        L=1,
        estimator=LeastSquares(),
        basis_function=basis_function,
    ).fit(X=None, y=y_train)


def _fit_rmss_non_polynomial_nar(y_train, basis_function):
    return RMSS(
        ylag=2,
        xlag=2,
        model_type="NAR",
        n_terms=3,
        order_selection=False,
        estimator=LeastSquares(),
        basis_function=basis_function,
    ).fit(X=None, y=y_train)


def _fit_er_non_polynomial_nar(y_train, basis_function):
    return ER(
        ylag=2,
        xlag=2,
        model_type="NAR",
        n_perm=5,
        random_state=193,
        estimator=LeastSquares(),
        basis_function=basis_function,
    ).fit(X=None, y=y_train)


def _fit_exact_ar1_nar(include_bias):
    offset = 0.25 if include_bias else 0.0
    y = np.empty((40, 1))
    y[0, 0] = 3.0
    for k in range(1, len(y)):
        y[k, 0] = 0.5 * y[k - 1, 0] + offset

    return FROLS(
        ylag=1,
        xlag=1,
        model_type="NAR",
        n_terms=1 + int(include_bias),
        order_selection=False,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=1, include_bias=include_bias),
    ).fit(X=None, y=y)


def _segmented_nar_free_run_reference(model, y, steps_ahead):
    reference = np.full(y.shape, np.nan, dtype=np.result_type(y.dtype, float))
    reference[: model.max_lag] = y[: model.max_lag]

    for block_start in range(model.max_lag, len(y), steps_ahead):
        block_horizon = min(steps_ahead, len(y) - block_start)
        initial_condition = y[block_start - model.max_lag : block_start]
        free_run = model.predict(
            X=None,
            y=initial_condition,
            steps_ahead=None,
            forecast_horizon=block_horizon,
        )
        reference[block_start : block_start + block_horizon] = free_run[-block_horizon:]

    return reference


def _simulate_metamss(model, **kwargs):
    model.theta = np.ones((len(kwargs["model_code"]), 1))
    model.pivv = np.arange(len(kwargs["model_code"]), dtype=np.intp)
    return np.copy(kwargs["y_test"])


def _metamss_population(model):
    population = np.zeros((model.dimension, model.n_agents), dtype=int)
    population[[0, -1], 0] = 1
    population[:, 1] = 1
    return population


def _record_residual_features(matrices, fit, *args, **kwargs):
    features = fit(*args, **kwargs)
    if kwargs.get("model_type") == "NAR":
        matrices.append(features)
    return features


def _univariate_reference_expansion(family, values):
    """Return degree-one and degree-two columns from their definitions."""
    if family == "bernstein":
        return 2 * values * (1 - values), values**2
    if family == "legendre":
        return values, (3 * values**2 - 1) / 2
    if family == "hermite":
        return 2 * values, 4 * values**2 - 2
    if family == "hermite_normalized":
        return values, values**2 - 1
    if family == "laguerre":
        return 1 - values, 1 - 2 * values + values**2 / 2
    raise ValueError(f"Unknown univariate basis family: {family}")


@pytest.mark.parametrize("basis_cls", [Polynomial, Bilinear])
@pytest.mark.parametrize("include_bias", [True, False])
def test_polynomial_family_native_width_matches_codes(basis_cls, include_bias):
    regressor_matrix, regressor_code = _native_matrix_and_codes(
        basis_cls(degree=2, include_bias=include_bias)
    )

    has_constant_code = bool(np.any(np.all(regressor_code == 0, axis=1)))
    assert regressor_matrix.shape[1] == regressor_code.shape[0]
    assert has_constant_code is include_bias


@pytest.mark.parametrize(
    "basis_cls",
    [Bernstein, Legendre, Hermite, HermiteNormalized, Laguerre],
)
@pytest.mark.parametrize("include_bias", [True, False])
@pytest.mark.parametrize("ensemble", [True, False])
def test_univariate_native_width_matches_codes_in_every_bias_and_ensemble_mode(
    basis_cls,
    include_bias,
    ensemble,
):
    basis_function = basis_cls(
        degree=2,
        include_bias=include_bias,
        ensemble=ensemble,
    )
    regressor_matrix, regressor_code = _native_matrix_and_codes(basis_function)

    has_constant_code = bool(np.any(np.all(regressor_code == 0, axis=1)))
    assert regressor_matrix.shape[1] == regressor_code.shape[0]
    assert has_constant_code is include_bias


@pytest.mark.parametrize(
    ("basis_cls", "family"),
    [
        pytest.param(Bernstein, "bernstein", id="bernstein"),
        pytest.param(Legendre, "legendre", id="legendre"),
        pytest.param(Hermite, "hermite", id="hermite"),
        pytest.param(
            HermiteNormalized,
            "hermite_normalized",
            id="hermite-normalized",
        ),
        pytest.param(Laguerre, "laguerre", id="laguerre"),
    ],
)
def test_univariate_columns_and_codes_follow_the_same_explicit_order(
    basis_cls,
    family,
):
    data = np.array(
        [
            [1.0, 0.1, 0.2],
            [1.0, 0.3, 0.4],
            [1.0, 0.5, 0.6],
            [1.0, 0.7, 0.8],
        ]
    )
    basis_function = basis_cls(degree=2, include_bias=False, ensemble=False)
    regressor_matrix = basis_function.fit(data, max_lag=1, ylag=1, xlag=1)
    regressor_code = RegressorDictionary(
        xlag=1,
        ylag=1,
        basis_function=basis_function,
    ).regressor_space(n_inputs=1, n_features=regressor_matrix.shape[1])

    y_degree_one, y_degree_two = _univariate_reference_expansion(
        family,
        data[1:, 1],
    )
    x_degree_one, x_degree_two = _univariate_reference_expansion(
        family,
        data[1:, 2],
    )
    expected_columns = [
        (np.array([1001, 0]), y_degree_one),
        (np.array([1001, 1001]), y_degree_two),
        (np.array([2001, 0]), x_degree_one),
        (np.array([2001, 2001]), x_degree_two),
    ]

    for column, (expected_code, expected_values) in enumerate(expected_columns):
        assert_array_equal(regressor_code[column], expected_code)
        assert_allclose(
            regressor_matrix[:, column],
            expected_values,
            rtol=0,
            atol=1e-15,
        )


@pytest.mark.parametrize("ensemble", [True, False])
def test_fourier_width_matches_codes_without_adding_a_bias_option(ensemble):
    basis_function = Fourier(degree=2, n=2, ensemble=ensemble)
    regressor_matrix, regressor_code = _native_matrix_and_codes(basis_function)

    assert "include_bias" not in inspect.signature(Fourier).parameters
    assert regressor_matrix.shape[1] == regressor_code.shape[0]
    assert not np.any(np.all(regressor_code == 0, axis=1))
    assert not np.any(np.all(regressor_matrix == 1.0, axis=0))


def test_fourier_rejects_include_bias_to_keep_its_public_api_unchanged():
    with pytest.raises(TypeError):
        Fourier(include_bias=False)


def test_legendre_frols_keeps_selected_codes_and_parameters_aligned():
    x_train, x_valid, y_train, y_valid = _generate_siso_data()
    model = FROLS(
        ylag=2,
        xlag=2,
        n_terms=3,
        order_selection=False,
        estimator=LeastSquares(),
        basis_function=Legendre(degree=2, include_bias=False),
    ).fit(X=x_train, y=y_train)

    selected = model.pivv[: model.n_terms]
    prediction = model.predict(X=x_valid, y=y_valid, steps_ahead=1)

    assert_array_equal(model.final_model, model.regressor_code[selected])
    assert model.theta.shape == (model.final_model.shape[0], 1)
    assert not np.any(np.all(model.regressor_code == 0, axis=1))
    assert prediction.shape == y_valid.shape
    assert np.all(np.isfinite(prediction))


def test_bilinear_frols_without_bias_fits_and_predicts_recursively():
    x_train, x_valid, y_train, y_valid = _generate_siso_data()
    model = FROLS(
        ylag=2,
        xlag=2,
        n_terms=3,
        order_selection=False,
        estimator=LeastSquares(),
        basis_function=Bilinear(degree=2, include_bias=False),
    ).fit(X=x_train, y=y_train)

    selected = model.pivv[: model.n_terms]

    assert_array_equal(model.final_model, model.regressor_code[selected])
    assert model.theta.shape == (model.final_model.shape[0], 1)
    assert not np.any(np.all(model.regressor_code == 0, axis=1))
    for steps_ahead in (None, 1, 3):
        prediction = model.predict(
            X=x_valid,
            y=y_valid,
            steps_ahead=steps_ahead,
        )
        assert prediction.shape == y_valid.shape
        assert_allclose(prediction, y_valid, rtol=1e-9, atol=1e-10)


def test_bilinear_frols_default_matches_explicit_bias_true():
    x_train, x_valid, y_train, y_valid = _generate_siso_data()

    default = _fit_frols_model(Bilinear(degree=2), x_train, y_train)
    explicit = _fit_frols_model(Bilinear(degree=2, include_bias=True), x_train, y_train)
    default_prediction = default.predict(X=x_valid, y=y_valid, steps_ahead=1)
    explicit_prediction = explicit.predict(X=x_valid, y=y_valid, steps_ahead=1)

    assert_array_equal(default.regressor_code, explicit.regressor_code)
    assert_array_equal(default.pivv, explicit.pivv)
    assert_array_equal(default.final_model, explicit.final_model)
    assert_allclose(default.theta, explicit.theta, rtol=0, atol=0)
    assert_allclose(default_prediction, explicit_prediction, rtol=0, atol=0)


def test_polynomial_without_bias_aligns_a_miso_model_and_one_step_prediction():
    rng = np.random.default_rng(193)
    x = rng.uniform(-0.9, 0.9, size=(200, 2))
    y = np.zeros((200, 1))
    for k in range(2, y.shape[0]):
        y[k, 0] = (
            0.25 * y[k - 1, 0]
            + 0.8 * x[k - 1, 0]
            - 0.4 * x[k - 2, 1]
            + 0.15 * x[k - 1, 0] * x[k - 1, 1]
        )

    split = 150
    model = FROLS(
        ylag=2,
        xlag=[[1, 2], [1, 2]],
        n_terms=4,
        order_selection=False,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=2, include_bias=False),
    ).fit(X=x[:split], y=y[:split])
    prediction = model.predict(X=x[split:], y=y[split:], steps_ahead=1)
    selected = model.pivv[: model.n_terms]

    assert_array_equal(model.final_model, model.regressor_code[selected])
    assert model.theta.shape == (model.final_model.shape[0], 1)
    assert not np.any(np.all(model.regressor_code == 0, axis=1))
    assert_allclose(prediction, y[split:], rtol=1e-10, atol=1e-12)


def test_bilinear_without_bias_accepts_array_api_inputs():
    xp = pytest.importorskip("array_api_strict")
    data = np.array(
        [
            [1.0, 0.1, -0.4],
            [1.0, 0.3, 0.8],
            [1.0, -0.6, 0.5],
            [1.0, 0.9, -0.2],
        ]
    )
    basis_function = Bilinear(degree=2, include_bias=False)
    expected = basis_function.fit(data, max_lag=1, ylag=1, xlag=1)

    with config_context(array_api_dispatch=True):
        result = basis_function.fit(
            xp.asarray(data),
            max_lag=1,
            ylag=1,
            xlag=1,
        )

    assert result.__array_namespace__().__name__ == xp.__name__
    xp_assert_allclose(result, expected, rtol=0, atol=1.5e-12)


def test_general_narx_without_bias_uses_external_intercept_setting():
    x_train, x_valid, y_train, y_valid = _generate_siso_data()
    estimator = LinearRegression(fit_intercept=False)
    model = NARX(
        ylag=2,
        xlag=2,
        base_estimator=estimator,
        basis_function=Polynomial(degree=2, include_bias=False),
    ).fit(X=x_train, y=y_train)

    prediction = model.predict(X=x_valid, y=y_valid, steps_ahead=1)

    assert estimator.fit_intercept is False
    assert estimator.intercept_ == 0.0
    assert estimator.coef_.shape[0] == model.regressor_code.shape[0]
    assert_array_equal(model.final_model, model.regressor_code)
    assert not np.any(np.all(model.regressor_code == 0, axis=1))
    assert_allclose(prediction, y_valid, rtol=1e-9, atol=1e-10)


def test_aols_without_bias_keeps_selection_and_parameters_aligned():
    x_train, x_valid, y_train, y_valid = _generate_siso_data()
    model = AOLS(
        ylag=2,
        xlag=2,
        k=3,
        L=1,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=2, include_bias=False),
    ).fit(X=x_train, y=y_train)

    prediction = model.predict(X=x_valid, y=y_valid, steps_ahead=1)

    assert_array_equal(model.final_model, model.regressor_code[model.pivv])
    assert model.theta.shape == (model.final_model.shape[0], 1)
    assert not np.any(np.all(model.regressor_code == 0, axis=1))
    assert prediction.shape == y_valid.shape
    assert np.all(np.isfinite(prediction))


@pytest.mark.parametrize("model_cls", [UOFR, OSF])
def test_ofr_variants_without_bias_keep_selection_aligned(model_cls):
    x_train, x_valid, y_train, y_valid = _generate_siso_data()
    model = model_cls(
        ylag=2,
        xlag=2,
        n_terms=3,
        order_selection=False,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=2, include_bias=False),
    ).fit(X=x_train, y=y_train)

    prediction = model.predict(X=x_valid, y=y_valid, steps_ahead=1)
    selected = model.pivv[: model.n_terms]

    assert_array_equal(model.final_model, model.regressor_code[selected])
    assert model.theta.shape == (model.final_model.shape[0], 1)
    assert not np.any(np.all(model.regressor_code == 0, axis=1))
    assert prediction.shape == y_valid.shape
    assert np.all(np.isfinite(prediction))


@pytest.mark.parametrize(
    ("model_type", "uses_input"),
    [("NAR", False), ("NFIR", True)],
)
def test_frols_without_bias_aligns_nar_and_nfir_models(model_type, uses_input):
    x_train, x_valid, y_train, y_valid = _generate_siso_data()
    model = FROLS(
        ylag=2,
        xlag=2,
        model_type=model_type,
        n_terms=2,
        order_selection=False,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=2, include_bias=False),
    ).fit(X=x_train if uses_input else None, y=y_train)

    prediction = model.predict(
        X=x_valid if uses_input else None,
        y=y_valid,
        steps_ahead=1,
    )
    selected = model.pivv[: model.n_terms]

    assert_array_equal(model.final_model, model.regressor_code[selected])
    assert model.theta.shape == (model.final_model.shape[0], 1)
    assert not np.any(np.all(model.regressor_code == 0, axis=1))
    assert prediction.shape == y_valid.shape
    assert np.all(np.isfinite(prediction))


@pytest.mark.parametrize(
    "steps_ahead",
    [
        pytest.param(2, id="two-steps"),
        pytest.param(3, id="three-steps"),
        pytest.param(24, id="longer-than-remaining-horizon"),
    ],
)
@pytest.mark.parametrize("include_bias", [True, False])
def test_frols_nar_n_step_matches_segmented_free_runs(
    include_bias,
    steps_ahead,
):
    y_train, y_valid = _generate_nar_data()
    model = _fit_frols_nar(y_train, include_bias)
    expected = _segmented_nar_free_run_reference(
        model,
        y_valid,
        steps_ahead,
    )

    prediction = model.predict(
        X=None,
        y=y_valid,
        steps_ahead=steps_ahead,
    )

    assert prediction.shape == y_valid.shape
    assert np.all(np.isfinite(prediction))
    assert_array_equal(prediction[: model.max_lag], y_valid[: model.max_lag])
    assert_allclose(prediction, expected, rtol=1e-12, atol=1e-12)
    assert_allclose(prediction[-1], expected[-1], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("include_bias", [True, False])
def test_frols_nar_one_step_and_free_run_regressions(include_bias):
    y_train, y_valid = _generate_nar_data()
    model = _fit_frols_nar(y_train, include_bias)
    remaining_horizon = int(len(y_valid) - model.max_lag)

    one_step = model.predict(X=None, y=y_valid, steps_ahead=1)
    one_step_reference = _segmented_nar_free_run_reference(model, y_valid, 1)
    free_run = model.predict(
        X=None,
        y=y_valid[: model.max_lag],
        forecast_horizon=remaining_horizon,
    )
    free_run_reference = _segmented_nar_free_run_reference(
        model,
        y_valid,
        remaining_horizon + 1,
    )

    assert one_step.shape == y_valid.shape
    assert free_run.shape == y_valid.shape
    assert np.all(np.isfinite(one_step))
    assert np.all(np.isfinite(free_run))
    assert_array_equal(one_step[: model.max_lag], y_valid[: model.max_lag])
    assert_array_equal(free_run[: model.max_lag], y_valid[: model.max_lag])
    assert_allclose(one_step, one_step_reference, rtol=1e-12, atol=1e-12)
    assert_allclose(free_run, free_run_reference, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("include_bias", [True, False])
def test_frols_nar_recursive_predictions_promote_integer_output(include_bias):
    model = _fit_exact_ar1_nar(include_bias)
    y_integer = np.array([[3], [100], [100], [100], [100]])

    if include_bias:
        expected_one_step = np.array([[3], [1.75], [50.25], [50.25], [50.25]])
        expected_n_step = np.array([[3], [1.75], [1.125], [50.25], [25.375]])
        expected_free_run = np.array([[3], [1.75], [1.125], [0.8125], [0.65625]])
    else:
        expected_one_step = np.array([[3], [1.5], [50], [50], [50]])
        expected_n_step = np.array([[3], [1.5], [0.75], [50], [25]])
        expected_free_run = np.array([[3], [1.5], [0.75], [0.375], [0.1875]])

    one_step = model.predict(X=None, y=y_integer, steps_ahead=1)
    n_step = model.predict(X=None, y=y_integer, steps_ahead=2)
    free_run = model.predict(X=None, y=y_integer[:1], forecast_horizon=4)

    assert np.issubdtype(one_step.dtype, np.floating)
    assert np.issubdtype(n_step.dtype, np.floating)
    assert np.issubdtype(free_run.dtype, np.floating)
    assert_allclose(one_step, expected_one_step, rtol=1e-12, atol=1e-12)
    assert_allclose(n_step, expected_n_step, rtol=1e-12, atol=1e-12)
    assert_allclose(free_run, expected_free_run, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "fit_model",
    [
        pytest.param(_fit_aols_nar, id="aols"),
        pytest.param(_fit_rmss_nar, id="rmss"),
    ],
)
@pytest.mark.parametrize("include_bias", [True, False])
def test_base_mss_nar_consumers_match_segmented_free_runs(
    fit_model,
    include_bias,
):
    y_train, y_valid = _generate_nar_data()
    model = fit_model(y_train, include_bias)
    expected = _segmented_nar_free_run_reference(model, y_valid, 3)

    prediction = model.predict(X=None, y=y_valid, steps_ahead=3)

    assert prediction.shape == y_valid.shape
    assert np.all(np.isfinite(prediction))
    assert_array_equal(prediction[: model.max_lag], y_valid[: model.max_lag])
    assert_allclose(prediction, expected, rtol=1e-12, atol=1e-12)
    assert_allclose(prediction[-2:], expected[-2:], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("steps_ahead", "forecast_horizon"),
    [
        pytest.param(2, None, id="two-steps-without-horizon"),
        pytest.param(3, 1, id="three-steps-with-short-horizon"),
        pytest.param(3, 100, id="three-steps-with-long-horizon"),
        pytest.param(24, None, id="step-longer-than-remaining-horizon"),
    ],
)
def test_frols_non_polynomial_nar_uses_observed_series_as_n_step_horizon(
    steps_ahead,
    forecast_horizon,
):
    y_train, y_valid = _generate_nar_data()
    model = _fit_frols_non_polynomial_nar(y_train, Fourier(degree=2, n=1))
    expected = _segmented_nar_free_run_reference(model, y_valid, steps_ahead)
    original_y = y_valid.copy()

    prediction = model.predict(
        X=None,
        y=y_valid,
        steps_ahead=steps_ahead,
        forecast_horizon=forecast_horizon,
    )

    assert prediction.shape == y_valid.shape
    assert np.all(np.isfinite(prediction))
    assert_array_equal(y_valid, original_y)
    assert_array_equal(prediction[: model.max_lag], y_valid[: model.max_lag])
    assert_allclose(prediction, expected, rtol=1e-12, atol=1e-12)
    assert_allclose(prediction[-1], expected[-1], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "fit_model",
    [
        pytest.param(_fit_frols_non_polynomial_nar, id="frols"),
        pytest.param(_fit_aols_non_polynomial_nar, id="aols"),
        pytest.param(_fit_rmss_non_polynomial_nar, id="rmss"),
        pytest.param(_fit_er_non_polynomial_nar, id="er"),
    ],
)
def test_base_mss_non_polynomial_nar_consumers_match_segmented_free_runs(
    fit_model,
):
    y_train, y_valid = _generate_nar_data()
    model = fit_model(y_train, Fourier(degree=2, n=1))
    expected = _segmented_nar_free_run_reference(model, y_valid, 3)

    prediction = model.predict(X=None, y=y_valid, steps_ahead=3)

    assert prediction.shape == y_valid.shape
    assert np.all(np.isfinite(prediction))
    assert_array_equal(prediction[: model.max_lag], y_valid[: model.max_lag])
    assert_allclose(prediction, expected, rtol=1e-12, atol=1e-12)
    assert_allclose(prediction[-2:], expected[-2:], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("basis_cls", "basis_kwargs"),
    [
        pytest.param(Fourier, {"n": 1}, id="fourier"),
        pytest.param(Bilinear, {"include_bias": False}, id="bilinear"),
        pytest.param(Bernstein, {"include_bias": False}, id="bernstein"),
        pytest.param(Legendre, {"include_bias": True}, id="legendre-with-bias"),
        pytest.param(Legendre, {"include_bias": False}, id="legendre-without-bias"),
        pytest.param(Laguerre, {"include_bias": False}, id="laguerre"),
        pytest.param(Hermite, {"include_bias": False}, id="hermite"),
        pytest.param(
            HermiteNormalized,
            {"include_bias": False},
            id="hermite-normalized",
        ),
    ],
)
def test_frols_nar_n_step_supports_every_non_polynomial_basis(
    basis_cls,
    basis_kwargs,
):
    y_train, y_valid = _generate_nar_data()
    basis_function = basis_cls(degree=2, **basis_kwargs)
    model = _fit_frols_non_polynomial_nar(y_train, basis_function)
    expected = _segmented_nar_free_run_reference(model, y_valid, 3)

    prediction = model.predict(
        X=None,
        y=y_valid,
        steps_ahead=3,
        forecast_horizon=1,
    )

    assert prediction.shape == y_valid.shape
    assert np.all(np.isfinite(prediction))
    assert_array_equal(prediction[: model.max_lag], y_valid[: model.max_lag])
    assert_allclose(prediction, expected, rtol=1e-12, atol=1e-12)


def test_frols_non_polynomial_nar_n_step_validates_boundaries_and_integer_data():
    y_train, y_valid = _generate_nar_data()
    model = _fit_frols_non_polynomial_nar(y_train, Fourier(degree=2, n=1))

    for invalid_steps in (0, -1, 1.5):
        with pytest.raises(
            ValueError,
            match="steps_ahead must be integer and > zero",
        ):
            model.predict(X=None, y=y_valid, steps_ahead=invalid_steps)

    prefix_only = model.predict(
        X=None,
        y=y_valid[: model.max_lag],
        steps_ahead=3,
        forecast_horizon=100,
    )
    assert_array_equal(prefix_only, y_valid[: model.max_lag])

    with pytest.raises(ValueError, match="Insufficient initial condition"):
        model.predict(
            X=None,
            y=y_valid[: model.max_lag - 1],
            steps_ahead=3,
        )

    y_integer = np.rint(100 * y_valid).astype(np.int64)
    expected = _segmented_nar_free_run_reference(model, y_integer, 2)
    prediction = model.predict(X=None, y=y_integer, steps_ahead=2)

    assert np.issubdtype(prediction.dtype, np.floating)
    assert_allclose(prediction, expected, rtol=1e-12, atol=1e-12)


def test_frols_non_polynomial_nar_n_step_preserves_array_api_namespace():
    xp = pytest.importorskip("array_api_strict")
    y_train, y_valid = _generate_nar_data()
    model = _fit_frols_non_polynomial_nar(y_train, Fourier(degree=2, n=1))
    expected = model.predict(X=None, y=y_valid, steps_ahead=3)

    with config_context(array_api_dispatch=True):
        prediction = model.predict(
            X=None,
            y=xp.asarray(y_valid),
            steps_ahead=3,
            forecast_horizon=100,
        )

    assert prediction.__array_namespace__().__name__ == xp.__name__
    xp_assert_allclose(prediction, expected, rtol=1e-12, atol=1e-12)


def test_rmss_without_bias_keeps_selection_and_parameters_aligned():
    x_train, _, y_train, _ = _generate_siso_data(n_samples=64)
    model = RMSS(
        ylag=2,
        xlag=2,
        n_terms=3,
        order_selection=False,
        average_theta=False,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=2, include_bias=False),
    )

    with pytest.warns(UserWarning, match="average_theta=False"):
        model.fit(X=x_train, y=y_train)

    selected = model.pivv[: model.n_terms]
    assert_array_equal(model.final_model, model.regressor_code[selected])
    assert model.theta.shape == (model.final_model.shape[0], 1)
    assert not np.any(np.all(model.regressor_code == 0, axis=1))
    assert np.all(np.isfinite(model.theta))


def test_metamss_without_bias_uses_reduced_search_space(monkeypatch):
    x_train, _, y_train, _ = _generate_siso_data(n_samples=64)
    model = MetaMSS(
        xlag=1,
        ylag=1,
        maxiter=1,
        n_agents=2,
        random_state=193,
        basis_function=Polynomial(degree=2, include_bias=False),
    )

    monkeypatch.setattr(
        model,
        "generate_random_population",
        partial(_metamss_population, model),
    )
    monkeypatch.setattr(
        model,
        "evaluate_objective_function",
        MagicMock(return_value=np.array([0.0, 1.0])),
    )
    monkeypatch.setattr(model, "simulate", partial(_simulate_metamss, model))

    model.fit(X=x_train, y=y_train)

    with_bias_dimension = (
        RegressorDictionary(
            xlag=1,
            ylag=1,
            basis_function=Polynomial(degree=2, include_bias=True),
        )
        .regressor_space(n_inputs=1)
        .shape[0]
    )
    selected = np.flatnonzero(model.optimal_model)

    assert model.dimension == with_bias_dimension - 1
    assert not np.any(np.all(model.regressor_code == 0, axis=1))
    assert_array_equal(model.final_model, model.regressor_code[selected])
    assert model.theta.shape == (model.final_model.shape[0], 1)


def test_extended_least_squares_without_bias_keeps_selected_model_aligned():
    x_train, x_valid, y_train, y_valid = _generate_siso_data(noise_scale=0.01)
    estimator = LeastSquares(unbiased=True, uiter=2)
    estimator.unbiased_estimator = MagicMock(wraps=estimator.unbiased_estimator)
    basis_function = Polynomial(degree=2, include_bias=False)
    residual_feature_matrices = []
    original_fit = basis_function.fit
    basis_function.fit = MagicMock(
        side_effect=partial(
            _record_residual_features,
            residual_feature_matrices,
            original_fit,
        )
    )
    model = FROLS(
        ylag=2,
        xlag=2,
        elag=2,
        n_terms=3,
        order_selection=False,
        estimator=estimator,
        basis_function=basis_function,
    ).fit(X=x_train, y=y_train)

    assert estimator.unbiased_estimator.call_count == 1
    assert len(residual_feature_matrices) == estimator.uiter
    assert all(matrix.shape[1] == 5 for matrix in residual_feature_matrices)
    assert all(
        not np.any(np.all(matrix == 1.0, axis=0))
        for matrix in residual_feature_matrices
    )

    prediction = model.predict(X=x_valid, y=y_valid, steps_ahead=1)
    selected = model.pivv[: model.n_terms]

    assert estimator.unbiased is True
    assert_array_equal(model.final_model, model.regressor_code[selected])
    assert model.theta.shape == (model.final_model.shape[0], 1)
    assert not np.any(np.all(model.regressor_code == 0, axis=1))
    assert prediction.shape == y_valid.shape
    assert np.all(np.isfinite(model.theta))
    assert np.all(np.isfinite(prediction))
