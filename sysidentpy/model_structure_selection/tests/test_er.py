# pylint: disable=protected-access,redefined-outer-name
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal
from numpy.testing import assert_equal
from numpy.testing import assert_raises

from sysidentpy import config_context
from sysidentpy.basis_function import Fourier, Legendre, Polynomial
from sysidentpy.model_structure_selection import ER
from sysidentpy.parameter_estimation.estimators import LeastSquares
from sysidentpy.tests.test_narmax_base import create_test_data

x, y, _ = create_test_data()
train_percentage = 90
split_data = int(len(x) * (train_percentage / 100))

X_train = x[0:split_data, 0]
X_test = x[split_data::, 0]

y1 = y[0:split_data, 0]
y_test = y[split_data::, 0]
y_train = y1.copy()

y_train = np.reshape(y_train, (len(y_train), 1))
X_train = np.reshape(X_train, (len(X_train), 1))

y_test = np.reshape(y_test, (len(y_test), 1))
X_test = np.reshape(X_test, (len(X_test), 1))


def _segmented_nar_reference(model, y, steps_ahead):
    reference = np.empty_like(y, dtype=float)
    reference[: model.max_lag] = y[: model.max_lag]

    for block_start in range(model.max_lag, len(y), steps_ahead):
        block_horizon = min(steps_ahead, len(y) - block_start)
        initial_conditions = y[block_start - model.max_lag : block_start]
        free_run = model.predict(
            X=None,
            y=initial_conditions,
            forecast_horizon=block_horizon,
        )
        reference[block_start : block_start + block_horizon] = free_run[-block_horizon:]

    return reference


def test_default_values():
    default = {
        "ylag": 1,
        "xlag": 1,
        "q": 0.99,
        "h": 0.01,
        "k": 2,
        "mutual_information_estimator": "mutual_information_knn",
        "n_perm": 200,
        "p": np.inf,
        "skip_forward": False,
        "model_type": "NARMAX",
        "random_state": None,
    }
    model = ER(basis_function=Polynomial(degree=2))
    model_values = [
        model.ylag,
        model.xlag,
        model.q,
        model.h,
        model.k,
        model.mutual_information_estimator,
        model.n_perm,
        model.p,
        model.skip_forward,
        model.model_type,
        model.random_state,
    ]
    assert list(default.values()) == model_values
    assert isinstance(model.estimator, LeastSquares)
    assert isinstance(model.basis_function, Polynomial)


def test_validate_ylag():
    assert_raises(ValueError, ER, ylag=-1, basis_function=Polynomial(degree=2))
    assert_raises(ValueError, ER, ylag=1.3, basis_function=Polynomial(degree=2))


def test_validate_xlag():
    assert_raises(ValueError, ER, xlag=-1, basis_function=Polynomial(degree=2))
    assert_raises(ValueError, ER, xlag=1.3, basis_function=Polynomial(degree=2))


def test_k():
    assert_raises(ValueError, ER, k=-1, basis_function=Polynomial(degree=2))
    assert_raises(ValueError, ER, k=1.3, basis_function=Polynomial(degree=2))


def test_n_perm():
    assert_raises(ValueError, ER, n_perm=-1, basis_function=Polynomial(degree=2))
    assert_raises(ValueError, ER, n_perm=1.3, basis_function=Polynomial(degree=2))


def test_q():
    assert_raises(ValueError, ER, q=-1, basis_function=Polynomial(degree=2))
    assert_raises(ValueError, ER, q=1.3, basis_function=Polynomial(degree=2))


def test_skip_forward():
    assert_raises(TypeError, ER, skip_forward=1, basis_function=Polynomial(degree=2))
    assert_raises(
        TypeError, ER, skip_forward="True", basis_function=Polynomial(degree=2)
    )
    assert_raises(TypeError, ER, skip_forward=None, basis_function=Polynomial(degree=2))


def test_model_type_validation():
    with pytest.raises(ValueError, match="model_type must be NARMAX"):
        ER(model_type="FOO", basis_function=Polynomial(degree=2))


def test_model_prediction():
    model = ER(
        ylag=2,
        xlag=2,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=2),
    )
    model.fit(X=X_train, y=y_train)
    assert_raises(Exception, model.predict, X=X_test, y=y_test[:1])


def test_er_rejects_array_api_dispatch_with_clear_error():
    xp = pytest.importorskip("array_api_strict")
    model = ER(
        ylag=1,
        xlag=1,
        n_perm=1,
        random_state=0,
        basis_function=Polynomial(degree=1),
    )

    with config_context(array_api_dispatch=True):
        with pytest.raises(NotImplementedError, match=r"ER.*requires NumPy"):
            model.fit(X=xp.asarray(X_train[:10]), y=xp.asarray(y_train[:10]))


def test_fit_requires_y_argument():
    model = ER(
        ylag=2,
        xlag=2,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=1),
    )
    with pytest.raises(ValueError, match="y cannot be None"):
        model.fit(X=X_train[:5], y=None)


def test_fit_sets_single_input_when_x_missing():
    model = ER(
        ylag=2,
        xlag=2,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=1),
        model_type="NAR",
        n_perm=1,
    )
    model.fit(y=y_train[:20])
    assert model.n_inputs == 1


def test_fit_emits_warning_for_large_regressor_space():
    model = ER(
        ylag=3,
        xlag=3,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=4),
        n_perm=1,
        random_state=0,
    )
    with pytest.warns(UserWarning, match="higher number of possible regressors"):
        model.fit(X=X_train[:60], y=y_train[:60])


def test_mutual_information_knn():
    model = ER(
        ylag=2,
        xlag=2,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=1),
    )
    x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    y = np.array([0.3, 0.87, 0, 0.1, 0.9]).reshape(-1, 1)

    r = model.mutual_information_knn(x, y)
    assert_almost_equal(r, 0.6000, decimal=3)


def test_mutual_information_knn_argpartition_order(monkeypatch):
    model = ER(
        ylag=2,
        xlag=2,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=1),
    )
    signal = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)

    expected = model.mutual_information_knn(signal, signal)

    original_argpartition = np.argpartition
    head = model.k + 1

    def shuffled_argpartition(array, kth, axis=-1, kind="introselect", order=None):
        result = original_argpartition(array, kth, axis=axis, kind=kind, order=order)
        if axis != -1 or not np.isscalar(kth) or kth < head:
            return result
        leading = result[..., :head]
        trailing = result[..., head:]
        leading = leading[..., ::-1]
        return np.concatenate([leading, trailing], axis=axis)

    monkeypatch.setattr(np, "argpartition", shuffled_argpartition)

    assert_almost_equal(model.mutual_information_knn(signal, signal), expected)


def test_conditional_mutual_information_knn():
    model = ER(
        ylag=2,
        xlag=2,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=1),
    )
    a = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    b = np.array([0.3, 0.87, 0, 0.1, 0.9]).reshape(-1, 1)
    c = np.array([90, 12, 212, 13, 15]).reshape(-1, 1)

    r = model.conditional_mutual_information(a, b, c)
    assert_almost_equal(r, 0.2, decimal=3)


def test_tolerance_estimator(monkeypatch):
    basis_function = Polynomial(degree=1)
    model = ER(
        ylag=2,
        xlag=2,
        estimator=LeastSquares(),
        basis_function=basis_function,
        random_state=42,
        q=0.8,
        n_perm=5,
    )
    a = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)

    samples = iter([0.1, 0.2, 0.3, 0.4, 0.4])

    def fake_mutual_information(_, __):
        return next(samples)

    monkeypatch.setattr(model, "mutual_information_knn", fake_mutual_information)
    r = model.tolerance_estimator(a)
    assert_almost_equal(r, 0.4, decimal=4)


def test_entropic_forward_flags_unsuccessful_when_too_many_terms():
    class DeterministicER(ER):
        def tolerance_estimator(self, _y):
            return -1.0

        def mutual_information_knn(self, *_args, **_kwargs):
            return 1.0

        def conditional_mutual_information(self, *_args, **_kwargs):
            return 1.0

    model = DeterministicER(ylag=1, xlag=1, basis_function=Polynomial(degree=1))
    rng = np.random.default_rng(0)
    reg_matrix = rng.standard_normal((12, 10))
    y = rng.standard_normal((12, 1))

    selected_terms, success = model.entropic_regression_forward(reg_matrix, y)

    assert success is False
    assert len(selected_terms) == 9


def test_entropic_backward_keeps_all_terms_above_tolerance(monkeypatch):
    model = ER(basis_function=Polynomial(degree=1))
    model.tol = 0.5
    monkeypatch.setattr(
        model, "conditional_mutual_information", lambda *_args, **_kwargs: 1.0
    )
    reg_matrix = np.eye(3)
    y = np.ones((3, 1))

    selected = model.entropic_regression_backward(reg_matrix, y, [0, 1, 2])

    assert_equal(selected, np.array([0, 1, 2]))


def test_entropic_backward_removes_terms_equal_to_tolerance(monkeypatch):
    model = ER(basis_function=Polynomial(degree=1))
    model.tol = 0.5
    monkeypatch.setattr(
        model, "conditional_mutual_information", lambda *_args, **_kwargs: 0.5
    )
    reg_matrix = np.eye(3)
    y = np.ones((3, 1))

    selected = model.entropic_regression_backward(reg_matrix, y, [0, 1, 2])

    assert len(selected) == 1


def test_entropic_backward_removes_then_stops_above_tolerance(monkeypatch):
    model = ER(basis_function=Polynomial(degree=1))
    model.tol = 0.5
    mutual_information = iter([0.1, 0.4, 0.6, 0.7, 0.8])
    monkeypatch.setattr(
        model,
        "conditional_mutual_information",
        lambda *_args, **_kwargs: next(mutual_information),
    )
    reg_matrix = np.eye(3)
    y = np.ones((3, 1))

    selected = model.entropic_regression_backward(reg_matrix, y, [0, 1, 2])

    assert_equal(selected, np.array([1, 2]))


def test_fit_estimates_and_uses_one_tolerance(monkeypatch):
    model = ER(
        ylag=1,
        xlag=1,
        n_perm=1,
        random_state=0,
        basis_function=Polynomial(degree=1),
    )
    calls = 0

    def fake_tolerance(_y):
        nonlocal calls
        calls += 1
        return 100.0

    monkeypatch.setattr(model, "tolerance_estimator", fake_tolerance)
    monkeypatch.setattr(model, "mutual_information_knn", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(
        model, "conditional_mutual_information", lambda *_args, **_kwargs: 0.0
    )

    model.fit(X=X_train[:10], y=y_train[:10])

    assert calls == 1
    assert model.tol == model.estimated_tolerance == 100.0


def test_er_recovers_known_polynomial_structure_end_to_end():
    x_data, y_data, _ = create_test_data()
    model = ER(
        ylag=2,
        xlag=2,
        n_perm=20,
        random_state=0,
        basis_function=Polynomial(degree=2),
    )

    model.fit(X=x_data, y=y_data)

    expected_codes = np.array(
        [
            [2002, 0],
            [1002, 0],
            [2001, 1001],
            [2002, 1002],
            [1001, 1001],
        ]
    )
    expected_theta = np.array([0.6, -0.5, 0.7, -0.7, 0.2])
    assert_array_equal(model.final_model, expected_codes)
    assert_allclose(model.theta.ravel(), expected_theta, atol=1e-8)


def test_predict_polynomial_variants_cover_all_branches():
    model = ER(
        ylag=2,
        xlag=2,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=1),
        skip_forward=True,
        n_perm=1,
        random_state=0,
    )
    model.fit(X=X_train[:30], y=y_train[:30])
    window = model.max_lag + 8
    x_window = X_test[:window]
    y_window = y_test[:window]

    assert_equal(model.predict(X=x_window, y=y_window).shape, (window, 1))
    assert_equal(
        model.predict(X=x_window, y=y_window, steps_ahead=1).shape,
        (window, 1),
    )
    assert_equal(
        model.predict(X=x_window, y=y_window, steps_ahead=3).shape,
        (window, 1),
    )


def test_predict_polynomial_without_inputs_uses_forecast_horizon():
    model = ER(
        ylag=[1, 2],
        xlag=1,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=1),
        model_type="NAR",
        skip_forward=True,
        n_perm=1,
        random_state=0,
    )
    model.fit(y=y_train[:40])
    horizon = 4
    initial_conditions = y_test[: model.max_lag]
    yhat = model.predict(X=None, y=initial_conditions, forecast_horizon=horizon)
    assert_equal(yhat.shape, (model.max_lag + horizon, 1))


@pytest.mark.parametrize(
    "basis_function",
    [
        pytest.param(Fourier(degree=2, n=1), id="fourier"),
        pytest.param(
            Legendre(degree=2, include_bias=True),
            id="legendre-with-bias",
        ),
        pytest.param(
            Legendre(degree=2, include_bias=False),
            id="legendre-without-bias",
        ),
    ],
)
def test_predict_non_polynomial_variants_cover_nar_branches(basis_function):
    model = ER(
        ylag=[1, 2],
        xlag=2,
        estimator=LeastSquares(),
        basis_function=basis_function,
        model_type="NAR",
        skip_forward=True,
        n_perm=1,
        random_state=0,
    )
    model.fit(y=y_train[:40])
    horizon = 5
    window = model.max_lag + horizon
    y_window = y_test[:window]

    assert_equal(
        model.predict(X=None, y=y_window, forecast_horizon=horizon).shape,
        (window, 1),
    )
    assert_equal(
        model.predict(X=None, y=y_window, steps_ahead=1).shape,
        (window, 1),
    )

    n_step = model.predict(X=None, y=y_window, steps_ahead=3)
    with_ignored_horizon = model.predict(
        X=None,
        y=y_window,
        steps_ahead=3,
        forecast_horizon=1,
    )
    expected = _segmented_nar_reference(model, y_window, 3)

    assert_equal(n_step.shape, y_window.shape)
    np.testing.assert_array_equal(
        n_step[: model.max_lag],
        y_window[: model.max_lag],
    )
    np.testing.assert_allclose(n_step, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        with_ignored_horizon,
        expected,
        rtol=1e-10,
        atol=1e-12,
    )


def test_non_polynomial_predict_rejects_non_positive_steps():
    model = ER(
        ylag=[1, 2],
        basis_function=Fourier(degree=1),
        model_type="NAR",
        estimator=LeastSquares(),
        skip_forward=True,
        n_perm=1,
        random_state=0,
    ).fit(y=y_train[:40])

    with pytest.raises(ValueError, match="steps_ahead must be"):
        model.predict(X=None, y=y_test[:5], steps_ahead=0)


def test_nfir_prediction_modes_are_equivalent():
    model = ER(
        ylag=[1, 2],
        xlag=2,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=2),
        model_type="NFIR",
        skip_forward=True,
        n_perm=1,
        random_state=0,
    )
    model.fit(X=X_train[:40], y=y_train[:40])
    x_window = X_test[:10]
    y_window = y_test[:10]
    free_run = model.predict(X=x_window, y=y_window)
    one_step = model.predict(X=x_window, y=y_window, steps_ahead=1)
    n_step = model.predict(X=x_window, y=y_window, steps_ahead=3)

    np.testing.assert_allclose(one_step, free_run, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(n_step, free_run, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(free_run[: model.max_lag], y_window[: model.max_lag])


def test_predict_nfir_model_uses_specific_branch():
    model = ER(
        ylag=1,
        xlag=2,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=1),
        model_type="NFIR",
        skip_forward=True,
        n_perm=1,
        random_state=0,
    )
    model.fit(X=X_train[:30], y=y_train[:30])
    window = model.max_lag + 6
    x_window = X_test[:window]
    y_window = y_test[:window]
    assert_equal(model.predict(X=x_window, y=y_window).shape, (window, 1))


def test_fit_skip_forward_skips_forward_stage(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("forward stage should be skipped")

    model = ER(
        ylag=2,
        xlag=2,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=1),
        skip_forward=True,
        n_perm=1,
        q=0.5,
        random_state=0,
    )
    monkeypatch.setattr(model, "entropic_regression_forward", forbidden)
    monkeypatch.setattr(
        model,
        "entropic_regression_backward",
        lambda *_args, **_kwargs: np.array([0]),
    )
    model.fit(X=X_train[:10], y=y_train[:10])
    assert_equal(model.pivv[0], 0)


def test_fit_without_bias_does_not_force_first_regressor(monkeypatch):
    model = ER(
        ylag=2,
        xlag=2,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=1, include_bias=False),
        skip_forward=True,
        n_perm=1,
        random_state=0,
    )
    monkeypatch.setattr(
        model,
        "entropic_regression_backward",
        lambda *_args, **_kwargs: np.array([1]),
    )

    model.fit(X=X_train[:10], y=y_train[:10])

    assert_array_equal(model.pivv, np.array([1]))
    assert_array_equal(model.final_model, model.regressor_code[[1]])
    assert not np.any(np.all(model.final_model == 0, axis=1))


def test_fit_locates_nonpolynomial_bias_outside_first_column(monkeypatch):
    model = ER(
        ylag=2,
        xlag=2,
        estimator=LeastSquares(),
        basis_function=Legendre(degree=2, include_bias=True, ensemble=True),
        skip_forward=True,
        n_perm=1,
        h=0.0,
        random_state=0,
    )
    monkeypatch.setattr(
        model,
        "entropic_regression_backward",
        lambda *_args, **_kwargs: np.array([0]),
    )

    model.fit(X=X_train[:10], y=y_train[:10])

    bias_index = np.flatnonzero(np.all(model.regressor_code == 0, axis=1))[0]
    assert bias_index > 0
    assert_array_equal(model.pivv, np.array([bias_index, 0]))
    assert_array_equal(model.final_model, model.regressor_code[model.pivv])
    assert model.theta.shape == (2, 1)


def test_large_h_removes_constant_term():
    model = ER(
        ylag=2,
        xlag=2,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=2),
        h=100.0,
        random_state=0,
    )
    model.fit(X=X_train[:50], y=y_train[:50])
    assert 0 not in model.pivv


def test_model_prediction_rejects_unknown_type():
    model = ER(
        ylag=2,
        xlag=2,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=2),
    )
    model.fit(X=X_train, y=y_train)
    model.model_type = "UNKNOWN"
    assert_raises(ValueError, model._model_prediction, X_test, y_test)


def test_basis_function_predict_handles_missing_input():
    basis_function = Fourier(degree=2, n=1)
    model = ER(
        ylag=[1, 2],
        xlag=2,
        estimator=LeastSquares(),
        basis_function=basis_function,
        model_type="NAR",
    )
    model.fit(X=X_train, y=y_train)
    fitted_n_inputs = model.n_inputs
    horizon = 3
    yhat = model._basis_function_predict(
        x=None, y_initial=y_test, forecast_horizon=horizon
    )
    assert_equal(yhat.shape, (horizon, 1))
    assert_equal(model.n_inputs, fitted_n_inputs)
