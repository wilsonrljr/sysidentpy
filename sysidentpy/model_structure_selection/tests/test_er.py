# ruff: noqa: SLF001
# pylint: disable=protected-access,redefined-outer-name
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from numpy.testing import assert_raises
from sysidentpy.model_structure_selection import ER
from sysidentpy.basis_function import Polynomial, Fourier
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


def test_predict_fourier_variants_cover_non_pol_branches():
    basis_function = Fourier(degree=2, n=1)
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
    horizon = 4
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
    assert_equal(
        model.predict(
            X=None, y=y_window, steps_ahead=2, forecast_horizon=horizon
        ).shape,
        (window, 1),
    )


def test_er_basis_function_n_steps_horizon_returns_column_vector():
    basis_function = Fourier(degree=2, n=1)
    model = ER(
        ylag=[1, 2],
        xlag=2,
        estimator=LeastSquares(),
        basis_function=basis_function,
        skip_forward=True,
        n_perm=1,
        random_state=0,
    )
    model.fit(X=X_train[:40], y=y_train[:40])
    window_len = model.max_lag + 3
    steps_ahead = window_len - model.max_lag
    forecast_horizon = window_len
    yhat = model._basis_function_n_steps_horizon(
        x=X_test[:window_len],
        y=y_test[:window_len],
        steps_ahead=steps_ahead,
        forecast_horizon=forecast_horizon,
    )
    assert_equal(tuple(yhat.shape), (forecast_horizon - model.max_lag, 1))


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


def test_large_h_removes_constant_term():
    model = ER(
        ylag=2,
        xlag=2,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=2),
        h=100.0,
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
    horizon = 3
    yhat = model._basis_function_predict(
        x=None, y_initial=y_test, forecast_horizon=horizon
    )
    assert_equal(yhat.shape, (horizon, 1))
    assert_equal(model.n_inputs, 0)
