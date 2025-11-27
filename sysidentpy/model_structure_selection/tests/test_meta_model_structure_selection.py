import numpy as np
import pytest

from sysidentpy.model_structure_selection import (
    meta_model_structure_selection as meta_module,
)
from sysidentpy.model_structure_selection.meta_model_structure_selection import MetaMSS
from sysidentpy.simulation import SimulateNARMAX


def test_meta_mss_fit_requires_y():
    model = MetaMSS()
    with pytest.raises(ValueError, match="y cannot be None"):
        model.fit(y=None)


def test_meta_mss_fit_sets_default_input_count(monkeypatch):
    model = MetaMSS(maxiter=1, n_agents=1, random_state=0)
    captured = {}

    def fake_split(X, y, test_size):
        captured["n_inputs"] = model.n_inputs
        raise RuntimeError("stop early")

    monkeypatch.setattr(meta_module, "train_test_split", fake_split)

    y = np.arange(8, dtype=float).reshape(-1, 1)
    with pytest.raises(RuntimeError, match="stop early"):
        model.fit(X=None, y=y)

    assert captured["n_inputs"] == 1


def test_evaluate_objective_function_penalizes_empty_agent():
    model = MetaMSS()
    model.regressor_code = np.ones((1, 1))
    population = np.zeros((1, 1), dtype=int)
    x_train = np.zeros((2, 1))
    y_train = np.zeros((2, 1))
    x_test = np.zeros((1, 1))
    y_test = np.zeros((1, 1))

    fitness = model.evaluate_objective_function(
        x_train, y_train, x_test, y_test, population
    )

    assert fitness == [30]


def test_evaluate_objective_function_runs_loss_and_print(monkeypatch, capsys):
    model = MetaMSS()
    model.theta = []
    model.pivv = None
    model.dimension = 2
    model.regressor_code = np.array([[1, 0], [0, 1]])
    model.steps_ahead = None
    model.tested_models = []

    monkeypatch.setattr(MetaMSS, "_get_max_lag", lambda self: 1)

    def fake_simulate(self, **kwargs):
        return np.copy(kwargs["y_test"])

    monkeypatch.setattr(MetaMSS, "simulate", fake_simulate)

    class DummyBasis:
        def fit(self, *args, **kwargs):
            return np.ones((2, 2))

    model.basis_function = DummyBasis()

    def fake_t_test(self, psi, theta, residues):
        return np.array([], dtype=int), np.zeros((1, 1)), np.zeros((1, 1))

    monkeypatch.setattr(MetaMSS, "perform_t_test", fake_t_test)

    x_train = np.ones((4, 1))
    y_train = np.ones((4, 1))
    x_test = np.ones((2, 1))
    y_test = np.ones((2, 1))
    population = np.ones((2, 1), dtype=int)

    fitness = model.evaluate_objective_function(
        x_train, y_train, x_test, y_test, population
    )
    printed = capsys.readouterr().out.strip()

    assert fitness == [30]
    assert printed != ""
    assert len(model.tested_models) == 1


def test_perform_t_test_handles_nan_variance():
    model = MetaMSS()
    psi = np.ones((2, 2))
    theta = np.ones((2, 1))
    residues = np.zeros((2, 1))

    pos, t_test, tail2p = model.perform_t_test(psi, theta, residues)

    assert isinstance(pos, np.ndarray)
    assert np.all(np.isfinite(t_test))
    assert tail2p.shape == t_test.shape


def test_aic_computation():
    model = MetaMSS()
    y = np.array([[1.0], [2.0], [3.0]])
    yhat = np.array([[1.1], [1.9], [3.1]])

    result = model.aic(y, yhat, 2)
    expected = y.shape[0] * np.log(np.mean((y - yhat) ** 2)) + 2 * 2

    assert pytest.approx(result) == expected


def test_bic_computation():
    model = MetaMSS()
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    yhat = np.array([[0.9], [2.1], [3.0], [4.2]])

    result = model.bic(y, yhat, 3)
    mse = np.mean((y - yhat) ** 2)
    expected = y.shape[0] * np.log(mse) + 3 + np.log(y.shape[0])

    assert pytest.approx(result) == expected


def test_metamss_loss_returns_fallback_for_nan():
    model = MetaMSS()
    model.dimension = 3
    y_test = np.ones((3, 1))
    yhat = np.ones((3, 1))

    fitness = model.metamss_loss(y_test, yhat, n_terms=1)

    assert fitness == 30


def test_predict_free_run_uses_model_prediction(monkeypatch):
    model = MetaMSS()
    model.max_lag = 1
    calls = {}

    def fake_model_prediction(self, X, y, forecast_horizon=None):
        calls["args"] = (X, y, forecast_horizon)
        return np.array([[10.0]])

    monkeypatch.setattr(MetaMSS, "_model_prediction", fake_model_prediction)

    X = np.arange(4, dtype=float).reshape(-1, 1)
    y = np.arange(4, dtype=float).reshape(-1, 1)
    result = model.predict(X=X, y=y, forecast_horizon=2)

    assert calls["args"][2] == 2
    assert np.array_equal(result[: model.max_lag], y[: model.max_lag])
    assert np.array_equal(result[model.max_lag :], np.array([[10.0]]))


def test_predict_one_step_branch(monkeypatch):
    model = MetaMSS()
    model.max_lag = 1

    def fake_one_step(self, X, y):
        return np.array([[5.0], [6.0]])

    monkeypatch.setattr(MetaMSS, "_one_step_ahead_prediction", fake_one_step)

    X = np.arange(4, dtype=float).reshape(-1, 1)
    y = np.arange(4, dtype=float).reshape(-1, 1)
    result = model.predict(X=X, y=y, steps_ahead=1)

    assert np.array_equal(result[: model.max_lag], y[: model.max_lag])
    assert np.array_equal(result[model.max_lag :], np.array([[5.0], [6.0]]))


def test_predict_n_step_branch(monkeypatch):
    model = MetaMSS()
    model.max_lag = 2

    def fake_n_step(self, X, y, steps_ahead=None):
        return np.array([[7.0]])

    monkeypatch.setattr(MetaMSS, "_n_step_ahead_prediction", fake_n_step)

    X = np.arange(6, dtype=float).reshape(-1, 1)
    y = np.arange(6, dtype=float).reshape(-1, 1)
    result = model.predict(X=X, y=y, steps_ahead=2)

    assert np.array_equal(result[: model.max_lag], y[: model.max_lag])
    assert np.array_equal(result[model.max_lag :], np.array([[7.0]]))


def test_one_step_ahead_prediction_returns_column_vector(monkeypatch):
    def fake_super(self, x, y):
        return np.array([1.0, 2.0])

    monkeypatch.setattr(SimulateNARMAX, "_one_step_ahead_prediction", fake_super)

    model = MetaMSS()
    result = model._one_step_ahead_prediction(np.zeros((2, 1)), np.zeros((2, 1)))

    assert result.shape == (2, 1)
    assert np.array_equal(result, np.array([[1.0], [2.0]]))


def test_n_step_ahead_prediction_passthrough(monkeypatch):
    def fake_super(self, x, y, steps):
        return np.array([[8.0]])

    monkeypatch.setattr(SimulateNARMAX, "_n_step_ahead_prediction", fake_super)

    model = MetaMSS()
    result = model._n_step_ahead_prediction(
        np.zeros((2, 1)), np.zeros((2, 1)), steps_ahead=3
    )

    assert np.array_equal(result, np.array([[8.0]]))


def test_model_prediction_dispatches_and_validates(monkeypatch):
    model = MetaMSS()
    model.model_type = "NARMAX"
    narmax_called = {}

    def fake_narmax(self, x, y_initial, forecast_horizon):
        narmax_called["called"] = True
        return np.array([[2.0]])

    monkeypatch.setattr(MetaMSS, "_narmax_predict", fake_narmax)

    result = model._model_prediction(
        np.zeros((1, 1)), np.zeros((1, 1)), forecast_horizon=3
    )
    assert narmax_called["called"]
    assert np.array_equal(result, np.array([[2.0]]))

    nfir_called = {}

    def fake_nfir(self, x, y_initial):
        nfir_called["called"] = True
        return np.array([[3.0]])

    monkeypatch.setattr(MetaMSS, "_nfir_predict", fake_nfir)

    model.model_type = "NFIR"
    result = model._model_prediction(np.zeros((1, 1)), np.zeros((1, 1)))
    assert nfir_called["called"]
    assert np.array_equal(result, np.array([[3.0]]))

    model.model_type = "UNKNOWN"
    with pytest.raises(ValueError, match="model_type must be"):
        model._model_prediction(np.zeros((1, 1)), np.zeros((1, 1)))


def test_nfir_predict_delegates_to_super(monkeypatch):
    def fake_super(self, x, y_initial):
        return np.array([[4.0]])

    monkeypatch.setattr(SimulateNARMAX, "_nfir_predict", fake_super)

    model = MetaMSS()
    result = model._nfir_predict(np.zeros((1, 1)), np.zeros((1, 1)))

    assert np.array_equal(result, np.array([[4.0]]))
