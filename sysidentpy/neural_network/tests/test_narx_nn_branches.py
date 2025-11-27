from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

import sysidentpy.neural_network.narx_nn as narx_module
from sysidentpy.basis_function import Polynomial
from sysidentpy.neural_network import NARXNN

# pylint: disable=protected-access


class _DummyBasis:
    degree = 1
    ensemble = False

    def fit(self, lagged_data, *args, **kwargs):
        _ = kwargs
        _ = args
        return np.ones((len(lagged_data), 1), dtype=np.float32)

    def transform(self, lagged_data, *args, **kwargs):
        _ = args
        _ = kwargs
        return np.ones((lagged_data.shape[0], 1), dtype=np.float32)


def _make_model(**kwargs):
    model = NARXNN(basis_function=kwargs.pop("basis_function", Polynomial()), **kwargs)
    return model


def test_loss_func_must_be_string():
    with pytest.raises(TypeError, match="loss_func must be provided as string"):
        _make_model(loss_func=123)


def test_optimizer_must_be_string():
    with pytest.raises(TypeError, match="optimizer must be provided as string"):
        _make_model(optimizer=object())


def test_sanitize_lag_validations():
    with pytest.raises(ValueError, match="list cannot be empty"):
        NARXNN._sanitize_lag([], "ylag")

    with pytest.raises(ValueError, match="All elements of ylag must be integers"):
        NARXNN._sanitize_lag([1, "a"], "ylag")

    with pytest.raises(ValueError, match="must be >= 1"):
        NARXNN._sanitize_lag([0, 1], "ylag")


def test_forward_numpy_moves_tensor_to_device(monkeypatch):
    class FakeTensor:
        def __init__(self, array):
            self.array = np.asarray(array, dtype=np.float32)
            self.to_calls = []

        def to(self, device, non_blocking=False):
            self.to_calls.append((device, non_blocking))
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.array

    class EchoNet:
        training = True

        def __call__(self, tensor):
            return tensor

    model = _make_model(net=EchoNet())
    fake_tensor_holder = {}

    def fake_from_numpy(array):
        tensor = FakeTensor(array)
        fake_tensor_holder["tensor"] = tensor
        return tensor

    monkeypatch.setattr(narx_module.torch, "from_numpy", fake_from_numpy)
    model.device = SimpleNamespace(type="cuda")

    result = model._forward_numpy(np.array([[1.0]], dtype=np.float32))

    assert np.array_equal(result, np.array([[1.0]], dtype=np.float32))
    assert fake_tensor_holder["tensor"].to_calls[0][0] == model.device


def test_seed_torch_generators_calls_cuda_seed(monkeypatch):
    model = _make_model(random_state=4)
    calls = {}

    monkeypatch.setattr(
        narx_module.torch,
        "manual_seed",
        lambda value: calls.setdefault("cpu", value),
    )
    monkeypatch.setattr(narx_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        narx_module.torch.cuda,
        "manual_seed_all",
        lambda value: calls.setdefault("cuda", value),
    )

    model._seed_torch_generators()

    assert calls["cpu"] == 4
    assert calls["cuda"] == 4


def test_split_data_requires_y():
    model = _make_model()
    with pytest.raises(ValueError, match="y cannot be None"):
        model.split_data(np.ones((4, 1), dtype=np.float32), None)


def test_split_data_with_none_input_sets_default_inputs(monkeypatch):
    model = _make_model()

    monkeypatch.setattr(
        narx_module,
        "build_lagged_matrix",
        lambda *args, **kwargs: np.ones((3, 2), dtype=np.float32),
    )
    model.basis_function.fit = lambda *args, **kwargs: np.ones((3, 2), dtype=np.float32)

    model.split_data(None, np.ones((5, 1), dtype=np.float32))
    assert model.n_inputs == 1


def test_get_data_uses_default_shuffle(monkeypatch):
    recorded = {}

    class FakeLoader:
        def __init__(self, *args, **kwargs):
            recorded["args"] = args
            recorded["kwargs"] = kwargs

    monkeypatch.setattr(narx_module, "DataLoader", FakeLoader)
    model = _make_model()
    model.shuffle_batches = True
    model.device = SimpleNamespace(type="cpu")

    model.get_data("dataset", shuffle=None)

    assert recorded["kwargs"]["shuffle"] is True
    assert recorded["kwargs"]["pin_memory"] is False


def test_fit_verbose_tracks_losses(monkeypatch):
    class FakeTensor:
        def to(self, *_args, **_kwargs):
            return self

    train_dl = [(FakeTensor(), FakeTensor())]
    valid_dl = [(FakeTensor(), FakeTensor())]
    loaders = deque([train_dl, valid_dl])

    class FakeNet:
        def __init__(self):
            self.training = True

        def train(self):
            self.training = True

        def eval(self):
            self.training = False

    model = _make_model(net=FakeNet(), epochs=1, verbose=True)

    def fake_loss_batch(*_args, **_kwargs):
        return 0.5, 2

    model.loss_batch = fake_loss_batch

    def fake_define_opt():
        return SimpleNamespace()

    model.define_opt = fake_define_opt
    monkeypatch.setattr(model, "data_transform", lambda *a, **k: loaders.popleft())

    X = np.ones((4, 1), dtype=np.float32)
    y = np.ones((4, 1), dtype=np.float32)

    model.fit(X=X, y=y, X_test=X, y_test=y)

    assert model.train_loss == [0.5]
    assert model.val_loss == [0.5]


def test_one_step_prediction_requires_y():
    model = _make_model()
    with pytest.raises(ValueError, match="y cannot be None"):
        model._one_step_ahead_prediction(np.zeros((2, 1), dtype=np.float32), None)


def test_n_step_prediction_needs_initial_conditions():
    model = _make_model()
    model.max_lag = 3
    model.n_inputs = 1
    y = np.ones((2, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="Insufficient initial condition elements"):
        model._n_step_ahead_prediction(
            np.ones((4, 1), dtype=np.float32), y, steps_ahead=1
        )


def test_model_prediction_invalid_type():
    model = _make_model()
    model.model_type = "UNKNOWN"
    with pytest.raises(ValueError, match="model_type must be NARMAX, NAR or NFIR"):
        model._model_prediction(
            np.ones((2, 1), dtype=np.float32),
            np.ones((2, 1), dtype=np.float32),
        )


def test_narmax_predict_requires_enough_initial_conditions():
    model = _make_model()
    model.max_lag = 2
    with pytest.raises(ValueError, match="Insufficient initial condition elements"):
        model._narmax_predict(
            np.ones((3, 1), dtype=np.float32),
            np.ones((1, 1), dtype=np.float32),
        )


def test_narmax_predict_handles_missing_inputs():
    model = _make_model()
    model.max_lag = 1
    model.n_inputs = 0
    model.final_model = np.array([[0]], dtype=int)
    model._code2exponents = lambda **_kwargs: np.zeros(1, dtype=np.float32)
    model._scalar_forward = lambda *_args, **_kwargs: 0.0

    y_initial = np.ones((1, 1), dtype=np.float32)
    result = model._narmax_predict(x=None, y_initial=y_initial, forecast_horizon=2)

    assert result.shape == (3, 1)


def test_narmax_predict_sets_nar_inputs_to_zero():
    model = _make_model(model_type="NAR")
    model.max_lag = 1
    model.n_inputs = 2
    model.final_model = np.array([[0]], dtype=int)
    model._code2exponents = lambda **_kwargs: np.zeros(1, dtype=np.float32)
    model._scalar_forward = lambda *_args, **_kwargs: 0.0

    x = np.ones((3, 2), dtype=np.float32)
    y_initial = np.ones((1, 1), dtype=np.float32)
    model._narmax_predict(x=x, y_initial=y_initial)

    assert model.n_inputs == 0


def test_basis_function_predict_modes():
    model = _make_model(basis_function=_DummyBasis())
    model.max_lag = 1
    model.xlag = 1
    model.ylag = 1
    model.n_inputs = 1
    model._scalar_forward = lambda *_args, **_kwargs: 0.0

    x = np.ones((3, 1), dtype=np.float32)
    y_initial = np.ones((1, 1), dtype=np.float32)

    model.model_type = "NARMAX"
    assert model._basis_function_predict(x, y_initial).shape == (3, 1)

    model.model_type = "NAR"
    model.n_inputs = 1
    output = model._basis_function_predict(None, y_initial, forecast_horizon=2)
    assert output.shape[0] == 3
    assert model.n_inputs == 0

    model.model_type = "NFIR"
    assert model._basis_function_predict(x, y_initial).shape == (3, 1)

    model.model_type = "UNKNOWN"
    with pytest.raises(ValueError, match="Unrecognized model type"):
        model._basis_function_predict(x, y_initial)


def test_basis_function_n_step_prediction_validations():
    model = _make_model(basis_function=_DummyBasis())
    model.max_lag = 2

    with pytest.raises(ValueError, match="Insufficient initial condition elements"):
        model._basis_function_n_step_prediction(
            np.ones((4, 1), dtype=np.float32),
            np.ones((1, 1), dtype=np.float32),
            1,
            1,
        )

    model.max_lag = 1
    model.model_type = "NAR"
    model._basis_function_predict = lambda *args, **kwargs: np.arange(
        kwargs.get("forecast_horizon", 2), dtype=np.float32
    ).reshape(-1, 1)
    y = np.ones((4, 1), dtype=np.float32)
    result = model._basis_function_n_step_prediction(
        x=None,
        y=y,
        steps_ahead=1,
        forecast_horizon=2,
    )
    assert result.shape[0] == 3


def test_basis_function_n_step_prediction_modes():
    model = _make_model(basis_function=_DummyBasis())
    model.max_lag = 1
    model.model_type = "NARMAX"

    def fake_predict(_self, *args, **kwargs):
        y_slice = kwargs.get("y_initial")
        if y_slice is None and len(args) > 1:
            y_slice = args[1]
        if y_slice is None:
            y_slice = kwargs.get("x")
        horizon = kwargs.get("forecast_horizon", y_slice.shape[0])
        length = max(horizon, y_slice.shape[0])
        return np.arange(length, dtype=np.float32).reshape(-1, 1)

    model._basis_function_predict = fake_predict.__get__(model, NARXNN)

    x = np.ones((4, 1), dtype=np.float32)
    y = np.ones((4, 1), dtype=np.float32)
    result = model._basis_function_n_step_prediction(
        x, y, steps_ahead=1, forecast_horizon=1
    )
    assert result.shape == (4, 1)

    model.model_type = "NAR"
    result = model._basis_function_n_step_prediction(
        x=None, y=y, steps_ahead=1, forecast_horizon=2
    )
    assert result.shape == (3, 1)

    model.model_type = "NFIR"
    result = model._basis_function_n_step_prediction(
        x=x, y=y, steps_ahead=1, forecast_horizon=1
    )
    assert result.shape == (4, 1)

    model.model_type = "UNKNOWN"
    with pytest.raises(ValueError, match="model_type must be NARMAX, NAR or NFIR"):
        model._basis_function_n_step_prediction(x, y, steps_ahead=1, forecast_horizon=1)
