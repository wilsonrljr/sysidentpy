import builtins
from collections import deque
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

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


class _EarlyStoppingNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return x[:, :1] * self.weight


def _fit_with_validation_losses(monkeypatch, validation_losses, **kwargs):
    net = _EarlyStoppingNet()
    model = _make_model(
        net=net,
        early_stopping=True,
        epochs=kwargs.pop("epochs", len(validation_losses)),
        **kwargs,
    )
    batch = (
        torch.ones((2, 1), dtype=torch.float32),
        torch.ones((2, 1), dtype=torch.float32),
    )
    loaders = deque([[batch], [batch]])
    epochs_completed = 0

    monkeypatch.setattr(model, "data_transform", lambda *a, **k: loaders.popleft())
    monkeypatch.setattr(model, "define_opt", object)

    def fake_loss_batch(x, _y, opt=None):
        nonlocal epochs_completed
        if opt is not None:
            epochs_completed += 1
            with torch.no_grad():
                net.weight.fill_(epochs_completed)
            return 0.5, len(x)
        return validation_losses[epochs_completed - 1], len(x)

    monkeypatch.setattr(model, "loss_batch", fake_loss_batch)
    data = np.ones((4, 1), dtype=np.float32)
    model.fit(X=data, y=data, X_test=data, y_test=data)
    return model, epochs_completed


def test_loss_func_must_be_string():
    with pytest.raises(TypeError, match="loss_func must be provided as string"):
        _make_model(loss_func=123)


def test_optimizer_must_be_string():
    with pytest.raises(TypeError, match="optimizer must be provided as string"):
        _make_model(optimizer=object())


def test_optional_torch_import_paths(monkeypatch):
    original_import = builtins.__import__

    def import_without_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch intentionally unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_torch)
    spec = importlib.util.spec_from_file_location(
        "sysidentpy.neural_network._narx_nn_without_torch",
        narx_module.__file__,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.torch is None
    with pytest.raises(ImportError, match="PyTorch is required"):
        module.NARXNN()

    def import_without_narx_nn(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "narx_nn" and level == 1:
            raise ImportError("narx_nn intentionally unavailable")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_narx_nn)
    init_path = Path(narx_module.__file__).with_name("__init__.py")
    package_spec = importlib.util.spec_from_file_location(
        "sysidentpy.neural_network",
        init_path,
        submodule_search_locations=[str(init_path.parent)],
    )
    package = importlib.util.module_from_spec(package_spec)
    package_spec.loader.exec_module(package)
    assert not hasattr(package, "NARXNN")


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    [
        ({"early_stopping": 1}, TypeError, "early_stopping must be False or True"),
        ({"patience": 0}, ValueError, "patience must be integer and > zero"),
        ({"patience": 1.5}, ValueError, "patience must be integer and > zero"),
        ({"patience": True}, ValueError, "patience must be integer and > zero"),
        ({"min_delta": -0.1}, ValueError, "min_delta must be a finite number"),
        ({"min_delta": np.inf}, ValueError, "min_delta must be a finite number"),
        ({"min_delta": True}, ValueError, "min_delta must be a finite number"),
        ({"min_delta": "0"}, ValueError, "min_delta must be a finite number"),
    ],
)
def test_early_stopping_parameter_validation(kwargs, exception, message):
    with pytest.raises(exception, match=message):
        _make_model(**kwargs)


def test_sanitize_lag_validations():
    model = _make_model()

    with pytest.raises(ValueError, match="list cannot be empty"):
        model._sanitize_lag([], "ylag")

    with pytest.raises(ValueError, match="All elements of ylag must be integers"):
        model._sanitize_lag([1, "a"], "ylag")

    with pytest.raises(ValueError, match="must be >= 1"):
        model._sanitize_lag([0, 1], "ylag")


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


def test_seed_torch_generators_ignores_missing_random_state(monkeypatch):
    model = _make_model(random_state=None)
    monkeypatch.setattr(
        narx_module.torch,
        "manual_seed",
        lambda *_args: pytest.fail("manual_seed should not be called"),
    )

    model._seed_torch_generators()


def test_loss_batch_without_optimizer_does_not_update_weights():
    net = torch.nn.Linear(1, 1)
    model = _make_model(net=net)
    x = torch.ones((2, 1), dtype=torch.float32)
    y = torch.zeros((2, 1), dtype=torch.float32)
    weights_before = {
        name: value.detach().clone() for name, value in net.state_dict().items()
    }

    loss, batch_size = model.loss_batch(x, y)

    assert np.isfinite(loss)
    assert batch_size == 2
    for name, value in net.state_dict().items():
        torch.testing.assert_close(value, weights_before[name])


def test_split_data_requires_y():
    model = _make_model()
    with pytest.raises(ValueError, match="y cannot be None"):
        model.split_data(np.ones((4, 1), dtype=np.float32), None)


def test_data_transform_requires_y():
    model = _make_model()
    with pytest.raises(ValueError, match="y cannot be None"):
        model.data_transform(np.ones((4, 1), dtype=np.float32), None)


def test_split_data_with_none_input_sets_default_inputs(monkeypatch):
    model = _make_model()

    monkeypatch.setattr(
        narx_module,
        "build_lagged_matrix",
        lambda *args, **kwargs: np.ones((3, 2), dtype=np.float32),
    )
    model.basis_function.fit = lambda *args, **kwargs: np.ones((3, 6), dtype=np.float32)

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


def test_fit_without_early_stopping_runs_all_epochs(monkeypatch):
    model = _make_model(net=_EarlyStoppingNet(), epochs=4)
    batch = (
        torch.ones((2, 1), dtype=torch.float32),
        torch.ones((2, 1), dtype=torch.float32),
    )
    transform_calls = 0
    epochs_completed = 0

    def fake_data_transform(*_args, **_kwargs):
        nonlocal transform_calls
        transform_calls += 1
        return [batch]

    def fake_loss_batch(x, _y, opt=None):
        nonlocal epochs_completed
        assert opt is not None
        epochs_completed += 1
        return 0.5, len(x)

    monkeypatch.setattr(model, "data_transform", fake_data_transform)
    monkeypatch.setattr(model, "define_opt", object)
    monkeypatch.setattr(model, "loss_batch", fake_loss_batch)

    data = np.ones((4, 1), dtype=np.float32)
    model.fit(X=data, y=data)

    assert epochs_completed == model.epochs
    assert transform_calls == 1
    assert model.train_loss == []
    assert model.val_loss == []


def test_fit_requires_defined_network():
    model = _make_model()
    data = np.ones((4, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="defined before training"):
        model.fit(X=data, y=data)


def test_early_stopping_requires_validation_data(monkeypatch):
    model = _make_model(net=_EarlyStoppingNet(), early_stopping=True, epochs=1)
    batch = (
        torch.ones((2, 1), dtype=torch.float32),
        torch.ones((2, 1), dtype=torch.float32),
    )
    monkeypatch.setattr(model, "data_transform", lambda *a, **k: [batch])

    data = np.ones((4, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="early_stopping=True"):
        model.fit(X=data, y=data)


def test_invalid_early_stopping_fit_does_not_reset_network():
    net = torch.nn.Linear(2, 1)
    model = _make_model(
        net=net,
        random_state=7,
        early_stopping=True,
        epochs=1,
    )
    state_before = {
        name: value.detach().clone() for name, value in net.state_dict().items()
    }
    data = np.ones((4, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="early_stopping=True"):
        model.fit(X=data, y=data)

    for name, value in net.state_dict().items():
        torch.testing.assert_close(value, state_before[name])


def test_data_transform_rejects_data_without_post_lag_samples():
    model = _make_model()
    data = np.ones((1, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="more samples than the maximum lag"):
        model.data_transform(data, data)


def test_early_stopping_interrupts_and_restores_best_weights(monkeypatch):
    model, epochs_completed = _fit_with_validation_losses(
        monkeypatch,
        [1.0, 0.8, 0.9, 0.7, 0.6],
        patience=1,
    )

    assert epochs_completed == 3
    assert model.train_loss == [0.5, 0.5, 0.5]
    assert model.val_loss == [1.0, 0.8, 0.9]
    assert model.net.weight.item() == pytest.approx(2.0)


def test_early_stopping_patience_resets_after_improvement(monkeypatch):
    model, epochs_completed = _fit_with_validation_losses(
        monkeypatch,
        [1.0, 1.1, 0.9, 1.0, 1.1, 0.5],
        patience=2,
    )

    assert epochs_completed == 5
    assert model.val_loss == [1.0, 1.1, 0.9, 1.0, 1.1]
    assert model.net.weight.item() == pytest.approx(3.0)


def test_early_stopping_respects_min_delta(monkeypatch):
    model, epochs_completed = _fit_with_validation_losses(
        monkeypatch,
        [1.0, 0.95, 0.7],
        patience=1,
        min_delta=0.1,
    )

    assert epochs_completed == 2
    assert model.val_loss == [1.0, 0.95]
    assert model.net.weight.item() == pytest.approx(2.0)


def test_early_stopping_restores_best_weights_after_all_epochs(monkeypatch):
    model, epochs_completed = _fit_with_validation_losses(
        monkeypatch,
        [1.0, 0.5, 0.75],
        patience=10,
    )

    assert epochs_completed == 3
    assert model.net.weight.item() == pytest.approx(2.0)


def test_early_stopping_accumulates_small_improvements(monkeypatch):
    model, epochs_completed = _fit_with_validation_losses(
        monkeypatch,
        [1.0, 0.95, 0.89, 0.90, 0.91],
        patience=2,
        min_delta=0.1,
    )

    assert epochs_completed == 5
    assert model.net.weight.item() == pytest.approx(3.0)


def test_early_stopping_rejects_non_finite_validation_loss(monkeypatch):
    with pytest.raises(ValueError, match="Validation loss must be finite"):
        _fit_with_validation_losses(monkeypatch, [np.nan], patience=1)


def test_validation_loss_is_weighted_by_batch_size(monkeypatch):
    model = _make_model(net=_EarlyStoppingNet(), epochs=1, verbose=True)
    train_batch = (
        torch.ones((3, 1), dtype=torch.float32),
        torch.ones((3, 1), dtype=torch.float32),
    )
    validation_batches = [
        (
            torch.ones((2, 1), dtype=torch.float32),
            torch.ones((2, 1), dtype=torch.float32),
        ),
        (
            torch.ones((1, 1), dtype=torch.float32),
            torch.ones((1, 1), dtype=torch.float32),
        ),
    ]
    loaders = deque([[train_batch], validation_batches])

    monkeypatch.setattr(model, "data_transform", lambda *a, **k: loaders.popleft())
    monkeypatch.setattr(model, "define_opt", object)

    def fake_loss_batch(x, _y, opt=None):
        if opt is not None:
            return 2.0, len(x)
        return (1.0 if len(x) == 2 else 4.0), len(x)

    monkeypatch.setattr(model, "loss_batch", fake_loss_batch)
    data = np.ones((4, 1), dtype=np.float32)

    model.fit(X=data, y=data, X_test=data, y_test=data)

    assert model.train_loss == [2.0]
    assert model.val_loss == [2.0]


def test_repeated_early_stopping_fits_are_independent():
    data = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(-1, 1)
    y = 0.5 * data
    net = torch.nn.Sequential(
        torch.nn.Linear(2, 4),
        torch.nn.Tanh(),
        torch.nn.Linear(4, 1),
    )
    model = _make_model(
        net=net,
        basis_function=Polynomial(degree=1),
        epochs=5,
        learning_rate=0.0,
        optimizer="SGD",
        random_state=3,
        early_stopping=True,
        patience=2,
    )

    model.fit(X=data[:8], y=y[:8], X_test=data[8:], y_test=y[8:])
    first_losses = model.val_loss.copy()
    first_state = {
        name: value.detach().clone() for name, value in net.state_dict().items()
    }

    model.fit(X=data[:8], y=y[:8], X_test=data[8:], y_test=y[8:])

    assert len(first_losses) == len(model.val_loss) == 3
    np.testing.assert_allclose(model.val_loss, first_losses, rtol=0, atol=0)
    for name, value in net.state_dict().items():
        torch.testing.assert_close(value, first_state[name])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_early_stopping_preserves_cuda_model_device():
    data = np.linspace(0.0, 1.0, 8, dtype=np.float32).reshape(-1, 1)
    y = 0.5 * data
    net = torch.nn.Linear(2, 1).to("cuda")
    model = _make_model(
        net=net,
        basis_function=Polynomial(degree=1),
        device="cuda",
        epochs=3,
        learning_rate=0.0,
        early_stopping=True,
        patience=1,
    )

    model.fit(X=data[:5], y=y[:5], X_test=data[5:], y_test=y[5:])

    assert len(model.val_loss) == 2
    assert all(parameter.device.type == "cuda" for parameter in net.parameters())


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


def test_model_prediction_dispatches_nfir(monkeypatch):
    model = _make_model(model_type="NFIR")
    expected = np.ones((2, 1), dtype=np.float32)
    monkeypatch.setattr(model, "_nfir_predict", lambda *_args: expected)

    result = model._model_prediction(
        np.ones((2, 1), dtype=np.float32),
        np.ones((2, 1), dtype=np.float32),
    )

    assert result is expected


def test_predict_preserves_eval_mode(monkeypatch):
    model = _make_model(net=_EarlyStoppingNet())
    model.net.eval()
    expected = np.ones((2, 1), dtype=np.float32)
    monkeypatch.setattr(model, "_model_prediction", lambda *_args, **_kwargs: expected)

    result = model.predict(
        X=np.ones((2, 1), dtype=np.float32),
        y=np.ones((2, 1), dtype=np.float32),
    )

    assert result is expected
    assert model.net.training is False


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
    assert result.shape == y.shape
    np.testing.assert_array_equal(result[: model.max_lag], y[: model.max_lag])


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
    assert result.shape == y.shape
    np.testing.assert_array_equal(result[: model.max_lag], y[: model.max_lag])

    model.model_type = "NFIR"
    result = model._basis_function_n_step_prediction(
        x=x, y=y, steps_ahead=1, forecast_horizon=1
    )
    assert result.shape == (4, 1)

    model.model_type = "UNKNOWN"
    with pytest.raises(ValueError, match="model_type must be NARMAX, NAR or NFIR"):
        model._basis_function_n_step_prediction(x, y, steps_ahead=1, forecast_horizon=1)
