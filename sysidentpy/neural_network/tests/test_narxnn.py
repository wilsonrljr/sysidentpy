from unittest.mock import MagicMock, patch

import numpy as np
import torch
from numpy.testing import assert_almost_equal, assert_equal, assert_raises
from torch import nn

from sysidentpy.basis_function import Fourier, Polynomial
from sysidentpy.neural_network import NARXNN
from sysidentpy.utils.information_matrix import build_lagged_matrix
from sysidentpy.utils.narmax_tools import regressor_code
from sysidentpy.tests.test_narmax_base import create_test_data
import pytest
from sysidentpy.neural_network.narx_nn import _check_cuda

torch.manual_seed(0)


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

basis_function = Polynomial(degree=1)
regressors = regressor_code(
    X=X_train,
    xlag=2,
    model_type="NFIR",
    model_representation="neural_network",
    basis_function=basis_function,
)
n_features = regressors.shape[0]


class NARX(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(n_features, 30)
        self.lin2 = nn.Linear(30, 30)
        self.lin3 = nn.Linear(30, 1)
        self.tanh = nn.Tanh()

    def forward(self, xb):
        z = self.lin(xb)
        z = self.tanh(z)
        z = self.lin2(z)
        z = self.tanh(z)
        z = self.lin3(z)
        return z


class _ExpandedPolynomial(Polynomial):
    def fit(
        self,
        data,
        max_lag=1,
        ylag=1,
        xlag=1,
        model_type="NARMAX",
        predefined_regressors=None,
    ):
        base_features = super().fit(
            data,
            max_lag,
            ylag,
            xlag,
            model_type,
            predefined_regressors=None,
        )
        custom_feature = data[max_lag:, 1:2] ** 3
        features = np.column_stack([base_features, custom_feature])
        if predefined_regressors is None:
            return features
        return features[:, predefined_regressors]


class _DeterministicNARNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[0.4, -0.15]], dtype=torch.float32))

    def forward(self, xb):
        return self.linear(xb)


class _DeterministicFourierNARNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.tensor([[0.12, -0.08, 0.04, 0.2]], dtype=torch.float32)
            )

    def forward(self, xb):
        return self.linear(xb)


def _first_regressor_value(values):
    return float(np.asarray(values).reshape(-1)[0])


def _build_deterministic_nar_model(y, include_bias):
    model = NARXNN(
        net=_DeterministicNARNet(),
        ylag=2,
        xlag=2,
        model_type="NAR",
        basis_function=Polynomial(degree=1, include_bias=include_bias),
    )
    model.split_data(None, y)
    return model


def _build_deterministic_fourier_nar_model(y):
    model = NARXNN(
        net=_DeterministicFourierNARNet(),
        ylag=2,
        xlag=2,
        model_type="NAR",
        basis_function=Fourier(n=1, degree=1, ensemble=False),
    )
    model.split_data(None, y)
    return model


def _segmented_nar_free_run_reference(model, y, steps_ahead):
    reference = np.empty_like(y, dtype=np.float32)
    reference[: model.max_lag] = y[: model.max_lag]
    for start in range(model.max_lag, len(y), steps_ahead):
        block_horizon = min(steps_ahead, len(y) - start)
        initial_conditions = y[start - model.max_lag : start]
        free_run = model.predict(
            X=None,
            y=initial_conditions,
            forecast_horizon=block_horizon,
        )
        reference[start : start + block_horizon] = free_run[-block_horizon:]

    return reference


def _fourier_nar_one_step_reference(model, y):
    lagged_data = build_lagged_matrix(
        None,
        y,
        model.xlag,
        model.ylag,
        model.model_type,
    )
    regressor_matrix = model.basis_function.transform(
        lagged_data,
        model.max_lag,
        model.ylag,
        model.xlag,
        model.model_type,
    )
    regressor_matrix, _ = model._prepare_regressor_matrix(regressor_matrix, 1)
    prediction = model._forward_numpy(regressor_matrix)
    return np.concatenate([y[: model.max_lag], prediction], axis=0).astype(np.float32)


def _fourier_nar_free_run_reference(model, y_initial, forecast_horizon):
    reference = np.full(
        (model.max_lag + forecast_horizon, 1),
        np.nan,
        dtype=np.float32,
    )
    reference[: model.max_lag] = y_initial[: model.max_lag]

    for index in range(model.max_lag, len(reference)):
        context = np.concatenate(
            [
                reference[index - model.max_lag : index],
                np.zeros((1, 1), dtype=np.float32),
            ],
            axis=0,
        )
        reference[index] = _fourier_nar_one_step_reference(model, context)[-1]

    return reference


def test_default_values():
    default = {
        "ylag": 1,
        "xlag": 1,
        "model_type": "NARMAX",
        "batch_size": 100,
        "shuffle_batches": False,
        "learning_rate": 0.01,
        "epochs": 200,
        "optimizer": "Adam",
        "net": None,
        "train_percentage": 80,
        "verbose": False,
        "optim_params": {},
        "random_state": None,
        "early_stopping": False,
        "patience": 10,
        "min_delta": 0.0,
    }
    model = NARXNN(basis_function=Polynomial())
    model_values = [
        model.ylag,
        model.xlag,
        model.model_type,
        model.batch_size,
        model.shuffle_batches,
        model.learning_rate,
        model.epochs,
        model.optimizer,
        model.net,
        model.train_percentage,
        model.verbose,
        model.optim_params,
        model.random_state,
        model.early_stopping,
        model.patience,
        model.min_delta,
    ]
    assert list(default.values()) == model_values


def test_reference_network_forward_pass_yields_single_output():
    net = NARX()
    sample = torch.zeros((2, n_features), dtype=torch.float32)
    output = net(sample)
    assert_equal(tuple(output.shape), (2, 1))


def test_validate():
    assert_raises(ValueError, NARXNN, ylag=-1, basis_function=Polynomial(degree=1))
    assert_raises(ValueError, NARXNN, ylag=1.3, basis_function=Polynomial(degree=1))
    assert_raises(ValueError, NARXNN, xlag=1.3, basis_function=Polynomial(degree=1))
    assert_raises(ValueError, NARXNN, xlag=-1, basis_function=Polynomial(degree=1))
    assert_raises(
        ValueError, NARXNN, train_percentage=0, basis_function=Polynomial(degree=1)
    )
    assert_raises(
        ValueError, NARXNN, train_percentage=101, basis_function=Polynomial(degree=1)
    )
    assert_raises(
        ValueError,
        NARXNN,
        basis_function=Polynomial(degree=1),
        optimizer="NotAnOpt",
    )
    assert_raises(
        ValueError,
        NARXNN,
        basis_function=Polynomial(degree=1),
        loss_func="not_a_loss",
    )
    assert_raises(
        TypeError,
        NARXNN,
        basis_function=Polynomial(degree=1),
        optim_params=[("lr", 0.1)],
    )
    assert_raises(
        TypeError,
        NARXNN,
        basis_function=Polynomial(degree=1),
        shuffle_batches="yes",
    )


def test_sanitize_lag_sequence_conversion():
    model = NARXNN(basis_function=Polynomial(degree=1))
    sanitized = model._sanitize_lag([1, 2, 3], "ylag")
    assert sanitized == [1, 2, 3]


def test_seed_torch_generators_sets_seed(monkeypatch):
    model = NARXNN(basis_function=Polynomial(degree=1), random_state=7)
    recorded = {}

    def fake_manual_seed(value):
        recorded["cpu"] = value

    monkeypatch.setattr(torch, "manual_seed", fake_manual_seed)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model._seed_torch_generators()
    assert recorded["cpu"] == 7


def test_reset_network_parameters_requires_net():
    model = NARXNN(basis_function=Polynomial(degree=1))
    assert_raises(ValueError, model._reset_network_parameters)


def test_reset_network_parameters_invokes_reset():
    class TinyNet(nn.Linear):
        def __init__(self):
            super().__init__(1, 1)
            self.reset_called = False

        def reset_parameters(self):
            self.reset_called = True
            super().reset_parameters()

    net = TinyNet()
    model = NARXNN(net=net, basis_function=Polynomial(degree=1))
    model._reset_network_parameters()
    assert net.reset_called is True


def test_data_transform_allows_shuffle_override(monkeypatch):
    model = NARXNN(basis_function=Polynomial(degree=1))
    captured = {}

    def fake_get_data(train_ds, *, shuffle=None):
        captured["shuffle"] = shuffle
        return (train_ds, shuffle)

    monkeypatch.setattr(model, "get_data", fake_get_data)
    dataloader = model.data_transform(X_train[:10], y_train[:10], shuffle=False)
    assert captured["shuffle"] is False
    assert isinstance(dataloader, tuple)


def test_polynomial_bias_modes_produce_same_neural_inputs_and_one_step_matrix():
    with_bias = NARXNN(
        basis_function=Polynomial(degree=2, include_bias=True),
        xlag=2,
        ylag=2,
    )
    without_bias = NARXNN(
        basis_function=Polynomial(degree=2, include_bias=False),
        xlag=2,
        ylag=2,
    )

    matrix_with_bias, _ = with_bias.split_data(X_train[:20], y_train[:20])
    matrix_without_bias, _ = without_bias.split_data(X_train[:20], y_train[:20])

    np.testing.assert_allclose(matrix_with_bias, matrix_without_bias)
    np.testing.assert_array_equal(with_bias.regressor_code, without_bias.regressor_code)
    assert matrix_with_bias.shape[1] == with_bias.regressor_code.shape[0]
    assert not np.any(np.all(with_bias.regressor_code == 0, axis=1))

    n_predictions = X_test[:10].shape[0] - with_bias.max_lag
    with_bias_forward = MagicMock(return_value=np.zeros((n_predictions, 1)))
    without_bias_forward = MagicMock(return_value=np.zeros((n_predictions, 1)))
    with_bias._forward_numpy = with_bias_forward
    without_bias._forward_numpy = without_bias_forward
    with_bias._one_step_ahead_prediction(X_test[:10], y_test[:10])
    without_bias._one_step_ahead_prediction(X_test[:10], y_test[:10])

    matrix_with_bias = with_bias_forward.call_args.args[0]
    matrix_without_bias = without_bias_forward.call_args.args[0]
    np.testing.assert_allclose(matrix_with_bias, matrix_without_bias)
    assert matrix_with_bias.shape[1] == with_bias.regressor_code.shape[0]


@pytest.mark.parametrize("include_bias", [True, False])
@pytest.mark.parametrize("steps_ahead", [2, 3, 30])
def test_polynomial_nar_n_step_matches_segmented_free_run(include_bias, steps_ahead):
    sample = np.arange(25, dtype=np.float32)
    y_data = (0.1 * sample + 0.5 * np.sin(sample / 2)).reshape(-1, 1)
    model = _build_deterministic_nar_model(y_data, include_bias)

    reference = _segmented_nar_free_run_reference(model, y_data, steps_ahead)
    prediction = model.predict(X=None, y=y_data, steps_ahead=steps_ahead)

    assert prediction.shape == y_data.shape
    np.testing.assert_array_equal(prediction[: model.max_lag], y_data[: model.max_lag])
    np.testing.assert_allclose(prediction, reference, rtol=1e-6, atol=1e-6)
    assert np.all(np.isfinite(prediction))


@pytest.mark.parametrize("forecast_horizon", [None, 1, 100])
@pytest.mark.parametrize("steps_ahead", [2, 3, 30])
@pytest.mark.parametrize("training", [True, False])
def test_fourier_nar_n_step_matches_segmented_free_run(
    steps_ahead,
    forecast_horizon,
    training,
):
    sample = np.arange(25, dtype=np.float32)
    y_data = (0.1 * sample + 0.5 * np.sin(sample / 2)).reshape(-1, 1)
    model = _build_deterministic_fourier_nar_model(y_data)
    model.net.train(training)

    reference = _segmented_nar_free_run_reference(model, y_data, steps_ahead)
    prediction = model.predict(
        X=None,
        y=y_data,
        steps_ahead=steps_ahead,
        forecast_horizon=forecast_horizon,
    )

    assert prediction.shape == y_data.shape
    np.testing.assert_array_equal(prediction[: model.max_lag], y_data[: model.max_lag])
    np.testing.assert_allclose(prediction, reference, rtol=1e-6, atol=1e-6)
    assert np.all(np.isfinite(prediction))
    assert model.net.training is training


def test_fourier_nar_n_step_with_only_initial_conditions_returns_prefix():
    y_data = np.array([[0.2], [-0.1], [0.3]], dtype=np.float32)
    model = _build_deterministic_fourier_nar_model(y_data)
    initial_conditions = y_data[: model.max_lag]

    prediction = model.predict(X=None, y=initial_conditions, steps_ahead=3)

    assert prediction.shape == initial_conditions.shape
    np.testing.assert_array_equal(prediction, initial_conditions)


def test_fourier_nar_n_step_requires_initial_conditions():
    sample = np.arange(10, dtype=np.float32)
    y_data = np.sin(sample / 2).reshape(-1, 1)
    model = _build_deterministic_fourier_nar_model(y_data)

    with pytest.raises(ValueError, match="Insufficient initial condition elements"):
        model.predict(X=None, y=y_data[: model.max_lag - 1], steps_ahead=2)


def test_fourier_nar_one_step_and_free_run_regressions():
    sample = np.arange(25, dtype=np.float32)
    y_data = (0.1 * sample + 0.5 * np.sin(sample / 2)).reshape(-1, 1)
    model = _build_deterministic_fourier_nar_model(y_data)
    forecast_horizon = 5

    expected_one_step = _fourier_nar_one_step_reference(model, y_data)
    expected_free_run = _fourier_nar_free_run_reference(
        model,
        y_data[: model.max_lag],
        forecast_horizon,
    )
    one_step = model.predict(X=None, y=y_data, steps_ahead=1)
    free_run = model.predict(
        X=None,
        y=y_data[: model.max_lag],
        forecast_horizon=forecast_horizon,
    )

    assert one_step.shape == y_data.shape
    assert free_run.shape == (model.max_lag + forecast_horizon, 1)
    np.testing.assert_array_equal(one_step[: model.max_lag], y_data[: model.max_lag])
    np.testing.assert_array_equal(free_run[: model.max_lag], y_data[: model.max_lag])
    np.testing.assert_allclose(one_step, expected_one_step, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(free_run, expected_free_run, rtol=1e-6, atol=1e-6)


def test_custom_polynomial_layout_removes_only_one_bias_code():
    x_data = X_train[:20]
    y_data = y_train[:20]
    basis_function = _ExpandedPolynomial(degree=1, include_bias=True)
    model = NARXNN(basis_function=basis_function, xlag=1, ylag=1)
    lagged_data = build_lagged_matrix(
        x_data, y_data, model.xlag, model.ylag, model.model_type
    )
    full_matrix = basis_function.fit(lagged_data, max_lag=1, ylag=1, xlag=1)

    regressor_matrix, _ = model.split_data(x_data, y_data)

    np.testing.assert_allclose(regressor_matrix, full_matrix[:, 1:])
    assert regressor_matrix.shape[1] == full_matrix.shape[1] - 1
    assert model.regressor_code.shape[0] == regressor_matrix.shape[1]
    assert np.count_nonzero(np.all(model.regressor_code == 0, axis=1)) == 1


@pytest.mark.parametrize("include_bias", [True, False])
def test_polynomial_nfir_recursive_prediction_uses_input_exponent_block(include_bias):
    x_data = np.arange(1, 7, dtype=float).reshape(-1, 1)
    y_data = np.zeros_like(x_data)
    model = NARXNN(
        xlag=1,
        ylag=1,
        model_type="NFIR",
        basis_function=Polynomial(degree=1, include_bias=include_bias),
    )
    regressor_matrix, _ = model.split_data(x_data, y_data)
    model._scalar_forward = _first_regressor_value

    prediction = model._nfir_predict(x_data, y_data)

    np.testing.assert_allclose(regressor_matrix[:, 0], x_data[:-1, 0])
    np.testing.assert_allclose(prediction[1:, 0], regressor_matrix[:, 0])


def test_fit_verbose_requires_validation_data():
    model = NARXNN(
        net=NARX(),
        basis_function=Polynomial(degree=1),
        verbose=True,
        epochs=1,
    )
    assert_raises(ValueError, model.fit, X=X_train[:50], y=y_train[:50])


def test_predict_requires_defined_network():
    model = NARXNN(basis_function=Polynomial(degree=1))
    assert_raises(ValueError, model.predict, X=X_test, y=y_test)


def test_narmax_predict_requires_forecast_horizon_when_no_input():
    model = NARXNN(basis_function=Polynomial(degree=1))
    model.max_lag = 2
    model.n_inputs = 1
    model.final_model = np.array([[1001]])
    model.theta = np.array([[0.1]])
    model._scalar_forward = lambda arr: float(np.sum(arr))
    y_initial = np.ones((model.max_lag, 1))
    with pytest.raises(ValueError, match="forecast_horizon cannot be None"):
        model._narmax_predict(x=None, y_initial=y_initial, forecast_horizon=None)


def test_fit_raise():
    assert_raises(
        ValueError,
        NARXNN,
        basis_function=Polynomial(degree=1),
        model_type="NARARMAX",
    )


def test_fit_raise_y():
    model = NARXNN(basis_function=Polynomial(degree=2))
    assert_raises(ValueError, model.fit, X=X_train, y=None)


def test_fit_lag_nar():
    basis_function = Polynomial(degree=1)
    regressors = regressor_code(
        X=X_train,
        xlag=2,
        ylag=2,
        model_type="NAR",
        model_representation="neural_network",
        basis_function=basis_function,
    )
    n_features = regressors.shape[0]

    class NARX(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(n_features, 30)
            self.lin2 = nn.Linear(30, 30)
            self.lin3 = nn.Linear(30, 1)
            self.tanh = nn.Tanh()

        def forward(self, xb):
            z = self.lin(xb)
            z = self.tanh(z)
            z = self.lin2(z)
            z = self.tanh(z)
            z = self.lin3(z)
            return z

    model = NARXNN(
        net=NARX(),
        ylag=2,
        xlag=2,
        basis_function=basis_function,
        model_type="NAR",
        loss_func="mse_loss",
        optimizer="Adam",
        epochs=10,
        verbose=False,
        random_state=0,
        optim_params={
            "betas": (0.9, 0.999),
            "eps": 1e-05,
        },  # optional parameters of the optimizer
    )

    model.fit(X=X_train, y=y_train)
    assert_equal(model.max_lag, 2)


def test_fit_lag_nfir():
    basis_function = Polynomial(degree=1)
    regressors = regressor_code(
        X=X_train,
        xlag=2,
        ylag=2,
        model_type="NFIR",
        model_representation="neural_network",
        basis_function=basis_function,
    )
    n_features = regressors.shape[0]

    class NARX(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(n_features, 30)
            self.lin2 = nn.Linear(30, 30)
            self.lin3 = nn.Linear(30, 1)
            self.tanh = nn.Tanh()

        def forward(self, xb):
            z = self.lin(xb)
            z = self.tanh(z)
            z = self.lin2(z)
            z = self.tanh(z)
            z = self.lin3(z)
            return z

    model = NARXNN(
        net=NARX(),
        ylag=2,
        xlag=2,
        basis_function=basis_function,
        model_type="NFIR",
        loss_func="mse_loss",
        optimizer="Adam",
        epochs=10,
        verbose=False,
        random_state=0,
        optim_params={
            "betas": (0.9, 0.999),
            "eps": 1e-05,
        },  # optional parameters of the optimizer
    )

    model.fit(X=X_train, y=y_train)
    model.fit(X=X_train, y=y_train)
    assert_equal(model.max_lag, 2)


def test_fit_lag_narmax():
    basis_function = Polynomial(degree=1)
    regressors = regressor_code(
        X=X_train,
        xlag=2,
        ylag=2,
        model_type="NARMAX",
        model_representation="neural_network",
        basis_function=basis_function,
    )
    n_features = regressors.shape[0]

    class NARX(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(n_features, 30)
            self.lin2 = nn.Linear(30, 30)
            self.lin3 = nn.Linear(30, 1)
            self.tanh = nn.Tanh()

        def forward(self, xb):
            z = self.lin(xb)
            z = self.tanh(z)
            z = self.lin2(z)
            z = self.tanh(z)
            z = self.lin3(z)
            return z

    model = NARXNN(
        net=NARX(),
        ylag=2,
        xlag=2,
        basis_function=basis_function,
        model_type="NARMAX",
        loss_func="mse_loss",
        optimizer="Adam",
        epochs=10,
        verbose=False,
        random_state=0,
        optim_params={
            "betas": (0.9, 0.999),
            "eps": 1e-05,
        },  # optional parameters of the optimizer
    )

    model.fit(X=X_train, y=y_train)
    assert_equal(model.max_lag, 2)


def test_fit_lag_narmax_fourier():
    basis_function = Fourier(degree=1)
    regressors = regressor_code(
        X=X_train,
        xlag=2,
        ylag=2,
        model_type="NARMAX",
        model_representation="neural_network",
        basis_function=basis_function,
    )
    n_features = regressors.shape[0]

    class NARX(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(n_features, 30)
            self.lin2 = nn.Linear(30, 30)
            self.lin3 = nn.Linear(30, 1)
            self.tanh = nn.Tanh()

        def forward(self, xb):
            z = self.lin(xb)
            z = self.tanh(z)
            z = self.lin2(z)
            z = self.tanh(z)
            z = self.lin3(z)
            return z

    model = NARXNN(
        net=NARX(),
        ylag=2,
        xlag=2,
        epochs=10,
        basis_function=basis_function,
        random_state=0,
        optim_params={
            "betas": (0.9, 0.999),
            "eps": 1e-05,
        },  # optional parameters of the optimizer
    )

    model.fit(X=X_train, y=y_train)
    assert_equal(model.max_lag, 2)


def test_model_predict():
    basis_function = Polynomial(degree=2)
    regressors = regressor_code(
        X=X_train,
        xlag=2,
        ylag=2,
        model_type="NARMAX",
        model_representation="neural_network",
        basis_function=basis_function,
    )
    n_features = regressors.shape[0]

    class NARX(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(n_features, 30)
            self.lin2 = nn.Linear(30, 30)
            self.lin3 = nn.Linear(30, 1)
            self.tanh = nn.Tanh()

        def forward(self, xb):
            z = self.lin(xb)
            z = self.tanh(z)
            z = self.lin2(z)
            z = self.tanh(z)
            z = self.lin3(z)
            return z

    model = NARXNN(
        net=NARX(),
        ylag=2,
        xlag=2,
        epochs=2000,
        basis_function=basis_function,
        random_state=0,
        optim_params={
            "betas": (0.9, 0.999),
            "eps": 1e-05,
        },  # optional parameters of the optimizer
    )

    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test)
    assert_almost_equal(yhat.mean(), y_test.mean(), decimal=2)


def test_steps_1():
    basis_function = Polynomial(degree=1)
    regressors = regressor_code(
        X=X_train,
        xlag=2,
        ylag=2,
        model_type="NARMAX",
        model_representation="neural_network",
        basis_function=basis_function,
    )
    n_features = regressors.shape[0]

    class NARX(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(n_features, 30)
            self.lin2 = nn.Linear(30, 30)
            self.lin3 = nn.Linear(30, 1)
            self.tanh = nn.Tanh()

        def forward(self, xb):
            z = self.lin(xb)
            z = self.tanh(z)
            z = self.lin2(z)
            z = self.tanh(z)
            z = self.lin3(z)
            return z

    model = NARXNN(
        net=NARX(),
        ylag=2,
        xlag=2,
        epochs=2000,
        basis_function=basis_function,
        random_state=0,
        optim_params={
            "betas": (0.9, 0.999),
            "eps": 1e-05,
        },  # optional parameters of the optimizer
    )

    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test, steps_ahead=1)
    mean_diff = abs(yhat.mean() - y_test.mean())
    # The learned mean oscillates slightly across torch/numpy releases,
    # but remains below this threshold when the network converges.
    assert mean_diff < 3e-2


def test_steps_3():
    basis_function = Polynomial(degree=1)
    regressors = regressor_code(
        X=X_train,
        xlag=2,
        ylag=2,
        model_type="NARMAX",
        model_representation="neural_network",
        basis_function=basis_function,
    )
    n_features = regressors.shape[0]

    class NARX(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(n_features, 30)
            self.lin2 = nn.Linear(30, 30)
            self.lin3 = nn.Linear(30, 1)
            self.tanh = nn.Tanh()

        def forward(self, xb):
            z = self.lin(xb)
            z = self.tanh(z)
            z = self.lin2(z)
            z = self.tanh(z)
            z = self.lin3(z)
            return z

    model = NARXNN(
        net=NARX(),
        ylag=2,
        xlag=2,
        epochs=2000,
        basis_function=basis_function,
        random_state=0,
        optim_params={
            "betas": (0.9, 0.999),
            "eps": 1e-05,
        },  # optional parameters of the optimizer
    )

    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test, steps_ahead=3)
    mean_diff = abs(yhat.mean() - y_test.mean())
    assert mean_diff < 3e-2


def test_raise_batch_size():
    assert_raises(
        ValueError, NARXNN, batch_size=0.3, basis_function=Polynomial(degree=2)
    )


def test_raise_epochs():
    assert_raises(ValueError, NARXNN, epochs=0.3, basis_function=Polynomial(degree=2))


def test_raise_train_percentage():
    assert_raises(
        ValueError, NARXNN, train_percentage=-1, basis_function=Polynomial(degree=2)
    )


def test_raise_verbose():
    assert_raises(TypeError, NARXNN, verbose=None, basis_function=Polynomial(degree=2))


def test_raise_device():
    assert_raises(ValueError, NARXNN, device="CPU", basis_function=Polynomial(degree=2))


def test_model_predict_fourier():
    basis_function = Fourier(degree=1)
    regressors = regressor_code(
        X=X_train,
        xlag=2,
        ylag=2,
        model_type="NARMAX",
        model_representation="neural_network",
        basis_function=basis_function,
    )
    n_features = regressors.shape[0]

    class NARX(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(n_features, 30)
            self.lin2 = nn.Linear(30, 30)
            self.lin3 = nn.Linear(30, 1)
            self.tanh = nn.Tanh()

        def forward(self, xb):
            z = self.lin(xb)
            z = self.tanh(z)
            z = self.lin2(z)
            z = self.tanh(z)
            z = self.lin3(z)
            return z

    model = NARXNN(
        net=NARX(),
        ylag=2,
        xlag=2,
        epochs=2000,
        learning_rate=0.001,
        basis_function=basis_function,
        random_state=0,
        optim_params={
            "betas": (0.9, 0.999),
            "eps": 1e-05,
        },  # optional parameters of the optimizer
    )

    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test)
    assert_almost_equal(yhat.mean(), y_test.mean(), decimal=2)


def test_steps_1_fourier():
    basis_function = Fourier(degree=1)
    regressors = regressor_code(
        X=X_train,
        xlag=2,
        ylag=2,
        model_type="NARMAX",
        model_representation="neural_network",
        basis_function=basis_function,
    )
    n_features = regressors.shape[0]

    class NARX(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(n_features, 30)
            self.lin2 = nn.Linear(30, 30)
            self.lin3 = nn.Linear(30, 1)
            self.tanh = nn.Tanh()

        def forward(self, xb):
            z = self.lin(xb)
            z = self.tanh(z)
            z = self.lin2(z)
            z = self.tanh(z)
            z = self.lin3(z)
            return z

    model = NARXNN(
        net=NARX(),
        ylag=2,
        xlag=2,
        epochs=1000,
        learning_rate=0.001,
        basis_function=basis_function,
        random_state=0,
        optim_params={
            "betas": (0.9, 0.999),
            "eps": 1e-03,
        },  # optional parameters of the optimizer
    )

    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test, steps_ahead=1)
    assert_almost_equal(yhat.mean(), y_test.mean(), decimal=2)


def test_steps_3_fourier():
    basis_function = Fourier(degree=1)
    regressors = regressor_code(
        X=X_train,
        xlag=2,
        ylag=2,
        model_type="NARMAX",
        model_representation="neural_network",
        basis_function=basis_function,
    )
    n_features = regressors.shape[0]

    class NARX(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(n_features, 30)
            self.lin2 = nn.Linear(30, 30)
            self.lin3 = nn.Linear(30, 1)
            self.tanh = nn.Tanh()

        def forward(self, xb):
            z = self.lin(xb)
            z = self.tanh(z)
            z = self.lin2(z)
            z = self.tanh(z)
            z = self.lin3(z)
            return z

    model = NARXNN(
        net=NARX(),
        ylag=2,
        xlag=2,
        epochs=2000,
        basis_function=basis_function,
        random_state=0,
        optim_params={
            "betas": (0.9, 0.999),
            "eps": 1e-05,
        },  # optional parameters of the optimizer
    )

    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test, steps_ahead=3)
    mean_diff = abs(yhat.mean() - y_test.mean())
    assert mean_diff < 3e-2


def test_check_cuda_cpu():
    """Test if _check_cuda correctly returns 'cpu' when requested."""
    assert _check_cuda("cpu") == torch.device("cpu")


def test_check_cuda_invalid():
    """Test if _check_cuda raises a ValueError for an invalid device."""
    with pytest.raises(ValueError, match="device must be 'cpu' or 'cuda'"):
        _check_cuda("invalid_device")


def test_check_cuda_available():
    """Test if _check_cuda returns 'cuda' when CUDA is available."""
    with patch("torch.cuda.is_available", return_value=True):
        assert _check_cuda("cuda") == torch.device("cuda")


def test_check_cuda_unavailable():
    """Test if _check_cuda falls back to 'cpu' when CUDA is unavailable."""
    with (
        patch("torch.cuda.is_available", return_value=False),
        pytest.warns(UserWarning, match="No CUDA available"),
    ):
        assert _check_cuda("cuda") == torch.device("cpu")


def test_fit_verbose_raises_error():
    """Fit raises ValueError if verbose=True but no validation data is provided."""
    model = NARXNN(verbose=True)  # Assuming 'verbose' is an argument to the class

    X_train = np.random.rand(10, 1)
    y_train = np.random.rand(10, 1)

    with pytest.raises(
        ValueError, match="X_test and y_test cannot be None if you set verbose=True"
    ):
        model.fit(X=X_train, y=y_train, X_test=None, y_test=None)


def test_fit_verbose_false_does_not_raise():
    """Fit does not raise an error when verbose=False and validation data is missing."""
    basis_function = Polynomial(degree=1)
    regressors = regressor_code(
        X=X_train,
        xlag=2,
        ylag=2,
        model_type="NARMAX",
        model_representation="neural_network",
        basis_function=basis_function,
    )
    n_features = regressors.shape[0]

    class NARX(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(n_features, 30)
            self.lin2 = nn.Linear(30, 30)
            self.lin3 = nn.Linear(30, 1)
            self.tanh = nn.Tanh()

        def forward(self, xb):
            z = self.lin(xb)
            z = self.tanh(z)
            z = self.lin2(z)
            z = self.tanh(z)
            z = self.lin3(z)
            return z

    model = NARXNN(
        net=NARX(),
        ylag=2,
        xlag=2,
        epochs=2,
        basis_function=basis_function,
        random_state=0,
        optim_params={
            "betas": (0.9, 0.999),
            "eps": 1e-05,
        },  # optional parameters of the optimizer
    )
    model.fit(
        X=X_train[:30].reshape(-1, 1),
        y=y_train[:30].reshape(-1, 1),
        X_test=None,
        y_test=None,
    )


def test_nfir():
    basis_function = Polynomial(degree=1)
    regressors = regressor_code(
        X=X_train,
        xlag=2,
        model_type="NFIR",
        model_representation="neural_network",
        basis_function=basis_function,
    )
    n_features = regressors.shape[0]

    class NARX(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(n_features, 30)
            self.lin2 = nn.Linear(30, 30)
            self.lin3 = nn.Linear(30, 1)
            self.tanh = nn.Tanh()

        def forward(self, xb):
            z = self.lin(xb)
            z = self.tanh(z)
            z = self.lin2(z)
            z = self.tanh(z)
            z = self.lin3(z)
            return z

    model = NARXNN(
        net=NARX(),
        ylag=2,
        xlag=2,
        epochs=2,
        basis_function=basis_function,
        model_type="NFIR",
        random_state=0,
        optim_params={
            "betas": (0.9, 0.999),
            "eps": 1e-05,
        },  # optional parameters of the optimizer
    )

    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test, steps_ahead=1)
    assert isinstance(yhat, np.ndarray)


def test_nfir_predict_output_shape():
    """Test that _nfir_predict returns output of expected shape."""
    basis_function = Polynomial(degree=1)
    regressors = regressor_code(
        X=X_train,
        xlag=2,
        model_type="NFIR",
        model_representation="neural_network",
        basis_function=basis_function,
    )
    n_features = regressors.shape[0]

    class NARX(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(n_features, 30)
            self.lin2 = nn.Linear(30, 30)
            self.lin3 = nn.Linear(30, 1)
            self.tanh = nn.Tanh()

        def forward(self, xb):
            z = self.lin(xb)
            z = self.tanh(z)
            z = self.lin2(z)
            z = self.tanh(z)
            z = self.lin3(z)
            return z

    model = NARXNN(
        net=NARX(),
        ylag=2,
        xlag=2,
        epochs=2,
        basis_function=basis_function,
        model_type="NFIR",
        random_state=0,
        optim_params={
            "betas": (0.9, 0.999),
            "eps": 1e-05,
        },  # optional parameters of the optimizer
    )

    model.fit(
        X=X_train[:30].reshape(-1, 1),
        y=y_train[:30].reshape(-1, 1),
    )
    y_output = model._nfir_predict(X_test, y_test)

    assert y_output.shape == y_test.shape, "Output shape mismatch."


def test_nfir_predict_initial_values():
    """Test that the first max_lag values in the output match y_initial."""
    basis_function = Polynomial(degree=1)
    regressors = regressor_code(
        X=X_train,
        xlag=2,
        model_type="NFIR",
        model_representation="neural_network",
        basis_function=basis_function,
    )
    n_features = regressors.shape[0]

    class NARX(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(n_features, 30)
            self.lin2 = nn.Linear(30, 30)
            self.lin3 = nn.Linear(30, 1)
            self.tanh = nn.Tanh()

        def forward(self, xb):
            z = self.lin(xb)
            z = self.tanh(z)
            z = self.lin2(z)
            z = self.tanh(z)
            z = self.lin3(z)
            return z

    model = NARXNN(
        net=NARX(),
        ylag=2,
        xlag=2,
        epochs=2,
        basis_function=basis_function,
        model_type="NFIR",
        random_state=0,
        optim_params={
            "betas": (0.9, 0.999),
            "eps": 1e-05,
        },
    )
    model.fit(
        X=X_train[:30].reshape(-1, 1),
        y=y_train[:30].reshape(-1, 1),
    )

    y_output = model._nfir_predict(X_test, y_test)

    np.testing.assert_almost_equal(
        y_output[: model.max_lag],
        y_test[: model.max_lag],
        decimal=5,
        err_msg="Initial values do not match y_initial.",
    )


def test_basis_n_step():
    basis_function = Fourier(degree=1)
    regressors = regressor_code(
        X=X_train,
        xlag=2,
        ylag=2,
        model_representation="neural_network",
        basis_function=basis_function,
    )
    n_features = regressors.shape[0]

    class NARX(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(n_features, 30)
            self.lin2 = nn.Linear(30, 30)
            self.lin3 = nn.Linear(30, 1)
            self.tanh = nn.Tanh()

        def forward(self, xb):
            z = self.lin(xb)
            z = self.tanh(z)
            z = self.lin2(z)
            z = self.tanh(z)
            z = self.lin3(z)
            return z

    model = NARXNN(
        net=NARX(),
        ylag=2,
        xlag=2,
        epochs=2,
        basis_function=basis_function,
        random_state=0,
        optim_params={
            "betas": (0.9, 0.999),
            "eps": 1e-05,
        },  # optional parameters of the optimizer
    )

    model.fit(X=X_train, y=y_train)
    yhat = model._basis_function_n_step_prediction(
        x=X_test, y=y_test, steps_ahead=2, forecast_horizon=1
    )
    assert isinstance(yhat, np.ndarray)


def test_basis_n_step_shape():
    """Test that _nfir_predict returns output of expected shape."""
    basis_function = Fourier(degree=1)
    regressors = regressor_code(
        X=X_train,
        xlag=2,
        ylag=2,
        model_representation="neural_network",
        basis_function=basis_function,
    )
    n_features = regressors.shape[0]

    class NARX(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(n_features, 30)
            self.lin2 = nn.Linear(30, 30)
            self.lin3 = nn.Linear(30, 1)
            self.tanh = nn.Tanh()

        def forward(self, xb):
            z = self.lin(xb)
            z = self.tanh(z)
            z = self.lin2(z)
            z = self.tanh(z)
            z = self.lin3(z)
            return z

    model = NARXNN(
        net=NARX(),
        ylag=2,
        xlag=2,
        epochs=2,
        basis_function=basis_function,
        random_state=0,
        optim_params={
            "betas": (0.9, 0.999),
            "eps": 1e-05,
        },
    )

    model.fit(
        X=X_train[:30].reshape(-1, 1),
        y=y_train[:30].reshape(-1, 1),
    )
    y_output = model._basis_function_n_step_prediction(
        X_test, y_test, steps_ahead=2, forecast_horizon=1
    )

    assert y_output.shape == y_test.shape, "Output shape mismatch."


def test_basis_n_step_initial_values():
    """Test that the first max_lag values in the output match y_initial."""
    basis_function = Fourier(degree=1)
    regressors = regressor_code(
        X=X_train,
        xlag=2,
        ylag=2,
        model_representation="neural_network",
        basis_function=basis_function,
    )
    n_features = regressors.shape[0]

    class NARX(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(n_features, 30)
            self.lin2 = nn.Linear(30, 30)
            self.lin3 = nn.Linear(30, 1)
            self.tanh = nn.Tanh()

        def forward(self, xb):
            z = self.lin(xb)
            z = self.tanh(z)
            z = self.lin2(z)
            z = self.tanh(z)
            z = self.lin3(z)
            return z

    model = NARXNN(
        net=NARX(),
        ylag=2,
        xlag=2,
        epochs=2,
        basis_function=basis_function,
        random_state=0,
        optim_params={
            "betas": (0.9, 0.999),
            "eps": 1e-05,
        },
    )
    model.fit(
        X=X_train[:30].reshape(-1, 1),
        y=y_train[:30].reshape(-1, 1),
    )

    y_output = model._basis_function_n_step_prediction(
        X_test, y_test, steps_ahead=2, forecast_horizon=1
    )

    np.testing.assert_almost_equal(
        y_output[: model.max_lag],
        y_test[: model.max_lag],
        decimal=5,
        err_msg="Initial values do not match y_initial.",
    )
