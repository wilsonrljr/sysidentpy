"""Build Polynomial NARMAX Models."""

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause

import logging
import sys
import warnings
from collections.abc import Mapping
from copy import deepcopy
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch import optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None  # type: ignore[assignment]

from .._lib._array_api import get_namespace, _require_numpy_namespace
from ..narmax_base import BaseMSS
from ..basis_function import Polynomial
from sysidentpy.utils.information_matrix import (
    build_output_matrix,
    build_input_matrix,
    build_input_output_matrix,
    build_lagged_matrix,
)
from ..utils.check_arrays import num_features

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)


def _check_cuda(device):
    if device not in ["cpu", "cuda"]:
        raise ValueError(f"device must be 'cpu' or 'cuda'. Got {device}")

    if device == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    warnings.warn(
        "No CUDA available. We set the device as CPU",
        stacklevel=2,
    )

    return torch.device("cpu")


def convert_to_tensor(reg_matrix, y):
    """Return the lagged matrix and the y values given the maximum lags.

    Based on Pytorch official docs:
    https://pytorch.org/tutorials/beginner/nn_tutorial.html

    Parameters
    ----------
    reg_matrix : ndarray of floats
        The information matrix of the model.
    y : ndarray of floats
        The output data

    Returns
    -------
    Tensor: tensor
        tensors that have the same size of the first dimension.

    """
    reg_matrix = np.ascontiguousarray(np.asarray(reg_matrix, dtype=np.float32))
    y = np.ascontiguousarray(np.asarray(y, dtype=np.float32))
    return TensorDataset(torch.from_numpy(reg_matrix), torch.from_numpy(y))


class NARXNN(BaseMSS):
    r"""NARX Neural Network model built on top of Pytorch.

    Neural networks are models composed of interconnected layers of nodes
    (neurons) designed for tasks like classification and regression. Each neuron
    is a basic unit within these networks. Mathematically, a neuron is
    represented by a function $f$ that takes an input vector
    $\mathbf{x} = [x_1, x_2, \ldots, x_n]$ and generates an output $y$.
    This function usually involves a weighted sum of the inputs, an optional
    bias term $b$, and an activation function $\phi$:

    $$
    y = \phi \left( \sum_{i=1}^{n} w_i x_i + b \right)
    \tag{2.31}
    $$

    where $\mathbf{w} = [w_1, w_2, \ldots, w_n]$ are the weights associated with the
    inputs. The activation function $\phi$ introduces nonlinearity into the model,
    allowing the network to learn complex patterns.

    Currently we support a Series-Parallel (open-loop) Feedforward Network training
    process, which make the training process easier, and we convert the
    NARX network from Series-Parallel to the Parallel (closed-loop) configuration for
    prediction.

    Parameters
    ----------
    ylag : int, default=2
        The maximum lag of the output.
    xlag : int, default=2
        The maximum lag of the input.
    basis_function: Polynomial or Fourier basis functions
        Defines which basis function will be used in the model.
    model_type: str, default="NARMAX"
        The user can choose "NARMAX", "NAR" and "NFIR" models
    batch_size : int, default=100
        Size of mini-batches of data for stochastic optimizers
    shuffle_batches : bool, default=False
        Whether to shuffle mini-batches during training.
    learning_rate : float, default=0.01
        Learning rate schedule for weight updates
    epochs : int, default=100
        Number of training epochs
    loss_func : str, default='mse_loss'
        Select the loss function available in torch.nn.functional
    optimizer : str, default='SGD'
        The solver for weight optimization
    optim_params : dict, default=None
        Optional parameters for the optimizer
    net : default=None
        The defined network using nn.Module
    verbose : bool, default=False
        Show the training and validation loss at each iteration
    random_state : int or None, default=None
        Controls the seeding used to reset the neural network parameters before
        training. When provided, the model weights are reinitialized with the
        same seed at every call to ``fit`` to guarantee deterministic behaviour.
    early_stopping : bool, default=False
        Whether to stop training when the validation loss stops improving. Validation
        data must be provided through ``X_test`` and ``y_test`` when calling ``fit``.
    patience : int, default=10
        Number of consecutive epochs without sufficient validation loss improvement
        before training is stopped. Only used when ``early_stopping=True``.
    min_delta : float, default=0.0
        Minimum decrease in validation loss required to qualify as an improvement.

    Examples
    --------
    >>> from torch import nn
    >>> import numpy as np
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> from sysidentpy.metrics import mean_squared_error
    >>> from sysidentpy.utils.generate_data import get_siso_data
    >>> from sysidentpy.neural_network import NARXNN
    >>> from sysidentpy.basis_function import Polynomial
    >>> from sysidentpy.utils.generate_data import get_siso_data
    >>> basis_function = Polynomial(degree=2)
    >>> x_train, x_valid, y_train, y_valid = get_siso_data(
    ...     n=1000,
    ...     colored_noise=False,
    ...     sigma=0.01,
    ...     train_percentage=80
    ... )
    >>> narx_nn = NARXNN(
    ...     ylag=2,
    ...     xlag=2,
    ...     basis_function=basis_function,
    ...     model_type="NARMAX",
    ...     loss_func='mse_loss',
    ...     optimizer='Adam',
    ...     epochs=200,
    ...     verbose=False,
    ...     optim_params={'betas': (0.9, 0.999), 'eps': 1e-05} # for the optimizer
    ... )
    >>> class Net(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.lin = nn.Linear(4, 10)
    ...         self.lin2 = nn.Linear(10, 10)
    ...         self.lin3 = nn.Linear(10, 1)
    ...         self.tanh = nn.Tanh()
    >>>
    ...     def forward(self, xb):
    ...         z = self.lin(xb)
    ...         z = self.tanh(z)
    ...         z = self.lin2(z)
    ...         z = self.tanh(z)
    ...         z = self.lin3(z)
    ...         return z
    >>>
    >>> narx_nn.net = Net()
    >>> neural_narx.fit(x=x_train, y=y_train)
    >>> yhat = neural_narx.predict(x=x_valid, y=y_valid)
    >>> print(mean_squared_error(y_valid, yhat))
    0.000131

    References
    ----------
    - Manuscript: Orthogonal least squares methods and their application
       to non-linear system identification
       <https://eprints.soton.ac.uk/251147/1/778742007_content.pdf>`_

    """

    def __init__(
        self,
        *,
        ylag=1,
        xlag=1,
        model_type="NARMAX",
        basis_function=Polynomial(),
        batch_size=100,
        learning_rate=0.01,
        epochs=200,
        loss_func="mse_loss",
        optimizer="Adam",
        net=None,
        train_percentage=80,
        verbose=False,
        optim_params=None,
        device="cpu",
        shuffle_batches=False,
        random_state: Optional[int] = None,
        early_stopping=False,
        patience=10,
        min_delta=0.0,
    ):
        if torch is None:
            raise ImportError(
                "PyTorch is required for NARXNN. "
                "Install it with: pip install sysidentpy[all]"
            )

        self.ylag = ylag
        self.xlag = xlag
        self.basis_function = basis_function
        self.model_type = model_type
        self.non_degree = basis_function.degree
        self.max_lag = self._get_max_lag()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_func_name = loss_func
        self.loss_func = None
        self.optimizer_name = optimizer
        self.optimizer = optimizer
        self.optimizer_cls = None
        self.net = net
        self.train_percentage = train_percentage
        self.verbose = verbose
        self.shuffle_batches = shuffle_batches
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        if optim_params is None:
            self.optim_params = {}
        elif isinstance(optim_params, Mapping):
            self.optim_params = dict(optim_params)
        else:
            self.optim_params = optim_params
        self.device = _check_cuda(device)
        self.regressor_code = None
        self.train_loss = None
        self.val_loss = None
        self.ensemble = None
        self.n_inputs = None
        self.final_model = None
        self._validate_params()
        self.loss_func = getattr(F, self.loss_func_name)
        self.optimizer_cls = getattr(optim, self.optimizer_name)

    def _validate_params(self):
        """Validate input params."""
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError(
                f"bacth_size must be integer and > zero. Got {self.batch_size}"
            )

        if not isinstance(self.epochs, int) or self.epochs < 1:
            raise ValueError(f"epochs must be integer and > zero. Got {self.epochs}")

        if (
            not isinstance(self.train_percentage, int)
            or self.train_percentage <= 0
            or self.train_percentage > 100
        ):
            raise ValueError(
                "train_percentage must be an integer between 1 and 100. "
                f"Got {self.train_percentage}"
            )

        if not isinstance(self.verbose, bool):
            raise TypeError(f"verbose must be False or True. Got {self.verbose}")

        if not isinstance(self.shuffle_batches, bool):
            raise TypeError(
                f"shuffle_batches must be False or True. Got {self.shuffle_batches}"
            )

        if not isinstance(self.early_stopping, bool):
            raise TypeError(
                f"early_stopping must be False or True. Got {self.early_stopping}"
            )

        if (
            isinstance(self.patience, bool)
            or not isinstance(self.patience, (int, np.integer))
            or self.patience < 1
        ):
            raise ValueError(
                f"patience must be integer and > zero. Got {self.patience}"
            )

        if (
            isinstance(self.min_delta, bool)
            or not isinstance(self.min_delta, (int, float, np.integer, np.floating))
            or not np.isfinite(self.min_delta)
            or self.min_delta < 0
        ):
            raise ValueError(
                f"min_delta must be a finite number and >= zero. Got {self.min_delta}"
            )

        self.ylag = self._sanitize_lag(self.ylag, "ylag")
        self.xlag = self._sanitize_lag(self.xlag, "xlag")

        if self.model_type not in ["NARMAX", "NAR", "NFIR"]:
            raise ValueError(
                f"model_type must be NARMAX, NAR or NFIR. Got {self.model_type}"
            )

        if not isinstance(self.optim_params, dict):
            raise TypeError(
                "optim_params must be a mapping (e.g. dict). "
                f"Got {type(self.optim_params).__name__}"
            )

        if not isinstance(self.loss_func_name, str):
            raise TypeError(
                f"loss_func must be provided as string. Got {self.loss_func_name}"
            )
        if not hasattr(F, self.loss_func_name):
            raise ValueError(
                f"loss_func {self.loss_func_name} not available in torch.nn.functional"
            )

        if not isinstance(self.optimizer_name, str):
            raise TypeError(
                f"optimizer must be provided as string. Got {self.optimizer_name}"
            )
        if not hasattr(optim, self.optimizer_name):
            raise ValueError(
                f"optimizer {self.optimizer_name} not available in torch.optim"
            )

    def _sanitize_lag(self, value, name):
        if isinstance(value, int):
            if value < 1:
                raise ValueError(f"{name} must be >= 1. Got {value}")
            return value

        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                raise ValueError(f"{name} list cannot be empty")
            sanitized = []
            for idx, lag in enumerate(value):
                if not isinstance(lag, (int, np.integer)):
                    raise ValueError(
                        f"All elements of {name} must be integers. "
                        f"Found {type(lag).__name__} at position {idx}"
                    )
                if lag < 1:
                    raise ValueError(
                        f"All elements of {name} must be >= 1. "
                        f"Found {lag} at position {idx}"
                    )
                sanitized.append(int(lag))
            return sanitized

        raise ValueError(
            f"{name} must be an int or a sequence of ints. Got {type(value).__name__}"
        )

    def _as_float_array(self, array):
        return np.ascontiguousarray(np.asarray(array, dtype=np.float32))

    def _forward_numpy(self, array):
        tensor = torch.from_numpy(self._as_float_array(array))
        if self.device.type != "cpu":
            tensor = tensor.to(self.device, non_blocking=True)
        return self.net(tensor).detach().cpu().numpy()

    def _scalar_forward(self, array):
        return float(self._forward_numpy(array).reshape(-1)[0])

    def _prepare_regressor_matrix(self, reg_matrix, n_inputs):
        """Align neural input columns with their canonical regressor codes."""
        regressor_code = self._regressor_space_for_feature_matrix(
            n_inputs, n_features=reg_matrix.shape[1]
        )
        if not isinstance(self.basis_function, Polynomial):
            return reg_matrix, regressor_code

        bias_indices = np.flatnonzero(np.all(regressor_code == 0, axis=1))
        if bias_indices.size == 0:
            return reg_matrix, regressor_code

        # Native Polynomial layouts contain one bias. Compatibility fallbacks for
        # custom subclasses can repeat approximate codes when extending the layout;
        # removing every zero code would silently discard those custom features.
        keep_columns = np.ones(regressor_code.shape[0], dtype=bool)
        keep_columns[bias_indices[0]] = False
        return reg_matrix[:, keep_columns], regressor_code[keep_columns, :]

    def define_opt(self):
        """Define the optimizer using the user parameters."""
        return self.optimizer_cls(
            self.net.parameters(), lr=self.learning_rate, **self.optim_params
        )

    def _seed_torch_generators(self):
        if self.random_state is None:
            return
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    def _reset_network_parameters(self):
        if self.net is None:
            raise ValueError("The neural network must be defined before training")

        def _reset_fn(module):
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        self.net.apply(_reset_fn)

    def loss_batch(self, x, y, opt=None):
        """Compute the loss for one batch.

        Parameters
        ----------
        x : ndarray of floats
            The regressor matrix.
        y : ndarray of floats
            The output data.
        opt: Torch optimizer
            Chosen by the user.

        Returns
        -------
        loss : float
            The loss of one batch.

        """
        loss = self.loss_func(self.net(x), y)

        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()

        return loss.item(), len(x)

    def split_data(self, x, y):
        """Return the lagged matrix and the y values given the maximum lags.

        Parameters
        ----------
        x : ndarray of floats
            The input data.
        y : ndarray of floats
            The output data.

        Returns
        -------
        y : ndarray of floats
            The y values considering the lags.
        reg_matrix : ndarray of floats
            The information matrix of the model.

        """
        if y is None:
            raise ValueError("y cannot be None")

        self.max_lag = self._get_max_lag()
        lagged_data = build_lagged_matrix(x, y, self.xlag, self.ylag, self.model_type)

        reg_matrix = self.basis_function.fit(
            lagged_data,
            self.max_lag,
            self.ylag,
            self.xlag,
            self.model_type,
            predefined_regressors=None,
        )

        if x is not None:
            self.n_inputs = num_features(x)
        else:
            self.n_inputs = 1  # only used to create the regressor space base

        reg_matrix, self.regressor_code = self._prepare_regressor_matrix(
            reg_matrix, self.n_inputs
        )

        self.final_model = self.regressor_code.copy()
        reg_matrix = np.atleast_1d(reg_matrix).astype(np.float32)

        y = np.atleast_1d(y[self.max_lag :]).astype(np.float32)
        return reg_matrix, y

    def get_data(self, train_ds, *, shuffle=None):
        """Return the lagged matrix and the y values given the maximum lags.

        Based on Pytorch official docs:
        https://pytorch.org/tutorials/beginner/nn_tutorial.html

        Parameters
        ----------
        train_ds: tensor
            Tensors that have the same size of the first dimension.

        Returns
        -------
        Dataloader: dataloader
            tensors that have the same size of the first dimension.

        """
        pin_memory = False if self.device.type == "cpu" else True
        if shuffle is None:
            shuffle = self.shuffle_batches
        return DataLoader(
            train_ds,
            batch_size=self.batch_size,
            pin_memory=pin_memory,
            shuffle=shuffle,
        )

    def data_transform(self, x, y, *, shuffle=None):
        """Return the data transformed in tensors using Dataloader.

        Parameters
        ----------
        x : ndarray of floats
            The input data.
        y : ndarray of floats
            The output data.

        Returns
        -------
        Tensors : Dataloader

        """
        if y is None:
            raise ValueError("y cannot be None")

        self.max_lag = self._get_max_lag()
        if len(y) <= self.max_lag:
            raise ValueError(
                "y must contain more samples than the maximum lag. "
                f"Got {len(y)} samples and max_lag={self.max_lag}"
            )

        x_train, y_train = self.split_data(x, y)
        train_ds = convert_to_tensor(x_train, y_train)
        train_dl = self.get_data(train_ds, shuffle=shuffle)
        return train_dl

    def fit(self, *, X=None, y=None, X_test=None, y_test=None):
        """Train a NARX Neural Network model.

        This is an training pipeline that allows a friendly usage
        by the user. The training pipeline was based on
        https://pytorch.org/tutorials/beginner/nn_tutorial.html

        Parameters
        ----------
        X : ndarray of floats
            The input data to be used in the training process.
        y : ndarray of floats
            The output data to be used in the training process.
        X_test : ndarray of floats
            The input data to be used in the validation process. Required when
            ``verbose=True`` or ``early_stopping=True``.
        y_test : ndarray of floats
            The output data to be used in the validation process. Required when
            ``verbose=True`` or ``early_stopping=True``.

        Returns
        -------
        net : nn.Module
            The model fitted.
        train_loss: ndarrays of floats
            The training loss of each batch
        val_loss: ndarrays of floats
            The validation loss of each batch

        """
        monitor_validation = self.verbose or self.early_stopping
        if monitor_validation and (X_test is None or y_test is None):
            if self.early_stopping:
                raise ValueError(
                    "X_test and y_test cannot be None if you set early_stopping=True"
                )
            raise ValueError("X_test and y_test cannot be None if you set verbose=True")

        if self.net is None:
            raise ValueError("The neural network must be defined before training")

        xp = get_namespace(y) if X is None else get_namespace(X, y)
        _require_numpy_namespace(xp, feature="NARXNN", dependency="PyTorch/NumPy")

        if self.random_state is not None:
            self._seed_torch_generators()
            self._reset_network_parameters()

        train_dl = self.data_transform(X, y, shuffle=self.shuffle_batches)
        if monitor_validation:
            valid_dl = self.data_transform(X_test, y_test, shuffle=False)

        opt = self.define_opt()
        self.val_loss = []
        self.train_loss = []
        best_val_loss = float("inf")
        patience_reference_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0
        for epoch in range(self.epochs):
            self.net.train()
            epoch_loss = 0.0
            seen_samples = 0
            for input_data, output_data in train_dl:
                X_batch = input_data.to(self.device, non_blocking=True)
                y_batch = output_data.to(self.device, non_blocking=True)
                batch_loss, batch_size = self.loss_batch(X_batch, y_batch, opt=opt)
                if monitor_validation:
                    epoch_loss += batch_loss * batch_size
                    seen_samples += batch_size

            if monitor_validation:
                train_metric = epoch_loss / max(seen_samples, 1)
                self.train_loss.append(train_metric)

                self.net.eval()
                val_loss = 0.0
                val_count = 0
                with torch.no_grad():
                    for X_val, y_val in valid_dl:
                        loss_val, batch_size = self.loss_batch(
                            X_val.to(self.device, non_blocking=True),
                            y_val.to(self.device, non_blocking=True),
                        )
                        val_loss += loss_val * batch_size
                        val_count += batch_size
                validation_metric = val_loss / max(val_count, 1)
                self.val_loss.append(validation_metric)

                if self.early_stopping:
                    if not np.isfinite(validation_metric):
                        raise ValueError(
                            "Validation loss must be finite when early stopping is "
                            f"enabled. Got {validation_metric}"
                        )
                    if validation_metric < best_val_loss:
                        best_val_loss = validation_metric
                        best_state = deepcopy(self.net.state_dict())

                    if validation_metric < patience_reference_loss - self.min_delta:
                        patience_reference_loss = validation_metric
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1

                if self.verbose:
                    logging.info(
                        "Train metrics: %s | Validation metrics: %s",
                        self.train_loss[epoch],
                        self.val_loss[epoch],
                    )

                if self.early_stopping and epochs_without_improvement >= self.patience:
                    break

        if self.early_stopping and best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict(self, *, X=None, y=None, steps_ahead=None, forecast_horizon=None):
        """Return the predicted given an input and initial values.

        The predict function allows a friendly usage by the user.
        Given a trained model, predict values given
        a new set of data.

        The method supports free-run, one-step-ahead and n-step-ahead
        prediction.

        Parameters
        ----------
        X : ndarray of floats
            The input data to be used in the prediction process.
        y : ndarray of floats
            The output data to be used in the prediction process.
        steps_ahead : int (default = None)
            The user can use free run simulation, one-step ahead prediction
            and n-step ahead prediction.
        forecast_horizon : int, default=None
            Number of values predicted beyond the initial conditions for a NAR
            free-run prediction when ``X`` is ``None``.

        Returns
        -------
        yhat : ndarray of float32 of shape (n_predictions, 1)
            Predicted values including the initial conditions.

        """
        if self.net is None:
            raise ValueError("The neural network must be defined before prediction")

        xp = get_namespace(y) if X is None else get_namespace(X, y)
        _require_numpy_namespace(xp, feature="NARXNN", dependency="PyTorch/NumPy")

        training_modes = [(module, module.training) for module in self.net.modules()]
        self.net.eval()
        try:
            with torch.no_grad():
                result = self._predict(
                    X=X,
                    y=y,
                    steps_ahead=steps_ahead,
                    forecast_horizon=forecast_horizon,
                    allow_cpu_fallback=False,
                )
        finally:
            for module, training in training_modes:
                module.training = training

        return np.asarray(result, dtype=np.float32)

    def _one_step_ahead_prediction(self, x_base, y=None):
        """Perform the 1-step-ahead prediction of a model.

        Parameters
        ----------
        y : array-like of shape = max_lag
            Initial conditions values of the model
            to start recursive process.
        x : ndarray of floats of shape = n_samples
            Vector with input values to be used in model simulation.

        Returns
        -------
        yhat : ndarray of floats
               The 1-step-ahead predicted values of the model.

        """
        if y is None:
            raise ValueError("y cannot be None")

        n_inputs = num_features(x_base) if x_base is not None else 1

        lagged_data = build_lagged_matrix(
            x_base, y, self.xlag, self.ylag, self.model_type
        )

        x_base = self.basis_function.transform(
            lagged_data, self.max_lag, self.ylag, self.xlag, self.model_type
        )
        x_base, _ = self._prepare_regressor_matrix(x_base, n_inputs)

        predictions = self._forward_numpy(x_base)
        return predictions.astype(np.float32).reshape(-1, 1)

    def _narmax_predict(self, x, y_initial, forecast_horizon=None):
        if len(y_initial) < self.max_lag:
            raise ValueError(
                "Insufficient initial condition elements! Expected at least"
                f" {self.max_lag} elements."
            )

        n_inputs = self._prediction_n_inputs()
        if x is not None:
            forecast_horizon = x.shape[0]
            if n_inputs > 0:
                x = self._as_float_array(x).reshape(-1, n_inputs)
            else:
                x = None
        else:
            if forecast_horizon is None:
                raise ValueError(
                    "forecast_horizon cannot be None when x is None"
                    " for NARXNN prediction"
                )
            forecast_horizon = forecast_horizon + self.max_lag

        y_output = np.full(forecast_horizon, np.nan, dtype=np.float32)
        y_output[: self.max_lag] = y_initial[: self.max_lag, 0]

        model_exponents = np.asarray(
            self._get_prediction_exponents(),
            dtype=np.float32,
        )
        raw_regressor = np.zeros(model_exponents.shape[1], dtype=np.float32)
        regressor_powers = np.empty(model_exponents.shape, dtype=np.float32)
        regressor_value = np.empty(model_exponents.shape[0], dtype=np.float32)
        for i in range(self.max_lag, forecast_horizon):
            init = 0
            final = self.max_lag
            k = int(i - self.max_lag)
            raw_regressor[:final] = y_output[k:i]
            for j in range(n_inputs):
                init += self.max_lag
                final += self.max_lag
                raw_regressor[init:final] = x[k:i, j]

            np.power(raw_regressor, model_exponents, out=regressor_powers)
            np.prod(regressor_powers, axis=1, out=regressor_value)
            y_output[i] = self._scalar_forward(regressor_value)
        return y_output[self.max_lag :].reshape(-1, 1)

    def _nfir_predict(self, x, y_initial):
        n_inputs = self._prediction_n_inputs()
        x = self._as_float_array(x).reshape(-1, n_inputs)
        y_output = np.full(x.shape[0], np.nan, dtype=np.float32)
        y_output[: self.max_lag] = y_initial[: self.max_lag, 0]
        model_exponents = np.asarray(
            self._get_prediction_exponents(),
            dtype=np.float32,
        )
        raw_regressor = np.zeros(model_exponents.shape[1], dtype=np.float32)
        regressor_powers = np.empty(model_exponents.shape, dtype=np.float32)
        regressor_value = np.empty(model_exponents.shape[0], dtype=np.float32)
        for i in range(self.max_lag, x.shape[0]):
            # ``_code2exponents`` always reserves the first block for output
            # lags. NFIR codes only use the following input blocks.
            init = self.max_lag
            final = 2 * self.max_lag
            k = int(i - self.max_lag)
            for j in range(n_inputs):
                raw_regressor[init:final] = x[k:i, j]
                init += self.max_lag
                final += self.max_lag

            np.power(raw_regressor, model_exponents, out=regressor_powers)
            np.prod(regressor_powers, axis=1, out=regressor_value)
            y_output[i] = self._scalar_forward(regressor_value)
        return y_output[self.max_lag :].reshape(-1, 1)

    def _basis_function_predict(self, x, y_initial, forecast_horizon=None):
        if x is not None:
            forecast_horizon = x.shape[0]
        else:
            forecast_horizon = forecast_horizon + self.max_lag

        yhat = np.full(forecast_horizon, np.nan, dtype=np.float32)
        yhat[: self.max_lag] = y_initial[: self.max_lag, 0]

        analyzed_elements_number = self.max_lag + 1

        for i in range(forecast_horizon - self.max_lag):
            if self.model_type == "NARMAX":
                lagged_data = build_input_output_matrix(
                    x[i : i + analyzed_elements_number],
                    yhat[i : i + analyzed_elements_number].reshape(-1, 1),
                    self.xlag,
                    self.ylag,
                )
            elif self.model_type == "NAR":
                lagged_data = build_output_matrix(
                    yhat[i : i + analyzed_elements_number].reshape(-1, 1), self.ylag
                )
            elif self.model_type == "NFIR":
                lagged_data = build_input_matrix(
                    x[i : i + analyzed_elements_number], self.xlag
                )
            else:
                raise ValueError(
                    "Unrecognized model type. The model_type should be NARMAX, NAR or"
                    " NFIR."
                )

            x_tmp = self.basis_function.transform(
                lagged_data, self.max_lag, self.ylag, self.xlag, self.model_type
            )
            yhat[i + self.max_lag] = self._scalar_forward(x_tmp)
        return yhat[self.max_lag :].reshape(-1, 1)
