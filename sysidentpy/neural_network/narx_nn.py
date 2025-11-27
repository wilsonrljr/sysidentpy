"""Build Polynomial NARMAX Models."""

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause


import logging
import sys
import warnings
from collections.abc import Mapping
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from ..narmax_base import BaseMSS
from ..basis_function import Polynomial
from sysidentpy.utils.information_matrix import (
    build_output_matrix,
    build_input_matrix,
    build_input_output_matrix,
    build_lagged_matrix,
)
from ..utils.check_arrays import check_positive_int, num_features

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

    if device == "cuda":
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
    ):
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

    @staticmethod
    def _sanitize_lag(value, name):
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

    @staticmethod
    def _as_float_array(array):
        return np.ascontiguousarray(np.asarray(array, dtype=np.float32))

    def _forward_numpy(self, array):
        tensor = torch.from_numpy(self._as_float_array(array))
        if self.device.type != "cpu":
            tensor = tensor.to(self.device, non_blocking=True)
        return self.net(tensor).detach().cpu().numpy()

    def _scalar_forward(self, array):
        return float(self._forward_numpy(array).reshape(-1)[0])

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

        if isinstance(self.basis_function, Polynomial):
            reg_matrix = self.basis_function.fit(
                lagged_data,
                self.max_lag,
                self.ylag,
                self.xlag,
                self.model_type,
                predefined_regressors=None,
            )
            reg_matrix = reg_matrix[:, 1:]
        else:
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

        self.regressor_code = self.regressor_space(self.n_inputs)
        repetition = len(reg_matrix)
        if not isinstance(self.basis_function, Polynomial):
            tmp_code = np.sort(
                np.tile(self.regressor_code[1:, :], (repetition, 1)),
                axis=0,
            )
            self.regressor_code = tmp_code[list(range(len(reg_matrix))), :].copy()
        else:
            self.regressor_code = self.regressor_code[
                1:
            ]  # removes the column of the constant

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
            The input data to be used in the prediction process.
        y_test : ndarray of floats
            The output data (initial conditions) to be used in the prediction process.

        Returns
        -------
        net : nn.Module
            The model fitted.
        train_loss: ndarrays of floats
            The training loss of each batch
        val_loss: ndarrays of floats
            The validation loss of each batch

        """
        if self.random_state is not None:
            self._seed_torch_generators()
            self._reset_network_parameters()

        train_dl = self.data_transform(X, y, shuffle=self.shuffle_batches)
        if self.verbose:
            if X_test is None or y_test is None:
                raise ValueError(
                    "X_test and y_test cannot be None if you set verbose=True"
                )
            valid_dl = self.data_transform(X_test, y_test, shuffle=False)

        opt = self.define_opt()
        self.val_loss = []
        self.train_loss = []
        for epoch in range(self.epochs):
            self.net.train()
            epoch_loss = 0.0
            seen_samples = 0
            for input_data, output_data in train_dl:
                X_batch = input_data.to(self.device, non_blocking=True)
                y_batch = output_data.to(self.device, non_blocking=True)
                batch_loss, batch_size = self.loss_batch(X_batch, y_batch, opt=opt)
                if self.verbose:
                    epoch_loss += batch_loss * batch_size
                    seen_samples += batch_size

            if self.verbose:
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
                self.val_loss.append(val_loss / max(val_count, 1))
                logging.info(
                    "Train metrics: %s | Validation metrics: %s",
                    self.train_loss[epoch],
                    self.val_loss[epoch],
                )
        return self

    def predict(self, *, X=None, y=None, steps_ahead=None, forecast_horizon=None):
        """Return the predicted given an input and initial values.

        The predict function allows a friendly usage by the user.
        Given a trained model, predict values given
        a new set of data.

        This method accept y values mainly for prediction n-steps ahead
        (to be implemented in the future).

        Currently, we only support infinity-steps-ahead prediction,
        but run 1-step-ahead prediction manually is straightforward.

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
            The number of predictions over the time.

        Returns
        -------
        yhat : ndarray of floats
            The predicted values of the model.

        """
        if self.net is None:
            raise ValueError("The neural network must be defined before prediction")

        was_training = self.net.training
        self.net.eval()
        try:
            with torch.no_grad():
                if isinstance(self.basis_function, Polynomial):
                    if steps_ahead is None:
                        result = self._model_prediction(
                            X, y, forecast_horizon=forecast_horizon
                        )
                    elif steps_ahead == 1:
                        result = self._one_step_ahead_prediction(X, y)
                    else:
                        check_positive_int(steps_ahead, "steps_ahead")
                        result = self._n_step_ahead_prediction(
                            X, y, steps_ahead=steps_ahead
                        )
                else:
                    if steps_ahead is None:
                        result = self._basis_function_predict(
                            X, y, forecast_horizon=forecast_horizon
                        )
                    elif steps_ahead == 1:
                        result = self._one_step_ahead_prediction(X, y)
                    else:
                        check_positive_int(steps_ahead, "steps_ahead")
                        result = self._basis_function_n_step_prediction(
                            X,
                            y,
                            steps_ahead=steps_ahead,
                            forecast_horizon=forecast_horizon,
                        )
        finally:
            if was_training:
                self.net.train()

        return result

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

        lagged_data = build_lagged_matrix(
            x_base, y, self.xlag, self.ylag, self.model_type
        )

        if isinstance(self.basis_function, Polynomial):
            x_base = self.basis_function.transform(
                lagged_data, self.max_lag, self.ylag, self.xlag, self.model_type
            )
            x_base = x_base[:, 1:]
        else:
            x_base = self.basis_function.transform(
                lagged_data, self.max_lag, self.ylag, self.xlag, self.model_type
            )

        predictions = self._forward_numpy(x_base)
        yhat = np.concatenate(
            [y.ravel()[: self.max_lag].flatten(), predictions.ravel()]
        )
        return yhat.astype(np.float32).reshape(-1, 1)

    def _n_step_ahead_prediction(self, x, y, steps_ahead):
        """Perform the n-steps-ahead prediction of a model.

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
               The n-steps-ahead predicted values of the model.

        """
        if len(y) < self.max_lag:
            raise ValueError(
                "Insufficient initial condition elements! Expected at least"
                f" {self.max_lag} elements."
            )

        yhat = np.zeros(x.shape[0], dtype=float)
        yhat.fill(np.nan)
        yhat[: self.max_lag] = y[: self.max_lag, 0]
        i = self.max_lag
        x = x.reshape(-1, self.n_inputs)
        while i < len(y):
            k = int(i - self.max_lag)
            if i + steps_ahead > len(y):
                steps_ahead = len(y) - i  # predicts the remaining values

            yhat[i : i + steps_ahead] = self._model_prediction(
                x[k : i + steps_ahead], y[k : i + steps_ahead]
            )[-steps_ahead:].ravel()

            i += steps_ahead

        yhat = yhat.ravel()
        return yhat.reshape(-1, 1)

    def _model_prediction(self, x, y_initial, forecast_horizon=None):
        """Perform the infinity steps-ahead simulation of a model.

        Parameters
        ----------
        y_initial : array-like of shape = max_lag
            Number of initial conditions values of output
            to start recursive process.
        x : ndarray of floats of shape = n_samples
            Vector with input values to be used in model simulation.

        Returns
        -------
        yhat : ndarray of floats
               The predicted values of the model.

        """
        if self.model_type in ["NARMAX", "NAR"]:
            return self._narmax_predict(x, y_initial, forecast_horizon)

        if self.model_type == "NFIR":
            return self._nfir_predict(x, y_initial)

        raise ValueError(
            f"model_type must be NARMAX, NAR or NFIR. Got {self.model_type}"
        )

    def _narmax_predict(self, x, y_initial, forecast_horizon=None):
        if len(y_initial) < self.max_lag:
            raise ValueError(
                "Insufficient initial condition elements! Expected at least"
                f" {self.max_lag} elements."
            )

        if x is not None:
            x = self._as_float_array(x).reshape(-1, self.n_inputs)
            forecast_horizon = x.shape[0]
        else:
            if forecast_horizon is None:
                raise ValueError(
                    "forecast_horizon cannot be None when x is None for NARXNN prediction"
                )
            forecast_horizon = forecast_horizon + self.max_lag

        if self.model_type == "NAR":
            self.n_inputs = 0

        y_output = np.full(forecast_horizon, np.nan, dtype=np.float32)
        y_output[: self.max_lag] = y_initial[: self.max_lag, 0]

        model_exponents = np.vstack(
            [self._code2exponents(code=model) for model in self.final_model]
        )
        raw_regressor = np.zeros(model_exponents.shape[1], dtype=np.float32)
        regressor_powers = np.empty(model_exponents.shape, dtype=np.float32)
        regressor_value = np.empty(model_exponents.shape[0], dtype=np.float32)
        for i in range(self.max_lag, forecast_horizon):
            init = 0
            final = self.max_lag
            k = int(i - self.max_lag)
            raw_regressor[:final] = y_output[k:i]
            for j in range(self.n_inputs):
                init += self.max_lag
                final += self.max_lag
                raw_regressor[init:final] = x[k:i, j]

            np.power(raw_regressor, model_exponents, out=regressor_powers)
            np.prod(regressor_powers, axis=1, out=regressor_value)
            y_output[i] = self._scalar_forward(regressor_value)
        return y_output.reshape(-1, 1)

    def _nfir_predict(self, x, y_initial):
        x = self._as_float_array(x).reshape(-1, self.n_inputs)
        y_output = np.full(x.shape[0], np.nan, dtype=np.float32)
        y_output[: self.max_lag] = y_initial[: self.max_lag, 0]
        model_exponents = np.vstack(
            [self._code2exponents(code=model) for model in self.final_model]
        )
        raw_regressor = np.zeros(model_exponents.shape[1], dtype=np.float32)
        regressor_powers = np.empty(model_exponents.shape, dtype=np.float32)
        regressor_value = np.empty(model_exponents.shape[0], dtype=np.float32)
        for i in range(self.max_lag, x.shape[0]):
            init = 0
            final = self.max_lag
            k = int(i - self.max_lag)
            for j in range(self.n_inputs):
                raw_regressor[init:final] = x[k:i, j]
                init += self.max_lag
                final += self.max_lag

            np.power(raw_regressor, model_exponents, out=regressor_powers)
            np.prod(regressor_powers, axis=1, out=regressor_value)
            y_output[i] = self._scalar_forward(regressor_value)
        return y_output.reshape(-1, 1)

    def _basis_function_predict(self, x, y_initial, forecast_horizon=None):
        if x is not None:
            forecast_horizon = x.shape[0]
        else:
            forecast_horizon = forecast_horizon + self.max_lag

        if self.model_type == "NAR":
            self.n_inputs = 0

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
        return yhat.reshape(-1, 1)

    def _basis_function_n_step_prediction(self, x, y, steps_ahead, forecast_horizon):
        """Perform the n-steps-ahead prediction of a model.

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
               The n-steps-ahead predicted values of the model.

        """
        if len(y) < self.max_lag:
            raise ValueError(
                "Insufficient initial condition elements! Expected at least"
                f" {self.max_lag} elements."
            )

        if x is not None:
            forecast_horizon = x.shape[0]
        else:
            forecast_horizon = forecast_horizon + self.max_lag

        yhat = np.zeros(forecast_horizon, dtype=float)
        yhat.fill(np.nan)
        yhat[: self.max_lag] = y[: self.max_lag, 0]

        i = self.max_lag

        while i < len(y):
            k = int(i - self.max_lag)
            if i + steps_ahead > len(y):
                steps_ahead = len(y) - i  # predicts the remaining values

            if self.model_type == "NARMAX":
                yhat[i : i + steps_ahead] = self._basis_function_predict(
                    x[k : i + steps_ahead], y[k : i + steps_ahead]
                )[-steps_ahead:].ravel()
            elif self.model_type == "NAR":
                yhat[i : i + steps_ahead] = self._basis_function_predict(
                    x=None,
                    y_initial=y[k : i + steps_ahead],
                    forecast_horizon=forecast_horizon,
                )[-forecast_horizon : -forecast_horizon + steps_ahead].ravel()
            elif self.model_type == "NFIR":
                yhat[i : i + steps_ahead] = self._basis_function_predict(
                    x=x[k : i + steps_ahead],
                    y_initial=y[k : i + steps_ahead],
                )[-steps_ahead:].ravel()
            else:
                raise ValueError(
                    f"model_type must be NARMAX, NAR or NFIR. Got {self.model_type}"
                )

            i += steps_ahead

        return yhat.reshape(-1, 1)
