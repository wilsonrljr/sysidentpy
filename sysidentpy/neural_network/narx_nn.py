""" Build Polynomial NARMAX Models """

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause


import logging
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from ..narmax_base import BaseMSS
from ..basis_function import Polynomial
from ..utils._check_arrays import _check_positive_int, _num_features

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)


class NARXNN(BaseMSS):
    """NARX Neural Network model build on top of Pytorch

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

    Examples
    --------
    >>> from torch import nn
    >>> import numpy as np
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> from sysidentpy.metrics import mean_squared_error
    >>> from sysidentpy.utils.generate_data import get_siso_data
    >>> from sysidentpy.neural_network import NARXNN
    >>> from sysidentpy.utils.generate_data import get_siso_data
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
    >>> neural_narx.fit(X=x_train, y=y_train)
    >>> yhat = neural_narx.predict(X=x_valid, y=y_valid)
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
    ):
        self.ylag = ylag
        self.xlag = xlag
        self.basis_function = basis_function
        self.model_type = model_type
        self.build_matrix = self.get_build_io_method(model_type)
        self.non_degree = basis_function.degree
        self.max_lag = self._get_max_lag()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_func = getattr(F, loss_func)
        self.optimizer = optimizer
        self.net = net
        self.train_percentage = train_percentage
        self.verbose = verbose
        self.optim_params = optim_params
        self.device = self._check_cuda(device)
        self.regressor_code = None
        self.train_loss = None
        self.val_loss = None
        self.ensemble = None
        self.n_inputs = None
        self.final_model = None
        self._validate_params()

    def _validate_params(self):
        """Validate input params."""

        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError(
                f"bacth_size must be integer and > zero. Got {self.batch_size}"
            )

        if not isinstance(self.epochs, int) or self.epochs < 1:
            raise ValueError(f"epochs must be integer and > zero. Got {self.epochs}")

        if not isinstance(self.train_percentage, int) or self.train_percentage < 0:
            raise ValueError(
                f"bacth_size must be integer and > zero. Got {self.train_percentage}"
            )

        if not isinstance(self.verbose, bool):
            raise TypeError(f"verbose must be False or True. Got {self.verbose}")

        if isinstance(self.ylag, int) and self.ylag < 1:
            raise ValueError(f"ylag must be integer and > zero. Got {self.ylag}")

        if isinstance(self.xlag, int) and self.xlag < 1:
            raise ValueError(f"xlag must be integer and > zero. Got {self.xlag}")

        if not isinstance(self.xlag, (int, list)):
            raise ValueError(f"xlag must be integer and > zero. Got {self.xlag}")

        if not isinstance(self.ylag, (int, list)):
            raise ValueError(f"ylag must be integer and > zero. Got {self.ylag}")

        if self.model_type not in ["NARMAX", "NAR", "NFIR"]:
            raise ValueError(
                f"model_type must be NARMAX, NAR or NFIR. Got {self.model_type}"
            )

    def _check_cuda(self, device):
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

    def define_opt(self):
        """Defines the optimizer using the user parameters."""
        opt = getattr(optim, self.optimizer)
        return opt(self.net.parameters(), lr=self.learning_rate, **self.optim_params)

    def loss_batch(self, X, y, opt=None):
        """Compute the loss for one batch.

        Parameters
        ----------
        X : ndarray of floats
            The regressor matrix.
        y : ndarray of floats
            The output data.

        Returns
        -------
        loss : float
            The loss of one batch.

        """
        loss = self.loss_func(self.net(X), y)

        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()

        return loss.item(), len(X)

    def split_data(self, X, y):
        """Return the lagged matrix and the y values given the maximum lags.

        Parameters
        ----------
        X : ndarray of floats
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
        lagged_data = self.build_matrix(X, y)

        basis_name = self.basis_function.__class__.__name__
        if basis_name == "Polynomial":
            reg_matrix = self.basis_function.fit(
                lagged_data, self.max_lag, predefined_regressors=None
            )
            reg_matrix = reg_matrix[:, 1:]
        else:
            reg_matrix, self.ensemble = self.basis_function.fit(
                lagged_data, self.max_lag, predefined_regressors=None
            )

        if X is not None:
            self.n_inputs = _num_features(X)
        else:
            self.n_inputs = 1  # only used to create the regressor space base

        self.regressor_code = self.regressor_space(self.n_inputs)
        if basis_name != "Polynomial" and self.basis_function.ensemble:
            basis_code = np.sort(
                np.tile(
                    self.regressor_code[1:, :], (self.basis_function.repetition, 1)
                ),
                axis=0,
            )
            self.regressor_code = np.concatenate([self.regressor_code[1:], basis_code])
        elif basis_name != "Polynomial" and self.basis_function.ensemble is False:
            self.regressor_code = np.sort(
                np.tile(
                    self.regressor_code[1:, :], (self.basis_function.repetition, 1)
                ),
                axis=0,
            )

        if basis_name == "Polynomial":
            self.regressor_code = self.regressor_code[
                1:
            ]  # removes the column of the constant

        self.final_model = self.regressor_code.copy()
        reg_matrix = np.atleast_1d(reg_matrix).astype(np.float32)

        y = np.atleast_1d(y[self.max_lag :]).astype(np.float32)
        return reg_matrix, y

    def convert_to_tensor(self, reg_matrix, y):
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
        reg_matrix, y = map(torch.tensor, (reg_matrix, y))
        return TensorDataset(reg_matrix, y)

    def get_data(self, train_ds):
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
        return DataLoader(
            train_ds, batch_size=self.batch_size, pin_memory=pin_memory, shuffle=False
        )

    def data_transform(self, X, y):
        """Return the data transformed in tensors using Dataloader.

        Parameters
        ----------
        X : ndarray of floats
            The input data.
        y : ndarray of floats
            The output data.

        Returns
        -------
        Tensors : Dataloader

        """
        if y is None:
            raise ValueError("y cannot be None")

        x_train, y_train = self.split_data(X, y)
        train_ds = self.convert_to_tensor(x_train, y_train)
        train_dl = self.get_data(train_ds)
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
        train_dl = self.data_transform(X, y)
        if self.verbose:
            if X_test is None or y_test is None:
                raise ValueError(
                    "X_test and y_test cannot be None if you set verbose=True"
                )
            valid_dl = self.data_transform(X_test, y_test)

        opt = self.define_opt()
        self.val_loss = []
        self.train_loss = []
        for epoch in range(self.epochs):
            self.net.train()
            for X, y in train_dl:
                X, y = X.to(self.device), y.to(self.device)
                self.loss_batch(X, y, opt=opt)

            if self.verbose:
                train_losses, train_nums = zip(
                    *[
                        self.loss_batch(X.to(self.device), y.to(self.device))
                        for X, y in train_dl
                    ]
                )
                self.train_loss.append(
                    np.sum(np.multiply(train_losses, train_nums)) / np.sum(train_nums)
                )

                self.net.eval()
                with torch.no_grad():
                    losses, nums = zip(
                        *[
                            self.loss_batch(X.to(self.device), y.to(self.device))
                            for X, y in valid_dl
                        ]
                    )
                self.val_loss.append(np.sum(np.multiply(losses, nums)) / np.sum(nums))

                logging.info(
                    "Train metrics: "
                    + str(self.train_loss[epoch])
                    + " | Validation metrics: "
                    + str(self.val_loss[epoch])
                )
        return self

    def predict(self, *, X=None, y=None, steps_ahead=None, forecast_horizon=None):
        """Return the predicted given an input and initial values.

        The predict function allows a friendly usage by the user.
        Given a trained model, predict values given
        a new set of data.

        This method accept y values mainly for prediction n-steps ahead
        (to be implemented in the future).

        Currently we only support infinity-steps-ahead prediction,
        but run 1-step-ahead prediction manually is straightforward.

        Parameters
        ----------
        X : ndarray of floats
            The input data to be used in the prediction process.
        y : ndarray of floats
            The output data to be used in the prediction process.

        Returns
        -------
        yhat : ndarray of floats
            The predicted values of the model.

        """
        if self.basis_function.__class__.__name__ == "Polynomial":
            if steps_ahead is None:
                return self._model_prediction(X, y, forecast_horizon=forecast_horizon)
            if steps_ahead == 1:
                return self._one_step_ahead_prediction(X, y)

            _check_positive_int(steps_ahead, "steps_ahead")
            return self._n_step_ahead_prediction(X, y, steps_ahead=steps_ahead)

        if steps_ahead is None:
            return self._basis_function_predict(X, y, forecast_horizon=forecast_horizon)
        if steps_ahead == 1:
            return self._one_step_ahead_prediction(X, y)

        return self._basis_function_n_step_prediction(
            X, y, steps_ahead=steps_ahead, forecast_horizon=forecast_horizon
        )

    def _one_step_ahead_prediction(self, X, y):
        """Perform the 1-step-ahead prediction of a model.

        Parameters
        ----------
        y : array-like of shape = max_lag
            Initial conditions values of the model
            to start recursive process.
        X : ndarray of floats of shape = n_samples
            Vector with input values to be used in model simulation.

        Returns
        -------
        yhat : ndarray of floats
               The 1-step-ahead predicted values of the model.

        """
        lagged_data = self.build_matrix(X, y)

        basis_name = self.basis_function.__class__.__name__
        if basis_name == "Polynomial":
            X_base = self.basis_function.transform(
                lagged_data,
                self.max_lag,
            )
            X_base = X_base[:, 1:]
        else:
            X_base, _ = self.basis_function.transform(
                lagged_data,
                self.max_lag,
            )

        yhat = np.zeros(X.shape[0], dtype=float)
        X_base = np.atleast_1d(X_base).astype(np.float32)
        yhat = yhat.astype(np.float32)
        x_valid, _ = map(torch.tensor, (X_base, yhat))
        yhat = self.net(x_valid.to(self.device)).detach().cpu().numpy()
        yhat = np.concatenate([y.ravel()[: self.max_lag].flatten(), yhat.ravel()])
        return yhat.reshape(-1, 1)

    def _n_step_ahead_prediction(self, X, y, steps_ahead):
        """Perform the n-steps-ahead prediction of a model.

        Parameters
        ----------
        y : array-like of shape = max_lag
            Initial conditions values of the model
            to start recursive process.
        X : ndarray of floats of shape = n_samples
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

        yhat = np.zeros(X.shape[0], dtype=float)
        yhat.fill(np.nan)
        yhat[: self.max_lag] = y[: self.max_lag, 0]
        i = self.max_lag
        X = X.reshape(-1, self.n_inputs)
        while i < len(y):
            k = int(i - self.max_lag)
            if i + steps_ahead > len(y):
                steps_ahead = len(y) - i  # predicts the remaining values

            yhat[i : i + steps_ahead] = self._model_prediction(
                X[k : i + steps_ahead], y[k : i + steps_ahead]
            )[-steps_ahead:].ravel()

            i += steps_ahead

        yhat = yhat.ravel()
        return yhat.reshape(-1, 1)

    def _model_prediction(self, X, y_initial, forecast_horizon=None):
        """Perform the infinity steps-ahead simulation of a model.

        Parameters
        ----------
        y_initial : array-like of shape = max_lag
            Number of initial conditions values of output
            to start recursive process.
        X : ndarray of floats of shape = n_samples
            Vector with input values to be used in model simulation.

        Returns
        -------
        yhat : ndarray of floats
               The predicted values of the model.

        """
        if self.model_type in ["NARMAX", "NAR"]:
            return self._narmax_predict(X, y_initial, forecast_horizon)

        if self.model_type == "NFIR":
            return self._nfir_predict(X, y_initial)

        raise ValueError(
            f"model_type must be NARMAX, NAR or NFIR. Got {self.model_type}"
        )

    def _narmax_predict(self, X, y_initial, forecast_horizon):
        if len(y_initial) < self.max_lag:
            raise ValueError(
                "Insufficient initial condition elements! Expected at least"
                f" {self.max_lag} elements."
            )

        if X is not None:
            forecast_horizon = X.shape[0]
        else:
            forecast_horizon = forecast_horizon + self.max_lag

        if self.model_type == "NAR":
            self.n_inputs = 0

        y_output = np.zeros(forecast_horizon, dtype=float)
        y_output.fill(np.nan)
        y_output[: self.max_lag] = y_initial[: self.max_lag, 0]

        model_exponents = [
            self._code2exponents(code=model) for model in self.final_model
        ]
        raw_regressor = np.zeros(len(model_exponents[0]), dtype=float)
        for i in range(self.max_lag, forecast_horizon):
            init = 0
            final = self.max_lag
            k = int(i - self.max_lag)
            raw_regressor[:final] = y_output[k:i]
            for j in range(self.n_inputs):
                init += self.max_lag
                final += self.max_lag
                raw_regressor[init:final] = X[k:i, j]

            regressor_value = np.zeros(len(model_exponents))
            for j, model_exponent in enumerate(model_exponents):
                regressor_value[j] = np.prod(np.power(raw_regressor, model_exponent))

            regressor_value = np.atleast_1d(regressor_value).astype(np.float32)
            y_output = y_output.astype(np.float32)
            x_valid, _ = map(torch.tensor, (regressor_value, y_output))
            y_output[i] = self.net(x_valid.to(self.device))[0].detach().cpu().numpy()
        return y_output.reshape(-1, 1)

    def _nfir_predict(self, X, y_initial):
        y_output = np.zeros(X.shape[0], dtype=float)
        y_output.fill(np.nan)
        y_output[: self.max_lag] = y_initial[: self.max_lag, 0]
        X = X.reshape(-1, self.n_inputs)
        model_exponents = [
            self._code2exponents(code=model) for model in self.final_model
        ]
        raw_regressor = np.zeros(len(model_exponents[0]), dtype=float)
        for i in range(self.max_lag, X.shape[0]):
            init = 0
            final = self.max_lag
            k = int(i - self.max_lag)
            for j in range(self.n_inputs):
                raw_regressor[init:final] = X[k:i, j]
                init += self.max_lag
                final += self.max_lag

            regressor_value = np.zeros(len(model_exponents))
            for j, model_exponent in enumerate(model_exponents):
                regressor_value[j] = np.prod(np.power(raw_regressor, model_exponent))

            regressor_value = np.atleast_1d(regressor_value).astype(np.float32)
            y_output = y_output.astype(np.float32)
            x_valid, _ = map(torch.tensor, (regressor_value, y_output))
            y_output[i] = self.net(x_valid.to(self.device))[0].detach().cpu().numpy()
        return y_output.reshape(-1, 1)

    def _basis_function_predict(self, X, y_initial, forecast_horizon=None):
        if X is not None:
            forecast_horizon = X.shape[0]
        else:
            forecast_horizon = forecast_horizon + self.max_lag

        if self.model_type == "NAR":
            self.n_inputs = 0

        yhat = np.zeros(forecast_horizon, dtype=float)
        yhat.fill(np.nan)
        yhat[: self.max_lag] = y_initial[: self.max_lag, 0]

        analyzed_elements_number = self.max_lag + 1

        for i in range(0, forecast_horizon - self.max_lag):
            if self.model_type == "NARMAX":
                lagged_data = self.build_input_output_matrix(
                    X[i : i + analyzed_elements_number],
                    yhat[i : i + analyzed_elements_number].reshape(-1, 1),
                )
            elif self.model_type == "NAR":
                lagged_data = self.build_output_matrix(
                    yhat[i : i + analyzed_elements_number].reshape(-1, 1)
                )
            elif self.model_type == "NFIR":
                lagged_data = self.build_input_matrix(
                    X[i : i + analyzed_elements_number]
                )
            else:
                raise ValueError(
                    "Unrecognized model type. The model_type should be NARMAX, NAR or"
                    " NFIR."
                )

            X_tmp, _ = self.basis_function.transform(
                lagged_data,
                self.max_lag,
            )
            X_tmp = np.atleast_1d(X_tmp).astype(np.float32)
            yhat = yhat.astype(np.float32)
            x_valid, _ = map(torch.tensor, (X_tmp, yhat))
            yhat[i + self.max_lag] = (
                self.net(x_valid.to(self.device))[0].detach().cpu().numpy()
            )
        return yhat.reshape(-1, 1)

    def _basis_function_n_step_prediction(self, X, y, steps_ahead, forecast_horizon):
        """Perform the n-steps-ahead prediction of a model.

        Parameters
        ----------
        y : array-like of shape = max_lag
            Initial conditions values of the model
            to start recursive process.
        X : ndarray of floats of shape = n_samples
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

        if X is not None:
            forecast_horizon = X.shape[0]
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
                    X[k : i + steps_ahead], y[k : i + steps_ahead]
                )[-steps_ahead:].ravel()
            elif self.model_type == "NAR":
                yhat[i : i + steps_ahead] = self._basis_function_predict(
                    X=None,
                    y_initial=y[k : i + steps_ahead],
                    forecast_horizon=forecast_horizon,
                )[-forecast_horizon : -forecast_horizon + steps_ahead].ravel()
            elif self.model_type == "NFIR":
                yhat[i : i + steps_ahead] = self._basis_function_predict(
                    X=X[k : i + steps_ahead],
                    y_initial=y[k : i + steps_ahead],
                )[-steps_ahead:].ravel()
            else:
                raise ValueError(
                    f"model_type must be NARMAX, NAR or NFIR. Got {self.model_type}"
                )

            i += steps_ahead

        return yhat.reshape(-1, 1)

    def _basis_function_n_steps_horizon(self, X, y, steps_ahead, forecast_horizon):
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
                    X[k : i + steps_ahead], y[k : i + steps_ahead]
                )[-forecast_horizon : -forecast_horizon + steps_ahead].ravel()
            elif self.model_type == "NAR":
                yhat[i : i + steps_ahead] = self._basis_function_predict(
                    X=None,
                    y_initial=y[k : i + steps_ahead],
                    forecast_horizon=forecast_horizon,
                )[-forecast_horizon : -forecast_horizon + steps_ahead].ravel()
            elif self.model_type == "NFIR":
                yhat[i : i + steps_ahead] = self._basis_function_predict(
                    X=X[k : i + steps_ahead],
                    y_initial=y[k : i + steps_ahead],
                )[-forecast_horizon : -forecast_horizon + steps_ahead].ravel()
            else:
                raise ValueError(
                    f"model_type must be NARMAX, NAR or NFIR. Got {self.model_type}"
                )

            i += steps_ahead

        yhat = yhat.ravel()
        return yhat.reshape(-1, 1)
