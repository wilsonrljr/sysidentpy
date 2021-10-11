""" Build Polynomial NARMAX Models """

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause


import logging
import numpy as np
from ..base import GenerateRegressors
from ..base import InformationMatrix
from ..residues.residues_correlation import ResiduesAnalysis
from ..utils._check_arrays import check_X_y
from ..utils.deprecation import deprecated
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch import optim
import torch.nn.functional as F
import torch
import sys


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)


class NARXNN(GenerateRegressors, InformationMatrix, ResiduesAnalysis):
    """NARX Neural Network model build on top of Pytorch

    Currently we support a Series-Parallel (open-loop) Feedforward Network training
    process, which make the training process easier, and we convert the
    NARX network from Series-Parallel to the Parallel (closed-loop) configuration for prediction.

    Parameters
    ----------
    non_degree : int, default=1
        The nonlinearity degree of the polynomial function.
    ylag : int, default=2
        The maximum lag of the output.
    xlag : int, default=2
        The maximum lag of the input.
    n_inputs : int, default=1
        The number of inputs of the system.
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
    opt_params : dict, default=None
        Optional parameters for the optimizer
    net : default=None
        The defined network using nn.Module
    train_percentage : int, default=80
        The percentage of training data to split the dataset
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
    >>> from sysidentpy.utils.generate_data import get_miso_data, get_siso_data
    >>> x_train, x_valid, y_train, y_valid = get_siso_data(n=1000,
    >>>                                                 colored_noise=False,
    >>>                                                sigma=0.01,
    >>>                                                train_percentage=80)
    >>> narx_nn = NARXNN(ylag=2,
    ...                  xlag=2,
    ...                  loss_func='mse_loss',
    ...                  optimizer='Adam',
    ...                  epochs=200,
    ...                  verbose=False)
    >>> train_dl = narx_nn.data_transform(x_train, y_train)
    >>> valid_dl = narx_nn.data_transform(x_valid, y_valid)
    >>> class Net(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.lin = nn.Linear(4, 10)
    ...         self.lin2 = nn.Linear(10, 10)
    ...         self.lin3 = nn.Linear(10, 1)
    ...         self.tanh = nn.Tanh()
    ...
    ...     def forward(self, xb):
    ...         z = self.lin(xb)
    ...         z = self.tanh(z)
    ...         z = self.lin2(z)
    ...         z = self.tanh(z)
    ...         z = self.lin3(z)
    ...         return z

    >>> narx_nn.net = Net()
    >>> neural_narx.fit(train_dl, valid_dl)
    >>> yhat = neural_narx.predict(x_valid, y_valid)
    >>> print(mean_squared_error(y_valid, yhat))
    0.000131

    References
    ----------
    .. [1]`Manuscript: Orthogonal least squares methods and their application
       to non-linear system identification
       <https://eprints.soton.ac.uk/251147/1/778742007_content.pdf>`_
    """
    @deprecated(version='v0.1.7', future_version='v0.2.0',
            alternative="NARXNN(ylag=2, xlag=2, basis_function='Some basis function')")
    def __init__(
        self,
        non_degree=1,
        ylag=2,
        xlag=2,
        n_inputs=1,
        batch_size=100,  # batch size
        learning_rate=0.01,  # learning rate
        epochs=200,  # how many epochs to train for
        loss_func="mse_loss",
        optimizer="Adam",
        net=None,
        train_percentage=80,
        verbose=False,
        optim_params=None
        # **opt_params
    ):
        self.non_degree = non_degree
        self.ylag = ylag
        self.xlag = xlag
        self._n_inputs = n_inputs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_func = getattr(F, loss_func)
        self.optimizer = optimizer
        [self.regressor_code, self.max_lag] = GenerateRegressors().regressor_space(
            non_degree, xlag, ylag, n_inputs
        )
        self.regressor_code = self.regressor_code[1:]
        self.net = net
        self.train_percentage = train_percentage
        self.verbose = verbose
        self.optim_params = optim_params
        self._validate_params()

    def _validate_params(self):
        """Validate input params."""

        if not isinstance(self._n_inputs, int) or self._n_inputs < 1:
            raise ValueError(
                "n_inputs must be integer and > zero. Got %f" % self._n_inputs
            )

        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError(
                "bacth_size must be integer and > zero. Got %f" % self.batch_size
            )

        if not isinstance(self.epochs, int) or self.epochs < 1:
            raise ValueError("epochs must be integer and > zero. Got %f" % self.epochs)

        if not isinstance(self.train_percentage, int) or self.train_percentage < 0:
            raise ValueError(
                "bacth_size must be integer and > zero. Got %f" % self.train_percentage
            )

        if not isinstance(self.verbose, bool):
            raise TypeError("verbose must be False or True. Got %f" % self.verbose)

    def define_opt(self):
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
        # [:, 1] removes the column of the constant
        reg_matrix = InformationMatrix().build_information_matrix(
            X, y, self.xlag, self.ylag, self.non_degree
        )[:, 1:]

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
        Tensor: tensor
            Tensors that have the same size of the first dimension.

        Returns
        -------
        Dataloader: dataloader
            tensors that have the same size of the first dimension.
        """
        return DataLoader(train_ds, batch_size=self.batch_size, shuffle=False)

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

        check_X_y(X, y)

        x_train, y_train = self.split_data(X, y)
        train_ds = self.convert_to_tensor(x_train, y_train)
        train_dl = self.get_data(train_ds)
        return train_dl

    def fit(self, train_dl, valid_dl):
        """Train a NARX Neural Network model.

        This is an training pipeline that allows a friendly usage
        by the user. The training pipeline was based on
        https://pytorch.org/tutorials/beginner/nn_tutorial.html

        Parameters
        ----------
        train_dl : Tensor
            The input data to be used in the training process.
        valid_dl : Tensor
            The output data to be used in the training process.

        Returns
        -------
        net : nn.Module
            The model fitted.
        train_loss: ndarrays of floats
            The training loss of each batch
        val_loss: ndarrays of floats
            The validation loss of each batch
        """
        opt = self.define_opt()
        self.val_loss = []
        self.train_loss = []
        for epoch in range(self.epochs):
            self.net.train()
            for X, y in train_dl:
                self.loss_batch(X, y, opt=opt)

            if self.verbose == True:
                train_losses, train_nums = zip(
                    *[self.loss_batch(X, y) for X, y in train_dl]
                )
                self.train_loss.append(
                    np.sum(np.multiply(train_losses, train_nums)) / np.sum(train_nums)
                )

                self.net.eval()
                with torch.no_grad():
                    losses, nums = zip(*[self.loss_batch(X, y) for X, y in valid_dl])
                self.val_loss.append(np.sum(np.multiply(losses, nums)) / np.sum(nums))

                logging.info(
                    "Train metrics: "
                    + str(self.train_loss[epoch])
                    + " | Validation metrics: "
                    + str(self.val_loss[epoch])
                )
        return self

    def one_step_ahead_prediction(self, valid_dl):
        x_valid, y_valid = valid_dl.dataset[:]
        yhat = self.net(x_valid).detach().numpy()
        return yhat

    def predict(self, X, y_initial):
        """Return the predicted given an input and initial values.

        The predict function allows a friendly usage by the user.
        Given a trained model, predict values given
        a new set of data.

        This method accept y values mainly for prediction n-steps ahead
        (to be implemented in the future)

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
        yhat = np.zeros((len(X), 1))

        # Discard unnecessary initial values
        yhat[0 : self.max_lag] = y_initial[0 : self.max_lag]
        analised_elements_number = self.max_lag + 1

        for i in range(0, len(X) - self.max_lag):
            reg_matrix = InformationMatrix().build_information_matrix(
                X[i : i + analised_elements_number],
                yhat[i : i + analised_elements_number],
                self.xlag,
                self.ylag,
                self.non_degree,
            )[:, 1:]

            reg_matrix = np.atleast_1d(reg_matrix).astype(np.float32)
            yhat = yhat.astype(np.float32)
            x_valid, y_valid = map(torch.tensor, (reg_matrix, yhat))
            a = self.net(x_valid)
            yhat[i + self.max_lag] = a[:, 0].detach().numpy()
        return yhat
