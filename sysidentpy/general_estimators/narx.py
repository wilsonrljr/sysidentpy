""" Build NARX Models Using general estimators """

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause


import logging
import numpy as np
from ..base import GenerateRegressors
from ..base import InformationMatrix
from ..residues.residues_correlation import ResiduesAnalysis
from ..utils._check_arrays import check_X_y

import sys

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)


class NARX(GenerateRegressors, InformationMatrix, ResiduesAnalysis):
    """NARX model build on top of general estimators

    Currently is possible to use any estimator that have a fit/predict
    as an Autoregressive Model. We use our GenerateRegressors and
    InformationMatrix classes to handle the creation of the lagged
    features and we are able to use a simple fit and prediction function
    to run infinity-steps-ahead prediction.

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
    fit_params : dict, default=None
        Optional parameters of the fit function of the baseline estimator
    base_estimator : default=None
        The defined base estimator of the sklearn
    verbose : bool, default=False
        Print messages

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> from sysidentpy.metrics import mean_squared_error
    >>> from sysidentpy.utils.generate_data import get_siso_data
    >>> from sysidentpy.general_estimators import NARX
    >>> from sklearn.linear_model import BayesianRidge # to use as base estimator
    >>> x_train, x_valid, y_train, y_valid = get_siso_data(n=1000,
    >>>                                                    colored_noise=False,
    >>>                                                    sigma=0.01,
    >>>                                                    train_percentage=80)
    >>> BayesianRidge_narx = NARX(base_estimator=BayesianRidge(),
    ...                           xlag=2,
    ...                           ylag=2
    ... )
    >>> BayesianRidge_narx.fit(x_train, y_train)
    >>> yhat = BayesianRidge_narx.predict(x_valid, y_valid)
    >>> print(mean_squared_error(y_valid, yhat))
    0.000131
    """

    def __init__(
        self,
        non_degree=1,
        ylag=2,
        xlag=2,
        n_inputs=1,
        base_estimator=None,
        fit_params={},
    ):
        self.non_degree = non_degree
        self.ylag = ylag
        self.xlag = xlag
        self._n_inputs = n_inputs
        [self.regressor_code, self.max_lag] = GenerateRegressors().regressor_space(
            non_degree, xlag, ylag, n_inputs
        )
        self.regressor_code = self.regressor_code[1:]
        self.base_estimator = base_estimator
        self.fit_params = fit_params
        self._validate_params()

    def _validate_params(self):
        """Validate input params."""

        if not isinstance(self._n_inputs, int) or self._n_inputs < 1:
            raise ValueError(
                "n_inputs must be integer and > zero. Got %f" % self._n_inputs
            )

    def data_preparation(self, X, y):
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
        logging.info("Creating the regressor matrix")
        reg_matrix = InformationMatrix().build_information_matrix(
            X, y, self.xlag, self.ylag, self.non_degree
        )
        logging.info(
            "The regressor matrix have " + str(reg_matrix.shape[1]) + " features"
        )
        reg_matrix = reg_matrix

        y = y[self.max_lag :]
        return reg_matrix, y

    def fit(self, X, y):
        """Train a NARX Neural Network model.

        This is an training pipeline that allows a friendly usage
        by the user. All the lagged features are built using the
        SysIdentPy classes and we use the fit method of the base
        estimator of the sklearn to fit the model.

        Parameters
        ----------
        X : ndarrays of floats
            The input data to be used in the training process.
        y : ndarrays of floats
            The output data to be used in the training process.

        Returns
        -------
        base_estimator : sklearn estimator
            The model fitted.
        """
        if y is None:
            raise ValueError("y cannot be None")

        check_X_y(X, y)

        logging.info("Training the model")
        X, y = self.data_preparation(X, y)
        self.base_estimator.fit(X, y, **self.fit_params)
        logging.info("Done! Model is built!")
        return self

    def predict(self, X, y_initial):
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
            )

            a = self.base_estimator.predict(reg_matrix)
            yhat[i + self.max_lag] = a[0]
        return yhat
