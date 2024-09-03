"""Build NARX Models Using general estimators."""

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause


import logging
import sys
from typing import Any, List, Union, Optional


import numpy as np
from numpy.typing import NDArray

from ..narmax_base import BaseMSS
from ..basis_function import Polynomial, Fourier
from ..utils._check_arrays import _check_positive_int, _num_features

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)


class NARX(BaseMSS):
    """NARX model build on top of general estimators.

    Currently is possible to use any estimator that have a fit/predict
    as an Autoregressive Model. We use our GenerateRegressors and
    InformationMatrix classes to handle the creation of the lagged
    features and we are able to use a simple fit and prediction function
    to run infinity-steps-ahead prediction.

    Parameters
    ----------
    ylag : int, default=2
        The maximum lag of the output.
    xlag : int, default=2
        The maximum lag of the input.
    fit_params : dict, default=None
        Optional parameters of the fit function of the baseline estimator
    base_estimator : default=None
        The defined base estimator of the sklearn

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> from sysidentpy.metrics import mean_squared_error
    >>> from sysidentpy.utils.generate_data import get_siso_data
    >>> from sysidentpy.general_estimators import NARX
    >>> from sklearn.linear_model import BayesianRidge
    >>> from sysidentpy.basis_function._basis_function import Polynomial
    >>> from sysidentpy.utils.display_results import results
    >>> from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
    >>> from sysidentpy.residues.residues_correlation import(
    ...    compute_residues_autocorrelation,
    ...    compute_cross_correlation
    ... )
    >>> from sklearn.linear_model import BayesianRidge # to use as base estimator
    >>> x_train, x_valid, y_train, y_valid = get_siso_data(
    ...    n=1000,
    ...    colored_noise=False,
    ...    sigma=0.01,
    ...    train_percentage=80
    ... )
    >>> BayesianRidge_narx = NARX(
    ...     base_estimator=BayesianRidge(),
    ...     xlag=2,
    ...     ylag=2,
    ...     basis_function=basis_function,
    ...     model_type="NARMAX",
    ... )
    >>> BayesianRidge_narx.fit(x_train, y_train)
    >>> yhat = BayesianRidge_narx.predict(x_valid, y_valid)
    >>> print("MSE: ", mean_squared_error(y_valid, yhat))
    >>> plot_results(y=y_valid, yhat=yhat, n=1000)
    >>> ee = compute_residues_autocorrelation(y_valid, yhat)
    >>> plot_residues_correlation(data=ee, title="Residues", ylabel="$e^2$")
    >>> x1e = compute_cross_correlation(y_valid, yhat, x_valid)
    >>> plot_residues_correlation(data=x1e, title="Residues", ylabel="$x_1e$")
    0.000131

    """

    def __init__(
        self,
        *,
        ylag: Union[List[Any], Any] = 1,
        xlag: Union[List[Any], Any] = 1,
        model_type: str = "NARMAX",
        basis_function: Union[Polynomial, Fourier] = Polynomial(),
        base_estimator=None,
        fit_params=None,
    ):
        self.basis_function = basis_function
        self.model_type = model_type
        self.build_matrix = self.get_build_io_method(model_type)
        self.non_degree = basis_function.degree
        self.ylag = ylag
        self.xlag = xlag
        self.max_lag = self._get_max_lag()
        self.base_estimator = base_estimator
        if fit_params is None:
            fit_params = {}

        self.fit_params = fit_params
        self.ensemble = None
        self.n_inputs = None
        self.regressor_code = None
        self._validate_params()

    def _validate_params(self):
        """Validate input params."""
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

    def fit(self, *, X=None, y=None):
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

        self.max_lag = self._get_max_lag()
        lagged_data = self.build_matrix(X, y)
        reg_matrix = self.basis_function.fit(
            lagged_data,
            self.max_lag,
            self.ylag,
            self.xlag,
            self.model_type,
            predefined_regressors=None,
        )

        if X is not None:
            self.n_inputs = _num_features(X)
        else:
            self.n_inputs = 1  # just to create the regressor space base

        self.regressor_code = self.regressor_space(self.n_inputs)
        self.final_model = self.regressor_code
        y = y[self.max_lag :].ravel()

        self.base_estimator.fit(reg_matrix, y, **self.fit_params)
        return self

    def predict(
        self,
        *,
        X: Optional[NDArray] = None,
        y: Optional[NDArray] = None,
        steps_ahead: Optional[int] = None,
        forecast_horizon: Optional[int] = 1,
    ) -> NDArray:
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
        if isinstance(self.basis_function, Polynomial):
            if steps_ahead is None:
                yhat = self._model_prediction(X, y, forecast_horizon=forecast_horizon)
                yhat = np.concatenate([y[: self.max_lag], yhat], axis=0)
                return yhat

            if steps_ahead == 1:
                yhat = self._one_step_ahead_prediction(X, y)
                yhat = np.concatenate([y[: self.max_lag], yhat], axis=0)
                return yhat

            _check_positive_int(steps_ahead, "steps_ahead")
            yhat = self._n_step_ahead_prediction(X, y, steps_ahead=steps_ahead)
            yhat = np.concatenate([y[: self.max_lag], yhat], axis=0)
            return yhat

        if steps_ahead is None:
            yhat = self._basis_function_predict(X, y, forecast_horizon=forecast_horizon)
            yhat = np.concatenate([y[: self.max_lag], yhat], axis=0)
            return yhat
        if steps_ahead == 1:
            yhat = self._one_step_ahead_prediction(X, y)
            yhat = np.concatenate([y[: self.max_lag], yhat], axis=0)
            return yhat

        yhat = self._basis_function_n_step_prediction(
            X, y, steps_ahead=steps_ahead, forecast_horizon=forecast_horizon
        )
        yhat = np.concatenate([y[: self.max_lag], yhat], axis=0)
        return yhat

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
        X_base = self.basis_function.transform(
            lagged_data, self.max_lag, self.ylag, self.xlag, self.model_type
        )

        yhat = self.base_estimator.predict(X_base)
        # yhat = np.concatenate([y[: self.max_lag].flatten(), yhat])  # delete this one
        return yhat.reshape(-1, 1)

    def _nar_step_ahead(self, y, steps_ahead):
        if len(y) < self.max_lag:
            raise ValueError(
                "Insufficient initial condition elements! Expected at least"
                f" {self.max_lag} elements."
            )

        to_remove = int(np.ceil((len(y) - self.max_lag) / steps_ahead))
        yhat = np.zeros(len(y) + steps_ahead, dtype=float)
        yhat.fill(np.nan)
        yhat[: self.max_lag] = y[: self.max_lag, 0]
        i = self.max_lag

        steps = [step for step in range(0, to_remove * steps_ahead, steps_ahead)]
        if len(steps) > 1:
            for step in steps[:-1]:
                yhat[i : i + steps_ahead] = self._model_prediction(
                    X=None, y_initial=y[step:i], forecast_horizon=steps_ahead
                )[-steps_ahead:].ravel()
                i += steps_ahead

            steps_ahead = np.sum(np.isnan(yhat))
            yhat[i : i + steps_ahead] = self._model_prediction(
                X=None, y_initial=y[steps[-1] : i]
            )[-steps_ahead:].ravel()
        else:
            yhat[i : i + steps_ahead] = self._model_prediction(
                X=None, y_initial=y[0:i], forecast_horizon=steps_ahead
            )[-steps_ahead:].ravel()

        yhat = yhat.ravel()[self.max_lag : :]
        return yhat.reshape(-1, 1)

    def narmax_n_step_ahead(self, X, y, steps_ahead):
        """N steps ahead prediction method for NARMAX model."""
        if len(y) < self.max_lag:
            raise ValueError(
                "Insufficient initial condition elements! Expected at least"
                f" {self.max_lag} elements."
            )

        to_remove = int(np.ceil((len(y) - self.max_lag) / steps_ahead))
        X = X.reshape(-1, self.n_inputs)
        yhat = np.zeros(X.shape[0], dtype=float)
        yhat.fill(np.nan)
        yhat[: self.max_lag] = y[: self.max_lag, 0]
        i = self.max_lag
        steps = [step for step in range(0, to_remove * steps_ahead, steps_ahead)]
        if len(steps) > 1:
            for step in steps[:-1]:
                yhat[i : i + steps_ahead] = self._model_prediction(
                    X=X[step : i + steps_ahead],
                    y_initial=y[step:i],
                )[-steps_ahead:].ravel()
                i += steps_ahead

            steps_ahead = np.sum(np.isnan(yhat))
            yhat[i : i + steps_ahead] = self._model_prediction(
                X=X[steps[-1] : i + steps_ahead],
                y_initial=y[steps[-1] : i],
            )[-steps_ahead:].ravel()
        else:
            yhat[i : i + steps_ahead] = self._model_prediction(
                X=X[0 : i + steps_ahead],
                y_initial=y[0:i],
            )[-steps_ahead:].ravel()

        yhat = yhat.ravel()[self.max_lag : :]
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
        steps_ahead : int (default = None)
            The user can use free run simulation, one-step ahead prediction
            and n-step ahead prediction.

        Returns
        -------
        yhat : ndarray of floats
               The n-steps-ahead predicted values of the model.

        """
        if self.model_type == "NARMAX":
            return self.narmax_n_step_ahead(X, y, steps_ahead)

        if self.model_type == "NAR":
            return self._nar_step_ahead(y, steps_ahead)

    def _model_prediction(self, X, y_initial, forecast_horizon=None):
        """Perform the infinity steps-ahead simulation of a model.

        Parameters
        ----------
        y_initial : array-like of shape = max_lag
            Number of initial conditions values of output
            to start recursive process.
        X : ndarray of floats of shape = n_samples
            Vector with input values to be used in model simulation.
        forecast_horizon : int, default=None
            The number of predictions over the time.

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

            y_output[i] = self.base_estimator.predict(regressor_value.reshape(1, -1))[0]
        return y_output[self.max_lag : :].reshape(-1, 1)

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
            raw_regressor[:final] = y_output[k:i]
            for j in range(self.n_inputs):
                init += self.max_lag
                final += self.max_lag
                raw_regressor[init:final] = X[k:i, j]

            regressor_value = np.zeros(len(model_exponents))
            for j, model_exponent in enumerate(model_exponents):
                regressor_value[j] = np.prod(np.power(raw_regressor, model_exponent))

            y_output[i] = self.base_estimator.predict(regressor_value.reshape(1, -1))[0]
        return y_output[self.max_lag : :].reshape(-1, 1)

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

        for i in range(forecast_horizon - self.max_lag):
            lagged_data = self.build_matrix(
                X[i : i + analyzed_elements_number],
                yhat[i : i + analyzed_elements_number].reshape(-1, 1),
            )
            X_tmp = self.basis_function.transform(
                lagged_data, self.max_lag, self.ylag, self.xlag, self.model_type
            )

            a = self.base_estimator.predict(X_tmp)
            yhat[i + self.max_lag] = a[0]

        return yhat[self.max_lag :].reshape(-1, 1)

    def _basis_function_n_step_prediction(self, X, y, steps_ahead, forecast_horizon):
        """Perform the n-steps-ahead prediction of a model.

        Parameters
        ----------
        y : array-like of shape = max_lag
            Initial conditions values of the model
            to start recursive process.
        X : ndarray of floats of shape = n_samples
            Vector with input values to be used in model simulation.
        steps_ahead : int (default = None)
            The user can use free run simulation, one-step ahead prediction
            and n-step ahead prediction.
        forecast_horizon : int, default=None
            The number of predictions over the time.

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
                    X[k : i + steps_ahead],
                    y[k : i + steps_ahead],
                    forecast_horizon=forecast_horizon,
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
                    forecast_horizon=forecast_horizon,
                )[-steps_ahead:].ravel()
            else:
                raise ValueError(
                    f"model_type must be NARMAX, NAR or NFIR. Got {self.model_type}"
                )

            i += steps_ahead

        return yhat[self.max_lag : :].reshape(-1, 1)

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
                    X[k : i + steps_ahead],
                    y[k : i + steps_ahead],
                    forecast_horizon,
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
                    forecast_horizon=forecast_horizon,
                )[-forecast_horizon : -forecast_horizon + steps_ahead].ravel()
            else:
                raise ValueError(
                    f"model_type must be NARMAX, NAR or NFIR. Got {self.model_type}"
                )

            i += steps_ahead

        yhat = yhat.ravel()
        return yhat[self.max_lag : :].reshape(-1, 1)
