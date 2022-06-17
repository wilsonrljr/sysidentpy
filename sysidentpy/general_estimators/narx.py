""" Build NARX Models Using general estimators """

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause


import logging
import numpy as np

# from ..base import GenerateRegressors
# from ..base import InformationMatrix
# from ..residues.residues_correlation import ResiduesAnalysis
# from ..utils._check_arrays import check_X_y, _check_positive_int
from collections import Counter
from ..utils.deprecation import deprecated
import sys
from ..narmax_base import GenerateRegressors, InformationMatrix, ModelInformation
from ..utils._check_arrays import _check_positive_int, _num_features, check_X_y


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)


class ModelPrediction:
    def predict(self, X, y, steps_ahead=None):
        """Return the predicted values given an input.

        The predict function allows a friendly usage by the user.
        Given a previously trained model, predict values given
        a new set of data.

        This method accept y values mainly for prediction n-steps ahead
        (to be implemented in the future)

        Parameters
        ----------
        X : ndarray of floats
            The input data to be used in the prediction process.
        y : ndarray of floats
            The output data to be used in the prediction process.
        steps_ahead = int (default = None)
            The forecast horizon.

        Returns
        -------
        yhat : ndarray of floats
            The predicted values of the model.

        """
        if self.basis_function.__class__.__name__ == "Polynomial":
            if steps_ahead is None:
                return self._model_prediction(X, y)
            elif steps_ahead == 1:
                return self._one_step_ahead_prediction(X, y)
            else:
                _check_positive_int(steps_ahead, "steps_ahead")
                return self._n_step_ahead_prediction(X, y, steps_ahead=steps_ahead)
        else:
            if steps_ahead is None:
                return self._basis_function_predict(X, y)
            elif steps_ahead == 1:
                return self._one_step_ahead_prediction(X, y)

    def _code2exponents(self, code):
        """
        Convert regressor code to exponents array.

        Parameters
        ----------
        code : 1D-array of int
            Codification of one regressor.
        """
        regressors = np.array(list(set(code)))
        regressors_count = Counter(code)

        if np.all(regressors == 0):
            return np.zeros(self.max_lag * (1 + self._n_inputs))

        else:
            exponents = np.array([], dtype=float)
            elements = np.round(np.divide(regressors, 1000), 0)[
                (regressors > 0)
            ].astype(int)

            for j in range(1, self._n_inputs + 2):
                base_exponents = np.zeros(self.max_lag, dtype=float)
                if j in elements:
                    for i in range(1, self.max_lag + 1):
                        regressor_code = int(j * 1000 + i)
                        base_exponents[-i] = regressors_count[regressor_code]
                    exponents = np.append(exponents, base_exponents)

                else:
                    exponents = np.append(exponents, base_exponents)

            return exponents

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
        if self.model_type == "NAR":
            lagged_data = self.build_output_matrix(y, self.ylag)
        elif self.model_type == "NFIR":
            lagged_data = self.build_input_matrix(X, self.xlag)
        elif self.model_type == "NARMAX":
            lagged_data = self.build_input_output_matrix(X, y, self.xlag, self.ylag)
        else:
            raise ValueError(
                "Unrecognized model type. The model_type should be NARMAX, NAR or NFIR."
            )

        if self.basis_function.__class__.__name__ == "Polynomial":
            X_base = self.basis_function.transform(
                lagged_data,
                self.max_lag,
                # predefined_regressors=self.pivv[: len(self.final_model)],
            )
        else:
            X_base, _ = self.basis_function.transform(
                lagged_data,
                self.max_lag,
                # predefined_regressors=self.pivv[: len(self.final_model)],
            )

        # yhat = np.dot(X_base, self.theta.flatten())
        yhat = self.base_estimator.predict(X_base)
        yhat = np.concatenate([y[: self.max_lag].flatten(), yhat])
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
            raise Exception("Insufficient initial conditions elements!")

        yhat = np.zeros(X.shape[0], dtype=float)
        yhat.fill(np.nan)
        yhat[: self.max_lag] = y[: self.max_lag, 0]
        i = self.max_lag
        X = X.reshape(-1, self._n_inputs)
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
        elif self.model_type == "NFIR":
            return self._nfir_predict(X, y_initial)
        else:
            raise Exception(
                "model_type do not exist! Model type must be NARMAX, NAR or NFIR"
            )

    def _narmax_predict(self, X, y_initial, forecast_horizon):
        if len(y_initial) < self.max_lag:
            raise Exception("Insufficient initial conditions elements!")

        # X = X.reshape(-1, self._n_inputs)
        if X is not None:
            forecast_horizon = X.shape[0]
        else:
            forecast_horizon = forecast_horizon + self.max_lag

        if self.model_type == "NAR":
            self._n_inputs = 0

        y_output = np.zeros(forecast_horizon, dtype=float)
        y_output.fill(np.nan)
        y_output[: self.max_lag] = y_initial[: self.max_lag, 0]

        model_exponents = [self._code2exponents(model) for model in self.final_model]
        raw_regressor = np.zeros(len(model_exponents[0]), dtype=float)
        for i in range(self.max_lag, forecast_horizon):
            init = 0
            final = self.max_lag
            k = int(i - self.max_lag)
            raw_regressor[:final] = y_output[k:i]
            for j in range(self._n_inputs):
                init += self.max_lag
                final += self.max_lag
                raw_regressor[init:final] = X[k:i, j]

            regressor_value = np.zeros(len(model_exponents))
            for j in range(len(model_exponents)):
                regressor_value[j] = np.prod(
                    np.power(raw_regressor, model_exponents[j])
                )

            # y_output[i] = np.dot(regressor_value, self.theta.flatten())
            y_output[i] = self.base_estimator.predict(regressor_value)
        return y_output.reshape(-1, 1)

    def _nfir_predict(self, X, y_initial):
        y_output = np.zeros(X.shape[0], dtype=float)
        y_output.fill(np.nan)
        y_output[: self.max_lag] = y_initial[: self.max_lag, 0]
        X = X.reshape(-1, self._n_inputs)
        model_exponents = [self._code2exponents(model) for model in self.final_model]
        raw_regressor = np.zeros(len(model_exponents[0]), dtype=float)
        for i in range(self.max_lag, X.shape[0]):
            init = 0
            final = self.max_lag
            k = int(i - self.max_lag)
            for j in range(self._n_inputs):
                raw_regressor[init:final] = X[k:i, j]
                init += self.max_lag
                final += self.max_lag

            regressor_value = np.zeros(len(model_exponents))
            for j in range(len(model_exponents)):
                regressor_value[j] = np.prod(
                    np.power(raw_regressor, model_exponents[j])
                )

            # y_output[i] = np.dot(regressor_value, self.theta.flatten())
            y_output[i] = self.base_estimator.predict(regressor_value)
        return y_output.reshape(-1, 1)

    def _basis_function_predict(self, X, y_initial, forecast_horizon=None):
        if X is not None:
            forecast_horizon = X.shape[0]
        else:
            forecast_horizon = forecast_horizon + self.max_lag

        if self.model_type == "NAR":
            self._n_inputs = 0

        yhat = np.zeros(forecast_horizon, dtype=float)
        yhat.fill(np.nan)
        yhat[: self.max_lag] = y_initial[: self.max_lag, 0]

        # Discard unnecessary initial values
        # yhat[0:self.max_lag] = y_initial[0:self.max_lag]
        analyzed_elements_number = self.max_lag + 1

        for i in range(0, forecast_horizon - self.max_lag):
            if self.model_type == "NARMAX":
                lagged_data = self.build_input_output_matrix(
                    X[i : i + analyzed_elements_number],
                    yhat[i : i + analyzed_elements_number].reshape(-1, 1),
                    self.xlag,
                    self.ylag,
                )
            elif self.model_type == "NAR":
                lagged_data = self.build_output_matrix(
                    yhat[i : i + analyzed_elements_number].reshape(-1, 1), self.ylag
                )
            elif self.model_type == "NFIR":
                lagged_data = self.build_input_matrix(
                    X[i : i + analyzed_elements_number], self.xlag
                )
            else:
                raise ValueError(
                    "Unrecognized model type. The model_type should be NARMAX, NAR or NFIR."
                )

            X_tmp, _ = self.basis_function.transform(
                lagged_data,
                self.max_lag,
                # predefined_regressors=self.pivv[: len(self.final_model)],
            )

            # a = X_tmp @ theta
            a = self.base_estimator.predict(X_tmp)
            yhat[i + self.max_lag] = a[0]

        return yhat.reshape(-1, 1)

    def basis_function_n_step_prediction(self, X, y, steps_ahead, forecast_horizon):
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
            raise Exception("Insufficient initial conditions elements!")

        if X is not None:
            forecast_horizon = X.shape[0]
        else:
            forecast_horizon = forecast_horizon + self.max_lag

        yhat = np.zeros(forecast_horizon, dtype=float)
        yhat.fill(np.nan)
        yhat[: self.max_lag] = y[: self.max_lag, 0]

        # Discard unnecessary initial values
        analyzed_elements_number = self.max_lag + 1
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
                    "Unrecognized model type. The model_type should be NARMAX, NAR or NFIR."
                )

            # yhat[i : i + steps_ahead] = self._basis_function_predict(
            #     X[k : i + steps_ahead], y[k : i + steps_ahead], self.theta
            # )[-steps_ahead:].ravel()

            i += steps_ahead

        # yhat = yhat.ravel()
        return yhat.reshape(-1, 1)

    def _basis_function_n_steps_horizon(self, X, y, steps_ahead, forecast_horizon):
        yhat = np.zeros(forecast_horizon, dtype=float)
        yhat.fill(np.nan)
        yhat[: self.max_lag] = y[: self.max_lag, 0]

        # Discard unnecessary initial values
        analyzed_elements_number = self.max_lag + 1
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
                    "Unrecognized model type. The model_type should be NARMAX, NAR or NFIR."
                )

            i += steps_ahead

        yhat = yhat.ravel()
        return yhat.reshape(-1, 1)


class NARX(GenerateRegressors, InformationMatrix, ModelInformation, ModelPrediction):
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
    >>> from sklearn.linear_model import BayesianRidge
    >>> from sysidentpy.basis_function._basis_function import Polynomial
    >>> from sysidentpy.utils.display_results import results
    >>> from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
    >>> from sysidentpy.residues.residues_correlation import compute_residues_autocorrelation, compute_cross_correlation
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
        ylag=2,
        xlag=2,
        model_type="NARMAX",
        basis_function=None,
        base_estimator=None,
        fit_params={},
    ):

        self.basis_function = basis_function
        self.model_type = model_type
        self.non_degree = basis_function.degree
        self.ylag = ylag
        self.xlag = xlag
        self.max_lag = self._get_max_lag(ylag, xlag)
        self.base_estimator = base_estimator
        self.fit_params = fit_params
        self._validate_params()

    def _validate_params(self):
        """Validate input params."""
        if isinstance(self.ylag, int) and self.ylag < 1:
            raise ValueError("ylag must be integer and > zero. Got %f" % self.ylag)

        if isinstance(self.xlag, int) and self.xlag < 1:
            raise ValueError("xlag must be integer and > zero. Got %f" % self.xlag)

        if not isinstance(self.xlag, (int, list)):
            raise ValueError("xlag must be integer and > zero. Got %f" % self.xlag)

        if not isinstance(self.ylag, (int, list)):
            raise ValueError("ylag must be integer and > zero. Got %f" % self.ylag)

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

        if self.model_type == "NAR":
            lagged_data = self.build_output_matrix(y, self.ylag)
            self.max_lag = self._get_max_lag(ylag=self.ylag)
        elif self.model_type == "NFIR":
            lagged_data = self.build_input_matrix(X, self.xlag)
            self.max_lag = self._get_max_lag(xlag=self.xlag)
        elif self.model_type == "NARMAX":
            check_X_y(X, y)
            self.max_lag = self._get_max_lag(ylag=self.ylag, xlag=self.xlag)
            lagged_data = self.build_input_output_matrix(X, y, self.xlag, self.ylag)
        else:
            raise ValueError(
                "Unrecognized model type. The model_type should be NARMAX, NAR or NFIR."
            )

        if self.basis_function.__class__.__name__ == "Polynomial":
            reg_matrix = self.basis_function.fit(
                lagged_data, self.max_lag, predefined_regressors=None
            )
        else:
            reg_matrix, self.ensemble = self.basis_function.fit(
                lagged_data, self.max_lag, predefined_regressors=None
            )

        if X is not None:
            self._n_inputs = _num_features(X)
        else:
            self._n_inputs = 1  # just to create the regressor space base

        self.regressor_code = self.regressor_space(
            self.non_degree, self.xlag, self.ylag, self._n_inputs, self.model_type
        )
        self.final_model = self.regressor_code
        y = y[self.max_lag :].ravel()

        self.base_estimator.fit(reg_matrix, y, **self.fit_params)
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
            elif steps_ahead == 1:
                return self._one_step_ahead_prediction(X, y)
            else:
                _check_positive_int(steps_ahead, "steps_ahead")
                return self._n_step_ahead_prediction(X, y, steps_ahead=steps_ahead)
        else:
            if steps_ahead is None:
                return self._basis_function_predict(
                    X, y, forecast_horizon=forecast_horizon
                )
            elif steps_ahead == 1:
                return self._one_step_ahead_prediction(X, y)
            else:
                return self.basis_function_n_step_prediction(
                    X, y, steps_ahead=steps_ahead, forecast_horizon=forecast_horizon
                )
