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
from sysidentpy.utils.information_matrix import build_lagged_matrix
from ..basis_function import Polynomial, Fourier
from ..utils.check_arrays import check_positive_int, num_features

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)


class NARX(BaseMSS):
    r"""NARX model build on top of general estimators.

    The Nonlinear AutoRegressive with eXogenous inputs (NARX) model is mathematically
    described by:

    $$
        y(t) = F\left(\mathbf{\phi}(t)\right) + \epsilon(t)
    $$

    where $\mathbf{\phi}(t)$ is the regression vector composed of lagged inputs and
    outputs:

    $$
        \mathbf{\phi}(t) = [y(t-1), \ldots, y(t-n_y), x(t-1), \ldots, x(t-n_x)]
    $$

    Here, $n_y$ (``ylag``) and $n_x$ (``xlag``) are the maximum lags for the output and
    input, respectively. The function $F$ is approximated by the base estimator. For
    NARMAX models, the regression vector includes lagged residuals:

    $$
        \mathbf{\phi}(t) = [y(t-1), \ldots, y(t-n_y), x(t-1), \ldots, x(t-n_x),
        \epsilon(t-1), \ldots, \epsilon(t-n_e)]
    $$

    where $n_e$ is determined by the basis function and ``model_type`` parameter.

    This implementation uses ``GenerateRegressors`` and ``InformationMatrix`` to
    construct lagged features and allows infinite-step-ahead prediction via iterative
    methods.

    Parameters
    ----------
    ylag : int, default=2
        The maximum lag order of the output $n_y$ (number of past output terms used).
    xlag : int, default=2
        The maximum lag order of the input $n_x$ (number of past input terms used).
    fit_params : dict, default=None
        Additional parameters to pass to the ``fit`` method of the base estimator.
    base_estimator : estimator object, default=None
        An sklearn-compatible estimator with ``fit`` and ``predict`` methods.
    basis_function : basis function object, default=Polynomial
        Nonlinear transformation applied to regressors (e.g., Polynomial, Fourier).
    model_type : {"NARMAX", "NAR", "NFIR"}, default="NARMAX"
        Model structure. Use "NARMAX" to include lagged residuals in the regression
        vector.

    Examples
    --------
    >>> import numpy as np
    >>> from sysidentpy.general_estimators import NARX
    >>> from sklearn.linear_model import BayesianRidge
    >>> from sysidentpy.basis_function import Polynomial
    >>> from sysidentpy.utils.generate_data import get_siso_data
    >>> # Generate data and fit model
    >>> x_train, x_valid, y_train, y_valid = get_siso_data(n=1000)
    >>> basis_function = Polynomial(degree=2)
    >>> model = NARX(
    ...     base_estimator=BayesianRidge(),
    ...     xlag=2,
    ...     ylag=2,
    ...     basis_function=basis_function,
    ...     model_type="NARMAX"
    ... )
    >>> model.fit(x_train, y_train)
    >>> yhat = model.predict(x_valid, y_valid)
    >>> # Evaluation and plotting code here
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

        This is a training pipeline that allows a friendly usage
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
        lagged_data = build_lagged_matrix(X, y, self.xlag, self.ylag, self.model_type)
        reg_matrix = self.basis_function.fit(
            lagged_data,
            self.max_lag,
            self.ylag,
            self.xlag,
            self.model_type,
            predefined_regressors=None,
        )

        if X is not None:
            self.n_inputs = num_features(X)
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
        if isinstance(self.basis_function, Polynomial):
            if steps_ahead is None:
                yhat = self._model_prediction(X, y, forecast_horizon=forecast_horizon)
                yhat = np.concatenate([y[: self.max_lag], yhat], axis=0)
                return yhat

            if steps_ahead == 1:
                yhat = self._one_step_ahead_prediction(X, y)
                yhat = np.concatenate([y[: self.max_lag], yhat], axis=0)
                return yhat

            check_positive_int(steps_ahead, "steps_ahead")
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

    def _one_step_ahead_prediction(self, x, y):
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
        lagged_data = build_lagged_matrix(x, y, self.xlag, self.ylag, self.model_type)
        x_base = self.basis_function.transform(
            lagged_data, self.max_lag, self.ylag, self.xlag, self.model_type
        )

        yhat = self.base_estimator.predict(x_base)
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
                    x=None, y_initial=y[step:i], forecast_horizon=steps_ahead
                )[-steps_ahead:].ravel()
                i += steps_ahead

            steps_ahead = np.sum(np.isnan(yhat))
            yhat[i : i + steps_ahead] = self._model_prediction(
                x=None, y_initial=y[steps[-1] : i]
            )[-steps_ahead:].ravel()
        else:
            yhat[i : i + steps_ahead] = self._model_prediction(
                x=None, y_initial=y[0:i], forecast_horizon=steps_ahead
            )[-steps_ahead:].ravel()

        yhat = yhat.ravel()[self.max_lag : :]
        return yhat.reshape(-1, 1)

    def narmax_n_step_ahead(self, x, y, steps_ahead):
        """N steps ahead prediction method for NARMAX model."""
        if len(y) < self.max_lag:
            raise ValueError(
                "Insufficient initial condition elements! Expected at least"
                f" {self.max_lag} elements."
            )

        to_remove = int(np.ceil((len(y) - self.max_lag) / steps_ahead))
        x = x.reshape(-1, self.n_inputs)
        yhat = np.zeros(x.shape[0], dtype=float)
        yhat.fill(np.nan)
        yhat[: self.max_lag] = y[: self.max_lag, 0]
        i = self.max_lag
        steps = [step for step in range(0, to_remove * steps_ahead, steps_ahead)]
        if len(steps) > 1:
            for step in steps[:-1]:
                yhat[i : i + steps_ahead] = self._model_prediction(
                    x=x[step : i + steps_ahead],
                    y_initial=y[step:i],
                )[-steps_ahead:].ravel()
                i += steps_ahead

            steps_ahead = np.sum(np.isnan(yhat))
            yhat[i : i + steps_ahead] = self._model_prediction(
                x=x[steps[-1] : i + steps_ahead],
                y_initial=y[steps[-1] : i],
            )[-steps_ahead:].ravel()
        else:
            yhat[i : i + steps_ahead] = self._model_prediction(
                x=x[0 : i + steps_ahead],
                y_initial=y[0:i],
            )[-steps_ahead:].ravel()

        yhat = yhat.ravel()[self.max_lag : :]
        return yhat.reshape(-1, 1)

    def _n_step_ahead_prediction(self, x, y, steps_ahead):
        """Perform the n-steps-ahead prediction of a model.

        Parameters
        ----------
        y : array-like of shape = max_lag
            Initial conditions values of the model
            to start recursive process.
        x : ndarray of floats of shape = n_samples
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
            return self.narmax_n_step_ahead(x, y, steps_ahead)

        if self.model_type == "NAR":
            return self._nar_step_ahead(y, steps_ahead)

    def _model_prediction(self, x, y_initial, forecast_horizon=None):
        """Perform the infinity steps-ahead simulation of a model.

        Parameters
        ----------
        y_initial : array-like of shape = max_lag
            Number of initial conditions values of output
            to start recursive process.
        x : ndarray of floats of shape = n_samples
            Vector with input values to be used in model simulation.
        forecast_horizon : int, default=None
            The number of predictions over the time.

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

    def _narmax_predict(self, x, y_initial, forecast_horizon):
        if len(y_initial) < self.max_lag:
            raise ValueError(
                "Insufficient initial condition elements! Expected at least"
                f" {self.max_lag} elements."
            )

        if x is not None:
            forecast_horizon = x.shape[0]
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
                raw_regressor[init:final] = x[k:i, j]

            regressor_value = np.zeros(len(model_exponents))
            for j, model_exponent in enumerate(model_exponents):
                regressor_value[j] = np.prod(np.power(raw_regressor, model_exponent))

            y_output[i] = self.base_estimator.predict(regressor_value.reshape(1, -1))[0]
        return y_output[self.max_lag : :].reshape(-1, 1)

    def _nfir_predict(self, x, y_initial):
        y_output = np.zeros(x.shape[0], dtype=float)
        y_output.fill(np.nan)
        y_output[: self.max_lag] = y_initial[: self.max_lag, 0]
        x = x.reshape(-1, self.n_inputs)
        model_exponents = [
            self._code2exponents(code=model) for model in self.final_model
        ]
        raw_regressor = np.zeros(len(model_exponents[0]), dtype=float)
        for i in range(self.max_lag, x.shape[0]):
            init = 0
            final = self.max_lag
            k = int(i - self.max_lag)
            raw_regressor[:final] = y_output[k:i]
            for j in range(self.n_inputs):
                init += self.max_lag
                final += self.max_lag
                raw_regressor[init:final] = x[k:i, j]

            regressor_value = np.zeros(len(model_exponents))
            for j, model_exponent in enumerate(model_exponents):
                regressor_value[j] = np.prod(np.power(raw_regressor, model_exponent))

            y_output[i] = self.base_estimator.predict(regressor_value.reshape(1, -1))[0]
        return y_output[self.max_lag : :].reshape(-1, 1)

    def _basis_function_predict(self, x, y_initial, forecast_horizon=None):
        if x is not None:
            forecast_horizon = x.shape[0]
        else:
            forecast_horizon = forecast_horizon + self.max_lag

        if self.model_type == "NAR":
            self.n_inputs = 0

        yhat = np.zeros(forecast_horizon, dtype=float)
        yhat.fill(np.nan)
        yhat[: self.max_lag] = y_initial[: self.max_lag, 0]

        analyzed_elements_number = self.max_lag + 1

        for i in range(forecast_horizon - self.max_lag):
            lagged_data = build_lagged_matrix(
                x[i : i + analyzed_elements_number],
                yhat[i : i + analyzed_elements_number].reshape(-1, 1),
                self.xlag,
                self.ylag,
                self.model_type,
            )
            x_tmp = self.basis_function.transform(
                lagged_data, self.max_lag, self.ylag, self.xlag, self.model_type
            )

            a = self.base_estimator.predict(x_tmp)
            yhat[i + self.max_lag] = a[0]

        return yhat[self.max_lag :].reshape(-1, 1)

    def _basis_function_n_step_prediction(self, x, y, steps_ahead, forecast_horizon):
        """Perform the n-steps-ahead prediction of a model.

        Parameters
        ----------
        y : array-like of shape = max_lag
            Initial conditions values of the model
            to start recursive process.
        x : ndarray of floats of shape = n_samples
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
                    x[k : i + steps_ahead],
                    y[k : i + steps_ahead],
                    forecast_horizon=forecast_horizon,
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
                    forecast_horizon=forecast_horizon,
                )[-steps_ahead:].ravel()
            else:
                raise ValueError(
                    f"model_type must be NARMAX, NAR or NFIR. Got {self.model_type}"
                )

            i += steps_ahead

        return yhat[self.max_lag : :].reshape(-1, 1)

    def _basis_function_n_steps_horizon(self, x, y, steps_ahead, forecast_horizon):
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
                    x[k : i + steps_ahead],
                    y[k : i + steps_ahead],
                    forecast_horizon,
                )[-forecast_horizon : -forecast_horizon + steps_ahead].ravel()
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
                    forecast_horizon=forecast_horizon,
                )[-forecast_horizon : -forecast_horizon + steps_ahead].ravel()
            else:
                raise ValueError(
                    f"model_type must be NARMAX, NAR or NFIR. Got {self.model_type}"
                )

            i += steps_ahead

        yhat = yhat.ravel()
        return yhat[self.max_lag : :].reshape(-1, 1)
