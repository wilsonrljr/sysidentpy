"""Simulation methods for NARMAX models."""

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause

from typing import Union

import numpy as np

from sysidentpy.narmax_base import house, rowhouse
from ..basis_function import Fourier, Polynomial
from ..narmax_base import BaseMSS
from sysidentpy.utils.information_matrix import build_lagged_matrix

from sysidentpy.utils.simulation import (
    get_index_from_regressor_code,
    list_output_regressor_code,
    list_input_regressor_code,
)

from sysidentpy.utils.lags import (
    get_lag_from_regressor_code,
    get_max_lag_from_model_code,
)

from ..utils.check_arrays import check_positive_int, num_features
from ..parameter_estimation.estimators import (
    LeastSquares,
    RidgeRegression,
    RecursiveLeastSquares,
    TotalLeastSquares,
    LeastMeanSquareMixedNorm,
    LeastMeanSquares,
    LeastMeanSquaresFourth,
    LeastMeanSquaresLeaky,
    LeastMeanSquaresNormalizedLeaky,
    LeastMeanSquaresNormalizedSignRegressor,
    LeastMeanSquaresNormalizedSignSign,
    LeastMeanSquaresSignError,
    LeastMeanSquaresSignSign,
    AffineLeastMeanSquares,
    NormalizedLeastMeanSquares,
    NormalizedLeastMeanSquaresSignError,
    LeastMeanSquaresSignRegressor,
)

Estimators = Union[
    LeastSquares,
    RidgeRegression,
    RecursiveLeastSquares,
    TotalLeastSquares,
    LeastMeanSquareMixedNorm,
    LeastMeanSquares,
    LeastMeanSquaresFourth,
    LeastMeanSquaresLeaky,
    LeastMeanSquaresNormalizedLeaky,
    LeastMeanSquaresNormalizedSignRegressor,
    LeastMeanSquaresNormalizedSignSign,
    LeastMeanSquaresSignError,
    LeastMeanSquaresSignSign,
    AffineLeastMeanSquares,
    NormalizedLeastMeanSquares,
    NormalizedLeastMeanSquaresSignError,
    LeastMeanSquaresSignRegressor,
]


class SimulateNARMAX(BaseMSS):
    r"""Simulates a Polynomial NARMAX model.

    The NARMAX (Nonlinear AutoRegressive Moving Average with eXogenous inputs) model
    is described as:

    $$
    y_k = \mathcal{F}^\ell \Big[y_{k-1}, \dotsc, y_{k-n_y}, x_{k-d}, x_{k-d-1},
    \dotsc, x_{k-d-n_x}, e_{k-1}, \dotsc, e_{k-n_e} \Big] + e_k
    $$

    where:

    - $ n_y \in \mathbb{N}^* $, $ n_x \in \mathbb{N} $, and $ n_e \in \mathbb{N} $ are
        the maximum lags for the system output, input, and noise, respectively.
    - $ x_k \in \mathbb{R}^{n_x} $ is the system input, and $ y_k \in \mathbb{R}^{n_y} $
        is the system output at discrete time $ k \in \mathbb{N} $.
    - $ e_k \in \mathbb{R}^{n_e} $ represents uncertainties and possible noise at
        discrete time $ k $.
    - $ \mathcal{F}^\ell $ is a nonlinear function of the input and output regressors
        with nonlinearity degree $ \ell \in \mathbb{N} $.
    - $ d $ is a time delay, typically set to $ d=1 $.

    This class provides tools for simulating NARMAX models using a chosen basis function
    and estimation method.

    Parameters
    ----------
    estimator : Estimators, default=RecursiveLeastSquares()
        The parameter estimation method used for model identification.
    elag : int or list, default=2
        Specifies the maximum lags for the error variables.
        If an integer, it applies to both input and output lags.
        If a list, it should contain specific lag values for different variables.
    estimate_parameter : bool, default=True
        Whether to estimate model parameters. Set to `True` unless pre-estimated
        parameters are provided.
    calculate_err : bool, default=False
        If `True`, uses the Error Reduction Ratio (ERR) algorithm to select regressors.
    model_type : str, default="NARMAX"
        Defines the model type. Supported values: `"NARMAX"`, `"ARX"`, `"OE"`, etc.
    basis_function : Polynomial or Fourier, default=Polynomial()
        The basis function used to define the model's nonlinear terms.
    eps : float, default=np.finfo(np.float64).eps
        A small numerical constant used for normalization.

    Attributes
    ----------
    n_inputs : int
        Number of input variables.
    xlag : int or list
        Lags for the input variables.
    ylag : int or list
        Lags for the output variables.
    n_terms : int
        Number of terms in the final model.
    err : array-like
        Error Reduction Ratio (ERR) values for the selected regressors.
    final_model : array-like
        The structure of the identified model.
    theta : array-like
        Estimated parameters of the model.
    pivv : array-like
        Pivot vector for variable selection.
    non_degree : int
        Degree of nonlinearity used in the model.

    Examples
    --------
    >>> import numpy as np
    >>> from sysidentpy.simulation import SimulateNARMAX
    >>> from sysidentpy.basis_function import Polynomial
    >>> x_train = np.random.rand(1000, 1)
    >>> y_train = np.random.rand(1000, 1)
    >>> basis_function = Polynomial(degree=2)
    >>> simulator = SimulateNARMAX(basis_function=basis_function)
    >>> model = np.array([
    ...     [1001, 0],       # y(k-1)
    ...     [2001, 1001],    # x1(k-1)y(k-1)
    ...     [2002, 0]        # x1(k-2)
    ... ])
    >>> theta = np.array([[0.2, 0.9, 0.1]]).T  # Model parameters
    >>> y_pred = simulator.simulate(
    ...     X_test=x_train, y_test=y_train,
    ...     model_code=model, theta=theta
    ... )
    """

    def __init__(
        self,
        *,
        estimator: Estimators = RecursiveLeastSquares(),
        elag: Union[int, list] = 2,
        estimate_parameter: bool = True,
        calculate_err: bool = False,
        model_type: str = "NARMAX",
        basis_function: Union[Polynomial, Fourier] = Polynomial(),
        eps: np.float64 = np.finfo(np.float64).eps,
    ):
        self.elag = elag
        self.model_type = model_type
        self.basis_function = basis_function
        self.estimator = estimator
        self.estimate_parameter = estimate_parameter
        self.calculate_err = calculate_err
        self.eps = eps
        self.n_inputs = None
        self.xlag = None
        self.ylag = None
        self.n_terms = None
        self.err = None
        self.final_model = None
        self.theta = None
        self.pivv = None
        self.non_degree = None
        self._validate_simulate_params()

    def _validate_simulate_params(self):
        if not isinstance(self.estimate_parameter, bool):
            raise TypeError(
                "estimate_parameter must be False or True. Got"
                f" {self.estimate_parameter}"
            )

        if not isinstance(self.calculate_err, bool):
            raise TypeError(
                f"calculate_err must be False or True. Got {self.calculate_err}"
            )

        if self.basis_function is None:
            raise TypeError(f"basis_function can't be. Got {self.basis_function}")

        if self.model_type not in ["NARMAX", "NAR", "NFIR"]:
            raise ValueError(
                f"model_type must be NARMAX, NAR, or NFIR. Got {self.model_type}"
            )

    def _check_simulate_params(self, y_train, y_test, model_code, steps_ahead, theta):
        if not isinstance(self.basis_function, Polynomial):
            raise NotImplementedError(
                "Currently, SimulateNARMAX only works for polynomial models."
            )

        if y_test is None:
            raise ValueError("y_test cannot be None")

        if not isinstance(model_code, np.ndarray):
            raise TypeError(f"model_code must be an np.np.ndarray. Got {model_code}")

        if not isinstance(steps_ahead, (int, type(None))):
            raise ValueError(
                f"steps_ahead must be None or integer > zero. Got {steps_ahead}"
            )

        if not isinstance(theta, np.ndarray) and not self.estimate_parameter:
            raise TypeError(
                "If estimate_parameter is False, theta must be an np.ndarray. Got"
                f" {theta}"
            )

        if self.estimate_parameter:
            if not all(isinstance(i, np.ndarray) for i in [y_train]):
                raise TypeError(
                    "If estimate_parameter is True, X_train and y_train must be an"
                    f" np.ndarray. Got {type(y_train)}"
                )

    def simulate(
        self,
        *,
        X_train=None,
        y_train=None,
        X_test=None,
        y_test=None,
        model_code=None,
        steps_ahead=None,
        theta=None,
        forecast_horizon=None,
    ):
        """Simulate the response of a NARMAX model based on user-defined parameters.

        This method simulates the system's response using a predefined model structure
        (`model_code`) and estimated parameters (`theta`). It allows for both
        training-based parameter estimation and direct simulation using precomputed
        parameters.

        Parameters
        ----------
        X_train : array-like, shape (n_samples, n_features), optional
            Input data used for parameter estimation during training.
            Required if `estimate_parameter=True`.
        y_train : array-like, shape (n_samples, 1), optional
            Output (target) data used for parameter estimation during training.
            Required if `estimate_parameter=True`.
        X_test : array-like, shape (n_samples, n_features), optional
            Input data used for simulation (prediction).
        y_test : array-like, shape (n_samples, 1), optional
            Output data used as initial conditions for simulation.
        model_code : array-like, shape (n_terms, n_columns)
            Encoded representation of the model's regressors, defining
            the input-output relationships in the system.
        steps_ahead : int, optional
            Number of steps ahead for multi-step prediction. If `None`, defaults to
            one-step-ahead prediction.
        theta : array-like, shape (n_terms, 1), optional
            Precomputed model parameters. Required if `estimate_parameter=False`.
        forecast_horizon : int, optional
            Number of time steps to predict in open-loop forecasting.
            Used mainly for NAR and NARMA-type models.

        Returns
        -------
        yhat : array-like, shape (n_samples, 1)
            Predicted output values of the system based on the given inputs.

        Raises
        ------
        ValueError
            If necessary parameters are missing, such as `y_train` when
            `estimate_parameter=True` or `theta` when `estimate_parameter=False`.

        Notes
        -----
        - If `estimate_parameter=True`, the method first estimates the parameters using
        the provided training data (`X_train`, `y_train`) and the chosen basis function.
        - If `estimate_parameter=False`, the method assumes `theta` contains the model
            parameters.
        - The forecast horizon is automatically adjusted for NAR models if not provided.
        - The method internally computes the lag structure based on `model_code` to
            define regressors.

        Examples
        --------
        >>> import numpy as np
        >>> from sysidentpy.simulation import SimulateNARMAX
        >>> from sysidentpy.basis_function import Polynomial
        >>> X_train = np.random.rand(1000, 1)
        >>> y_train = np.random.rand(1000, 1)
        >>> X_test = np.random.rand(200, 1)
        >>> y_test = np.random.rand(200, 1)
        >>> basis_function = Polynomial(degree=2)
        >>> simulator = SimulateNARMAX(basis_function=basis_function)
        >>> model = np.array([
        ...     [1001, 0],       # y(k-1)
        ...     [2001, 1001],    # x1(k-1)y(k-1)
        ...     [2002, 0]        # x1(k-2)
        ... ])
        >>> theta = np.array([[0.2, 0.9, 0.1]]).T  # Precomputed model parameters
        >>> y_pred = simulator.simulate(
        ...     X_train=X_train, y_train=y_train,
        ...     X_test=X_test, y_test=y_test,
        ...     model_code=model, theta=theta
        ... )

        """
        self._check_simulate_params(y_train, y_test, model_code, steps_ahead, theta)

        if X_test is not None:
            self.n_inputs = num_features(X_test)
        else:
            self.n_inputs = 1  # just to create the regressor space base

        xlag_code = list_input_regressor_code(model_code)
        ylag_code = list_output_regressor_code(model_code)
        self.xlag = get_lag_from_regressor_code(xlag_code)
        self.ylag = get_lag_from_regressor_code(ylag_code)
        self.max_lag = max(self.xlag, self.ylag)
        if self.n_inputs != 1:
            self.xlag = self.n_inputs * [list(range(1, self.max_lag + 1))]

        # for MetaMSS NAR modelling
        if self.model_type == "NAR" and forecast_horizon is None:
            forecast_horizon = y_test.shape[0] - self.max_lag

        self.non_degree = model_code.shape[1]
        regressor_code = self.regressor_space(self.n_inputs)

        self.pivv = get_index_from_regressor_code(regressor_code, model_code)
        self.final_model = regressor_code[self.pivv]
        # to use in the predict function
        self.n_terms = self.final_model.shape[0]
        if self.estimate_parameter and not self.calculate_err:
            self.max_lag = self._get_max_lag()
            lagged_data = build_lagged_matrix(
                X_train, y_train, self.xlag, self.ylag, self.model_type
            )
            psi = self.basis_function.fit(
                lagged_data,
                self.max_lag,
                self.ylag,
                self.xlag,
                self.model_type,
                predefined_regressors=self.pivv,
            )

            self.theta = self.estimator.optimize(
                psi, y_train[self.max_lag :, 0].reshape(-1, 1)
            )
            if self.estimator.unbiased is True:
                self.theta = self.estimator.unbiased_estimator(
                    psi,
                    y_train[self.max_lag :, 0].reshape(-1, 1),
                    self.theta,
                    self.elag,
                    self.max_lag,
                    self.estimator,
                    self.basis_function,
                    self.estimator.uiter,
                )

            self.err = self.n_terms * [0]
        elif not self.estimate_parameter:
            self.theta = theta
            self.err = self.n_terms * [0]
        else:
            self.max_lag = self._get_max_lag()
            lagged_data = build_lagged_matrix(
                X_train, y_train, self.xlag, self.ylag, self.model_type
            )
            psi = self.basis_function.fit(
                lagged_data,
                self.max_lag,
                self.ylag,
                self.xlag,
                self.model_type,
                predefined_regressors=self.pivv,
            )

            _, self.err, _, _ = self.error_reduction_ratio(
                psi, y_train, self.n_terms, self.final_model
            )
            self.theta = self.estimator.optimize(
                psi, y_train[self.max_lag :, 0].reshape(-1, 1)
            )
            if self.estimator.unbiased is True:
                self.theta = self.estimator.unbiased_estimator(
                    psi,
                    y_train[self.max_lag :, 0].reshape(-1, 1),
                    self.theta,
                    self.elag,
                    self.max_lag,
                    self.estimator,
                    self.basis_function,
                    self.estimator.uiter,
                )

        return self.predict(
            X=X_test,
            y=y_test,
            steps_ahead=steps_ahead,
            forecast_horizon=forecast_horizon,
        )

    def error_reduction_ratio(self, psi, y, process_term_number, regressor_code):
        """Perform the Error Reduction Ration algorithm.

        Parameters
        ----------
        psi : array_like
            The information matrix of the model.
        y : array-like
            The target data used in the identification process.
        process_term_number : int
            Number of Process Terms defined by the user.
        regressor_code : array_like
            The regressor code list given the xlag and ylag for a MISO model.

        Returns
        -------
        model_code : array_like
            Model defined by the user to simulate.
        err : array-like
            The respective ERR calculated for each regressor.
        piv : array-like
            Contains the index to put the regressors in the correct order
            based on err values.
        psi_orthogonal : array_like
            The updated and orthogonal information matrix.

        References
        ----------
        - Manuscript: Orthogonal least squares methods and their application
           to non-linear system identification
           https://eprints.soton.ac.uk/251147/1/778742007_content.pdf
        - Manuscript (portuguese): Identificação de Sistemas não Lineares
           Utilizando Modelos NARMAX Polinomiais - Uma Revisão
           e Novos Resultados

        """
        squared_y = np.dot(y[self.max_lag :].T, y[self.max_lag :])
        tmp_psi = psi.copy()
        y = y[self.max_lag :, 0].reshape(-1, 1)
        tmp_y = y.copy()
        dimension = tmp_psi.shape[1]
        piv = np.arange(dimension)
        tmp_err = np.zeros(dimension)
        err = np.zeros(dimension)

        for i in np.arange(0, dimension):
            for j in np.arange(i, dimension):
                # Add `eps` in the denominator to omit division by zero if
                # denominator is zero
                tmp_err[j] = (
                    (np.dot(tmp_psi[i:, j].T, tmp_y[i:]) ** 2)
                    / (np.dot(tmp_psi[i:, j].T, tmp_psi[i:, j]) * squared_y + self.eps)
                )[0, 0]

            if i == process_term_number:
                break

            piv_index = np.argmax(tmp_err[i:]) + i
            err[i] = tmp_err[piv_index]
            tmp_psi[:, [piv_index, i]] = tmp_psi[:, [i, piv_index]]
            piv[[piv_index, i]] = piv[[i, piv_index]]

            v = house(tmp_psi[i:, i])

            row_result = rowhouse(tmp_psi[i:, i:], v)

            tmp_y[i:] = rowhouse(tmp_y[i:], v)

            tmp_psi[i:, i:] = np.copy(row_result)

        tmp_piv = piv[0:process_term_number]
        psi_orthogonal = psi[:, tmp_piv]
        model_code = regressor_code[tmp_piv, :].copy()
        return model_code, err, piv, psi_orthogonal

    def predict(self, *, X=None, y=None, steps_ahead=None, forecast_horizon=None):
        """Return the predicted values given an input.

        The predict function allows a friendly usage by the user.
        Given a previously trained model, predict values given
        a new set of data.

        This method accept y values mainly for prediction n-steps ahead
        (to be implemented in the future)

        Parameters
        ----------
        X : array_like
            The input data to be used in the prediction process.
        y : array_like
            The output data to be used in the prediction process.
        steps_ahead : int
            The user can use free run simulation, one-step ahead prediction
            and n-step ahead prediction. The default is None
        forecast_horizon : int
            The number of predictions over the time. The default is None

        Returns
        -------
        yhat : array_like
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
        x : array_like of shape = n_samples
            Vector with input values to be used in model simulation.

        Returns
        -------
        yhat : array_like
               The 1-step-ahead predicted values of the model.

        """
        lagged_data = build_lagged_matrix(x, y, self.xlag, self.ylag, self.model_type)
        x_base = self.basis_function.transform(
            lagged_data,
            self.max_lag,
            self.ylag,
            self.xlag,
            self.model_type,
            predefined_regressors=self.pivv[: len(self.final_model)],
        )

        yhat = super()._one_step_ahead_prediction(x_base)
        return yhat.reshape(-1, 1)

    def _n_step_ahead_prediction(self, x, y, steps_ahead):
        """Perform the n-steps-ahead prediction of a model.

        Parameters
        ----------
        y : array-like of shape = max_lag
            Initial conditions values of the model
            to start recursive process.
        x : array_like of shape = n_samples
            Vector with input values to be used in model simulation.

        Returns
        -------
        yhat : array_like
               The n-steps-ahead predicted values of the model.

        """
        yhat = super()._n_step_ahead_prediction(x, y, steps_ahead)
        return yhat

    def _model_prediction(self, x, y_initial, forecast_horizon=None):
        """Perform the infinity steps-ahead simulation of a model.

        Parameters
        ----------
        y_initial : array-like of shape = max_lag
            Number of initial conditions values of output
            to start recursive process.
        x : array_like of shape = n_samples
            Vector with input values to be used in model simulation.

        Returns
        -------
        yhat : array_like
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

        y_output = super()._narmax_predict(x, y_initial, forecast_horizon)
        return y_output

    def _nfir_predict(self, x, y_initial):
        y_output = super()._nfir_predict(x, y_initial)
        return y_output

    def _basis_function_predict(self, x, y_initial, forecast_horizon=None):
        """Not implemented."""
        raise NotImplementedError(
            "You can only use Polynomial Basis Function in SimulateNARMAX for now."
        )

    def _basis_function_n_step_prediction(self, x, y, steps_ahead, forecast_horizon):
        """Not implemented."""
        raise NotImplementedError(
            "You can only use Polynomial Basis Function in SimulateNARMAX for now."
        )

    def _basis_function_n_steps_horizon(self, x, y, steps_ahead, forecast_horizon):
        """Not implemented."""
        raise NotImplementedError(
            "You can only use Polynomial Basis Function in SimulateNARMAX for now."
        )

    def fit(self, *, X=None, y=None):
        """Not implemented."""
        raise NotImplementedError(
            "There is no fit method in Simulate because the model is predefined."
        )
