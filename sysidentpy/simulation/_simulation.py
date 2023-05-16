""" Simulation methods for NARMAX models """

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause

from typing import Union

import numpy as np

from ..basis_function import Fourier, Polynomial
from ..narmax_base import BaseMSS, Orthogonalization
from ..parameter_estimation.estimators import Estimators
from ..utils._check_arrays import _check_positive_int, _num_features


class SimulateNARMAX(Estimators, BaseMSS):
    r"""Simulation of Polynomial NARMAX model

    The NARMAX model is described as:

    $$
        y_k= F^\ell[y_{k-1}, \dotsc, y_{k-n_y},x_{k-d}, x_{k-d-1},
        \dotsc, x_{k-d-n_x}, e_{k-1}, \dotsc, e_{k-n_e}] + e_k
    $$

    where $n_y\in \mathbb{N}^*$, $n_x \in \mathbb{N}$, $n_e \in \mathbb{N}$,
    are the maximum lags for the system output and input respectively;
    $x_k \in \mathbb{R}^{n_x}$ is the system input and $y_k \in \mathbb{R}^{n_y}$
    is the system output at discrete time $k \in \mathbb{N}^n$;
    $e_k \in \mathbb{R}^{n_e}$ stands for uncertainties and possible noise
    at discrete time $k$. In this case, $\mathcal{F}^\ell$ is some nonlinear function
    of the input and output regressors with nonlinearity degree $\ell \in \mathbb{N}$
    and $d$ is a time delay typically set to $d=1$.

    Parameters
    ----------
    estimator : str, default="least_squares"
        The parameter estimation method.
    extended_least_squares : bool, default=False
        Whether to use extended least squares method
        for parameter estimation.
        Note that we define a specific set of noise regressors.
    estimate_parameter : bool, default=False
        Whether to use a method for parameter estimation.
        Must be True if the user do not enter the pre-estimated parameters.
        Note that we define a specific set of noise regressors.
    calculate_err : bool, default=False
        Whether to use a ERR algorithm to the pre-defined regressors.
    lam : float, default=0.98
        Forgetting factor of the Recursive Least Squares method.
    delta : float, default=0.01
        Normalization factor of the P matrix.
    offset_covariance : float, default=0.2
        The offset covariance factor of the affine least mean squares
        filter.
    mu : float, default=0.01
        The convergence coefficient (learning rate) of the filter.
    eps : float
        Normalization factor of the normalized filters.
    gama : float, default=0.2
        The leakage factor of the Leaky LMS method.
    weight : float, default=0.02
        Weight factor to control the proportions of the error norms
        and offers an extra degree of freedom within the adaptation
        of the LMS mixed norm method.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from sysidentpy.simulation import SimulateNARMAX
    >>> from sysidentpy.basis_function._basis_function import Polynomial
    >>> from sysidentpy.metrics import root_relative_squared_error
    >>> from sysidentpy.utils.generate_data import get_miso_data, get_siso_data
    >>> x_train, x_valid, y_train, y_valid = get_siso_data(n=1000,
    ...                                                    colored_noise=True,
    ...                                                    sigma=0.2,
    ...                                                    train_percentage=90)
    >>> basis_function = Polynomial(degree=2)
    >>> s = SimulateNARMAX(basis_function=basis_function)
    >>> model = np.array(
    ...     [
    ...     [1001,    0], # y(k-1)
    ...     [2001, 1001], # x1(k-1)y(k-1)
    ...     [2002,    0], # x1(k-2)
    ...     ]
    ...                 )
    >>> # theta must be a numpy array of shape (n, 1) where n
    ... is the number of regressors
    >>> theta = np.array([[0.2, 0.9, 0.1]]).T
    >>> yhat = s.simulate(
    ...     X_test=x_test,
    ...     y_test=y_test,
    ...     model_code=model,
    ...     theta=theta,
    ...     )
    >>> r = pd.DataFrame(
    ...     results(
    ...         model.final_model, model.theta, model.err,
    ...         model.n_terms, err_precision=8, dtype='sci'
    ...         ),
    ...     columns=['Regressors', 'Parameters', 'ERR'])
    >>> print(r)
        Regressors Parameters         ERR
    0        x1(k-2)     0.9000       0.0
    1         y(k-1)     0.1999       0.0
    2  x1(k-1)y(k-1)     0.1000       0.0
    """

    def __init__(
        self,
        *,
        estimator: str = "recursive_least_squares",
        elag: Union[int, list] = 2,
        extended_least_squares: bool = False,
        lam: float = 0.98,
        delta: float = 0.01,
        offset_covariance: float = 0.2,
        mu: float = 0.01,
        eps: np.float64 = np.finfo(np.float64).eps,
        gama: float = 0.2,
        weight: float = 0.02,
        estimate_parameter: bool = True,
        calculate_err: bool = False,
        model_type: str = "NARMAX",
        basis_function: Union[Polynomial, Fourier] = Polynomial(),
    ):
        super().__init__(
            lam=lam,
            delta=delta,
            offset_covariance=offset_covariance,
            mu=mu,
            eps=eps,
            gama=gama,
            weight=weight,
        )
        self.elag = elag
        self.model_type = model_type
        self.build_matrix = self.get_build_io_method(model_type)
        self.basis_function = basis_function
        self.estimator = estimator
        self.extended_least_squares = extended_least_squares
        self.estimate_parameter = estimate_parameter
        self.calculate_err = calculate_err
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
        if self.basis_function.__class__.__name__ != "Polynomial":
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
        """Simulate a model defined by the user.

        Parameters
        ----------
        X_train : ndarray of floats
            The input data to be used in the training process.
        y_train : ndarray of floats
            The output data to be used in the training process.
        X_test : ndarray of floats
            The input data to be used in the prediction process.
        y_test : ndarray of floats
            The output data (initial conditions) to be used in the prediction process.
        model_code : ndarray of int
            Flattened list of input or output regressors.
        steps_ahead = int, default = None
            The forecast horizon.
        theta : array-like of shape = number_of_model_elements
            The parameters of the model.

        Returns
        -------
        yhat : ndarray of floats
            The predicted values of the model.
        results : string
            Where:
                First column represents each regressor element;
                Second column represents associated parameter;
                Third column represents the error reduction ratio associated
                to each regressor.

        """
        self._check_simulate_params(y_train, y_test, model_code, steps_ahead, theta)

        if X_test is not None:
            self.n_inputs = _num_features(X_test)
        else:
            self.n_inputs = 1  # just to create the regressor space base

        xlag_code = self._list_input_regressor_code(model_code)
        ylag_code = self._list_output_regressor_code(model_code)
        self.xlag = self._get_lag_from_regressor_code(xlag_code)
        self.ylag = self._get_lag_from_regressor_code(ylag_code)
        self.max_lag = max(self.xlag, self.ylag)
        if self.n_inputs != 1:
            self.xlag = self.n_inputs * [list(range(1, self.max_lag + 1))]

        # for MetaMSS NAR modelling
        if self.model_type == "NAR" and forecast_horizon is None:
            forecast_horizon = y_test.shape[0] - self.max_lag

        self.non_degree = model_code.shape[1]
        regressor_code = self.regressor_space(self.n_inputs)

        self.pivv = self._get_index_from_regressor_code(regressor_code, model_code)
        self.final_model = regressor_code[self.pivv]
        # to use in the predict function
        self.n_terms = self.final_model.shape[0]
        if self.estimate_parameter and not self.calculate_err:
            self.max_lag = self._get_max_lag()
            lagged_data = self.build_matrix(X_train, y_train)
            psi = self.basis_function.fit(
                lagged_data, self.max_lag, predefined_regressors=self.pivv
            )

            self.theta = getattr(self, self.estimator)(psi, y_train)
            if self.extended_least_squares is True:
                self.theta = self._unbiased_estimator(
                    psi, y_train, self.theta, self.elag, self.max_lag, self.estimator
                )

            self.err = self.n_terms * [0]
        elif not self.estimate_parameter:
            self.theta = theta
            self.err = self.n_terms * [0]
        else:
            self.max_lag = self._get_max_lag()
            lagged_data = self.build_matrix(X_train, y_train)
            psi = self.basis_function.fit(
                lagged_data, self.max_lag, predefined_regressors=self.pivv
            )

            _, self.err, _, _ = self.error_reduction_ratio(
                psi, y_train, self.n_terms, self.final_model
            )
            self.theta = getattr(self, self.estimator)(psi, y_train)
            if self.extended_least_squares is True:
                self.theta = self._unbiased_estimator(
                    psi, y_train, self.theta, self.non_degree, self.elag, self.max_lag
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
        y : array-like of shape = n_samples
            The target data used in the identification process.
        psi : ndarray of floats
            The information matrix of the model.
        process_term_number : int
            Number of Process Terms defined by the user.

        Returns
        -------
        err : array-like of shape = number_of_model_elements
            The respective ERR calculated for each regressor.
        piv : array-like of shape = number_of_model_elements
            Contains the index to put the regressors in the correct order
            based on err values.
        psi_orthogonal : ndarray of floats
            The updated and orthogonal information matrix.

        References
        ----------
        - Manuscript: Orthogonal least squares methods and their application
           to non-linear system identification
           https://eprints.soton.ac.uk/251147/1/778742007_content.pdf
        - Manuscript (portuguese): Identificação de Sistemas não Lineares
           Utilizando Modelos NARMAX Polinomiais – Uma Revisão
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
                tmp_err[j] = (np.dot(tmp_psi[i:, j].T, tmp_y[i:]) ** 2) / (
                    np.dot(tmp_psi[i:, j].T, tmp_psi[i:, j]) * squared_y + self.eps
                )

            if i == process_term_number:
                break

            piv_index = np.argmax(tmp_err[i:]) + i
            err[i] = tmp_err[piv_index]
            tmp_psi[:, [piv_index, i]] = tmp_psi[:, [i, piv_index]]
            piv[[piv_index, i]] = piv[[i, piv_index]]

            v = Orthogonalization().house(tmp_psi[i:, i])

            row_result = Orthogonalization().rowhouse(tmp_psi[i:, i:], v)

            tmp_y[i:] = Orthogonalization().rowhouse(tmp_y[i:], v)

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
        if self.basis_function.__class__.__name__ == "Polynomial":
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
        if self.basis_function.__class__.__name__ == "Polynomial":
            X_base = self.basis_function.transform(
                lagged_data,
                self.max_lag,
                predefined_regressors=self.pivv[: len(self.final_model)],
            )
        else:
            X_base, _ = self.basis_function.transform(
                lagged_data,
                self.max_lag,
                predefined_regressors=self.pivv[: len(self.final_model)],
            )

        yhat = super()._one_step_ahead_prediction(X_base)
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
        yhat = super()._n_step_ahead_prediction(X, y, steps_ahead)
        return yhat

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

        y_output = super()._narmax_predict(X, y_initial, forecast_horizon)
        return y_output

    def _nfir_predict(self, X, y_initial):
        y_output = super()._nfir_predict(X, y_initial)
        return y_output

    def _basis_function_predict(self, X, y_initial, forecast_horizon=None):
        """not implemented"""
        raise NotImplementedError(
            "You can only use Polynomial Basis Function in SimulateNARMAX for now."
        )

    def _basis_function_n_step_prediction(self, X, y, steps_ahead, forecast_horizon):
        """not implemented"""
        raise NotImplementedError(
            "You can only use Polynomial Basis Function in SimulateNARMAX for now."
        )

    def _basis_function_n_steps_horizon(self, X, y, steps_ahead, forecast_horizon):
        """not implemented"""
        raise NotImplementedError(
            "You can only use Polynomial Basis Function in SimulateNARMAX for now."
        )

    def fit(self, *, X=None, y=None):
        """not implemented"""
        raise NotImplementedError(
            "There is no fit method in Simulate because the model is predefined."
        )
