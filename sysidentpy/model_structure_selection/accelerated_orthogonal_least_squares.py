""" NARMAX Models using the Accelerated Orthogonal Least-Squares algorithm"""

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause
from collections import Counter

import numpy as np
from numpy import linalg as LA
from ..narmax_base import (
    GenerateRegressors,
    InformationMatrix,
    ModelInformation,
)
from ..parameter_estimation.estimators import Estimators
from ..utils._check_arrays import _check_positive_int, _num_features, check_X_y


class AOLS(
    Estimators,
    GenerateRegressors,
    ModelInformation,
    InformationMatrix,
):
    r"""Accelerated Orthogonal Least Squares Algorithm

    Build Polynomial NARMAX model using the Accelerated Orthogonal Least-Squares ([1]_).
    This algorithm is based on the Matlab code available on:
    https://github.com/realabolfazl/AOLS/

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
    ylag : int, default=2
        The maximum lag of the output.
    xlag : int, default=2
        The maximum lag of the input.
    k : int, default=1
        The sparsity level.
    L : int, default=1
        Number of selected indices per iteration.
    threshold : float, default=10e10
        The desired accuracy.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from sysidentpy.model_structure_selection import AOLS
    >>> from sysidentpy.basis_function._basis_function import Polynomial
    >>> from sysidentpy.utils.display_results import results
    >>> from sysidentpy.metrics import root_relative_squared_error
    >>> from sysidentpy.utils.generate_data import get_miso_data, get_siso_data
    >>> x_train, x_valid, y_train, y_valid = get_siso_data(n=1000,
    ...                                                    colored_noise=True,
    ...                                                    sigma=0.2,
    ...                                                    train_percentage=90)
    >>> basis_function = Polynomial(degree=2)
    >>> model = AOLS(basis_function=basis_function,
    ...              ylag=2, xlag=2
    ...              )
    >>> model.fit(x_train, y_train)
    >>> yhat = model.predict(x_valid, y_valid)
    >>> rrse = root_relative_squared_error(y_valid, yhat)
    >>> print(rrse)
    0.001993603325328823
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

    References
    ----------
    - Manuscript: Accelerated Orthogonal Least-Squares for Large-Scale
       Sparse Reconstruction
       https://www.sciencedirect.com/science/article/abs/pii/S1051200418305311
    - Code:
       https://github.com/realabolfazl/AOLS/

    """

    def __init__(
        self,
        *,
        ylag=2,
        xlag=2,
        k=1,
        L=1,
        threshold=10e-10,
        model_type="NARMAX",
        estimator="least_squares",
        basis_function=None,
        lam=0.98,
        delta=0.01,
        offset_covariance=0.2,
        mu=0.01,
        eps=np.finfo(np.float64).eps,
        gama=0.2,
        weight=0.02,
    ):
        self.basis_function = basis_function
        self.model_type = model_type
        self.xlag = xlag
        self.ylag = ylag
        self.non_degree = basis_function.degree
        self.max_lag = self._get_max_lag(ylag, xlag)
        self.k = k
        self.L = L
        self.estimator = estimator
        self.threshold = threshold
        super().__init__(
            lam=lam,
            delta=delta,
            offset_covariance=offset_covariance,
            mu=mu,
            eps=eps,
            gama=gama,
            weight=weight,
        )
        self.ensemble = None
        self.res = None
        self._n_inputs = None
        self.theta = None
        self.regressor_code = None
        self.pivv = None
        self.final_model = None
        self.n_terms = None
        self.err = None
        self._validate_params()

    def _validate_params(self):
        """Validate input params."""
        if isinstance(self.ylag, int) and self.ylag < 1:
            raise ValueError(f"ylag must be integer and > zero. Got {self.ylag}")

        if isinstance(self.xlag, int) and self.xlag < 1:
            raise ValueError(f"xlag must be integer and > zero. Got {self.xlag}")

        if not isinstance(self.xlag, (int, list)):
            raise ValueError(f"xlag must be integer and > zero. Got {self.ylag}")

        if not isinstance(self.ylag, (int, list)):
            raise ValueError(f"ylag must be integer and > zero. Got {self.ylag}")

        if not isinstance(self.k, int) or self.k < 1:
            raise ValueError(f"k must be integer and > zero. Got {self.k}")

        if not isinstance(self.L, int) or self.L < 1:
            raise ValueError(f"k must be integer and > zero. Got {self.L}")

        if not isinstance(self.threshold, (int, float)) or self.threshold < 0:
            raise ValueError(
                f"threshold must be integer and > zero. Got {self.threshold}"
            )

    def aols(self, psi, y):
        """Perform the Accelerated Orthogonal Least-Squares algorithm.

        Parameters
        ----------
        y : array-like of shape = n_samples
            The target data used in the identification process.
        psi : ndarray of floats
            The information matrix of the model.

        Returns
        -------
        theta : array-like of shape = number_of_model_elements
            The respective ERR calculated for each regressor.
        piv : array-like of shape = number_of_model_elements
            Contains the index to put the regressors in the correct order
            based on err values.
        residual_norm : float
            The final residual norm.

        References
        ----------
        - Manuscript: Accelerated Orthogonal Least-Squares for Large-Scale
           Sparse Reconstruction
           https://www.sciencedirect.com/science/article/abs/pii/S1051200418305311

        """
        n, m = psi.shape
        theta = np.zeros([m, 1])
        r = y[self.max_lag :].reshape(-1, 1).copy()
        it = 0
        max_iter = int(min(self.k, np.floor(n / self.L)))
        aols_index = np.zeros(max_iter * self.L)
        U = np.zeros([n, max_iter * self.L])
        T = psi.copy()
        while LA.norm(r) > self.threshold and it < max_iter:
            it = it + 1
            temp_in = (it - 1) * self.L
            if it > 1:
                T = T - U[:, temp_in].reshape(-1, 1) @ (
                    U[:, temp_in].reshape(-1, 1).T @ psi
                )

            q = ((r.T @ psi) / np.sum(psi * T, axis=0)).ravel()
            TT = np.sum(T**2, axis=0) * (q**2)
            sub_ind = list(aols_index[:temp_in].astype(int))
            TT[sub_ind] = 0
            sorting_indices = np.argsort(TT)[::-1].ravel()
            aols_index[temp_in : temp_in + self.L] = sorting_indices[: self.L]
            for i in range(self.L):
                TEMP = T[:, sorting_indices[i]].reshape(-1, 1) * q[sorting_indices[i]]
                U[:, temp_in + i] = (TEMP / np.linalg.norm(TEMP, axis=0)).ravel()
                r = r - TEMP
                if i == self.L:
                    break

                T = T - U[:, temp_in + i].reshape(-1, 1) @ (
                    U[:, temp_in + i].reshape(-1, 1).T @ psi
                )
                q = ((r.T @ psi) / np.sum(psi * T, axis=0)).ravel()

        aols_index = aols_index[aols_index > 0].ravel().astype(int)
        residual_norm = LA.norm(r)
        theta[aols_index] = getattr(self, self.estimator)(psi[:, aols_index], y)
        if self.L > 1:
            sorting_indices = np.argsort(np.abs(theta))[::-1]
            aols_index = sorting_indices[: self.k].ravel().astype(int)
            theta[aols_index] = getattr(self, self.estimator)(psi[:, aols_index], y)
            residual_norm = LA.norm(
                y[self.max_lag :].reshape(-1, 1)
                - psi[:, aols_index] @ theta[aols_index]
            )

        pivv = np.argwhere(theta.ravel() != 0).ravel()
        theta = theta[theta != 0]
        return theta.reshape(-1, 1), pivv, residual_norm

    def fit(self, *, X=None, y=None):
        """Fit polynomial NARMAX model using AOLS algorithm.

        The 'fit' function allows a friendly usage by the user.
        Given two arguments, X and y, fit training data.

        Parameters
        ----------
        X : ndarray of floats
            The input data to be used in the training process.
        y : ndarray of floats
            The output data to be used in the training process.

        Returns
        -------
        model : ndarray of int
            The model code representation.
        piv : array-like of shape = number_of_model_elements
            Contains the index to put the regressors in the correct order
            based on err values.
        theta : array-like of shape = number_of_model_elements
            The estimated parameters of the model.
        err : array-like of shape = number_of_model_elements
            The respective ERR calculated for each regressor.
        info_values : array-like of shape = n_regressor
            Vector with values of akaike's information criterion
            for models with N terms (where N is the
            vector position + 1).

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

        (self.theta, self.pivv, self.res) = self.aols(reg_matrix, y)
        if self.basis_function.__class__.__name__ == "Polynomial":
            self.final_model = self.regressor_code[self.pivv, :].copy()
        elif self.basis_function.__class__.__name__ != "Polynomial" and self.ensemble:
            basis_code = np.sort(
                np.tile(
                    self.regressor_code[1:, :], (self.basis_function.repetition, 1)
                ),
                axis=0,
            )
            self.regressor_code = np.concatenate([self.regressor_code[1:], basis_code])
            self.final_model = self.regressor_code[self.pivv, :].copy()
        else:
            self.regressor_code = np.sort(
                np.tile(
                    self.regressor_code[1:, :], (self.basis_function.repetition, 1)
                ),
                axis=0,
            )
            self.final_model = self.regressor_code[self.pivv, :].copy()

        # self.max_lag = self._get_max_lag_from_model_code(self.final_model)
        self.n_terms = len(
            self.theta
        )  # the number of terms we selected (necessary in the 'results' methods)
        self.err = self.n_terms * [
            0
        ]  # just to use the `results` method. Will be changed in next update.
        return self

    def predict(self, X=None, y=None, steps_ahead=None, forecast_horizon=None):
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
                return self._model_prediction(X, y, forecast_horizon=forecast_horizon)
            if steps_ahead == 1:
                return self._one_step_ahead_prediction(X, y)

            _check_positive_int(steps_ahead, "steps_ahead")
            return self._n_step_ahead_prediction(X, y, steps_ahead=steps_ahead)

        if steps_ahead is None:
            return self._basis_function_predict(
                X, y, self.theta, forecast_horizon=forecast_horizon
            )
        if steps_ahead == 1:
            return self._one_step_ahead_prediction(X, y)

        return self.basis_function_n_step_prediction(
            X, y, steps_ahead=steps_ahead, forecast_horizon=forecast_horizon
        )

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
            # self.max_lag = ModelInformation()._get_max_lag(ylag=self.ylag)
        elif self.model_type == "NFIR":
            lagged_data = self.build_input_matrix(X, self.xlag)
            # self.max_lag = ModelInformation()._get_max_lag(xlag=self.xlag)
        elif self.model_type == "NARMAX":
            # check_X_y(X, y)
            # self.max_lag = ModelInformation()._get_max_lag(
            # ylag=self.ylag, xlag=self.xlag)
            lagged_data = self.build_input_output_matrix(X, y, self.xlag, self.ylag)
        else:
            raise ValueError(
                "Unrecognized model type. The model_type should be NARMAX, NAR or NFIR."
            )

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

        # piv_final_model = self.pivv[: len(self.final_model)]
        # X_base = X_base[:, piv_final_model]
        yhat = np.dot(X_base, self.theta.flatten())
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
            for j, model_exponent in enumerate(model_exponents):
                regressor_value[j] = np.prod(np.power(raw_regressor, model_exponent))

            y_output[i] = np.dot(regressor_value, self.theta.flatten())
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
            for j, model_exponent in enumerate(model_exponents):
                regressor_value[j] = np.prod(np.power(raw_regressor, model_exponent))

            y_output[i] = np.dot(regressor_value, self.theta.flatten())
        return y_output.reshape(-1, 1)

    def _basis_function_predict(self, X, y_initial, theta, forecast_horizon=None):
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
                    "Unrecognized model type. The model_type should be NARMAX, NAR or"
                    " NFIR."
                )

            X_tmp, _ = self.basis_function.transform(
                lagged_data,
                self.max_lag,
                predefined_regressors=self.pivv[: len(self.final_model)],
            )

            a = X_tmp @ theta
            yhat[i + self.max_lag] = a[:, 0]

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
        i = self.max_lag

        while i < len(y):
            k = int(i - self.max_lag)
            if i + steps_ahead > len(y):
                steps_ahead = len(y) - i  # predicts the remaining values

            if self.model_type == "NARMAX":
                yhat[i : i + steps_ahead] = self._basis_function_predict(
                    X[k : i + steps_ahead], y[k : i + steps_ahead], self.theta
                )[-steps_ahead:].ravel()
            elif self.model_type == "NAR":
                yhat[i : i + steps_ahead] = self._basis_function_predict(
                    X=None,
                    y_initial=y[k : i + steps_ahead],
                    theta=self.theta,
                    forecast_horizon=forecast_horizon,
                )[-forecast_horizon : -forecast_horizon + steps_ahead].ravel()
            elif self.model_type == "NFIR":
                yhat[i : i + steps_ahead] = self._basis_function_predict(
                    X=X[k : i + steps_ahead],
                    y_initial=y[k : i + steps_ahead],
                    theta=self.theta,
                )[-steps_ahead:].ravel()
            else:
                raise ValueError(
                    "Unrecognized model type. The model_type should be NARMAX, NAR or"
                    " NFIR."
                )

            i += steps_ahead
        return yhat.reshape(-1, 1)

    def _basis_function_n_steps_horizon(self, X, y, steps_ahead, forecast_horizon):
        yhat = np.zeros(forecast_horizon, dtype=float)
        yhat.fill(np.nan)
        yhat[: self.max_lag] = y[: self.max_lag, 0]

        # Discard unnecessary initial values
        i = self.max_lag

        while i < len(y):
            k = int(i - self.max_lag)
            if i + steps_ahead > len(y):
                steps_ahead = len(y) - i  # predicts the remaining values

            if self.model_type == "NARMAX":
                yhat[i : i + steps_ahead] = self._basis_function_predict(
                    X[k : i + steps_ahead], y[k : i + steps_ahead], self.theta
                )[-forecast_horizon : -forecast_horizon + steps_ahead].ravel()
            elif self.model_type == "NAR":
                yhat[i : i + steps_ahead] = self._basis_function_predict(
                    X=None,
                    y_initial=y[k : i + steps_ahead],
                    theta=self.theta,
                    forecast_horizon=forecast_horizon,
                )[-forecast_horizon : -forecast_horizon + steps_ahead].ravel()
            elif self.model_type == "NFIR":
                yhat[i : i + steps_ahead] = self._basis_function_predict(
                    X=X[k : i + steps_ahead],
                    y_initial=y[k : i + steps_ahead],
                    theta=self.theta,
                )[-forecast_horizon : -forecast_horizon + steps_ahead].ravel()
            else:
                raise ValueError(
                    "Unrecognized model type. The model_type should be NARMAX, NAR or"
                    " NFIR."
                )

            i += steps_ahead

        yhat = yhat.ravel()
        return yhat.reshape(-1, 1)
