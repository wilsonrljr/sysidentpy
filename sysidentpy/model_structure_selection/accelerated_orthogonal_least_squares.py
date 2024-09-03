"""NARMAX Models using the Accelerated Orthogonal Least-Squares algorithm."""

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause
from typing import Tuple, Union, Optional

import numpy as np
from numpy import linalg as LA

from ..narmax_base import BaseMSS
from ..basis_function import Fourier, Polynomial
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
from ..utils._check_arrays import _check_positive_int, _num_features

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


class AOLS(BaseMSS):
    r"""Accelerated Orthogonal Least Squares Algorithm.

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
        ylag: Union[int, list] = 2,
        xlag: Union[int, list] = 2,
        k: int = 1,
        L: int = 1,
        threshold: float = 10e-10,
        model_type: str = "NARMAX",
        estimator: Estimators = LeastSquares(),
        basis_function: Union[Polynomial, Fourier] = Polynomial(),
    ):
        self.basis_function = basis_function
        self.model_type = model_type
        self.build_matrix = self.get_build_io_method(model_type)
        self.xlag = xlag
        self.ylag = ylag
        self.max_lag = self._get_max_lag()
        self.k = k
        self.L = L
        self.estimator = estimator
        self.threshold = threshold
        self.res = None
        self.n_inputs = None
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
            raise ValueError(f"xlag must be integer and > zero. Got {self.xlag}")

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

    def aols(
        self, psi: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform the Accelerated Orthogonal Least-Squares algorithm.

        Parameters
        ----------
        psi : ndarray of floats
            The information matrix of the model.
        y : array-like of shape = n_samples
            The target data used in the identification process.

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
        theta[aols_index] = self.estimator.optimize(
            psi[:, aols_index], y[self.max_lag :, 0].reshape(-1, 1)
        )
        if self.L > 1:
            sorting_indices = np.argsort(np.abs(theta))[::-1]
            aols_index = sorting_indices[: self.k].ravel().astype(int)
            theta[aols_index] = self.estimator.optimize(
                psi[:, aols_index], y[self.max_lag :, 0].reshape(-1, 1)
            )
            residual_norm = LA.norm(
                y[self.max_lag :].reshape(-1, 1)
                - psi[:, aols_index] @ theta[aols_index]
            )

        pivv = np.argwhere(theta.ravel() != 0).ravel()
        theta = theta[theta != 0]
        return theta.reshape(-1, 1), pivv, residual_norm

    def fit(self, *, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None):
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
        (self.theta, self.pivv, self.res) = self.aols(reg_matrix, y)
        repetition = len(reg_matrix)
        if isinstance(self.basis_function, Polynomial):
            self.final_model = self.regressor_code[self.pivv, :].copy()
        else:
            self.regressor_code = np.sort(
                np.tile(self.regressor_code[1:, :], (repetition, 1)),
                axis=0,
            )
            self.final_model = self.regressor_code[self.pivv, :].copy()

        self.n_terms = len(
            self.theta
        )  # the number of terms we selected (necessary in the 'results' methods)
        self.err = self.n_terms * [
            0
        ]  # just to use the `results` method. Will be changed in next update.
        return self

    def predict(
        self,
        *,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        steps_ahead: Optional[int] = None,
        forecast_horizon: int = 0,
    ) -> np.ndarray:
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

    def _one_step_ahead_prediction(
        self, X: Optional[np.ndarray], y: Optional[np.ndarray]
    ) -> np.ndarray:
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
            lagged_data,
            self.max_lag,
            self.ylag,
            self.xlag,
            self.model_type,
            predefined_regressors=self.pivv[: len(self.final_model)],
        )

        yhat = super()._one_step_ahead_prediction(X_base)
        return yhat.reshape(-1, 1)

    def _n_step_ahead_prediction(
        self,
        X: Optional[np.ndarray],
        y: Optional[np.ndarray],
        steps_ahead: Optional[int],
    ) -> np.ndarray:
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
        yhat = super()._n_step_ahead_prediction(X, y, steps_ahead)
        return yhat

    def _model_prediction(
        self,
        X: Optional[np.ndarray],
        y_initial: Optional[np.ndarray],
        forecast_horizon: int = 1,
    ) -> np.ndarray:
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

    def _narmax_predict(
        self,
        X: Optional[np.ndarray],
        y_initial: Optional[np.ndarray],
        forecast_horizon: int = 1,
    ) -> np.ndarray:
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

    def _nfir_predict(
        self, X: Optional[np.ndarray], y_initial: Optional[np.ndarray]
    ) -> np.ndarray:
        y_output = super()._nfir_predict(X, y_initial)
        return y_output

    def _basis_function_predict(
        self,
        X: Optional[np.ndarray],
        y_initial: Optional[np.ndarray],
        forecast_horizon: int = 1,
    ) -> np.ndarray:
        if X is not None:
            forecast_horizon = X.shape[0]
        else:
            forecast_horizon = forecast_horizon + self.max_lag

        if self.model_type == "NAR":
            self.n_inputs = 0

        yhat = super()._basis_function_predict(X, y_initial, forecast_horizon)
        return yhat.reshape(-1, 1)

    def _basis_function_n_step_prediction(
        self,
        X: Optional[np.ndarray],
        y: Optional[np.ndarray],
        steps_ahead: Optional[int],
        forecast_horizon: int,
    ) -> np.ndarray:
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

        yhat = super()._basis_function_n_step_prediction(
            X, y, steps_ahead, forecast_horizon
        )
        return yhat.reshape(-1, 1)

    def _basis_function_n_steps_horizon(
        self,
        X: Optional[np.ndarray],
        y: Optional[np.ndarray],
        steps_ahead: Optional[int],
        forecast_horizon: int,
    ) -> np.ndarray:
        yhat = super()._basis_function_n_steps_horizon(
            X, y, steps_ahead, forecast_horizon
        )
        return yhat.reshape(-1, 1)
