"""Base classes for NARMAX estimator."""

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause


from abc import ABCMeta, abstractmethod
from collections import Counter
from itertools import chain, combinations_with_replacement
from typing import Any, List, Tuple, Union, Optional

import numpy as np

from sysidentpy.utils.information_matrix import (
    build_output_matrix,
    build_input_matrix,
    build_input_output_matrix,
)
from .basis_function import Fourier, Polynomial


class RegressorDictionary:
    """Base class for Model Structure Selection."""

    def __init__(
        self,
        xlag: Union[List[Any], Any] = 1,
        ylag: Union[List[Any], Any] = 1,
        basis_function: Union[Polynomial, Fourier] = Polynomial(),
        model_type: str = "NARMAX",
    ):
        self.xlag = xlag
        self.ylag = ylag
        self.basis_function = basis_function
        self.model_type = model_type

    def create_narmax_code(self, n_inputs: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create the code representation of the regressors.

        This function generates a codification from all possibles
        regressors given the maximum lag of the input and output.
        This is used to write the final terms of the model in a
        readable form. [1001] -> y(k-1).
        This code format was based on a dissertation from UFMG. See
        reference below.

        Parameters
        ----------
        n_inputs : int
            Number of input variables.

        Returns
        -------
        x_vec : ndarray of int
            List of the input lags.
        y_vec : ndarray of int
            List of the output lags.

        Examples
        --------
        The codification is defined as:

        100n = y(k-n)
        200n = u(k-n)
        [100n, 100n] = y(k-n)y(k-n)
        [200n, 200n] = u(k-n)u(k-n)

        References
        ----------
        - Master Thesis: Barbosa, Alípio Monteiro.
            Técnicas de otimização bi-objetivo para a determinação
            da estrutura de modelos NARX (2010).

        """
        if self.basis_function.degree < 1:
            raise ValueError(
                f"degree must be integer and > zero. Got {self.basis_function.degree}"
            )

        if np.min(np.minimum(self.ylag, 1)) < 1:
            raise ValueError(
                f"ylag must be integer or list and > zero. Got {self.ylag}"
            )

        if (
            np.min(
                np.min(
                    np.array(list(chain.from_iterable([[self.xlag]])), dtype="object")
                )
            )
            < 1
        ):
            raise ValueError(
                f"xlag must be integer or list and > zero. Got {self.xlag}"
            )

        y_vec = self.get_y_lag_list()

        if n_inputs == 1:
            x_vec = self.get_siso_x_lag_list()
        else:
            x_vec = self.get_miso_x_lag_list(n_inputs)

        return x_vec, y_vec

    def get_y_lag_list(self) -> np.ndarray:
        """Return y regressor code list.

        Returns
        -------
        y_vec = ndarray of ints
            The y regressor code list given the ylag.

        """
        if isinstance(self.ylag, list):
            # create only the lags passed from list
            y_vec = []
            y_vec.extend([lag + 1000 for lag in self.ylag])
            return np.array(y_vec)

        # create a range of lags if passed a int value
        return np.arange(1001, 1001 + self.ylag)

    def get_siso_x_lag_list(self) -> np.ndarray:
        """Return x regressor code list for SISO models.

        Returns
        -------
        x_vec_tmp = ndarray of ints
            The x regressor code list given the xlag for a SISO model.

        """
        if isinstance(self.xlag, list):
            # create only the lags passed from list
            x_vec_tmp = []
            x_vec_tmp.extend([lag + 2000 for lag in self.xlag])
            return np.array(x_vec_tmp)

        # create a range of lags if passed a int value
        return np.arange(2001, 2001 + self.xlag)

    def get_miso_x_lag_list(self, n_inputs: int) -> np.ndarray:
        """Return x regressor code list for MISO models.

        Parameters
        ----------
        n_inputs : int
            Number of input variables.

        Returns
        -------
        x_vec = ndarray of ints
            The x regressor code list given the xlag for a MISO model.

        """
        # only list are allowed if n_inputs > 1
        # the user must entered list of the desired lags explicitly
        x_vec_tmp = []
        for i in range(n_inputs):
            if isinstance(self.xlag[i], list):
                # create 200n, 300n,..., 400n to describe each input
                x_vec_tmp.extend([lag + 2000 + i * 1000 for lag in self.xlag[i]])
            elif isinstance(self.xlag[i], int) and n_inputs > 1:
                x_vec_tmp.extend(
                    [np.arange(2001 + i * 1000, 2001 + i * 1000 + self.xlag[i])]
                )

        # if x_vec is a nested list, ensure all elements are arrays
        all_arrays = [np.array([i]) if isinstance(i, int) else i for i in x_vec_tmp]
        return np.concatenate([i for i in all_arrays])

    def regressor_space(self, n_inputs: int) -> np.ndarray:
        """Create regressor code based on model type.

        Parameters
        ----------
        n_inputs : int
            Number of input variables.

        Returns
        -------
        regressor_code = ndarray of ints
            The regressor code list given the xlag and ylag for a MISO model.

        """
        x_vec, y_vec = self.create_narmax_code(n_inputs)
        reg_aux = np.array([0])
        if self.model_type == "NARMAX":
            reg_aux = np.concatenate([reg_aux, y_vec, x_vec])
        elif self.model_type == "NAR":
            reg_aux = np.concatenate([reg_aux, y_vec])
        elif self.model_type == "NFIR":
            reg_aux = np.concatenate([reg_aux, x_vec])
        else:
            raise ValueError(
                "Unrecognized model type. Model type should be NARMAX, NAR or NFIR"
            )

        regressor_code = list(
            combinations_with_replacement(reg_aux, self.basis_function.degree)
        )

        regressor_code = np.array(regressor_code)
        regressor_code = regressor_code[:, regressor_code.shape[1] :: -1]
        return regressor_code

    def _get_max_lag(self):
        """Get the max lag defined by the user.

        Returns
        -------
        max_lag = int
            The max lag value defined by the user.
        """
        ny = np.max(list(chain.from_iterable([[self.ylag]])))
        nx = np.max(list(chain.from_iterable([[np.array(self.xlag, dtype=object)]])))
        return np.max([ny, np.max(nx)])


class BaseMSS(RegressorDictionary, metaclass=ABCMeta):
    """Base class for Model Structure Selection."""

    @abstractmethod
    def __init__(self):
        super().__init__(self)
        self.max_lag = None
        self.n_inputs = None
        self.theta = None
        self.final_model = None
        self.pivv = None

    @abstractmethod
    def fit(self, *, X, y):
        """Abstract method."""

    @abstractmethod
    def predict(
        self,
        *,
        X: Optional[np.ndarray] = None,
        y: np.ndarray,
        steps_ahead: Optional[int] = None,
        forecast_horizon: int = 1,
    ) -> np.ndarray:
        """Abstract method."""

    def _code2exponents(self, *, code: np.ndarray) -> np.ndarray:
        """Convert regressor code to exponents array.

        Parameters
        ----------
        code : 1D-array of int
            Codification of one regressor.

        Returns
        -------
        exponents = ndarray of ints
        """
        regressors = np.array(list(set(code)))
        regressors_count = Counter(code)

        if np.all(regressors == 0):
            return np.zeros(self.max_lag * (1 + self.n_inputs))

        exponents = np.array([], dtype=float)
        elements = np.round(np.divide(regressors, 1000), 0)[(regressors > 0)].astype(
            int
        )

        for j in range(1, self.n_inputs + 2):
            base_exponents = np.zeros(self.max_lag, dtype=float)
            if j in elements:
                for i in range(1, self.max_lag + 1):
                    regressor_code = int(j * 1000 + i)
                    base_exponents[-i] = regressors_count[regressor_code]
                exponents = np.append(exponents, base_exponents)

            else:
                exponents = np.append(exponents, base_exponents)

        return exponents

    def _one_step_ahead_prediction(self, x_base: np.ndarray) -> np.ndarray:
        """Perform the 1-step-ahead prediction of a model.

        Parameters
        ----------
        x_base : ndarray of floats of shape = n_samples
            Regressor matrix with input-output arrays.

        Returns
        -------
        yhat : ndarray of floats
               The 1-step-ahead predicted values of the model.

        """
        yhat = np.dot(x_base, self.theta.flatten())
        return yhat.reshape(-1, 1)

    @abstractmethod
    def _model_prediction(
        self,
        x: Optional[np.ndarray],
        y_initial: np.ndarray,
        forecast_horizon: int = 1,
    ) -> np.ndarray:
        """Model prediction wrapper."""

    def _narmax_predict(
        self,
        x: np.ndarray,
        y_initial: np.ndarray,
        forecast_horizon: int = 1,
    ) -> np.ndarray:
        """narmax_predict method."""
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

            y_output[i] = np.dot(regressor_value, self.theta.flatten())
        return y_output[self.max_lag : :].reshape(-1, 1)

    @abstractmethod
    def _nfir_predict(self, x: np.ndarray, y_initial: np.ndarray) -> np.ndarray:
        """Nfir predict method."""
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

            y_output[i] = np.dot(regressor_value, self.theta.flatten())
        return y_output[self.max_lag : :].reshape(-1, 1)

    def _nar_step_ahead(self, y: np.ndarray, steps_ahead: int) -> np.ndarray:
        if len(y) < self.max_lag:
            raise ValueError(
                "Insufficient initial condition elements! Expected at least"
                f" {self.max_lag} elements."
            )

        to_remove = int(np.ceil((len(y) - self.max_lag) / steps_ahead))
        yhat_length = len(y) + steps_ahead
        yhat = np.zeros(yhat_length, dtype=float)
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

    def narmax_n_step_ahead(
        self,
        x: np.ndarray,
        y: np.ndarray,
        steps_ahead: Optional[int],
    ) -> np.ndarray:
        """n_steps ahead prediction method for NARMAX model."""
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

    @abstractmethod
    def _n_step_ahead_prediction(
        self,
        x: Optional[np.ndarray],
        y: Optional[np.ndarray],
        steps_ahead: Optional[int],
    ) -> np.ndarray:
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
            Predicted values for NARMAX and NAR models.
        """
        if self.model_type == "NARMAX":
            return self.narmax_n_step_ahead(x, y, steps_ahead)

        if self.model_type == "NAR":
            return self._nar_step_ahead(y, steps_ahead)

        raise ValueError(
            "n_steps_ahead prediction will be implemented for NFIR models in v0.4.*"
        )

    @abstractmethod
    def _basis_function_predict(
        self,
        x: Optional[np.ndarray],
        y_initial: np.ndarray,
        forecast_horizon: int = 1,
    ) -> np.ndarray:
        """Basis function prediction."""
        yhat = np.zeros(forecast_horizon, dtype=float)
        yhat.fill(np.nan)
        yhat[: self.max_lag] = y_initial[: self.max_lag, 0]

        # Discard unnecessary initial values
        analyzed_elements_number = self.max_lag + 1

        for i in range(forecast_horizon - self.max_lag):
            if self.model_type == "NARMAX":
                lagged_data = build_input_output_matrix(
                    x[i : i + analyzed_elements_number],
                    yhat[i : i + analyzed_elements_number].reshape(-1, 1),
                    self.xlag,
                    self.ylag,
                )
            elif self.model_type == "NAR":
                lagged_data = build_output_matrix(
                    yhat[i : i + analyzed_elements_number].reshape(-1, 1), self.ylag
                )
            elif self.model_type == "NFIR":
                lagged_data = build_input_matrix(
                    x[i : i + analyzed_elements_number], self.xlag
                )
            else:
                raise ValueError(
                    f"model_type must be NARMAX, NAR or NFIR. Got {self.model_type}"
                )

            x_tmp = self.basis_function.transform(
                lagged_data,
                self.max_lag,
                self.ylag,
                self.xlag,
                self.model_type,
                predefined_regressors=self.pivv[: len(self.final_model)],
            )

            a = x_tmp @ self.theta
            yhat[i + self.max_lag] = a.item()

        return yhat[self.max_lag :].reshape(-1, 1)

    @abstractmethod
    def _basis_function_n_step_prediction(
        self,
        x: Optional[np.ndarray],
        y: np.ndarray,
        steps_ahead: int,
        forecast_horizon: int,
    ) -> np.ndarray:
        """Basis function n step ahead."""
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

        return yhat[self.max_lag :].reshape(-1, 1)

    def _basis_function_n_steps_horizon(
        self,
        x: Optional[np.ndarray],
        y: np.ndarray,
        steps_ahead: int,
        forecast_horizon: int,
    ) -> np.ndarray:
        """Basis n steps horizon."""
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
                    x[k : i + steps_ahead], y[k : i + steps_ahead], forecast_horizon
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
        return yhat[self.max_lag :].reshape(-1, 1)


def house(x: np.ndarray) -> np.ndarray:
    """Perform a Householder reflection of vector.

    Parameters
    ----------
    x : array-like of shape = number_of_training_samples
        The respective column of the matrix of regressors in each
        iteration of ERR function.

    Returns
    -------
    v : array-like of shape = number_of_training_samples
        The reflection of the array x.

    References
    ----------
    - Manuscript: Chen, S., Billings, S. A., & Luo, W. (1989).
        Orthogonal least squares methods and their application to non-linear
        system identification.

    """
    u = np.linalg.norm(x, 2)
    if u != 0:
        aux_b = x[0] + np.sign(x[0]) * u
        x = x[1:] / (aux_b + np.finfo(np.float64).eps)
        x = np.concatenate((np.array([1]), x))
    return x


def rowhouse(RA: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Perform a row Householder transformation.

    Parameters
    ----------
    RA : array-like of shape = number_of_training_samples
        The respective column of the matrix of regressors in each
        iteration of ERR function.
    v : array-like of shape = number_of_training_samples
        The reflected vector obtained by using the householder reflection.

    Returns
    -------
    B : array-like of shape = number_of_training_samples

    References
    ----------
    - Manuscript: Chen, S., Billings, S. A., & Luo, W. (1989).
        Orthogonal least squares methods and their application to
        non-linear system identification. International Journal of
        control, 50(5), 1873-1896.

    """
    b = -2 / np.dot(v.T, v)
    w = b * np.dot(RA.T, v)
    w = w.reshape(1, -1)
    v = v.reshape(-1, 1)
    RA = RA + v * w
    B = RA
    return B
