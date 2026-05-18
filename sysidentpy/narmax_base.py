"""Base classes for NARMAX estimator."""

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause


import math
from abc import ABCMeta, abstractmethod
from collections import Counter
from itertools import chain, combinations_with_replacement
from typing import Any, List, Tuple, Union, Optional

import numpy as np

from sysidentpy._lib._array_api import (
    _asarray,
    _copy,
    _concat,
    _is_numpy_namespace,
    _zeros,
    _pow,
    _to_numpy,
    _vector_norm,
    _vstack,
    device as _device,
    get_namespace,
)
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
        if (
            isinstance(self.basis_function, Polynomial)
            and not getattr(self.basis_function, "include_bias", True)
        ):
            # combinations_with_replacement emits the (0,0,...,0) pure-bias tuple
            # first; Polynomial.fit drops the matching psi column, so drop the row
            # here to keep regressor_code aligned with psi.
            regressor_code = regressor_code[1:]
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
        self._polynomial_narmax_predict_cache = None
        self._polynomial_narmax_predict_cache_key = None

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

    def _predict_on_cpu(
        self,
        *,
        X: Optional[np.ndarray],
        y: np.ndarray,
        steps_ahead: Optional[int],
        forecast_horizon: Optional[int],
        original_xp,
        target_device,
    ) -> np.ndarray:
        """Run predict on CPU and convert the result back to the original backend.

        Sequential NARX prediction (free-run and n-step-ahead) is inherently
        recursive: y[t] depends on y[t-1].  Each iteration operates on a tiny
        regressor vector, so GPU kernel-launch overhead dominates and makes the
        loop ~20-30x slower than NumPy.  This helper transparently moves inputs
        to CPU, runs the existing (fast) NumPy predict path, and returns the
        result in the caller's original namespace/device.
        """
        X_np = _to_numpy(X) if X is not None else None
        y_np = _to_numpy(y)

        original_theta = self.theta
        try:
            if self.theta is not None:
                self.theta = _to_numpy(self.theta)
            yhat_np = self.predict(
                X=X_np,
                y=y_np,
                steps_ahead=steps_ahead,
                forecast_horizon=forecast_horizon,
            )
        finally:
            self.theta = original_theta

        return _asarray(
            yhat_np, xp=original_xp, dtype=y.dtype, target_device=target_device
        )

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

    def _get_polynomial_narmax_predict_cache_key(self):
        final_model = np.asarray(self.final_model)
        degree = getattr(self.basis_function, "degree", None)
        return (
            self.model_type,
            self.max_lag,
            self.n_inputs,
            degree,
            final_model.shape,
            final_model.dtype.str,
            final_model.tobytes(),
        )

    def _get_polynomial_narmax_predict_exponents(self) -> np.ndarray:
        cache_key = self._get_polynomial_narmax_predict_cache_key()
        cached_key = getattr(self, "_polynomial_narmax_predict_cache_key", None)
        if cached_key != cache_key or not hasattr(
            self, "_polynomial_narmax_predict_cache"
        ):
            final_model = np.asarray(self.final_model)
            if final_model.shape[0] == 0:
                exponent_matrix = np.zeros(
                    (0, self.max_lag * (1 + self.n_inputs)),
                    dtype=float,
                )
            else:
                exponent_matrix = np.vstack(
                    [self._code2exponents(code=model) for model in final_model]
                )
            self._polynomial_narmax_predict_cache = exponent_matrix
            self._polynomial_narmax_predict_cache_key = cache_key

        return self._polynomial_narmax_predict_cache

    def _should_use_polynomial_narmax_fast_path(
        self,
        x: Optional[np.ndarray],
        y_initial: np.ndarray,
        forecast_horizon: int,
    ) -> bool:
        if x is None:
            return False

        has_supported_state = (
            self.model_type == "NARMAX"
            and isinstance(self.basis_function, Polynomial)
            and self.max_lag is not None
            and self.n_inputs is not None
            and self.theta is not None
            and self.final_model is not None
            and self.n_inputs > 0
            and y_initial.shape[0] > self.max_lag
            and forecast_horizon >= self.max_lag
        )
        if not has_supported_state:
            return False

        xp = get_namespace(x, y_initial)
        namespace_name = getattr(xp, "__name__", xp.__class__.__name__)
        return (
            _is_numpy_namespace(xp)
            or "torch" in namespace_name
            or "cupy" in namespace_name
        )

    def _shift_regressor_block(
        self,
        xp,
        raw_regressor,
        start: int,
        stop: int,
        value,
    ):
        if stop - start > 1:
            raw_regressor[start : stop - 1] = _copy(
                xp,
                raw_regressor[start + 1 : stop],
            )
        raw_regressor[stop - 1] = value

    def _polynomial_narmax_predict_fast(
        self,
        x: np.ndarray,
        y_initial: np.ndarray,
        forecast_horizon: int,
    ) -> np.ndarray:
        xp = get_namespace(x, y_initial)
        _dtype = x.dtype
        target_device = _device(x, y_initial)
        x = xp.reshape(x, (-1, self.n_inputs))
        n_predictions = forecast_horizon - self.max_lag
        if n_predictions <= 0:
            return xp.reshape(
                _zeros(xp, 0, dtype=_dtype, target_device=target_device),
                (-1, 1),
            )

        model_exponents = _asarray(
            self._get_polynomial_narmax_predict_exponents(),
            xp=xp,
            dtype=_dtype,
            target_device=target_device,
        )
        raw_regressor = _zeros(
            xp,
            model_exponents.shape[1],
            dtype=_dtype,
            target_device=target_device,
        )
        raw_regressor[: self.max_lag] = y_initial[: self.max_lag, 0]
        for input_index in range(self.n_inputs):
            start = self.max_lag * (1 + input_index)
            stop = start + self.max_lag
            raw_regressor[start:stop] = x[: self.max_lag, input_index]

        theta = xp.reshape(
            _asarray(
                self.theta,
                xp=xp,
                dtype=_dtype,
                target_device=target_device,
            ),
            (-1,),
        )
        predictions = _zeros(
            xp,
            n_predictions,
            dtype=_dtype,
            target_device=target_device,
        )

        for step in range(n_predictions):
            regressor_powers = _pow(xp, raw_regressor, model_exponents)
            regressor_value = xp.prod(regressor_powers, axis=1)
            y_next = regressor_value @ theta
            predictions[step] = y_next

            if step == n_predictions - 1:
                continue

            self._shift_regressor_block(
                xp,
                raw_regressor,
                0,
                self.max_lag,
                y_next,
            )
            new_input_index = self.max_lag + step
            for input_index in range(self.n_inputs):
                start = self.max_lag * (1 + input_index)
                stop = start + self.max_lag
                self._shift_regressor_block(
                    xp,
                    raw_regressor,
                    start,
                    stop,
                    x[new_input_index, input_index],
                )

        return xp.reshape(predictions, (-1, 1))

    def _narmax_predict_reference(
        self,
        x: np.ndarray,
        y_initial: np.ndarray,
        forecast_horizon: int = 1,
    ) -> np.ndarray:
        """Return the reference recursive NARMAX prediction."""
        xp = get_namespace(x, y_initial)
        _dtype = x.dtype if x is not None else y_initial.dtype
        target_device = _device(x, y_initial)
        y_output = _zeros(
            xp,
            forecast_horizon,
            dtype=_dtype,
            target_device=target_device,
        )
        y_output = y_output * float("nan")
        y_output[: self.max_lag] = y_initial[: self.max_lag, 0]

        model_exponents = _vstack(
            xp,
            [
                _asarray(
                    self._code2exponents(code=model),
                    xp=xp,
                    target_device=target_device,
                )
                for model in self.final_model
            ],
        )
        raw_regressor = _zeros(
            xp,
            model_exponents.shape[1],
            dtype=_dtype,
            target_device=target_device,
        )
        theta = xp.reshape(
            _asarray(
                self.theta,
                xp=xp,
                dtype=_dtype,
                target_device=target_device,
            ),
            (-1,),
        )
        for i in range(self.max_lag, forecast_horizon):
            init = 0
            final = self.max_lag
            k = int(i - self.max_lag)
            raw_regressor[:final] = y_output[k:i]
            for j in range(self.n_inputs):
                init += self.max_lag
                final += self.max_lag
                raw_regressor[init:final] = x[k:i, j]

            regressor_powers = _pow(xp, raw_regressor, model_exponents)
            regressor_value = xp.prod(regressor_powers, axis=1)
            y_output[i] = regressor_value @ theta
        return xp.reshape(y_output[self.max_lag : :], (-1, 1))

    def _one_step_ahead_prediction(
        self,
        x_base: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Perform the 1-step-ahead prediction of a model.

        Parameters
        ----------
        x_base : ndarray of floats of shape = n_samples
            Regressor matrix with input-output arrays.
        y : ndarray, optional
            Unused placeholder to keep signature compatible with subclasses.

        Returns
        -------
        yhat : ndarray of floats
               The 1-step-ahead predicted values of the model.

        """
        _ = y  # keeps signature aligned with subclasses
        xp = get_namespace(x_base)
        theta = xp.reshape(
            _asarray(
                self.theta,
                xp=xp,
                dtype=x_base.dtype,
                target_device=_device(x_base),
            ),
            (-1,),
        )
        yhat = x_base @ theta
        return xp.reshape(yhat, (-1, 1))

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
        if self._should_use_polynomial_narmax_fast_path(
            x,
            y_initial,
            forecast_horizon,
        ):
            return self._polynomial_narmax_predict_fast(
                x,
                y_initial,
                forecast_horizon,
            )

        return self._narmax_predict_reference(x, y_initial, forecast_horizon)

    @abstractmethod
    def _nfir_predict(self, x: np.ndarray, y_initial: np.ndarray) -> np.ndarray:
        """Nfir predict method."""
        xp = get_namespace(x, y_initial)
        target_device = _device(x, y_initial)
        y_output = _zeros(xp, x.shape[0], dtype=x.dtype, target_device=target_device)
        y_output = y_output * float("nan")
        y_output[: self.max_lag] = y_initial[: self.max_lag, 0]
        x = xp.reshape(x, (-1, self.n_inputs))
        model_exponents = _vstack(
            xp,
            [
                _asarray(
                    self._code2exponents(code=model),
                    xp=xp,
                    target_device=target_device,
                )
                for model in self.final_model
            ],
        )
        raw_regressor = _zeros(
            xp,
            model_exponents.shape[1],
            dtype=x.dtype,
            target_device=target_device,
        )
        theta = xp.reshape(
            _asarray(
                self.theta,
                xp=xp,
                dtype=x.dtype,
                target_device=target_device,
            ),
            (-1,),
        )
        for i in range(self.max_lag, x.shape[0]):
            init = 0
            final = self.max_lag
            k = int(i - self.max_lag)
            raw_regressor[:final] = y_output[k:i]
            for j in range(self.n_inputs):
                init += self.max_lag
                final += self.max_lag
                raw_regressor[init:final] = x[k:i, j]

            regressor_powers = _pow(xp, raw_regressor, model_exponents)
            regressor_value = xp.prod(regressor_powers, axis=1)
            y_output[i] = regressor_value @ theta
        return xp.reshape(y_output[self.max_lag : :], (-1, 1))

    def _nar_step_ahead(self, y: np.ndarray, steps_ahead: int) -> np.ndarray:
        xp = get_namespace(y)
        if len(y) < self.max_lag:
            raise ValueError(
                "Insufficient initial condition elements! Expected at least"
                f" {self.max_lag} elements."
            )

        to_remove = math.ceil((len(y) - self.max_lag) / steps_ahead)
        yhat_length = len(y) + steps_ahead
        yhat = _zeros(xp, yhat_length, dtype=y.dtype, target_device=_device(y))
        yhat = yhat * float("nan")
        yhat[: self.max_lag] = y[: self.max_lag, 0]
        i = self.max_lag

        steps = [step for step in range(0, to_remove * steps_ahead, steps_ahead)]
        if len(steps) > 1:
            for step in steps[:-1]:
                yhat[i : i + steps_ahead] = xp.reshape(
                    self._model_prediction(
                        x=None, y_initial=y[step:i], forecast_horizon=steps_ahead
                    )[-steps_ahead:],
                    (-1,),
                )
                i += steps_ahead

            steps_ahead = int(xp.sum(xp.asarray(xp.isnan(yhat), dtype=xp.int32)))
            yhat[i : i + steps_ahead] = xp.reshape(
                self._model_prediction(x=None, y_initial=y[steps[-1] : i])[
                    -steps_ahead:
                ],
                (-1,),
            )
        else:
            yhat[i : i + steps_ahead] = xp.reshape(
                self._model_prediction(
                    x=None, y_initial=y[0:i], forecast_horizon=steps_ahead
                )[-steps_ahead:],
                (-1,),
            )

        yhat = xp.reshape(yhat, (-1,))[self.max_lag : :]
        return xp.reshape(yhat, (-1, 1))

    def narmax_n_step_ahead(
        self,
        x: np.ndarray,
        y: np.ndarray,
        steps_ahead: Optional[int],
    ) -> np.ndarray:
        """n_steps ahead prediction method for NARMAX model."""
        xp = get_namespace(x, y)
        if len(y) < self.max_lag:
            raise ValueError(
                "Insufficient initial condition elements! Expected at least"
                f" {self.max_lag} elements."
            )

        to_remove = math.ceil((len(y) - self.max_lag) / steps_ahead)
        x = xp.reshape(x, (-1, self.n_inputs))
        yhat = _zeros(xp, x.shape[0], dtype=x.dtype, target_device=_device(x, y))
        yhat = yhat * float("nan")
        yhat[: self.max_lag] = y[: self.max_lag, 0]
        i = self.max_lag
        steps = [step for step in range(0, to_remove * steps_ahead, steps_ahead)]
        if len(steps) > 1:
            for step in steps[:-1]:
                yhat[i : i + steps_ahead] = xp.reshape(
                    self._model_prediction(
                        x=x[step : i + steps_ahead],
                        y_initial=y[step:i],
                    )[-steps_ahead:],
                    (-1,),
                )
                i += steps_ahead

            steps_ahead = int(xp.sum(xp.asarray(xp.isnan(yhat), dtype=xp.int32)))
            yhat[i : i + steps_ahead] = xp.reshape(
                self._model_prediction(
                    x=x[steps[-1] : i + steps_ahead],
                    y_initial=y[steps[-1] : i],
                )[-steps_ahead:],
                (-1,),
            )
        else:
            yhat[i : i + steps_ahead] = xp.reshape(
                self._model_prediction(
                    x=x[0 : i + steps_ahead],
                    y_initial=y[0:i],
                )[-steps_ahead:],
                (-1,),
            )

        yhat = xp.reshape(yhat, (-1,))[self.max_lag : :]
        return xp.reshape(yhat, (-1, 1))

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
        xp = get_namespace(y_initial)
        yhat = _zeros(
            xp,
            forecast_horizon,
            dtype=y_initial.dtype,
            target_device=_device(x, y_initial),
        )
        yhat = yhat * float("nan")
        yhat[: self.max_lag] = y_initial[: self.max_lag, 0]

        # Discard unnecessary initial values
        analyzed_elements_number = self.max_lag + 1

        for i in range(forecast_horizon - self.max_lag):
            if self.model_type == "NARMAX":
                lagged_data = build_input_output_matrix(
                    x[i : i + analyzed_elements_number],
                    xp.reshape(yhat[i : i + analyzed_elements_number], (-1, 1)),
                    self.xlag,
                    self.ylag,
                )
            elif self.model_type == "NAR":
                lagged_data = build_output_matrix(
                    xp.reshape(yhat[i : i + analyzed_elements_number], (-1, 1)),
                    self.ylag,
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

        return xp.reshape(yhat[self.max_lag :], (-1, 1))

    @abstractmethod
    def _basis_function_n_step_prediction(
        self,
        x: Optional[np.ndarray],
        y: np.ndarray,
        steps_ahead: int,
        forecast_horizon: int,
    ) -> np.ndarray:
        """Basis function n step ahead."""
        xp = get_namespace(y)
        yhat = _zeros(xp, forecast_horizon, dtype=y.dtype, target_device=_device(x, y))
        yhat = yhat * float("nan")
        yhat[: self.max_lag] = y[: self.max_lag, 0]

        # Discard unnecessary initial values
        i = self.max_lag

        while i < len(y):
            k = int(i - self.max_lag)
            if i + steps_ahead > len(y):
                steps_ahead = len(y) - i  # predicts the remaining values

            if self.model_type == "NARMAX":
                yhat[i : i + steps_ahead] = xp.reshape(
                    self._basis_function_predict(
                        x[k : i + steps_ahead],
                        y[k : i + steps_ahead],
                        forecast_horizon=forecast_horizon,
                    )[-steps_ahead:],
                    (-1,),
                )
            elif self.model_type == "NAR":
                yhat[i : i + steps_ahead] = xp.reshape(
                    self._basis_function_predict(
                        x=None,
                        y_initial=y[k : i + steps_ahead],
                        forecast_horizon=forecast_horizon,
                    )[-forecast_horizon : -forecast_horizon + steps_ahead],
                    (-1,),
                )
            elif self.model_type == "NFIR":
                yhat[i : i + steps_ahead] = xp.reshape(
                    self._basis_function_predict(
                        x=x[k : i + steps_ahead],
                        y_initial=y[k : i + steps_ahead],
                        forecast_horizon=forecast_horizon,
                    )[-steps_ahead:],
                    (-1,),
                )
            else:
                raise ValueError(
                    f"model_type must be NARMAX, NAR or NFIR. Got {self.model_type}"
                )

            i += steps_ahead

        return xp.reshape(yhat[self.max_lag :], (-1, 1))

    def _basis_function_n_steps_horizon(
        self,
        x: Optional[np.ndarray],
        y: np.ndarray,
        steps_ahead: int,
        forecast_horizon: int,
    ) -> np.ndarray:
        """Basis n steps horizon."""
        xp = get_namespace(y)
        yhat = _zeros(xp, forecast_horizon, dtype=y.dtype, target_device=_device(x, y))
        yhat = yhat * float("nan")
        yhat[: self.max_lag] = y[: self.max_lag, 0]

        # Discard unnecessary initial values
        i = self.max_lag

        while i < len(y):
            k = int(i - self.max_lag)
            if i + steps_ahead > len(y):
                steps_ahead = len(y) - i  # predicts the remaining values

            if self.model_type == "NARMAX":
                yhat[i : i + steps_ahead] = xp.reshape(
                    self._basis_function_predict(
                        x[k : i + steps_ahead], y[k : i + steps_ahead], forecast_horizon
                    )[-forecast_horizon : -forecast_horizon + steps_ahead],
                    (-1,),
                )
            elif self.model_type == "NAR":
                yhat[i : i + steps_ahead] = xp.reshape(
                    self._basis_function_predict(
                        x=None,
                        y_initial=y[k : i + steps_ahead],
                        forecast_horizon=forecast_horizon,
                    )[-forecast_horizon : -forecast_horizon + steps_ahead],
                    (-1,),
                )
            elif self.model_type == "NFIR":
                yhat[i : i + steps_ahead] = xp.reshape(
                    self._basis_function_predict(
                        x=x[k : i + steps_ahead],
                        y_initial=y[k : i + steps_ahead],
                        forecast_horizon=forecast_horizon,
                    )[-forecast_horizon : -forecast_horizon + steps_ahead],
                    (-1,),
                )
            else:
                raise ValueError(
                    f"model_type must be NARMAX, NAR or NFIR. Got {self.model_type}"
                )

            i += steps_ahead

        yhat = xp.reshape(yhat, (-1,))
        return xp.reshape(yhat[self.max_lag :], (-1, 1))


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
    xp = get_namespace(x)
    u = float(_to_numpy(_vector_norm(xp, x)))
    if u != 0:
        eps_value = float(np.finfo(np.float64).eps)
        aux_b = x[0] + xp.sign(x[0]) * u
        x = x[1:] / (aux_b + eps_value)
        x = _concat(
            xp,
            [
                _asarray([1.0], xp=xp, dtype=x.dtype, target_device=_device(x)),
                x,
            ],
        )
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
    xp = get_namespace(RA, v)
    input_was_vector = RA.ndim == 1
    if input_was_vector:
        RA = xp.reshape(RA, (-1, 1))

    v_column = xp.reshape(v, (-1, 1))
    b = -2 / xp.sum(v * v)
    w = b * xp.sum(RA * v_column, axis=0)
    B = RA + v_column * xp.reshape(w, (1, -1))

    if input_was_vector:
        return xp.reshape(B, (-1,))

    return B
