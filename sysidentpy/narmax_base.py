"""Base classes for NARMAX estimator."""

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause

import math
from abc import ABCMeta, abstractmethod
from collections import Counter
from copy import copy
from itertools import chain, combinations_with_replacement
from operator import index
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from sysidentpy._config import config_context
from sysidentpy._lib._array_api import (
    _get_namespace_and_device,
    _asarray,
    _copy,
    _concat,
    _is_numpy_namespace,
    _zeros,
    _pow,
    _supports_numpy_metadata_indices,
    _to_numpy,
    _vector_norm,
    device as _device,
    get_namespace,
)
from sysidentpy.utils.information_matrix import (
    build_output_matrix,
    build_input_matrix,
    build_input_output_matrix,
)
from .basis_function import Fourier, Polynomial
from .basis_function.basis_function_base import BaseBasisFunction


def _find_method_owner(cls: type, method_name: str) -> Optional[type]:
    """Return the first class in the MRO that defines a method."""
    for parent in cls.__mro__:
        if method_name in parent.__dict__:
            return parent
    return None


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

    def _build_basis_feature_codes(
        self, base_codes: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        """Build feature codes and report whether their layout is canonical."""
        feature_code_builder = getattr(self.basis_function, "_get_feature_codes", None)
        basis_type = type(self.basis_function)
        feature_code_owner = _find_method_owner(basis_type, "_get_feature_codes")
        fit_owner = _find_method_owner(basis_type, "fit")
        declares_feature_codes = "_get_feature_codes" in basis_type.__dict__
        has_canonical_feature_codes = callable(feature_code_builder) and (
            declares_feature_codes
            or (
                feature_code_owner is not BaseBasisFunction
                and feature_code_owner is fit_owner
            )
        )
        # An inherited hook still provides the best candidate ordering for a
        # custom layout; only native/explicit hooks receive strict width checks.
        if callable(feature_code_builder) and feature_code_owner is not None:
            regressor_code = np.asarray(
                feature_code_builder(
                    base_codes,
                    xlag=self.xlag,
                    ylag=self.ylag,
                    model_type=self.model_type,
                )
            )
        else:
            # Backward-compatible fallback for custom basis functions that do not
            # provide their own feature-code layout.
            regressor_code = np.asarray(
                list(
                    combinations_with_replacement(
                        base_codes, self.basis_function.degree
                    )
                )
            )
            regressor_code = regressor_code[:, ::-1]

        return regressor_code, has_canonical_feature_codes

    def _align_feature_code_width(
        self,
        regressor_code: np.ndarray,
        n_features: Optional[int],
        has_canonical_feature_codes: bool,
    ) -> np.ndarray:
        """Validate feature-code shape and align compatible custom layouts."""
        if regressor_code.ndim != 2:
            raise ValueError(
                f"Feature codes must be a 2D array. Got shape {regressor_code.shape}."
            )
        if regressor_code.shape[1] != self.basis_function.degree:
            raise ValueError(
                "Feature codes must have one column per basis-function degree. "
                f"Expected {self.basis_function.degree} columns, but got "
                f"{regressor_code.shape[1]}."
            )

        if n_features is None:
            return regressor_code

        if has_canonical_feature_codes:
            if regressor_code.shape[0] != n_features:
                raise ValueError(
                    "The basis function generated a feature matrix with "
                    f"{n_features} columns, but its feature-code hook generated "
                    f"{regressor_code.shape[0]} rows."
                )
            return regressor_code

        if regressor_code.shape[0] >= n_features:
            return regressor_code[:n_features]

        if regressor_code.shape[0] == 0:
            return np.zeros((n_features, self.basis_function.degree), dtype=int)

        repetitions = math.ceil(n_features / regressor_code.shape[0])
        return np.tile(regressor_code, (repetitions, 1))[:n_features]

    def regressor_space(
        self,
        n_inputs: int,
        n_features: Optional[int] = None,
    ) -> np.ndarray:
        """Create regressor code based on model type.

        Parameters
        ----------
        n_inputs : int
            Number of input variables.
        n_features : int, optional
            Number of columns generated by the basis function. When provided,
            validate that the feature codes and information matrix are aligned.

        Returns
        -------
        regressor_code = ndarray of ints
            The regressor code list given the xlag and ylag for a MISO model.

        """
        n_features = self._validate_n_features(n_features)
        x_vec, y_vec = self.create_narmax_code(n_inputs)
        base_codes = np.array([0])
        if self.model_type == "NARMAX":
            base_codes = np.concatenate([base_codes, y_vec, x_vec])
        elif self.model_type == "NAR":
            base_codes = np.concatenate([base_codes, y_vec])
        elif self.model_type == "NFIR":
            base_codes = np.concatenate([base_codes, x_vec])
        else:
            raise ValueError(
                "Unrecognized model type. Model type should be NARMAX, NAR or NFIR"
            )

        regressor_code, is_canonical = self._build_basis_feature_codes(base_codes)
        return self._align_feature_code_width(
            regressor_code,
            n_features,
            is_canonical,
        )

    def _validate_n_features(self, n_features: Optional[int]) -> Optional[int]:
        """Validate an optional feature-matrix width."""
        if n_features is None:
            return None

        if isinstance(n_features, bool) or not isinstance(
            n_features, (int, np.integer)
        ):
            raise TypeError(
                f"n_features must be a non-negative integer or None. Got {n_features}"
            )
        if n_features < 0:
            raise ValueError(
                f"n_features must be a non-negative integer or None. Got {n_features}"
            )

        return int(n_features)

    def _regressor_space_for_feature_matrix(
        self,
        n_inputs: int,
        n_features: int,
    ) -> np.ndarray:
        """Build codes for a fitted matrix while preserving legacy overrides.

        Model subclasses could override the historical one-argument public
        ``regressor_space`` method. Internal fit paths therefore call such an
        override with its original signature and only use the new width argument
        for the native implementation.
        """
        if type(self).regressor_space is RegressorDictionary.regressor_space:
            return RegressorDictionary.regressor_space(
                self,
                n_inputs,
                n_features=n_features,
            )

        n_features = self._validate_n_features(n_features)
        regressor_code = np.asarray(self.regressor_space(n_inputs))
        if regressor_code.ndim != 2:
            raise ValueError(
                f"Feature codes must be a 2D array. Got shape {regressor_code.shape}."
            )
        if regressor_code.shape[0] != n_features:
            raise ValueError(
                "The overridden regressor_space generated "
                f"{regressor_code.shape[0]} rows for a feature matrix with "
                f"{n_features} columns."
            )

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
        self._prediction_exponents_cache = None
        self._prediction_exponents_cache_key = None

    @abstractmethod
    def fit(self, *, X, y):
        """Abstract method."""

    def predict(
        self,
        *,
        X: Optional[np.ndarray] = None,
        y: np.ndarray,
        steps_ahead: Optional[int] = None,
        forecast_horizon: int = 1,
    ) -> np.ndarray:
        """Return predictions for the configured model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_inputs), optional
            Input data. It is required for NARMAX and NFIR models. For NAR
            free-run prediction, a provided array preserves the historical
            behavior where its number of rows defines the prediction length.
        y : ndarray of shape (n_samples, 1)
            Output data. The first ``max_lag`` rows are the initial conditions.
        steps_ahead : int, optional
            ``None`` selects free-run prediction, 1 selects one-step-ahead
            prediction, and larger values select blockwise n-step prediction.
        forecast_horizon : int, default=1
            Number of values predicted beyond the initial conditions for a NAR
            free-run prediction when ``X`` is ``None``.

        Returns
        -------
        ndarray of shape (n_predictions, 1)
            Predictions including the initial conditions.
        """
        return self._predict(
            X=X,
            y=y,
            steps_ahead=steps_ahead,
            forecast_horizon=forecast_horizon,
        )

    def _normalize_prediction_integer(
        self,
        value,
        name: str,
        *,
        allow_zero: bool,
    ) -> int:
        """Normalize an integer used by the prediction API."""
        value_dtype = getattr(value, "dtype", None)
        dtype_name = getattr(value_dtype, "name", None)
        if dtype_name is None and value_dtype is not None:
            dtype_name = str(value_dtype).rsplit(".", maxsplit=1)[-1]
        if isinstance(value, (bool, np.bool_)) or dtype_name in ("bool", "bool_"):
            raise ValueError(f"{name} must be an integer. Got {value!r}.")

        try:
            normalized_value = index(value)
        except TypeError as exc:
            raise ValueError(f"{name} must be an integer. Got {value!r}.") from exc

        minimum = 0 if allow_zero else 1
        if normalized_value < minimum:
            comparison = "greater than or equal to zero" if allow_zero else "positive"
            raise ValueError(f"{name} must be {comparison}. Got {normalized_value}.")
        return normalized_value

    def _validate_prediction_inputs(
        self,
        X,
        y,
        steps_ahead,
        forecast_horizon,
    ) -> tuple[Optional[int], Optional[int]]:
        """Validate and normalize inputs at the public prediction boundary."""
        self._validate_prediction_array_shapes(X, y)

        normalized_steps = None
        if steps_ahead is not None:
            normalized_steps = self._normalize_prediction_integer(
                steps_ahead,
                "steps_ahead",
                allow_zero=False,
            )

        if self.model_type in ("NARMAX", "NFIR"):
            self._validate_input_model_prediction(X, y, normalized_steps)
            return normalized_steps, forecast_horizon

        if self.model_type != "NAR":
            raise ValueError(
                f"model_type must be NARMAX, NAR or NFIR. Got {self.model_type}"
            )

        normalized_horizon = self._validate_nar_prediction(
            X,
            normalized_steps,
            forecast_horizon,
        )
        return normalized_steps, normalized_horizon

    def _validate_prediction_array_shapes(self, X, y) -> None:
        """Validate prediction arrays and the initial-condition length."""
        if y is None:
            raise ValueError("y cannot be None for prediction.")
        if getattr(y, "ndim", None) != 2 or y.shape[1] != 1:
            raise ValueError(
                "y must be a 2D array with a single column. "
                f"Got shape {getattr(y, 'shape', None)}."
            )
        if y.shape[0] < self.max_lag:
            raise ValueError(
                "Insufficient initial condition elements. Expected at least "
                f"{self.max_lag} samples, got {y.shape[0]}."
            )

        if X is not None and getattr(X, "ndim", None) != 2:
            raise ValueError(
                f"X must be a 2D array. Got shape {getattr(X, 'shape', None)}."
            )

    def _validate_input_model_prediction(self, X, y, steps_ahead) -> None:
        """Validate a NARMAX or NFIR prediction request."""
        if X is None:
            raise ValueError(f"X cannot be None for {self.model_type} prediction.")
        if self.n_inputs is not None and X.shape[1] != self.n_inputs:
            raise ValueError(
                f"X must have {self.n_inputs} input column(s). Got {X.shape[1]}."
            )
        if steps_ahead is not None and X.shape[0] != y.shape[0]:
            raise ValueError(
                "X and y must contain the same number of samples for "
                f"{self.model_type} one-step and n-step prediction. "
                f"Got {X.shape[0]} and {y.shape[0]}."
            )
        if steps_ahead is None and X.shape[0] < self.max_lag:
            raise ValueError(
                f"X must contain at least {self.max_lag} samples for free-run "
                f"prediction. Got {X.shape[0]}."
            )

    def _validate_nar_prediction(self, X, steps_ahead, forecast_horizon):
        """Validate a NAR request and return its normalized free-run horizon."""
        if steps_ahead is not None:
            return forecast_horizon
        if X is not None:
            if X.shape[0] < self.max_lag:
                raise ValueError(
                    f"X must contain at least {self.max_lag} samples for "
                    f"free-run prediction. Got {X.shape[0]}."
                )
            return forecast_horizon
        if forecast_horizon is None:
            raise ValueError(
                "forecast_horizon cannot be None when X is None for NAR "
                "free-run prediction."
            )

        return self._normalize_prediction_integer(
            forecast_horizon,
            "forecast_horizon",
            allow_zero=True,
        )

    def _prediction_has_empty_suffix(
        self,
        X,
        y,
        steps_ahead: Optional[int],
        forecast_horizon: Optional[int],
    ) -> bool:
        """Return whether a prediction request contains no target samples."""
        if steps_ahead is not None:
            return y.shape[0] == self.max_lag
        if self.model_type == "NAR" and X is None:
            return forecast_horizon == 0
        return X.shape[0] == self.max_lag

    def _prediction_dispatch(
        self,
        *,
        X,
        y,
        steps_ahead: Optional[int],
        forecast_horizon: Optional[int],
    ):
        """Dispatch validated prediction inputs and prepend initial conditions."""
        xp = get_namespace(X, y)
        prefix = y[: self.max_lag, ...]
        if self._prediction_has_empty_suffix(X, y, steps_ahead, forecast_horizon):
            return _copy(xp, prefix)

        if self.model_type == "NFIR":
            yhat = self._one_step_ahead_prediction(X, y)
            return self._prepend_initial_conditions(xp, prefix, yhat)

        if isinstance(self.basis_function, Polynomial):
            if steps_ahead is None:
                yhat = self._model_prediction(X, y, forecast_horizon=forecast_horizon)
            elif steps_ahead == 1:
                yhat = self._one_step_ahead_prediction(X, y)
            else:
                yhat = self._n_step_ahead_prediction(X, y, steps_ahead=steps_ahead)
        elif steps_ahead is None:
            yhat = self._basis_function_predict(
                X,
                y,
                forecast_horizon=forecast_horizon,
            )
        elif steps_ahead == 1:
            yhat = self._one_step_ahead_prediction(X, y)
        else:
            yhat = self._basis_function_n_step_prediction(
                X,
                y,
                steps_ahead=steps_ahead,
                forecast_horizon=forecast_horizon,
            )
        return self._prepend_initial_conditions(xp, prefix, yhat)

    def _prepend_initial_conditions(self, xp, prefix, yhat):
        """Return predictions in the public namespace with one prefix."""
        target_device = _device(prefix)
        yhat = _asarray(
            yhat,
            xp=xp,
            target_device=target_device,
        )
        output_dtype = self._prediction_dtype(xp, prefix.dtype, yhat.dtype)
        yhat = _asarray(
            yhat,
            xp=xp,
            dtype=output_dtype,
            target_device=target_device,
        )
        prefix = _asarray(
            prefix,
            xp=xp,
            dtype=output_dtype,
            target_device=target_device,
        )
        return _concat(xp, [prefix, yhat], axis=0)

    def _predict(
        self,
        *,
        X,
        y,
        steps_ahead,
        forecast_horizon,
        allow_cpu_fallback: bool = True,
    ):
        """Implement the shared public prediction workflow."""
        steps_ahead, forecast_horizon = self._validate_prediction_inputs(
            X,
            y,
            steps_ahead,
            forecast_horizon,
        )
        if self.model_type == "NAR" and steps_ahead is not None:
            X = None
        xp, target_device = _get_namespace_and_device(X, y)
        if self._prediction_has_empty_suffix(X, y, steps_ahead, forecast_horizon):
            return self._prediction_dispatch(
                X=X,
                y=y,
                steps_ahead=steps_ahead,
                forecast_horizon=forecast_horizon,
            )

        requires_recursive_fallback = self.model_type != "NFIR" and steps_ahead != 1
        requires_basis_fallback = not isinstance(
            self.basis_function, Polynomial
        ) and not _supports_numpy_metadata_indices(xp)
        requires_cpu_fallback = requires_recursive_fallback or requires_basis_fallback
        if allow_cpu_fallback and requires_cpu_fallback and not _is_numpy_namespace(xp):
            return self._predict_on_cpu(
                X=X,
                y=y,
                steps_ahead=steps_ahead,
                forecast_horizon=forecast_horizon,
                original_xp=xp,
                target_device=target_device,
            )
        return self._prediction_dispatch(
            X=X,
            y=y,
            steps_ahead=steps_ahead,
            forecast_horizon=forecast_horizon,
        )

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
        loop slower than NumPy. This boundary also supports basis functions that
        still use NumPy metadata for column selection. Inputs are transferred
        once, prediction runs on a shallow model copy, and the result is returned
        in the caller's original namespace and device.
        """
        X_np = _to_numpy(X) if X is not None else None
        y_np = _to_numpy(y)

        if isinstance(self.basis_function, Polynomial) and self.final_model is not None:
            self._get_prediction_exponents()
        cpu_model = copy(self)
        theta = getattr(self, "theta", None)
        if theta is not None:
            cpu_model.theta = _to_numpy(theta)
        with config_context(array_api_dispatch=False):
            yhat_np = cpu_model._prediction_dispatch(
                X=X_np,
                y=y_np,
                steps_ahead=steps_ahead,
                forecast_horizon=forecast_horizon,
            )

        output_dtype = self._prediction_output_dtype(
            original_xp,
            X,
            y,
            prediction_dtype=yhat_np.dtype,
        )

        return _asarray(
            yhat_np,
            xp=original_xp,
            dtype=output_dtype,
            target_device=target_device,
        )

    def _prediction_n_inputs(self) -> int:
        """Return the number of inputs used by prediction kernels."""
        if self.model_type == "NAR":
            return 0
        return self.n_inputs

    def _prediction_dtype(self, xp, *dtypes):
        """Resolve a floating compute dtype for prediction operands."""
        normalized_dtypes = []
        default_float = xp.asarray(1.0).dtype
        for dtype in dtypes:
            if dtype is None:
                continue
            if _is_numpy_namespace(xp):
                normalized = (
                    default_float if np.dtype(dtype).kind in "biu" else np.dtype(dtype)
                )
            else:
                dtype_name = getattr(dtype, "name", None)
                if dtype_name is None:
                    dtype_name = str(dtype).rsplit(".", maxsplit=1)[-1]
                if dtype_name == "bool_":
                    dtype_name = "bool"
                namespace_dtype = getattr(xp, dtype_name, dtype)
                normalized = (
                    default_float
                    if xp.isdtype(namespace_dtype, ("bool", "integral"))
                    else namespace_dtype
                )
            normalized_dtypes.append(normalized)

        if not normalized_dtypes:
            return default_float
        result_dtype = normalized_dtypes[0]
        for dtype in normalized_dtypes[1:]:
            result_dtype = xp.result_type(result_dtype, dtype)
        return result_dtype

    def _prediction_output_dtype(self, xp, X, y, prediction_dtype=None):
        """Return the public output dtype for the active prediction namespace."""
        data_dtype = y.dtype if self.model_type == "NAR" or X is None else X.dtype
        theta = getattr(self, "theta", None)
        theta_dtype = getattr(theta, "dtype", None)
        return self._prediction_dtype(
            xp,
            data_dtype,
            y.dtype,
            theta_dtype,
            prediction_dtype,
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

        n_inputs = self._prediction_n_inputs()
        if np.all(regressors == 0):
            return np.zeros(self.max_lag * (1 + n_inputs))

        exponents = np.array([], dtype=float)
        elements = np.round(np.divide(regressors, 1000), 0)[(regressors > 0)].astype(
            int
        )

        for j in range(1, n_inputs + 2):
            base_exponents = np.zeros(self.max_lag, dtype=float)
            if j in elements:
                for i in range(1, self.max_lag + 1):
                    regressor_code = int(j * 1000 + i)
                    base_exponents[-i] = regressors_count[regressor_code]
                exponents = np.append(exponents, base_exponents)

            else:
                exponents = np.append(exponents, base_exponents)

        return exponents

    def _get_prediction_exponents_cache_key(self):
        final_model = np.asarray(self.final_model)
        degree = getattr(self.basis_function, "degree", None)
        return (
            self.model_type,
            self.max_lag,
            self._prediction_n_inputs(),
            degree,
            final_model.shape,
            final_model.dtype.str,
            final_model.tobytes(),
        )

    def _get_prediction_exponents(self) -> np.ndarray:
        """Return a cached exponent matrix for the selected Polynomial model."""
        cache_key = self._get_prediction_exponents_cache_key()
        cached_key = getattr(self, "_prediction_exponents_cache_key", None)
        if cached_key != cache_key or not hasattr(self, "_prediction_exponents_cache"):
            final_model = np.asarray(self.final_model)
            if final_model.shape[0] == 0:
                exponent_matrix = np.zeros(
                    (0, self.max_lag * (1 + self._prediction_n_inputs())),
                    dtype=float,
                )
            else:
                exponent_matrix = np.vstack(
                    [self._code2exponents(code=model) for model in final_model]
                )
            exponent_matrix.setflags(write=False)
            self._prediction_exponents_cache = exponent_matrix
            self._prediction_exponents_cache_key = cache_key

        return self._prediction_exponents_cache

    def _get_polynomial_narmax_predict_exponents(self) -> np.ndarray:
        """Return Polynomial prediction exponents for backward compatibility."""
        return self._get_prediction_exponents()

    def _narmax_predict_reference(
        self,
        x: np.ndarray,
        y_initial: np.ndarray,
        forecast_horizon: int = 1,
    ) -> np.ndarray:
        """Return the reference recursive NARMAX prediction."""
        xp = get_namespace(x, y_initial)
        data_dtype = y_initial.dtype if self.model_type == "NAR" else x.dtype
        _dtype = self._prediction_dtype(
            xp,
            data_dtype,
            y_initial.dtype,
            getattr(self.theta, "dtype", None),
        )

        target_device = _device(x, y_initial)
        y_output = _zeros(
            xp,
            forecast_horizon,
            dtype=_dtype,
            target_device=target_device,
        )
        y_output = y_output * float("nan")
        y_output[: self.max_lag] = _asarray(
            y_initial[: self.max_lag, 0],
            xp=xp,
            dtype=_dtype,
            target_device=target_device,
        )

        model_exponents = _asarray(
            self._get_prediction_exponents(),
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
            for j in range(self._prediction_n_inputs()):
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
        target_device = _device(x_base)
        prediction_dtype = self._prediction_dtype(
            xp,
            x_base.dtype,
            getattr(self.theta, "dtype", None),
        )
        x_base = _asarray(
            x_base,
            xp=xp,
            dtype=prediction_dtype,
            target_device=target_device,
        )
        theta = xp.reshape(
            _asarray(
                self.theta,
                xp=xp,
                dtype=prediction_dtype,
                target_device=target_device,
            ),
            (-1,),
        )
        yhat = x_base @ theta
        return xp.reshape(yhat, (-1, 1))

    def _model_prediction(
        self,
        x: Optional[np.ndarray],
        y_initial: np.ndarray,
        forecast_horizon: int = 1,
    ) -> np.ndarray:
        """Dispatch free-run prediction to the configured model kernel."""
        if self.model_type in ("NARMAX", "NAR"):
            return self._narmax_predict(x, y_initial, forecast_horizon)
        if self.model_type == "NFIR":
            return self._nfir_predict(x, y_initial)
        raise ValueError(
            f"model_type must be NARMAX, NAR or NFIR. Got {self.model_type}"
        )

    def _resolve_recursive_prediction_length(self, x, forecast_horizon: int) -> int:
        """Return the total recursive buffer length for a prediction request."""
        if x is not None:
            return x.shape[0]
        return forecast_horizon + self.max_lag

    def _narmax_predict(
        self,
        x: np.ndarray,
        y_initial: np.ndarray,
        forecast_horizon: int = 1,
    ) -> np.ndarray:
        """narmax_predict method."""
        if y_initial.shape[0] < self.max_lag:
            raise ValueError(
                "Insufficient initial condition elements! Expected at least"
                f" {self.max_lag} elements."
            )
        prediction_length = self._resolve_recursive_prediction_length(
            x,
            forecast_horizon,
        )
        return self._narmax_predict_reference(x, y_initial, prediction_length)

    def _nfir_predict(self, x: np.ndarray, y_initial: np.ndarray) -> np.ndarray:
        """Nfir predict method."""
        xp = get_namespace(x, y_initial)
        target_device = _device(x, y_initial)
        prediction_dtype = self._prediction_dtype(
            xp,
            x.dtype,
            y_initial.dtype,
            getattr(self.theta, "dtype", None),
        )
        x = _asarray(
            x,
            xp=xp,
            dtype=prediction_dtype,
            target_device=target_device,
        )
        y_output = _zeros(
            xp,
            x.shape[0],
            dtype=prediction_dtype,
            target_device=target_device,
        )
        y_output = y_output * float("nan")
        y_output[: self.max_lag] = _asarray(
            y_initial[: self.max_lag, 0],
            xp=xp,
            dtype=prediction_dtype,
            target_device=target_device,
        )
        x = xp.reshape(x, (-1, self.n_inputs))
        model_exponents = _asarray(
            self._get_prediction_exponents(),
            xp=xp,
            dtype=prediction_dtype,
            target_device=target_device,
        )
        raw_regressor = _zeros(
            xp,
            model_exponents.shape[1],
            dtype=prediction_dtype,
            target_device=target_device,
        )
        theta = xp.reshape(
            _asarray(
                self.theta,
                xp=xp,
                dtype=prediction_dtype,
                target_device=target_device,
            ),
            (-1,),
        )
        for i in range(self.max_lag, x.shape[0]):
            init = 0
            final = self.max_lag
            k = int(i - self.max_lag)
            raw_regressor[:final] = y_output[k:i]
            for j in range(self._prediction_n_inputs()):
                init += self.max_lag
                final += self.max_lag
                raw_regressor[init:final] = x[k:i, j]

            regressor_powers = _pow(xp, raw_regressor, model_exponents)
            regressor_value = xp.prod(regressor_powers, axis=1)
            y_output[i] = regressor_value @ theta
        return xp.reshape(y_output[self.max_lag : :], (-1, 1))

    def _store_prediction_block(
        self,
        *,
        xp,
        prediction_buffer,
        block_prediction,
        block_horizon: int,
        output_start: int,
        n_predictions: int,
        target_device,
    ):
        """Store one recursive block without narrowing the predictor dtype."""
        block_values = xp.reshape(
            _asarray(
                block_prediction[-block_horizon:],
                xp=xp,
                target_device=target_device,
            ),
            (-1,),
        )
        prediction_dtype = self._prediction_dtype(
            xp,
            getattr(prediction_buffer, "dtype", None),
            block_values.dtype,
        )
        if prediction_buffer is None:
            prediction_buffer = _zeros(
                xp,
                n_predictions,
                dtype=prediction_dtype,
                target_device=target_device,
            )
        elif prediction_buffer.dtype != prediction_dtype:
            prediction_buffer = _asarray(
                prediction_buffer,
                xp=xp,
                dtype=prediction_dtype,
                target_device=target_device,
            )

        block_values = _asarray(
            block_values,
            xp=xp,
            dtype=prediction_dtype,
            target_device=target_device,
        )
        prediction_buffer[output_start : output_start + block_horizon] = block_values
        return prediction_buffer

    def _blockwise_prediction(self, *, x, y, steps_ahead: int) -> np.ndarray:
        """Return recursive predictions over observation-anchored blocks."""
        xp = get_namespace(x, y)
        target_device = _device(x, y)
        prediction_dtype = self._prediction_dtype(
            xp,
            getattr(x, "dtype", None),
            y.dtype,
            getattr(getattr(self, "theta", None), "dtype", None),
        )
        if x is not None:
            x = xp.reshape(
                _asarray(
                    x,
                    xp=xp,
                    dtype=prediction_dtype,
                    target_device=target_device,
                ),
                (-1, self._prediction_n_inputs()),
            )

        n_samples = y.shape[0]
        n_predictions = n_samples - self.max_lag
        if n_predictions == 0:
            return xp.reshape(
                _zeros(
                    xp,
                    0,
                    dtype=prediction_dtype,
                    target_device=target_device,
                ),
                (-1, 1),
            )

        prediction_method = self._model_prediction
        if not isinstance(self.basis_function, Polynomial):
            prediction_method = self._basis_function_predict

        prediction_buffer = None
        block_end = self.max_lag
        while block_end < n_samples:
            block_horizon = min(steps_ahead, n_samples - block_end)
            block_start = block_end - self.max_lag
            x_window = None
            if x is not None:
                x_window = x[block_start : block_end + block_horizon]
            block_prediction = prediction_method(
                x=x_window,
                y_initial=y[block_start:block_end],
                forecast_horizon=block_horizon,
            )
            prediction_buffer = self._store_prediction_block(
                xp=xp,
                prediction_buffer=prediction_buffer,
                block_prediction=block_prediction,
                block_horizon=block_horizon,
                output_start=block_start,
                n_predictions=n_predictions,
                target_device=target_device,
            )
            block_end += block_horizon

        return xp.reshape(prediction_buffer, (-1, 1))

    def _nar_step_ahead(self, y: np.ndarray, steps_ahead: int) -> np.ndarray:
        """Return blockwise NAR predictions without initial conditions.

        Parameters
        ----------
        y : ndarray of shape (n_samples, 1)
            Observed output values. The first ``max_lag`` samples provide the
            initial conditions for the first prediction block.
        steps_ahead : int
            Maximum number of recursive predictions in each block.

        Returns
        -------
        ndarray of shape (n_samples - max_lag, 1)
            Predicted values after the initial conditions. Each block restarts
            from the immediately preceding observed outputs.

        Raises
        ------
        ValueError
            If ``steps_ahead`` is not a positive integer or if ``y`` does not
            contain enough initial conditions.
        """
        steps_ahead = self._normalize_prediction_integer(
            steps_ahead,
            "steps_ahead",
            allow_zero=False,
        )
        n_samples = y.shape[0]
        if n_samples < self.max_lag:
            raise ValueError(
                "Insufficient initial condition elements! Expected at least"
                f" {self.max_lag} elements."
            )

        return self._blockwise_prediction(x=None, y=y, steps_ahead=steps_ahead)

    def narmax_n_step_ahead(
        self,
        x: np.ndarray,
        y: np.ndarray,
        steps_ahead: Optional[int],
    ) -> np.ndarray:
        """Return blockwise NARMAX predictions without initial conditions."""
        steps_ahead = self._normalize_prediction_integer(
            steps_ahead,
            "steps_ahead",
            allow_zero=False,
        )
        if x is None:
            raise ValueError("X cannot be None for NARMAX n-step prediction.")
        n_samples = y.shape[0]
        if n_samples < self.max_lag:
            raise ValueError(
                "Insufficient initial condition elements! Expected at least"
                f" {self.max_lag} elements."
            )

        if x.shape[0] != n_samples:
            raise ValueError(
                "X and y must contain the same number of samples for NARMAX "
                f"n-step prediction. Got {x.shape[0]} and {n_samples}."
            )

        return self._blockwise_prediction(x=x, y=y, steps_ahead=steps_ahead)

    def _n_step_ahead_prediction(
        self,
        x: Optional[np.ndarray],
        y: Optional[np.ndarray],
        steps_ahead: Optional[int],
    ) -> np.ndarray:
        """Perform the n-steps-ahead prediction of a model.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_inputs), optional
            Input values for NARMAX prediction. NAR prediction ignores this
            argument, and NFIR prediction uses it in the feed-forward kernel.
        y : ndarray of shape (n_samples, 1)
            Observed output values. Its length defines the prediction interval,
            and its first ``max_lag`` rows provide the initial conditions.
        steps_ahead : int
            Maximum number of recursive predictions in each block.

        Returns
        -------
        yhat : ndarray of floats
            Predicted values for NARMAX, NAR and NFIR models.
        """
        if self.model_type == "NARMAX":
            return self.narmax_n_step_ahead(x, y, steps_ahead)

        if self.model_type == "NAR":
            return self._nar_step_ahead(y, steps_ahead)

        if self.model_type == "NFIR":
            return self._model_prediction(x, y)

        raise ValueError(
            f"model_type must be NARMAX, NAR or NFIR. Got {self.model_type}"
        )

    def _basis_function_predict(
        self,
        x: Optional[np.ndarray],
        y_initial: np.ndarray,
        forecast_horizon: int = 1,
    ) -> np.ndarray:
        """Basis function prediction."""
        forecast_horizon = self._resolve_recursive_prediction_length(
            x,
            forecast_horizon,
        )
        xp = get_namespace(x, y_initial)
        target_device = _device(x, y_initial)
        prediction_dtype = self._prediction_dtype(
            xp,
            None if self.model_type == "NAR" else getattr(x, "dtype", None),
            y_initial.dtype,
            getattr(self.theta, "dtype", None),
        )
        yhat = _zeros(
            xp,
            forecast_horizon,
            dtype=prediction_dtype,
            target_device=target_device,
        )
        yhat = yhat * float("nan")
        yhat[: self.max_lag] = _asarray(
            y_initial[: self.max_lag, 0],
            xp=xp,
            dtype=prediction_dtype,
            target_device=target_device,
        )
        theta = _asarray(
            self.theta,
            xp=xp,
            dtype=prediction_dtype,
            target_device=target_device,
        )

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
            x_tmp = _asarray(
                x_tmp,
                xp=xp,
                dtype=prediction_dtype,
                target_device=target_device,
            )
            a = x_tmp @ theta
            yhat[i + self.max_lag] = a.item()

        return xp.reshape(yhat[self.max_lag :], (-1, 1))

    def _basis_function_n_step_prediction(
        self,
        x: Optional[np.ndarray],
        y: np.ndarray,
        steps_ahead: int,
        forecast_horizon: int,
    ) -> np.ndarray:
        """Basis function n step ahead."""
        if self.model_type == "NAR":
            return self._nar_step_ahead(y, steps_ahead)

        if self.model_type == "NARMAX":
            return self.narmax_n_step_ahead(x, y, steps_ahead)

        if self.model_type == "NFIR":
            return self._basis_function_predict(
                x=x,
                y_initial=y,
                forecast_horizon=x.shape[0],
            )

        raise ValueError(
            f"model_type must be NARMAX, NAR or NFIR. Got {self.model_type}"
        )


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
