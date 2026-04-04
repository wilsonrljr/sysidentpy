"""Bilinear Basis Function for NARMAX models."""

import warnings
from itertools import chain, combinations_with_replacement
from typing import Optional

import numpy as np

from sysidentpy._lib._array_api import (
    _column_stack,
    _is_numpy_namespace,
    _ones,
    _to_numpy,
    device as _device,
    get_namespace,
    is_array_api_obj,
)
from sysidentpy.utils.lags import get_max_xlag, get_max_ylag

from .basis_function_base import BaseBasisFunction


class Bilinear(BaseBasisFunction):
    r"""Build Bilinear basis function.

    A general bilinear input-output model takes the form

    $$
        y(k) = a_0 + \sum_{i=1}^{n_y} a_i y(k-i) + \sum_{i=1}^{n_u} b_i u(k-i) +
        \sum_{i=1}^{n_y} \sum_{j=1}^{n_u} c_{ij} y(k-i) u(k-j)
    $$

    This is a special case of the Polynomial NARMAX model.

    Bilinear system theory has been widely studied and it plays an important role in the
    context of continuous-time systems.  This is because, roughly speaking, the set of
    bilinear systems is dense in the space of continuous-time systems and any continuous
    causal functional can be arbitrarily well approximated by bilinear systems within
    any bounded time interval (see for example Fliess and Normand-Cyrot 1982). Moreover,
    many real continuous-time processes are naturally in bilinear form. A few examples
    are distillation columns (España and Landau 1978), nuclear and thermal control
    processes (Mohler 1973).

    Sampling the continuous-time bilinear system, however, produces a NARMAX model
    which is more complex than a discrete-time bilinear model.

    Parameters
    ----------
    degree : int (max_degree), default=2
        The maximum degree of the polynomial features.

    Notes
    -----
    Be aware that the number of features in the output array scales
    significantly as the number of inputs, the max lag of the input and output, and
    degree increases. High degrees can cause overfitting.
    """

    def __init__(
        self,
        degree: int = 2,
    ):
        self.degree = degree

    def _normalize_predefined_regressors(
        self,
        predefined_regressors: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Normalize regressor indices to NumPy metadata for Python indexing."""
        if predefined_regressors is None:
            return None

        if is_array_api_obj(predefined_regressors):
            predefined_regressors = _to_numpy(predefined_regressors)

        return np.asarray(predefined_regressors, dtype=np.intp).reshape(-1)

    def _get_combination_list(
        self,
        data: np.ndarray,
        ylag: int,
        xlag: int,
        predefined_regressors: Optional[np.ndarray] = None,
    ) -> list[tuple[int, ...]]:
        """Return bilinear term combinations filtered by lag configuration."""
        iterable_list = range(data.shape[1])
        combination_list = list(
            combinations_with_replacement(iterable_list, self.degree)
        )

        ny = get_max_ylag(ylag)
        combination_ylag = list(
            combinations_with_replacement(list(range(1, ny + 1)), self.degree)
        )
        if isinstance(xlag, int):
            xlag = [xlag]

        combination_xlag = []
        ni = 0
        for lag in xlag:
            nx = get_max_xlag(lag)
            combination_lag = list(
                combinations_with_replacement(
                    list(range(ny + 1 + ni, nx + ny + 1 + ni)), self.degree
                )
            )
            combination_xlag.append(combination_lag)
            ni += nx

        combination_xlag = list(chain.from_iterable(combination_xlag))
        combinations_xy = combination_xlag + combination_ylag
        combination_list = list(set(combination_list) - set(combinations_xy))

        predefined_regressors = self._normalize_predefined_regressors(
            predefined_regressors
        )
        if predefined_regressors is None:
            return combination_list

        return [combination_list[index] for index in predefined_regressors]

    def _evaluate_terms(
        self,
        data: np.ndarray,
        combination_list: list[tuple[int, ...]],
    ) -> np.ndarray:
        """Build the bilinear terms while preserving backend/device."""
        xp = get_namespace(data)
        if _is_numpy_namespace(xp):
            return _column_stack(
                xp,
                [
                    xp.prod(data[:, combination], axis=1)
                    for combination in combination_list
                ],
            )

        target_device = _device(data)
        terms = []
        for combination in combination_list:
            term = _ones(
                xp,
                (data.shape[0],),
                dtype=data.dtype,
                target_device=target_device,
            )
            for col in combination:
                term = term * data[:, int(col)]
            terms.append(term)

        return _column_stack(xp, terms)

    def fit(
        self,
        data: np.ndarray,
        max_lag: int = 1,
        ylag: int = 1,
        xlag: int = 1,
        model_type: str = "NARMAX",
        predefined_regressors: Optional[np.ndarray] = None,
    ):
        """Build the Bilinear information matrix.

        Each column of the information matrix represents a candidate
        regressor. The set of candidate regressors are based on xlag,
        ylag, and degree defined by the user.

        Parameters
        ----------
        data : ndarray of floats
            The lagged matrix built with respect to each lag and column.
        max_lag : int
            Target data used on training phase.
        ylag : ndarray of int
            The range of lags according to user definition.
        xlag : ndarray of int
            The range of lags according to user definition.
        model_type : str
            The type of the model (NARMAX, NAR or NFIR).
        predefined_regressors : ndarray of int
            The index of the selected regressors by the Model Structure
            Selection algorithm.

        Returns
        -------
        psi = ndarray of floats
            The lagged matrix built in respect with each lag and column.

        """
        if self.degree == 1:
            warnings.warn(
                "You choose a bilinear basis function and nonlinear degree = 1."
                "In this case, you have a linear polynomial model.",
                stacklevel=2,
            )

        combination_list = self._get_combination_list(
            data,
            ylag,
            xlag,
            predefined_regressors,
        )
        psi = self._evaluate_terms(data, combination_list)
        psi = psi[max_lag:, :]
        return psi

    def transform(
        self,
        data: np.ndarray,
        max_lag: int = 1,
        ylag: int = 1,
        xlag: int = 1,
        model_type: str = "NARMAX",
        predefined_regressors: Optional[np.ndarray] = None,
    ):
        """Build Polynomial Basis Functions.

        Parameters
        ----------
        data : ndarray of floats
            The lagged matrix built with respect to each lag and column.
        max_lag : int
            Maximum lag of list of regressors.
        ylag : ndarray of int
            The range of lags according to user definition.
        xlag : ndarray of int
            The range of lags according to user definition.
        model_type : str
            The type of the model (NARMAX, NAR or NFIR).
        predefined_regressors: ndarray
            Regressors to be filtered in the transformation.

        Returns
        -------
        x_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.

        """
        return self.fit(data, max_lag, ylag, xlag, model_type, predefined_regressors)
