"""Polynomial Basis Function for NARMAX models."""

from itertools import combinations_with_replacement
from typing import Optional, Tuple, Dict

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
from .basis_function_base import BaseBasisFunction


class Polynomial(BaseBasisFunction):
    r"""Build polynomial basis function.

    Generate a new feature matrix consisting of all polynomial combinations
    of the features with degree less than or equal to the specified degree.

    $$
        y_k = \sum_{i=1}^{p}\Theta_i \times \prod_{j=0}^{n_x}u_{k-j}^{b_i, j}
        \prod_{l=1}^{n_e}e_{k-l}^{d_i, l}\prod_{m=1}^{n_y}y_{k-m}^{a_i, m}
    $$

    where $p$ is the number of regressors, $\Theta_i$ are the
    model parameters, and $a_i, m, b_i, j$ and $d_i, l \in \mathbb{N}$
    are the exponents of the output, input and noise terms, respectively.

    Parameters
    ----------
    degree : int (max_degree), default=2
        The maximum degree of the polynomial features.
    include_bias : bool, default=False
        Whether to include the bias (constant) term in the output feature matrix.
        When set to True, a column of ones is prepended to the feature matrix.

    Notes
    -----
    Be aware that the number of features in the output array scales
    significantly as the number of inputs, the max lag of the input and output, and
    degree increases. High degrees can cause overfitting.
    """

    def __init__(
        self,
        degree: int = 2,
        include_bias: bool = False,
    ):
        self.degree = degree
        self.include_bias = include_bias
        # Cache combination indices per (n_features, degree) to avoid rebuilding
        self._combination_cache: Dict[Tuple[int, int], np.ndarray] = {}

    def _get_combination_indices(self, n_features: int) -> np.ndarray:
        """Return cached column-index combinations for the current degree."""
        key = (n_features, self.degree)
        if key not in self._combination_cache:
            iterable = range(n_features)
            combos = np.array(
                list(combinations_with_replacement(iterable, self.degree)),
                dtype=np.int32,
            )
            self._combination_cache[key] = combos
        return self._combination_cache[key]

    def _normalize_predefined_regressors(
        self,
        predefined_regressors: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Normalize regressor indices to NumPy metadata for cached lookups."""
        if predefined_regressors is None:
            return None

        if is_array_api_obj(predefined_regressors):
            predefined_regressors = _to_numpy(predefined_regressors)

        return np.asarray(predefined_regressors, dtype=np.intp).reshape(-1)

    def _evaluate_terms(
        self,
        data: np.ndarray,
        predefined_regressors: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Vectorized polynomial feature construction without Python loops."""
        xp = get_namespace(data)
        n_features = data.shape[1]
        combos = self._get_combination_indices(n_features)
        predefined_regressors = self._normalize_predefined_regressors(
            predefined_regressors
        )
        if predefined_regressors is not None:
            combos = combos[predefined_regressors]

        if not _is_numpy_namespace(xp):
            target_device = _device(data)
            terms = []
            for combo in combos:
                term = _ones(
                    xp,
                    (data.shape[0],),
                    dtype=data.dtype,
                    target_device=target_device,
                )
                for col in combo:
                    term = term * data[:, int(col)]
                terms.append(term)
            return _column_stack(xp, terms)

        # Start with ones so we can multiply each degree slice in place
        n_samples = data.shape[0]
        n_terms = combos.shape[0]
        psi = _ones(xp, (n_samples, n_terms), dtype=data.dtype)

        # Multiply column-wise using the cached combination indices
        for degree_idx in range(self.degree):
            cols = combos[:, degree_idx]
            psi = psi * data[:, cols]

        return psi

    def fit(
        self,
        data: np.ndarray,
        max_lag: int = 1,
        ylag: int = 1,
        xlag: int = 1,
        model_type: str = "NARMAX",
        predefined_regressors: Optional[np.ndarray] = None,
    ):
        """Build the Polynomial information matrix.

        Each columns of the information matrix represents a candidate
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
        xp = get_namespace(data)

        # Create combinations of all columns based on its index
        # Note: predefined_regressors filtering happens inside _evaluate_terms
        # only when include_bias is False. When include_bias is True, we need
        # to handle it after adding the bias column.
        if self.include_bias:
            # Evaluate all terms first, then add bias
            psi = self._evaluate_terms(data, predefined_regressors=None)
            psi = psi[max_lag:, :]

            # Add bias column at the beginning
            target_device = _device(data) if not _is_numpy_namespace(xp) else None
            bias_column = _ones(
                xp,
                (psi.shape[0], 1),
                dtype=psi.dtype,
                target_device=target_device,
            )
            psi = _column_stack(xp, [bias_column, psi])

            # Apply predefined_regressors filtering after adding bias
            if predefined_regressors is not None:
                predefined_regressors = self._normalize_predefined_regressors(
                    predefined_regressors
                )
                psi = psi[:, predefined_regressors]
        else:
            # Original behavior: no bias term
            psi = self._evaluate_terms(data, predefined_regressors)
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
