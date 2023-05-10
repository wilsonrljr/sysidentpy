""" Basis Function for NARMAX models """

from itertools import combinations_with_replacement
from typing import Union

import numpy as np

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

    Notes
    -----
    Be aware that the number of features in the output array scales
    significantly as the number of inputs, the max lag of the input and output, and
    degree increases. High degrees can cause overfitting.

    """

    def __init__(
        self,
        degree=2,
    ):
        self.degree = degree

    def fit(
        self,
        data: np.ndarray,
        max_lag: int = 1,
        predefined_regressors: Union[np.ndarray, None] = None,
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
        predefined_regressors : ndarray of int
            The index of the selected regressors by the Model Structure
            Selection algorithm.

        Returns
        -------
        psi = ndarray of floats
            The lagged matrix built in respect with each lag and column.

        """
        # Create combinations of all columns based on its index
        iterable_list = range(data.shape[1])
        combinations = list(combinations_with_replacement(iterable_list, self.degree))
        if predefined_regressors is not None:
            combinations = [combinations[index] for index in predefined_regressors]

        psi = np.column_stack(
            [
                np.prod(data[:, combinations[i]], axis=1)
                for i in range(len(combinations))
            ]
        )
        psi = psi[max_lag:, :]
        return psi

    def transform(
        self,
        data: np.ndarray,
        max_lag: int = 1,
        predefined_regressors: Union[np.ndarray, None] = None,
    ):
        return self.fit(data, max_lag, predefined_regressors)


class Fourier:
    """Build Fourier basis function.
    Generate a new feature matrix consisting of all Fourier features
    with respect to the number of harmonics.

    Parameters
    ----------
    degree : int (max_degree), default=2
        The maximum degree of the polynomial features.

    Notes
    -----
    Be aware that the number of features in the output array scales
    significantly as the number of inputs, the max lag of the input and output.

    """

    def __init__(self, n=1, p=2 * np.pi, degree=1, ensemble=True):
        self.n = n
        self.p = p
        self.degree = degree
        self.ensemble = ensemble
        self.repetition = None

    def _fourier_expansion(self, data, n):
        base = np.column_stack(
            [
                np.cos(2 * np.pi * data * n / self.p),
                np.sin(2 * np.pi * data * n / self.p),
            ]
        )
        return base

    def fit(
        self,
        data: np.ndarray,
        max_lag: int = 1,
        predefined_regressors: Union[np.ndarray, None] = None,
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
        predefined_regressors : ndarray of int
            The index of the selected regressors by the Model Structure
            Selection algorithm.

        Returns
        -------
        psi = ndarray of floats
            The lagged matrix built in respect with each lag and column.

        """
        # remove intercept (because the data always have the intercept)
        if self.degree > 1:
            data = Polynomial().fit(data, max_lag, predefined_regressors=None)
            data = data[:, 1:]
        else:
            data = data[max_lag:, 1:]

        columns = list(range(data.shape[1]))
        harmonics = list(range(1, self.n + 1))
        psi = np.zeros([len(data), 1])

        for col in columns:
            base_col = np.column_stack(
                [self._fourier_expansion(data[:, col], h) for h in harmonics]
            )
            psi = np.column_stack([psi, base_col])

        self.repetition = self.n * 2
        if self.ensemble:
            psi = psi[:, 1:]
            psi = np.column_stack([data, psi])
        else:
            psi = psi[:, 1:]

        if predefined_regressors is None:
            return psi, self.ensemble

        return psi[:, predefined_regressors], self.ensemble

    def transform(
        self,
        data: np.ndarray,
        max_lag: int = 1,
        predefined_regressors: Union[np.ndarray, None] = None,
    ):
        return self.fit(data, max_lag, predefined_regressors)
