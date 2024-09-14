"""Polynomial Basis Function for NARMAX models."""

from itertools import combinations_with_replacement
from typing import Optional
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
        degree: int = 2,
    ):
        self.degree = degree

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
        # Create combinations of all columns based on its index
        iterable_list = self.get_iterable_list(ylag, xlag, model_type)
        combination_list = list(
            combinations_with_replacement(iterable_list, self.degree)
        )
        if predefined_regressors is not None:
            combination_list = [
                combination_list[index] for index in predefined_regressors
            ]

        psi = np.column_stack(
            [
                np.prod(data[:, combination_list[i]], axis=1)
                for i in range(len(combination_list))
            ]
        )
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
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.

        """
        return self.fit(data, max_lag, ylag, xlag, model_type, predefined_regressors)
