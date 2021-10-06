import numpy as np
from itertools import combinations_with_replacement
from ..narmax_base import InformationMatrix


class PolynomialBasis(InformationMatrix):
    def __init__(
        self,
        non_degree=2,
    ):

        self.non_degree = non_degree
        
    def build_polynomial_basis(self, data, non_degree, max_lag, predefined_regressors=None):
        """Build the information matrix.

        Each columns of the information matrix represents a candidate
        regressor. The set of candidate regressors are based on xlag,
        ylag, and non_degree entered by the user.

        Parameters
        ----------
        model : ndarray of int
            The model code representation.
        y : array-like
            Target data used on training phase.
        ylag : int
            The maximum lag of output regressors.
        non_degree : int
            The desired maximum nonlinearity degree.

        Returns
        -------
        lagged_data = ndarray of floats
            The lagged matrix built in respect with each lag and column.

        """
        # Create combinations of all columns based on its index
        iterable_list = range(data.shape[1])
        combinations = list(combinations_with_replacement(iterable_list, non_degree))
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