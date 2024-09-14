"""Legendre Basis Function for NARMAX models."""

from typing import Optional

import numpy as np
from scipy.special import eval_legendre

from sysidentpy.basis_function.basis_function_base import BaseBasisFunction


class Legendre(BaseBasisFunction):
    r"""Build Legendre basis function expansion.

    This class constructs a feature matrix consisting of Legendre polynomial basis
    functions up to a specified degree. Legendre polynomials, denoted by $P_n(x)$,
    are orthogonal polynomials over the interval $[-1, 1]$ with respect to the
    uniform weight function $w(x) = 1$. They are widely used in numerical analysis,
    curve fitting, and approximation theory.

    The Legendre polynomial $P_n(x)$ of degree $n$ is defined by the following
    recurrence relation:

    $$
    P_0(x) = 1
    $$

    $$
    P_1(x) = x
    $$

    $$
    (n+1) P_{n+1}(x) = (2n + 1)x P_n(x) - n P_{n-1}(x)
    $$

    where $P_n(x)$ represents the Legendre polynomial of degree $n$.

    Parameters
    ----------
    degree : int, default=2
        The maximum degree of the Legendre polynomial basis functions to be generated.

    include_bias : bool, default=True
        Whether to include the bias (constant) term in the output feature matrix.

    ensemble : bool, default=False
        If True, the original data is concatenated with the polynomial features.

    Notes
    -----
    The number of features in the output matrix increases as the degree of the
    polynomial increases, which can lead to a high-dimensional feature space.
    Consider using dimensionality reduction techniques if overfitting becomes an issue.

    References
    ----------
    - Wikipedia: Legendre polynomial
        https://en.wikipedia.org/wiki/Legendre_polynomials

    """

    def __init__(
        self,
        degree: int = 1,
        include_bias: bool = True,
        ensemble: bool = False,
    ):
        self.degree = degree
        self.include_bias = include_bias
        self.ensemble = ensemble

    def _legendre_expansion(self, data: np.ndarray):
        num_samples = data.shape[0]
        basis = np.zeros((num_samples, self.degree + 1))
        for n in range(self.degree + 1):
            basis[:, n] = eval_legendre(n, data)
        return basis

    def fit(
        self,
        data: np.ndarray,
        max_lag: int = 1,
        ylag: int = 1,
        xlag: int = 1,
        model_type: str = "NARMAX",
        predefined_regressors: Optional[np.ndarray] = None,
    ):
        # remove intercept (because data always have the intercept)
        data = data[max_lag:, 1:]

        n_features = data.shape[1]
        psi = [self._legendre_expansion(data[:, col]) for col in range(n_features)]
        # remove P0(x) = 1 from every column expansion
        psi = [basis[:, 1:] for basis in psi]
        psi = np.hstack(psi)
        psi = np.nan_to_num(psi, 0)
        if self.include_bias:
            bias_column = np.ones((psi.shape[0], 1))
            psi = np.hstack((bias_column, psi))

        if self.ensemble:
            psi = np.column_stack([data, psi])

        if predefined_regressors is None:
            return psi

        return psi[:, predefined_regressors]

    def transform(
        self,
        data: np.ndarray,
        max_lag: int = 1,
        ylag: int = 1,
        xlag: int = 1,
        model_type: str = "NARMAX",
        predefined_regressors: Optional[np.ndarray] = None,
    ):
        """Build Bersntein Basis Functions.

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
