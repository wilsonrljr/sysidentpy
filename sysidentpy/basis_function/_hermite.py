"""Hermite Basis Function for NARMAX models."""

from typing import Optional

import numpy as np
from scipy.special import eval_hermite

from sysidentpy.basis_function.basis_function_base import BaseBasisFunction


class Hermite(BaseBasisFunction):
    r"""Build Hermite basis function expansion.

    This class constructs a feature matrix consisting of Hermite polynomial basis
    functions up to a specified degree. Hermite polynomials, denoted by $H_n(x)$,
    are orthogonal polynomials over the interval $(-\infty, \infty)$ with respect
    to the weight function $w(x) = e^{-x^2}$. These polynomials are widely used in
    probability theory, quantum mechanics, and numerical analysis, particularly in
    solving the quantum harmonic oscillator and in the field of statistics.

    **Physicist's Hermite polynomials** $H_n(x)$, often used in physics:
    $$
    H_n(x) = (-1)^n e^{x^2} \frac{d^n}{dx^n} e^{-x^2}
    $$

    The Hermite polynomial $H_n(x)$ of degree $n$ can be also defined by the following
    recurrence relation:

    $$
    H_0(x) = 1
    $$

    $$
    H_1(x) = 2x
    $$

    $$
    H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)
    $$

    where $H_n(x)$ represents the Hermite polynomial of degree $n$.

    Parameters
    ----------
    degree : int, default=2
        The maximum degree of the Hermite polynomial basis functions to be generated.

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
    - Wikipedia: Hermite polynomial
        https://en.wikipedia.org/wiki/Hermite_polynomials
    - Scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_hermite.html

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

    def _hermite_expansion(self, data: np.ndarray):
        num_samples = data.shape[0]
        basis = np.zeros((num_samples, self.degree + 1))
        for n in range(self.degree + 1):
            basis[:, n] = eval_hermite(n, data)
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
        psi = [self._hermite_expansion(data[:, col]) for col in range(n_features)]
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
