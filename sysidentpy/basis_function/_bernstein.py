"""Bersntein Basis Function for NARMAX models."""

from typing import Optional
import numpy as np
from scipy.stats import binom

from .basis_function_base import BaseBasisFunction
from ..utils.deprecation import deprecated


@deprecated(
    version="v0.5.0",
    future_version="v0.6.0",
    message=(
        " `bias` and `n` are deprecated in 0.5.0 and will be removed in 0.6.0."
        " Use `include_bias` and `degree`, respectively, instead."
    ),
)
class Bernstein(BaseBasisFunction):
    r"""Build Bersntein basis function.

    Generate Bernstein basis functions.

    This class constructs a new feature matrix consisting of Bernstein basis functions
    for a given degree. Bernstein polynomials are useful in numerical analysis, curve
    fitting, and approximation theory due to their smoothness and the ability to
    approximate any continuous function on a closed interval.

    The Bernstein polynomial of degree \(n\) for a variable \(x\) is defined as:

    $$
        B_{i,n}(x) = \binom{n}{i} x^i (1 - x)^{n - i} \quad \text{for} \quad i = 0, 1,
        \ldots, n
    $$

    where \(\binom{n}{i}\) is the binomial coefficient, given by:

    $$
        \binom{n}{i} = \frac{n!}{i! (n - i)!}
    $$

    Bernstein polynomials form a basis for the space of polynomials of degree at most
    \(n\). They are particularly useful in approximation theory because they can
    approximate any continuous function on the interval \([0, 1]\) as \(n\) increases.

    Be aware that the number of features in the output array scales significantly with
    the number of inputs, the maximum lag of the input, and the polynomial degree.

    Parameters
    ----------
    degree : int (max_degree), default=1
        The maximum degree of the polynomial features.
    bias : bool, default=True
        Whether to include the bias (constant) term in the output feature matrix.
        deprecated in v.0.5.0
           `bias` is deprecated in 0.5.0 and will be removed in 0.6.0.
           Use `include_bias` instead.
    n : int, default=1
        The maximum degree of the bersntein polynomial features.
        deprecated in v.0.5.0
           `n` is deprecated in 0.5.0 and will be removed in 0.6.0.
           Use `degree` instead.

    Notes
    -----
    Be aware that the number of features in the output array scales
    significantly as the number of inputs, the max lag of the input and output.

    References
    ----------
    - Blog: this method is based on the content provided by Alex Shtoff in his blog.
        The content is easy to follow and every user is referred to is blog to check
        not only the Bersntein method, but also different topics that Alex discuss
        there!
        https://alexshtf.github.io/2024/01/21/Bernstein.html
    - Wikipedia: Bernstein polynomial
        https://en.wikipedia.org/wiki/Bernstein_polynomial

    """

    def __init__(
        self,
        degree: int = 1,
        n: Optional[int] = None,
        bias: Optional[bool] = None,
        include_bias: bool = True,
        ensemble: bool = False,
    ):
        if n is not None:
            self.degree = n
        else:
            self.degree = degree

        if bias is not None:
            self.include_bias = bias
        else:
            self.include_bias = include_bias

        self.ensemble = ensemble

    def _bernstein_expansion(self, data: np.ndarray):
        k = np.arange(1 + self.degree)
        base = binom.pmf(k, self.degree, data[:, None])
        return base

    def fit(
        self,
        data: np.ndarray,
        max_lag: int = 1,
        ylag: int = 1,
        xlag: int = 1,
        model_type: str = "NARMAX",
        predefined_regressors: Optional[np.ndarray] = None,
    ):
        # remove intercept (because the data always have the intercept)
        data = data[max_lag:, 1:]

        n_features = data.shape[1]
        psi = [self._bernstein_expansion(data[:, col]) for col in range(n_features)]
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
