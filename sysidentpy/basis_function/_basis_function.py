"""Basis Function for NARMAX models."""

from itertools import combinations_with_replacement
from typing import Optional
import numpy as np
from scipy.stats import binom

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
        iterable_list =  self.get_iterable_list(ylag, xlag, model_type)
        combinations = list(combinations_with_replacement(iterable_list, self.degree))
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
        ylag: int=1,
        xlag: int=1,
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


class Fourier(BaseBasisFunction):
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

    def __init__(
        self, n: int = 1, p: float = 2 * np.pi, degree: int = 1, ensemble: bool = True
    ):
        self.n = n
        self.p = p
        self.degree = degree
        self.ensemble = ensemble

    def _fourier_expansion(self, data: np.ndarray, n: int):
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
        # remove intercept (because the data always have the intercept)
        if self.degree > 1:
            data = Polynomial().fit(data, max_lag, ylag, xlag, model_type, predefined_regressors=None)
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

        if self.ensemble:
            psi = psi[:, 1:]
            psi = np.column_stack([data, psi])
        else:
            psi = psi[:, 1:]

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
        """Build Fourier Basis Functions.

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


class Bersntein(BaseBasisFunction):
    r"""Build Bersntein basis function.

    Generate Bernstein basis functions.

    This class constructs a new feature matrix consisting of Bernstein basis functions
    for a given degree. Bernstein polynomials are useful in numerical analysis, curve
    fitting, and approximation theory due to their smoothness and the ability to
    approximate any continuous function on a closed interval.

    The Bernstein polynomial of degree \(n\) for a variable \(x\) is defined as:

    .. math::
        B_{i,n}(x) = \binom{n}{i} x^i (1 - x)^{n - i} \quad \text{for} \quad i = 0, 1,
        \ldots, n

    where \(\binom{n}{i}\) is the binomial coefficient, given by:

    .. math::
        \binom{n}{i} = \frac{n!}{i! (n - i)!}

    Bernstein polynomials form a basis for the space of polynomials of degree at most
    \(n\). They are particularly useful in approximation theory because they can
    approximate any continuous function on the interval \([0, 1]\) as \(n\) increases.

    Be aware that the number of features in the output array scales significantly with
    the number of inputs, the maximum lag of the input, and the polynomial degree.

    Parameters
    ----------
    degree : int (max_degree), default=2
        The maximum degree of the polynomial features.

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
        self, degree: int = 1, n: int = 1, bias: bool = True, ensemble: bool = False
    ):
        self.degree = degree
        self.n = n
        self.bias = bias
        self.ensemble = ensemble

    def _bernstein_expansion(self, data: np.ndarray):
        k = np.arange(1 + self.n)
        base = binom.pmf(k, self.n, data[:, None])
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
        if self.degree > 1:
            data = Polynomial().fit(data, max_lag, ylag, xlag, model_type, predefined_regressors=None)
            data = data[:, 1:]
        else:
            data = data[max_lag:, 1:]

        n_features = data.shape[1]
        psi = [self._bernstein_expansion(data[:, col]) for col in range(n_features)]
        if not self.bias:
            psi = [basis[:, 1:] for basis in psi]

        psi = np.hstack(psi)
        bias_column = np.ones((psi.shape[0], 1))
        psi = np.hstack((bias_column, psi))
        psi = np.nan_to_num(psi, 0)
        if self.ensemble:
            psi = psi[:, 1:]
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
        predefined_regressors: ndarray
            Regressors to be filtered in the transformation.

        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.

        """
        return self.fit(data, max_lag, ylag, xlag, model_type, predefined_regressors)

class Bilinear(BaseBasisFunction):
    r"""Build Bilinear basis function.

    A general bilinear input-output model takes the form

    .. math::
        y(k) = a_0 + \sum_{i=1}^{n_y} a_i y(k-i) + \sum_{i=1}^{n_u} b_i u(k-i) +
        \sum_{i=1}^{n_y} \sum_{j=1}^{n_u} c_{ij} y(k-i) u(k-j)

    This is a special case of the Polynomial NARMAX model.

    Bilinear system theory has been widely studied and it plays an important role in the context of continuous-time
    systems.  This is because, roughly speaking, the set of bilinear
    systems is dense in the space of continuous-time systems and any continuous causal
    functional can be arbitrarily well approximated by bilinear systems within any
    bounded time interval (see for example Fliess and Normand-Cyrot 1982). Moreover,
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
        iterable_list =  self.get_iterable_list(ylag, xlag, model_type)
        combinations = list(combinations_with_replacement(iterable_list, self.degree))
        if self.degree == 1:
            Warning('You choose a bilinear basis function and nonlinear degree = 1. In this case, you have a linear polynomial model.')
        else:
            ny = self.get_max_ylag(ylag)
            nx = self.get_max_xlag(xlag)
            combination_ylag = list(combinations_with_replacement(list(range(1, ny + 1)), self.degree))
            combination_xlag = list(combinations_with_replacement(list(range(ny + 1, nx + ny + 1)), self.degree))
            combinations_xy = combination_xlag + combination_ylag
            combinations = list(set(combinations)-set(combinations_xy))

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
