"""Fourier Basis Function for NARMAX models."""

from typing import Optional
import numpy as np

from sysidentpy.basis_function import Polynomial
from .basis_function_base import BaseBasisFunction


class Fourier(BaseBasisFunction):
    r"""Build Fourier basis function.

    Generate a new feature matrix consisting of all Fourier features
    with respect to the number of harmonics.

    The Fourier expansion is given by:

    If you set $\mathcal{F}$ as the Fourier extension

    $$
    \mathcal{F}(x) = [\cos(\pi x), \sin(\pi x), \cos(2\pi x), \sin(2\pi x), \ldots,
    \cos(N\pi x), \sin(N\pi x)]
    $$

    In this case, the Fourier ARX representation will be:

    \begin{aligned}
    y_k = &\Big[ \cos(\pi y_{k-1}), \sin(\pi y_{k-1}), \cos(2\pi y_{k-1}),
    \sin(2\pi y_{k-1}), \ldots, \cos(N\pi y_{k-1}), \sin(N\pi y_{k-1}), \\
    &\ \ \cos(\pi y_{k-2}), \sin(\pi y_{k-2}), \ldots, \cos(N\pi y_{k-n_y}),
    \sin(N\pi y_{k-n_y}), \\
    &\ \ \cos(\pi x_{k-1}), \sin(\pi x_{k-1}), \cos(2\pi x_{k-1}), \sin(2\pi x_{k-1}),
    \ldots, \cos(N\pi x_{k-n_x}), \sin(N\pi x_{k-n_x}) \Big] \\
    &\ \ + e_k
    \end{aligned}

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
            data = Polynomial().fit(
                data, max_lag, ylag, xlag, model_type, predefined_regressors=None
            )
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
