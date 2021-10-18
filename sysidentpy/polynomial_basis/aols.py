""" Build Polynomial NARMAX Models using the Accelerated Orthogonal Least-Squares algorithm """

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause


import warnings
import numpy as np
from numpy import linalg as LA
from collections import Counter
from ..base import GenerateRegressors
from ..base import _get_max_lag
from ..base import ModelInformation
from ..parameter_estimation.estimators import Estimators
from ..residues.residues_correlation import ResiduesAnalysis
from ..utils._check_arrays import check_X_y
from .narmax import PolynomialNarmax
from ..utils.deprecation import deprecated


@deprecated(
    version="v0.1.7",
    future_version="v0.2.0",
    alternative="from sysidentpy.model_structure_selection import AOLS. \n Check the documentation for more details.",
)
class AOLS(PolynomialNarmax):
    """Polynomial NARMAX model

    Build Polynomial NARMAX model using the Accelerated Orthogonal Least-Squares.
    This algorithm is based on the Matlab code available on:
    https://github.com/realabolfazl/AOLS/

    Parameters
    ----------
    non_degree : int, default=2
        The nonlinearity degree of the polynomial function.
    ylag : int, default=2
        The maximum lag of the output.
    xlag : int, default=2
        The maximum lag of the input.
    n_inputs : int, default=1
        The number of inputs of the system.
    k : int, default=1
        The sparsity level.
    l : int, default=1
        Number of selected indices per iteration.
    threshold : float, default=10e10
        The desired accuracy.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from sysidentpy.polynomial_basis import AOLS
    >>> from sysidentpy.metrics import root_relative_squared_error
    >>> from sysidentpy.utils.generate_data import get_miso_data, get_siso_data
    >>> x_train, x_valid, y_train, y_valid = get_siso_data(n=1000,
    ...                                                    colored_noise=True,
    ...                                                    sigma=0.2,
    ...                                                    train_percentage=90)
    >>> model = AOLS(non_degree=2,
    ...              order_selection=True,
    ...              n_info_values=10,
    ...              extended_least_squares=False,
    ...              ylag=2, xlag=2,
    ...              info_criteria='aic',
    ...              estimator='least_squares',
    ...              )
    >>> model.fit(x_train, y_train)
    >>> yhat = model.predict(x_valid, y_valid)
    >>> rrse = root_relative_squared_error(y_valid, yhat)
    >>> print(rrse)
    0.001993603325328823
    >>> results = pd.DataFrame(model.results(err_precision=8,
    ...                                      dtype='dec'),
    ...                        columns=['Regressors', 'Parameters', 'ERR'])
    >>> print(results)
        Regressors Parameters         ERR
    0        x1(k-2)     0.9000       0.0
    1         y(k-1)     0.1999       0.0
    2  x1(k-1)y(k-1)     0.1000       0.0

    References
    ----------
    [1] [1] Manuscript: Accelerated Orthogonal Least-Squares for Large-Scale
        Sparse Reconstruction
            https://www.sciencedirect.com/science/article/abs/pii/S1051200418305311
    [2] Code:
        https://github.com/realabolfazl/AOLS/
    """

    def __init__(
        self, non_degree=2, ylag=2, xlag=2, n_inputs=1, k=1, l=1, threshold=10e-10
    ):

        # self.non_degree = non_degree
        # self.ylag = ylag
        # self.xlag = xlag
        [self.regressor_code, self.max_lag] = self.regressor_space(
            non_degree, xlag, ylag, n_inputs
        )

        self.k = k
        self.l = l
        self.threshold = threshold
        self._validate_params()
        super().__init__(  # n_inputs=n_inputs,
            non_degree=non_degree,
            ylag=ylag,
            xlag=xlag,
            n_inputs=n_inputs,
        )
        self._n_inputs = n_inputs

    def _validate_params(self):
        """Validate input params."""
        if not isinstance(self.k, int) or self.k < 1:
            raise ValueError("k must be integer and > zero. Got %f" % self.k)

        if not isinstance(self.l, int) or self.l < 1:
            raise ValueError("k must be integer and > zero. Got %f" % self.l)

        if not isinstance(self.threshold, (int, float)) or self.threshold < 0:
            raise ValueError(
                "threshold must be integer and > zero. Got %f" % self.threshold
            )

    def aols(self, psi, y):
        """Perform the Accelerated Orthogonal Least-Squares algorithm.

        Parameters
        ----------
        y : array-like of shape = n_samples
            The target data used in the identification process.
        psi : ndarray of floats
            The information matrix of the model.
        process_term_number : int
            Number of Process Terms defined by the user.

        Returns
        -------
        theta : array-like of shape = number_of_model_elements
            The respective ERR calculated for each regressor.
        piv : array-like of shape = number_of_model_elements
            Contains the index to put the regressors in the correct order
            based on err values.
        residual_norm : float
            The final residual norm.

        References
        ----------
        [1] Manuscript: Accelerated Orthogonal Least-Squares for Large-Scale
        Sparse Reconstruction
            https://www.sciencedirect.com/science/article/abs/pii/S1051200418305311
        """
        n, m = psi.shape
        theta = np.zeros([m, 1])
        r = y.copy()
        it = 0
        max_iter = int(min(self.k, np.floor(n / self.l)))
        AOLS_index = np.zeros(max_iter * self.l)
        U = np.zeros([n, max_iter * self.l])
        T = psi.copy()
        while LA.norm(r) > self.threshold and it < max_iter:
            it = it + 1
            temp_in = (it - 1) * self.l
            if it > 1:
                T = T - U[:, temp_in].reshape(-1, 1) @ (
                    U[:, temp_in].reshape(-1, 1).T @ psi
                )

            q = ((r.T @ psi) / np.sum(psi * T, axis=0)).ravel()
            TT = np.sum(T ** 2, axis=0) * (q ** 2)
            sub_ind = list(AOLS_index[:temp_in].astype(int))
            TT[sub_ind] = 0
            sorting_indices = np.argsort(TT)[::-1].ravel()
            AOLS_index[temp_in : temp_in + self.l] = sorting_indices[: self.l]
            for i in range(self.l):
                TEMP = T[:, sorting_indices[i]].reshape(-1, 1) * q[sorting_indices[i]]
                U[:, temp_in + i] = (TEMP / np.linalg.norm(TEMP, axis=0)).ravel()
                r = r - TEMP
                if i == self.l:
                    break

                T = T - U[:, temp_in + i].reshape(-1, 1) @ (
                    U[:, temp_in + i].reshape(-1, 1).T @ psi
                )
                q = ((r.T @ psi) / np.sum(psi * T, axis=0)).ravel()

        AOLS_index = AOLS_index[AOLS_index > 0].ravel().astype(int)
        residual_norm = LA.norm(r)
        theta[AOLS_index] = LA.lstsq(psi[:, AOLS_index], y, rcond=None)[0]
        if self.l > 1:
            sorting_indices = np.argsort(np.abs(theta))[::-1]
            AOLS_index = sorting_indices[: self.k].ravel().astype(int)
            theta[AOLS_index] = LA.lstsq(psi[:, AOLS_index], y, rcond=None)[0]
            residual_norm = LA.norm(y - psi[:, AOLS_index] @ theta[AOLS_index])

        pivv = np.argwhere(theta.ravel() > 0).ravel()
        theta = theta[theta > 0]
        return theta.reshape(-1, 1), pivv, residual_norm

    def fit(self, X, y):
        """Fit polynomial NARMAX model using AOLS algorithm.

        The 'fit' function allows a friendly usage by the user.
        Given two arguments, X and y, fit training data.

        Parameters
        ----------
        X : ndarray of floats
            The input data to be used in the training process.
        y : ndarray of floats
            The output data to be used in the training process.

        Returns
        -------
        model : ndarray of ints
            The model code represetation.
        piv : array-like of shape = number_of_model_elements
            Contains the index to put the regressors in the correct order
            based on err values.
        theta : array-like of shape = number_of_model_elements
            The estimated parameters of the model.
        err : array-like of shape = number_of_model_elements
            The respective ERR calculated for each regressor.
        info_values : array-like of shape = n_regressor
            Vector with values of akaike's information criterion
            for models with N terms (where N is the
            vector position + 1).

        """
        if y is None:
            raise ValueError("y cannot be None")

        check_X_y(X, y)

        reg_Matrix = self.build_information_matrix(
            X, y, self.xlag, self.ylag, self.non_degree
        )
        # self.max_lag = _get_max_lag(self.ylag, self.xlag)
        y = y[self.max_lag :].reshape(-1, 1)

        (self.theta, self.pivv, self.res) = self.aols(reg_Matrix, y)
        self.final_model = self.regressor_code[self.pivv, :].copy()
        self.max_lag = ModelInformation()._get_max_lag_from_model_code(self.final_model)
        self.n_terms = len(
            self.theta
        )  # the number of terms we selected (necessary in the 'results' methods)
        self.err = self.n_terms * [
            0
        ]  # just to use the `results` method. Will be changed in next update.

        return self

    def predict(self, X, y, steps_ahead=None):
        """Return the predicted values given an input.

        The predict function allows a friendly usage by the user.
        Given a previously trained model, predict values given
        a new set of data.

        This method accept y values mainly for prediction n-steps ahead
        (to be implemented in the future)

        Parameters
        ----------
        X : ndarray of floats
            The input data to be used in the prediction process.
        y : ndarray of floats
            The output data to be used in the prediction process.
        steps_ahead = int (default = None)
            The forecast horizon.

        Returns
        -------
        yhat : ndarray of floats
            The predicted values of the model.

        """
        if steps_ahead is None:
            return self._model_prediction(X, y)
        elif steps_ahead == 1:
            return self._one_step_ahead_prediction(X, y)
        else:
            self._check_positive_int(steps_ahead, "steps_ahead")
            return self._n_step_ahead_prediction(X, y, steps_ahead=steps_ahead)
