"""Build NARMAX Models using UOFR algorithm."""

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause

from typing import Union, Tuple, Optional

import numpy as np

from sysidentpy.narmax_base import house, rowhouse

from ..basis_function import Fourier, Polynomial
from .ofr_base import OFRBase, get_info_criteria

from ..parameter_estimation.estimators import (
    LeastSquares,
    RidgeRegression,
    RecursiveLeastSquares,
    TotalLeastSquares,
    LeastMeanSquareMixedNorm,
    LeastMeanSquares,
    LeastMeanSquaresFourth,
    LeastMeanSquaresLeaky,
    LeastMeanSquaresNormalizedLeaky,
    LeastMeanSquaresNormalizedSignRegressor,
    LeastMeanSquaresNormalizedSignSign,
    LeastMeanSquaresSignError,
    LeastMeanSquaresSignSign,
    AffineLeastMeanSquares,
    NormalizedLeastMeanSquares,
    NormalizedLeastMeanSquaresSignError,
    LeastMeanSquaresSignRegressor,
)

Estimators = Union[
    LeastSquares,
    RidgeRegression,
    RecursiveLeastSquares,
    TotalLeastSquares,
    LeastMeanSquareMixedNorm,
    LeastMeanSquares,
    LeastMeanSquaresFourth,
    LeastMeanSquaresLeaky,
    LeastMeanSquaresNormalizedLeaky,
    LeastMeanSquaresNormalizedSignRegressor,
    LeastMeanSquaresNormalizedSignSign,
    LeastMeanSquaresSignError,
    LeastMeanSquaresSignSign,
    AffineLeastMeanSquares,
    NormalizedLeastMeanSquares,
    NormalizedLeastMeanSquaresSignError,
    LeastMeanSquaresSignRegressor,
]


class UOFR(OFRBase):
    r"""Ultra Orthogonal Forward Regression algorithm.

    This class uses the UOFR algorithm ([1]) to build NARMAX models.
    The NARMAX model is described as:

    $$
        y_k= F[y_{k-1}, \dotsc, y_{k-n_y},x_{k-d}, x_{k-d-1},
        \dotsc, x_{k-d-n_x}, e_{k-1}, \dotsc, e_{k-n_e}] + e_k
    $$

    where $n_y\in \mathbb{N}^*$, $n_x \in \mathbb{N}$, $n_e \in \mathbb{N}$,
    are the maximum lags for the system output and input respectively;
    $x_k \in \mathbb{R}^{n_x}$ is the system input and $y_k \in \mathbb{R}^{n_y}$
    is the system output at discrete time $k \in \mathbb{N}^n$;
    $e_k \in \mathbb{R}^{n_e}4 stands for uncertainties and possible noise
    at discrete time $k$. In this case, $\mathcal{F}$ is some nonlinear function
    of the input and output regressors and $d$ is a time delay typically set to
     $d=1$.

    Parameters
    ----------
    ylag : int, default=2
        The maximum lag of the output.
    xlag : int, default=2
        The maximum lag of the input.
    elag : int, default=2
        The maximum lag of the residues regressors.
    order_selection: bool, default=False
        Whether to use information criteria for order selection.
    info_criteria : str, default="aic"
        The information criteria method to be used.
    n_terms : int, default=None
        The number of the model terms to be selected.
        Note that n_terms overwrite the information criteria
        values.
    n_info_values : int, default=10
        The number of iterations of the information
        criteria method.
    estimator : str, default="least_squares"
        The parameter estimation method.
    model_type: str, default="NARMAX"
        The user can choose "NARMAX", "NAR" and "NFIR" models
    eps : float, default=np.finfo(np.float64).eps
        Normalization factor of the normalized filters.
    alpha : float, default=np.finfo(np.float64).eps
        Regularization parameter used in ridge regression.
        Ridge regression parameter that regularizes the algorithm to prevent over
        fitting. If the input is a noisy signal, the ridge parameter is likely to be
        set close to the noise level, at least as a starting point.
        Entered through the self data structure.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from sysidentpy.model_structure_selection import FROLS
    >>> from sysidentpy.basis_function import Polynomial
    >>> from sysidentpy.utils.display_results import results
    >>> from sysidentpy.metrics import root_relative_squared_error
    >>> from sysidentpy.utils.generate_data import get_miso_data, get_siso_data
    >>> x_train, x_valid, y_train, y_valid = get_siso_data(n=1000,
    ...                                                    colored_noise=True,
    ...                                                    sigma=0.2,
    ...                                                    train_percentage=90)
    >>> basis_function = Polynomial(degree=2)
    >>> model = UOFR(basis_function=basis_function,
    ...               order_selection=True,
    ...               n_info_values=10,
    ...               extended_least_squares=False,
    ...               ylag=2,
    ...               xlag=2,
    ...               info_criteria='aic',
    ...               )
    >>> model.fit(x_train, y_train)
    >>> yhat = model.predict(x_valid, y_valid)
    >>> rrse = root_relative_squared_error(y_valid, yhat)
    >>> print(rrse)
    0.001993603325328823
    >>> r = pd.DataFrame(
    ...     results(
    ...         model.final_model, model.theta, model.err,
    ...         model.n_terms, err_precision=8, dtype='sci'
    ...         ),
    ...     columns=['Regressors', 'Parameters', 'ERR'])
    >>> print(r)
        Regressors Parameters         ERR
    0        x1(k-2)     0.9000       0.0
    1         y(k-1)     0.1999       0.0
    2  x1(k-1)y(k-1)     0.1000       0.0

    References
    ----------
    - Manuscript: Ultra-Orthogonal Forward Regression Algorithms for the
        Identification of Non-Linear Dynamic Systems
       https://eprints.whiterose.ac.uk/107310/1/UOFR%20Algorithms%20R1.pdf

    """

    def __init__(
        self,
        *,
        ylag: Union[int, list] = 2,
        xlag: Union[int, list] = 2,
        elag: Union[int, list] = 2,
        order_selection: bool = True,
        info_criteria: str = "aic",
        n_terms: Union[int, None] = None,
        n_info_values: int = 15,
        estimator: Estimators = RecursiveLeastSquares(),
        basis_function: Union[Polynomial, Fourier] = Polynomial(),
        model_type: str = "NARMAX",
        eps: np.float64 = np.finfo(np.float64).eps,
        alpha: float = 0,
        err_tol: Optional[float] = None,
    ):
        self.order_selection = order_selection
        self.ylag = ylag
        self.xlag = xlag
        self.max_lag = self._get_max_lag()
        self.info_criteria = info_criteria
        self.info_criteria_function = get_info_criteria(info_criteria)
        self.n_info_values = n_info_values
        self.n_terms = n_terms
        self.estimator = estimator
        self.elag = elag
        self.model_type = model_type
        self.basis_function = basis_function
        self.eps = eps
        if isinstance(self.estimator, RidgeRegression):
            self.alpha = self.estimator.alpha
        else:
            self.alpha = alpha

        self.err_tol = err_tol
        self._validate_params()
        self.n_inputs = None
        self.regressor_code = None
        self.info_values = None
        self.err = None
        self.final_model = None
        self.theta = None
        self.pivv = None

    def gaussian_test_function(self, t: np.ndarray, order: int) -> np.ndarray:
        """Generate Gaussian-like test function derivatives."""
        sigma = 1.0  # Adjust based on signal characteristics
        gaussian = np.exp(-(t**2) / (2 * sigma**2))
        derivative = np.gradient(gaussian, t)
        for _ in range(order - 1):
            derivative = np.gradient(derivative, t)
        return derivative

    def normalize_test_function(self, phi_j: np.ndarray) -> np.ndarray:
        """Normalize derivatives."""
        norm = np.linalg.norm(phi_j, ord=2)
        return phi_j / norm if norm != 0 else phi_j

    def compute_modulated_signal(
        self, signal: np.ndarray, phi_bar_j: np.ndarray
    ) -> np.ndarray:
        modulated = np.convolve(signal.flatten(), phi_bar_j, mode="valid")
        return modulated  # Length = len(signal) - len(phi_bar_j) + 1

    def augment_uls_terms(
        self, y: np.ndarray, psi: np.ndarray, m: int = 2, test_support: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Augment signals for ULS with matching row counts."""
        modulated_length = len(y) - test_support + 1
        num_terms = psi.shape[1]
        t = np.linspace(-3, 3, test_support)

        # Initialize y_augmented and psi_augmented with original truncated signals
        y_augmented = y[:modulated_length].reshape(-1, 1)
        psi_augmented = psi[:modulated_length, :]

        for j in range(1, m + 1):
            phi_j = self.gaussian_test_function(t, order=j)
            phi_bar_j = self.normalize_test_function(phi_j)
            y_j = self.compute_modulated_signal(y, phi_bar_j).reshape(-1, 1)
            y_augmented = np.vstack([y_augmented, y_j])
            modulated_terms = np.zeros((modulated_length, num_terms))
            for term in range(num_terms):
                x_j = self.compute_modulated_signal(psi[:, term], phi_bar_j)
                modulated_terms[:, term] = x_j

            psi_augmented = np.vstack([psi_augmented, modulated_terms])

        return y_augmented, psi_augmented

    def sobolev_error_reduction_ratio(
        self,
        psi: np.ndarray,
        y: np.ndarray,
        process_term_number: int,
        m: int = 2,
        test_support: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Define Ultra Orthogonal Least Squares."""
        y = y[self.max_lag :, 0].reshape(-1, 1)
        y_augmented, psi_augmented = self.augment_uls_terms(y, psi, m, test_support)
        y_augmented = y_augmented.reshape(-1, 1)
        # Compute ERR on the augmented ULS matrix
        squared_y = np.dot(y_augmented.T, y_augmented)
        psi_working = psi_augmented.copy()
        y_working = y_augmented.copy()
        num_terms = psi_working.shape[1]
        piv = np.arange(num_terms)
        candidate_err = np.zeros(num_terms)
        err = np.zeros(num_terms)

        for step_idx in np.arange(0, num_terms):
            for term_idx in np.arange(step_idx, num_terms):
                candidate_err[term_idx] = (
                    (
                        np.dot(psi_working[step_idx:, term_idx].T, y_working[step_idx:])
                        ** 2
                    )
                    / (
                        (
                            np.dot(
                                psi_working[step_idx:, term_idx].T,
                                psi_working[step_idx:, term_idx],
                            )
                            + self.alpha
                        )
                        * squared_y
                    )
                    + self.eps
                )[0, 0]

            max_err_idx = np.argmax(candidate_err[step_idx:]) + step_idx
            err[step_idx] = candidate_err[max_err_idx]
            if step_idx == process_term_number:
                break

            if (self.err_tol is not None) and (err.cumsum()[step_idx] >= self.err_tol):
                self.n_terms = step_idx + 1
                process_term_number = step_idx + 1
                break

            psi_working[:, [max_err_idx, step_idx]] = psi_working[
                :, [step_idx, max_err_idx]
            ]
            piv[[max_err_idx, step_idx]] = piv[[step_idx, max_err_idx]]
            reflector = house(psi_working[step_idx:, step_idx])
            row_result = rowhouse(psi_working[step_idx:, step_idx:], reflector)
            y_working[step_idx:] = rowhouse(y_working[step_idx:], reflector)
            psi_working[step_idx:, step_idx:] = np.copy(row_result)

        tmp_piv = piv[0:process_term_number]
        psi_orthogonal = psi[:, tmp_piv]
        return err, tmp_piv, psi_orthogonal

    def run_mss_algorithm(
        self, psi: np.ndarray, y: np.ndarray, process_term_number: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.sobolev_error_reduction_ratio(psi, y, process_term_number)

    def fit(self, *, X: Optional[np.ndarray] = None, y: np.ndarray):
        """Fit polynomial NARMAX model.

        This is an 'alpha' version of the 'fit' function which allows
        a friendly usage by the user. Given two arguments, x and y, fit
        training data.

        Parameters
        ----------
        X : ndarray of floats
            The input data to be used in the training process.
        y : ndarray of floats
            The output data to be used in the training process.

        Returns
        -------
        model : ndarray of int
            The model code representation.
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
        super().fit(X=X, y=y)
        return self

    def predict(
        self,
        *,
        X: Optional[np.ndarray] = None,
        y: np.ndarray,
        steps_ahead: Optional[int] = None,
        forecast_horizon: Optional[int] = None,
    ) -> np.ndarray:
        yhat = super().predict(
            X=X, y=y, steps_ahead=steps_ahead, forecast_horizon=forecast_horizon
        )
        return yhat
