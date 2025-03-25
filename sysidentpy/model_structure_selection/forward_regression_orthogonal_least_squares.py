"""Build Polynomial NARMAX Models using FROLS algorithm."""

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
#           Luan Pascoal da Costa Andrade <luan_pascoal13@hotmail.com>
#           Samuel Carlos Pessoa Oliveira <samuelcpoliveira@gmail.com>
#           Samir Angelo Milani Martins <martins@ufsj.edu.br>
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


class FROLS(OFRBase):
    r"""Forward Regression Orthogonal Least Squares algorithm.

    This class uses the FROLS algorithm ([1]_, [2]_) to build NARMAX models.
    The NARMAX model is described as:

    $$
        y_k= F^\ell[y_{k-1}, \dotsc, y_{k-n_y},x_{k-d}, x_{k-d-1},
        \dotsc, x_{k-d-n_x}, e_{k-1}, \dotsc, e_{k-n_e}] + e_k
    $$

    where $n_y\in \mathbb{N}^*$, $n_x \in \mathbb{N}$, $n_e \in \mathbb{N}$,
    are the maximum lags for the system output and input respectively;
    $x_k \in \mathbb{R}^{n_x}$ is the system input and $y_k \in \mathbb{R}^{n_y}$
    is the system output at discrete time $k \in \mathbb{N}^n$;
    $e_k \in \mathbb{R}^{n_e}4 stands for uncertainties and possible noise
    at discrete time $k$. In this case, $\mathcal{F}^\ell$ is some nonlinear function
    of the input and output regressors with nonlinearity degree $\ell \in \mathbb{N}$
    and $d$ is a time delay typically set to $d=1$.

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
    >>> model = FROLS(basis_function=basis_function,
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
    - Manuscript: Orthogonal least squares methods and their application
       to non-linear system identification
       https://eprints.soton.ac.uk/251147/1/778742007_content.pdf
    - Manuscript (portuguese): Identificação de Sistemas não Lineares
       Utilizando Modelos NARMAX Polinomiais - Uma Revisão
       e Novos Resultados

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

    def error_reduction_ratio(
        self, psi: np.ndarray, y: np.ndarray, process_term_number: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform the Error Reduction Ration algorithm.

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
        err : array-like of shape = number_of_model_elements
            The respective ERR calculated for each regressor.
        piv : array-like of shape = number_of_model_elements
            Contains the index to put the regressors in the correct order
            based on err values.
        psi_orthogonal : ndarray of floats
            The updated and orthogonal information matrix.

        References
        ----------
        - Manuscript: Orthogonal least squares methods and their application
           to non-linear system identification
           https://eprints.soton.ac.uk/251147/1/778742007_content.pdf
        - Manuscript (portuguese): Identificação de Sistemas não Lineares
           Utilizando Modelos NARMAX Polinomiais - Uma Revisão
           e Novos Resultados

        """
        squared_y = np.dot(y[self.max_lag :].T, y[self.max_lag :])
        tmp_psi = psi.copy()
        y = y[self.max_lag :, 0].reshape(-1, 1)
        tmp_y = y.copy()
        dimension = tmp_psi.shape[1]
        piv = np.arange(dimension)
        tmp_err = np.zeros(dimension)
        err = np.zeros(dimension)

        for i in np.arange(0, dimension):
            for j in np.arange(i, dimension):
                # Add `eps` in the denominator to omit division by zero if
                # denominator is zero
                # To implement regularized regression (ridge regression), add
                # alpha to psi.T @ psi.   See S. Chen, Local regularization assisted
                # orthogonal least squares regression, Neurocomputing 69 (2006) 559-585.
                # The version implemented below uses the same regularization for every
                # feature, # What Chen refers to Uniform regularized orthogonal least
                # squares (UROLS) Set to tiny (self.eps) when you are not regularizing.
                # alpha = eps is the default.
                tmp_err[j] = (
                    (np.dot(tmp_psi[i:, j].T, tmp_y[i:]) ** 2)
                    / (
                        (np.dot(tmp_psi[i:, j].T, tmp_psi[i:, j]) + self.alpha)
                        * squared_y
                    )
                    + self.eps
                )[0, 0]

            piv_index = np.argmax(tmp_err[i:]) + i
            err[i] = tmp_err[piv_index]
            if i == process_term_number:
                break

            if (self.err_tol is not None) and (err.cumsum()[i] >= self.err_tol):
                self.n_terms = i + 1
                process_term_number = i + 1
                break

            tmp_psi[:, [piv_index, i]] = tmp_psi[:, [i, piv_index]]
            piv[[piv_index, i]] = piv[[i, piv_index]]
            v = house(tmp_psi[i:, i])
            row_result = rowhouse(tmp_psi[i:, i:], v)
            tmp_y[i:] = rowhouse(tmp_y[i:], v)
            tmp_psi[i:, i:] = np.copy(row_result)

        tmp_piv = piv[0:process_term_number]
        psi_orthogonal = psi[:, tmp_piv]
        return err, tmp_piv, psi_orthogonal

    def run_mss_algorithm(
        self, psi: np.ndarray, y: np.ndarray, process_term_number: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.error_reduction_ratio(psi, y, process_term_number)

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
        steps_ahead : int (default = None)
            The user can use free run simulation, one-step ahead prediction
            and n-step ahead prediction.
        forecast_horizon : int, default=None
            The number of predictions over the time.

        Returns
        -------
        yhat : ndarray of floats
            The predicted values of the model.

        """
        yhat = super().predict(
            X=X, y=y, steps_ahead=steps_ahead, forecast_horizon=forecast_horizon
        )
        return yhat
