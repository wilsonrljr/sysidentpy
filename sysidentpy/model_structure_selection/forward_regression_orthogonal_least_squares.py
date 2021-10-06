""" Build Polynomial NARMAX Models using FROLS algorithm """

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
#           Luan Pascoal da Costa Andrade <luan_pascoal13@hotmail.com>
#           Samuel Carlos Pessoa Oliveira <samuelcpoliveira@gmail.com>
#           Samir Angelo Milani Martins <martins@ufsj.edu.br>
# License: BSD 3 clause


import warnings
import numpy as np
from collections import Counter
from ..base import GenerateRegressors
from ..base import HouseHolder
from ..base import InformationMatrix
from ..parameter_estimation.estimators import Estimators
from ..residues.residues_correlation import ResiduesAnalysis
from ..utils._check_arrays import check_X_y, _check_positive_int
from ..base import _get_max_lag
import warnings


class FROLS(
    Estimators, GenerateRegressors, HouseHolder, InformationMatrix, ResiduesAnalysis
):
    """Forward Regression Orthogonal Least Squares algorithm.

    Parameters
    ----------
    ylag : int, default=2
        The maximum lag of the output.
    xlag : int, default=2
        The maximum lag of the input.
    order_selection: bool, default=False
        Whether to use information criteria for order selection.
    info_criteria : str, default="aic"
        The information criteria method to be used.
    n_terms : int, default=None
        The number of the model terms to be selected.
        Note that n_terms overwrite the information criteria
        values.
    n_inputs : int, default=1
        The number of inputs of the system.
    n_info_values : int, default=10
        The number of iterations of the information
        criteria method.
    estimator : str, default="least_squares"
        The parameter estimation method.
    extended_least_squares : bool, default=False
        Whether to use extended least squares method
        for parameter estimation.
        Note that we define a specific set of noise regressors.
    aux_lag : int, default=1
        Temporary lag value used only for parameter estimation.
        This value is overwritten by the max_lag value and will
        be removed in v0.1.4.
    lam : float, default=0.98
        Forgetting factor of the Recursive Least Squares method.
    delta : float, default=0.01
        Normalization factor of the P matrix.
    offset_covariance : float, default=0.2
        The offset covariance factor of the affine least mean squares
        filter.
    mu : float, default=0.01
        The convergence coefficient (learning rate) of the filter.
    eps : float
        Normalization factor of the normalized filters.
    gama : float, default=0.2
        The leakage factor of the Leaky LMS method.
    weight : float, default=0.02
        Weight factor to control the proportions of the error norms
        and offers an extra degree of freedom within the adaptation
        of the LMS mixed norm method.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from sysidentpy.polynomial_basis import PolynomialNarmax
    >>> from sysidentpy.metrics import root_relative_squared_error
    >>> from sysidentpy.utils.generate_data import get_miso_data, get_siso_data
    >>> x_train, x_valid, y_train, y_valid = get_siso_data(n=1000,
    ...                                                    colored_noise=True,
    ...                                                    sigma=0.2,
    ...                                                    train_percentage=90)
    >>> model = PolynomialNarmax(non_degree=2,
    ...                          order_selection=True,
    ...                          n_info_values=10,
    ...                          extended_least_squares=False,
    ...                          ylag=2, xlag=2,
    ...                          info_criteria='aic',
    ...                          estimator='least_squares',
    ...                          )
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
    0        x1(k-2)     0.9000  0.95556574
    1         y(k-1)     0.1999  0.04107943
    2  x1(k-1)y(k-1)     0.1000  0.00335113

    References
    ----------
    [1] Manuscript: Orthogonal least squares methods and their application
        to non-linear system identification
        https://eprints.soton.ac.uk/251147/1/778742007_content.pdf
    [2] Manuscript (portuguese): Identificação de Sistemas não Lineares
        Utilizando Modelos NARMAX Polinomiais – Uma Revisão
        e Novos Resultados
    """

    def __init__(
        self,
        ylag=2,
        xlag=2,
        order_selection=False,
        info_criteria="aic",
        n_terms=None,
        n_inputs=1,
        n_info_values=10,
        estimator="recursive_least_squares",
        extended_least_squares=True,
        lam=0.98,
        delta=0.01,
        offset_covariance=0.2,
        mu=0.01,
        eps=np.finfo(np.float64).eps,
        gama=0.2,
        weight=0.02,
        basis_function=None,
        model_type="NARMAX"
    ):

        self.non_degree = basis_function.non_degree
        self._order_selection = order_selection
        self._n_inputs = n_inputs
        self.ylag = ylag
        self.xlag = xlag
        self.regressor_code = self.regressor_space(
            self.non_degree, xlag, ylag, n_inputs, model_type
        )
        self.max_lag = _get_max_lag(ylag, xlag)
        self.info_criteria = info_criteria
        self.n_info_values = n_info_values
        self.n_terms = n_terms
        self.estimator = estimator
        self._extended_least_squares = extended_least_squares
        self._eps = eps
        self._mu = mu
        self._offset_covariance = offset_covariance
        self._lam = lam
        self._delta = delta
        self._gama = gama
        self._weight = weight  # 0 < weight <1
        self.model_type = model_type
        self._validate_params()
        self.basis_function = basis_function

    def _validate_params(self):
        """Validate input params."""
        if not isinstance(self.n_info_values, int) or self.n_info_values < 1:
            raise ValueError(
                "n_info_values must be integer and > zero. Got %f" % self.n_info_values
            )

        if not isinstance(self._n_inputs, int) or self._n_inputs < 1:
            raise ValueError(
                "n_inputs must be integer and > zero. Got %f" % self._n_inputs
            )

        if not isinstance(self._order_selection, bool):
            raise TypeError(
                "order_selection must be False or True. Got %f" % self._order_selection
            )

        if not isinstance(self._extended_least_squares, bool):
            raise TypeError(
                "extended_least_squares must be False or True. Got %f"
                % self._extended_least_squares
            )

        if self.info_criteria not in ["aic", "bic", "fpe", "lilc"]:
            raise ValueError(
                "info_criteria must be aic, bic, fpe or lilc. Got %s"
                % self.info_criteria
            )
            
        if self.model_type not in ["NARMAX", "NAR", "NFIR"]:
            raise ValueError(
                "model_type must be NARMAX, NAR or NFIR. Got %s"
                % self.model_type
            )

        if (
            not isinstance(self.n_terms, int) or self.n_terms < 1
        ) and self.n_terms is not None:
            raise ValueError(
                "n_terms must be integer and > zero. Got %f" % self.n_terms
            )

        if self.n_terms is not None and self.n_terms > self.regressor_code.shape[0]:
            self.n_terms = self.regressor_code.shape[0]
            warnings.warn(
                (
                    "n_terms is greater than the maximum number of "
                    "all regressors space considering the chosen y_lag,"
                    "u_lag, and non_degree. We set as "
                    "%d "
                )
                % self.regressor_code.shape[0],
                stacklevel=2,
            )
            
    def error_reduction_ratio(self, psi, y, process_term_number):
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
        [1] Manuscript: Orthogonal least squares methods and their application
            to non-linear system identification
            https://eprints.soton.ac.uk/251147/1/778742007_content.pdf

        [2] Manuscript (portuguese): Identificação de Sistemas não Lineares
            Utilizando Modelos NARMAX Polinomiais – Uma Revisão
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
                tmp_err[j] = (np.dot(tmp_psi[i:, j].T, tmp_y[i:]) ** 2) / (
                    np.dot(tmp_psi[i:, j].T, tmp_psi[i:, j]) * squared_y + self._eps
                )

            if i == process_term_number:
                break

            piv_index = np.argmax(tmp_err[i:]) + i
            err[i] = tmp_err[piv_index]
            tmp_psi[:, [piv_index, i]] = tmp_psi[:, [i, piv_index]]
            piv[[piv_index, i]] = piv[[i, piv_index]]

            v = self._house(tmp_psi[i:, i])

            row_result = self._rowhouse(tmp_psi[i:, i:], v)

            tmp_y[i:] = self._rowhouse(tmp_y[i:], v)

            tmp_psi[i:, i:] = np.copy(row_result)

        tmp_piv = piv[0:process_term_number]
        psi_orthogonal = psi[:, tmp_piv]
        model_code = self.regressor_code[tmp_piv, :].copy()
        return model_code, err, piv, psi_orthogonal
    
    def information_criterion(self, X_base, y):
        """Determine the model order.

        This function uses a information criterion to determine the model size.
        'Akaike'-  Akaike's Information Criterion with
                   critical value 2 (AIC) (default).
        'Bayes' -  Bayes Information Criterion (BIC).
        'FPE'   -  Final Prediction Error (FPE).
        'LILC'  -  Khundrin’s law ofiterated logarithm criterion (LILC).

        Parameters
        ----------
        y : array-like of shape = n_samples
            Target values of the system.
        X : array-like of shape = n_samples
            Input system values measured by the user.

        Returns
        -------
        output_vector : array-like of shape = n_regressor
            Vector with values of akaike's information criterion
            for models with N terms (where N is the
            vector position + 1).

        References
        ----------

        """
        if (
            self.n_info_values is not None
            and self.n_info_values > self.regressor_code.shape[0]
        ):
            self.n_info_values = self.regressor_code.shape[0]
            warnings.warn(
                (
                    "n_info_values is greater than the maximum number "
                    "of all regressors space considering the chosen "
                    "y_lag, u_lag, and non_degree. We set as "
                    "%d "
                )
                % self.regressor_code.shape[0],
                stacklevel=2,
            )

        output_vector = np.zeros(self.n_info_values)
        output_vector[:] = np.nan

        n_samples = len(y) - self.max_lag

        for i in range(0, self.n_info_values):
            n_theta = i + 1
            regressor_matrix = self.error_reduction_ratio(X_base, y, n_theta)[3]

            tmp_theta = getattr(self, self.estimator)(regressor_matrix, y)

            tmp_yhat = np.dot(regressor_matrix, tmp_theta)
            tmp_residual = y[self.max_lag :] - tmp_yhat
            e_var = np.var(tmp_residual, ddof=1)

            output_vector[i] = self.compute_info_value(n_theta, n_samples, e_var)

        return output_vector

    def compute_info_value(self, n_theta, n_samples, e_var):
        """Compute the information criteria value.

        This function returns the information criteria concerning each
        number of regressor. The information criteria can be AIC, BIC,
        LILC and FPE.

        Parameters
        ----------
        n_theta : int
            Number of parameters of the model.
        n_samples : int
            Number of samples given the maximum lag.
        e_var : float
            Variance of the residues

        Returns
        -------
        info_criteria_value : float
            The computed value given the information criteria selected by the
            user.

        """
        if self.info_criteria == "bic":
            model_factor = n_theta * np.log(n_samples)
        elif self.info_criteria == "fpe":
            model_factor = n_samples * np.log(
                (n_samples + n_theta) / (n_samples - n_theta)
            )
        elif self.info_criteria == "lilc":
            model_factor = 2 * n_theta * np.log(np.log(n_samples))
        else:  # AIC
            model_factor = +2 * n_theta

        e_factor = n_samples * np.log(e_var)
        info_criteria_value = e_factor + model_factor

        return info_criteria_value