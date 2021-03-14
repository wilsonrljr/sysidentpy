""" Build Polynomial NARMAX Models """

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
from ..utils._check_arrays import check_X_y


class PolynomialNarmax(
    GenerateRegressors, HouseHolder, InformationMatrix, ResiduesAnalysis, Estimators
):
    """Polynomial NARXMAX model

    Parameters
    ----------
    non_degree : int, default=2
        The nonlinearity degree of the polynomial function.
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
    extended_least_squres : bool, default=False
        Whether to use extended least squres method
        for parameter estimation.
        Note that we define a specific set of noise regressors.
    aux_lag : int, default=1
        Temporary lag value used only for parameter estimation.
        This value is overwriten by the max_lag value and will
        be removed in v0.1.4.
    lam : float, default=0.98
        Forgetting factor of the Recursive Least Squares method.
    delta : float, default=0.01
        Normalization factor of the P matrix.
    offset_covariance : float, default=0.2
        The offset covariance factor of the affine least mean squares
        filter.
    mu : float, defaul=0.01
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
        Utilizando Modelos NARMAX Polinomiais–Uma Revisão
        e Novos Resultados
    """

    def __init__(
        self,
        non_degree=2,
        ylag=2,
        xlag=2,
        order_selection=False,
        info_criteria="aic",
        n_terms=None,
        n_inputs=1,
        n_info_values=10,
        estimator="recursive_least_squares",
        extended_least_squares=True,
        aux_lag=1,
        lam=0.98,
        delta=0.01,
        offset_covariance=0.2,
        mu=0.01,
        eps=np.finfo(np.float64).eps,
        gama=0.2,
        weight=0.02,
    ):

        self.non_degree = non_degree
        self._order_selection = order_selection
        self._n_inputs = n_inputs
        self.ylag = ylag
        self.xlag = xlag
        [self.regressor_code, self.max_lag] = GenerateRegressors().regressor_space(
            non_degree, xlag, ylag, n_inputs
        )

        self.info_criteria = info_criteria
        self.n_info_values = n_info_values
        self.n_terms = n_terms
        self.estimator = estimator
        self._extended_least_squares = extended_least_squares
        self._eps = eps
        self._mu = mu
        self._offset_covariance = offset_covariance
        self._aux_lag = aux_lag
        self._lam = lam
        self._delta = delta
        self._gama = gama
        self._weight = weight  # 0 < weight <1
        self._validate_params()

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
            Utilizando Modelos NARMAX Polinomiais–Uma Revisão
            e Novos Resultados

        """
        squared_y = y[self.max_lag :].T @ y[self.max_lag :]
        tmp_psi = np.array(psi)
        y = np.array([y[self.max_lag :, 0]]).T
        tmp_y = np.copy(y)
        [n, dimension] = tmp_psi.shape
        piv = np.arange(dimension)
        tmp_err = np.zeros(dimension)
        err = np.zeros(dimension)

        for i in np.arange(0, dimension):
            for j in np.arange(i, dimension):
                num = np.array(tmp_psi[i:n, j].T @ tmp_y[i:n])
                num = np.power(num, 2)
                den = np.array((tmp_psi[i:n, j].T @ tmp_psi[i:n, j]) * squared_y)
                tmp_err[j] = num / den

            if i == process_term_number:
                break

            tmp_err = list(tmp_err)
            piv_index = tmp_err.index(max(tmp_err[i:]))
            err[i] = tmp_err[piv_index]
            tmp_psi[:, [piv_index, i]] = tmp_psi[:, [i, piv_index]]
            piv[[piv_index, i]] = piv[[i, piv_index]]
            x = tmp_psi[i:n, i]

            v = HouseHolder()._house(x)

            aux_1 = tmp_psi[i:n, i:dimension]

            row_result = HouseHolder()._rowhouse(aux_1, v)

            tmp_y[i:n] = HouseHolder()._rowhouse(tmp_y[i:n], v)

            tmp_psi[i:n, i:dimension] = np.copy(row_result)

        tmp_piv = piv[0:process_term_number]
        tmp_psi = np.array(psi)
        psi_orthogonal = np.copy(tmp_psi[:, tmp_piv])
        tmp_psi = np.array(psi)
        regressor_code_buffer = self.regressor_code
        model_code = np.copy(regressor_code_buffer[tmp_piv, :])
        return model_code, err, piv, psi_orthogonal

    def fit(self, X, y):
        """Fit polynomial NARMAX model.

        This is an 'alpha' version of the 'fit' function which allows
        a friendly usage by the user. Given two arguments, X and y, fit
        training data.

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

        reg_Matrix = InformationMatrix().build_information_matrix(
            X, y, self.xlag, self.ylag, self.non_degree
        )

        if self._order_selection is True:
            self.info_values = self.information_criterion(X, y)

        if self.n_terms is None and self._order_selection is True:
            model_length = np.where(self.info_values == np.amin(self.info_values))
            model_length = int(model_length[0] + 1)
            self.n_terms = model_length
        elif self.n_terms is None and self._order_selection is not True:
            raise ValueError(
                "If order_selection is False, you must define n_terms value."
            )
        else:
            model_length = self.n_terms

        (self.final_model, self.err, self.pivv, psi) = self.error_reduction_ratio(
            reg_Matrix, y, model_length
        )

        # I know... the 'method' below needs attention
        parameter_estimation = Estimators(
            aux_lag=self.max_lag,
            lam=self._lam,
            delta=self._delta,
            offset_covariance=self._offset_covariance,
            mu=self._mu,
            eps=self._eps,
            gama=self._gama,
            weight=self._weight,
        )
        self.theta = getattr(parameter_estimation, self.estimator)(psi, y)

        if self._extended_least_squares is True:
            self.theta = self._unbiased_estimator(
                psi, X, y, self.theta, self.max_lag, parameter_estimation
            )
        return self

    def _unbiased_estimator(
        self, psi, X, y, biased_theta, aux_lag, parameter_estimation
    ):
        """Estimate the model parameters using Extended Least Squares method.

        Parameters
        ----------
        psi : ndarray of floats
            The information matrix of the model.
        X : ndarray of floats
            The input data to be used in the training process.
        y_train : array-like of shape = y_training
            The data used to training the model.
        biased_theta : array-like of shape = number_of_model_elements
            The estimated biased parameters of the model.

        Returns
        -------
        theta : array-like of shape = number_of_model_elements
            The estimated unbiased parameters of the model.

        References
        ----------
        [1] Manuscript: Sorenson, H. W. (1970). Least-squares estimation:
            from Gauss to Kalman. IEEE spectrum, 7(7), 63-68.
            http://pzs.dstu.dp.ua/DataMining/mls/bibl/Gauss2Kalman.pdf
        [2] Book (Portuguese): Aguirre, L. A. (2007). Introduçaoa identificaçao
            de sistemas: técnicas lineares enao-lineares aplicadas a sistemas
            reais. Editora da UFMG. 3a ediçao.
        [3] Manuscript: Markovsky, I., & Van Huffel, S. (2007).
            Overview of total least-squares methods.
            Signal processing, 87(10), 2283-2302.
            https://eprints.soton.ac.uk/263855/1/tls_overview.pdf
        [4] Wikipedia entry on Least Squares
            https://en.wikipedia.org/wiki/Least_squares

        """
        e = y[aux_lag:, 0].reshape(-1, 1) - psi @ biased_theta
        for i in range(30):
            e = np.concatenate([np.zeros([aux_lag, 1]), e], axis=0)
            ee = np.concatenate([e, X], axis=1)
            elag = [[1, aux_lag]] * ee.shape[1]
            psi_extended = InformationMatrix().build_information_matrix(
                ee, y, elag, 1, 2
            )

            psi_extended = psi_extended[:, [2, 3, 7, 8, 11, 12, 13, 14, 15, 16, 17]]

            psi_e = np.concatenate([psi, psi_extended], axis=1)
            unbiased_theta = getattr(parameter_estimation, self.estimator)(psi_e, y)
            e = y[aux_lag:, 0].reshape(-1, 1) - psi_e @ unbiased_theta.reshape(-1, 1)

        return unbiased_theta[0 : len(self.final_model), 0].reshape(-1, 1)

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

    def _code2exponents(self, code):
        """
        Convert regressor code to exponents array.

        Parameters
        ----------
        code : 1D-array of ints
            Codification of one regressor.
        """
        regressors = np.array(list(set(code)))
        regressors_count = Counter(code)

        if np.all(regressors == 0):
            return np.zeros(self.max_lag * (1 + self._n_inputs))

        else:
            exponents = np.array([], dtype=float)
            elements = np.round(np.divide(regressors, 1000), 0)[
                (regressors > 0)
            ].astype(int)

            for j in range(1, self._n_inputs + 2):
                base_exponents = np.zeros(self.max_lag, dtype=float)
                if j in elements:
                    for i in range(1, self.max_lag + 1):
                        regressor_code = int(j * 1000 + i)
                        base_exponents[-i] = regressors_count[regressor_code]
                    exponents = np.append(exponents, base_exponents)

                else:
                    exponents = np.append(exponents, base_exponents)

            return exponents

    def _check_positive_int(self, value, name):
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be integer and > zero. Got {value}")

    def _one_step_ahead_prediction(self, X, y):
        """Perform the 1-step-ahead prediction of a model.

        Parameters
        ----------
        y : array-like of shape = max_lag
            Initial conditions values of the model
            to start recursive process.
        X : ndarray of floats of shape = n_samples
            Vector with entrace values to be used in model simulation.

        Returns
        -------
        yhat : ndarray of floats
               The 1-step-ahead predicted values of the model.

        """
        X_base = InformationMatrix().build_information_matrix(
            X, y, self.xlag, self.ylag, self.non_degree
        )
        piv_final_model = self.pivv[: len(self.final_model)]
        X_base = X_base[:, piv_final_model]
        yhat = np.dot(X_base, self.theta.flatten())
        yhat = np.concatenate([y[: self.max_lag].flatten(), yhat])
        return yhat.reshape(-1, 1)

    def _n_step_ahead_prediction(self, X, y, steps_ahead):
        """Perform the n-steps-ahead prediction of a model.

        Parameters
        ----------
        y : array-like of shape = max_lag
            Initial conditions values of the model
            to start recursive process.
        X : ndarray of floats of shape = n_samples
            Vector with entrace values to be used in model simulation.

        Returns
        -------
        yhat : ndarray of floats
               The n-steps-ahead predicted values of the model.

        """
        if len(y) < self.max_lag:
            raise Exception("Insufficient initial conditions elements!")

        yhat = np.zeros(X.shape[0], dtype=float)
        yhat.fill(np.nan)
        yhat[: self.max_lag] = y[: self.max_lag, 0]
        i = self.max_lag
        X = X.reshape(-1, self._n_inputs)
        while i < len(y):
            k = int(i - self.max_lag)
            if i + steps_ahead > len(y):
                steps_ahead = len(y) - i  # predicts the remaining values

            yhat[i : i + steps_ahead] = self._model_prediction(
                X[k : i + steps_ahead], y[k : i + steps_ahead]
            )[-steps_ahead:].ravel()

            i += steps_ahead

        yhat = yhat.ravel()
        return yhat.reshape(-1, 1)

    def _model_prediction(self, X, y_initial):
        """Perform the infinity steps-ahead simulation of a model.

        Parameters
        ----------
        y_initial : array-like of shape = max_lag
            Number of initial conditions values of output mensured
            to start recursive process.
        X : ndarray of floats of shape = n_samples
            Vector with entrace values to be used in model simulation.

        Returns
        -------
        yhat : ndarray of floats
               The predicted values of the model.

        """
        if len(y_initial) < self.max_lag:
            raise Exception("Insufficient initial conditions elements!")

        X = X.reshape(-1, self._n_inputs)
        y_output = np.zeros(X.shape[0], dtype=float)
        y_output.fill(np.nan)
        y_output[: self.max_lag] = y_initial[: self.max_lag, 0]

        model_exponents = [self._code2exponents(model) for model in self.final_model]
        raw_regressor = np.zeros(len(model_exponents[0]), dtype=float)

        for i in range(self.max_lag, X.shape[0]):
            init = 0
            final = self.max_lag
            k = int(i - self.max_lag)
            raw_regressor[:final] = y_output[k:i]
            for j in range(self._n_inputs):
                init += self.max_lag
                final += self.max_lag
                raw_regressor[init:final] = X[k:i, j]

            regressor_value = np.zeros(len(model_exponents))
            for j in range(len(model_exponents)):
                regressor_value[j] = np.prod(
                    np.power(raw_regressor, model_exponents[j])
                )

            y_output[i] = np.dot(regressor_value, self.theta.flatten())
        return y_output.reshape(-1, 1)

    def information_criterion(self, X, y):
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
        X_base = InformationMatrix().build_information_matrix(
            X, y, self.xlag, self.ylag, self.non_degree
        )

        n_samples = len(y) - self.max_lag

        parameter_estimation = Estimators(
            aux_lag=self.max_lag,
            lam=self._lam,
            delta=self._delta,
            offset_covariance=self._offset_covariance,
            mu=self._mu,
            eps=self._eps,
            gama=self._gama,
            weight=self._weight,
        )

        for i in range(0, self.n_info_values):
            n_theta = i + 1
            regressor_matrix = self.error_reduction_ratio(X_base, y, n_theta)[3]

            tmp_theta = getattr(parameter_estimation, self.estimator)(
                regressor_matrix, y
            )

            tmp_yhat = regressor_matrix @ tmp_theta
            tmp_residual = y[self.max_lag :] - tmp_yhat
            e_var = np.var(tmp_residual, ddof=1)

            output_vector[i] = self.compute_info_value(n_theta, n_samples, e_var)

            # output_vector[i] = e_factor + model_factor

        return output_vector

    def results(self, theta_precision=4, err_precision=8, dtype="dec"):
        """Write the model regressors, parameters and ERR values.

        This function returns the model regressors, its respectives parameter
        and ERR value on a string matrix.

        Parameters
        ----------
        theta_precision : int (default: 4)
            Precision of shown parameters values.
        err_precision : int (default: 8)
            Precision of shown ERR values.
        dtype : string (default: 'dec')
            Type of representation:
            sci - Scientific notation;
            dec - Decimal notation.

        Returns
        -------
        output_matrix : string
            Where:
                First column represents each regressor element;
                Second column represents associated parameter;
                Third column represents the error reduction ratio associated
                to each regressor.

        """
        if not isinstance(theta_precision, int) or theta_precision < 1:
            raise ValueError(
                "theta_precision must be integer and > zero. Got %f" % theta_precision
            )

        if not isinstance(err_precision, int) or err_precision < 1:
            raise ValueError(
                "err_precision must be integer and > zero. Got %f" % err_precision
            )

        if dtype not in ("dec", "sci"):
            raise ValueError("dtype must be dec or sci. Got %s" % dtype)

        output_matrix = []
        theta_output_format = "{:." + str(theta_precision)
        err_output_format = "{:." + str(err_precision)

        if dtype == "dec":
            theta_output_format = theta_output_format + "f}"
            err_output_format = err_output_format + "f}"
        else:
            theta_output_format = theta_output_format + "E}"
            err_output_format = err_output_format + "E}"

        for i in range(0, self.n_terms):
            if np.max(self.final_model[i]) < 1:
                tmp_regressor = str(1)
            else:
                regressor_dic = Counter(self.final_model[i])
                regressor_string = []
                for j in range(0, len(list(regressor_dic.keys()))):
                    regressor_key = list(regressor_dic.keys())[j]
                    if regressor_key < 1:
                        translated_key = ""
                        translated_exponent = ""
                    else:
                        delay_string = str(
                            int(regressor_key - np.floor(regressor_key / 1000) * 1000)
                        )
                        if int(regressor_key / 1000) < 2:
                            translated_key = "y(k-" + delay_string + ")"
                        else:
                            translated_key = (
                                "x"
                                + str(int(regressor_key / 1000) - 1)
                                + "(k-"
                                + delay_string
                                + ")"
                            )
                        if regressor_dic[regressor_key] < 2:
                            translated_exponent = ""
                        else:
                            translated_exponent = "^" + str(
                                regressor_dic[regressor_key]
                            )
                    regressor_string.append(translated_key + translated_exponent)
                tmp_regressor = "".join(regressor_string)

            current_parameter = theta_output_format.format(self.theta[i, 0])
            current_err = err_output_format.format(self.err[i])
            current_output = [tmp_regressor, current_parameter, current_err]
            output_matrix.append(current_output)

        return output_matrix

    def compute_info_value(self, n_theta, n_samples, e_var):
        """Compute the information criteria value.

        This function returns the information criteria concerning each
        number of regressor. The informotion criteria can be AIC, BIC,
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
