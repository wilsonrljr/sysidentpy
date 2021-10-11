""" Simulation methods for NARMAX models """

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause

import numpy as np
from ..utils._check_arrays import check_X_y
from ..narmax_base import GenerateRegressors, ModelPrediction
from ..narmax_base import HouseHolder
from ..narmax_base import InformationMatrix
from ..narmax_base import ModelInformation
from ..narmax_base import ModelPrediction
from ..parameter_estimation.estimators import Estimators


class SimulateNARMAX(Estimators, GenerateRegressors, HouseHolder,
    ModelInformation, InformationMatrix, ModelPrediction):
    """Simulation of Polynomial NARMAX model

    Parameters
    ----------
    n_inputs : int, default=1
        The number of inputs of the system.
    estimator : str, default="least_squares"
        The parameter estimation method.
    extended_least_squares : bool, default=False
        Whether to use extended least squares method
        for parameter estimation.
        Note that we define a specific set of noise regressors.
    estimate_parameter : bool, default=False
        Whether to use a method for parameter estimation.
        Must be True if the user do not enter the pre-estimated parameters.
        Note that we define a specific set of noise regressors.
    calculate_err : bool, default=False
        Whether to use a ERR algorithm to the pre-defined regressors.
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
    >>> from sysidentpy.polynomial_basis.simulation import SimulatePolynomialNarmax
    >>> from sysidentpy.metrics import root_relative_squared_error
    >>> from sysidentpy.utils.generate_data import get_miso_data, get_siso_data
    >>> x_train, x_valid, y_train, y_valid = get_siso_data(n=1000,
    ...                                                    colored_noise=True,
    ...                                                    sigma=0.2,
    ...                                                    train_percentage=90)
    >>> s = SimulatePolynomialNarmax()
    >>> model = np.array(
    ...     [
    ...     [1001,    0], # y(k-1)
    ...     [2001, 1001], # x1(k-1)y(k-1)
    ...     [2002,    0], # x1(k-2)
    ...     ]
    ...                 )
    >>> # theta must be a numpy array of shape (n, 1) where n is the number of regressors
    >>> theta = np.array([[0.2, 0.9, 0.1]]).T
    >>> yhat, results = s.simulate(
    ...     X_test=x_test,
    ...     y_test=y_test,
    ...     model_code=model,
    ...     theta=theta,
    ...     plot=True)
    >>> results = pd.DataFrame(model.results(err_precision=8,
    ...                                      dtype='dec'),
    ...                        columns=['Regressors', 'Parameters', 'ERR'])
    >>> print(results)
        Regressors Parameters         ERR
    0        x1(k-2)     0.9000  0.95556574
    1         y(k-1)     0.1999  0.04107943
    2  x1(k-1)y(k-1)     0.1000  0.00335113

    """
    def __init__(
        self,
        n_inputs=1,
        estimator="recursive_least_squares",
        extended_least_squares=False,
        lam=0.98,
        delta=0.01,
        offset_covariance=0.2,
        mu=0.01,
        eps=np.finfo(np.float64).eps,
        gama=0.2,
        weight=0.02,
        estimate_parameter=True,
        calculate_err=False,
        model_type="NARMAX",
        basis_function=None,
    ):

        super().__init__(
            lam=lam,
            delta=delta,
            offset_covariance=offset_covariance,
            mu=mu,
            eps=eps,
            gama=gama,
            weight=weight,
        )
        self.model_type = model_type
        self.basis_function=basis_function
        self.estimator = estimator
        self._extended_least_squares = extended_least_squares
        self.estimate_parameter = estimate_parameter
        self.calculate_err = calculate_err
        self._n_inputs = n_inputs

    def _validate_simulate_params(self):
        if not isinstance(self.estimate_parameter, bool):
            raise TypeError(
                f"estimate_parameter must be False or True. Got {self.estimate_parameter}"
            )
        
        if not isinstance(self.calculate_err, bool):
            raise TypeError(
                f"calculate_err must be False or True. Got {self.calculate_err}"
            )

    def simulate(
        self,
        X_train=None,
        y_train=None,
        X_test=None,
        y_test=None,
        model_code=None,
        steps_ahead=None,
        theta=None,
        plot=True,
    ):
        """Simulate a model defined by the user.

        Parameters
        ----------
        X_train : ndarray of floats
            The input data to be used in the training process.
        y_train : ndarray of floats
            The output data to be used in the training process.
        X_test : ndarray of floats
            The input data to be used in the prediction process.
        y_test : ndarray of floats
            The output data (initial conditions) to be used in the prediction process.
        model_code : ndarray of int
            Flattened list of input or output regressors.
        steps_ahead = int, default = None
            The forecast horizon.
        theta : array-like of shape = number_of_model_elements
            The parameters of the model.
        plot : bool, default=True
            Indicate if the user wants to plot or not.
        Returns
        -------
        yhat : ndarray of floats
            The predicted values of the model.
        results : string
            Where:
                First column represents each regressor element;
                Second column represents associated parameter;
                Third column represents the error reduction ratio associated
                to each regressor.

        """
        if y_test is None:
            raise ValueError("y_test cannot be None")

        if not isinstance(model_code, np.ndarray):
            raise TypeError(f"model_code must be an np.np.ndarray. Got {model_code}")

        if not isinstance(steps_ahead, (int, type(None))):
            raise ValueError(
                f"steps_ahead must be None or integer > zero. Got {steps_ahead}"
            )
        if not isinstance(plot, bool):
            raise TypeError(f"plot must be True or False. Got {plot}")
        if not isinstance(theta, np.ndarray) and not self.estimate_parameter:
            raise TypeError(
                f"If estimate_parameter is False, theta must be an np.np.ndarray. Got {theta}"
            )

        check_X_y(X_test, y_test)
        if self.estimate_parameter:
            if not all(isinstance(i, np.ndarray) for i in [X_train, y_train]):
                raise TypeError(
                    f"If estimate_parameter is True, X_train and y_train must be an np.ndarray. Got {type(X_train), type(y_train)}"
                )
            check_X_y(X_train, y_train)
            if y_train is None:
                raise ValueError("y_train cannot be None")

        xlag_code = self._list_input_regressor_code(model_code)
        ylag_code = self._list_output_regressor_code(model_code)
        self.xlag = self._get_lag_from_regressor_code(xlag_code)
        self.ylag = self._get_lag_from_regressor_code(ylag_code)
        self.max_lag = max(self.xlag, self.ylag)
        if self._n_inputs != 1:
            self.xlag = self._n_inputs * [list(range(1, self.max_lag + 1))]

        self.non_degree = model_code.shape[1]
        regressor_code = self.regressor_space(
            self.non_degree, self.xlag, self.ylag, self._n_inputs, self.model_type
        )

        self.pivv = self._get_index_from_regressor_code(regressor_code, model_code)
        self.final_model = regressor_code[self.pivv]
        # self.regressor_code = self.final_model
        # to use in the predict function
        self.n_terms = self.final_model.shape[0]
        
        if not self.estimate_parameter:
            self.theta = theta
            self.err = self.n_terms * [0]
        elif self.estimate_parameter and not self.calculate_err:
            if self.model_type == "NAR":
                lagged_data = self.build_output_matrix(y_train, self.ylag, self.non_degree)
                self.max_lag = self._get_max_lag(ylag=self.ylag)
            elif self.model_type == "NFIR":
                lagged_data = self.build_input_matrix(X_train, self.xlag, self.non_degree)
                self.max_lag = self._get_max_lag(xlag=self.xlag)
            elif self.model_type == "NARMAX":
                check_X_y(X_train, y_train)
                self.max_lag = self._get_max_lag(ylag=self.ylag, xlag=self.xlag)
                lagged_data = self.build_input_output_matrix(X_train, y_train, self.xlag, self.ylag, self.non_degree)
            else:
                raise ValueError("Unrecognized model type. The model_type should be NARMAX, NAR or NFIR.")
            psi = self.basis_function.build_polynomial_basis(
                lagged_data, self.non_degree, self.max_lag, predefined_regressors=self.pivv)

            self.theta = getattr(self, self.estimator)(psi, y_train)
            if self._extended_least_squares is True:
                self.theta = self._unbiased_estimator(psi, y_train, self.theta, self.non_degree, self.elag, self.max_lag)
            
            self.err = self.n_terms * [0]
        else:
            if self.model_type == "NAR":
                lagged_data = self.build_output_matrix(y_train, self.ylag, self.non_degree)
                self.max_lag = self._get_max_lag(ylag=self.ylag)
            elif self.model_type == "NFIR":
                lagged_data = self.build_input_matrix(X_train, self.xlag, self.non_degree)
                self.max_lag = self._get_max_lag(xlag=self.xlag)
            elif self.model_type == "NARMAX":
                check_X_y(X_train, y_train)
                self.max_lag = self._get_max_lag(ylag=self.ylag, xlag=self.xlag)
                lagged_data = self.build_input_output_matrix(X_train, y_train, self.xlag, self.ylag, self.non_degree)
            else:
                raise ValueError("Unrecognized model type. The model_type should be NARMAX, NAR or NFIR.")
            psi = self.basis_function.build_polynomial_basis(
                lagged_data, self.non_degree, self.max_lag, predefined_regressors=self.pivv)

            _, self.err, self.pivv, _ = self.error_reduction_ratio(
                psi, y_train, self.n_terms, self.final_model
            )
            self.theta = getattr(self, self.estimator)(psi, y_train)
            if self._extended_least_squares is True:
                self.theta = self._unbiased_estimator(psi, y_train, self.theta, self.non_degree, self.elag, self.max_lag)

        yhat = self.predict(X_test, y_test, steps_ahead)
        # results = self.results(err_precision=8, dtype="dec")

        # if plot:
        #     ee, ex, _, _ = self.residuals(X_test, y_test, yhat)
        #     self.plot_result(y_test, yhat, ee, ex)
        return yhat

    def error_reduction_ratio(self, psi, y, process_term_number, regressor_code):
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
        model_code = regressor_code[tmp_piv, :].copy()
        return model_code, err, piv, psi_orthogonal