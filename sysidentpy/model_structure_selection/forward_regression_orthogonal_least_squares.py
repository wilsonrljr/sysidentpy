""" Build Polynomial NARMAX Models using FROLS algorithm """

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
#           Luan Pascoal da Costa Andrade <luan_pascoal13@hotmail.com>
#           Samuel Carlos Pessoa Oliveira <samuelcpoliveira@gmail.com>
#           Samir Angelo Milani Martins <martins@ufsj.edu.br>
# License: BSD 3 clause

import warnings
from typing import Union

import numpy as np

from sysidentpy.utils._check_arrays import _check_positive_int, _num_features

from ..basis_function import Fourier, Polynomial
from ..narmax_base import BaseMSS, Orthogonalization
from ..parameter_estimation.estimators import Estimators
from ..utils.deprecation import deprecated


@deprecated(
    version="v0.3.0",
    future_version="v0.4.0",
    message=(
        "Passing a string to define the estimator will rise an error in v0.4.0."
        " \n You'll have to use FROLS(estimator=LeastSquares()) instead. \n The"
        " only change is that you'll have to define the estimator first instead"
        " of passing a string like 'least_squares'. \n This change will make"
        " easier to implement new estimators and it'll improve code"
        " readability."
    ),
)
class FROLS(Estimators, BaseMSS):
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
        The maximum lag of the residues.
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
    extended_least_squares : bool, default=False
        Whether to use extended least squares method
        for parameter estimation.
        Note that we define a specific set of noise regressors.
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
    model_type: str, default="NARMAX"
        The user can choose "NARMAX", "NAR" and "NFIR" models

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from sysidentpy.model_structure_selection import FROLS
    >>> from sysidentpy.basis_function._basis_function import Polynomial
    >>> from sysidentpy.utils.display_results import results
    >>> from sysidentpy.metrics import root_relative_squared_error
    >>> from sysidentpy.utils.generate_data import get_miso_data, get_siso_data
    >>> x_train, x_valid, y_train, y_valid = get_siso_data(n=1000,
    ...                                                    colored_noise=True,
    ...                                                    sigma=0.2,
    ...                                                    train_percentage=90)
    >>> basis_function = Polynomial(degree=2)
    >>> model = PolynomialNarmax(basis_function=basis_function,
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
       Utilizando Modelos NARMAX Polinomiais – Uma Revisão
       e Novos Resultados

    """

    def __init__(
        self,
        *,
        ylag: Union[int, list] = 2,
        xlag: Union[int, list] = 2,
        elag: Union[int, list] = 2,
        order_selection: bool = False,
        info_criteria: str = "aic",
        n_terms: Union[int, None] = None,
        n_info_values: int = 10,
        estimator: str = "recursive_least_squares",
        extended_least_squares: bool = False,
        lam: float = 0.98,
        delta: float = 0.01,
        offset_covariance: float = 0.2,
        mu: float = 0.01,
        eps: np.float64 = np.finfo(np.float64).eps,
        gama: float = 0.2,
        weight: float = 0.02,
        basis_function: Union[Polynomial, Fourier] = Polynomial(),
        model_type: str = "NARMAX",
    ):
        self.non_degree = basis_function.degree
        self.order_selection = order_selection
        self.ylag = ylag
        self.xlag = xlag
        self.max_lag = self._get_max_lag()
        self.info_criteria = info_criteria
        self.info_criteria_function = self.get_info_criteria(info_criteria)
        self.n_info_values = n_info_values
        self.n_terms = n_terms
        self.estimator = estimator
        self.extended_least_squares = extended_least_squares
        self.elag = elag
        self.model_type = model_type
        self.build_matrix = self.get_build_io_method(model_type)
        self._validate_params()
        self.basis_function = basis_function
        super().__init__(
            lam=lam,
            delta=delta,
            offset_covariance=offset_covariance,
            mu=mu,
            eps=eps,
            gama=gama,
            weight=weight,
            basis_function=basis_function,
        )
        self.ensemble = None
        self.n_inputs = None
        self.regressor_code = None
        self.info_values = None
        self.err = None
        self.final_model = None
        self.theta = None
        self.pivv = None

    def _validate_params(self):
        """Validate input params."""
        if not isinstance(self.n_info_values, int) or self.n_info_values < 1:
            raise ValueError(
                f"n_info_values must be integer and > zero. Got {self.n_info_values}"
            )

        if isinstance(self.ylag, int) and self.ylag < 1:
            raise ValueError(f"ylag must be integer and > zero. Got {self.ylag}")

        if isinstance(self.xlag, int) and self.xlag < 1:
            raise ValueError(f"xlag must be integer and > zero. Got {self.xlag}")

        if not isinstance(self.xlag, (int, list)):
            raise ValueError(f"xlag must be integer and > zero. Got {self.xlag}")

        if not isinstance(self.ylag, (int, list)):
            raise ValueError(f"ylag must be integer and > zero. Got {self.ylag}")

        if not isinstance(self.order_selection, bool):
            raise TypeError(
                f"order_selection must be False or True. Got {self.order_selection}"
            )

        if not isinstance(self.extended_least_squares, bool):
            raise TypeError(
                "extended_least_squares must be False or True. Got"
                f" {self.extended_least_squares}"
            )

        if self.info_criteria not in ["aic", "bic", "fpe", "lilc"]:
            raise ValueError(
                f"info_criteria must be aic, bic, fpe or lilc. Got {self.info_criteria}"
            )

        if self.model_type not in ["NARMAX", "NAR", "NFIR"]:
            raise ValueError(
                f"model_type must be NARMAX, NAR or NFIR. Got {self.model_type}"
            )

        if (
            not isinstance(self.n_terms, int) or self.n_terms < 1
        ) and self.n_terms is not None:
            raise ValueError(f"n_terms must be integer and > zero. Got {self.n_terms}")

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
        - Manuscript: Orthogonal least squares methods and their application
           to non-linear system identification
           https://eprints.soton.ac.uk/251147/1/778742007_content.pdf
        - Manuscript (portuguese): Identificação de Sistemas não Lineares
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
                    np.dot(tmp_psi[i:, j].T, tmp_psi[i:, j]) * squared_y + self.eps
                )

            if i == process_term_number:
                break

            piv_index = np.argmax(tmp_err[i:]) + i
            err[i] = tmp_err[piv_index]
            tmp_psi[:, [piv_index, i]] = tmp_psi[:, [i, piv_index]]
            piv[[piv_index, i]] = piv[[i, piv_index]]

            v = Orthogonalization().house(tmp_psi[i:, i])

            row_result = Orthogonalization().rowhouse(tmp_psi[i:, i:], v)

            tmp_y[i:] = Orthogonalization().rowhouse(tmp_y[i:], v)

            tmp_psi[i:, i:] = np.copy(row_result)

        tmp_piv = piv[0:process_term_number]
        psi_orthogonal = psi[:, tmp_piv]
        return err, piv, psi_orthogonal

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
        X_base : array-like of shape = n_samples
            Input system values measured by the user.

        Returns
        -------
        output_vector : array-like of shape = n_regressor
            Vector with values of akaike's information criterion
            for models with N terms (where N is the
            vector position + 1).

        """
        if self.n_info_values is not None and self.n_info_values > X_base.shape[1]:
            self.n_info_values = X_base.shape[1]
            warnings.warn(
                (
                    "n_info_values is greater than the maximum number of all"
                    " regressors space considering the chosen y_lag, u_lag, and"
                    f" non_degree. We set as {X_base.shape[1]}"
                ),
                stacklevel=2,
            )

        output_vector = np.zeros(self.n_info_values)
        output_vector[:] = np.nan

        n_samples = len(y) - self.max_lag

        for i in range(0, self.n_info_values):
            n_theta = i + 1
            regressor_matrix = self.error_reduction_ratio(X_base, y, n_theta)[2]

            tmp_theta = getattr(self, self.estimator)(regressor_matrix, y)

            tmp_yhat = np.dot(regressor_matrix, tmp_theta)
            tmp_residual = y[self.max_lag :] - tmp_yhat
            e_var = np.var(tmp_residual, ddof=1)

            # output_vector[i] = self.compute_info_value(n_theta, n_samples, e_var)
            output_vector[i] = self.info_criteria_function(n_theta, n_samples, e_var)

        return output_vector

    def get_info_criteria(self, info_criteria):
        """get info criteria"""
        info_criteria_options = {
            "aic": self.aic,
            "bic": self.bic,
            "fpe": self.fpe,
            "lilc": self.lilc,
        }
        return info_criteria_options.get(info_criteria)

    def bic(self, n_theta, n_samples, e_var):
        """Compute the Bayesian information criteria value.

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
        model_factor = n_theta * np.log(n_samples)
        e_factor = n_samples * np.log(e_var)
        info_criteria_value = e_factor + model_factor

        return info_criteria_value

    def aic(self, n_theta, n_samples, e_var):
        """Compute the Akaike information criteria value.

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
        model_factor = 2 * n_theta
        e_factor = n_samples * np.log(e_var)
        info_criteria_value = e_factor + model_factor

        return info_criteria_value

    def fpe(self, n_theta, n_samples, e_var):
        """Compute the Final Error Prediction value.

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
        model_factor = n_samples * np.log((n_samples + n_theta) / (n_samples - n_theta))
        e_factor = n_samples * np.log(e_var)
        info_criteria_value = e_factor + model_factor

        return info_criteria_value

    def lilc(self, n_theta, n_samples, e_var):
        """Compute the Lilc information criteria value.

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
        model_factor = 2 * n_theta * np.log(np.log(n_samples))
        e_factor = n_samples * np.log(e_var)
        info_criteria_value = e_factor + model_factor

        return info_criteria_value

    def fit(self, *, X=None, y=None):
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
        if y is None:
            raise ValueError("y cannot be None")

        self.max_lag = self._get_max_lag()
        lagged_data = self.build_matrix(X, y)

        if self.basis_function.__class__.__name__ == "Polynomial":
            reg_matrix = self.basis_function.fit(
                lagged_data, self.max_lag, predefined_regressors=None
            )
        else:
            reg_matrix, self.ensemble = self.basis_function.fit(
                lagged_data, self.max_lag, predefined_regressors=None
            )

        if X is not None:
            self.n_inputs = _num_features(X)
        else:
            self.n_inputs = 1  # just to create the regressor space base

        self.regressor_code = self.regressor_space(self.n_inputs)

        if self.order_selection is True:
            self.info_values = self.information_criterion(reg_matrix, y)

        if self.n_terms is None and self.order_selection is True:
            model_length = np.where(self.info_values == np.amin(self.info_values))
            model_length = int(model_length[0] + 1)
            self.n_terms = model_length
        elif self.n_terms is None and self.order_selection is not True:
            raise ValueError(
                "If order_selection is False, you must define n_terms value."
            )
        else:
            model_length = self.n_terms

        (self.err, self.pivv, psi) = self.error_reduction_ratio(
            reg_matrix, y, model_length
        )

        tmp_piv = self.pivv[0:model_length]
        if self.basis_function.__class__.__name__ == "Polynomial":
            self.final_model = self.regressor_code[tmp_piv, :].copy()
        elif self.basis_function.__class__.__name__ != "Polynomial" and self.ensemble:
            basis_code = np.sort(
                np.tile(
                    self.regressor_code[1:, :], (self.basis_function.repetition, 1)
                ),
                axis=0,
            )
            self.regressor_code = np.concatenate([self.regressor_code[1:], basis_code])
            self.final_model = self.regressor_code[tmp_piv, :].copy()
        else:
            self.regressor_code = np.sort(
                np.tile(
                    self.regressor_code[1:, :], (self.basis_function.repetition, 1)
                ),
                axis=0,
            )
            self.final_model = self.regressor_code[tmp_piv, :].copy()

        self.theta = getattr(self, self.estimator)(psi, y)
        if self.extended_least_squares is True:
            self.theta = self._unbiased_estimator(
                psi, y, self.theta, self.elag, self.max_lag, self.estimator
            )
        return self

    def predict(self, *, X=None, y=None, steps_ahead=None, forecast_horizon=None):
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
        if self.basis_function.__class__.__name__ == "Polynomial":
            if steps_ahead is None:
                yhat = self._model_prediction(X, y, forecast_horizon=forecast_horizon)
                yhat = np.concatenate([y[: self.max_lag], yhat], axis=0)
                return yhat
            if steps_ahead == 1:
                yhat = self._one_step_ahead_prediction(X, y)
                yhat = np.concatenate([y[: self.max_lag], yhat], axis=0)
                return yhat

            _check_positive_int(steps_ahead, "steps_ahead")
            yhat = self._n_step_ahead_prediction(X, y, steps_ahead=steps_ahead)
            yhat = np.concatenate([y[: self.max_lag], yhat], axis=0)
            return yhat

        if steps_ahead is None:
            yhat = self._basis_function_predict(X, y, forecast_horizon)
            yhat = np.concatenate([y[: self.max_lag], yhat], axis=0)
            return yhat
        if steps_ahead == 1:
            yhat = self._one_step_ahead_prediction(X, y)
            yhat = np.concatenate([y[: self.max_lag], yhat], axis=0)
            return yhat

        yhat = self._basis_function_n_step_prediction(
            X, y, steps_ahead, forecast_horizon
        )
        yhat = np.concatenate([y[: self.max_lag], yhat], axis=0)
        return yhat

    def _one_step_ahead_prediction(self, X, y):
        """Perform the 1-step-ahead prediction of a model.

        Parameters
        ----------
        y : array-like of shape = max_lag
            Initial conditions values of the model
            to start recursive process.
        X : ndarray of floats of shape = n_samples
            Vector with input values to be used in model simulation.

        Returns
        -------
        yhat : ndarray of floats
               The 1-step-ahead predicted values of the model.

        """
        lagged_data = self.build_matrix(X, y)

        if self.basis_function.__class__.__name__ == "Polynomial":
            X_base = self.basis_function.transform(
                lagged_data,
                self.max_lag,
                predefined_regressors=self.pivv[: len(self.final_model)],
            )
        else:
            X_base, _ = self.basis_function.transform(
                lagged_data,
                self.max_lag,
                predefined_regressors=self.pivv[: len(self.final_model)],
            )

        yhat = super()._one_step_ahead_prediction(X_base)
        return yhat.reshape(-1, 1)

    def _n_step_ahead_prediction(self, X, y, steps_ahead):
        """Perform the n-steps-ahead prediction of a model.

        Parameters
        ----------
        y : array-like of shape = max_lag
            Initial conditions values of the model
            to start recursive process.
        X : ndarray of floats of shape = n_samples
            Vector with input values to be used in model simulation.

        Returns
        -------
        yhat : ndarray of floats
               The n-steps-ahead predicted values of the model.

        """
        yhat = super()._n_step_ahead_prediction(X, y, steps_ahead)
        return yhat

    def _model_prediction(self, X, y_initial, forecast_horizon=0):
        """Perform the infinity steps-ahead simulation of a model.

        Parameters
        ----------
        y_initial : array-like of shape = max_lag
            Number of initial conditions values of output
            to start recursive process.
        X : ndarray of floats of shape = n_samples
            Vector with input values to be used in model simulation.

        Returns
        -------
        yhat : ndarray of floats
               The predicted values of the model.

        """
        if self.model_type in ["NARMAX", "NAR"]:
            return self._narmax_predict(X, y_initial, forecast_horizon)

        if self.model_type == "NFIR":
            return self._nfir_predict(X, y_initial)

        raise ValueError(
            f"model_type must be NARMAX, NAR or NFIR. Got {self.model_type}"
        )

    def _narmax_predict(self, X, y_initial, forecast_horizon=0):
        if len(y_initial) < self.max_lag:
            raise ValueError(
                "Insufficient initial condition elements! Expected at least"
                f" {self.max_lag} elements."
            )

        if X is not None:
            forecast_horizon = X.shape[0]
        else:
            forecast_horizon = forecast_horizon + self.max_lag

        if self.model_type == "NAR":
            self.n_inputs = 0

        y_output = super()._narmax_predict(X, y_initial, forecast_horizon)
        return y_output

    def _nfir_predict(self, X, y_initial):
        y_output = super()._nfir_predict(X, y_initial)
        return y_output

    def _basis_function_predict(self, X, y_initial, forecast_horizon=None):
        if X is not None:
            forecast_horizon = X.shape[0]
        else:
            forecast_horizon = forecast_horizon + self.max_lag

        if self.model_type == "NAR":
            self.n_inputs = 0

        yhat = super()._basis_function_predict(X, y_initial, forecast_horizon)
        return yhat.reshape(-1, 1)

    def _basis_function_n_step_prediction(self, X, y, steps_ahead, forecast_horizon):
        """Perform the n-steps-ahead prediction of a model.

        Parameters
        ----------
        y : array-like of shape = max_lag
            Initial conditions values of the model
            to start recursive process.
        X : ndarray of floats of shape = n_samples
            Vector with input values to be used in model simulation.

        Returns
        -------
        yhat : ndarray of floats
               The n-steps-ahead predicted values of the model.

        """
        if len(y) < self.max_lag:
            raise ValueError(
                "Insufficient initial condition elements! Expected at least"
                f" {self.max_lag} elements."
            )

        if X is not None:
            forecast_horizon = X.shape[0]
        else:
            forecast_horizon = forecast_horizon + self.max_lag

        yhat = super()._basis_function_n_step_prediction(
            X, y, steps_ahead, forecast_horizon
        )
        return yhat.reshape(-1, 1)

    def _basis_function_n_steps_horizon(self, X, y, steps_ahead, forecast_horizon):
        yhat = super()._basis_function_n_steps_horizon(
            X, y, steps_ahead, forecast_horizon
        )
        return yhat.reshape(-1, 1)
