"""Base classes for NARMAX estimator."""

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause


import numpy as np
from itertools import combinations_with_replacement
from itertools import chain
from collections import Counter
from .utils._check_arrays import check_X_y, _check_positive_int
import warnings


class GenerateRegressors:
    """Polynomial NARMAX model

    Provides the main functions to generate the regressor dictionary
    and regressor codes for polynomial basis.
    """

    def create_narmax_code(self, non_degree, xlag, ylag, n_inputs):
        """Create the code representation of the regressors.

        This function generates a codification from all possibles
        regressors given the maximum lag of the input and output.
        This is used to write the final terms of the model in a
        readable form. [1001] -> y(k-1).
        This code format was based on a dissertation from UFMG. See
        reference below.

        Parameters
        ----------
        non_degree : int
            The desired maximum nonlinearity degree.
        ylag : int
            The maximum lag of output regressors.
        xlag : int
            The maximum lag of input regressors.

        Returns
        -------
        max_lag : int
            This value can be used by another functions.
        regressor_code : ndarray of int
            Matrix codification of all possible regressors.

        Examples
        --------
        The codification is defined as:

        >>> 100n = y(k-n)
        >>> 200n = u(k-n)
        >>> [100n 100n] = y(k-n)y(k-n)
        >>> [200n 200n] = u(k-n)u(k-n)

        References
        ----------
        [1] Master Thesis: Barbosa, Alípio Monteiro.
            Técnicas de otimização bi-objetivo para a determinação
            da estrutura de modelos NARX (2010).

        """
        if not isinstance(non_degree, int) or non_degree < 1:
            raise ValueError(
                "non_degree must be integer and > zero. Got %f" % non_degree
            )

        if not isinstance(ylag, (int, list)) or np.min(np.minimum(ylag, 1)) < 1:
            raise ValueError("ylag must be integer or list and > zero. Got %f" % ylag)

        if (
            not isinstance(xlag, (int, list))
            # or np.min(np.minimum(xlag, 1)) < 1):
            or np.min(np.min(list(chain.from_iterable([[xlag]])))) < 1
        ):
            raise ValueError("xlag must be integer or list and > zero. Got %f" % xlag)

        if not isinstance(n_inputs, int) or n_inputs < 1:
            raise ValueError("n_inputs must be integer and > zero. Got %f" % n_inputs)

        if isinstance(ylag, list):
            # create only the lags passed from list
            y_vec = []
            y_vec.extend([lag + 1000 for lag in ylag])
            y_vec = np.array(y_vec)
        else:
            # create a range of lags if passed a int value
            y_vec = np.arange(1001, 1001 + ylag)

        if isinstance(xlag, list) and n_inputs == 1:
            # create only the lags passed from list
            x_vec_tmp = []
            x_vec_tmp.extend([lag + 2000 for lag in xlag])
            x_vec_tmp = np.array(x_vec_tmp)
        elif isinstance(xlag, int) and n_inputs == 1:
            # create a range of lags if passed a int value
            x_vec_tmp = np.arange(2001, 2001 + xlag)
        elif n_inputs > 1:
            # only list are allowed if n_inputs > 1
            # the user must entered list of the desired lags explicitly
            x_vec_tmp = []
            for i in range(n_inputs):
                if isinstance(xlag[i], list) and n_inputs > 1:
                    # create 200n, 300n,..., 400n to describe each input
                    x_vec_tmp.extend([lag + 2000 + i * 1000 for lag in xlag[i]])
                elif isinstance(xlag[i], int) and n_inputs > 1:
                    x_vec_tmp.extend(
                        [np.arange(2001 + i * 1000, 2001 + i * 1000 + xlag[i])]
                    )

        if n_inputs > 1:
            # if x_vec is a nested list, ensure all elements are arrays
            all_arrays = [np.array([i]) if isinstance(i, int) else i for i in x_vec_tmp]
            x_vec = np.concatenate([i for i in all_arrays])
        else:
            x_vec = x_vec_tmp

        return x_vec, y_vec
    
    def regressor_space(self, non_degree, xlag, ylag, n_inputs, model_type="NARMAX"):
        x_vec, y_vec = self.create_narmax_code(non_degree, xlag, ylag, n_inputs)
        reg_aux = np.array([0])
        if model_type == "NARMAX":
            reg_aux = np.concatenate([reg_aux, y_vec, x_vec])
        elif model_type == "NAR":
            reg_aux = np.concatenate([reg_aux, y_vec])
        elif model_type == "NFIR":
            reg_aux = np.concatenate([reg_aux, x_vec])
        else:
            raise Exception("Unrecognized model type. Model type should be NARMAX, NAR or NFIR")

        regressor_code = list(combinations_with_replacement(reg_aux, non_degree))

        regressor_code = np.array(regressor_code)
        regressor_code = regressor_code[:, regressor_code.shape[1] :: -1]
        return regressor_code
    
    
class HouseHolder:
    """Householder reflection and transformation."""

    def _house(self, x):
        """Perform a Householder reflection of vector.

        Parameters
        ----------
        x : array-like of shape = number_of_training_samples
            The respective column of the matrix of regressors in each
            iteration of ERR function.

        Returns
        -------
        v : array-like of shape = number_of_training_samples
            The reflection of the array x.

        References
        ----------
        [1] Manuscript: Chen, S., Billings, S. A., & Luo, W. (1989).
            Orthogonal least squares methods and their application to non-linear system identification.

        """
        u = np.linalg.norm(x, 2)
        if u != 0:
            aux_b = x[0] + np.sign(x[0]) * u
            x = x[1:] / aux_b
            x = np.concatenate((np.array([1]), x))
        return x

    def _rowhouse(self, RA, v):
        """Perform a row Householder transformation.

        Parameters
        ----------
        RA : array-like of shape = number_of_training_samples
            The respective column of the matrix of regressors in each
            iteration of ERR function.
        v : array-like of shape = number_of_training_samples
            The reflected vector obtained by using the householder reflection.

        Returns
        -------
        B : array-like of shape = number_of_training_samples

        References
        ----------
        [1] Manuscript: Chen, S., Billings, S. A., & Luo, W. (1989).
            Orthogonal least squares methods and their application to
            non-linear system identification. International Journal of
            control, 50(5), 1873-1896.

        """
        b = -2 / np.dot(v.T, v)
        w = b * np.dot(RA.T, v)
        w = w.reshape(1, -1)
        v = v.reshape(-1, 1)
        RA = RA + v * w
        B = RA
        return B
    
class ModelInformation:
    def _list_output_regressor_code(self, model_code):
        """Create a flattened array of output regressors.

        Parameters
        ----------
        model_code : ndarray of int
            Model defined by the user to simulate.

        Returns
        -------
        model_code : ndarray of int
            Flattened list of output regressors.

        """
        regressor_code = [
            code for code in model_code.ravel() if (code != 0) and (str(code)[0] == "1")
        ]

        return np.asarray(regressor_code)

    def _list_input_regressor_code(self, model_code):
        """Create a flattened array of input regressors.

        Parameters
        ----------
        model_code : ndarray of int
            Model defined by the user to simulate.

        Returns
        -------
        model_code : ndarray of int
            Flattened list of output regressors.

        """
        regressor_code = [
            code for code in model_code.ravel() if (code != 0) and (str(code)[0] != "1")
        ]
        return np.asarray(regressor_code)

    def _get_lag_from_regressor_code(self, regressors):
        """Get the maximum lag from array of regressors.

        Parameters
        ----------
        regressors : ndarray of int
            Flattened list of input or output regressors.

        Returns
        -------
        max_lag : int
            Maximum lag of list of regressors.

        """
        lag_list = [
            int(i) for i in regressors.astype("str") for i in [np.sum(int(i[2:]))]
        ]
        if len(lag_list) != 0:
            return max(lag_list)
        else:
            return 1
        
    def _get_max_lag_from_model_code(self, model_code):
        xlag_code = self._list_input_regressor_code(model_code)
        ylag_code = self._list_output_regressor_code(model_code)
        xlag = self._get_lag_from_regressor_code(xlag_code)
        ylag = self._get_lag_from_regressor_code(ylag_code)
        return max(xlag, ylag)
    
    def _get_max_lag(ylag=1, xlag=1):
        """Get the max lag defined by the user.

        Parameters
        ----------
        ylag : int
            The maximum lag of output regressors.
        xlag : int
            The maximum lag of input regressors.

        Returns
        -------
        max_lag = int
            The max lag value defined by the user.
        """
        ny = np.max(list(chain.from_iterable([[ylag]])))
        nx = np.max(list(chain.from_iterable([[xlag]])))
        return np.max([ny, np.max(nx)])
    
class InformationMatrix:
    """Class for methods regarding preprocessing of columns"""

    def shift_column(self, col_to_shift, lag):
        """Shift values based on a lag.

        Parameters
        ----------
        col_to_shift : array-like of shape = n_samples
            The samples of the input or output.
        lag : int
            The respective lag of the regressor.

        Returns
        -------
        tmp_column : array-like of shape = n_samples
            The shifted array of the input or output.

        Examples
        --------
        >>> y = [1, 2, 3, 4, 5]
        >>> shift_column(y, 1)
        [0, 1, 2, 3, 4]

        """
        n_samples = col_to_shift.shape[0]
        tmp_column = np.zeros((n_samples, 1))
        aux = col_to_shift[0 : n_samples - lag]
        aux = np.reshape(aux, (len(aux), 1))
        tmp_column[lag:, 0] = aux[:, 0]
        return tmp_column

    def _process_xlag(self, X, xlag):
        """Create the list of lags to be used for the inputs

        Parameters
        ----------
        X : array-like
            Input data used on training phase.
        xlag : int
            The maximum lag of input regressors.

        Returns
        -------
        x_lag : ndarray of int
            The range of lags according to user definition.

        """
        n_inputs = X.shape[1]
        if isinstance(xlag, int) and n_inputs > 1:
            raise ValueError(
                "If n_inputs > 1, xlag must be a nested list. Got %f" % xlag
            )

        if isinstance(xlag, int):
            xlag = range(1, xlag + 1)

        return n_inputs, xlag

    def _process_ylag(self, ylag):
        """Create the list of lags to be used for the outputs

        Parameters
        ----------
        ylag : int, list
            The maximum lag of input regressors.

        Returns
        -------
        y_lag : ndarray of int
            The range of lags according to user definition.

        """
        if isinstance(ylag, int):
            ylag = range(1, ylag + 1)

        return ylag

    def _create_lagged_X(self, X, xlag, n_inputs):
        """Create a lagged matrix of inputs without combinations.

        Parameters
        ----------
        X : array-like
            Input data used on training phase.
        xlag : int
            The maximum lag of input regressors.
        n_inputs : int
            Number of input variables.

        Returns
        -------
        x_lagged : ndarray of floats
            A lagged input matrix formed by the input regressors
            without combinations.

        """
        if n_inputs == 1:
            x_lagged = np.column_stack(
                [self.shift_column(X[:, 0], lag) for lag in xlag]
            )
        else:
            x_lagged = np.zeros([len(X), 1])  # just to stack other columns
            # if user input a nested list like [[1, 2], 4], the following
            # line convert it to [[1, 2], [4]].
            # Remember, for multiple inputs all lags must be entered explicitly
            xlag = [[i] if isinstance(i, int) else i for i in xlag]
            for col in range(n_inputs):
                x_lagged_col = np.column_stack(
                    [self.shift_column(X[:, col], lag) for lag in xlag[col]]
                )
                x_lagged = np.column_stack([x_lagged, x_lagged_col])

            x_lagged = x_lagged[:, 1:]  # remove the column of 0 created above

        return x_lagged

    def _create_lagged_y(self, y, ylag):
        """Create a lagged matrix of the output without combinations.

        Parameters
        ----------
        y : array-like
            Output data used on training phase.
        ylag : int
            The maximum lag of output regressors.

        Returns
        -------
        y_lagged : ndarray of floats
            A lagged input matrix formed by the output regressors
            without combinations.

        """
        y_lagged = np.column_stack([self.shift_column(y[:, 0], lag) for lag in ylag])
        return y_lagged

    def initial_lagged_matrix(self, X, y, xlag, ylag):
        """Build a lagged matrix concerning each lag for each column.

        Parameters
        ----------
        model : ndarray of int
            The model code representation.
        y : array-like
            Target data used on training phase.
        X : array-like
            Input data used on training phase.
        ylag : int
            The maximum lag of output regressors.
        xlag : int
            The maximum lag of input regressors.

        Returns
        -------
        lagged_data : ndarray of floats
            The lagged matrix built in respect with each lag and column.

        Examples
        --------
        Let X and y be the input and output values of shape Nx1.
        If the chosen lags are 2 for both input and output
        the initial lagged matrix will be formed by Y[k-1], Y[k-2],
        X[k-1], and X[k-2].

        """
        n_inputs, xlag = self._process_xlag(X, xlag)

        ylag = self._process_ylag(ylag)

        x_lagged = self._create_lagged_X(X, xlag, n_inputs)

        y_lagged = self._create_lagged_y(y, ylag)
        lagged_data = np.concatenate([y_lagged, x_lagged], axis=1)
        return lagged_data
    
    def build_output_matrix(self, y, ylag, non_degree, predefined_regressors=None):
        """Build the information matrix.

        Each columns of the information matrix represents a candidate
        regressor. The set of candidate regressors are based on xlag,
        ylag, and non_degree entered by the user.

        Parameters
        ----------
        model : ndarray of int
            The model code representation.
        y : array-like
            Target data used on training phase.
        ylag : int
            The maximum lag of output regressors.
        non_degree : int
            The desired maximum nonlinearity degree.

        Returns
        -------
        lagged_data = ndarray of floats
            The lagged matrix built in respect with each lag and column.

        """
        # Generate a lagged data which each column is a input or output
        # related to its respective lags. With this approach we can create
        # the information matrix by using all possible combination of
        # the columns as a product in the iterations
        ylag = self._process_ylag(ylag=ylag)
        y_lagged = self._create_lagged_y(y, ylag)
        constant = np.ones([y_lagged.shape[0], 1])
        data = np.concatenate([constant, y_lagged], axis=1)
        return data
    
    def build_input_matrix(self, X, xlag, non_degree, predefined_regressors=None):
        """Build the information matrix.

        Each columns of the information matrix represents a candidate
        regressor. The set of candidate regressors are based on xlag,
        ylag, and non_degree entered by the user.

        Parameters
        ----------
        model : ndarray of int
            The model code representation.
        X : array-like
            Input data used on training phase.
        xlag : int
            The maximum lag of input regressors.
        non_degree : int
            The desired maximum nonlinearity degree.

        Returns
        -------
        lagged_data = ndarray of floats
            The lagged matrix built in respect with each lag and column.

        """
        # Generate a lagged data which each column is a input or output
        # related to its respective lags. With this approach we can create
        # the information matrix by using all possible combination of
        # the columns as a product in the iterations
        
        n_inputs, xlag = self._process_xlag(X, xlag)

        x_lagged = self._create_lagged_X(X, xlag, n_inputs)
        
        constant = np.ones([x_lagged.shape[0], 1])
        data = np.concatenate([constant, x_lagged], axis=1)

        return data

    def build_input_output_matrix(self, X, y, xlag, ylag, non_degree, predefined_regressors=None):
        """Build the information matrix.

        Each columns of the information matrix represents a candidate
        regressor. The set of candidate regressors are based on xlag,
        ylag, and non_degree entered by the user.

        Parameters
        ----------
        model : ndarray of int
            The model code representation.
        y : array-like
            Target data used on training phase.
        X : array-like
            Input data used on training phase.
        ylag : int
            The maximum lag of output regressors.
        xlag : int
            The maximum lag of input regressors.
        non_degree : int
            The desired maximum nonlinearity degree.

        Returns
        -------
        lagged_data = ndarray of floats
            The lagged matrix built in respect with each lag and column.

        """
        # Generate a lagged data which each column is a input or output
        # related to its respective lags. With this approach we can create
        # the information matrix by using all possible combination of
        # the columns as a product in the iterations
        lagged_data = self.initial_lagged_matrix(X, y, xlag=xlag, ylag=ylag)

        constant = np.ones([lagged_data.shape[0], 1])
        data = np.concatenate([constant, lagged_data], axis=1)        
        return data
    
class ModelPrediction:
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
            _check_positive_int(steps_ahead, "steps_ahead")
            return self._n_step_ahead_prediction(X, y, steps_ahead=steps_ahead)

    def _code2exponents(self, code):
        """
        Convert regressor code to exponents array.

        Parameters
        ----------
        code : 1D-array of int
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
        if self.model_type == "NAR":
            warnings.warn(
                (
                    "Because the user chooses NAR model , the model built"
                    "will be of the form y(k) = F(y[k-1], y[k-2], ..., y[k-n]) + e(k)"
                ),
                stacklevel=2,
            )
            lagged_data = self.build_output_matrix(y, self.ylag, self.non_degree)
            self.max_lag = ModelInformation()._get_max_lag(ylag=self.ylag)
        elif self.model_type == "NFIR":
            warnings.warn(
                (
                    "Because the user chooses the NFIR model, the model built"
                    "will be of the form y(k) = F(X[k-1], X[k-2], ..., X[k-n]) + e(k)"
                ),
                stacklevel=2,
            )
            lagged_data = self.build_input_matrix(X, self.xlag, self.non_degree)
            self.max_lag = ModelInformation()._get_max_lag(xlag=self.xlag)
        elif self.model_type == "NARMAX":
            warnings.warn(
                (
                    "Because the user chooses NARMAX model, the model built"
                    "will be of the form y(k) = F(y[k-1], y[k-2], ..., y[k-n], X[k-1], X[k-2], ..., X[k-n]) + e(k)"
                ),
                stacklevel=2,
            )
            check_X_y(X, y)
            self.max_lag = ModelInformation()._get_max_lag(ylag=self.ylag, xlag=self.xlag)
            lagged_data = self.build_input_output_matrix(X, y, self.xlag, self.ylag, self.non_degree)
        else:
            raise ValueError("Unrecognized model type. The model_type should be NARMAX, NAR or NFIR.")
        
        X_base = self.basis_function.build_polynomial_basis(
            lagged_data, self.non_degree, self.max_lag, predefined_regressors=None)
        
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
            Vector with input values to be used in model simulation.

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
            return self._narmax_predict(X, y_initial)
        elif self.model_type == "NFIR":
            return self._nfir_predict(X)
        else:
            raise Exception("model_type do not exist! Model type must be NARMAX, NAR or NFIR")
    
    def _narmax_predict(self, X, y_initial):
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

    def _nfir_predict(self, X):
        y_output = np.zeros(X.shape[0], dtype=float)
        y_output.fill(np.nan)
        X = X.reshape(-1, self._n_inputs)
        model_exponents = [self._code2exponents(model) for model in self.final_model]
        raw_regressor = np.zeros(len(model_exponents[0]), dtype=float)
        for i in range(self.max_lag, X.shape[0]):
            init = 0
            final = self.max_lag
            k = int(i - self.max_lag)
            for j in range(self._n_inputs):
                raw_regressor[init:final] = X[k:i, j]
                init += self.max_lag
                final += self.max_lag

            regressor_value = np.zeros(len(model_exponents))
            for j in range(len(model_exponents)):
                regressor_value[j] = np.prod(
                    np.power(raw_regressor, model_exponents[j])
                )

            y_output[i] = np.dot(regressor_value, self.theta.flatten())
        return y_output.reshape(-1, 1)