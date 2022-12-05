"""Base classes for NARMAX estimator."""

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause


from collections import Counter
from itertools import chain, combinations_with_replacement

import numpy as np

from .utils._check_arrays import _num_features


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
        - Master Thesis: Barbosa, Alípio Monteiro.
            Técnicas de otimização bi-objetivo para a determinação
            da estrutura de modelos NARX (2010).

        """
        if not isinstance(non_degree, int) or non_degree < 1:
            raise ValueError(f"non_degree must be integer and > zero. Got {non_degree}")

        if not isinstance(ylag, (int, list)) or np.min(np.minimum(ylag, 1)) < 1:
            raise ValueError(f"ylag must be integer or list and > zero. Got {ylag}")

        if (
            not isinstance(xlag, (int, list))
            # or np.min(np.minimum(xlag, 1)) < 1):
            or np.min(np.min(list(chain.from_iterable([[xlag]])))) < 1
        ):
            raise ValueError(f"xlag must be integer or list and > zero. Got {xlag}")

        if not isinstance(n_inputs, int) or n_inputs < 1:
            raise ValueError(f"n_inputs must be integer and > zero. Got {n_inputs}")

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
        """Create regressor code based on model type."""
        x_vec, y_vec = self.create_narmax_code(non_degree, xlag, ylag, n_inputs)
        reg_aux = np.array([0])
        if model_type == "NARMAX":
            reg_aux = np.concatenate([reg_aux, y_vec, x_vec])
        elif model_type == "NAR":
            reg_aux = np.concatenate([reg_aux, y_vec])
        elif model_type == "NFIR":
            reg_aux = np.concatenate([reg_aux, x_vec])
        else:
            raise ValueError(
                "Unrecognized model type. Model type should be NARMAX, NAR or NFIR"
            )

        regressor_code = list(combinations_with_replacement(reg_aux, non_degree))

        regressor_code = np.array(regressor_code)
        regressor_code = regressor_code[:, regressor_code.shape[1] :: -1]
        return regressor_code


class HouseHolder:
    """Householder reflection and transformation."""

    def house(self, x):
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
        - Manuscript: Chen, S., Billings, S. A., & Luo, W. (1989).
            Orthogonal least squares methods and their application to non-linear
            system identification.

        """
        u = np.linalg.norm(x, 2)
        if u != 0:
            aux_b = x[0] + np.sign(x[0]) * u
            x = x[1:] / aux_b
            x = np.concatenate((np.array([1]), x))
        return x

    def rowhouse(self, RA, v):
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
        - Manuscript: Chen, S., Billings, S. A., & Luo, W. (1989).
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
    """Methods to get model information"""

    def _get_index_from_regressor_code(self, regressor_code, model_code):
        """Get the index of user regressor in regressor space.

        Took from: https://stackoverflow.com/questions/38674027/find-the-row-indexes-of-several-values-in-a-numpy-array/38674038#38674038

        Parameters
        ----------
        regressor_code : ndarray of int
            Matrix codification of all possible regressors.
        model_code : ndarray of int
            Model defined by the user to simulate.

        Returns
        -------
        model_index : ndarray of int
            Index of model code in the regressor space.

        """
        dims = regressor_code.max(0) + 1
        model_index = np.where(
            np.in1d(
                np.ravel_multi_index(regressor_code.T, dims),
                np.ravel_multi_index(model_code.T, dims),
            )
        )[0]
        return model_index

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

    def _get_max_lag(self, ylag=1, xlag=1):
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
        aux = col_to_shift[0 : n_samples - lag].reshape(-1, 1)
        # aux = np.reshape(aux, (len(aux), 1))
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
        n_inputs = _num_features(X)
        if isinstance(xlag, int) and n_inputs > 1:
            raise ValueError(f"If n_inputs > 1, xlag must be a nested list. Got {xlag}")

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

    def build_output_matrix(self, y, ylag):
        """Build the information matrix of output values.

        Each columns of the information matrix represents a candidate
        regressor. The set of candidate regressors are based on xlag,
        ylag, and non_degree entered by the user.

        Parameters
        ----------
        y : array-like
            Target data used on training phase.
        ylag : int
            The maximum lag of output regressors.

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

    def build_input_matrix(self, X, xlag):
        """Build the information matrix of input values.

        Each columns of the information matrix represents a candidate
        regressor. The set of candidate regressors are based on xlag,
        ylag, and non_degree entered by the user.

        Parameters
        ----------
        X : array-like
            Input data used on training phase.
        xlag : int
            The maximum lag of input regressors.

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

    def build_input_output_matrix(self, X, y, xlag, ylag):
        """Build the information matrix.

        Each columns of the information matrix represents a candidate
        regressor. The set of candidate regressors are based on xlag,
        ylag, and non_degree entered by the user.

        Parameters
        ----------
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
    """Model prediction methods."""

    def __init__(
        self,
        model_type="NARMAX",
        basis_function=None,
        max_lag=1,
        xlag=1,
        ylag=1,
        final_model=None,
        theta=None,
        pivv=None,
    ):
        self.model_type = model_type
        self.im = InformationMatrix()
        self.basis_function = basis_function
        self.max_lag = max_lag
        self.xlag = xlag
        self.ylag = ylag
        self.final_model = final_model
        self.theta = theta
        self.pivv = pivv

    def _code2exponents(self, *, code, max_lag, n_inputs):
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
            return np.zeros(max_lag * (1 + n_inputs))

        exponents = np.array([], dtype=float)
        elements = np.round(np.divide(regressors, 1000), 0)[(regressors > 0)].astype(
            int
        )

        for j in range(1, n_inputs + 2):
            base_exponents = np.zeros(max_lag, dtype=float)
            if j in elements:
                for i in range(1, max_lag + 1):
                    regressor_code = int(j * 1000 + i)
                    base_exponents[-i] = regressors_count[regressor_code]
                exponents = np.append(exponents, base_exponents)

            else:
                exponents = np.append(exponents, base_exponents)

        return exponents

    def _one_step_ahead_prediction(
        self, *, X, y, ylag, xlag, max_lag, final_model, theta, pivv
    ):
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
            lagged_data = self.im.build_output_matrix(y, ylag)
        elif self.model_type == "NFIR":
            lagged_data = self.im.build_input_matrix(X, xlag)
        elif self.model_type == "NARMAX":
            lagged_data = self.im.build_input_output_matrix(X, y, xlag, ylag)
        else:
            raise ValueError(
                "Unrecognized model type. The model_type should be NARMAX, NAR or NFIR."
            )

        if self.basis_function.__class__.__name__ == "Polynomial":
            X_base = self.basis_function.transform(
                lagged_data,
                max_lag,
                predefined_regressors=pivv[: len(final_model)],
            )
        else:
            X_base, _ = self.basis_function.transform(
                lagged_data,
                max_lag,
                predefined_regressors=pivv[: len(final_model)],
            )

        yhat = np.dot(X_base, theta.flatten())
        yhat = np.concatenate([y[:max_lag].flatten(), yhat])
        return yhat.reshape(-1, 1)

    def _n_step_ahead_prediction(
        self, *, X, y, steps_ahead, theta, final_model, max_lag, n_inputs
    ):
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
        if len(y) < max_lag:
            raise Exception("Insufficient initial conditions elements!")

        yhat = np.zeros(X.shape[0], dtype=float)
        yhat.fill(np.nan)
        yhat[:max_lag] = y[:max_lag, 0]
        i = max_lag
        X = X.reshape(-1, n_inputs)
        while i < len(y):
            k = int(i - max_lag)
            if i + steps_ahead > len(y):
                steps_ahead = len(y) - i  # predicts the remaining values

            yhat[i : i + steps_ahead] = self._model_prediction(
                X=X[k : i + steps_ahead],
                y_initial=y[k : i + steps_ahead],
                max_lag=max_lag,
                theta=theta,
                n_inputs=n_inputs,
                final_model=final_model,
            )[-steps_ahead:].ravel()

            i += steps_ahead

        yhat = yhat.ravel()
        return yhat.reshape(-1, 1)

    def _model_prediction(
        self,
        *,
        X,
        y_initial,
        max_lag,
        theta,
        n_inputs,
        final_model,
        forecast_horizon=None,
    ):
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
            return self._narmax_predict(
                X=X,
                y_initial=y_initial,
                forecast_horizon=forecast_horizon,
                max_lag=max_lag,
                n_inputs=n_inputs,
                final_model=final_model,
                theta=theta,
            )
        if self.model_type == "NFIR":
            return self._nfir_predict(
                X=X,
                y_initial=y_initial,
                max_lag=max_lag,
                n_inputs=n_inputs,
                final_model=final_model,
                theta=theta,
            )

        raise Exception(
            "model_type do not exist! Model type must be NARMAX, NAR or NFIR"
        )

    def _narmax_predict(
        self,
        *,
        X,
        y_initial,
        forecast_horizon,
        max_lag,
        n_inputs,
        final_model,
        theta,
    ):
        if len(y_initial) < max_lag:
            raise Exception("Insufficient initial conditions elements!")

        if X is not None:
            forecast_horizon = X.shape[0]
        else:
            forecast_horizon = forecast_horizon + max_lag

        if self.model_type == "NAR":
            n_inputs = 0

        y_output = np.zeros(forecast_horizon, dtype=float)
        y_output.fill(np.nan)
        y_output[:max_lag] = y_initial[:max_lag, 0]

        model_exponents = [
            self._code2exponents(code=model, max_lag=max_lag, n_inputs=n_inputs)
            for model in final_model
        ]
        raw_regressor = np.zeros(len(model_exponents[0]), dtype=float)
        for i in range(max_lag, forecast_horizon):
            init = 0
            final = max_lag
            k = int(i - max_lag)
            raw_regressor[:final] = y_output[k:i]
            for j in range(n_inputs):
                init += max_lag
                final += max_lag
                raw_regressor[init:final] = X[k:i, j]

            regressor_value = np.zeros(len(model_exponents))
            for j, model_exponent in enumerate(model_exponents):
                regressor_value[j] = np.prod(np.power(raw_regressor, model_exponent))

            y_output[i] = np.dot(regressor_value, theta.flatten())
        return y_output.reshape(-1, 1)

    def _nfir_predict(self, *, X, y_initial, max_lag, n_inputs, final_model, theta):
        y_output = np.zeros(X.shape[0], dtype=float)
        y_output.fill(np.nan)
        y_output[:max_lag] = y_initial[:max_lag, 0]
        X = X.reshape(-1, n_inputs)
        model_exponents = [
            self._code2exponents(code=model, max_lag=max_lag, n_inputs=n_inputs)
            for model in final_model
        ]
        raw_regressor = np.zeros(len(model_exponents[0]), dtype=float)
        for i in range(max_lag, X.shape[0]):
            init = 0
            final = max_lag
            k = int(i - max_lag)
            for j in range(n_inputs):
                raw_regressor[init:final] = X[k:i, j]
                init += max_lag
                final += max_lag

            regressor_value = np.zeros(len(model_exponents))
            for j, model_exponent in enumerate(model_exponents):
                regressor_value[j] = np.prod(np.power(raw_regressor, model_exponent))

            y_output[i] = np.dot(regressor_value, theta.flatten())
        return y_output.reshape(-1, 1)

    def _basis_function_predict(
        self,
        X,
        y_initial,
        xlag,
        ylag,
        theta,
        max_lag,
        final_model,
        pivv,
        forecast_horizon=None,
    ):
        if X is not None:
            forecast_horizon = X.shape[0]
        else:
            forecast_horizon = forecast_horizon + max_lag

        if self.model_type == "NAR":
            n_inputs = 0

        yhat = np.zeros(forecast_horizon, dtype=float)
        yhat.fill(np.nan)
        yhat[:max_lag] = y_initial[:max_lag, 0]

        analyzed_elements_number = max_lag + 1

        for i in range(0, forecast_horizon - max_lag):
            if self.model_type == "NARMAX":
                lagged_data = self.im.build_input_output_matrix(
                    X[i : i + analyzed_elements_number],
                    yhat[i : i + analyzed_elements_number].reshape(-1, 1),
                    xlag,
                    ylag,
                )
            elif self.model_type == "NAR":
                lagged_data = self.im.build_output_matrix(
                    yhat[i : i + analyzed_elements_number].reshape(-1, 1), ylag
                )
            elif self.model_type == "NFIR":
                lagged_data = self.im.build_input_matrix(
                    X[i : i + analyzed_elements_number], xlag
                )
            else:
                raise ValueError(
                    "Unrecognized model type. The model_type should be NARMAX, NAR or"
                    " NFIR."
                )

            X_tmp, _ = self.basis_function.transform(
                lagged_data,
                max_lag,
                predefined_regressors=pivv[: len(final_model)],
            )

            a = X_tmp @ theta
            yhat[i + max_lag] = a[:, 0]

        return yhat.reshape(-1, 1)

    def basis_function_n_step_prediction(
        self,
        *,
        X,
        y,
        xlag,
        ylag,
        max_lag,
        steps_ahead,
        theta,
        pivv,
        final_model,
        forecast_horizon,
    ):
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
        if len(y) < max_lag:
            raise Exception("Insufficient initial conditions elements!")

        if X is not None:
            forecast_horizon = X.shape[0]
        else:
            forecast_horizon = forecast_horizon + max_lag

        yhat = np.zeros(forecast_horizon, dtype=float)
        yhat.fill(np.nan)
        yhat[:max_lag] = y[:max_lag, 0]

        # Discard unnecessary initial values
        i = max_lag

        while i < len(y):
            k = int(i - max_lag)
            if i + steps_ahead > len(y):
                steps_ahead = len(y) - i  # predicts the remaining values

            if self.model_type == "NARMAX":
                yhat[i : i + steps_ahead] = self._basis_function_predict(
                    X=X[k : i + steps_ahead],
                    y_initial=y[k : i + steps_ahead],
                    theta=theta,
                    max_lag=max_lag,
                    final_model=final_model,
                    xlag=xlag,
                    ylag=ylag,
                    pivv=pivv,
                )[-steps_ahead:].ravel()
            elif self.model_type == "NAR":
                yhat[i : i + steps_ahead] = self._basis_function_predict(
                    X=None,
                    y_initial=y[k : i + steps_ahead],
                    theta=theta,
                    forecast_horizon=forecast_horizon,
                    max_lag=max_lag,
                    xlag=xlag,
                    ylag=ylag,
                    pivv=pivv,
                    final_model=final_model,
                )[-forecast_horizon : -forecast_horizon + steps_ahead].ravel()
            elif self.model_type == "NFIR":
                yhat[i : i + steps_ahead] = self._basis_function_predict(
                    X=X[k : i + steps_ahead],
                    y_initial=y[k : i + steps_ahead],
                    theta=theta,
                    max_lag=max_lag,
                    xlag=xlag,
                    ylag=ylag,
                    pivv=pivv,
                    final_model=final_model,
                )[-steps_ahead:].ravel()
            else:
                raise ValueError(
                    "Unrecognized model type. The model_type should be NARMAX, NAR or"
                    " NFIR."
                )

            i += steps_ahead
        return yhat.reshape(-1, 1)

    def _basis_function_n_steps_horizon(
        self,
        *,
        X,
        y,
        xlag,
        ylag,
        pivv,
        final_model,
        max_lag,
        theta,
        steps_ahead,
        forecast_horizon,
    ):
        yhat = np.zeros(forecast_horizon, dtype=float)
        yhat.fill(np.nan)
        yhat[:max_lag] = y[:max_lag, 0]

        # Discard unnecessary initial values
        i = max_lag

        while i < len(y):
            k = int(i - max_lag)
            if i + steps_ahead > len(y):
                steps_ahead = len(y) - i  # predicts the remaining values

            if self.model_type == "NARMAX":
                yhat[i : i + steps_ahead] = self._basis_function_predict(
                    X=X[k : i + steps_ahead],
                    y_initial=y[k : i + steps_ahead],
                    theta=theta,
                    max_lag=max_lag,
                    xlag=xlag,
                    ylag=ylag,
                    pivv=pivv,
                    final_model=final_model,
                )[-forecast_horizon : -forecast_horizon + steps_ahead].ravel()
            elif self.model_type == "NAR":
                yhat[i : i + steps_ahead] = self._basis_function_predict(
                    X=None,
                    y_initial=y[k : i + steps_ahead],
                    theta=theta,
                    forecast_horizon=forecast_horizon,
                    max_lag=max_lag,
                    xlag=xlag,
                    ylag=ylag,
                    pivv=pivv,
                    final_model=final_model,
                )[-forecast_horizon : -forecast_horizon + steps_ahead].ravel()
            elif self.model_type == "NFIR":
                yhat[i : i + steps_ahead] = self._basis_function_predict(
                    X=X[k : i + steps_ahead],
                    y_initial=y[k : i + steps_ahead],
                    theta=theta,
                    max_lag=max_lag,
                    xlag=xlag,
                    ylag=ylag,
                    pivv=pivv,
                    final_model=final_model,
                )[-forecast_horizon : -forecast_horizon + steps_ahead].ravel()
            else:
                raise ValueError(
                    "Unrecognized model type. The model_type should be NARMAX, NAR or"
                    " NFIR."
                )

            i += steps_ahead

        yhat = yhat.ravel()
        return yhat.reshape(-1, 1)
