"""Base classes for NARX estimator."""

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
#           Luan Pascoal da Costa Andrade <luan_pascoal13@hotmail.com>
#           Samuel Carlos Pessoa Oliveira <samuelcpoliveira@gmail.com>
#           Samir Angelo Milani Martins <martins@ufsj.edu.br>
# License: BSD 3 clause


import numpy as np
from itertools import combinations_with_replacement
from itertools import chain


def _get_max_lag(ylag, xlag):
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


class GenerateRegressors:
    """Polynomial NARMAX model

    Provides the main functions to generate the regressor dictionary
    and regressor codes for polynomial basis.
    """

    def regressor_space(self, non_degree, xlag, ylag, n_inputs):
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
            Técnicas de otimizaçao bi-objetivo para a determinaçao
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
            # the user must entered list of the desired lags explicity
            x_vec_tmp = []
            for i in range(n_inputs):
                if isinstance(xlag[i], list) and n_inputs > 1:
                    # create 200n, 300n,..., 400n to describe each input
                    x_vec_tmp.extend([lag + 2000 + i * 1000 for lag in xlag[i]])
                elif isinstance(xlag[i], int) and n_inputs > 1:
                    x_vec_tmp.extend(
                        [np.arange(2001 + i * 1000, 2001 + i * 1000 + xlag[i])]
                    )

        reg_aux = np.array([0])
        if n_inputs > 1:
            # if x_vec is a nested list, ensure all elements are arrays
            all_arrays = [np.array([i]) if isinstance(i, int) else i for i in x_vec_tmp]
            x_vec = np.concatenate([i for i in all_arrays])
        else:
            x_vec = x_vec_tmp

        reg_aux = np.concatenate([reg_aux, y_vec, x_vec])

        regressor_code = list(combinations_with_replacement(reg_aux, non_degree))

        regressor_code = np.array(regressor_code)
        regressor_code = regressor_code[:, regressor_code.shape[1] :: -1]
        max_lag = _get_max_lag(ylag, xlag)
        return regressor_code, max_lag


class HouseHolder:
    """Householder reflection and transformation."""

    def _house(self, x):
        """Perform a Househoulder reflection of vector.

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
        n = len(x)
        u = np.linalg.norm(x, 2)
        v = np.array(x)
        aux_a = np.array([1])
        if u != 0:
            aux_b = x[0] + np.sign(x[0]) * u
            v = np.array(v[1:n] / aux_b)
            v = np.concatenate((aux_a, v))
        return v

    def _rowhouse(self, RA, v):
        """Perform a row Househoulder transformation.

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
        b = -2 / (v.T @ v)
        w = b * RA.T @ v
        w = w.reshape(1, -1)
        v = v.reshape(-1, 1)
        RA = RA + v * w
        B = RA
        return B


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
            # Remember, for multiple inputs all lags must be entered explicity
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

    def build_information_matrix(self, X, y, xlag, ylag, non_degree):
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
        max_lag = _get_max_lag(ylag, xlag)

        # Generate a lagged data which each column is a input or output
        # related to its respective lags. With this approach we can create
        # the information matrix by using all possible combination of
        # the columns as a product in the iterations
        lagged_data = self.initial_lagged_matrix(X, y, xlag=xlag, ylag=ylag)

        constant = np.ones([lagged_data.shape[0], 1])
        data = np.concatenate([constant, lagged_data], axis=1)

        # Create combinations of all columns based on its index
        iterable_list = range(data.shape[1])
        combinations = list(combinations_with_replacement(iterable_list, non_degree))

        psi = np.column_stack(
            [
                np.prod(data[:, combinations[i]], axis=1)
                for i in range(len(combinations))
            ]
        )
        psi = psi[max_lag:, :]
        return psi
