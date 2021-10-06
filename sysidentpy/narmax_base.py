"""Base classes for NARMAX estimator."""

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause


import numpy as np
from itertools import combinations_with_replacement
from itertools import chain
from collections import Counter



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