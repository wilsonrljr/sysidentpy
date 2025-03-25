"""Affine Information Least Squares for NARMAX models."""

from typing import Tuple, List

import numpy as np

from sysidentpy.basis_function import Polynomial
from sysidentpy.utils.information_matrix import build_input_output_matrix
from sysidentpy.narmax_base import RegressorDictionary
from sysidentpy.utils.simulation import (
    get_index_from_regressor_code,
    list_output_regressor_code,
    list_input_regressor_code,
)

from sysidentpy.utils.lags import (
    get_lag_from_regressor_code,
    get_max_lag_from_model_code,
)


def get_term_clustering(qit: np.ndarray) -> np.ndarray:
    """Get the term clustering of the model.

    This function takes a matrix `qit` and compute the term clustering based
    on their values. It calculates the number of occurrences of each value
    for each row in the matrix.

    Parameters
    ----------
    qit : ndarray
        Input matrix containing terms clustering to be sorted.

    Returns
    -------
    N_aux : ndarray
        A new matrix with rows representing the number of occurrences of each value
        for each row in the input matrix `qit`. The columns correspond to different
        values.

    Examples
    --------
    >>> qit = np.array([[1, 2, 2],
    ...                 [1, 3, 1],
    ...                 [2, 2, 3]])
    >>> result = get_term_clustering(qit)
    >>> print(result)
    [[1. 2. 0. 0.]
    [2. 0. 1. 0.]
    [0. 2. 1. 0.]]

    Notes
    -----
    The function calculates the number of occurrences of each value (from 1 to
    the maximum value in the input matrix `qit`) for each row and returns a matrix
    where rows represent rows of the input matrix `qit`, and columns represent
    different values.

    """
    max_value = int(np.max(qit))
    counts_matrix = np.zeros((qit.shape[0], max_value))

    for k in range(1, max_value + 1):
        counts_matrix[:, k - 1] = np.sum(qit == k, axis=1)

    return counts_matrix.astype(int)


def get_cost_function(y: np.ndarray, psi: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Calculate the cost function based on residuals.

    Parameters
    ----------
    y : ndarray of floats
        The target data used in the identification process.
    psi : ndarray of floats, shape (n_samples, n_parameters)
        The matrix of regressors.
    theta : ndarray of floats
        The parameter vector.

    Returns
    -------
    cost_function : float
        The calculated cost function value.

    Notes
    -----
    This method computes the cost function value based on the residuals between
    the target data (y) and the predicted values using the regressors (dynamic
    and static) and parameter vector (theta). It quantifies the error in the
    model's predictions.

    """
    residuals = y - psi.dot(theta)
    return residuals.T.dot(residuals)


class AILS:
    """Affine Information Least Squares (AILS) for NARMAX Parameter Estimation.

    AILS is a non-iterative multiobjective Least Squares technique used for finding
    Pareto-set solutions in NARMAX (Nonlinear AutoRegressive Moving Average with
    eXogenous inputs) model parameter estimation. This method is suitable for
    linear-in-the-parameter model structures.

    Two types of auxiliary information can be incorporated: static function and
    steady-state gain.

    Parameters
    ----------
    static_gain : bool, default=True
        Flag indicating the presence of data related to steady-state gain.
    static_function : bool, default=True
        Flag indicating the presence of data concerning static function.
    final_model : ndarray, default=[[0], [0]]
        Model code representation.

    References
    ----------
    1. Nepomuceno, E. G., Takahashi, R. H. C., & Aguirre, L. A. (2007).
    "Multiobjective parameter estimation for nonlinear systems: Affine information and
    least-squares formulation."
    International Journal of Control, 80, 863-871.
    """

    def __init__(
        self,
        static_gain: bool = True,
        static_function: bool = True,
        final_model: np.ndarray = np.zeros((1, 1)),
        normalize: bool = True,
    ):
        self.n_inputs = np.max(final_model // 1000) - 1
        self.degree = np.shape(final_model)[1]
        self.max_lag = 1
        self.final_model = final_model
        self.static_gain = static_gain
        self.static_function = static_function
        self.normalize = normalize

    def build_linear_mapping(self):
        """Assemble the linear mapping matrix R using the regressor-space method.

        This function constructs the linear mapping matrix R, which plays a key role in
        mapping the parameter vector to the cluster coefficients. It also generates a
        row matrix qit that assists in locating terms within the linear mapping matrix.
        This qit matrix is later used in creating the static regressor matrix (Q).

        Returns
        -------
        R : ndarray of int
            A constant matrix of ones and zeros that maps the parameter vector to
            cluster coefficients.
        qit : ndarray of int
            A row matrix that helps locate terms within the linear mapping matrix R and
            is used in the creation of the static regressor matrix (Q).

        Notes
        -----
        The linear mapping matrix R is constructed using the regressor-space method.
        It plays a crucial role in the parameter estimation process, facilitating the
        mapping of parameter values to cluster coefficients. The qit matrix aids in
        term localization within the linear mapping matrix R and is subsequently used
        to build the static regressor matrix (Q).

        """
        xlag = [1] * self.n_inputs

        object_qit = RegressorDictionary(xlag=xlag, ylag=[1])
        # Given xlag and ylag equal to 1, there is no repetition of terms, which is
        # ideal for building qit.
        qit = object_qit.regressor_space(n_inputs=self.n_inputs) // 1000
        model = self.final_model // 1000
        R = np.all(qit[:, None, :] == model, axis=2).astype(int)
        # Find rows with all zeros in R (sum of row elements is 0)
        null_rows = list(np.where(np.sum(R, axis=1) == 0)[0])

        R = np.delete(R, null_rows, axis=0)
        qit = np.delete(qit, null_rows, axis=0)
        return R, get_term_clustering(qit)

    def build_static_function_information(
        self, x_static: np.ndarray, y_static: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construct a matrix of static regressors for a NARMAX model.

        Parameters
        ----------
        y_static : array-like, shape (n_samples_static_function,)
            Output of the static function.
        x_static : array-like, shape (n_samples_static_function,)
            Static function input.

        Returns
        -------
        Q_dot_R : ndarray of floats, shape (n_samples_static_function, n_parameters)
            The result of multiplying the matrix of static regressors (Q) with the
            linear mapping matrix (R), where n_parameters is the number of model
            parameters.
        static_covariance: ndarray of floats, shape (n_parameters, n_parameters)
            The covariance QR'QR
        static_response: ndarray of floats, shape (n_parameters,)
            The response QR'y

        Notes
        -----
        This function constructs a matrix of static regressors (Q) based on the provided
        static function outputs (y_static) and inputs (X_static). The linear mapping
        matrix (R) should be precomputed before calling this function. The result
        Q_dot_R represents the static regressors for the NARMAX model.

        """
        R, qit = self.build_linear_mapping()
        Q = y_static ** qit[:, 0]
        for k in range(self.n_inputs):
            Q *= x_static ** qit[:, 1 + k]

        Q = Q.reshape(len(y_static), len(qit))

        QR = Q.dot(R)
        static_covariance = QR.T.dot(QR)
        static_response = QR.T.dot(y_static)
        return QR, static_covariance, static_response

    def build_static_gain_information(
        self, x_static: np.ndarray, y_static: np.ndarray, gain: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construct a matrix of static regressors referring to the derivative (gain).

        Parameters
        ----------
        y_static : array-like, shape (n_samples_static_function,)
            Output of the static function.
        x_static : array-like, shape (n_samples_static_function,)
            Static function input.
        gain : array-like, shape (n_samples_static_gain,)
            Static gain input.

        Returns
        -------
        HR : ndarray of floats, shape (n_samples_static_function, n_parameters)
            The matrix of static regressors for the derivative (gain) multiplied by the
            linear mapping matrix R.
        gain_covariance : ndarray of floats, shape (n_parameters, n_parameters)
            The covariance matrix (HR'HR) for the gain-related regressors.
        gain_response : ndarray of floats, shape (n_parameters,)
            The response vector (HR'y) for the gain-related regressors.

        Notes
        -----
        This function constructs a matrix of static regressors (G+H) for the derivative
        (gain) based on the provided static function outputs (y_static), inputs
        (X_static), and gain values. The linear mapping matrix (R) should be
        precomputed before calling this function.

        """
        R, qit = self.build_linear_mapping()
        H = np.zeros((len(y_static), len(qit)))
        G = np.zeros((len(y_static), len(qit)))
        for i in range(len(y_static)):
            for j in range(1, len(qit)):
                if y_static[i, 0] == 0:
                    if (qit[j, 0]) == 1:
                        H[i, j] = gain[i][0]
                    else:
                        H[i, j] = 0
                else:
                    H[i, j] = (gain[i] * qit[j, 0] * y_static[i, 0] ** (qit[j, 0] - 1))[
                        0
                    ]
                for k in range(self.n_inputs):
                    if x_static[i, k] == 0:
                        if (qit[j, 1 + k]) == 1:
                            G[i, j] = 1
                        else:
                            G[i, j] = 0
                    else:
                        G[i, j] = qit[j, 1 + k] * x_static[i, k] ** (qit[j, 1 + k] - 1)

        HR = (G + H).dot(R)
        gain_covariance = HR.T.dot(HR)
        gain_response = HR.T.dot(gain)
        return HR, gain_covariance, gain_response

    def build_system_data(
        self,
        y: np.ndarray,
        static_gain: np.ndarray,
        static_function: np.ndarray,
    ) -> List[np.ndarray]:
        """Construct a list of output data components for the NARMAX system.

        Parameters
        ----------
        y : ndarray of floats
            The target data used in the identification process.
        static_gain : ndarray of floats
            Static gain output data.
        static_function : ndarray of floats
            Static function output data.

        Returns
        -------
        system_data : list of ndarrays
            A list containing data components, including the target data (y),
            static gain data (if present), and static function data (if present).

        Notes
        -----
        This method constructs a list of data components that are used in the NARMAX
        system identification process. The components may include the target data (y),
        static gain data (if enabled), and static function data (if enabled).

        """
        if not self.static_gain:
            return [y] + [static_function]

        if not self.static_function:
            return [y] + [static_gain]

        return [y] + [static_gain] + [static_function]

    def build_affine_data(
        self, psi: np.ndarray, HR: np.ndarray, QR: np.ndarray
    ) -> List[np.ndarray]:
        """Construct a list of affine data components for NARMAX modeling.

        Parameters
        ----------
        psi : ndarray of floats, shape (n_samples, n_parameters)
            The matrix of dynamic regressors.
        HR : ndarray of floats, shape (n_samples_static_gain, n_parameters)
            The matrix of static gain regressors.
        QR : ndarray of floats, shape (n_samples_static_function, n_parameters)
            The matrix of static function regressors.

        Returns
        -------
        affine_data : list of ndarrays
            A list containing affine data components, including the matrix of static
            regressors (psi), static gain regressors (if present), and static function
            regressors (if present).

        Notes
        -----
        This method constructs a list of affine data components used in the NARMAX
        modeling process. The components may include the matrix of static regressors
        (psi), static gain regressors (if enabled), and static function regressors
        (if enabled).

        """
        if not self.static_gain:
            return [psi] + [QR]

        if not self.static_function:
            return [psi] + [HR]

        return [psi] + [HR] + [QR]

    def build_psi(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Build the matrix of dynamic regressor for NARMAX modeling.

        Parameters
        ----------
        X : ndarray of floats
            The input data to be used in the training process.
        y : ndarray of floats
            The output data to be used in the training process.

        Returns
        -------
        psi : ndarray of floats, shape (n_samples, n_parameters)
            The matrix of dynamic regressors.

        """
        psi_builder = RegressorDictionary()
        xlag_code = list_input_regressor_code(self.final_model)
        ylag_code = list_output_regressor_code(self.final_model)
        xlag = get_lag_from_regressor_code(xlag_code)
        ylag = get_lag_from_regressor_code(ylag_code)
        self.max_lag = get_max_lag_from_model_code(self.final_model)
        if self.n_inputs != 1:
            xlag = self.n_inputs * [list(range(1, self.max_lag + 1))]

        psi_builder.xlag = xlag
        psi_builder.ylag = ylag
        regressor_code = psi_builder.regressor_space(self.n_inputs)
        pivv = get_index_from_regressor_code(regressor_code, self.final_model)
        self.final_model = regressor_code[pivv]

        lagged_data = build_input_output_matrix(x=X, y=y, xlag=xlag, ylag=ylag)

        psi = Polynomial(degree=self.degree).fit(
            lagged_data,
            max_lag=self.max_lag,
            ylag=ylag,
            xlag=xlag,
            predefined_regressors=pivv,
        )
        return psi

    def estimate(
        self,
        y_static: np.ndarray = np.zeros(1),
        X_static: np.ndarray = np.zeros(1),
        gain: np.ndarray = np.zeros(1),
        y: np.ndarray = np.zeros(1),
        X: np.ndarray = np.zeros((1, 1)),
        weighing_matrix: np.ndarray = np.zeros((1, 1)),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.int64]:
        """Estimate the parameters via multi-objective techniques.

        Parameters
        ----------
        y_static : array-like of shape = n_samples_static_function, default = ([0])
            Output of static function.
        X_static : array-like of shape = n_samples_static_function, default = ([0])
            Static function input.
        gain : array-like of shape = n_samples_static_gain, default = ([0])
            Static gain input.
        y : array-like of shape = n_samples, default = ([0])
            The target data used in the identification process.
        X : ndarray of floats, default = ([[0],[0]])
            Matrix of static regressors.
        weighing_matrix: ndarray
            Weighing matrix for defining the weight of each objective.

        Returns
        -------
        J : ndarray
            Matrix referring to the objectives.
        euclidean_norm : ndarray
            Matrix of the Euclidean norm.
        theta : ndarray
            Matrix with parameters for each weight.
        HR : ndarray
            H matrix multiplied by R.
        QR : ndarray
            Q matrix multiplied by R.
        position : ndarray, default = ([[0],[0]])
            Position of the best theta set.

        """
        psi = self.build_psi(X, y)
        y = y[self.max_lag :]
        HR, QR = np.zeros((1, 1)), np.zeros((1, 1))
        n_parameters = weighing_matrix.shape[1]
        num_objectives = self.static_function + self.static_gain + 1
        euclidean_norm = np.zeros(n_parameters)
        theta = np.zeros((n_parameters, self.final_model.shape[0]))
        dynamic_covariance = psi.T.dot(psi)
        dynamic_response = psi.T.dot(y)

        if self.static_function:
            QR, static_covariance, static_response = (
                self.build_static_function_information(X_static, y_static)
            )
        if self.static_gain:
            HR, gain_covariance, gain_response = self.build_static_gain_information(
                X_static, y_static, gain
            )
        J = np.zeros((num_objectives, n_parameters))
        system_data = self.build_system_data(y, gain, y_static)
        affine_information_data = self.build_affine_data(psi, HR, QR)
        for i in range(n_parameters):
            theta1 = weighing_matrix[0, i] * dynamic_covariance
            theta2 = weighing_matrix[0, i] * dynamic_response

            w = 1
            if self.static_function:
                theta1 += weighing_matrix[w, i] * static_covariance
                theta2 += weighing_matrix[w, i] * static_response.reshape(-1, 1)
                w += 1

            if self.static_gain:
                theta1 += weighing_matrix[w, i] * gain_covariance
                theta2 += weighing_matrix[w, i] * gain_response.reshape(-1, 1)
                w += 1

            tmp_theta = np.linalg.lstsq(theta1, theta2, rcond=None)[0]
            theta[i, :] = tmp_theta.T

            for j in range(num_objectives):
                residuals = get_cost_function(
                    system_data[j], affine_information_data[j], tmp_theta
                )
                J[j, i] = residuals[0]

            euclidean_norm[i] = np.linalg.norm(J[:, i])

        if self.normalize is True:
            J /= np.max(J, axis=1)[:, np.newaxis]
            euclidean_norm /= np.max(euclidean_norm)

            euclidean_norm = euclidean_norm / np.max(euclidean_norm)

        position = np.argmin(euclidean_norm)
        return (
            J,
            euclidean_norm,
            theta,
            HR,
            QR,
            position,
        )
