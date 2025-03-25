"""Methods for parameter estimation."""

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause


import warnings

from typing import Union, Optional

import numpy as np

from scipy.optimize import nnls
from scipy.optimize import lsq_linear
from scipy.sparse.linalg import lsmr

from .estimators_base import BaseEstimator
from .estimators_base import _validate_params, _initial_values
from ..utils.deprecation import deprecated
from ..utils.check_arrays import check_linear_dependence_rows


class EstimatorError(Exception):
    """Generic Python-exception-derived object raised by estimator functions.

    General purpose exception class, derived from Python's ValueError
    class, programmatically raised in estimators functions when a Estimator-related
    condition would prevent further correct execution of the function.

    Parameters
    ----------
    None

    """


class LeastSquares(BaseEstimator):
    """Ordinary Least Squares for linear parameter estimation.

    The Least Squares method minimizes the sum of the squared differences
    between the observed and predicted values. It is used to estimate the
    parameters of a linear model.

    Parameters
    ----------
    unbiased : bool, optional
        If True, applies an unbiased estimator. Default is False.
    uiter : int, optional
        Number of iterations for the unbiased estimator. Default is 20.

    References
    ----------
    - Sorenson, H. W. (1970). Least-squares estimation: from Gauss to Kalman.
      IEEE spectrum, 7(7), 63-68.
      http://pzs.dstu.dp.ua/DataMining/mls/bibl/Gauss2Kalman.pdf
    - Aguirre, L. A. (2007). Introdução identificação de sistemas: técnicas
      lineares e não-lineares aplicadas a sistemas reais. Editora da UFMG. 3a edição.
    - Markovsky, I., & Van Huffel, S. (2007). Overview of total least-squares methods.
      Signal processing, 87(10), 2283-2302.
      https://eprints.soton.ac.uk/263855/1/tls_overview.pdf
    - Wikipedia entry on Least Squares
      https://en.wikipedia.org/wiki/Least_squares
    """

    def __init__(self, *, unbiased: bool = False, uiter: int = 20):
        self.unbiased = unbiased
        self.uiter = uiter
        _validate_params(vars(self))

    def optimize(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
        r"""Estimate the model parameters using the Least Squares method.

        The Least Squares method solves the following optimization problem:

        $$
        \min_{\theta} \| \psi \theta - y \|_2^2
        $$

        where $\psi$ is the information matrix, $y$ is the observed data,
        and $\theta$ are the model parameters to be estimated.

        Parameters
        ----------
        psi : ndarray of floats
            The information matrix of the model.
        y : array-like of shape (n_samples, 1)
            The data used to train the model.

        Returns
        -------
        theta : array-like of shape (n_features, 1)
            The estimated parameters of the model.
        """
        check_linear_dependence_rows(psi)
        theta = np.linalg.lstsq(psi, y, rcond=None)[0]
        return theta


@deprecated(
    version="v0.6.0",
    future_version="v0.7.0",
    message=(
        " `solver` is deprecated in v0.5.4 and will be removed in v0.7.0."
        " A single solver option will be retained moving forward."
    ),
)
class RidgeRegression(BaseEstimator):
    """Ridge Regression estimator using classic and SVD methods.

    This class implements Ridge Regression, a type of linear regression that includes
    an L2 penalty to prevent overfitting. The implementation offers two methods for
    parameter estimation: a classic approach and an approach based on Singular Value
    Decomposition (SVD).

    Parameters
    ----------
    alpha : np.float64, optional (default=np.finfo(np.float64).eps)
        Regularization strength; must be a positive float. Regularization improves the
        conditioning of the problem and reduces the variance of the estimates. Larger
        values specify stronger regularization. If the input is a noisy signal,
        the ridge parameter is likely to be set close to the noise level, at least as
        a starting point. Entered through the self data structure.
    solver : str, optional (default="svd")
        Solver to use in the parameter estimation procedure.

    Methods
    -------
    ridge_regression_classic(psi, y)
        Estimate the model parameters using the classic ridge regression method.
    ridge_regression(psi, y)
        Estimate the model parameters using the SVD-based ridge regression method.
    optimize(psi, y)
        Optimize the model parameters using the chosen method (SVD or classic).

    References
    ----------
    - Wikipedia entry on ridge regression
      https://en.wikipedia.org/wiki/Ridge_regression
    - D. J. Gauthier, E. Bollt, A. Griffith, W. A. S. Barbosa, 'Next generation
      reservoir computing,' Nat. Commun. 12, 5564 (2021).
      https://www.nature.com/articles/s41467-021-25801-2
    - Hoerl, A. E.; Kennard, R. W. Ridge regression: applications to nonorthogonal
      problems. Technometrics, Taylor & Francis, v. 12, n. 1, p. 69-82, 1970.
    - StackExchange: whuber. The proof of shrinking coefficients using ridge regression
      through "spectral decomposition".
      Cross Validated, accessed 21 September 2023,
      https://stats.stackexchange.com/q/220324
    """

    def __init__(
        self,
        *,
        alpha: np.float64 = np.finfo(np.float64).eps,
        solver: str = "svd",
        unbiased: bool = False,
        uiter: int = 30,
    ):
        self.alpha = alpha
        self.solver = solver
        self.uiter = uiter
        self.unbiased = unbiased
        _validate_params(vars(self))

    def ridge_regression_classic(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Estimate the model parameters using ridge regression.

           Based on the least_squares module and uses the same data format but you need
           to pass alpha in the call to FROLS.

        Parameters
        ----------
        psi : ndarray of floats
            The information matrix of the model.
        y : array-like of shape = y_training
            The data used to training the model.

        Returns
        -------
        theta : array-like of shape = number_of_model_elements
            The estimated parameters of the model.

        References
        ----------
        - Wikipedia entry on ridge regression
          https://en.wikipedia.org/wiki/Ridge_regression

        alpha multiplied by the identity matrix (np.eye) favors models (theta) that
        have small size using an L2 norm.  This prevents over fitting of the model.
        For applications where preventing overfitting is important, see, for example,
        D. J. Gauthier, E. Bollt, A. Griffith, W. A. S. Barbosa, 'Next generation
        reservoir computing,' Nat. Commun. 12, 5564 (2021).
        https://www.nature.com/articles/s41467-021-25801-2

        """
        check_linear_dependence_rows(psi)

        theta = (
            np.linalg.pinv(psi.T @ psi + self.alpha * np.eye(psi.shape[1])) @ psi.T @ y
        )
        return theta

    def ridge_regression(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Estimate the model parameters using SVD and Ridge Regression method.

        Parameters
        ----------
        psi : ndarray of floats
            The information matrix of the model.
        y : array-like of shape = y_training
            The data used to training the model.

        Returns
        -------
        theta : array-like of shape = number_of_model_elements
            The estimated parameters of the model.

        References
        ----------
        - Manuscript: Hoerl, A. E.; Kennard, R. W. Ridge regression:
                      applications to nonorthogonal problems. Technometrics,
                      Taylor & Francis, v. 12, n. 1, p. 69-82, 1970.

        - StackExchange: whuber. The proof of shrinking coefficients using ridge
                         regression through "spectral decomposition".
                         Cross Validated, accessed 21 September 2023,
                         https://stats.stackexchange.com/q/220324
        """
        check_linear_dependence_rows(psi)
        try:
            U, S, Vh = np.linalg.svd(psi, full_matrices=False)
            S = np.diag(S)
            i = np.identity(len(S))
            theta = Vh.T @ np.linalg.inv(S**2 + self.alpha * i) @ S @ U.T @ y
        except EstimatorError:
            warnings.warn(
                "The SVD computation did not converge."
                "Theta values will be calculated with the classic algorithm.",
                stacklevel=2,
            )

            theta = self.ridge_regression_classic(psi, y)

        return theta

    def optimize(self, psi: np.ndarray, y):
        if self.solver == "svd":
            return self.ridge_regression(psi, y)

        return self.ridge_regression_classic(psi, y)


class TotalLeastSquares(BaseEstimator):
    """Estimate the model parameters using the Total Least Squares (TLS) method.

    The Total Least Squares method is used to solve the problem of fitting a model
    to data when both the independent variables (psi) and the dependent variable (y)
    are subject to errors. This method minimizes the orthogonal distances from the
    data points to the fitted model, which is more appropriate when errors are present
    in all variables.

    Parameters
    ----------
    unbiased : bool, optional
        If True, applies an unbiased estimator. Default is False.
    uiter : int, optional
        Number of iterations for the unbiased estimator. Default is 30.

    References
    ----------
    - Golub, G. H., & Van Loan, C. F. (1980). An analysis of the total least squares
    problem.
      SIAM journal on numerical analysis, 17(6), 883-893.
    - Markovsky, I., & Van Huffel, S. (2007). Overview of total least-squares methods.
      Signal processing, 87(10), 2283-2302. https://eprints.soton.ac.uk/263855/1/tls_overview.pdf
    - Wikipedia entry on Total Least Squares: https://en.wikipedia.org/wiki/Total_least_squares
    """

    def __init__(self, *, unbiased: bool = False, uiter: int = 30):
        self.unbiased = unbiased
        self.uiter = uiter
        _validate_params(vars(self))

    def optimize(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
        r"""Estimate the model parameters using the Total Least Squares method.

        The TLS method solves the following problem:

        $$
            \min_{E, f} \| [E, f] \|_F \quad \text{subject to}
            \quad (psi + E) \theta = y + f
        $$

        where $E$ and $f$ are the error matrices for $psi$ and $y$ respectively,
        and $\| \cdot \|_F$ denotes the Frobenius norm.

        Parameters
        ----------
        psi : ndarray of floats
            The information matrix of the model.
        y : array-like of shape (n_samples, 1)
            The data used to train the model.

        Returns
        -------
        theta : array-like of shape (n_features, 1)
            The estimated parameters of the model.
        """
        check_linear_dependence_rows(psi)
        full = np.hstack((psi, y))
        n = psi.shape[1]
        _, _, v = np.linalg.svd(full, full_matrices=True)
        theta = -v.T[:n, n:] / v.T[n:, n:]
        return theta.reshape(-1, 1)


class RecursiveLeastSquares(BaseEstimator):
    """Recursive Least Squares (RLS) filter for parameter estimation.

    The Recursive Least Squares method is used to estimate the parameters of a model
    by minimizing the sum of the squares of the differences between the observed and
    predicted values. This method incorporates a forgetting factor to give more weight
    to recent observations.

    Parameters
    ----------
    lam : float, default=0.98
        Forgetting factor of the Recursive Least Squares method.
    delta : float, default=0.01
        Normalization factor of the P matrix.
    unbiased : bool, optional
        If True, applies an unbiased estimator. Default is False.
    uiter : int, optional
        Number of iterations for the unbiased estimator. Default is 30.

    Attributes
    ----------
    lam : float
        Forgetting factor of the Recursive Least Squares method.
    delta : float
        Normalization factor of the P matrix.
    xi : np.ndarray
        The estimation error at each iteration.
    theta_evolution : np.ndarray
        Evolution of the estimated parameters over iterations.

    Methods
    -------
    optimize(psi: np.ndarray, y: np.ndarray) -> np.ndarray
        Estimate the model parameters using the Recursive Least Squares method.

    References
    ----------
    - Book (Portuguese): Aguirre, L. A. (2007). Introdução identificação
       de sistemas: técnicas lineares e não-lineares aplicadas a sistemas
       reais. Editora da UFMG. 3a edição.
    """

    def __init__(
        self,
        *,
        delta: float = 0.01,
        lam: float = 0.98,
        unbiased: bool = False,
        uiter: int = 30,
    ):
        self.delta = delta
        self.lam = lam
        self.unbiased = unbiased
        self.uiter = uiter
        _validate_params(vars(self))
        self.xi: Optional[np.ndarray] = None
        self.theta_evolution: Optional[np.ndarray] = None

    def optimize(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
        r"""Estimate the model parameters using the Recursive Least Squares method.

        The implementation considers the forgetting factor.

        Parameters
        ----------
        psi : ndarray of floats
            The information matrix of the model.
        y : array-like of shape = y_training
            The data used to train the model.

        Returns
        -------
        theta : array-like of shape = number_of_model_elements
            The estimated parameters of the model.

        Notes
        -----
        The RLS algorithm updates the parameter estimates recursively as follows:

        1. Initialize the parameter vector `theta` and the covariance matrix `P`:

           $$
           \\theta_0 = \\mathbf{0}, \\quad P_0 = \\frac{1}{\\delta} I
           $$

        2. For each new observation `(psi_i, y_i)`, update the estimates:

           $$
           k_i = \\frac{\\lambda^{-1} P_{i-1} \\psi_i}{1 +
           \\lambda^{-1} \\psi_i^T P_{i-1} \\psi_i}
           $$

           $$
           \\theta_i = \\theta_{i-1} + k_i (y_i - \\psi_i^T \\theta_{i-1})
           $$

           $$
           P_i = \\lambda^{-1} (P_{i-1} - k_i \\psi_i^T P_{i-1})
           $$

        References
        ----------
        - Book (Portuguese): Aguirre, L. A. (2007). Introdução identificação
           de sistemas: técnicas lineares e não-lineares aplicadas a sistemas
           reais. Editora da UFMG. 3a edição.
        """
        n_theta, n, theta, self.xi = _initial_values(psi)
        p = np.eye(n_theta) / self.delta

        for i in range(2, n):
            psi_tmp = psi[i, :].reshape(-1, 1)
            k_numerator = self.lam ** (-1) * p.dot(psi_tmp)
            k_denominator = 1 + self.lam ** (-1) * psi_tmp.T.dot(p).dot(psi_tmp)
            k = np.divide(k_numerator, k_denominator)
            self.xi[i, 0] = y[i, 0] - np.dot(psi_tmp.T, theta[:, i - 1])[0]
            tmp_list = theta[:, i - 1].reshape(-1, 1) + k.dot(self.xi[i, 0])
            theta[:, i] = tmp_list.flatten()

            p1 = p.dot(psi[i, :].reshape(-1, 1)).dot(psi[i, :].reshape(-1, 1).T).dot(p)
            p2 = (
                psi[i, :].reshape(-1, 1).T.dot(p).dot(psi[i, :].reshape(-1, 1))
                + self.lam
            )

            p_numerator = p - np.divide(p1, p2)
            p = np.divide(p_numerator, self.lam)

        self.theta_evolution = theta.copy()
        return theta[:, -1].reshape(-1, 1)


class AffineLeastMeanSquares(BaseEstimator):
    """Affine Least Mean Squares (ALMS) filter for parameter estimation.

    The ALMS filter is an adaptive filter used to estimate the parameters of a model.
    It incorporates an offset covariance factor to improve the stability and convergence
    of the parameter estimation process.

    Parameters
    ----------
    mu : float, default=0.01
        The learning rate or step size for the LMS algorithm.
    offset_covariance : float, default=0.2
        The offset covariance factor of the affine least mean squares filter.
    unbiased : bool, optional
        If True, applies an unbiased estimator. Default is False.
    uiter : int, optional
        Number of iterations for the unbiased estimator. Default is 30.

    Attributes
    ----------
    mu : float
        The learning rate or step size for the LMS algorithm.
    offset_covariance : float
        The offset covariance factor of the affine least mean squares filter.
    xi : np.ndarray or None
        The estimation error at each iteration. Initialized as None and updated during
        optimization.

    Methods
    -------
    optimize(psi: np.ndarray, y: np.ndarray) -> np.ndarray
        Estimate the model parameters using the ALMS filter.

    References
    ----------
    - Poularikas, A. D. (2017). Adaptive filtering: Fundamentals of least mean squares
    with MATLAB®. CRC Press.
    """

    def __init__(
        self,
        *,
        mu: float = 0.01,
        offset_covariance: float = 0.2,
        unbiased: bool = False,
        uiter: int = 30,
    ):
        self.mu = mu
        self.offset_covariance = offset_covariance
        self.uiter = uiter
        self.unbiased = unbiased
        _validate_params(vars(self))
        self.xi: Optional[np.ndarray] = None

    def optimize(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
        r"""Estimate the model parameters using the Affine Least Mean Squares.

        The ALMS method updates the parameter estimates recursively as follows:

        1. Compute the estimation error:

           $$
           \xi = y - \psi \theta_{i-1}
           $$

        2. Update the parameter vector:

           $$
           \theta_i = \theta_{i-1} + \mu \psi (\psi^T \psi + \text{offset_covariance}
           \cdot I)^{-1} \xi
           $$

        Parameters
        ----------
        psi : ndarray of floats
            The information matrix of the model.
        y : array-like of shape (n_samples, 1)
            The data used to train the model.

        Returns
        -------
        theta : array-like of shape (n_features, 1)
            The estimated parameters of the model.

        Notes
        -----
        A more in-depth documentation of all methods for parameters estimation
        will be available soon. For now, please refer to the mentioned references.
        """
        n_theta, n, theta, self.xi = _initial_values(psi)

        for i in range(n_theta, n):
            self.xi = y - psi.dot(theta[:, i - 1].reshape(-1, 1))
            aux = (
                self.mu
                * psi
                @ np.linalg.pinv(psi.T @ psi + self.offset_covariance * np.eye(n_theta))
            )
            tmp_list = theta[:, i - 1].reshape(-1, 1) + aux.T.dot(self.xi)
            theta[:, i] = tmp_list.flatten()

        return theta[:, -1].reshape(-1, 1)


class LeastMeanSquares(BaseEstimator):
    """Least Mean Squares (LMS) filter for parameter estimation in adaptive filtering.

    The LMS algorithm is an adaptive filter used to estimate the parameters of a model
    by minimizing the mean square error between the observed and predicted values.

    Parameters
    ----------
    mu : float, default=0.01
        The learning rate or step size for the LMS algorithm.
    unbiased : bool, optional
        If True, applies an unbiased estimator. Default is False.
    uiter : int, optional
        Number of iterations for the unbiased estimator. Default is 30.

    Attributes
    ----------
    mu : float
        The learning rate or step size for the LMS algorithm.
    unbiased : bool
        Indicates whether an unbiased estimator is applied.
    uiter : int
        Number of iterations for the unbiased estimator.
    xi : np.ndarray or None
        The estimation error at each iteration. Initialized as None and updated during
        optimization.

    Methods
    -------
    optimize(psi: np.ndarray, y: np.ndarray) -> np.ndarray
        Estimate the model parameters using the LMS filter.

    References
    ----------
    - Haykin, S., & Widrow, B. (Eds.). (2003). Least-mean-square adaptive filters
    (Vol. 31). John Wiley & Sons.
    - Zipf, J. G. F. (2011). Classificação, análise estatística e novas estratégias de
    algoritmos LMS de passo variável.
    - Wikipedia entry on Least Mean Squares: https://en.wikipedia.org/wiki/Least_mean_squares_filter
    """

    def __init__(self, *, mu: float = 0.01, unbiased: bool = False, uiter: int = 30):
        self.mu = mu
        self.unbiased = unbiased
        self.uiter = uiter
        _validate_params(vars(self))
        self.xi: Optional[np.ndarray] = None

    def optimize(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
        r"""Estimate the model parameters using the Least Mean Squares filter.

        The LMS algorithm updates the parameter estimates recursively as follows:

        1. Compute the estimation error:

           $$
           \xi_i = y_i - \psi_i^T \theta_{i-1}
           $$

        2. Update the parameter vector:

           $$
           \theta_i = \theta_{i-1} + 2 \mu \xi_i \psi_i
           $$

        Parameters
        ----------
        psi : ndarray of floats
            The information matrix of the model.
        y : array-like of shape (n_samples, 1)
            The data used to train the model.

        Returns
        -------
        theta : array-like of shape (n_features, 1)
            The estimated parameters of the model.
        """
        n_theta, n, theta, self.xi = _initial_values(psi)

        for i in range(n_theta, n):
            psi_tmp = psi[i, :].reshape(-1, 1)
            self.xi[i, 0] = y[i, 0] - np.dot(psi_tmp.T, theta[:, i - 1])[0]
            tmp_list = (
                theta[:, i - 1].reshape(-1, 1) + 2 * self.mu * self.xi[i, 0] * psi_tmp
            )
            theta[:, i] = tmp_list.flatten()

        return theta[:, -1].reshape(-1, 1)


class LeastMeanSquaresSignError(BaseEstimator):
    """Least Mean Squares (LMS) filter for parameter estimation using sign-error.

    The sign-error LMS algorithm uses the sign of the error vector to update the filter
    coefficients.

    Parameters
    ----------
    mu : float, default=0.01
        The learning rate or step size for the LMS algorithm.
    unbiased : bool, optional
        If True, applies an unbiased estimator. Default is False.
    uiter : int, optional
        Number of iterations for the unbiased estimator. Default is 30.

    Attributes
    ----------
    mu : float
        The learning rate or step size for the LMS algorithm.
    xi : np.ndarray or None
        The estimation error at each iteration. Initialized as None and updated during
        optimization.

    Methods
    -------
    optimize(psi: np.ndarray, y: np.ndarray) -> np.ndarray
        Estimate the model parameters using the LMS filter.

    References
    ----------
    - Hayes, M. H. (2009). Statistical digital signal processing and modeling.
    John Wiley & Sons.
    - Zipf, J. G. F. (2011). Classificação, análise estatística e novas estratégias de
    algoritmos LMS de passo variável.
    - Wikipedia entry on Least Mean Squares: https://en.wikipedia.org/wiki/Least_mean_squares_filter
    """

    def __init__(self, *, mu: float = 0.01, unbiased: bool = False, uiter: int = 30):
        self.mu = mu
        self.uiter = uiter
        self.unbiased = unbiased
        _validate_params(vars(self))
        self.xi: Optional[np.ndarray] = None

    def optimize(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
        r"""Parameter estimation using the Sign-Error Least Mean Squares filter.

        The sign-error LMS algorithm updates the parameter estimates recursively as
        follows:

        1. Compute the estimation error:

           $$
           \xi_i = y_i - \psi_i^T \theta_{i-1}
           $$

        2. Update the parameter vector:

           $$
           \theta_i = \theta_{i-1} + \mu \cdot \text{sign}(\xi_i) \cdot \psi_i
           $$

        Parameters
        ----------
        psi : ndarray of floats
            The information matrix of the model.
        y : array-like of shape (n_samples, 1)
            The data used to train the model.

        Returns
        -------
        theta : array-like of shape (n_features, 1)
            The estimated parameters of the model.

        """
        n_theta, n, theta, self.xi = _initial_values(psi)

        for i in range(n_theta, n):
            psi_tmp = psi[i, :].reshape(-1, 1)
            self.xi[i, 0] = y[i, 0] - np.dot(psi_tmp.T, theta[:, i - 1])[0]
            tmp_list = (
                theta[:, i - 1].reshape(-1, 1)
                + self.mu * np.sign(self.xi[i, 0]) * psi_tmp
            )
            theta[:, i] = tmp_list.flatten()

        return theta[:, -1].reshape(-1, 1)


class NormalizedLeastMeanSquares(BaseEstimator):
    """Normalized Least Mean Squares (NLMS) filter for parameter estimation.

    The NLMS algorithm is an adaptive filter used to estimate the parameters of a model
    by minimizing the mean square error between the observed and predicted values. The
    normalization is used to avoid numerical instability when updating the estimated
    parameters.

    Parameters
    ----------
    mu : float, default=0.01
        The learning rate or step size for the LMS algorithm.
    eps : float, default=np.finfo(np.float64).eps
        Normalization factor of the normalized filters.

    Attributes
    ----------
    mu : float
        The learning rate or step size for the LMS algorithm.
    eps : float
        Normalization factor of the normalized filters.
    xi : np.ndarray or None
        The estimation error at each iteration. Initialized as None and updated during
        optimization.

    Methods
    -------
    optimize(psi: np.ndarray, y: np.ndarray) -> np.ndarray
        Estimate the model parameters using the NLMS filter.

    References
    ----------
    - Hayes, M. H. (2009). Statistical digital signal processing and modeling.
      John Wiley & Sons.
    - Zipf, J. G. F. (2011). Classificação, análise estatística e novas estratégias de
      algoritmos LMS de passo variável.
    - Wikipedia entry on Least Mean Squares: https://en.wikipedia.org/wiki/Least_mean_squares_filter
    """

    def __init__(
        self,
        *,
        mu: float = 0.01,
        eps: np.float64 = np.finfo(np.float64).eps,
        unbiased: bool = False,
        uiter: int = 30,
    ):
        self.mu = mu
        self.eps = eps
        self.unbiased = unbiased
        self.uiter = uiter
        _validate_params(vars(self))
        self.xi: Optional[np.ndarray] = None

    def optimize(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
        r"""Parameter estimation using the Normalized Least Mean Squares filter.

        The NLMS algorithm updates the parameter estimates recursively as follows:

        1. Compute the estimation error:

           $$
           \xi_i = y_i - \psi_i^T \theta_{i-1}
           $$

        2. Update the parameter vector:

           $$
           \theta_i = \theta_{i-1} + 2 \mu \xi_i \frac{\psi_i}{\epsilon +
           \psi_i^T \psi_i}
           $$

        Parameters
        ----------
        psi : ndarray of floats
            The information matrix of the model.
        y : array-like of shape (n_samples, 1)
            The data used to train the model.

        Returns
        -------
        theta : array-like of shape (n_features, 1)
            The estimated parameters of the model.

        """
        n_theta, n, theta, self.xi = _initial_values(psi)

        for i in range(n_theta, n):
            psi_tmp = psi[i, :].reshape(-1, 1)
            self.xi[i, 0] = y[i, 0] - np.dot(psi_tmp.T, theta[:, i - 1])[0]
            tmp_list = theta[:, i - 1].reshape(-1, 1) + 2 * self.mu * self.xi[i, 0] * (
                psi_tmp / (self.eps + np.dot(psi_tmp.T, psi_tmp))
            )
            theta[:, i] = tmp_list.flatten()

        return theta[:, -1].reshape(-1, 1)


class NormalizedLeastMeanSquaresSignError(BaseEstimator):
    """Normalized Least Mean Squares SignError (NLMSSE) filter for parameter estimation.

    The NLMSSE algorithm updates the parameter estimates recursively by normalizing
    the input signal to avoid numerical instability and using the sign of the error
    vector to adjust the filter coefficients.

    Parameters
    ----------
    mu : float, default=0.01
        The learning rate or step size for the LMS algorithm.
    eps : float, default=np.finfo(np.float64).eps
        Normalization factor of the normalized filters.

    Attributes
    ----------
    mu : float
        The learning rate or step size for the LMS algorithm.
    eps : float
        Normalization factor of the normalized filters.
    xi : np.ndarray or None
        The estimation error at each iteration. Initialized as None and updated during
        optimization.

    Methods
    -------
    optimize(psi: np.ndarray, y: np.ndarray) -> np.ndarray
        Estimate the model parameters using the NLMSSE filter.

    References
    ----------
    - Hayes, M. H. (2009). Statistical digital signal processing and modeling.
      John Wiley & Sons.
    - Zipf, J. G. F. (2011). Classificação, análise estatística e novas estratégias de
      algoritmos LMS de passo variável.
    - Wikipedia entry on Least Mean Squares: https://en.wikipedia.org/wiki/Least_mean_squares_filter
    """

    def __init__(
        self,
        *,
        mu: float = 0.01,
        eps: np.float64 = np.finfo(np.float64).eps,
        unbiased: bool = False,
        uiter: int = 30,
    ):
        self.mu = mu
        self.eps = eps
        self.unbiased = unbiased
        self.uiter = uiter
        _validate_params(vars(self))
        self.xi: Optional[np.ndarray] = None

    def optimize(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
        r"""Parameter estimation using the Normalized Sign-Error LMS filter.

        The NLMSSE algorithm updates the parameter estimates recursively as follows:

        1. Compute the estimation error:

           $$
           \xi_i = y_i - \psi_i^T \theta_{i-1}
           $$

        2. Update the parameter vector:

           $$
           \theta_i = \theta_{i-1} + 2 \mu \cdot \text{sign}(\xi_i) \cdot
           \frac{\psi_i}{\epsilon + \psi_i^T \psi_i}
           $$

        Parameters
        ----------
        psi : ndarray of floats
            The information matrix of the model.
        y : array-like of shape (n_samples, 1)
            The data used to train the model.

        Returns
        -------
        theta : array-like of shape (n_features, 1)
            The estimated parameters of the model.

        Notes
        -----
        The normalization is used to avoid numerical instability when updating
        the estimated parameters and the sign of the error vector is used to
        change the filter coefficients.
        """
        n_theta, n, theta, self.xi = _initial_values(psi)

        for i in range(n_theta, n):
            psi_tmp = psi[i, :].reshape(-1, 1)
            self.xi[i, 0] = y[i, 0] - np.dot(psi_tmp.T, theta[:, i - 1])[0]
            tmp_list = theta[:, i - 1].reshape(-1, 1) + 2 * self.mu * np.sign(
                self.xi[i, 0]
            ) * (psi_tmp / (self.eps + np.dot(psi_tmp.T, psi_tmp)))
            theta[:, i] = tmp_list.flatten()

        return theta[:, -1].reshape(-1, 1)


class LeastMeanSquaresSignRegressor(BaseEstimator):
    """Least Mean Squares (LMSSR) filter for parameter estimation.

    The sign-regressor LMS algorithm uses the sign of the matrix
    information to change the filter coefficients.

    Parameters
    ----------
    mu : float, default=0.01
        The learning rate or step size for the LMS algorithm.
    unbiased : bool, optional
        If True, applies an unbiased estimator. Default is False.
    uiter : int, optional
        Number of iterations for the unbiased estimator. Default is 30.

    Attributes
    ----------
    mu : float
        The learning rate or step size for the LMS algorithm.
    unbiased : bool
        Indicates whether an unbiased estimator is applied.
    uiter : int
        Number of iterations for the unbiased estimator.
    xi : np.ndarray or None
        The estimation error at each iteration. Initialized as None and updated during
        optimization.

    Methods
    -------
    optimize(psi: np.ndarray, y: np.ndarray) -> np.ndarray
        Estimate the model parameters using the LMS filter.

    References
    ----------
    - Hayes, M. H. (2009). Statistical digital signal processing and modeling.
      John Wiley & Sons.
    - Zipf, J. G. F. (2011). Classificação, análise estatística e novas estratégias de
      algoritmos LMS de passo variável.
    - Wikipedia entry on Least Mean Squares: https://en.wikipedia.org/wiki/Least_mean_squares_filter
    """

    def __init__(self, *, mu: float = 0.01, unbiased: bool = False, uiter: int = 30):
        self.mu = mu
        self.unbiased = unbiased
        self.uiter = uiter
        _validate_params(vars(self))
        self.xi: Optional[np.ndarray] = None

    def optimize(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
        r"""Parameter estimation using the Sign-Regressor LMS filter.

        The sign-regressor LMS algorithm updates the parameter estimates recursively
        as follows:

        1. Compute the estimation error:

           $$
           \xi_i = y_i - \psi_i^T \theta_{i-1}
           $$

        2. Update the parameter vector:

           $$
           \theta_i = \theta_{i-1} + \mu \cdot \xi_i \cdot \text{sign}(\psi_i)
           $$

        Parameters
        ----------
        psi : ndarray of floats
            The information matrix of the model.
        y : array-like of shape (n_samples, 1)
            The data used to train the model.

        Returns
        -------
        theta : array-like of shape (n_features, 1)
            The estimated parameters of the model.
        """
        n_theta, n, theta, self.xi = _initial_values(psi)

        for i in range(n_theta, n):
            psi_tmp = psi[i, :].reshape(-1, 1)
            self.xi[i, 0] = y[i, 0] - np.dot(psi_tmp.T, theta[:, i - 1])[0]
            tmp_list = theta[:, i - 1].reshape(-1, 1) + self.mu * self.xi[
                i, 0
            ] * np.sign(psi_tmp)
            theta[:, i] = tmp_list.flatten()

        return theta[:, -1].reshape(-1, 1)


class LeastMeanSquaresNormalizedSignRegressor(BaseEstimator):
    """Normalized Least Mean Squares SignRegressor filter for parameter estimation.

    The Normalized Sign-Regressor LMS algorithm updates the parameter estimates
    recursively by normalizing the input signal to avoid numerical instability
    and using the sign of the information matrix to adjust the filter coefficients.

    Parameters
    ----------
    mu : float, default=0.01
        The learning rate or step size for the LMS algorithm.
    eps : float, default=np.finfo(np.float64).eps
        Normalization factor of the normalized filters.

    Attributes
    ----------
    mu : float
        The learning rate or step size for the LMS algorithm.
    eps : float, default=np.finfo(np.float64).eps
        Normalization factor of the normalized filters.
    xi : np.ndarray or None
        The estimation error at each iteration. Initialized as None and updated during
        optimization.

    Methods
    -------
    optimize(psi: np.ndarray, y: np.ndarray) -> np.ndarray
        Estimate the model parameters using the Normalized Sign-Regressor LMS filter.

    References
    ----------
    - Hayes, M. H. (2009). Statistical digital signal processing and modeling.
      John Wiley & Sons.
    - Zipf, J. G. F. (2011). Classificação, análise estatística e novas estratégias de
      algoritmos LMS de passo variável.
    - Wikipedia entry on Least Mean Squares
      https://en.wikipedia.org/wiki/Least_mean_squares_filter
    """

    def __init__(
        self,
        *,
        mu: float = 0.01,
        eps: np.float64 = np.finfo(np.float64).eps,
        unbiased: bool = False,
        uiter: int = 30,
    ):
        self.mu = mu
        self.eps = eps
        self.unbiased = unbiased
        self.uiter = uiter
        _validate_params(vars(self))
        self.xi: Optional[np.ndarray] = None

    def optimize(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
        r"""Parameter estimation using the Normalized Sign-Regressor LMS filter.

        The Normalized Sign-Regressor LMS algorithm updates the parameter estimates
        recursively as follows:

        1. Compute the estimation error:

           $$
           \xi_i = y_i - \psi_i^T \theta_{i-1}
           $$

        2. Update the parameter vector:

           $$
           \theta_i = \theta_{i-1} + \mu \cdot \xi_i \cdot
           \frac{\text{sign}(\psi_i)}{\epsilon + \psi_i^T \psi_i}
           $$

        The normalization is used to avoid numerical instability when updating
        the estimated parameters and the sign of the information matrix is
        used to change the filter coefficients.

        Parameters
        ----------
        psi : ndarray of floats
            The information matrix of the model.
        y : array-like of shape (n_samples, 1)
            The data used to train the model.

        Returns
        -------
        theta : array-like of shape (n_features, 1)
            The estimated parameters of the model.
        """
        n_theta, n, theta, self.xi = _initial_values(psi)

        for i in range(n_theta, n):
            psi_tmp = psi[i, :].reshape(-1, 1)
            self.xi[i, 0] = y[i, 0] - np.dot(psi_tmp.T, theta[:, i - 1])[0]
            tmp_list = theta[:, i - 1].reshape(-1, 1) + self.mu * self.xi[i, 0] * (
                np.sign(psi_tmp) / (self.eps + np.dot(psi_tmp.T, psi_tmp))
            )
            theta[:, i] = tmp_list.flatten()

        return theta[:, -1].reshape(-1, 1)


class LeastMeanSquaresSignSign(BaseEstimator):
    """Least Mean Squares Sign-Sign (LMSSS) filter for parameter estimation.

    The LMSSS algorithm uses both the sign of the matrix information and the sign of
    the error vector to update the filter coefficients.

    Parameters
    ----------
    mu : float, default=0.01
        The learning rate or step size for the LMS algorithm.
    unbiased : bool, optional
        If True, applies an unbiased estimator. Default is False.
    uiter : int, optional
        Number of iterations for the unbiased estimator. Default is 30.

    Attributes
    ----------
    mu : float
        The learning rate or step size for the LMS algorithm.
    xi : np.ndarray or None
        The estimation error at each iteration. Initialized as None and updated during
        optimization.

    Methods
    -------
    optimize(psi: np.ndarray, y: np.ndarray) -> np.ndarray
        Estimate the model parameters using the LMSSS filter.

    References
    ----------
    - Hayes, M. H. (2009). Statistical digital signal processing and modeling.
    John Wiley & Sons.
    - Zipf, J. G. F. (2011). Classificação, análise estatística e novas estratégias de
    algoritmos LMS de passo variável.
    - Wikipedia entry on Least Mean Squares: https://en.wikipedia.org/wiki/Least_mean_squares_filter
    """

    def __init__(self, *, mu: float = 0.01, unbiased: bool = False, uiter: int = 30):
        self.mu = mu
        self.unbiased = unbiased
        self.uiter = uiter
        _validate_params(vars(self))
        self.xi: Optional[np.ndarray] = None

    def optimize(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
        r"""Parameter estimation using the Sign-Sign LMS filter.

        The LMSSS algorithm updates the parameter estimates recursively as follows:

        1. Compute the estimation error:

           $$
           \xi_i = y_i - \psi_i^T \theta_{i-1}
           $$

        2. Update the parameter vector:

           $$
           \theta_i = \theta_{i-1} + 2* \mu \cdot \text{sign}(\xi_i)
           \cdot \text{sign}(\psi_i)
           $$

        Parameters
        ----------
        psi : ndarray of floats
            The information matrix of the model.
        y : array-like of shape (n_samples, 1)
            The data used to train the model.

        Returns
        -------
        theta : array-like of shape (n_features, 1)
            The estimated parameters of the model.
        """
        n_theta, n, theta, self.xi = _initial_values(psi)

        for i in range(n_theta, n):
            psi_tmp = psi[i, :].reshape(-1, 1)
            self.xi[i, 0] = y[i, 0] - np.dot(psi_tmp.T, theta[:, i - 1])[0]
            tmp_list = theta[:, i - 1].reshape(-1, 1) + 2 * self.mu * np.sign(
                self.xi[i, 0]
            ) * np.sign(psi_tmp)
            theta[:, i] = tmp_list.flatten()

        return theta[:, -1].reshape(-1, 1)


class LeastMeanSquaresNormalizedSignSign(BaseEstimator):
    """Normalized Least Mean Squares SignSign (NLMSSS) filter for parameter estimation.

    The NLMSSS algorithm updates the parameter estimates recursively by normalizing
    the input signal to avoid numerical instability and using both the sign of the
    information matrix and the sign of the error vector to adjust the filter
    coefficients.

    Parameters
    ----------
    mu : float, default=0.01
        The learning rate or step size for the LMS algorithm.
    eps : float, default=np.finfo(np.float64).eps
        Normalization factor of the normalized filters.

    Attributes
    ----------
    mu : float
        The learning rate or step size for the LMS algorithm.
    eps : float
        Normalization factor of the normalized filters.
    xi : np.ndarray or None
        The estimation error at each iteration. Initialized as None and updated during
        optimization.

    Methods
    -------
    optimize(psi: np.ndarray, y: np.ndarray) -> np.ndarray
        Estimate the model parameters using the NLMSSS filter.

    References
    ----------
    - Hayes, M. H. (2009). Statistical digital signal processing and modeling.
      John Wiley & Sons.
    - Zipf, J. G. F. (2011). Classificação, análise estatística e novas estratégias de
      algoritmos LMS de passo variável.
    - Wikipedia entry on Least Mean Squares: https://en.wikipedia.org/wiki/Least_mean_squares_filter
    """

    def __init__(
        self,
        *,
        mu: float = 0.01,
        eps: np.float64 = np.finfo(np.float64).eps,
        unbiased: bool = False,
        uiter: int = 30,
    ):
        self.mu = mu
        self.eps = eps
        self.unbiased = unbiased
        self.uiter = uiter
        _validate_params(vars(self))
        self.xi: Optional[np.ndarray] = None

    def optimize(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
        r"""Parameter estimation using the Normalized Sign-Sign LMS filter.

        The NLMSSS algorithm updates the parameter estimates recursively as follows:

        1. Compute the estimation error:

           $$
           \xi_i = y_i - \psi_i^T \theta_{i-1}
           $$

        2. Update the parameter vector:

           $$
           \theta_i = \theta_{i-1} + 2 \mu \cdot \text{sign}(\xi_i) \cdot
           \frac{\text{sign}(\psi_i)}{\epsilon + \psi_i^T \psi_i}
           $$

        The normalization is used to avoid numerical instability when updating
        the estimated parameters and both the sign of the information matrix
        and the sign of the error vector are used to change the filter
        coefficients.

        Parameters
        ----------
        psi : ndarray of floats
            The information matrix of the model.
        y : array-like of shape (n_samples, 1)
            The data used to train the model.

        Returns
        -------
        theta : array-like of shape (n_features, 1)
            The estimated parameters of the model.

        References
        ----------
        - Hayes, M. H. (2009). Statistical digital signal processing and modeling.
          John Wiley & Sons.
        - Zipf, J. G. F. (2011). Classificação, análise estatística e novas estratégias
        de algoritmos LMS de passo variável.
        - Wikipedia entry on Least Mean Squares: https://en.wikipedia.org/wiki/Least_mean_squares_filter
        """
        n_theta, n, theta, self.xi = _initial_values(psi)

        for i in range(n_theta, n):
            psi_tmp = psi[i, :].reshape(-1, 1)
            self.xi[i, 0] = y[i, 0] - np.dot(psi_tmp.T, theta[:, i - 1])[0]
            tmp_list = theta[:, i - 1].reshape(-1, 1) + 2 * self.mu * np.sign(
                self.xi[i, 0]
            ) * (np.sign(psi_tmp) / (self.eps + np.dot(psi_tmp.T, psi_tmp)))
            theta[:, i] = tmp_list.flatten()

        return theta[:, -1].reshape(-1, 1)


class LeastMeanSquaresNormalizedLeaky(BaseEstimator):
    """Normalized Least Mean Squares Leaky (NLMSL) filter for parameter estimation.

    The NLMSL algorithm is an adaptive filter used to estimate the parameters of a model
    by minimizing the mean square error between the observed and predicted values. The
    normalization is used to avoid numerical instability when updating the estimated
    parameters, and the leakage factor helps to prevent coefficient drift.

    Parameters
    ----------
    mu : float, default=0.01
        The learning rate or step size for the LMS algorithm.
    eps : float, default=np.finfo(np.float64).eps
        Normalization factor of the normalized filters.
    gama : float, default=0.2
        The leakage factor of the Leaky LMS method.

    Attributes
    ----------
    mu : float
        The learning rate or step size for the LMS algorithm.
    eps : float, default=np.finfo(np.float64).eps
        Normalization factor of the normalized filters.
    gama : float, default=0.2
        The leakage factor of the Leaky LMS method.
    xi : np.ndarray or None
        The estimation error at each iteration. Initialized as None and updated during
        optimization.

    Methods
    -------
    optimize(psi: np.ndarray, y: np.ndarray) -> np.ndarray
        Estimate the model parameters using the NLMSL filter.

    References
    ----------
    - Hayes, M. H. (2009). Statistical digital signal processing and modeling.
      John Wiley & Sons.
    - Zipf, J. G. F. (2011). Classificação, análise estatística e novas estratégias de
      algoritmos LMS de passo variável.
    - Wikipedia entry on Least Mean Squares: https://en.wikipedia.org/wiki/Least_mean_squares_filter
    """

    def __init__(
        self,
        *,
        mu: float = 0.01,
        gama: float = 0.2,
        eps: np.float64 = np.finfo(np.float64).eps,
        unbiased: bool = False,
        uiter: int = 30,
    ):
        self.mu = mu
        self.eps = eps
        self.gama = gama
        self.unbiased = unbiased
        self.uiter = uiter
        _validate_params(vars(self))
        self.xi: Optional[np.ndarray] = None

    def optimize(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
        r"""Parameter estimation using the Normalized Leaky LMS filter.

        The NLMSL algorithm updates the parameter estimates recursively as follows:

        1. Compute the estimation error:

           $$
           \xi_i = y_i - \psi_i^T \theta_{i-1}
           $$

        2. Update the parameter vector:

           $$
           \theta_i = \theta_{i-1} (1 - \mu \gamma) + \mu \frac{\xi_i \psi_i}{\epsilon
           + \psi_i^T \psi_i}
           $$

        When the leakage factor, $\gamma$, is set to 0, there is no leakage in the
        estimation process.

        Parameters
        ----------
        psi : ndarray of floats
            The information matrix of the model.
        y : array-like of shape (n_samples, 1)
            The data used to train the model.

        Returns
        -------
        theta : array-like of shape (n_features, 1)
            The estimated parameters of the model.

        References
        ----------
        - Hayes, M. H. (2009). Statistical digital signal processing and modeling.
          John Wiley & Sons.
        - Zipf, J. G. F. (2011). Classificação, análise estatística e novas estratégias
          de algoritmos LMS de passo variável.
        - Wikipedia entry on Least Mean Squares: https://en.wikipedia.org/wiki/Least_mean_squares_filter
        """
        n_theta, n, theta, self.xi = _initial_values(psi)

        for i in range(n_theta, n):
            psi_tmp = psi[i, :].reshape(-1, 1)
            self.xi[i, 0] = y[i, 0] - np.dot(psi_tmp.T, theta[:, i - 1])[0]
            tmp_list = theta[:, i - 1].reshape(-1, 1) * (
                1 - self.mu * self.gama
            ) + self.mu * self.xi[i, 0] * psi_tmp / (
                self.eps + np.dot(psi_tmp.T, psi_tmp)
            )
            theta[:, i] = tmp_list.flatten()

        return theta[:, -1].reshape(-1, 1)


class LeastMeanSquaresLeaky(BaseEstimator):
    """Least Mean Squares Leaky (LMSL) filter for parameter estimation.

    The LMSL algorithm is an adaptive filter used to estimate the parameters of a model
    by minimizing the mean square error between the observed and predicted values. The
    leakage factor helps to prevent coefficient drift.

    Parameters
    ----------
    mu : float, default=0.01
        The learning rate or step size for the LMS algorithm.
    gama : float, default=0.2
        The leakage factor of the Leaky LMS method.
    unbiased : bool, optional
        If True, applies an unbiased estimator. Default is False.
    uiter : int, optional
        Number of iterations for the unbiased estimator. Default is 30.

    Attributes
    ----------
    mu : float
        The learning rate or step size for the LMS algorithm.
    gama : float, default=0.2
        The leakage factor of the Leaky LMS method.
    xi : np.ndarray or None
        The estimation error at each iteration. Initialized as None and updated during
        optimization.

    Methods
    -------
    optimize(psi: np.ndarray, y: np.ndarray) -> np.ndarray
        Estimate the model parameters using the LMSL filter.

    References
    ----------
    - Hayes, M. H. (2009). Statistical digital signal processing and modeling.
      John Wiley & Sons.
    - Zipf, J. G. F. (2011). Classificação, análise estatística e novas estratégias de
      algoritmos LMS de passo variável.
    - Wikipedia entry on Least Mean Squares: https://en.wikipedia.org/wiki/Least_mean_squares_filter
    """

    def __init__(
        self,
        *,
        mu: float = 0.01,
        gama: float = 0.001,
        unbiased: bool = False,
        uiter: int = 30,
    ):
        self.mu = mu
        self.gama = gama
        self.unbiased = unbiased
        self.uiter = uiter
        _validate_params(vars(self))
        self.xi: Optional[np.ndarray] = None

    def optimize(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
        r"""Parameter estimation using the Leaky LMS filter.

        The LMSL algorithm updates the parameter estimates recursively as follows:

        1. Compute the estimation error:

           $$
           \xi_i = y_i - \psi_i^T \theta_{i-1}
           $$

        2. Update the parameter vector:

           $$
           \theta_i = \theta_{i-1} (1 - \mu \gamma) + \mu \xi_i \psi_i
           $$

        When the leakage factor, $\gamma$, is set to 0, there is no leakage in the
        estimation process.

        Parameters
        ----------
        psi : ndarray of floats
            The information matrix of the model.
        y : array-like of shape (n_samples, 1)
            The data used to train the model.

        Returns
        -------
        theta : array-like of shape (n_features, 1)
            The estimated parameters of the model.

        References
        ----------
        - Hayes, M. H. (2009). Statistical digital signal processing and modeling.
          John Wiley & Sons.
        - Zipf, J. G. F. (2011). Classificação, análise estatística e novas estratégias
          de algoritmos LMS de passo variável.
        - Wikipedia entry on Least Mean Squares: https://en.wikipedia.org/wiki/Least_mean_squares_filter
        """
        n_theta, n, theta, self.xi = _initial_values(psi)

        for i in range(n_theta, n):
            psi_tmp = psi[i, :].reshape(-1, 1)
            self.xi[i, 0] = y[i, 0] - np.dot(psi_tmp.T, theta[:, i - 1])[0]
            tmp_list = (
                theta[:, i - 1].reshape(-1, 1) * (1 - self.mu * self.gama)
                + self.mu * self.xi[i, 0] * psi_tmp
            )
            theta[:, i] = tmp_list.flatten()

        return theta[:, -1].reshape(-1, 1)


class LeastMeanSquaresFourth(BaseEstimator):
    """Least Mean Squares Fourth (LMSF) filter for parameter estimation.

    The LMSF algorithm is an adaptive filter used to estimate the parameters of a model
    by using the mean fourth error cost function to eliminate the noise effectively.

    Parameters
    ----------
    mu : float, default=0.5
        The learning rate or step size for the LMS algorithm.
    unbiased : bool, optional
        If True, applies an unbiased estimator. Default is False.
    uiter : int, optional
        Number of iterations for the unbiased estimator. Default is 30.

    Attributes
    ----------
    mu : float
        The learning rate or step size for the LMS algorithm.
    unbiased : bool
        Indicates whether an unbiased estimator is applied.
    uiter : int
        Number of iterations for the unbiased estimator.
    xi : np.ndarray or None
        The estimation error at each iteration. Initialized as None and updated during
        optimization.

    Methods
    -------
    optimize(psi: np.ndarray, y: np.ndarray) -> np.ndarray
        Estimate the model parameters using the LMSF filter.

    References
    ----------
    - Hayes, M. H. (2009). Statistical digital signal processing and modeling.
      John Wiley & Sons.
    - Zipf, J. G. F. (2011). Classificação, análise estatística e novas estratégias de
      algoritmos LMS de passo variável.
    - Gui, G., Mehbodniya, A., & Adachi, F. (2013). Least mean square/fourth algorithm
      with application to sparse channel estimation. arXiv preprint arXiv:1304.3911.
      https://arxiv.org/pdf/1304.3911.pdf
    - Nascimento, V. H., & Bermudez, J. C. M. (2005, March). When is the least-mean
      fourth algorithm mean-square stable? In Proceedings.(ICASSP'05). IEEE
      International Conference on Acoustics, Speech, and Signal Processing, 2005.
      (Vol. 4, pp. iv-341). IEEE. http://www.lps.usp.br/vitor/artigos/icassp05.pdf
    - Wikipedia entry on Least Mean Squares: https://en.wikipedia.org/wiki/Least_mean_squares_filter
    """

    def __init__(self, *, mu: float = 0.5, unbiased: bool = False, uiter: int = 30):
        self.mu = mu
        self.unbiased = unbiased
        self.uiter = uiter
        _validate_params(vars(self))
        self.xi: Optional[np.ndarray] = None

    def optimize(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
        r"""Parameter estimation using the LMS Fourth filter.

        The LMSF algorithm updates the parameter estimates recursively as follows:

        1. Compute the estimation error:

           $$
           \xi_i = y_i - \psi_i^T \theta_{i-1}
           $$

        2. Update the parameter vector:

           $$
           \theta_i = \theta_{i-1} + \mu \psi_i \xi_i^3
           $$

        Parameters
        ----------
        psi : ndarray of floats
            The information matrix of the model.
        y : ndarray of floats of shape (n_samples, 1)
            The data used to train the model.

        Returns
        -------
        theta : ndarray of floats of shape (n_features, 1)
            The estimated parameters of the model.
        """
        n_theta, n, theta, self.xi = _initial_values(psi)

        for i in range(n_theta, n):
            psi_tmp = psi[i, :].reshape(-1, 1)
            self.xi[i, 0] = y[i, 0] - np.dot(psi_tmp.T, theta[:, i - 1])[0]
            tmp_list = (
                theta[:, i - 1].reshape(-1, 1) + self.mu * psi_tmp * self.xi[i, 0] ** 3
            )
            theta[:, i] = tmp_list.flatten()

        return theta[:, -1].reshape(-1, 1)


class LeastMeanSquareMixedNorm(BaseEstimator):
    """Least Mean Square Mixed Norm (LMS-MN) Adaptive Filter.

    This class implements the Mixed-norm Least Mean Square (LMS) adaptive filter
    algorithm, which incorporates an additional weight factor to control the
    proportions of the error norms, thus providing an extra degree of freedom
    in the adaptation process.

    Parameters
    ----------
    mu : float, optional
        The adaptation step size. Default is 0.01.
    weight : float, optional
        The weight factor for mixed-norm control. This factor controls the
        proportions of the error norms and offers an extra degree of freedom
        within the adaptation of the LMS mixed norm method.
    unbiased : bool, optional
        If True, applies an unbiased estimator. Default is False.
    uiter : int, optional
        Number of iterations for the unbiased estimator. Default is 30.

    Attributes
    ----------
    mu : float
        The adaptation step size.
    weight : float
        The weight factor for mixed-norm control.
    xi : ndarray or None
        The error signal, initialized to None.

    Methods
    -------
    optimize(psi: np.ndarray, y: np.ndarray) -> np.ndarray
        Estimate the model parameters using the LMSF filter.

    References
    ----------
    - Chambers, J. A., Tanrikulu, O., & Constantinides, A. G. (1994).
      Least mean mixed-norm adaptive filtering.
      Electronics letters, 30(19), 1574-1575.
      https://ieeexplore.ieee.org/document/326382
    - Dissertation (Portuguese): Zipf, J. G. F. (2011). Classificação,
      análise estatística e novas estratégias de algoritmos LMS de passo
      variável.
    - Wikipedia entry on Least Mean Squares
      https://en.wikipedia.org/wiki/Least_mean_squares_filter
    """

    def __init__(
        self,
        *,
        mu: float = 0.01,
        weight: float = 0.02,
        unbiased: bool = False,
        uiter: int = 30,
    ):
        self.mu = mu
        self.weight = weight
        self.unbiased = unbiased
        self.uiter = uiter
        _validate_params(vars(self))
        self.xi: Optional[np.ndarray] = None

    def optimize(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
        r"""Parameter estimation using the Mixed-norm LMS filter.

        The LMS-MN algorithm updates the parameter estimates recursively as follows:

        1. Compute the estimation error:

           $$
           \xi_i = y_i - \psi_i^T \theta_{i-1}
           $$

        2. Update the parameter vector:

           $$
           \theta_i = \theta_{i-1} + \mu \psi_i \xi_i (\text{weight}
           + (1 - \text{weight}) \xi_i^2)
           $$

        Parameters
        ----------
        psi : ndarray of floats
            The information matrix of the model.
        y : array-like of shape (n_samples, 1)
            The data used to train the model.

        Returns
        -------
        theta : array-like of shape (n_features, 1)
            The estimated parameters of the model.

        Notes
        -----
        A more in-depth documentation of all methods for parameter estimation
        will be available soon. For now, please refer to the mentioned references.
        """
        n_theta, n, theta, self.xi = _initial_values(psi)

        for i in range(n_theta, n):
            psi_tmp = psi[i, :].reshape(-1, 1)
            self.xi[i, 0] = y[i, 0] - np.dot(psi_tmp.T, theta[:, i - 1])[0]
            tmp_list = theta[:, i - 1].reshape(-1, 1) + self.mu * psi_tmp * self.xi[
                i, 0
            ] * (self.weight + (1 - self.weight) * self.xi[i, 0] ** 2)
            theta[:, i] = tmp_list.flatten()

        return theta[:, -1].reshape(-1, 1)


class NonNegativeLeastSquares(BaseEstimator):
    """Solve ``argmin_x || Ax - b ||_2`` for ``x >= 0``.

    This is a wrapper class for the `scipy.optimize.nnls` method.

    This problem, often called NonNegative Least Squares (NNLS), is a convex
    optimization problem with convex constraints. It typically arises when
    the ``x`` models quantities for which only nonnegative values are
    attainable; such as weights of ingredients, component costs, and so on.

    Parameters
    ----------
    unbiased : bool, optional
        If True, applies an unbiased estimator. Default is False.
    uiter : int, optional
        Number of iterations for the unbiased estimator. Default is 30.
    maxiter : int, optional
        Maximum number of iterations. Default value is ``3 * n`` where ``n``
        is the number of features.
    atol : float, optional
        Tolerance value used in the algorithm to assess closeness to zero in
        the projected residual ``(A.T @ (A x - b))`` entries. Increasing this
        value relaxes the solution constraints. A typical relaxation value can
        be selected as ``max(m, n) * np.linalg.norm(A, 1) * np.spacing(1.)``.
        Default is None.

    Attributes
    ----------
    unbiased : bool
        Indicates whether an unbiased estimator is applied.
    uiter : int
        Number of iterations for the unbiased estimator.
    maxiter : int
        Maximum number of iterations.
    atol : float
        Tolerance value for the algorithm.

    References
    ----------
    Lawson C., Hanson R.J., "Solving Least Squares Problems", SIAM,
       1995, :doi:`10.1137/1.9781611971217`
    Bro, Rasmus and de Jong, Sijmen, "A Fast Non-Negativity-Constrained Least
        Squares Algorithm", Journal Of Chemometrics, 1997,
        :doi:`10.1002/(SICI)1099-128X(199709/10)11:5<393::AID-CEM483>3.0.CO;2-L`

    Examples
    --------
    >>> import numpy as np
    >>> from sysidentpy.parameter_estimation import NonNegativeLeastSquares
    ...
    >>> A = np.array([[1, 0], [1, 0], [0, 1]])
    >>> b = np.array([2, 1, 1])
    >>> nnls_solver = NonNegativeLeastSquares()
    >>> x = nnls_solver.optimize(A, b)
    >>> print(x)
    [[1.5]
     [1. ]]

    >>> b = np.array([-1, -1, -1])
    >>> x = nnls_solver.optimize(A, b)
    >>> print(x)
    [[0.]
     [0.]]
    """

    def __init__(
        self, unbiased: bool = False, uiter: int = 30, maxiter=None, atol=None
    ):
        self.unbiased = unbiased
        self.uiter = uiter
        self.maxiter = maxiter
        self.atol = atol

    def optimize(self, psi, y):
        """Parameter estimation using the NonNegativeLeastSquares algorithm.

        Parameters
        ----------
        psi : ndarray of floats
            The information matrix of the model.
        y : ndarray of floats of shape (n_samples, 1)
            The data used to train the model.

        Returns
        -------
        theta : array-like of shape = number_of_model_elements
            The estimated parameters of the model.

        Notes
        -----
        This is a wrapper class for the `scipy.optimize.nnls` method.

        References
        ----------
        .. [1] scipy, https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html
        """
        theta, _ = nnls(psi, y.ravel(), maxiter=self.maxiter, atol=self.atol)
        return theta.reshape(-1, 1)


class BoundedVariableLeastSquares(BaseEstimator):
    """Solve a linear least-squares problem with bounds on the variables.

    This is a wrapper class for the `scipy.optimize.lsq_linear` method.

    Given a m-by-n design matrix A and a target vector b with m elements,
    `lsq_linear` solves the following optimization problem::

        minimize 0.5 * ||A x - b||**2
        subject to lb <= x <= ub

    This optimization problem is convex, hence a found minimum (if iterations
    have converged) is guaranteed to be global.

    Parameters
    ----------
    unbiased : bool
        Indicates whether an unbiased estimator is applied.
    uiter : int
        Number of iterations for the unbiased estimator.
    method : 'trf' or 'bvls', optional
        Method to perform minimization.

            * 'trf' : Trust Region Reflective algorithm adapted for a linear
              least-squares problem. This is an interior-point-like method
              and the required number of iterations is weakly correlated with
              the number of variables.
            * 'bvls' : Bounded-variable least-squares algorithm. This is
              an active set method, which requires the number of iterations
              comparable to the number of variables. Can't be used when `A` is
              sparse or LinearOperator.

        Default is 'trf'.
    tol : float, optional
        Tolerance parameter. The algorithm terminates if a relative change
        of the cost function is less than `tol` on the last iteration.
        Additionally, the first-order optimality measure is considered:

            * ``method='trf'`` terminates if the uniform norm of the gradient,
              scaled to account for the presence of the bounds, is less than
              `tol`.
            * ``method='bvls'`` terminates if Karush-Kuhn-Tucker conditions
              are satisfied within `tol` tolerance.

    lsq_solver : {None, 'exact', 'lsmr'}, optional
        Method of solving unbounded least-squares problems throughout
        iterations:

            * 'exact' : Use dense QR or SVD decomposition approach. Can't be
              used when `A` is sparse or LinearOperator.
            * 'lsmr' : Use `scipy.sparse.linalg.lsmr` iterative procedure
              which requires only matrix-vector product evaluations. Can't
              be used with ``method='bvls'``.

        If None (default), the solver is chosen based on type of `A`.
    lsmr_tol : None, float or 'auto', optional
        Tolerance parameters 'atol' and 'btol' for `scipy.sparse.linalg.lsmr`
        If None (default), it is set to ``1e-2 * tol``. If 'auto', the
        tolerance will be adjusted based on the optimality of the current
        iterate, which can speed up the optimization process, but is not always
        reliable.
    max_iter : None or int, optional
        Maximum number of iterations before termination. If None (default), it
        is set to 100 for ``method='trf'`` or to the number of variables for
        ``method='bvls'`` (not counting iterations for 'bvls' initialization).
    verbose : {0, 1, 2}, optional
        Level of algorithm's verbosity:

            * 0 : work silently (default).
            * 1 : display a termination report.
            * 2 : display progress during iterations.
    lsmr_maxiter : None or int, optional
        Maximum number of iterations for the lsmr least squares solver,
        if it is used (by setting ``lsq_solver='lsmr'``). If None (default), it
        uses lsmr's default of ``min(m, n)`` where ``m`` and ``n`` are the
        number of rows and columns of `A`, respectively. Has no effect if
        ``lsq_solver='exact'``.

    References
    ----------
    M. A. Branch, T. F. Coleman, and Y. Li, "A Subspace, Interior,
        and Conjugate Gradient Method for Large-Scale Bound-Constrained
        Minimization Problems," SIAM Journal on Scientific Computing,
        Vol. 21, Number 1, pp 1-23, 1999.
    P. B. Start and R. L. Parker, "Bounded-Variable Least-Squares:
        an Algorithm and Applications", Computational Statistics, 10,
        129-141, 1995.

    Notes
    -----
    This docstring is adapted from the `scipy.optimize.lsq_linear` method.

    Examples
    --------
    In this example, a problem with a large sparse matrix and bounds on the
    variables is solved.

    >>> import numpy as np
    >>> from scipy.sparse import rand
    >>> from sysidentpy.parameter_estimation import BoundedVariableLeastSquares
    >>> rng = np.random.default_rng()
    ...
    >>> m = 20000
    >>> n = 10000
    ...
    >>> A = rand(m, n, density=1e-4, random_state=rng)
    >>> b = rng.standard_normal(m)
    ...
    >>> lb = rng.standard_normal(n)
    >>> ub = lb + 1
    ...
    >>> res = BoundedVariableLeastSquares(A, b, bounds=(lb, ub), lsmr_tol='auto',
    verbose=1)
    The relative change of the cost function is less than `tol`.
    Number of iterations 16, initial cost 1.5039e+04, final cost 1.1112e+04,
    first-order optimality 4.66e-08.
    """

    def __init__(
        self,
        *,
        unbiased: bool = False,
        uiter: int = 30,
        bounds=(-np.inf, np.inf),
        method="trf",
        tol=1e-10,
        lsq_solver=None,
        lsmr_tol=None,
        max_iter=None,
        verbose=0,
        lsmr_maxiter=None,
    ):
        self.unbiased = unbiased
        self.uiter = uiter
        self.max_iter = max_iter
        self.bounds = bounds
        self.method = method
        self.tol = tol
        self.lsq_solver = lsq_solver
        self.lsmr_tol = lsmr_tol
        self.verbose = verbose
        self.lsmr_maxiter = lsmr_maxiter

    def optimize(self, psi, y):
        """Parameter estimation using the BoundedVariableLeastSquares algorithm.

        Parameters
        ----------
        psi : ndarray of floats
            The information matrix of the model.
        y : ndarray of floats of shape (n_samples, 1)
            The data used to train the model.

        Returns
        -------
        theta : array-like of shape = number_of_model_elements
            The estimated parameters of the model.

        Notes
        -----
        This is a wrapper class for the `scipy.optimize.lsq_linear` method.

        References
        ----------
        .. [1] scipy, https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html
        """
        theta = lsq_linear(
            psi,
            y.ravel(),
            bounds=self.bounds,
            method=self.method,
            tol=self.tol,
            lsq_solver=self.lsq_solver,
            lsmr_tol=self.lsmr_tol,
            max_iter=self.max_iter,
            verbose=self.verbose,
            lsmr_maxiter=self.lsmr_maxiter,
        )
        return theta.x.reshape(-1, 1)


class LeastSquaresMinimalResidual(BaseEstimator):
    """Iterative solver for least-squares minimal residual problems.

    This is a wrapper class for the `scipy.sparse.linalg.lsmr` method.

    lsmr solves the system of linear equations ``Ax = b``. If the system
    is inconsistent, it solves the least-squares problem ``min ||b - Ax||_2``.
    ``A`` is a rectangular matrix of dimension m-by-n, where all cases are
    allowed: m = n, m > n, or m < n. ``b`` is a vector of length m.
    The matrix A may be dense or sparse (usually sparse).

    Parameters
    ----------
    unbiased : bool, optional
        If True, applies an unbiased estimator. Default is False.
    uiter : int, optional
        Number of iterations for the unbiased estimator. Default is 30.

    Attributes
    ----------
    unbiased : bool
        Indicates whether an unbiased estimator is applied.
    uiter : int
        Number of iterations for the unbiased estimator.
    damp : float
        Damping factor for regularized least-squares. `lsmr` solves
        the regularized least-squares problem::

         min ||(b) - (  A   )x||
             ||(0)   (damp*I) ||_2

        where damp is a scalar.  If damp is None or 0, the system
        is solved without regularization. Default is 0.
    atol, btol : float, optional
        Stopping tolerances. `lsmr` continues iterations until a
        certain backward error estimate is smaller than some quantity
        depending on atol and btol.  Let ``r = b - Ax`` be the
        residual vector for the current approximate solution ``x``.
        If ``Ax = b`` seems to be consistent, `lsmr` terminates
        when ``norm(r) <= atol * norm(A) * norm(x) + btol * norm(b)``.
        Otherwise, `lsmr` terminates when ``norm(A^H r) <=
        atol * norm(A) * norm(r)``.  If both tolerances are 1.0e-6 (default),
        the final ``norm(r)`` should be accurate to about 6
        digits. (The final ``x`` will usually have fewer correct digits,
        depending on ``cond(A)`` and the size of LAMBDA.)  If `atol`
        or `btol` is None, a default value of 1.0e-6 will be used.
        Ideally, they should be estimates of the relative error in the
        entries of ``A`` and ``b`` respectively.  For example, if the entries
        of ``A`` have 7 correct digits, set ``atol = 1e-7``. This prevents
        the algorithm from doing unnecessary work beyond the
        uncertainty of the input data.
    conlim : float, optional
        `lsmr` terminates if an estimate of ``cond(A)`` exceeds
        `conlim`.  For compatible systems ``Ax = b``, conlim could be
        as large as 1.0e+12 (say).  For least-squares problems,
        `conlim` should be less than 1.0e+8. If `conlim` is None, the
        default value is 1e+8.  Maximum precision can be obtained by
        setting ``atol = btol = conlim = 0``, but the number of
        iterations may then be excessive. Default is 1e8.
    maxiter : int, optional
        `lsmr` terminates if the number of iterations reaches
        `maxiter`.  The default is ``maxiter = min(m, n)``.  For
        ill-conditioned systems, a larger value of `maxiter` may be
        needed. Default is False.
    show : bool, optional
        Print iterations logs if ``show=True``. Default is False.
    x0 : array_like, shape (n,), optional
        Initial guess of ``x``, if None zeros are used. Default is None.

    References
    ----------
    .. [1] D. C.-L. Fong and M. A. Saunders,
           "LSMR: An iterative algorithm for sparse least-squares problems",
           SIAM J. Sci. Comput., vol. 33, pp. 2950-2971, 2011.
           :arxiv:`1006.0758`
    .. [2] LSMR Software, https://web.stanford.edu/group/SOL/software/lsmr/

    Notes
    -----
    This docstring is adapted from the `scipy.sparse.linalg.lsmr` method.
    """

    def __init__(
        self,
        *,
        unbiased: bool = False,
        uiter: int = 30,
        damp=0.0,
        atol=1e-6,
        btol=1e-6,
        conlim=1e8,
        maxiter=None,
        show=False,
        x0=None,
    ):
        self.unbiased = unbiased
        self.uiter = uiter
        self.damp = damp
        self.atol = atol
        self.btol = btol
        self.conlim = conlim
        self.maxiter = maxiter
        self.show = show
        self.x0 = x0

    def optimize(self, psi, y):
        """Parameter estimation using the Mixed-norm LMS filter.

        Parameters
        ----------
        psi : ndarray of floats
            The information matrix of the model.
        y : ndarray of floats of shape (n_samples, 1)
            The data used to train the model.

        Returns
        -------
        theta : array-like of shape = number_of_model_elements
            The estimated parameters of the model.

        Notes
        -----
        This is a wrapper class for the `scipy.sparse.linalg.lsmr` method.

        References
        ----------
        .. [1] scipy, https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsmr.html
        """
        theta = lsmr(
            psi,
            y.ravel(),
            damp=self.damp,
            atol=self.atol,
            btol=self.btol,
            conlim=self.conlim,
            maxiter=self.maxiter,
            show=self.show,
            x0=self.x0,
        )[0]
        return theta.reshape(-1, 1)
