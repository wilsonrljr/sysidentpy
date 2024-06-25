"""Methods for parameter estimation."""

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause


import warnings

import numpy as np

from .estimators_base import BaseEstimator


class EstimatorError(Exception):
    """Generic Python-exception-derived object raised by estimator functions.

    General purpose exception class, derived from Python's ValueError
    class, programmatically raised in estimators functions when a Estimator-related
    condition would prevent further correct execution of the function.

    Parameters
    ----------
    None

    """


class Estimators:
    pass


class LeastSquares(BaseEstimator):
    """Ordinary Least Squares for linear parameter estimation.

    References
    ----------
    - Manuscript: Sorenson, H. W. (1970). Least-squares estimation:
        from Gauss to Kalman. IEEE spectrum, 7(7), 63-68.
        http://pzs.dstu.dp.ua/DataMining/mls/bibl/Gauss2Kalman.pdf
    - Book (Portuguese): Aguirre, L. A. (2007). Introdução identificação
        de sistemas: técnicas lineares e não-lineares aplicadas a sistemas
        reais. Editora da UFMG. 3a edição.
    - Manuscript: Markovsky, I., & Van Huffel, S. (2007).
        Overview of total least-squares methods.
        Signal processing, 87(10), 2283-2302.
        https://eprints.soton.ac.uk/263855/1/tls_overview.pdf
    - Wikipedia entry on Least Squares
        https://en.wikipedia.org/wiki/Least_squares
    """

    def __init__(self, *, unbiased: bool = False, uiter: int = 20):
        self.unbiased = unbiased
        self.uiter = uiter
        self._validate_params(vars(self))

    def optimize(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Estimate the model parameters using Least Squares method.

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

        """
        self._check_linear_dependence_rows(psi)
        theta = np.linalg.lstsq(psi, y, rcond=None)[0]
        return theta


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
        self._validate_params(vars(self))

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
        self._check_linear_dependence_rows(psi)

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
        self._check_linear_dependence_rows(psi)
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
    """Estimate the model parameters using Total Least Squares method.

    _extended_summary_

    Parameters
    ----------
    BaseEstimator : _type_
        _description_

    References
    ----------
    - Manuscript: Golub, G. H., & Van Loan, C. F. (1980).
        An analysis of the total least squares problem.
        SIAM journal on numerical analysis, 17(6), 883-893.
    - Manuscript: Markovsky, I., & Van Huffel, S. (2007).
        Overview of total least-squares methods.
        Signal processing, 87(10), 2283-2302.
        https://eprints.soton.ac.uk/263855/1/tls_overview.pdf
    - Wikipedia entry on Total Least Squares
        https://en.wikipedia.org/wiki/Total_least_squares
    """

    def __init__(self, *, unbiased: bool = False, uiter: int = 30):
        self.unbiased = unbiased
        self.uiter = uiter
        self._validate_params(vars(self))

    def optimize(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Estimate the model parameters using Total Least Squares method.

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

        """
        self._check_linear_dependence_rows(psi)
        full = np.hstack((psi, y))
        n = psi.shape[1]
        _, _, v = np.linalg.svd(full, full_matrices=True)
        theta = -v.T[:n, n:] / v.T[n:, n:]
        return theta.reshape(-1, 1)


class RecursiveLeastSquares(BaseEstimator):
    """_summary_.

    _extended_summary_

    Parameters
    ----------
    lam : float, default=0.98
        Forgetting factor of the Recursive Least Squares method.
    delta : float, default=0.01
        Normalization factor of the P matrix.
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
        self._validate_params(vars(self))
        self.xi = None
        self.theta_evolution = None

    def optimize(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Estimate the model parameters using the Recursive Least Squares method.

        The implementation consider the forgetting factor.

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

        Notes
        -----
        A more in-depth documentation of all methods for parameters estimation
        will be available soon. For now, please refer to the mentioned
        references.

        References
        ----------
        - Book (Portuguese): Aguirre, L. A. (2007). Introdução identificação
           de sistemas: técnicas lineares e não-lineares aplicadas a sistemas
           reais. Editora da UFMG. 3a edição.

        """
        n_theta, n, theta, self.xi = self._initial_values(psi)
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

    Parameters
    ----------
    mu : float, default=0.01
        The learning rate or step size for the LMS algorithm.
    offset_covariance : float, default=0.2
        The offset covariance factor of the affine least mean squares
        filter.

    Attributes
    ----------
    mu : float
        The learning rate or step size for the LMS algorithm.
    offset_covariance : float, default=0.2
        The offset covariance factor of the affine least mean squares
        filter.
    xi : np.ndarray or None
        The estimation error at each iteration. Initialized as None and updated during
        optimization.

    Methods
    -------
    optimize(psi: np.ndarray, y: np.ndarray) -> np.ndarray
        Estimate the model parameters using the LMS filter.
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
        self._validate_params(vars(self))
        self.xi = None

    def optimize(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Estimate the model parameters using the Affine Least Mean Squares.

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

        Notes
        -----
        A more in-depth documentation of all methods for parameters estimation
        will be available soon. For now, please refer to the mentioned
        references.

        References
        ----------
        - Book: Poularikas, A. D. (2017). Adaptive filtering: Fundamentals
           of least mean squares with MATLAB®. CRC Press.

        """
        n_theta, n, theta, self.xi = self._initial_values(psi)

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

    Parameters
    ----------
    mu : float, default=0.01
        The learning rate or step size for the LMS algorithm.

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
    """

    def __init__(self, *, mu: float = 0.01, unbiased: bool = False, uiter: int = 30):
        self.mu = mu
        self.unbiased = unbiased
        self.uiter = uiter
        self._validate_params(vars(self))
        self.xi = None

    def optimize(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Estimate the model parameters using the Least Mean Squares filter.

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

        Notes
        -----
        A more in-depth documentation of all methods for parameters estimation
        will be available soon. For now, please refer to the mentioned
        references.

        References
        ----------
        - Book: Haykin, S., & Widrow, B. (Eds.). (2003). Least-mean-square
           adaptive filters (Vol. 31). John Wiley & Sons.
        - Dissertation (Portuguese): Zipf, J. G. F. (2011). Classificação,
           análise estatística e novas estratégias de algoritmos LMS de passo
           variável.
        - Wikipedia entry on Least Mean Squares
           https://en.wikipedia.org/wiki/Least_mean_squares_filter

        """
        n_theta, n, theta, self.xi = self._initial_values(psi)

        for i in range(n_theta, n):
            psi_tmp = psi[i, :].reshape(-1, 1)
            self.xi[i, 0] = y[i, 0] - np.dot(psi_tmp.T, theta[:, i - 1])[0]
            tmp_list = (
                theta[:, i - 1].reshape(-1, 1) + 2 * self.mu * self.xi[i, 0] * psi_tmp
            )
            theta[:, i] = tmp_list.flatten()

        return theta[:, -1].reshape(-1, 1)


class LeastMeanSquaresSignError(BaseEstimator):
    """Least Mean Squares (LMS) filter for parameter estimation.

    Parameters
    ----------
    mu : float, default=0.01
        The learning rate or step size for the LMS algorithm.

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
    """

    def __init__(self, *, mu: float = 0.01, unbiased: bool = False, uiter: int = 30):
        self.mu = mu
        self.uiter = uiter
        self.unbiased = unbiased
        self._validate_params(vars(self))
        self.xi = None

    def optimize(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Parameter estimation using the Sign-Error  Least Mean Squares filter.

        The sign-error LMS algorithm uses the sign of the error vector
        to change the filter coefficients.

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

        Notes
        -----
        A more in-depth documentation of all methods for parameters estimation
        will be available soon. For now, please refer to the mentioned
        references.

        References
        ----------
        - Book: Hayes, M. H. (2009). Statistical digital signal processing
           and modeling. John Wiley & Sons.
        - Dissertation (Portuguese): Zipf, J. G. F. (2011). Classificação,
           análise estatística e novas estratégias de algoritmos LMS de passo
           variável.
        - Wikipedia entry on Least Mean Squares
           https://en.wikipedia.org/wiki/Least_mean_squares_filter

        """
        n_theta, n, theta, self.xi = self._initial_values(psi)

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
    """Normalized Least Mean Squares (ALMS) filter for parameter estimation.

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
        Estimate the model parameters using the LMS filter.
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
        self._validate_params(vars(self))
        self.xi = None

    def optimize(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Parameter estimation using the Normalized Least Mean Squares filter.

        The normalization is used to avoid numerical instability when updating
        the estimated parameters.

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

        Notes
        -----
        A more in-depth documentation of all methods for parameters estimation
        will be available soon. For now, please refer to the mentioned
        references.

        References
        ----------
        - Book: Hayes, M. H. (2009). Statistical digital signal processing
           and modeling. John Wiley & Sons.
        - Dissertation (Portuguese): Zipf, J. G. F. (2011). Classificação,
           análise estatística e novas estratégias de algoritmos LMS de passo
           variável.
        - Wikipedia entry on Least Mean Squares
           https://en.wikipedia.org/wiki/Least_mean_squares_filter

        """
        n_theta, n, theta, self.xi = self._initial_values(psi)

        for i in range(n_theta, n):
            psi_tmp = psi[i, :].reshape(-1, 1)
            self.xi[i, 0] = y[i, 0] - np.dot(psi_tmp.T, theta[:, i - 1])[0]
            tmp_list = theta[:, i - 1].reshape(-1, 1) + 2 * self.mu * self.xi[i, 0] * (
                psi_tmp / (self.eps + np.dot(psi_tmp.T, psi_tmp))
            )
            theta[:, i] = tmp_list.flatten()

        return theta[:, -1].reshape(-1, 1)


class NormalizedLeastMeanSquaresSignError(BaseEstimator):
    """Normalized Least Mean Squares SignError(NLMSSE) filter for parameter estimation.

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
        Estimate the model parameters using the LMS filter.
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
        self._validate_params(vars(self))
        self.xi = None

    def optimize(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Parameter estimation using the Normalized Sign-Error LMS filter.

        The normalization is used to avoid numerical instability when updating
        the estimated parameters and the sign of the error vector is used to
        to change the filter coefficients.

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

        Notes
        -----
        A more in-depth documentation of all methods for parameters estimation
        will be available soon. For now, please refer to the mentioned
        references.

        References
        ----------
        - Book: Hayes, M. H. (2009). Statistical digital signal processing
           and modeling. John Wiley & Sons.
        - Dissertation (Portuguese): Zipf, J. G. F. (2011). Classificação,
           análise estatística e novas estratégias de algoritmos LMS de passo
           variável.
        - Wikipedia entry on Least Mean Squares
           https://en.wikipedia.org/wiki/Least_mean_squares_filter

        """
        n_theta, n, theta, self.xi = self._initial_values(psi)

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

    Parameters
    ----------
    mu : float, default=0.01
        The learning rate or step size for the LMS algorithm.

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
    """

    def __init__(self, *, mu: float = 0.01, unbiased: bool = False, uiter: int = 30):
        self.mu = mu
        self.unbiased = unbiased
        self.uiter = uiter
        self._validate_params(vars(self))
        self.xi = None

    def optimize(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Parameter estimation using the  Sign-Regressor LMS filter.

        The sign-regressor LMS algorithm uses the sign of the matrix
        information to change the filter coefficients.

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

        Notes
        -----
        A more in-depth documentation of all methods for parameters estimation
        will be available soon. For now, please refer to the mentioned
        references.

        References
        ----------
        - Book: Hayes, M. H. (2009). Statistical digital signal processing
           and modeling. John Wiley & Sons.
        - Dissertation (Portuguese): Zipf, J. G. F. (2011). Classificação,
           análise estatística e novas estratégias de algoritmos LMS de passo
           variável.
        - Wikipedia entry on Least Mean Squares
           https://en.wikipedia.org/wiki/Least_mean_squares_filter

        """
        n_theta, n, theta, self.xi = self._initial_values(psi)

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
        Estimate the model parameters using the LMS filter.
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
        self._validate_params(vars(self))
        self.xi = None

    def optimize(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Parameter estimation using the Normalized Sign-Regressor LMS filter.

        The normalization is used to avoid numerical instability when updating
        the estimated parameters and the sign of the information matrix is
        used to change the filter coefficients.

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

        Notes
        -----
        A more in-depth documentation of all methods for parameters estimation
        will be available soon. For now, please refer to the mentioned
        references.

        References
        ----------
        .. [1] Book: Hayes, M. H. (2009). Statistical digital signal processing
           and modeling. John Wiley & Sons.
        .. [2] Dissertation (Portuguese): Zipf, J. G. F. (2011). Classificação,
           análise estatística e novas estratégias de algoritmos LMS de passo
           variável.
        .. [3] Wikipedia entry on Least Mean Squares
           https://en.wikipedia.org/wiki/Least_mean_squares_filter

        """
        n_theta, n, theta, self.xi = self._initial_values(psi)

        for i in range(n_theta, n):
            psi_tmp = psi[i, :].reshape(-1, 1)
            self.xi[i, 0] = y[i, 0] - np.dot(psi_tmp.T, theta[:, i - 1])[0]
            tmp_list = theta[:, i - 1].reshape(-1, 1) + self.mu * self.xi[i, 0] * (
                np.sign(psi_tmp) / (self.eps + np.dot(psi_tmp.T, psi_tmp))
            )
            theta[:, i] = tmp_list.flatten()

        return theta[:, -1].reshape(-1, 1)


class LeastMeanSquaresSignSign(BaseEstimator):
    """Least Mean Squares SignSign(LMSSS) filter for parameter estimation.

    Parameters
    ----------
    mu : float, default=0.01
        The learning rate or step size for the LMS algorithm.

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
    """

    def __init__(self, *, mu: float = 0.01, unbiased: bool = False, uiter: int = 30):
        self.mu = mu
        self.unbiased = unbiased
        self.uiter = uiter
        self._validate_params(vars(self))
        self.xi = None

    def optimize(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Parameter estimation using the  Sign-Sign LMS filter.

        The sign-regressor LMS algorithm uses both the sign of the matrix
        information and the sign of the error vector to change the filter
        coefficients.

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

        Notes
        -----
        A more in-depth documentation of all methods for parameters estimation
        will be available soon. For now, please refer to the mentioned
        references.

        References
        ----------
        - Book: Hayes, M. H. (2009). Statistical digital signal processing
           and modeling. John Wiley & Sons.
        - Dissertation (Portuguese): Zipf, J. G. F. (2011). Classificação,
           análise estatística e novas estratégias de algoritmos LMS de passo
           variável.
        - Wikipedia entry on Least Mean Squares
           https://en.wikipedia.org/wiki/Least_mean_squares_filter

        """
        n_theta, n, theta, self.xi = self._initial_values(psi)

        for i in range(n_theta, n):
            psi_tmp = psi[i, :].reshape(-1, 1)
            self.xi[i, 0] = y[i, 0] - np.dot(psi_tmp.T, theta[:, i - 1])[0]
            tmp_list = theta[:, i - 1].reshape(-1, 1) + 2 * self.mu * np.sign(
                self.xi[i, 0]
            ) * np.sign(psi_tmp)
            theta[:, i] = tmp_list.flatten()

        return theta[:, -1].reshape(-1, 1)


class LeastMeanSquaresNormalizedSignSign(BaseEstimator):
    """Normalized Least Mean Squares SignSign(NLMSSS) filter for parameter estimation.

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
        Estimate the model parameters using the LMS filter.
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
        self._validate_params(vars(self))
        self.xi = None

    def optimize(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Parameter estimation using the Normalized Sign-Sign LMS filter.

        The normalization is used to avoid numerical instability when updating
        the estimated parameters and both the sign of the information matrix
        and the sign of the error vector are used to change the filter
        coefficients.

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

        Notes
        -----
        A more in-depth documentation of all methods for parameters estimation
        will be available soon. For now, please refer to the mentioned
        references.

        References
        ----------
        - Book: Hayes, M. H. (2009). Statistical digital signal processing
           and modeling. John Wiley & Sons.
        - Dissertation (Portuguese): Zipf, J. G. F. (2011). Classificação,
           análise estatística e novas estratégias de algoritmos LMS de passo
           variável.
        - Wikipedia entry on Least Mean Squares
           https://en.wikipedia.org/wiki/Least_mean_squares_filter

        """
        n_theta, n, theta, self.xi = self._initial_values(psi)

        for i in range(n_theta, n):
            psi_tmp = psi[i, :].reshape(-1, 1)
            self.xi[i, 0] = y[i, 0] - np.dot(psi_tmp.T, theta[:, i - 1])[0]
            tmp_list = theta[:, i - 1].reshape(-1, 1) + 2 * self.mu * np.sign(
                self.xi[i, 0]
            ) * (np.sign(psi_tmp) / (self.eps + np.dot(psi_tmp.T, psi_tmp)))
            theta[:, i] = tmp_list.flatten()

        return theta[:, -1].reshape(-1, 1)


class LeastMeanSquaresNormalizedLeaky(BaseEstimator):
    """Normalized Least Mean Squares Leaky(NLMSL) filter for parameter estimation.

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
        Estimate the model parameters using the LMS filter.
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
        self._validate_params(vars(self))
        self.xi = None

    def optimize(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Parameter estimation using the  Normalized Leaky LMS filter.

        When the leakage factor, gama, is set to 0 then there is no
        leakage in the estimation process.

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

        Notes
        -----
        A more in-depth documentation of all methods for parameters estimation
        will be available soon. For now, please refer to the mentioned
        references.

        References
        ----------
        - Book: Hayes, M. H. (2009). Statistical digital signal processing
           and modeling. John Wiley & Sons.
        - Dissertation (Portuguese): Zipf, J. G. F. (2011). Classificação,
           análise estatística e novas estratégias de algoritmos LMS de passo
           variável.
        - Wikipedia entry on Least Mean Squares
           https://en.wikipedia.org/wiki/Least_mean_squares_filter

        """
        n_theta, n, theta, self.xi = self._initial_values(psi)

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
    """Least Mean Squares Leaky(LMSL) filter for parameter estimation.

    Parameters
    ----------
    mu : float, default=0.01
        The learning rate or step size for the LMS algorithm.
    gama : float, default=0.2
        The leakage factor of the Leaky LMS method.

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
        Estimate the model parameters using the LMS filter.
    """

    def __init__(
        self,
        *,
        mu: float = 0.01,
        gama: float = 0.2,
        unbiased: bool = False,
        uiter: int = 30,
    ):
        self.mu = mu
        self.gama = gama
        self.unbiased = unbiased
        self.uiter = uiter
        self._validate_params(vars(self))
        self.xi = None

    def optimize(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Parameter estimation using the  Leaky LMS filter.

        When the leakage factor, gama, is set to 0 then there is no
        leakage in the estimation process.

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

        Notes
        -----
        A more in-depth documentation of all methods for parameters estimation
        will be available soon. For now, please refer to the mentioned
        references.

        References
        ----------
        - Book: Hayes, M. H. (2009). Statistical digital signal processing
           and modeling. John Wiley & Sons.
        - Dissertation (Portuguese): Zipf, J. G. F. (2011). Classificação,
           análise estatística e novas estratégias de algoritmos LMS de passo
           variável.
        - Wikipedia entry on Least Mean Squares
           https://en.wikipedia.org/wiki/Least_mean_squares_filter

        """
        n_theta, n, theta, self.xi = self._initial_values(psi)

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
    """Least Mean Squares Fourth(LMSF) filter for parameter estimation.

    Parameters
    ----------
    mu : float, default=0.01
        The learning rate or step size for the LMS algorithm.

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
    """

    def __init__(self, *, mu: float = 0.01, unbiased: bool = False, uiter: int = 30):
        self.mu = mu
        self.unbiased = unbiased
        self.uiter = uiter
        self._validate_params(vars(self))
        self.xi = None

    def optimize(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Parameter estimation using the  LMS Fourth filter.

        When the leakage factor, gama, is set to 0 then there is no
        leakage in the estimation process.

        Parameters
        ----------
        psi : ndarray of floats
            The information matrix of the model.
        y : ndarray of floats of shape = y_training
            The data used to training the model.

        Returns
        -------
        theta : ndarray of floats of shape = number_of_model_elements
            The estimated parameters of the model.

        Notes
        -----
        A more in-depth documentation of all methods for parameters estimation
        will be available soon. For now, please refer to the mentioned
        references.

        References
        ----------
        - Book: Hayes, M. H. (2009). Statistical digital signal processing
           and modeling. John Wiley & Sons.
        - Dissertation (Portuguese): Zipf, J. G. F. (2011). Classificação,
           análise estatística e novas estratégias de algoritmos LMS de passo
           variável.
        - Manuscript:Gui, G., Mehbodniya, A., & Adachi, F. (2013).
           Least mean square/fourth algorithm with application to sparse
           channel estimation. arXiv preprint arXiv:1304.3911.
           https://arxiv.org/pdf/1304.3911.pdf
        - Manuscript: Nascimento, V. H., & Bermudez, J. C. M. (2005, March).
           When is the least-mean fourth algorithm mean-square stable?
           In Proceedings.(ICASSP'05). IEEE International Conference on
           Acoustics, Speech, and Signal Processing, 2005.
           (Vol. 4, pp. iv-341). IEEE.
           http://www.lps.usp.br/vitor/artigos/icassp05.pdf
        - Wikipedia entry on Least Mean Squares
           https://en.wikipedia.org/wiki/Least_mean_squares_filter

        """
        n_theta, n, theta, self.xi = self._initial_values(psi)

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
        The weight factor for mixed-norm control. Weight factor to control the
        proportions of the error norms and offers an extra degree of freedom within
        the adaptation of the LMS mixed norm method.

    Attributes
    ----------
    mu : float
        The adaptation step size.
    weight : float
        The weight factor for mixed-norm control. Weight factor to control the
        proportions of the error norms and offers an extra degree of freedom within
        the adaptation of the LMS mixed norm method.
    xi : ndarray or None
        The error signal, initialized to None.
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
        self._validate_params(vars(self))
        self.xi = None

    def optimize(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Parameter estimation using the Mixed-norm LMS filter.

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

        Notes
        -----
        A more in-depth documentation of all methods for parameters estimation
        will be available soon. For now, please refer to the mentioned
        references.

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
        n_theta, n, theta, self.xi = self._initial_values(psi)

        for i in range(n_theta, n):
            psi_tmp = psi[i, :].reshape(-1, 1)
            self.xi[i, 0] = y[i, 0] - np.dot(psi_tmp.T, theta[:, i - 1])[0]
            tmp_list = theta[:, i - 1].reshape(-1, 1) + self.mu * psi_tmp * self.xi[
                i, 0
            ] * (self.weight + (1 - self.weight) * self.xi[i, 0] ** 2)
            theta[:, i] = tmp_list.flatten()

        return theta[:, -1].reshape(-1, 1)
