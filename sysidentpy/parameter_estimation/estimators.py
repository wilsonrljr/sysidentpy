""" Least Squares Methods for parameter estimation """

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause


import numpy as np
import warnings
from ..narmax_base import InformationMatrix


class Estimators(InformationMatrix):
    """Ordinary Least Squares for linear parameter estimation"""

    def __init__(
        self,
        max_lag=1,
        lam=0.98,
        delta=0.01,
        offset_covariance=0.2,
        mu=0.01,
        eps=np.finfo(np.float64).eps,
        gama=0.2,
        weight=0.02,
    ):

        self._eps = eps
        self._mu = mu
        self._offset_covariance = offset_covariance
        self.max_lag = max_lag
        self._lam = lam
        self._delta = delta
        self._gama = gama
        self._weight = weight  # <0  e <1
        self._validate_params()

    def _validate_params(self):
        """Validate input params."""
        attributes = {
            "max_lag": self.max_lag,
            "lam": self._lam,
            "delta": self._delta,
            "offset_covariance": self._offset_covariance,
            "mu": self._mu,
            "eps": self._eps,
            "gama": self._gama,
            "weight": self._weight,
        }
        for attribute, value in attributes.items():
            if not isinstance(value, (np.integer, int, float)):
                raise ValueError(
                    (
                        f"{attribute} must be int or float (positive)."
                        f"Got {type(attribute)}"
                    )
                )

            if attribute in ["lam", "weight", "offset_covariance"]:
                if value > 1 or value < 0:
                    raise ValueError(
                        f"{attribute} must lies on [0 1] range. Got {value}"
                    )

            if value < 0:
                raise ValueError(
                    (
                        f"{attribute} must be positive. Got {value}"
                        f"Check the documentation for allowed values"
                    )
                )

    def _check_linear_dependence_rows(self, psi):
        if np.linalg.matrix_rank(psi) != psi.shape[1]:
            warnings.warn(
                (
                    "Psi matrix might have linearly dependent rows."
                    "Be careful and check your data"
                ),
                stacklevel=2,
            )

    def least_squares(self, psi, y):
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

        References
        ----------
        .. [1] Manuscript: Sorenson, H. W. (1970). Least-squares estimation:
           from Gauss to Kalman. IEEE spectrum, 7(7), 63-68.
           http://pzs.dstu.dp.ua/DataMining/mls/bibl/Gauss2Kalman.pdf
        .. [2] Book (Portuguese): Aguirre, L. A. (2007). Introdução identificação
           de sistemas: técnicas lineares e não-lineares aplicadas a sistemas
           reais. Editora da UFMG. 3a edição.
        .. [3] Manuscript: Markovsky, I., & Van Huffel, S. (2007).
           Overview of total least-squares methods.
           Signal processing, 87(10), 2283-2302.
           https://eprints.soton.ac.uk/263855/1/tls_overview.pdf
        .. [4] Wikipedia entry on Least Squares
           https://en.wikipedia.org/wiki/Least_squares

        """
        self._check_linear_dependence_rows(psi)

        y = y[self.max_lag :, 0].reshape(-1, 1)
        theta = np.linalg.lstsq(psi, y, rcond=None)[0]
        return theta

    def _unbiased_estimator(self, psi, y, theta, non_degree, elag, max_lag):
        """Estimate the model parameters using Extended Least Squares method.

        Parameters
        ----------
        psi : ndarray of floats
            The information matrix of the model.
        X : ndarray of floats
            The input data to be used in the training process.
        y : array-like of shape = y_training
            The data used to training the model.
        biased_theta : array-like of shape = number_of_model_elements
            The estimated biased parameters of the model.

        Returns
        -------
        theta : array-like of shape = number_of_model_elements
            The estimated unbiased parameters of the model.

        References
        ----------
        .. [1] Manuscript: Sorenson, H. W. (1970). Least-squares estimation:
           from Gauss to Kalman. IEEE spectrum, 7(7), 63-68.
           http://pzs.dstu.dp.ua/DataMining/mls/bibl/Gauss2Kalman.pdf
        .. [2] Book (Portuguese): Aguirre, L. A. (2007). Introdução a identificação
           de sistemas: técnicas lineares e não-lineares aplicadas a sistemas
           reais. Editora da UFMG. 3a edição.
        .. [3] Manuscript: Markovsky, I., & Van Huffel, S. (2007).
           Overview of total least-squares methods.
           Signal processing, 87(10), 2283-2302.
            https://eprints.soton.ac.uk/263855/1/tls_overview.pdf
        .. [4] Wikipedia entry on Least Squares
           https://en.wikipedia.org/wiki/Least_squares

        """
        e = y[max_lag:, 0].reshape(-1, 1) - np.dot(psi, theta)
        for i in range(30):
            e = np.concatenate([np.zeros([max_lag, 1]), e], axis=0)

            lagged_data = self.build_output_matrix(e, elag)

            e_regressors = self.basis_function.fit(
                lagged_data, max_lag, predefined_regressors=None
            )

            psi_extended = np.concatenate([psi, e_regressors], axis=1)
            unbiased_theta = getattr(self, self.estimator)(psi_extended, y)
            e = y[max_lag:, 0].reshape(-1, 1) - np.dot(
                psi_extended, unbiased_theta.reshape(-1, 1)
            )

        return unbiased_theta[0 : theta.shape[0], 0].reshape(-1, 1)

    def total_least_squares(self, psi, y):
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

        References
        ----------
        .. [1] Manuscript: Golub, G. H., & Van Loan, C. F. (1980).
           An analysis of the total least squares problem.
           SIAM journal on numerical analysis, 17(6), 883-893.
        .. [2] Manuscript: Markovsky, I., & Van Huffel, S. (2007).
           Overview of total least-squares methods.
           Signal processing, 87(10), 2283-2302.
           https://eprints.soton.ac.uk/263855/1/tls_overview.pdf
        .. [3] Wikipedia entry on Total Least Squares
           https://en.wikipedia.org/wiki/Total_least_squares

        """
        y = y[self.max_lag :, 0].reshape(-1, 1)
        full = np.hstack((psi, y))
        n = psi.shape[1]
        u, s, v = np.linalg.svd(full, full_matrices=True)
        theta = -v.T[:n, n:] / v.T[n:, n:]
        return theta.reshape(-1, 1)

    def _initial_values(self, y, psi):
        y = y[self.max_lag :, 0].reshape(-1, 1)
        n_theta = psi.shape[1]
        n = len(psi)
        theta = np.zeros([n_theta, n])
        xi = np.zeros([n, 1])
        return y, n_theta, n, theta, xi

    def recursive_least_squares(self, psi, y):
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
        .. [1] Book (Portuguese): Aguirre, L. A. (2007). Introdução identificação
           de sistemas: técnicas lineares e não-lineares aplicadas a sistemas
           reais. Editora da UFMG. 3a edição.

        """
        y, n_theta, n, theta, self.xi = self._initial_values(y, psi)

        p = np.eye(n_theta) / self._delta

        for i in range(2, n):
            psi_tmp = psi[i, :].reshape(-1, 1)
            k_numerator = self._lam ** (-1) * p.dot(psi_tmp)
            k_denominator = 1 + self._lam ** (-1) * psi_tmp.T.dot(p).dot(psi_tmp)
            k = np.divide(k_numerator, k_denominator)
            self.xi[i, 0] = y[i, 0] - np.dot(psi_tmp.T, theta[:, i - 1])
            theta[:, i] = list(theta[:, i - 1].reshape(-1, 1) + k.dot(self.xi[i, 0]))

            p1 = p.dot(psi[i, :].reshape(-1, 1)).dot(psi[i, :].reshape(-1, 1).T).dot(p)
            p2 = (
                psi[i, :].reshape(-1, 1).T.dot(p).dot(psi[i, :].reshape(-1, 1))
                + self._lam
            )

            p_numerator = p - np.divide(p1, p2)
            p = np.divide(p_numerator, self._lam)

        self.theta_evolution = theta.copy()
        return theta[:, -1].reshape(-1, 1)

    def affine_least_mean_squares(self, psi, y):
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
        .. [1] Book: Poularikas, A. D. (2017). Adaptive filtering: Fundamentals
           of least mean squares with MATLAB®. CRC Press.

        """
        y, n_theta, n, theta, self.xi = self._initial_values(y, psi)

        for i in range(n_theta, n):
            self.xi = y - psi.dot(theta[:, i - 1].reshape(-1, 1))
            aux = (
                self._mu
                * psi
                @ np.linalg.pinv(
                    psi.T @ psi + self._offset_covariance * np.eye(n_theta)
                )
            )
            theta[:, i] = list(theta[:, i - 1].reshape(-1, 1) + aux.T.dot(self.xi))

        return theta[:, -1].reshape(-1, 1)

    def least_mean_squares(self, psi, y):
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
        .. [1] Book: Haykin, S., & Widrow, B. (Eds.). (2003). Least-mean-square
           adaptive filters (Vol. 31). John Wiley & Sons.
        .. [2] Dissertation (Portuguese): Zipf, J. G. F. (2011). Classificação,
           análise estatística e novas estratégias de algoritmos LMS de passo
           variável.
        .. [3] Wikipedia entry on Least Mean Squares
           https://en.wikipedia.org/wiki/Least_mean_squares_filter

        """
        y, n_theta, n, theta, self.xi = self._initial_values(y, psi)

        for i in range(n_theta, n):
            psi_tmp = psi[i, :].reshape(-1, 1)
            self.xi[i, 0] = y[i, 0] - np.dot(psi_tmp.T, theta[:, i - 1])
            theta[:, i] = list(
                theta[:, i - 1].reshape(-1, 1) + 2 * self._mu * self.xi[i, 0] * psi_tmp
            )

        return theta[:, -1].reshape(-1, 1)

    def least_mean_squares_sign_error(self, psi, y):
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
        .. [1] `Book: Hayes, M. H. (2009). Statistical digital signal processing
           and modeling. John Wiley & Sons.
        .. [2] `Dissertation (Portuguese): Zipf, J. G. F. (2011). Classificação,
           análise estatística e novas estratégias de algoritmos LMS de passo
           variável.
        .. [3] Wikipedia entry on Least Mean Squares
           https://en.wikipedia.org/wiki/Least_mean_squares_filter

        """
        y, n_theta, n, theta, self.xi = self._initial_values(y, psi)

        for i in range(n_theta, n):
            psi_tmp = psi[i, :].reshape(-1, 1)
            self.xi[i, 0] = y[i, 0] - np.dot(psi_tmp.T, theta[:, i - 1])
            theta[:, i] = list(
                theta[:, i - 1].reshape(-1, 1)
                + self._mu * np.sign(self.xi[i, 0]) * psi_tmp
            )

        return theta[:, -1].reshape(-1, 1)

    def normalized_least_mean_squares(self, psi, y):
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
        .. [1] `Book: Hayes, M. H. (2009). Statistical digital signal processing
           and modeling. John Wiley & Sons.
        .. [2] Dissertation (Portuguese): Zipf, J. G. F. (2011). Classificação,
           análise estatística e novas estratégias de algoritmos LMS de passo
           variável.
        .. [3] Wikipedia entry on Least Mean Squares
           https://en.wikipedia.org/wiki/Least_mean_squares_filter

        """
        y, n_theta, n, theta, self.xi = self._initial_values(y, psi)

        for i in range(n_theta, n):
            psi_tmp = psi[i, :].reshape(-1, 1)
            self.xi[i, 0] = y[i, 0] - np.dot(psi_tmp.T, theta[:, i - 1])
            theta[:, i] = list(
                theta[:, i - 1].reshape(-1, 1)
                + 2
                * self._mu
                * self.xi[i, 0]
                * (psi_tmp / (self._eps + np.dot(psi_tmp.T, psi_tmp)))
            )

        return theta[:, -1].reshape(-1, 1)

    def least_mean_squares_normalized_sign_error(self, psi, y):
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
        .. [1] Book: Hayes, M. H. (2009). Statistical digital signal processing
           and modeling. John Wiley & Sons.
        .. [2] Dissertation (Portuguese): Zipf, J. G. F. (2011). Classificação,
           análise estatística e novas estratégias de algoritmos LMS de passo
           variável.
        .. [3] Wikipedia entry on Least Mean Squares
           https://en.wikipedia.org/wiki/Least_mean_squares_filter

        """
        y, n_theta, n, theta, self.xi = self._initial_values(y, psi)

        for i in range(n_theta, n):
            psi_tmp = psi[i, :].reshape(-1, 1)
            self.xi[i, 0] = y[i, 0] - np.dot(psi_tmp.T, theta[:, i - 1])
            theta[:, i] = list(
                theta[:, i - 1].reshape(-1, 1)
                + 2
                * self._mu
                * np.sign(self.xi[i, 0])
                * (psi_tmp / (self._eps + np.dot(psi_tmp.T, psi_tmp)))
            )

        return theta[:, -1].reshape(-1, 1)

    def least_mean_squares_sign_regressor(self, psi, y):
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
        .. [1] Book: Hayes, M. H. (2009). Statistical digital signal processing
           and modeling. John Wiley & Sons.
        .. [2] Dissertation (Portuguese): Zipf, J. G. F. (2011). Classificação,
           análise estatística e novas estratégias de algoritmos LMS de passo
           variável.
        .. [3] Wikipedia entry on Least Mean Squares
           https://en.wikipedia.org/wiki/Least_mean_squares_filter

        """
        y, n_theta, n, theta, self.xi = self._initial_values(y, psi)

        for i in range(n_theta, n):
            psi_tmp = psi[i, :].reshape(-1, 1)
            self.xi[i, 0] = y[i, 0] - np.dot(psi_tmp.T, theta[:, i - 1])
            theta[:, i] = list(
                theta[:, i - 1].reshape(-1, 1)
                + self._mu * self.xi[i, 0] * np.sign(psi_tmp)
            )

        return theta[:, -1].reshape(-1, 1)

    def least_mean_squares_normalized_sign_regressor(self, psi, y):
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
        y, n_theta, n, theta, self.xi = self._initial_values(y, psi)

        for i in range(n_theta, n):
            psi_tmp = psi[i, :].reshape(-1, 1)
            self.xi[i, 0] = y[i, 0] - np.dot(psi_tmp.T, theta[:, i - 1])
            theta[:, i] = list(
                theta[:, i - 1].reshape(-1, 1)
                + self._mu
                * self.xi[i, 0]
                * (np.sign(psi_tmp) / (self._eps + np.dot(psi_tmp.T, psi_tmp)))
            )

        return theta[:, -1].reshape(-1, 1)

    def least_mean_squares_sign_sign(self, psi, y):
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
        .. [1] Book: Hayes, M. H. (2009). Statistical digital signal processing
           and modeling. John Wiley & Sons.
        .. [2] Dissertation (Portuguese): Zipf, J. G. F. (2011). Classificação,
           análise estatística e novas estratégias de algoritmos LMS de passo
           variável.
        .. [3] Wikipedia entry on Least Mean Squares
           https://en.wikipedia.org/wiki/Least_mean_squares_filter

        """
        y, n_theta, n, theta, self.xi = self._initial_values(y, psi)

        for i in range(n_theta, n):
            psi_tmp = psi[i, :].reshape(-1, 1)
            self.xi[i, 0] = y[i, 0] - np.dot(psi_tmp.T, theta[:, i - 1])
            theta[:, i] = list(
                theta[:, i - 1].reshape(-1, 1)
                + 2 * self._mu * np.sign(self.xi[i, 0]) * np.sign(psi_tmp)
            )

        return theta[:, -1].reshape(-1, 1)

    def least_mean_squares_normalized_sign_sign(self, psi, y):
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
        .. [1] Book: Hayes, M. H. (2009). Statistical digital signal processing
           and modeling. John Wiley & Sons.
        .. [2] Dissertation (Portuguese): Zipf, J. G. F. (2011). Classificação,
           análise estatística e novas estratégias de algoritmos LMS de passo
           variável.
        .. [3] Wikipedia entry on Least Mean Squares
           https://en.wikipedia.org/wiki/Least_mean_squares_filter

        """
        y, n_theta, n, theta, self.xi = self._initial_values(y, psi)

        for i in range(n_theta, n):
            psi_tmp = psi[i, :].reshape(-1, 1)
            self.xi[i, 0] = y[i, 0] - np.dot(psi_tmp.T, theta[:, i - 1])
            theta[:, i] = list(
                theta[:, i - 1].reshape(-1, 1)
                + 2
                * self._mu
                * np.sign(self.xi[i, 0])
                * (np.sign(psi_tmp) / (self._eps + np.dot(psi_tmp.T, psi_tmp)))
            )

        return theta[:, -1].reshape(-1, 1)

    def least_mean_squares_normalized_leaky(self, psi, y):
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
        .. [1] Book: Hayes, M. H. (2009). Statistical digital signal processing
           and modeling. John Wiley & Sons.
        .. [2] Dissertation (Portuguese): Zipf, J. G. F. (2011). Classificação,
           análise estatística e novas estratégias de algoritmos LMS de passo
           variável.
        .. [3] Wikipedia entry on Least Mean Squares
           https://en.wikipedia.org/wiki/Least_mean_squares_filter

        """
        y, n_theta, n, theta, self.xi = self._initial_values(y, psi)

        for i in range(n_theta, n):
            psi_tmp = psi[i, :].reshape(-1, 1)
            self.xi[i, 0] = y[i, 0] - np.dot(psi_tmp.T, theta[:, i - 1])
            theta[:, i] = list(
                theta[:, i - 1].reshape(-1, 1) * (1 - self._mu * self._gama)
                + self._mu
                * self.xi[i, 0]
                * psi_tmp
                / (self._eps + np.dot(psi_tmp.T, psi_tmp))
            )

        return theta[:, -1].reshape(-1, 1)

    def least_mean_squares_leaky(self, psi, y):
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
        .. [1] Book: Hayes, M. H. (2009). Statistical digital signal processing
           and modeling. John Wiley & Sons.
        .. [2] Dissertation (Portuguese): Zipf, J. G. F. (2011). Classificação,
           análise estatística e novas estratégias de algoritmos LMS de passo
           variável.
        .. [3] Wikipedia entry on Least Mean Squares
           https://en.wikipedia.org/wiki/Least_mean_squares_filter

        """
        y, n_theta, n, theta, self.xi = self._initial_values(y, psi)

        for i in range(n_theta, n):
            psi_tmp = psi[i, :].reshape(-1, 1)
            self.xi[i, 0] = y[i, 0] - np.dot(psi_tmp.T, theta[:, i - 1])
            theta[:, i] = list(
                theta[:, i - 1].reshape(-1, 1) * (1 - self._mu * self._gama)
                + self._mu * self.xi[i, 0] * psi_tmp
            )

        return theta[:, -1].reshape(-1, 1)

    def least_mean_squares_fourth(self, psi, y):
        """Parameter estimation using the  LMS Fourth filter.

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
        .. [1] Book: Hayes, M. H. (2009). Statistical digital signal processing
           and modeling. John Wiley & Sons.
        .. [2] Dissertation (Portuguese): Zipf, J. G. F. (2011). Classificação,
           análise estatística e novas estratégias de algoritmos LMS de passo
           variável.
        .. [3] Manuscript:Gui, G., Mehbodniya, A., & Adachi, F. (2013).
           Least mean square/fourth algorithm with application to sparse
           channel estimation. arXiv preprint arXiv:1304.3911.
           https://arxiv.org/pdf/1304.3911.pdf
        .. [4] Manuscript: Nascimento, V. H., & Bermudez, J. C. M. (2005, March).
           When is the least-mean fourth algorithm mean-square stable?
           In Proceedings.(ICASSP'05). IEEE International Conference on
           Acoustics, Speech, and Signal Processing, 2005.
           (Vol. 4, pp. iv-341). IEEE.
           http://www.lps.usp.br/vitor/artigos/icassp05.pdf
        .. [5] Wikipedia entry on Least Mean Squares
           https://en.wikipedia.org/wiki/Least_mean_squares_filter

        """
        y, n_theta, n, theta, self.xi = self._initial_values(y, psi)

        for i in range(n_theta, n):
            psi_tmp = psi[i, :].reshape(-1, 1)
            self.xi[i, 0] = y[i, 0] - np.dot(psi_tmp.T, theta[:, i - 1])
            theta[:, i] = list(
                theta[:, i - 1].reshape(-1, 1) + self._mu * psi_tmp * self.xi[i, 0] ** 3
            )

        return theta[:, -1].reshape(-1, 1)

    def least_mean_squares_mixed_norm(self, psi, y):
        """Parameter estimation using the Mixed-norm LMS filter.

        The weight factor controls the proportions of the error norms
        and offers an extra degree of freedom within the adaptation.

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
        .. [1] Chambers, J. A., Tanrikulu, O., & Constantinides, A. G. (1994).
           Least mean mixed-norm adaptive filtering.
           Electronics letters, 30(19), 1574-1575.
           https://ieeexplore.ieee.org/document/326382
        .. [2] Dissertation (Portuguese): Zipf, J. G. F. (2011). Classificação,
           análise estatística e novas estratégias de algoritmos LMS de passo
           variável.
        .. [3] Wikipedia entry on Least Mean Squares
           https://en.wikipedia.org/wiki/Least_mean_squares_filter

        """
        y, n_theta, n, theta, self.xi = self._initial_values(y, psi)

        for i in range(n_theta, n):
            psi_tmp = psi[i, :].reshape(-1, 1)
            self.xi[i, 0] = y[i, 0] - np.dot(psi_tmp.T, theta[:, i - 1])
            theta[:, i] = list(
                theta[:, i - 1].reshape(-1, 1)
                + self._mu
                * psi_tmp
                * self.xi[i, 0]
                * (self._weight + (1 - self._weight) * self.xi[i, 0] ** 2)
            )

        return theta[:, -1].reshape(-1, 1)
