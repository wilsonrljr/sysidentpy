""" Build Polynomial NARMAX Models using the Entropic Regression algorithm """

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause

import warnings
from typing import Union

import numpy as np
from numpy import linalg as LA
from scipy.spatial.distance import cdist
from scipy.special import psi

from ..narmax_base import BaseMSS
from ..basis_function import Fourier, Polynomial
from ..parameter_estimation.estimators import Estimators
from ..utils._check_arrays import _check_positive_int, _num_features, check_random_state
from ..utils.deprecation import deprecated


@deprecated(
    version="v0.3.0",
    future_version="v0.4.0",
    message=(
        "Passing a string to define the estimator will rise an error in v0.4.0."
        " \n You'll have to use ER(estimator=LeastSquares()) instead. \n The"
        " only change is that you'll have to define the estimator first instead"
        " of passing a string like 'least_squares'. \n This change will make"
        " easier to implement new estimators and it'll improve code"
        " readability."
    ),
)
class ER(Estimators, BaseMSS):
    r"""Entropic Regression Algorithm

    Build Polynomial NARMAX model using the Entropic Regression Algorithm ([1]_).
    This algorithm is based on the Matlab package available on:
    https://github.com/almomaa/ERFit-Package

    The NARMAX model is described as:

    $$
        y_k= F^\ell[y_{k-1}, \dotsc, y_{k-n_y},x_{k-d}, x_{k-d-1}, \dotsc, x_{k-d-n_x},
        e_{k-1}, \dotsc, e_{k-n_e}] + e_k
    $$

    where $n_y\in \mathbb{N}^*$, $n_x \in \mathbb{N}$, $n_e \in \mathbb{N}$,
    are the maximum lags for the system output and input respectively;
    $x_k \in \mathbb{R}^{n_x}$ is the system input and $y_k \in \mathbb{R}^{n_y}$
    is the system output at discrete time $k \in \mathbb{N}^n$;
    $e_k \in \mathbb{R}^{n_e}$ stands for uncertainties and possible noise
    at discrete time $k$. In this case, $\mathcal{F}^\ell$ is some nonlinear function
    of the input and output regressors with nonlinearity degree $\ell \in \mathbb{N}$
    and $d$ is a time delay typically set to $d=1$.

    Parameters
    ----------
    ylag : int, default=2
        The maximum lag of the output.
    xlag : int, default=2
        The maximum lag of the input.
    k : int, default=2
        The kth nearest neighbor to be used in estimation.
    q : float, default=0.99
        Quantile to compute, which must be between 0 and 1 inclusive.
    p : default=inf,
        Lp Measure of the distance in Knn estimator.
    n_perm: int, default=200
        Number of permutation to be used in shuffle test
    estimator : str, default="least_squares"
        The parameter estimation method.
    skip_forward = bool, default=False
        To be used for difficult and highly uncertain problems.
        Skipping the forward selection results in more accurate solution,
        but comes with higher computational cost.
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
    >>> from sysidentpy.model_structure_selection import ER
    >>> from sysidentpy.basis_function._basis_function import Polynomial
    >>> from sysidentpy.utils.display_results import results
    >>> from sysidentpy.metrics import root_relative_squared_error
    >>> from sysidentpy.utils.generate_data import get_miso_data, get_siso_data
    >>> x_train, x_valid, y_train, y_valid = get_siso_data(n=1000,
    ...                                                    colored_noise=True,
    ...                                                    sigma=0.2,
    ...                                                    train_percentage=90)
    >>> basis_function = Polynomial(degree=2)
    >>> model = ER(basis_function=basis_function,
    ...              ylag=2, xlag=2
    ...              )
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
    - Abd AlRahman R. AlMomani, Jie Sun, and Erik Bollt. How Entropic
        Regression Beats the Outliers Problem in Nonlinear System
        Identification. Chaos 30, 013107 (2020).
    - Alexander Kraskov, Harald St¨ogbauer, and Peter Grassberger.
        Estimating mutual information. Physical Review E, 69:066-138,2004
    - Alexander Kraskov, Harald St¨ogbauer, and Peter Grassberger.
        Estimating mutual information. Physical Review E, 69:066-138,2004
    - Alexander Kraskov, Harald St¨ogbauer, and Peter Grassberger.
        Estimating mutual information. Physical Review E, 69:066-138,2004

    """

    def __init__(
        self,
        *,
        ylag: Union[int, list] = 1,
        xlag: Union[int, list] = 1,
        q: float = 0.99,
        estimator: str = "least_squares",
        extended_least_squares: bool = False,
        h: float = 0.01,
        k: int = 2,
        mutual_information_estimator: str = "mutual_information_knn",
        n_perm: int = 200,
        p: Union[float, int] = np.inf,
        skip_forward: bool = False,
        lam: float = 0.98,
        delta: float = 0.01,
        offset_covariance: float = 0.2,
        mu: float = 0.01,
        eps: float = np.finfo(np.float64).eps,
        gama: float = 0.2,
        weight: float = 0.02,
        model_type: str = "NARMAX",
        basis_function: Union[Polynomial, Fourier] = Polynomial(),
        random_state: Union[int, None] = None,
    ):
        self.basis_function = basis_function
        self.model_type = model_type
        self.build_matrix = self.get_build_io_method(model_type)
        self.xlag = xlag
        self.ylag = ylag
        self.non_degree = basis_function.degree
        self.max_lag = self._get_max_lag()
        self.k = k
        self.estimator = estimator
        self.extended_least_squares = extended_least_squares
        self.q = q
        self.h = h
        self.mutual_information_estimator = mutual_information_estimator
        self.n_perm = n_perm
        self.p = p
        self.skip_forward = skip_forward
        self.random_state = random_state
        self.rng = check_random_state(random_state)
        self.tol = None
        self.ensemble = None
        self.n_inputs = None
        self.estimated_tolerance = None
        self.regressor_code = None
        self.final_model = None
        self.theta = None
        self.n_terms = None
        self.err = None
        self.pivv = None
        self._validate_params()
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

    def _validate_params(self):
        """Validate input params."""
        if isinstance(self.ylag, int) and self.ylag < 1:
            raise ValueError(f"ylag must be integer and > zero. Got {self.ylag}")

        if isinstance(self.xlag, int) and self.xlag < 1:
            raise ValueError(f"xlag must be integer and > zero. Got {self.xlag}")

        if not isinstance(self.xlag, (int, list)):
            raise ValueError(f"xlag must be integer and > zero. Got {self.xlag}")

        if not isinstance(self.ylag, (int, list)):
            raise ValueError(f"ylag must be integer and > zero. Got {self.ylag}")

        if not isinstance(self.k, int) or self.k < 1:
            raise ValueError(f"k must be integer and > zero. Got {self.k}")

        if not isinstance(self.n_perm, int) or self.n_perm < 1:
            raise ValueError(f"n_perm must be integer and > zero. Got {self.n_perm}")

        if not isinstance(self.q, float) or self.q > 1 or self.q <= 0:
            raise ValueError(
                f"q must be float and must be between 0 and 1 inclusive. Got {self.q}"
            )

        if not isinstance(self.skip_forward, bool):
            raise TypeError(
                f"skip_forward must be False or True. Got {self.skip_forward}"
            )

        if not isinstance(self.extended_least_squares, bool):
            raise TypeError(
                "extended_least_squares must be False or True. Got"
                f" {self.extended_least_squares}"
            )

        if self.model_type not in ["NARMAX", "NAR", "NFIR"]:
            raise ValueError(
                f"model_type must be NARMAX, NAR or NFIR. Got {self.model_type}"
            )

    def mutual_information_knn(self, y, y_perm):
        """Finds the mutual information.

        Finds the mutual information between $x$ and $y$ given $z$.

        This code is based on Matlab Entropic Regression package.

        Parameters
        ----------
        y : ndarray of floats
            The source signal.
        y_perm : ndarray of floats
            The destination signal.

        Returns
        -------
        ksg_estimation : float
            The conditioned mutual information.

        References
        ----------
        - Abd AlRahman R. AlMomani, Jie Sun, and Erik Bollt. How Entropic
            Regression Beats the Outliers Problem in Nonlinear System
            Identification. Chaos 30, 013107 (2020).
        - Alexander Kraskov, Harald St¨ogbauer, and Peter Grassberger.
            Estimating mutual information. Physical Review E, 69:066-138,2004
        - Alexander Kraskov, Harald St¨ogbauer, and Peter Grassberger.
            Estimating mutual information. Physical Review E, 69:066-138,2004
        - Alexander Kraskov, Harald St¨ogbauer, and Peter Grassberger.
            Estimating mutual information. Physical Review E, 69:066-138,2004

        """
        joint_space = np.concatenate([y, y_perm], axis=1)
        smallest_distance = np.sort(
            cdist(joint_space, joint_space, "minkowski", p=self.p).T
        )
        idx = np.argpartition(smallest_distance[-1, :], self.k + 1)[: self.k + 1]
        smallest_distance = smallest_distance[:, idx]
        epsilon = smallest_distance[:, -1].reshape(-1, 1)
        smallest_distance_y = cdist(y, y, "minkowski", p=self.p)
        less_than_array_nx = np.array((smallest_distance_y < epsilon)).astype(int)
        nx = (np.sum(less_than_array_nx, axis=1) - 1).reshape(-1, 1)
        smallest_distance_y_perm = cdist(y_perm, y_perm, "minkowski", p=self.p)
        less_than_array_ny = np.array((smallest_distance_y_perm < epsilon)).astype(int)
        ny = (np.sum(less_than_array_ny, axis=1) - 1).reshape(-1, 1)
        arr = psi(nx + 1) + psi(ny + 1)
        ksg_estimation = (
            psi(self.k) + psi(y.shape[0]) - np.nanmean(arr[np.isfinite(arr)])
        )
        return ksg_estimation

    def entropic_regression_backward(self, reg_matrix, y, piv):
        """Entropic Regression Backward Greedy Feature Elimination.

        This algorithm is based on the Matlab package available on:
        https://github.com/almomaa/ERFit-Package

        Parameters
        ----------
        reg_matrix : ndarray of floats
            The input data to be used in the prediction process.
        y : ndarray of floats
            The output data to be used in the prediction process.
        piv : ndarray of ints
            The set of indices to investigate

        Returns
        -------
        piv : ndarray of ints
            The set of remaining indices after the
            Backward Greedy Feature Elimination.

        """
        min_value = -np.inf
        piv = np.array(piv)
        ix = []
        while (min_value <= self.tol) and (len(piv) > 1):
            initial_array = np.full((1, len(piv)), np.inf)
            for i in range(initial_array.shape[1]):
                if piv[i] not in []:  # if you want to keep any regressor
                    rem = np.setdiff1d(piv, piv[i])
                    f1 = reg_matrix[:, piv] @ LA.pinv(reg_matrix[:, piv]) @ y
                    f2 = reg_matrix[:, rem] @ LA.pinv(reg_matrix[:, rem]) @ y
                    initial_array[0, i] = self.conditional_mutual_information(y, f1, f2)

            ix = np.argmin(initial_array)
            min_value = initial_array[0, ix]
            piv = np.delete(piv, ix)

        return piv

    def entropic_regression_forward(self, reg_matrix, y):
        """Entropic Regression Forward Greedy Feature Selection.

        This algorithm is based on the Matlab package available on:
        https://github.com/almomaa/ERFit-Package

        Parameters
        ----------
        reg_matrix : ndarray of floats
            The input data to be used in the prediction process.
        y : ndarray of floats
            The output data to be used in the prediction process.

        Returns
        -------
        selected_terms : ndarray of ints
            The set of selected regressors after the
            Forward Greedy Feature Selection.
        success : boolean
            Indicate if the forward selection succeed.
            If high degree of uncertainty is detected, and many parameters are
            selected, the success flag will be set to false. Then, the
            backward elimination will be applied for all indices.

        """
        success = True
        ix = []
        selected_terms = []
        reg_matrix_columns = np.array(list(range(reg_matrix.shape[1])))
        self.tol = self.tolerance_estimator(y)
        ksg_max = getattr(self, self.mutual_information_estimator)(
            y, reg_matrix @ LA.pinv(reg_matrix) @ y
        )
        stop_criteria = False
        while stop_criteria is False:
            selected_terms = np.ravel(
                [*selected_terms, *np.array([reg_matrix_columns[ix]])]
            )
            if len(selected_terms) != 0:
                ksg_local = getattr(self, self.mutual_information_estimator)(
                    y,
                    reg_matrix[:, selected_terms]
                    @ LA.pinv(reg_matrix[:, selected_terms])
                    @ y,
                )
            else:
                ksg_local = getattr(self, self.mutual_information_estimator)(
                    y, np.zeros_like(y)
                )

            initial_vector = np.full((1, reg_matrix.shape[1]), -np.inf)
            for i in range(reg_matrix.shape[1]):
                if reg_matrix_columns[i] not in selected_terms:
                    f1 = (
                        reg_matrix[:, [*selected_terms, reg_matrix_columns[i]]]
                        @ LA.pinv(
                            reg_matrix[:, [*selected_terms, reg_matrix_columns[i]]]
                        )
                        @ y
                    )
                    if len(selected_terms) != 0:
                        f2 = (
                            reg_matrix[:, selected_terms]
                            @ LA.pinv(reg_matrix[:, selected_terms])
                            @ y
                        )
                    else:
                        f2 = np.zeros_like(y)
                    vp_estimation = self.conditional_mutual_information(y, f1, f2)
                    initial_vector[0, i] = vp_estimation
                else:
                    continue

            ix = np.nanargmax(initial_vector)
            max_value = initial_vector[0, ix]

            if (ksg_max - ksg_local <= self.tol) or (max_value <= self.tol):
                stop_criteria = True
            elif len(selected_terms) > np.max([8, reg_matrix.shape[1] / 2]):
                success = False
                stop_criteria = True

        return selected_terms, success

    def conditional_mutual_information(self, y, f1, f2):
        """Finds the conditional mutual information.
        Finds the conditioned mutual information between $y$ and $f1$ given $f2$.

        This code is based on Matlab Entropic Regression package.
        https://github.com/almomaa/ERFit-Package

        Parameters
        ----------
        y : ndarray of floats
            The source signal.
        f1 : ndarray of floats
            The destination signal.
        f2 : ndarray of floats
            The condition set.

        Returns
        -------
        vp_estimation : float
            The conditioned mutual information.

        References
        ----------
        - Abd AlRahman R. AlMomani, Jie Sun, and Erik Bollt. How Entropic
            Regression Beats the Outliers Problem in Nonlinear System
            Identification. Chaos 30, 013107 (2020).
        - Alexander Kraskov, Harald St¨ogbauer, and Peter Grassberger.
            Estimating mutual information. Physical Review E, 69:066-138,2004
        - Alexander Kraskov, Harald St¨ogbauer, and Peter Grassberger.
            Estimating mutual information. Physical Review E, 69:066-138,2004
        - Alexander Kraskov, Harald St¨ogbauer, and Peter Grassberger.
            Estimating mutual information. Physical Review E, 69:066-138,2004

        """
        joint_space = np.concatenate([y, f1, f2], axis=1)
        smallest_distance = np.sort(
            cdist(joint_space, joint_space, "minkowski", p=self.p).T
        )
        idx = np.argpartition(smallest_distance[-1, :], self.k + 1)[: self.k + 1]
        smallest_distance = smallest_distance[:, idx]
        epsilon = smallest_distance[:, -1].reshape(-1, 1)
        # Find number of points from (y,f2), (f1,f2), and (f2,f2) that lies withing the
        # k^{th} nearest neighbor distance from each point of themselves.
        smallest_distance_y_f2 = cdist(
            np.concatenate([y, f2], axis=1),
            np.concatenate([y, f2], axis=1),
            "minkowski",
            p=self.p,
        )
        less_than_array_y_f2 = np.array((smallest_distance_y_f2 < epsilon)).astype(int)
        y_f2 = (np.sum(less_than_array_y_f2, axis=1) - 1).reshape(-1, 1)

        smallest_distance_f1_f2 = cdist(
            np.concatenate([f1, f2], axis=1),
            np.concatenate([f1, f2], axis=1),
            "minkowski",
            p=self.p,
        )
        less_than_array_f1_f2 = np.array((smallest_distance_f1_f2 < epsilon)).astype(
            int
        )
        f1_f2 = (np.sum(less_than_array_f1_f2, axis=1) - 1).reshape(-1, 1)

        smallest_distance_f2 = cdist(f2, f2, "minkowski", p=self.p)
        less_than_array_f2 = np.array((smallest_distance_f2 < epsilon)).astype(int)
        f2_f2 = (np.sum(less_than_array_f2, axis=1) - 1).reshape(-1, 1)
        arr = psi(y_f2 + 1) + psi(f1_f2 + 1) - psi(f2_f2 + 1)
        vp_estimation = psi(self.k) - np.nanmean(arr[np.isfinite(arr)])
        return vp_estimation

    def tolerance_estimator(self, y):
        """Tolerance Estimation for mutual independence test.
        Finds the conditioned mutual information between $y$ and $f1$ given $f2$.

        This code is based on Matlab Entropic Regression package.
        https://github.com/almomaa/ERFit-Package

        Parameters
        ----------
        y : ndarray of floats
            The source signal.

        Returns
        -------
        tol : float
            The tolerance value given q.

        References
        ----------
        - Abd AlRahman R. AlMomani, Jie Sun, and Erik Bollt. How Entropic
            Regression Beats the Outliers Problem in Nonlinear System
            Identification. Chaos 30, 013107 (2020).
        - Alexander Kraskov, Harald St¨ogbauer, and Peter Grassberger.
            Estimating mutual information. Physical Review E, 69:066-138,2004
        - Alexander Kraskov, Harald St¨ogbauer, and Peter Grassberger.
            Estimating mutual information. Physical Review E, 69:066-138,2004
        - Alexander Kraskov, Harald St¨ogbauer, and Peter Grassberger.
            Estimating mutual information. Physical Review E, 69:066-138,2004

        """
        ksg_estimation = []
        # ksg_estimation = [
        #     getattr(self, self.mutual_information_estimator)(y,
        # self.rng.permutation(y))
        #     for i in range(self.n_perm)
        # ]

        for _ in range(self.n_perm):
            mutual_information_output = getattr(
                self, self.mutual_information_estimator
            )(y, self.rng.permutation(y))

            ksg_estimation.append(mutual_information_output)

        ksg_estimation = np.array(ksg_estimation)
        tol = np.quantile(ksg_estimation, self.q)
        return tol

    def fit(self, *, X=None, y=None):
        """Fit polynomial NARMAX model using AOLS algorithm.

        The 'fit' function allows a friendly usage by the user.
        Given two arguments, X and y, fit training data.

        The Entropic Regression algorithm is based on the Matlab package available on:
        https://github.com/almomaa/ERFit-Package

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
        theta : array-like of shape = number_of_model_elements
            The estimated parameters of the model.

        References
        ----------
        - Abd AlRahman R. AlMomani, Jie Sun, and Erik Bollt. How Entropic
            Regression Beats the Outliers Problem in Nonlinear System
            Identification. Chaos 30, 013107 (2020).
        - Alexander Kraskov, Harald St¨ogbauer, and Peter Grassberger.
            Estimating mutual information. Physical Review E, 69:066-138,2004
        - Alexander Kraskov, Harald St¨ogbauer, and Peter Grassberger.
            Estimating mutual information. Physical Review E, 69:066-138,2004
        - Alexander Kraskov, Harald St¨ogbauer, and Peter Grassberger.
            Estimating mutual information. Physical Review E, 69:066-138,2004

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

        if self.regressor_code.shape[0] > 90:
            warnings.warn(
                (
                    "Given the higher number of possible regressors"
                    f" ({self.regressor_code.shape[0]}), the Entropic Regression"
                    " algorithm may take long time to run. Consider reducing the"
                    " number of regressors "
                ),
                stacklevel=2,
            )

        y_full = y.copy()
        y = y[self.max_lag :].reshape(-1, 1)
        self.tol = 0
        ksg_estimation = []
        for _ in range(self.n_perm):
            mutual_information_output = getattr(
                self, self.mutual_information_estimator
            )(y, self.rng.permutation(y))
            ksg_estimation.append(mutual_information_output)

        ksg_estimation = np.array(ksg_estimation).reshape(-1, 1)
        self.tol = np.quantile(ksg_estimation, self.q)
        self.estimated_tolerance = self.tol
        success = False
        if not self.skip_forward:
            selected_terms, success = self.entropic_regression_forward(reg_matrix, y)

        if not success or self.skip_forward:
            selected_terms = np.array(list(range(reg_matrix.shape[1])))

        selected_terms_backward = self.entropic_regression_backward(
            reg_matrix[:, selected_terms], y, list(range(len(selected_terms)))
        )

        final_model = selected_terms[selected_terms_backward]
        # re-check for the constant term (add it to the estimated indices)
        if 0 not in final_model:
            final_model = np.array([0, *final_model])

        if self.basis_function.__class__.__name__ == "Polynomial":
            self.final_model = self.regressor_code[final_model, :].copy()
        elif self.basis_function.__class__.__name__ != "Polynomial" and self.ensemble:
            basis_code = np.sort(
                np.tile(
                    self.regressor_code[1:, :], (self.basis_function.repetition, 1)
                ),
                axis=0,
            )
            self.regressor_code = np.concatenate([self.regressor_code[1:], basis_code])
            self.final_model = self.regressor_code[final_model, :].copy()
        else:
            self.regressor_code = np.sort(
                np.tile(
                    self.regressor_code[1:, :], (self.basis_function.repetition, 1)
                ),
                axis=0,
            )
            self.final_model = self.regressor_code[final_model, :].copy()

        self.theta = getattr(self, self.estimator)(reg_matrix[:, final_model], y_full)
        if (np.abs(self.theta[0]) < self.h) and (
            np.sum((self.theta != 0).astype(int)) > 1
        ):
            self.theta = self.theta[1:].reshape(-1, 1)
            self.final_model = self.final_model[1:, :]
            final_model = final_model[1:]

        self.n_terms = len(
            self.theta
        )  # the number of terms we selected (necessary in the 'results' methods)
        self.err = self.n_terms * [
            0
        ]  # just to use the `results` method. Will be changed in next update.
        self.pivv = final_model
        return self

    def predict(self, *, X=None, y=None, steps_ahead=None, forecast_horizon=None):
        """Return the predicted values given an input.

        The predict function allows a friendly usage by the user.
        Given a previously trained model, predict values given
        a new set of data.

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
                yhat = yhat = np.concatenate([y[: self.max_lag], yhat], axis=0)
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
            yhat = self._basis_function_predict(X, y, forecast_horizon=forecast_horizon)
            yhat = np.concatenate([y[: self.max_lag], yhat], axis=0)
            return yhat
        if steps_ahead == 1:
            yhat = self._one_step_ahead_prediction(X, y)
            yhat = np.concatenate([y[: self.max_lag], yhat], axis=0)
            return yhat

        yhat = self._basis_function_n_step_prediction(
            X, y, steps_ahead=steps_ahead, forecast_horizon=forecast_horizon
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

    def _model_prediction(self, X, y_initial, forecast_horizon=None):
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
        elif self.model_type == "NFIR":
            return self._nfir_predict(X, y_initial)
        else:
            raise ValueError(
                f"model_type must be NARMAX, NAR or NFIR. Got {self.model_type}"
            )

    def _narmax_predict(self, X, y_initial, forecast_horizon):
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
