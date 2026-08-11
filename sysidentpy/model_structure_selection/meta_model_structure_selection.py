"""Meta Model Structure Selection."""

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause
from copy import deepcopy
from numbers import Real
from typing import Tuple, Union, Optional

import numpy as np
from scipy.stats import t

from .._lib._array_api import get_namespace, _require_numpy_namespace
from sysidentpy.utils.information_matrix import build_lagged_matrix
from ..basis_function import Polynomial
from ..metaheuristics import BPSOGSA
from ..metrics import mean_squared_error, root_relative_squared_error
from ..simulation import SimulateNARMAX
from ..utils.check_arrays import (
    num_features,
    check_random_state,
    check_x_y,
)
from ..utils.lags import get_max_lag_from_model_code, get_max_xlag, get_max_ylag
from ..utils.narmax_tools import train_test_split

from ..parameter_estimation.estimators import (
    LeastSquares,
    RidgeRegression,
    RecursiveLeastSquares,
    TotalLeastSquares,
    LeastMeanSquareMixedNorm,
    LeastMeanSquares,
    LeastMeanSquaresFourth,
    LeastMeanSquaresLeaky,
    LeastMeanSquaresNormalizedLeaky,
    LeastMeanSquaresNormalizedSignRegressor,
    LeastMeanSquaresNormalizedSignSign,
    LeastMeanSquaresSignError,
    LeastMeanSquaresSignSign,
    AffineLeastMeanSquares,
    NormalizedLeastMeanSquares,
    NormalizedLeastMeanSquaresSignError,
    LeastMeanSquaresSignRegressor,
)

Estimators = Union[
    LeastSquares,
    RidgeRegression,
    RecursiveLeastSquares,
    TotalLeastSquares,
    LeastMeanSquareMixedNorm,
    LeastMeanSquares,
    LeastMeanSquaresFourth,
    LeastMeanSquaresLeaky,
    LeastMeanSquaresNormalizedLeaky,
    LeastMeanSquaresNormalizedSignRegressor,
    LeastMeanSquaresNormalizedSignSign,
    LeastMeanSquaresSignError,
    LeastMeanSquaresSignSign,
    AffineLeastMeanSquares,
    NormalizedLeastMeanSquares,
    NormalizedLeastMeanSquaresSignError,
    LeastMeanSquaresSignRegressor,
]


class MetaMSS(SimulateNARMAX, BPSOGSA):
    r"""Meta-Model Structure Selection: Building Polynomial NARMAX model.

    This class uses the MetaMSS ([1]_, [2]_, [3]_) algorithm to build NARMAX models.
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
    loss_func : str, default="metamss_loss"
        The loss function to be minimized.
    estimator : str, default="least_squares"
        The parameter estimation method.
    estimate_parameter : bool, default=True
        Whether to estimate the model parameters.
    eps : float
        Normalization factor of the normalized filters.
    maxiter : int, default=30
        The maximum number of iterations.
    alpha : int, default=23
        The descending coefficient of the gravitational constant.
    g_zero : int, default=100
        The initial value of the gravitational constant.
    k_agents_percent: int, default=2
        Percent of agents applying force to the others in the last iteration.
    norm : int, default=-2
        The information criteria method to be used.
    power : int, default=2
        The number of the model terms to be selected.
        Note that n_terms overwrite the information criteria
        values.
    n_agents : int, default=10
        The number of agents to search the optimal solution.
    p_ones : float, default=0.5
        The probability of getting ones in the construction of the population.
        It must be greater than zero because MetaMSS cannot evaluate an empty model.
    p_zeros : float, default=0.5
        The probability of getting zeros in the construction of the population.
    random_state : int, numpy.random.Generator, numpy.random.RandomState, optional
        Controls all random draws made by the optimizer. An integer produces the
        same trajectory on each call to :meth:`fit`; a generator instance advances
        its state between calls.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from sysidentpy.model_structure_selection import MetaMSS
    >>> from sysidentpy.metrics import root_relative_squared_error
    >>> from sysidentpy.basis_function import Polynomial
    >>> from sysidentpy.utils.display_results import results
    >>> from sysidentpy.utils.generate_data import get_siso_data
    >>> x_train, x_valid, y_train, y_valid = get_siso_data(n=400,
    ...                                                    colored_noise=False,
    ...                                                    sigma=0.001,
    ...                                                    train_percentage=80)
    >>> basis_function = Polynomial(degree=2)
    >>> model = MetaMSS(
    ...     basis_function=basis_function,
    ...     norm=-2,
    ...     xlag=7,
    ...     ylag=7,
    ...     k_agents_percent=2,
    ...     estimate_parameter=True,
    ...     maxiter=30,
    ...     n_agents=10,
    ...     p_value=0.05,
    ...     loss_func='metamss_loss'
    ... )
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
    - Manuscript: Meta-Model Structure Selection: Building Polynomial NARX Model
       for Regression and Classification
       https://arxiv.org/pdf/2109.09917.pdf
    - Manuscript (Portuguese): Identificação de Sistemas Não Lineares
       Utilizando o Algoritmo Híbrido e Binário de Otimização por
       Enxame de Partículas e Busca Gravitacional
       DOI: 10.17648/sbai-2019-111317
    - Master thesis: Meta model structure selection: an algorithm for
       building polynomial NARX models for regression and classification

    """

    def __init__(
        self,
        *,
        maxiter: int = 30,
        alpha: int = 23,
        g_zero: int = 100,
        k_agents_percent: int = 2,
        norm: float = -2,
        power: int = 2,
        n_agents: int = 10,
        p_zeros: float = 0.5,
        p_ones: float = 0.5,
        p_value: float = 0.05,
        xlag: Union[int, list] = 1,
        ylag: Union[int, list] = 1,
        elag: Union[int, list] = 1,
        estimator: Estimators = LeastSquares(),
        eps: np.float64 = np.finfo(np.float64).eps,
        estimate_parameter: bool = True,
        loss_func: str = "metamss_loss",
        model_type: str = "NARMAX",
        basis_function: Polynomial = Polynomial(),
        steps_ahead: Optional[int] = None,
        random_state: int | np.random.Generator | np.random.RandomState | None = None,
        test_size: float = 0.25,
    ):
        super().__init__(
            estimator=estimator,
            eps=eps,
            estimate_parameter=estimate_parameter,
            model_type=model_type,
            basis_function=basis_function,
        )

        BPSOGSA.__init__(
            self,
            n_agents=n_agents,
            maxiter=maxiter,
            g_zero=g_zero,
            alpha=alpha,
            k_agents_percent=k_agents_percent,
            norm=norm,
            power=power,
            p_zeros=p_zeros,
            p_ones=p_ones,
            random_state=random_state,
        )

        self.xlag = xlag
        self.ylag = ylag
        self._search_xlag = deepcopy(xlag)
        self._search_ylag = deepcopy(ylag)
        self.elag = elag
        self.p_value = p_value
        self.estimator = estimator
        self.estimate_parameter = estimate_parameter
        self.loss_func = loss_func
        self.steps_ahead = steps_ahead
        self.random_state = random_state
        self.test_size = test_size
        self.n_inputs = None
        self.regressor_code = None
        self.best_model_history = None
        self.tested_models = None
        self.final_model = None
        self._search_space_max_lag = None
        self._validate_metamss_params()

    def _validate_metamss_params(self):
        if isinstance(self.ylag, int) and self.ylag < 1:
            raise ValueError(f"ylag must be integer and > zero. Got {self.ylag}")

        if isinstance(self.xlag, int) and self.xlag < 1:
            raise ValueError(f"xlag must be integer and > zero. Got {self.xlag}")

        if not isinstance(self.xlag, (int, list)):
            raise ValueError(f"xlag must be integer and > zero. Got {self.xlag}")

        if not isinstance(self.ylag, (int, list)):
            raise ValueError(f"ylag must be integer and > zero. Got {self.ylag}")

        if (
            isinstance(self.p_value, bool)
            or not isinstance(self.p_value, Real)
            or not np.isfinite(self.p_value)
            or not 0 <= self.p_value <= 1
        ):
            raise ValueError(
                "p_value must be a finite real number in the interval [0, 1]. "
                f"Got {self.p_value}"
            )

        if not np.isclose(self.p_zeros + self.p_ones, 1):
            raise ValueError("p_zeros and p_ones must sum to 1")

        if not 0 < self.p_ones <= 1 or not 0 <= self.p_zeros < 1:
            raise ValueError(
                "MetaMSS requires p_ones > 0 so that nonempty models can be sampled"
            )

    def _generate_nonempty_agent(self) -> np.ndarray:
        """Sample one nonempty candidate using the configured probabilities."""
        dimension = (
            self.regressor_code.shape[0]
            if self.regressor_code is not None
            else self.dimension
        )
        for _ in range(100):
            agent = self._rng.choice(
                [0, 1], size=dimension, p=[self.p_zeros, self.p_ones]
            )
            if np.any(agent):
                return agent

        raise RuntimeError(
            "Unable to sample a nonempty MetaMSS candidate after 100 attempts. "
            "Increase p_ones."
        )

    def fit(
        self,
        *,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ):
        """Fit the polynomial NARMAX model.

        Parameters
        ----------
        X : ndarray, optional
            The input data to be used in the training process.
        y : ndarray
            The output data to be used in the training process.

        Returns
        -------
        self : returns an instance of self.

        """
        if not isinstance(self.basis_function, Polynomial):
            raise NotImplementedError(
                "Currently MetaMSS only supports polynomial models."
            )
        if y is None:
            raise ValueError("y cannot be None")

        xp = get_namespace(y) if X is None else get_namespace(X, y)
        _require_numpy_namespace(xp, feature="MetaMSS", dependency="SciPy")
        if y.ndim != 2 or y.shape[1] != 1:
            raise ValueError(
                "MetaMSS requires y to be a 2D array with exactly one output "
                f"column. Got shape {y.shape}."
            )

        if X is not None:
            check_x_y(X, y)
            n_inputs = num_features(X)
        else:
            n_inputs = 1  # just to create the regressor space base

        search_xlag = deepcopy(getattr(self, "_search_xlag", self.xlag))
        search_ylag = deepcopy(getattr(self, "_search_ylag", self.ylag))
        search_space_max_lag = max(get_max_xlag(search_xlag), get_max_ylag(search_ylag))
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size
        )
        if y_train.shape[0] <= search_space_max_lag:
            raise ValueError(
                "The identification set must contain more samples than the "
                f"maximum search-space lag ({search_space_max_lag}). Got "
                f"{y_train.shape[0]} identification samples."
            )

        self.n_inputs = n_inputs
        self._search_xlag = deepcopy(search_xlag)
        self._search_ylag = deepcopy(search_ylag)
        self.xlag = deepcopy(search_xlag)
        self.ylag = deepcopy(search_ylag)
        self.max_lag = search_space_max_lag
        self._search_space_max_lag = search_space_max_lag
        self.regressor_code = self.regressor_space(self.n_inputs)
        self.dimension = self.regressor_code.shape[0]
        velocity = np.zeros([self.dimension, self.n_agents])
        self._rng = check_random_state(self.random_state)
        population = self.generate_random_population()
        empty_agents = np.flatnonzero(~np.any(population, axis=0))
        for column in empty_agents:
            population[:, column] = self._generate_nonempty_agent()
        self.best_by_iter = []
        self.mean_by_iter = []
        self.optimal_fitness_value = np.inf
        self.optimal_model = None
        self.best_model_history = []
        self.tested_models = []

        for i in range(self.maxiter):
            fitness = np.asarray(
                self.evaluate_objective_function(
                    x_train, y_train, x_test, y_test, population
                ),
                dtype=float,
            )
            finite_fitness = np.isfinite(fitness)
            if not np.any(finite_fitness):
                raise RuntimeError(
                    "MetaMSS could not evaluate any candidate to a finite fitness."
                )
            fitness = np.where(finite_fitness, fitness, np.inf)
            column_of_best_solution = np.argmin(fitness)
            current_best_fitness = fitness[column_of_best_solution]

            if (
                current_best_fitness < self.optimal_fitness_value
                or self.optimal_model is None
            ):
                self.optimal_fitness_value = current_best_fitness
                self.optimal_model = population[:, column_of_best_solution].copy()
                self.best_model_history.append(self.optimal_model)

            self.best_by_iter.append(self.optimal_fitness_value)
            self.mean_by_iter.append(np.mean(fitness[finite_fitness]))
            agent_mass = self.mass_calculation(fitness)
            gravitational_constant = self.calculate_gravitational_constant(i)
            acceleration = self.calculate_acceleration(
                population, agent_mass, gravitational_constant, i
            )
            velocity, population = self.update_velocity_position(
                population,
                acceleration,
                velocity,
                i,
            )

        self.final_model = self.regressor_code[self.optimal_model == 1].copy()
        final_lag = get_max_lag_from_model_code(self.final_model)
        x_validation, y_validation = self._validation_data_with_training_tail(
            x_train, y_train, x_test, y_test, final_lag
        )
        _ = self.simulate(
            X_train=x_train,
            y_train=y_train,
            X_test=x_validation,
            y_test=y_validation,
            model_code=self.final_model,
            steps_ahead=self.steps_ahead,
        )
        self.max_lag = self._get_max_lag()
        return self

    @staticmethod
    def _validation_data_with_training_tail(
        x_train: Optional[np.ndarray],
        y_train: np.ndarray,
        x_test: Optional[np.ndarray],
        y_test: np.ndarray,
        candidate_lag: int,
    ) -> tuple[Optional[np.ndarray], np.ndarray]:
        """Prepend identification data as validation initial conditions."""
        y_validation = np.concatenate((y_train[-candidate_lag:], y_test), axis=0)
        if x_test is None:
            return None, y_validation
        if x_train is None:
            raise ValueError("x_train cannot be None when x_test is provided")
        x_validation = np.concatenate((x_train[-candidate_lag:], x_test), axis=0)
        return x_validation, y_validation

    def evaluate_objective_function(
        self,
        x_train: Optional[np.ndarray],
        y_train: Optional[np.ndarray],
        x_test: Optional[np.ndarray],
        y_test: Optional[np.ndarray],
        population: np.ndarray,
    ):
        """Fit the polynomial NARMAX model.

        Parameters
        ----------
        x_train : ndarray of floats
            The input data to be used in the training process.
        y_train : ndarray of floats
            The output data to be used in the training process.
        x_test : ndarray of floats
            The input data to be used in the prediction process.
        y_test : ndarray of floats
            The output data (initial conditions) to be used in the prediction process.
        population : ndarray of zeros and ones
            The initial population of agents.

        Returns
        -------
        fitness_value : ndarray
            The fitness value of each agent.
        """
        if y_train is None or y_test is None:
            raise ValueError("y_train and y_test cannot be None")
        if self.regressor_code is None:
            raise RuntimeError("The regressor space must be built before evaluation.")
        if self.tested_models is None:
            self.tested_models = []

        fitness = []
        for agent in population.T:
            for _ in range(100):
                if np.all(agent == 0):
                    agent[:] = self._generate_nonempty_agent()

                m = self.regressor_code[agent == 1].copy()
                candidate_lag = get_max_lag_from_model_code(m)
                x_validation, y_validation = self._validation_data_with_training_tail(
                    x_train, y_train, x_test, y_test, candidate_lag
                )
                yhat = self.simulate(
                    X_train=x_train,
                    y_train=y_train,
                    X_test=x_validation,
                    y_test=y_validation,
                    model_code=m,
                    steps_ahead=self.steps_ahead,
                )

                candidate_theta = self.theta
                if candidate_theta is None:
                    raise RuntimeError(
                        "The candidate simulation did not estimate theta."
                    )

                lagged_data = build_lagged_matrix(
                    x_train, y_train, self.xlag, self.ylag, self.model_type
                )

                psi = self.basis_function.fit(
                    lagged_data,
                    candidate_lag,
                    self.ylag,
                    self.xlag,
                    self.model_type,
                    predefined_regressors=self.pivv,
                )

                identification_target = y_train[candidate_lag:, 0].reshape(-1, 1)
                identification_residues = identification_target - psi @ candidate_theta
                supports_ols_t_test = (
                    isinstance(self.estimator, LeastSquares)
                    and not self.estimator.unbiased
                )
                has_valid_design = (
                    supports_ols_t_test
                    and psi.shape[0] > psi.shape[1]
                    and np.linalg.matrix_rank(psi) == psi.shape[1]
                )
                if has_valid_design:
                    pos_insignificant_terms, _, _ = self.perform_t_test(
                        psi, candidate_theta, identification_residues
                    )
                else:
                    pos_insignificant_terms = np.array([], dtype=np.intp)

                n_removed_terms = pos_insignificant_terms.size
                selected_positions = np.flatnonzero(agent)
                agent[selected_positions[pos_insignificant_terms]] = 0

                if np.all(agent == 0):
                    agent[:] = self._generate_nonempty_agent()
                    continue

                m = self.regressor_code[agent == 1].copy()
                candidate_lag = get_max_lag_from_model_code(m)
                x_validation, y_validation = self._validation_data_with_training_tail(
                    x_train, y_train, x_test, y_test, candidate_lag
                )
                yhat = self.simulate(
                    X_train=x_train,
                    y_train=y_train,
                    X_test=x_validation,
                    y_test=y_validation,
                    model_code=m,
                    steps_ahead=self.steps_ahead,
                )

                self.final_model = m.copy()
                self.tested_models.append(m)
                if self.theta is None:
                    raise RuntimeError("The pruned simulation did not estimate theta.")

                n_terms = len(self.theta)
                if self.loss_func == "metamss_loss":
                    n_terms += n_removed_terms

                y_score = y_test
                yhat_score = yhat[candidate_lag:]
                d = getattr(self, self.loss_func)(y_score, yhat_score, n_terms)
                fitness.append(d)
                break
            else:
                fitness.append(np.inf)

        return fitness

    def perform_t_test(
        self, psi: np.ndarray, theta: np.ndarray, residues: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform the t-test given the p-value defined by the user.

        Parameters
        ----------
        psi : array
            the data matrix of regressors
        theta : array
            the parameters estimated via least squares algorithm
        residues : array
            the identification residues of the solution

        Returns
        -------
        pos_insignificant_terms : array
            these regressors in the actual candidate solution are removed
            from the population since they are insignificant
        t_test : array
            the values of the p_value of each regressor of the model
        tail2p: array
            The calculated two-tailed p-value.

        Raises
        ------
        ValueError
            If the arrays do not represent a single-output OLS problem, if the
            residuals or parameters are not aligned with ``psi``, if there are no
            residual degrees of freedom, or if ``psi`` is rank deficient.

        Notes
        -----
        ``residues`` must be the one-step-ahead identification residuals
        ``y - psi @ theta`` from the same design matrix and OLS estimate passed to
        this method. The finite-sample Student's t interpretation assumes a
        correctly specified linear-in-the-parameters model, full-column-rank
        regressors, exogeneity, and independent homoscedastic Gaussian errors.
        Regressors containing lagged outputs are generally predetermined rather
        than strictly exogenous, so the exact finite-sample interpretation need
        not hold for dynamic NARX models even with white innovations. MetaMSS uses
        the resulting p-values as a model-selection heuristic; after data-driven
        structure selection they should not be interpreted as confirmatory
        post-selection inference.

        The residual variance is evaluated in a scaled logarithmic form. This
        preserves the t-statistic when the output unit is changed and avoids
        overflow or underflow in the sum of squared residuals. For an exactly
        zero residual vector, the smallest positive ``float64`` variance is used
        to define the otherwise degenerate standard error.

        """
        psi = np.asarray(psi)
        theta = np.asarray(theta)
        residues = np.asarray(residues)
        if psi.ndim != 2:
            raise ValueError(f"psi must be a 2D matrix. Got shape {psi.shape}.")
        if theta.ndim == 1:
            theta = theta.reshape(-1, 1)
        elif theta.ndim != 2 or theta.shape[1] != 1:
            raise ValueError(
                "theta must contain one column for a single-output OLS model. "
                f"Got shape {theta.shape}."
            )
        if residues.ndim == 1:
            residues = residues.reshape(-1, 1)
        elif residues.ndim != 2 or residues.shape[1] != 1:
            raise ValueError(
                "residues must contain one column for a single-output OLS model. "
                f"Got shape {residues.shape}."
            )
        arrays = (psi, theta, residues)
        if any(
            not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array)
            for array in arrays
        ):
            raise ValueError("psi, theta, and residues must be real numeric arrays.")

        n_samples, n_parameters = psi.shape
        if n_parameters == 0:
            raise ValueError("The t-test requires at least one regressor in psi.")
        if theta.shape[0] != n_parameters:
            raise ValueError(
                "theta must have one row per regressor in psi. "
                f"Got {theta.shape[0]} rows and {n_parameters} regressors."
            )
        if residues.shape[0] != n_samples:
            raise ValueError(
                "residues and psi must contain the same number of samples. "
                f"Got {residues.shape[0]} and {n_samples}."
            )
        if n_samples <= n_parameters:
            raise ValueError(
                "The t-test requires more samples than regressors. "
                f"Got {n_samples} samples and {n_parameters} regressors."
            )
        if not (
            np.all(np.isfinite(psi))
            and np.all(np.isfinite(theta))
            and np.all(np.isfinite(residues))
        ):
            raise ValueError(
                "psi, theta, and residues must contain only finite values."
            )
        if np.linalg.matrix_rank(psi) < n_parameters:
            raise ValueError("The t-test requires a full-column-rank regressor matrix.")

        degree_of_freedom = n_samples - n_parameters
        upper_triangular = np.linalg.qr(psi, mode="r")
        inverse_triangular = np.linalg.solve(upper_triangular, np.eye(n_parameters))

        # diag((psi.T @ psi)^-1) is the squared row norm of R^-1. Computing
        # its logarithm from normalized rows avoids overflow/underflow without
        # changing the covariance represented by the QR factorization.
        inverse_scale = np.max(np.abs(inverse_triangular), axis=1)
        normalized_inverse = inverse_triangular / inverse_scale[:, np.newaxis]
        log_skk_diag = 2 * np.log(inverse_scale) + np.log(
            np.sum(normalized_inverse**2, axis=1)
        )

        residual_scale = float(np.max(np.abs(residues)))
        if residual_scale == 0:
            log_residual_variance = np.log(np.finfo(np.float64).tiny)
        else:
            normalized_residues = residues / residual_scale
            scaled_sum_of_squares = float(np.sum(normalized_residues**2))
            log_residual_variance = (
                2 * np.log(residual_scale)
                + np.log(scaled_sum_of_squares)
                - np.log(degree_of_freedom)
            )

        log_standard_error = 0.5 * (log_residual_variance + log_skk_diag)
        theta_values = theta.ravel()
        t_values = np.zeros(n_parameters, dtype=float)
        nonzero_theta = theta_values != 0
        with np.errstate(over="ignore", under="ignore"):
            t_values[nonzero_theta] = np.sign(theta_values[nonzero_theta]) * np.exp(
                np.log(np.abs(theta_values[nonzero_theta]))
                - log_standard_error[nonzero_theta]
            )
        t_test = t_values.reshape(-1, 1)

        tail2p = 2 * t.sf(np.abs(t_test), degree_of_freedom)

        pos_insignificant_terms = np.flatnonzero(tail2p.ravel() > self.p_value).reshape(
            1, -1
        )

        return pos_insignificant_terms, t_test, tail2p

    def aic(self, y_test: np.ndarray, yhat: np.ndarray, n_theta: int) -> float:
        """Calculate the Akaike Information Criterion.

        Parameters
        ----------
        y_test : ndarray of floats
            The output data (initial conditions) to be used in the prediction process.
        yhat : ndarray of floats
            The n-steps-ahead predicted values of the model.
        n_theta : ndarray of floats
            The number of model parameters.

        Returns
        -------
        aic : float
            The Akaike Information Criterion

        """
        mse = max(mean_squared_error(y_test, yhat), np.finfo(np.float64).eps)
        n = y_test.shape[0]
        return n * np.log(mse) + 2 * n_theta

    def bic(self, y_test: np.ndarray, yhat: np.ndarray, n_theta: int) -> float:
        """Calculate the Bayesian Information Criterion.

        Parameters
        ----------
        y_test : ndarray of floats
            The output data (initial conditions) to be used in the prediction process.
        yhat : ndarray of floats
            The n-steps-ahead predicted values of the model.
        n_theta : ndarray of floats
            The number of model parameters.

        Returns
        -------
        bic : float
            The Bayesian Information Criterion

        """
        mse = max(mean_squared_error(y_test, yhat), np.finfo(np.float64).eps)
        n = y_test.shape[0]
        return n * np.log(mse) + n_theta * np.log(n)

    def metamss_loss(self, y_test: np.ndarray, yhat: np.ndarray, n_terms: int) -> float:
        """Calculate the MetaMSS loss function.

        Parameters
        ----------
        y_test : ndarray of floats
            The output data (initial conditions) to be used in the prediction process.
        yhat : ndarray of floats
            The n-steps-ahead predicted values of the model.
        n_terms : ndarray of floats
            The number of model parameters.

        Returns
        -------
        metamss_loss : float
            The MetaMSS loss function

        """
        penalty_count = np.arange(0, self.dimension + 1)
        penalty_distribution = (np.log(n_terms + 1) ** (-1)) / self.dimension
        penalty = self.sigmoid_linear_unit_derivative(
            penalty_count, self.dimension / 2, penalty_distribution
        )

        penalty = penalty - np.min(penalty)
        rmse = root_relative_squared_error(y_test, yhat)
        fitness = rmse * penalty[n_terms]
        if not np.isfinite(fitness):
            fitness = 30

        return fitness

    def sigmoid_linear_unit_derivative(self, x, c, a):
        """Calculate the derivative of the Sigmoid Linear Unit function.

        The derivative of Sigmoid Linear Unit (dSiLU) function can be
        viewed as a overshooting version of the sigmoid function.

        Parameters
        ----------
        x : ndarray
            The range of the regressors space.
        a : float
            The rate of change.
        c : int
            Corresponds to the x value where y = 0.5.

        Returns
        -------
        penalty : ndarray of floats
            The values of the penalty function

        """
        return (
            1
            / (1 + np.exp(-a * (x - c)))
            * (1 + (a * (x - c)) * (1 - 1 / (1 + np.exp(-a * (x - c)))))
        )

    def predict(
        self,
        *,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        steps_ahead: Optional[int] = None,
        forecast_horizon: int = 1,
    ) -> np.ndarray:
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
        steps_ahead : int, optional
            ``None`` selects free-run simulation, 1 selects one-step-ahead
            prediction, and values greater than 1 select n-step-ahead prediction.
        forecast_horizon : int, default=1
            Number of values predicted beyond the initial conditions for a NAR
            free-run prediction when ``X`` is ``None``.

        Returns
        -------
        yhat : ndarray of floats
            The predicted values of the model.

        """
        if not isinstance(self.basis_function, Polynomial):
            raise NotImplementedError(
                "MetaMSS doesn't support basis functions other than polynomial yet.",
            )
        return super().predict(
            X=X,
            y=y,
            steps_ahead=steps_ahead,
            forecast_horizon=forecast_horizon,
        )

    def _basis_function_predict(self, x, y_initial, forecast_horizon=None):
        """Not implemented."""
        raise NotImplementedError(
            "You can only use Polynomial Basis Function in MetaMSS for now."
        )

    def _basis_function_n_step_prediction(self, x, y, steps_ahead, forecast_horizon):
        """Not implemented."""
        raise NotImplementedError(
            "You can only use Polynomial Basis Function in MetaMSS for now."
        )
