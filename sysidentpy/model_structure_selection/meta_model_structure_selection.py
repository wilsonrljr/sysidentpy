""" Meta Model Structure Selection"""
# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause
from typing import Tuple, Union

import numpy as np
from scipy.stats import t

from ..basis_function import Polynomial
from ..metaheuristics import BPSOGSA
from ..metrics import mean_squared_error, root_relative_squared_error
from ..simulation import SimulateNARMAX
from ..utils._check_arrays import (
    _check_positive_int,
    _num_features,
    check_random_state,
    check_X_y,
)
from ..utils.deprecation import deprecated


@deprecated(
    version="v0.3.0",
    future_version="v0.4.0",
    message=(
        "Passing a string to define the estimator will rise an error in v0.4.0."
        " \n You'll have to use MetaMSS(estimator=LeastSquares()) instead. \n The"
        " only change is that you'll have to define the estimator first instead"
        " of passing a string like 'least_squares'. \n This change will make"
        " easier to implement new estimators and it'll improve code"
        " readability."
    ),
)
class MetaMSS(SimulateNARMAX, BPSOGSA):
    r"""Meta-Model Structure Selection: Building Polynomial NARMAX model

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
    extended_least_squares : bool, default=False
        Whether to use extended least squares method
        for parameter estimation.
        Note that we define a specific set of noise regressors.
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
    p_zeros : float, default=0.5
        The probability of getting ones in the construction of the population.
    p_zeros : float, default=0.5
        The probability of getting zeros in the construction of the population.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from sysidentpy.model_structure_selection import MetaMSS
    >>> from sysidentpy.metrics import root_relative_squared_error
    >>> from sysidentpy.basis_function._basis_function import Polynomial
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
    ...     estimator="least_squares",
    ...     k_agents_percent=2,
    ...     estimate_parameter=True,
    ...     maxiter=30,
    ...     n_agents=10,
    ...     p_value=0.05,
    ...     loss_func='metamss_loss'
    ... )
    >>> model.fit(x_train, y_train, x_valid, y_valid)
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
        norm: Union[int, float] = -2,
        power: int = 2,
        n_agents: int = 10,
        p_zeros: float = 0.5,
        p_ones: float = 0.5,
        p_value: float = 0.05,
        xlag: Union[int, list] = 1,
        ylag: Union[int, list] = 1,
        elag: Union[int, list] = 1,
        estimator: str = "least_squares",
        extended_least_squares: bool = False,
        lam: float = 0.98,
        delta: float = 0.01,
        offset_covariance: float = 0.2,
        mu: float = 0.01,
        eps: np.float64 = np.finfo(np.float64).eps,
        gama: float = 0.2,
        weight: float = 0.02,
        estimate_parameter: bool = True,
        loss_func: str = "metamss_loss",
        model_type: str = "NARMAX",
        basis_function: Polynomial = Polynomial(),
        steps_ahead: Union[int, None] = None,
        random_state: Union[int, None] = None,
    ):
        super().__init__(
            estimator=estimator,
            extended_least_squares=extended_least_squares,
            lam=lam,
            delta=delta,
            offset_covariance=offset_covariance,
            mu=mu,
            eps=eps,
            gama=gama,
            weight=weight,
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
        )

        self.xlag = xlag
        self.ylag = ylag
        self.elag = elag
        self.non_degree = basis_function.degree
        self.p_value = p_value
        self.estimator = estimator
        self.estimate_parameter = estimate_parameter
        self.loss_func = loss_func
        self.steps_ahead = steps_ahead
        self.random_state = random_state
        self.build_matrix = self.get_build_io_method(model_type)
        self.n_inputs = None
        self.regressor_code = None
        self.best_model_history = None
        self.tested_models = None
        self.final_model = None
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

    def fit(self, *, X=None, y=None, X_test=None, y_test=None):
        """Fit the polynomial NARMAX model.

        Parameters
        ----------
        X_train : ndarray of floats
            The input data to be used in the training process.
        y_train : ndarray of floats
            The output data to be used in the training process.
        X_test : ndarray of floats
            The input data to be used in the prediction process.
        y_test : ndarray of floats
            The output data (initial conditions) to be used in the prediction process.

        Returns
        -------
        self : returns an instance of self.

        """
        if self.basis_function.__class__.__name__ != "Polynomial":
            raise NotImplementedError(
                "Currently MetaMSS only supports polynomial models."
            )
        if y is None:
            raise ValueError("y cannot be None")

        if X is not None:
            check_X_y(X, y)
            self.n_inputs = _num_features(X)
        else:
            self.n_inputs = 1  # just to create the regressor space base

        #  self.n_inputs = _num_features(X_train)
        self.max_lag = self._get_max_lag()
        self.regressor_code = self.regressor_space(self.n_inputs)
        self.dimension = self.regressor_code.shape[0]
        velocity = np.zeros([self.dimension, self.n_agents])
        self.random_state = check_random_state(self.random_state)
        population = self.generate_random_population(self.random_state)
        self.best_by_iter = []
        self.mean_by_iter = []
        self.optimal_fitness_value = np.inf
        self.optimal_model = None
        self.best_model_history = []
        self.tested_models = []
        for i in range(self.maxiter):
            fitness = self.evaluate_objective_function(X, y, X_test, y_test, population)
            column_of_best_solution = np.nanargmin(fitness)
            current_best_fitness = fitness[column_of_best_solution]

            if current_best_fitness < self.optimal_fitness_value:
                self.optimal_fitness_value = current_best_fitness
                self.optimal_model = population[:, column_of_best_solution].copy()
                self.best_model_history.append(self.optimal_model)

            self.best_by_iter.append(self.optimal_fitness_value)
            self.mean_by_iter.append(np.mean(fitness))
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
        _ = self.simulate(
            X_train=X,
            y_train=y,
            X_test=X_test,
            y_test=y_test,
            model_code=self.final_model,
            steps_ahead=self.steps_ahead,
        )
        self.max_lag = self._get_max_lag()
        return self

    def evaluate_objective_function(self, X_train, y_train, X_test, y_test, population):
        """Fit the polynomial NARMAX model.

        Parameters
        ----------
        X_train : ndarray of floats
            The input data to be used in the training process.
        y_train : ndarray of floats
            The output data to be used in the training process.
        X_test : ndarray of floats
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
        fitness = []
        for agent in population.T:
            if np.all(agent == 0):
                fitness.append(30)  # penalty for cases where there is no terms
                continue

            m = self.regressor_code[agent == 1].copy()
            yhat = self.simulate(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                model_code=m,
                steps_ahead=self.steps_ahead,
            )

            residues = y_test - yhat
            self.max_lag = self._get_max_lag()
            lagged_data = self.build_matrix(X_train, y_train)

            psi = self.basis_function.fit(
                lagged_data, self.max_lag, predefined_regressors=self.pivv
            )

            pos_insignificant_terms, _, _ = self.perform_t_test(
                psi, self.theta, residues
            )

            pos_aux = np.where(agent == 1)[0]
            pos_aux = pos_aux[pos_insignificant_terms]
            agent[pos_aux] = 0

            m = self.regressor_code[agent == 1].copy()

            if np.all(agent == 0):
                fitness.append(1000)  # just a big number as penalty
                continue

            yhat = self.simulate(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                model_code=m,
                steps_ahead=self.steps_ahead,
            )

            self.final_model = m.copy()
            self.tested_models.append(m)
            if len(self.theta) == 0:
                print(m)
            d = getattr(self, self.loss_func)(y_test, yhat, len(self.theta))
            fitness.append(d)

        return fitness

    def perform_t_test(
        self, psi: np.ndarray, theta: np.ndarray, residues: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform the t-test given the p-value defined by the user

        Arguments:
        ----------
            psi : array
                the data matrix of regressors
            theta : array
                the parameters estimated via least squares algorithm
            residues : array
                the identification residues of the solution
            p_value_confidence : double
                parameter selected by the user to perform the statistical t-test

        Returns:
        --------
            pos_insignificant_terms : array
                these regressors in the actual candidate solution are removed
                from the population since they are insignificant
            t_test : array
                the values of the p_value of each regressor of the model

        """
        sum_of_squared_residues = np.sum(residues**2)
        variance_of_residues = (sum_of_squared_residues) / (
            len(residues) - psi.shape[1]
        )
        if np.isnan(variance_of_residues):
            variance_of_residues = 4.3645e05

        skk = np.linalg.pinv(psi.T.dot(psi))
        skk_diag = np.diag(skk)
        var_e = variance_of_residues * skk_diag
        se_theta = np.sqrt(var_e)
        se_theta = se_theta.reshape(-1, 1)
        t_test = theta / se_theta
        degree_of_freedom = psi.shape[0] - psi.shape[1]

        tail2P = 2 * t.cdf(-np.abs(t_test), degree_of_freedom)

        pos_insignificant_terms = np.where(tail2P > self.p_value)[0]
        pos_insignificant_terms = pos_insignificant_terms.reshape(-1, 1).T
        if pos_insignificant_terms.shape == 0:
            return np.array([]), t_test, tail2P

        # t_test and tail2P will be returned in future updates
        return pos_insignificant_terms, t_test, tail2P

    def aic(self, y_test, yhat, n_theta):
        """Calculate the Akaike Information Criterion

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
        mse = mean_squared_error(y_test, yhat)
        n = y_test.shape[0]
        return n * np.log(mse) + 2 * n_theta

    def bic(self, y_test, yhat, n_theta):
        """Calculate the Bayesian Information Criterion

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
        mse = mean_squared_error(y_test, yhat)
        n = y_test.shape[0]
        return n * np.log(mse) + n_theta + np.log(n)

    def metamss_loss(self, y_test, yhat, n_terms):
        """Calculate the MetaMSS loss function

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
        penalty_count = np.arange(0, self.dimension)
        penalty_distribution = (np.log(n_terms + 1) ** (-1)) / self.dimension
        penalty = self.sigmoid_linear_unit_derivative(
            penalty_count, self.dimension / 2, penalty_distribution
        )

        penalty = penalty - np.min(penalty)
        rmse = root_relative_squared_error(y_test, yhat)
        fitness = rmse * penalty[n_terms]
        if np.isnan(fitness):
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

    def predict(self, *, X=None, y=None, steps_ahead=None, forecast_horizon=None):
        """Return the predicted values given an input.

        The predict function allows a friendly usage by the user.
        Given a previously trained model, predict values given
        a new set of data.

        This method accept y values mainly for prediction n-steps ahead
        (to be implemented in the future)

        Parameters
        ----------
        X_test : ndarray of floats
            The input data to be used in the prediction process.
        y_test : ndarray of floats
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
                yhat = np.concatenate([y[: self.max_lag], yhat], axis=0)
                return yhat
            if steps_ahead == 1:
                yhat = self._one_step_ahead_prediction(X, y)
                yhat = np.concatenate([y[: self.max_lag], yhat], axis=0)
                return yhat

            _check_positive_int(steps_ahead, "steps_ahead")
            yhat = self._n_step_ahead_prediction(X, y, steps_ahead=steps_ahead)
            yhat = np.concatenate([y[: self.max_lag], yhat], axis=0)
            return yhat

        raise NotImplementedError(
            "MetaMSS doesn't support basis functions other than polynomial yet.",
        )

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
        yhat = super()._one_step_ahead_prediction(X, y)
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
        if self.model_type == "NFIR":
            return self._nfir_predict(X, y_initial)

        raise ValueError(
            f"model_type must be NARMAX, NAR or NFIR. Got {self.model_type}"
        )

    def _narmax_predict(self, X, y_initial, forecast_horizon):
        y_output = super()._narmax_predict(X, y_initial, forecast_horizon)
        return y_output

    def _nfir_predict(self, X, y_initial):
        y_output = super()._nfir_predict(X, y_initial)
        return y_output

    def _basis_function_predict(self, X, y_initial, forecast_horizon=None):
        """not implemented"""
        raise NotImplementedError(
            "You can only use Polynomial Basis Function in MetaMSS for now."
        )

    def _basis_function_n_step_prediction(self, X, y, steps_ahead, forecast_horizon):
        """not implemented"""
        raise NotImplementedError(
            "You can only use Polynomial Basis Function in MetaMSS for now."
        )

    def _basis_function_n_steps_horizon(self, X, y, steps_ahead, forecast_horizon):
        """not implemented"""
        raise NotImplementedError(
            "You can only use Polynomial Basis Function in MetaMSS for now."
        )
