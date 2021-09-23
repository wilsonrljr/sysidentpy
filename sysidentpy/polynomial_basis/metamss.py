""" Meta Model Structure Selection"""
# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause

import numpy as np
from scipy.stats import t
import warnings
from .simulation import SimulatePolynomialNarmax
from ..parameter_estimation.estimators import Estimators
from ..utils._check_arrays import check_X_y
from ..metrics import root_relative_squared_error, mean_squared_error
from ..metaheuristics.bpsogsa import BPSOGSA


class MetaMSS(BPSOGSA, SimulatePolynomialNarmax):
    """Meta-Model Structure Selection: Building Polynomial NARXMAX model

    Parameters
    ----------
    non_degree : int, default=2
        The nonlinearity degree of the polynomial function.
    ylag : int, default=2
        The maximum lag of the output.
    xlag : int, default=2
        The maximum lag of the input.
    loss_func : str, default="metamss_loss"
        The loss function to be minimized.
    n_inputs : int, default=1
        The number of inputs of the system.
    estimator : str, default="least_squares"
        The parameter estimation method.
    estimate_parameter : bool, default=True
        Whether to estimate the model parameters.
    extended_least_squres : bool, default=False
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
    mu : float, defaul=0.01
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
    dimension : int, default=15
        The dimension of the search space.
        criteria method.
    p_zeros : float, default=0.5
        The probability of getting ones in the construction of the population.
    p_zeros : float, default=0.5
        The probability of getting zeros in the construction of the population.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from sysidentpy.metaheuristics import MetaMSS
    >>> from sysidentpy.metrics import root_relative_squared_error
    >>> from sysidentpy.utils.generate_data import get_siso_data
    >>> x_train, x_valid, y_train, y_valid = get_siso_data(n=400,
    ...                                                    colored_noise=False,
    ...                                                    sigma=0.001,
    ...                                                    train_percentage=80)
    >>> model = MetaMSS(
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
    >>> results = pd.DataFrame(model.results(err_precision=8,
    ...                                      dtype='dec'),
    ...                        columns=['Regressors', 'Parameters', 'ERR'])
    >>> print(results)
        Regressors Parameters         ERR
    0        x1(k-2)     0.9000  0.00000000
    1         y(k-1)     0.1999  0.00000000
    2  x1(k-1)y(k-1)     0.1000  0.00000000

    References
    ----------
    [1] Manuscript: Meta-Model Structure Selection: Building Polynomial NARX Model
        for Regression and Classification 
        https://arxiv.org/pdf/2109.09917.pdf
    [1] Manuscript (Portuguese): Identificação de Sistemas Não Lineares
        Utilizando o Algoritmo Híbrido e Binário de Otimização por
        Enxame de Partículas e Busca Gravitacional
        DOI: 10.17648/sbai-2019-111317
    [2] Master thesis: Meta model structure selection: an algorithm for
        building polynomial NARX models for regression and classification
    """

    def __init__(
        self,
        maxiter=30,
        alpha=23,
        g_zero=100,
        k_agents_percent=2,
        norm=-2,
        power=2,
        n_agents=10,
        p_zeros=0.5,
        p_ones=0.5,
        p_value=0.05,
        non_degree=2,
        xlag=2,
        ylag=2,
        n_inputs=1,
        estimator="least_squares",
        extended_least_squares=False,
        lam=0.98,
        delta=0.01,
        offset_covariance=0.2,
        mu=0.01,
        eps=np.finfo(np.float64).eps,
        gama=0.2,
        weight=0.02,
        estimate_parameter=False,
        loss_func="metamss_loss",
    ):
        super().__init__(
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
        SimulatePolynomialNarmax().__init__(
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
            n_inputs=n_inputs,
        )
        self.xlag = xlag
        self.ylag = ylag
        self.non_degree = non_degree
        self.p_value = p_value
        self._n_inputs = n_inputs
        self.estimator = estimator
        self.extended_least_squares = extended_least_squares
        self.lam = lam
        self.delta = delta
        self.offset_covariance = offset_covariance
        self.mu = mu
        self.eps = eps
        self.gama = gama
        self.weight = weight
        self.estimate_parameter = estimate_parameter
        self.loss_func = loss_func
        self.regressor_code, _ = self.regressor_space(
            non_degree=non_degree, xlag=xlag, ylag=ylag, n_inputs=n_inputs
        )
        self.dimension = self.regressor_code.shape[0]

    def fit(self, X_train, y_train, X_test, y_test):
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
        if y_train is None:
            raise ValueError("y cannot be None")

        check_X_y(X_train, y_train)

        velocity = np.zeros([self.dimension, self.n_agents])
        population = self.generate_random_population()
        self.best_by_iter = []
        self.mean_by_iter = []
        self.optimal_fitness_value = np.inf
        self.optimal_model = None
        self.best_model_history = []
        self.tested_models = []
        for iter in range(self.maxiter):
            fitness = self.evaluate_objective_function(
                X_train, y_train, X_test, y_test, population
            )
            column_of_best_solution = np.nanargmin(fitness)
            current_best_fitness = fitness[column_of_best_solution]

            if current_best_fitness < self.optimal_fitness_value:
                self.optimal_fitness_value = current_best_fitness
                self.optimal_model = population[:, column_of_best_solution].copy()
                self.best_model_history.append(self.optimal_model)

            self.best_by_iter.append(self.optimal_fitness_value)
            self.mean_by_iter.append(np.mean(fitness))
            agent_mass = self.mass_calculation(fitness)
            gravitational_constant = self.calculate_gravitational_constant(iter)
            acceleration = self.calculate_acceleration(
                population, agent_mass, gravitational_constant, iter
            )
            velocity, population = self.update_velocity_position(
                population,
                acceleration,
                velocity,
                iter,
            )

        self.final_model = self.regressor_code[self.optimal_model == 1].copy()
        yhat, _ = self.simulate(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_code=self.final_model,
            plot=False,
        )
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
                fitness.append(30) # penalty for cases where there is no terms
                continue

            m = self.regressor_code[agent == 1].copy()
            yhat, _ = self.simulate(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                model_code=m,
                plot=False,
            )

            residues = y_test - yhat
            psi = self.build_information_matrix(
                X_train, y_train, self.xlag, self.ylag, self.non_degree
            )
            psi = psi[:, self.pivv]
            pos_insig_terms, _, _ = self.perform_t_test(psi, self.theta, residues)

            pos_aux = np.where(agent == 1)[0]
            pos_aux = pos_aux[pos_insig_terms]
            agent[pos_aux] = 0

            m = self.regressor_code[agent == 1].copy()

            if np.all(agent == 0):
                fitness.append(30)
                continue

            yhat, _ = self.simulate(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                model_code=m,
                plot=False,
            )

            self.final_model = m.copy()
            self.tested_models.append(m)

            d = getattr(self, self.loss_func)(y_test, yhat, len(self.theta))
            fitness.append(d)

        return fitness

    def perform_t_test(self, psi, theta, residues):
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
            pos_insig_terms : array
                these regressors in the actual candidate solution are removed
                from the popolution since they are insignificant
            t_test : array
                the values of the p_value of each regressor of the model

        """

        sum_of_squared_residues = np.sum(residues ** 2)
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

        pos_insig_terms = np.where(tail2P > self.p_value)[0]
        pos_insig_terms = pos_insig_terms.reshape(-1, 1).T
        if pos_insig_terms.shape == 0:
            return np.array([]), t_test, tail2P
        else:
            return pos_insig_terms, t_test, tail2P

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
        n_theta : ndarray of floats
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
