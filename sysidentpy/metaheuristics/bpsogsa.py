""" Binary Hybrid Particle Swarm Optimization and Gravitational Search Algorithm"""
# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause

import numpy as np

from sysidentpy.utils._check_arrays import check_random_state


class BPSOGSA:
    """Binary Hybrid Particle Swarm Optimization and Gravitational Search Algorithm

    Parameters
    ----------
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
    >>> from sysidentpy.metaheuristics import BPSOGSA
    >>> opt = BPSOGSA(maxiter=100,
    ...               k_agents_percent=2,
    ...               n_agents=10,
    ...               dimension=20
    ...               )
    >>> opt.optimize()
    >>> plt.plot(opt.best_by_iter)
    >>> plt.show()
    >>> print(opt.optimal_fitness_value)

    References
    ----------
    - A New Hybrid PSOGSA Algorithm for Function Optimization,
       https://www.mathworks.com/matlabcentral/fileexchange/35939-hybrid-particle-swarm-optimization-and-gravitational-search-algorithm-psogsa
    - Manuscript: Particle swarm optimization: developments, applications and resources.
    - Manuscript: S-shaped versus v-shaped transfer functions for binary
       particle swarm optimization
    - Manuscript: BGSA: Binary Gravitational Search Algorithm.
    - Manuscript: A taxonomy of hybrid metaheuristics

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
        dimension=15,
        p_zeros=0.5,
        p_ones=0.5,
    ):

        self.dimension = dimension
        self.n_agents = n_agents
        self.maxiter = maxiter
        self.g_zero = g_zero
        self.alpha = alpha
        self.k_agents_percent = k_agents_percent
        self._norm = norm
        self._power = power
        self.p_zeros = p_zeros
        self.p_ones = p_ones
        self.best_by_iter = None
        self.mean_by_iter = None
        self.optimal_fitness_value = None
        self.optimal_model = None
        super(BPSOGSA, self).__init__()

    def evaluate_objective_function(self, candidate_solution):
        """Function to be optimized"""
        total = 0
        for candidate in candidate_solution:
            total += candidate**2
        return total

    def optimize(self):
        """Run the BPSOGSA algorithm.

        This algorithm is based on the Matlab implementation provided by the
        author of the BPSOGSA algorithm.

        References
        ----------
        - A New Hybrid PSOGSA Algorithm for Function Optimization.
           https://www.mathworks.com/matlabcentral/fileexchange/35939-hybrid-particle-swarm-optimization-and-gravitational-search-algorithm-psogsa
        - Manuscript: Particle swarm optimization: developments, applications and
            resources.
        - Manuscript: S-shaped versus v-shaped transfer functions for binary.
           particle swarm optimization
        - Manuscript: BGSA: Binary Gravitational Search Algorithm.
        - Manuscript: A taxonomy of hybrid metaheuristics.

        """
        velocity = np.zeros([self.dimension, self.n_agents])
        population = self.generate_random_population()
        self.best_by_iter = []
        self.mean_by_iter = []
        self.optimal_fitness_value = np.inf
        self.optimal_model = None

        for i in range(self.maxiter):
            fitness = self.evaluate_objective_function(population)

            column_of_best_solution = np.argmin(fitness)
            current_best_fitness = fitness[column_of_best_solution]

            if current_best_fitness < self.optimal_fitness_value:
                self.optimal_fitness_value = current_best_fitness
                self.optimal_model = population[:, column_of_best_solution].copy()

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

        return self

    def generate_random_population(self, random_state=None):
        """Generate the initial population of agents randomly

        Returns
        -------
        population : ndarray of zeros and ones
            The initial population of agents.

        """
        rng = check_random_state(random_state)
        population = rng.choice(
            [0, 1], size=(self.dimension, self.n_agents), p=[self.p_zeros, self.p_ones]
        )
        return population

    def mass_calculation(self, fitness_value):
        """Calculate the inertial masses of the agents.

        Parameters
        ----------
        fitness_value : ndarray
            The fitness value of each agent.

        Returns
        -------
        agent_mass : ndarray of floats
            The mass of each agent.

        """

        highest_fitness_value = np.nanmax(fitness_value)
        lowest_fitness_value = np.nanmin(fitness_value)

        column_fitness = len(fitness_value)
        if highest_fitness_value == lowest_fitness_value:
            agent_mass = np.ones([column_fitness, 1])
        else:
            best_fitness_value = lowest_fitness_value
            worst_fitness_value = highest_fitness_value
            agent_mass = (fitness_value - 0.99 * worst_fitness_value) / (
                best_fitness_value - worst_fitness_value
            )

        agent_mass = (5 * agent_mass) / np.sum(agent_mass)
        return agent_mass

    def calculate_gravitational_constant(self, iteration):
        """Update the gravitational constant.

        Parameters
        ----------
        iteration : int
            The specific time.

        Returns
        -------
        gravitational_constant : float
            The gravitational_constant at time defined by the iteration.

        """
        gravitational_constant = self.g_zero * np.exp(
            (-self.alpha * (iteration + 1)) / self.maxiter
        )

        return gravitational_constant

    def calculate_acceleration(
        self, population, agent_mass, gravitational_constant, iteration
    ):
        """Calculate the acceleration of each agent.

        Parameters
        ----------
        population : ndarray of zeros and ones
            The population defined by the agents.
        agent_mass : ndarray of floats
            The mass of each agent.
        gravitational_constant : float
            The gravitational_constant at time defined by the iteration.
        iteration : int
            The current iteration.

        Returns
        -------
        acceleration : ndarray of floats
            The acceleration of each agent.

        """

        k_best_agents = self.k_agents_percent + (1 - iteration / self.maxiter) * (
            100 - self.k_agents_percent
        )
        k_best_agents = round(self.n_agents * k_best_agents / 100)

        maximum_value_index = np.argsort(agent_mass)[::-1].ravel()
        gravitational_force = np.zeros([self.dimension, self.n_agents])
        for i in range(self.n_agents):
            for j in range(k_best_agents):
                if maximum_value_index[j] != i:
                    euclidian_distance = np.linalg.norm(
                        population[:, i] - population[:, maximum_value_index[j]],
                        self._norm,
                    )
                    gravitational_force[:, i] = gravitational_force[
                        :, i
                    ] + np.random.rand(self.dimension) * agent_mass[
                        maximum_value_index[j]
                    ] * (
                        population[:, maximum_value_index[j]] - population[:, i]
                    ) / (
                        euclidian_distance**self._power + np.finfo(np.float64).eps
                    )

        acceleration = gravitational_force * gravitational_constant
        return acceleration

    def update_velocity_position(
        self,
        population,
        acceleration,
        velocity,
        iteration,
    ):
        """Update the velocity and position of each agent.

        Parameters
        ----------
        population : ndarray of zeros and ones
            The population defined by the agents.
        acceleration : ndarray of floats
            The acceleration of each agent.
        velocity : ndarray of floats
            The velocity of each agent.
        iteration : int
            The current iteration.

        Returns
        -------
        velocity : ndarray of floats
            The updated velocity of each agent.
        population : ndarray of zeros and ones
            The updated population defined by the agents.

        """
        c_factor_local_best = -2 * ((iteration**3) / (self.maxiter**3)) + 2
        c_factor_global_best = 2 * ((iteration**3) / (self.maxiter**3)) + 2
        global_best = np.repeat(self.optimal_model, self.n_agents, axis=0).reshape(
            self.dimension, self.n_agents
        )

        velocity = (
            np.random.rand(self.dimension, self.n_agents) * velocity
            + c_factor_local_best * acceleration
            + c_factor_global_best * (global_best - population)
        )
        r = np.random.rand(self.dimension, self.n_agents)
        transform_to_binary = np.absolute(np.tanh((velocity)))
        ind = np.where(r < transform_to_binary)
        population[ind] = 1 - population[ind]
        return velocity, population
