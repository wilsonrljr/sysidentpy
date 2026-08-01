import numpy as np
import pytest

from numpy.testing import assert_allclose, assert_array_equal, assert_equal

from sysidentpy.metaheuristics import BPSOGSA


def test_validate():
    r = (
        BPSOGSA(
            maxiter=100,
            n_agents=10,
            dimension=3,
            random_state=42,
        )
        .optimize()
        .optimal_model
    )
    assert_equal(r, [0, 0, 0])


def test_same_integer_seed_reproduces_full_trajectory():
    first = BPSOGSA(maxiter=10, n_agents=6, dimension=5, random_state=42).optimize()
    second = BPSOGSA(maxiter=10, n_agents=6, dimension=5, random_state=42).optimize()

    assert_array_equal(first.optimal_model, second.optimal_model)
    assert_allclose(first.best_by_iter, second.best_by_iter)
    assert_allclose(first.mean_by_iter, second.mean_by_iter)


def test_integer_seed_restarts_repeated_optimize_calls():
    optimizer = BPSOGSA(maxiter=10, n_agents=6, dimension=5, random_state=42)

    optimizer.optimize()
    first_history = np.asarray(optimizer.mean_by_iter).copy()
    optimizer.optimize()

    assert_allclose(optimizer.mean_by_iter, first_history)


def test_optimize_supports_state_loaded_from_legacy_pickle():
    optimizer = BPSOGSA(maxiter=1, n_agents=3, dimension=2)
    del optimizer.random_state
    del optimizer._rng

    optimizer.optimize()

    assert optimizer.random_state is None
    assert optimizer.optimal_model is not None


def test_seeded_optimizer_does_not_depend_on_global_numpy_state():
    global_state = np.random.get_state()
    try:
        np.random.seed(0)
        first = BPSOGSA(maxiter=8, n_agents=5, dimension=4, random_state=7).optimize()
        np.random.seed(1234)
        second = BPSOGSA(maxiter=8, n_agents=5, dimension=4, random_state=7).optimize()
    finally:
        np.random.set_state(global_state)

    assert_allclose(first.mean_by_iter, second.mean_by_iter)


def test_generator_instances_advance_between_population_draws():
    random_states = [np.random.default_rng(42), np.random.RandomState(42)]
    for random_state in random_states:
        optimizer = BPSOGSA(n_agents=8, dimension=8, random_state=random_state)

        first = optimizer.generate_random_population()
        second = optimizer.generate_random_population()

        assert not np.array_equal(first, second)


def test_mass_calculation_handles_nonfinite_fitness_without_nan():
    optimizer = BPSOGSA(n_agents=3, dimension=2, random_state=42)

    masses = optimizer.mass_calculation([1.0, np.inf, np.nan])

    assert np.all(np.isfinite(masses))


def test_mass_calculation_handles_extreme_finite_fitness_without_overflow():
    optimizer = BPSOGSA(n_agents=3, dimension=2, random_state=42)

    masses = optimizer.mass_calculation([-1e308, 0.0, 1e308])

    assert np.all(np.isfinite(masses))
    assert_allclose(np.sum(masses), 5.0)


def test_mass_calculation_assigns_nonnegative_mass_and_zero_to_worst_agent():
    optimizer = BPSOGSA(n_agents=4, dimension=2, random_state=42)

    masses = optimizer.mass_calculation([1.0, 2.0, 3.0, 4.0])

    assert np.all(masses >= 0)
    assert masses[0] > masses[1] > masses[2] > masses[3]
    assert masses[3] == 0
    assert_allclose(np.sum(masses), 5.0)


def test_equal_fitness_acceleration_is_not_biased_to_first_agent():
    optimizer = BPSOGSA(
        n_agents=3,
        dimension=2,
        maxiter=1,
        k_agents_percent=100,
        random_state=42,
    )
    population = np.array([[0, 1, 1], [0, 0, 1]])
    masses = optimizer.mass_calculation([1.0, 1.0, 1.0])

    acceleration = optimizer.calculate_acceleration(
        population, masses, gravitational_constant=1.0, iteration=0
    )

    assert masses.shape == (3,)
    assert np.any(acceleration[:, 0] != 0)


def test_optimize_ignores_nonfinite_candidates_without_nonfinite_state():
    class MixedFitnessBPSOGSA(BPSOGSA):
        def evaluate_objective_function(self, candidate_solution):
            return np.array([1.0, np.nan, 2.0])

    optimizer = MixedFitnessBPSOGSA(
        maxiter=1, n_agents=3, dimension=2, random_state=42
    ).optimize()

    assert np.isfinite(optimizer.optimal_fitness_value)
    assert np.all(np.isfinite(optimizer.best_by_iter))
    assert np.all(np.isfinite(optimizer.mean_by_iter))
    assert_allclose(optimizer.mean_by_iter, [1.5])


def test_optimize_rejects_objective_with_no_finite_candidates():
    class InvalidFitnessBPSOGSA(BPSOGSA):
        def evaluate_objective_function(self, candidate_solution):
            return np.full(self.n_agents, np.nan)

    optimizer = InvalidFitnessBPSOGSA(
        maxiter=1, n_agents=3, dimension=2, random_state=42
    )

    with pytest.raises(ValueError, match="no finite fitness"):
        optimizer.optimize()
