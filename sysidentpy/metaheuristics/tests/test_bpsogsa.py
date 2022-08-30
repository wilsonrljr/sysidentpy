import random

from numpy.testing import assert_equal

from sysidentpy.metaheuristics import BPSOGSA

random.seed(42)


def test_validate():
    r = BPSOGSA(maxiter=5000, n_agents=10, dimension=3).optimize().optimal_model
    assert_equal(r, [0, 0, 0])
