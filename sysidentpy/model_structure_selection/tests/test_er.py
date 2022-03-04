import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from numpy.testing import assert_raises
from numpy.testing._private.utils import assert_allclose
from sysidentpy.model_structure_selection import ER
from sysidentpy.utils.generate_data import get_miso_data, get_siso_data
from sysidentpy.basis_function import Polynomial


def create_test_data(n=1000):
    theta = np.array([[0.6], [-0.5], [0.7], [-0.7], [0.2]])
    # lag = 2
    # for k in range(lag, len(x)):
    #     y[k] = theta[4]*y[k-1]**2 + theta[2]*y[k-1]*x[k-1] + theta[0]*x[k-2] \
    #         + theta[3]*y[k-2]*x[k-2] + theta[1]*y[k-2]

    # y = np.reshape(y, (len(y), 1))
    # x = np.reshape(x, (len(x), 1))
    # data = np.concatenate([x, y], axis=1)
    data = np.loadtxt("examples/datasets/data_for_testing.txt")
    x = data[:, 0].reshape(-1, 1)
    y = data[:, 1].reshape(-1, 1)
    return x, y, theta


def test_default_values():
    default = {
        "ylag": 2,
        "xlag": 2,
        "estimator": "least_squares",
        "extended_least_squares": False,
        "q": 0.99,
        "h": 0.01,
        "k": 2,
        "mutual_information_estimator": "mutual_information_knn",
        "n_perm": 200,
        "p": np.inf,
        "skip_forward": False,
        "lam": 0.98,
        "delta": 0.01,
        "offset_covariance": 0.2,
        "mu": 0.01,
        "eps": np.finfo(np.float64).eps,
        "gama": 0.2,
        "weight": 0.02,
        "model_type": "NARMAX",
        "random_state": None,
    }
    model = ER(basis_function=Polynomial(degree=2))
    model_values = [
        model.ylag,
        model.xlag,
        model.estimator,
        model._extended_least_squares,
        model.q,
        model.h,
        model.k,
        model.mutual_information_estimator,
        model.n_perm,
        model.p,
        model.skip_forward,
        model._lam,
        model._delta,
        model._offset_covariance,
        model._mu,
        model._eps,
        model._gama,
        model._weight,
        model.model_type,
        model.random_state,
    ]
    assert list(default.values()) == model_values


def test_validate_ylag():
    assert_raises(ValueError, ER, ylag=-1, basis_function=Polynomial(degree=2))
    assert_raises(ValueError, ER, ylag=1.3, basis_function=Polynomial(degree=2))


def test_validate_xlag():
    assert_raises(ValueError, ER, xlag=-1, basis_function=Polynomial(degree=2))
    assert_raises(ValueError, ER, xlag=1.3, basis_function=Polynomial(degree=2))


def test_k():
    assert_raises(ValueError, ER, k=-1, basis_function=Polynomial(degree=2))
    assert_raises(ValueError, ER, k=1.3, basis_function=Polynomial(degree=2))


def test_n_perm():
    assert_raises(ValueError, ER, n_perm=-1, basis_function=Polynomial(degree=2))
    assert_raises(ValueError, ER, n_perm=1.3, basis_function=Polynomial(degree=2))


def test_q():
    assert_raises(ValueError, ER, q=-1, basis_function=Polynomial(degree=2))
    assert_raises(ValueError, ER, q=1.3, basis_function=Polynomial(degree=2))
