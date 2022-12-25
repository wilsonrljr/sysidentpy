import numpy as np
from numpy.testing import assert_almost_equal
from numpy.testing import assert_raises
from sysidentpy.model_structure_selection import ER
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
        "ylag": 1,
        "xlag": 1,
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
        model.extended_least_squares,
        model.q,
        model.h,
        model.k,
        model.mutual_information_estimator,
        model.n_perm,
        model.p,
        model.skip_forward,
        model.lam,
        model.delta,
        model.offset_covariance,
        model.mu,
        model.eps,
        model.gama,
        model.weight,
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


def test_skip_forward():
    assert_raises(TypeError, ER, skip_forward=1, basis_function=Polynomial(degree=2))
    assert_raises(
        TypeError, ER, skip_forward="True", basis_function=Polynomial(degree=2)
    )
    assert_raises(TypeError, ER, skip_forward=None, basis_function=Polynomial(degree=2))


def test_extended_least_squares():
    assert_raises(
        TypeError, ER, extended_least_squares=1, basis_function=Polynomial(degree=2)
    )
    assert_raises(
        TypeError,
        ER,
        extended_least_squares="True",
        basis_function=Polynomial(degree=2),
    )
    assert_raises(
        TypeError,
        ER,
        extended_least_squares=None,
        basis_function=Polynomial(degree=2),
    )


def test_model_prediction():
    x, y, _ = create_test_data()
    basis_function = Polynomial(degree=2)
    train_percentage = 90
    split_data = int(len(x) * (train_percentage / 100))

    X_train = x[0:split_data, 0]
    X_test = x[split_data::, 0]

    y1 = y[0:split_data, 0]
    y_test = y[split_data::, 0]
    y_train = y1.copy()

    y_train = np.reshape(y_train, (len(y_train), 1))
    X_train = np.reshape(X_train, (len(X_train), 1))

    y_test = np.reshape(y_test, (len(y_test), 1))
    X_test = np.reshape(X_test, (len(X_test), 1))
    model = ER(
        ylag=2,
        xlag=2,
        estimator="least_squares",
        basis_function=basis_function,
    )
    model.fit(X=X_train, y=y_train)
    assert_raises(Exception, model.predict, X=X_test, y=y_test[:1])


def test_mutual_information_knn():
    basis_function = Polynomial(degree=1)
    model = ER(
        ylag=2,
        xlag=2,
        estimator="least_squares",
        basis_function=basis_function,
    )
    x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    y = np.array([0.3, 0.87, 0, 0.1, 0.9]).reshape(-1, 1)

    r = model.mutual_information_knn(x, y)
    assert_almost_equal(r, 0.6000, decimal=3)


def test_conditional_mutual_information_knn():
    basis_function = Polynomial(degree=1)
    model = ER(
        ylag=2,
        xlag=2,
        estimator="least_squares",
        basis_function=basis_function,
    )
    x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    y = np.array([0.3, 0.87, 0, 0.1, 0.9]).reshape(-1, 1)
    z = np.array([90, 12, 212, 13, 15]).reshape(-1, 1)

    r = model.conditional_mutual_information(x, y, z)
    assert_almost_equal(r, 0.2, decimal=3)


def test_tolerance_estimator():
    basis_function = Polynomial(degree=1)
    model = ER(
        ylag=2,
        xlag=2,
        estimator="least_squares",
        basis_function=basis_function,
        random_state=42,
    )
    x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    r = model.tolerance_estimator(x)
    assert_almost_equal(r, 2.6833, decimal=4)
