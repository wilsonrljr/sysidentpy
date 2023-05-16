import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_raises

from sysidentpy.basis_function._basis_function import Fourier, Polynomial
from sysidentpy.model_structure_selection.accelerated_orthogonal_least_squares import (
    AOLS,
)


def create_test_data(n=1000):
    # np.random.seed(42)
    # x = np.random.uniform(-1, 1, n).T
    # y = np.zeros((n, 1))
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
        "k": 1,
        "L": 1,
        "threshold": 10e-10,
        "model_type": "NARMAX",
    }
    model = AOLS(basis_function=Polynomial(degree=2))
    model_values = [
        model.ylag,
        model.xlag,
        model.k,
        model.L,
        model.threshold,
        model.model_type,
    ]
    print(model_values)
    assert list(default.values()) == model_values


def test_validate_ylag():
    assert_raises(ValueError, AOLS, ylag=-1, basis_function=Polynomial(degree=2))
    assert_raises(ValueError, AOLS, ylag=1.3, basis_function=Polynomial(degree=2))


def test_validate_xlag():
    assert_raises(ValueError, AOLS, xlag=-1, basis_function=Polynomial(degree=2))
    assert_raises(ValueError, AOLS, xlag=1.3, basis_function=Polynomial(degree=2))


def test_k():
    assert_raises(ValueError, AOLS, k=-1, basis_function=Polynomial(degree=2))
    assert_raises(ValueError, AOLS, k=1.3, basis_function=Polynomial(degree=2))


def test_n_terms():
    assert_raises(ValueError, AOLS, L=1.2, basis_function=Polynomial(degree=2))
    assert_raises(ValueError, AOLS, L=-1, basis_function=Polynomial(degree=2))


def test_threshold():
    assert_raises(ValueError, AOLS, threshold=-1.2, basis_function=Polynomial(degree=2))
    assert_raises(ValueError, AOLS, threshold=-1, basis_function=Polynomial(degree=2))


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
    model = AOLS(ylag=[1, 2], xlag=2, basis_function=basis_function)
    model.fit(X=X_train, y=y_train)
    assert_raises(Exception, model.predict, X=X_test, y=y_test[:1])


def test_model_predict_fourier_steps_none():
    x, y, _ = create_test_data()
    basis_function = Fourier(degree=2, n=1)
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
    model = AOLS(
        ylag=[1, 2],
        xlag=2,
        basis_function=basis_function,
    )
    model.fit(X=X_train, y=y_train)
    yhat = model._basis_function_predict(X=X_test, y_initial=y_test)
    assert_almost_equal(yhat.mean(), y_test.mean(), decimal=1)


def test_model_predict_fourier_steps_1():
    x, y, _ = create_test_data()
    basis_function = Fourier(degree=2, n=1)
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
    model = AOLS(
        ylag=[1, 2],
        xlag=2,
        basis_function=basis_function,
    )
    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test, steps_ahead=1)
    assert_almost_equal(yhat.mean(), y_test.mean(), decimal=1)


def test_model_predict_fourier_nar_inputs():
    x, y, _ = create_test_data()
    basis_function = Fourier(degree=2, n=1)
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
    model = AOLS(
        ylag=[1, 2],
        xlag=2,
        basis_function=basis_function,
        model_type="NAR",
    )
    model.fit(X=X_train, y=y_train)
    model.predict(X=X_test, y=y_test)
    assert_equal(model.n_inputs, 0)


def test_model_predict_fourier_raises():
    x, y, _ = create_test_data()
    basis_function = Fourier(degree=2, n=1)
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
    model = AOLS(
        ylag=[1, 2],
        xlag=2,
        basis_function=basis_function,
        model_type="NARMAX",
    )
    model.fit(X=X_train, y=y_train)
    assert_raises(
        Exception, model._basis_function_n_step_prediction, X=X_test, y=y_test[:1]
    )
