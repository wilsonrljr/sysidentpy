import numpy as np
import torch
from numpy.testing import assert_almost_equal, assert_equal, assert_raises

from sysidentpy.basis_function import Fourier, Polynomial
from sysidentpy.neural_network import NARXNN
from sysidentpy.utils.generate_data import get_siso_data


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


x, y, _ = create_test_data()
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


def test_default_values():
    default = {
        "ylag": 1,
        "xlag": 1,
        "model_type": "NARMAX",
        "batch_size": 100,
        "learning_rate": 0.01,
        "epochs": 200,
        # "loss_func": "mse_loss",
        "optimizer": "Adam",
        "net": None,
        "train_percentage": 80,
        "verbose": False,
        "optim_params": None,
        # "device": "cpu",
    }
    model = NARXNN(basis_function=Polynomial())
    model_values = [
        model.ylag,
        model.xlag,
        model.model_type,
        model.batch_size,
        model.learning_rate,
        model.epochs,
        # model.loss_func,
        model.optimizer,
        model.net,
        model.train_percentage,
        model.verbose,
        model.optim_params,
        # model.device,
    ]
    assert list(default.values()) == model_values


def test_validate():
    assert_raises(ValueError, NARXNN, ylag=-1, basis_function=Polynomial(degree=1))
    assert_raises(ValueError, NARXNN, ylag=1.3, basis_function=Polynomial(degree=1))
    assert_raises(ValueError, NARXNN, xlag=1.3, basis_function=Polynomial(degree=1))
    assert_raises(ValueError, NARXNN, xlag=-1, basis_function=Polynomial(degree=1))


def test_fit_raise():
    assert_raises(
        ValueError,
        NARXNN,
        basis_function=Polynomial(degree=1),
        model_type="NARARMAX",
    )


def test_fit_raise_y():
    model = NARXNN(basis_function=Polynomial(degree=2))
    assert_raises(ValueError, model.fit, X=X_train, y=None)
