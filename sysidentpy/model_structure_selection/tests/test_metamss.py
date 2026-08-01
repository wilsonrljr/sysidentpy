import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
from numpy.testing import assert_raises

from sysidentpy import config_context
from sysidentpy.basis_function import Polynomial
from sysidentpy.model_structure_selection import MetaMSS
from sysidentpy.parameter_estimation.estimators import LeastSquares
from sysidentpy.tests.test_narmax_base import create_test_data
from sysidentpy.utils.generate_data import get_siso_data

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


def test_metamss():
    # piv = np.array([4, 2, 7, 11, 5])
    model_code = np.array(
        [[1001, 0], [2002, 0], [2001, 1001]]  # y(k-1)  # x1(k-2)  # x1(k-1)y(k-1)
    )
    basis_function = Polynomial(degree=2)
    global_rng_state = np.random.get_state()
    try:
        np.random.seed(42)
        x_train, _x_test, y_train, _y_test = get_siso_data(
            n=1000, colored_noise=False, sigma=0.0001, train_percentage=90
        )
    finally:
        np.random.set_state(global_rng_state)

    model = MetaMSS(
        ylag=[1, 2],
        xlag=2,
        maxiter=30,
        n_agents=20,
        basis_function=basis_function,
        random_state=42,
        test_size=0.1,
    )
    model.fit(X=x_train, y=y_train)
    assert_array_equal(model.final_model, model_code)


def test_default_values():
    default = {
        "ylag": 1,
        "xlag": 1,
        "model_type": "NARMAX",
        "maxiter": 30,
        "alpha": 23,
        "g_zero": 100,
        "k_agents_percent": 2,
        "norm": -2,
        "power": 2,
        "n_agents": 10,
        "p_zeros": 0.5,
        "p_ones": 0.5,
        "p_value": 0.05,
        "eps": np.finfo(np.float64).eps,
        "steps_ahead": None,
        "estimate_parameter": True,
        "loss_func": "metamss_loss",
        "random_state": None,
    }
    model = MetaMSS(basis_function=Polynomial(degree=2))
    model_values = [
        model.ylag,
        model.xlag,
        model.model_type,
        model.maxiter,
        model.alpha,
        model.g_zero,
        model.k_agents_percent,
        model.norm,
        model.power,
        model.n_agents,
        model.p_zeros,
        model.p_ones,
        model.p_value,
        model.eps,
        model.steps_ahead,
        model.estimate_parameter,
        model.loss_func,
        model.random_state,
    ]
    assert list(default.values()) == model_values
    assert isinstance(model.estimator, LeastSquares)
    assert isinstance(model.basis_function, Polynomial)


def test_validate_ylag():
    assert_raises(ValueError, MetaMSS, ylag=-1, basis_function=Polynomial(degree=2))
    assert_raises(ValueError, MetaMSS, ylag=1.3, basis_function=Polynomial(degree=2))


def test_validate_xlag():
    assert_raises(ValueError, MetaMSS, xlag=-1, basis_function=Polynomial(degree=2))
    assert_raises(ValueError, MetaMSS, xlag=1.3, basis_function=Polynomial(degree=2))


def test_predict():
    X_train, X_test, y_train, y_test = get_siso_data(
        n=1000, colored_noise=False, sigma=0.0001, train_percentage=90
    )
    basis_function = Polynomial(degree=2)
    model = MetaMSS(
        ylag=[1, 2],
        xlag=2,
        maxiter=30,
        n_agents=10,
        basis_function=basis_function,
        random_state=42,
        test_size=0.1,
    )
    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test)
    assert_almost_equal(yhat, y_test, decimal=2)


def test_model_prediction():
    model = MetaMSS(
        ylag=[1, 2],
        xlag=2,
        maxiter=30,
        n_agents=20,
        basis_function=Polynomial(degree=2),
        random_state=42,
        test_size=0.1,
    )
    model.fit(X=X_train, y=y_train)
    assert_raises(Exception, model.predict, X=X_test, y=y_test[:1])


def test_metamss_rejects_array_api_dispatch_with_clear_error():
    xp = pytest.importorskip("array_api_strict")
    model = MetaMSS(
        ylag=1,
        xlag=1,
        maxiter=1,
        n_agents=2,
        basis_function=Polynomial(degree=1),
        random_state=0,
    )

    with config_context(array_api_dispatch=True):
        with pytest.raises(NotImplementedError, match=r"MetaMSS.*requires NumPy"):
            model.fit(X=xp.asarray(X_train[:10]), y=xp.asarray(y_train[:10]))


def test_metamss_same_seed_reproduces_optimization_history():
    x_train, _x_test, y_train, _y_test = get_siso_data(
        n=120, colored_noise=False, sigma=0.001, train_percentage=90
    )
    configuration = {
        "ylag": 2,
        "xlag": 2,
        "maxiter": 2,
        "n_agents": 4,
        "basis_function": Polynomial(degree=2),
        "random_state": 42,
        "test_size": 0.2,
    }

    first = MetaMSS(**configuration).fit(X=x_train, y=y_train)
    second = MetaMSS(**configuration).fit(X=x_train, y=y_train)

    assert_array_equal(first.final_model, second.final_model)
    assert_almost_equal(first.best_by_iter, second.best_by_iter)
    assert_almost_equal(first.mean_by_iter, second.mean_by_iter)


def test_metamss_integer_seed_restarts_repeated_fit_on_same_instance():
    x_train, _x_test, y_train, _y_test = get_siso_data(
        n=120, colored_noise=False, sigma=0.001, train_percentage=90
    )
    model = MetaMSS(
        ylag=2,
        xlag=2,
        maxiter=2,
        n_agents=4,
        basis_function=Polynomial(degree=2),
        random_state=42,
        test_size=0.2,
    )

    model.fit(X=x_train, y=y_train)
    first_model = model.final_model.copy()
    first_best = np.asarray(model.best_by_iter).copy()
    first_mean = np.asarray(model.mean_by_iter).copy()
    first_dimension = model.dimension
    first_space = model.regressor_code.copy()
    model.fit(X=x_train, y=y_train)

    assert_array_equal(model.final_model, first_model)
    assert_array_equal(model.regressor_code, first_space)
    assert model.dimension == first_dimension
    assert_almost_equal(model.best_by_iter, first_best)
    assert_almost_equal(model.mean_by_iter, first_mean)


def test_metamss_generator_advances_across_repeated_fit():
    random_state = np.random.default_rng(42)
    model = MetaMSS(
        ylag=2,
        xlag=2,
        maxiter=1,
        n_agents=3,
        basis_function=Polynomial(degree=2),
        random_state=random_state,
        test_size=0.2,
    )
    state_before = repr(random_state.bit_generator.state)

    model.fit(X=X_train, y=y_train)
    state_after_first_fit = repr(random_state.bit_generator.state)
    model.fit(X=X_train, y=y_train)
    state_after_second_fit = repr(random_state.bit_generator.state)

    assert state_before != state_after_first_fit
    assert state_after_first_fit != state_after_second_fit


def test_metamss_rejects_zero_probability_of_nonempty_model():
    with pytest.raises(ValueError, match="requires p_ones > 0"):
        MetaMSS(p_zeros=1, p_ones=0)
