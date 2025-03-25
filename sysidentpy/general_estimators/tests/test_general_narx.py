import numpy as np
from sklearn.linear_model import LinearRegression
from numpy.testing import (
    assert_almost_equal,
    assert_equal,
    assert_raises,
)

from sysidentpy.basis_function import Polynomial, Fourier
from sysidentpy.general_estimators import NARX
from sysidentpy.tests.test_narmax_base import create_test_data

base_estimator = LinearRegression()

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
    }
    model = NARX(basis_function=Polynomial(degree=2))
    model_values = [
        model.ylag,
        model.xlag,
        model.model_type,
    ]
    assert list(default.values()) == model_values


def test_model_nfir():
    basis_function = Polynomial(degree=2)
    model = NARX(
        xlag=2,
        basis_function=basis_function,
        model_type="NFIR",
        base_estimator=base_estimator,
    )
    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test)
    assert_almost_equal(yhat.mean(), y_test[model.max_lag : :].mean(), decimal=1)


def test_validate():
    assert_raises(ValueError, NARX, ylag=-1, basis_function=Polynomial(degree=1))
    assert_raises(ValueError, NARX, ylag=1.3, basis_function=Polynomial(degree=1))
    assert_raises(ValueError, NARX, xlag=1.3, basis_function=Polynomial(degree=1))
    assert_raises(ValueError, NARX, xlag=-1, basis_function=Polynomial(degree=1))


def test_fit_raise():
    assert_raises(
        ValueError,
        NARX,
        base_estimator=LinearRegression(),
        basis_function=Polynomial(degree=1),
        model_type="NARARMAX",
    )


def test_fit_raise_y():
    model = NARX(basis_function=Polynomial(degree=2), base_estimator=base_estimator)
    assert_raises(ValueError, model.fit, X=X_train, y=None)


def test_fit_lag_nar():
    model = NARX(
        basis_function=Polynomial(degree=2),
        model_type="NAR",
        base_estimator=base_estimator,
        xlag=2,
        ylag=2,
    )
    model.fit(X=X_train, y=y_train)
    assert_equal(model.max_lag, 2)


def test_fit_lag_nfir():
    model = NARX(
        basis_function=Polynomial(degree=2),
        model_type="NFIR",
        base_estimator=base_estimator,
        xlag=2,
        ylag=2,
    )
    model.fit(X=X_train, y=y_train)
    assert_equal(model.max_lag, 2)


def test_fit_lag_narmax():
    model = NARX(
        basis_function=Polynomial(degree=2),
        base_estimator=base_estimator,
        xlag=2,
        ylag=2,
    )
    model.fit(X=X_train, y=y_train)
    assert_equal(model.max_lag, 2)


def test_fit_lag_narmax_fourier():
    model = NARX(
        basis_function=Fourier(degree=2), base_estimator=base_estimator, xlag=2, ylag=2
    )
    model.fit(X=X_train, y=y_train)
    assert_equal(model.max_lag, 2)


def test_model_predict():
    basis_function = Polynomial(degree=2)
    model = NARX(
        ylag=[1, 2],
        xlag=2,
        basis_function=basis_function,
        base_estimator=base_estimator,
    )
    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test)
    assert_almost_equal(yhat, y_test, decimal=10)


def test_model_predict_steps_none():
    basis_function = Polynomial(degree=2)
    model = NARX(
        ylag=[1, 2],
        xlag=2,
        basis_function=basis_function,
        base_estimator=LinearRegression(),
    )
    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test, steps_ahead=1)
    assert_almost_equal(yhat, y_test, decimal=10)


def test_model_predict_steps_3():
    basis_function = Polynomial(degree=2)
    model = NARX(
        ylag=[1, 2],
        xlag=2,
        basis_function=basis_function,
        base_estimator=LinearRegression(),
    )
    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test, steps_ahead=3)
    assert_almost_equal(yhat, y_test, decimal=10)


def test_model_predict_fourier_steps_1():
    basis_function = Fourier(degree=2, n=1)
    model = NARX(
        ylag=[1, 2],
        xlag=2,
        basis_function=basis_function,
        base_estimator=LinearRegression(),
    )
    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test, steps_ahead=1)
    assert_almost_equal(yhat.mean(), 0.0016457328739105236, decimal=6)


def test_model_predict_fourier_nar_inputs():
    basis_function = Fourier(degree=2, n=1)
    model = NARX(
        ylag=[1, 2],
        xlag=2,
        basis_function=basis_function,
        model_type="NAR",
        base_estimator=LinearRegression(),
    )
    model.fit(X=X_train, y=y_train)
    model.predict(X=X_test, y=y_test)
    assert_equal(model.n_inputs, 0)


def test_model_predict_fourier_raises():
    basis_function = Fourier(degree=2, n=1)
    model = NARX(
        ylag=[1, 2],
        xlag=2,
        basis_function=basis_function,
        model_type="NARMAX",
        base_estimator=LinearRegression(),
    )
    model.fit(X=X_train, y=y_train)
    assert_raises(
        Exception, model._basis_function_n_step_prediction, X=X_test, y=y_test[:1]
    )


def test_model_predict_fourier_value_error():
    basis_function = Fourier(degree=2, n=1)
    model = NARX(
        ylag=[1, 2],
        xlag=2,
        basis_function=basis_function,
        model_type="NARMAX",
        base_estimator=LinearRegression(),
    )
    model.fit(X=X_train, y=y_train)
    model.model_type = "NARRARMAX"
    assert_raises(
        ValueError,
        model._basis_function_n_step_prediction,
        x=X_test,
        y=y_test,
        steps_ahead=1,
        forecast_horizon=None,
    )


def test_model_predict_fourier_horizon_error():
    basis_function = Fourier(degree=2, n=1)
    model = NARX(
        ylag=[1, 2],
        xlag=2,
        basis_function=basis_function,
        model_type="NARMAX",
        base_estimator=LinearRegression(),
    )
    model.fit(X=X_train, y=y_train)
    model.model_type = "NARRARMAX"
    assert_raises(
        ValueError,
        model._basis_function_n_steps_horizon,
        x=X_test,
        y=y_test,
        steps_ahead=1,
        forecast_horizon=10,
    )


def test_model_predict_nfir_cat():
    basis_function = Polynomial(degree=2)
    model = NARX(
        base_estimator=base_estimator,
        xlag=10,
        ylag=10,
        basis_function=basis_function,
        model_type="NFIR",
    )

    model.fit(X=X_train, y=y_train)
    # yhat = model.predict(x=x_valid, y=y_valid)
    assert_equal(model.max_lag, 10)


def test_model_predict_steps_1():
    basis_function = Polynomial(degree=1)
    model = NARX(
        base_estimator=base_estimator,
        xlag=2,
        ylag=2,
        basis_function=basis_function,
        model_type="NARMAX",
    )

    model.fit(X=X_train, y=y_train)
    # yhat = model.predict(x=x_valid, y=y_valid, steps_ahead=1)
    assert_equal(model.max_lag, 2)


def test_model_predict_fourier_none():
    basis_function = Fourier(degree=1)
    model = NARX(
        base_estimator=base_estimator,
        xlag=10,
        ylag=10,
        basis_function=basis_function,
        model_type="NARMAX",
    )

    model.fit(X=X_train, y=y_train)
    assert_equal(model.max_lag, 10)


def test_model_predict_fourier_1():
    basis_function = Fourier(degree=1)
    model = NARX(
        base_estimator=base_estimator,
        xlag=10,
        ylag=10,
        basis_function=basis_function,
        model_type="NARMAX",
    )

    model.fit(X=X_train, y=y_train)
    assert_equal(model.max_lag, 10)


def test_model_predict_fourier_n():
    basis_function = Fourier(degree=1)
    model = NARX(
        base_estimator=base_estimator,
        xlag=10,
        ylag=10,
        basis_function=basis_function,
        model_type="NARMAX",
    )

    model.fit(X=X_train, y=y_train)
    assert_equal(model.max_lag, 10)
