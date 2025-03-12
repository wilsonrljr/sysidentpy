import numpy as np
from numpy.testing import (
    assert_almost_equal,
    assert_array_equal,
    assert_equal,
    assert_raises,
)

from sysidentpy.basis_function import Polynomial, Fourier
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.narmax_base import (
    RegressorDictionary,
    Orthogonalization,
    InformationMatrix,
)
from sysidentpy.parameter_estimation.estimators import (
    LeastSquares,
    RecursiveLeastSquares,
)
from sysidentpy.utils.generate_data import get_miso_data, get_siso_data

IM = InformationMatrix()
HH = Orthogonalization()
GR = RegressorDictionary()
bf_polynomial = Polynomial(degree=2)
bf_fourier = Fourier(degree=2, n=1)


def create_test_data():
    r"""Load test data from an external source.

    The dataset is based on a nonlinear autoregressive model
     with exogenous inputs (NARX) given by:

    $$
    y[k] = \theta_4 y[k-1]^2 + \theta_2 y[k-1] x[k-1] + \theta_0 x[k-2]
          + \theta_3 y[k-2] x[k-2] + \theta_1 y[k-2]
    $$

    where:
    - $ x[k] $ is the input at time step $ k $
    - $ y[k] $ is the output at time step $ k $
    - $ \theta = [\theta_0, \theta_1, \theta_2, \theta_3, \theta_4] $
     are model parameters

    Returns
    -------
        x (numpy.ndarray): Input data of shape $ (n, 1) $.
        y (numpy.ndarray): Output data of shape $ (n, 1) $.
        $\theta$ (numpy.ndarray): Model parameters.

    """
    theta = np.array([[0.6], [-0.5], [0.7], [-0.7], [0.2]])

    # Load dataset from external source
    url = "https://raw.githubusercontent.com/wilsonrljr/sysidentpy-data/refs/heads/main/datasets/testing/data_for_testing.txt"
    data = np.loadtxt(url)

    # Extract input (x) and output (y)
    xt = data[:, 0].reshape(-1, 1)
    yt = data[:, 1].reshape(-1, 1)

    return xt, yt, theta


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


def test_create_narmax_code():
    output1 = np.array([2001, 2002]), ([1001, 1002])
    r1 = RegressorDictionary(
        xlag=2, ylag=2, basis_function=Polynomial(degree=1)
    ).create_narmax_code(n_inputs=1)
    assert_array_equal(output1, r1)


def test_regressor_space():
    output1 = np.array([[0], [1001], [1002], [2001], [2002]])
    r1 = RegressorDictionary(
        xlag=2, ylag=2, basis_function=Polynomial(degree=1)
    ).regressor_space(n_inputs=1)
    assert_array_equal(output1, r1)
    output2 = np.array(
        [
            [0, 0],
            [1001, 0],
            [1002, 0],
            [2001, 0],
            [2002, 0],
            [1001, 1001],
            [1002, 1001],
            [2001, 1001],
            [2002, 1001],
            [1002, 1002],
            [2001, 1002],
            [2002, 1002],
            [2001, 2001],
            [2002, 2001],
            [2002, 2002],
        ]
    )
    r2 = RegressorDictionary(
        xlag=2, ylag=2, basis_function=Polynomial(degree=2)
    ).regressor_space(n_inputs=1)
    assert_array_equal(output2, r2)
    output3 = np.array(
        [
            [0, 0],
            [1001, 0],
            [1002, 0],
            [2001, 0],
            [2002, 0],
            [3001, 0],
            [3002, 0],
            [1001, 1001],
            [1002, 1001],
            [2001, 1001],
            [2002, 1001],
            [3001, 1001],
            [3002, 1001],
            [1002, 1002],
            [2001, 1002],
            [2002, 1002],
            [3001, 1002],
            [3002, 1002],
            [2001, 2001],
            [2002, 2001],
            [3001, 2001],
            [3002, 2001],
            [2002, 2002],
            [3001, 2002],
            [3002, 2002],
            [3001, 3001],
            [3002, 3001],
            [3002, 3002],
        ]
    )
    r3 = RegressorDictionary(
        xlag=[[1, 2], [1, 2]], ylag=2, basis_function=Polynomial(degree=2)
    ).regressor_space(n_inputs=2)
    assert_array_equal(output3, r3)


def test_house():
    a = np.array(
        [
            0.42544384,
            0.39365905,
            0.22209413,
            0.69760074,
            0.88183369,
            0.24818225,
            0.78482346,
            0.26967285,
            0.53987842,
            0.17367185,
        ]
    )

    output = np.array(
        [
            1,
            0.18970318,
            0.10702653,
            0.33617182,
            0.42495315,
            0.11959832,
            0.3782042,
            0.12995458,
            0.26016588,
            0.08369197,
        ]
    )
    assert_almost_equal(HH.house(a), output)


def test_row_house():
    a = np.array(
        [
            0.42544384,
            0.39365905,
            0.22209413,
            0.69760074,
            0.88183369,
            0.24818225,
            0.78482346,
            0.26967285,
            0.53987842,
            0.17367185,
        ]
    ).reshape(-1, 1)

    b = np.array(
        [
            0.90009285,
            0.21392929,
            0.58429212,
            0.55761456,
            0.65178413,
            0.4061564,
            0.4353402,
            0.02365408,
            0.52291863,
            0.185921,
        ]
    ).reshape(-1, 1)

    output = np.array(
        [
            [-1.1861246],
            [0.01063002],
            [-0.82404988],
            [-0.30077851],
            [-0.28515117],
            [-0.47901921],
            [0.00536996],
            [0.22732148],
            [-0.39637961],
            [-0.15920982],
        ]
    )
    assert_almost_equal(HH.rowhouse(a, b), output)


def test_get_index_from_regressor_code():
    model = np.array(
        [
            [1001, 0],  # y(k-1)
            [2001, 1001],  # x1(k-1)y(k-1)
            [2002, 0],  # x1(k-2)
        ]
    )

    regressor_space = np.array(
        [
            [0, 0],
            [1001, 0],
            [2001, 0],
            [2002, 0],
            [1001, 1001],
            [2001, 1001],
            [2002, 1001],
            [2001, 2001],
            [2002, 2001],
            [2002, 2002],
        ]
    )
    index = RegressorDictionary(
        xlag=2, ylag=2, basis_function=Polynomial(degree=2)
    ).get_index_from_regressor_code(regressor_code=regressor_space, model_code=model)

    assert (index == np.array([1, 3, 5])).all()


def test_list_output_regressor():
    model = np.array(
        [
            [1001, 0],  # y(k-1)
            [2001, 1001],  # x1(k-1)y(k-1)
            [2002, 0],  # x1(k-2)
        ]
    )

    y_code = RegressorDictionary(
        xlag=2, ylag=2, basis_function=Polynomial(degree=2)
    ).list_output_regressor_code(model)
    assert (y_code == np.array([1001, 1001])).all()


def test_list_input_regressor():
    model = np.array(
        [
            [1001, 0],  # y(k-1)
            [2001, 1001],  # x1(k-1)y(k-1)
            [2002, 0],  # x1(k-2)
        ]
    )

    x_code = RegressorDictionary(
        xlag=2, ylag=1, basis_function=Polynomial(degree=2)
    ).list_input_regressor_code(model)
    assert (x_code == np.array([2001, 2002])).all()


def test_get_lag_from_regressor_code():
    list_regressor1 = np.array([2001, 2002])
    list_regressor2 = np.array([1004, 1002])
    max_lag1 = RegressorDictionary(
        xlag=2, ylag=2, basis_function=Polynomial(degree=1)
    ).get_lag_from_regressor_code(list_regressor1)
    max_lag2 = RegressorDictionary(
        xlag=2, ylag=2, basis_function=Polynomial(degree=1)
    ).get_lag_from_regressor_code(list_regressor2)

    assert max_lag1 == 2
    assert max_lag2 == 4


def test_get_max_lag():
    output1 = 1
    r = RegressorDictionary(
        xlag=1, ylag=1, basis_function=Polynomial(degree=1)
    )._get_max_lag()
    output2 = 3
    r2 = RegressorDictionary(
        xlag=1, ylag=3, basis_function=Polynomial(degree=1)
    )._get_max_lag()
    assert_equal(output1, r)
    assert_equal(output2, r2)


def test_shift_column():
    a = np.array(
        [
            0.42544384,
            0.39365905,
            0.22209413,
            0.69760074,
            0.88183369,
            0.24818225,
            0.78482346,
            0.26967285,
            0.53987842,
            0.17367185,
        ]
    ).reshape(-1, 1)

    output = np.array(
        [
            [0.0],
            [0.0],
            [0.42544384],
            [0.39365905],
            [0.22209413],
            [0.69760074],
            [0.88183369],
            [0.24818225],
            [0.78482346],
            [0.26967285],
        ]
    )
    r = IM.shift_column(a, 2)
    assert_almost_equal(output, r)


def test_process_xlag():
    a = np.array(
        [
            0.42544384,
            0.39365905,
            0.22209413,
            0.69760074,
            0.88183369,
            0.24818225,
            0.78482346,
            0.26967285,
            0.53987842,
            0.17367185,
        ]
    ).reshape(-1, 1)

    n_inputs, xlag = InformationMatrix(xlag=2)._process_xlag(a.reshape(-1, 1))
    output1 = 1
    output2 = list(range(1, 3))
    assert_equal(output1, n_inputs)
    assert_equal(output2, xlag)


def test_process_ylag():
    ylag = InformationMatrix(ylag=2)._process_ylag()
    output1 = list(range(1, 3))
    assert_equal(output1, ylag)


def test_errors():
    assert_raises(
        ValueError,
        RegressorDictionary(
            xlag=2, ylag=2, basis_function=Polynomial(degree=-1)
        ).create_narmax_code,
        n_inputs=1,
    )
    assert_raises(
        ValueError,
        RegressorDictionary(
            xlag=2, ylag=-2, basis_function=Polynomial(degree=1)
        ).create_narmax_code,
        n_inputs=1,
    )
    assert_raises(
        ValueError,
        RegressorDictionary(
            xlag=-2, ylag=2, basis_function=Polynomial(degree=1)
        ).create_narmax_code,
        n_inputs=1,
    )
    assert_raises(
        ValueError,
        RegressorDictionary(
            xlag=2, ylag=2, basis_function=Polynomial(degree=1)
        ).create_narmax_code,
        n_inputs=0,
    )


def test_create_narmax_code_ylist():
    output1 = np.array([2001, 2002]), ([1001, 1002])
    r1 = RegressorDictionary(
        xlag=2, ylag=[1, 2], basis_function=Polynomial(degree=1)
    ).create_narmax_code(n_inputs=1)
    assert_array_equal(output1, r1)


def test_create_narmax_code_xlist():
    output1 = np.array([2001, 2002]), ([1001, 1002])
    r1 = RegressorDictionary(
        xlag=[1, 2], ylag=2, basis_function=Polynomial(degree=1)
    ).create_narmax_code(n_inputs=1)
    assert_array_equal(output1, r1)


def test_create_narmax_code_miso():
    output1 = np.concatenate(
        np.array(
            [np.array([2001, 2002, 3001, 3002]), np.array([1001, 1002])], dtype=object
        )
    )
    r1 = RegressorDictionary(
        xlag=[[1, 2], [1, 2]], ylag=2, basis_function=Polynomial(degree=1)
    ).create_narmax_code(n_inputs=2)
    assert_array_equal(output1, np.concatenate(r1))


def test_regressor_space_raise():
    assert_raises(
        ValueError,
        RegressorDictionary(
            xlag=2, ylag=2, basis_function=Polynomial(degree=1), model_type="NARARMAX"
        ).regressor_space,
        n_inputs=1,
    )


def test_model_information_get_lag():
    laglist = np.array([2001, 2002, 3001, 3002, 1001, 1002])
    output = 2
    r1 = RegressorDictionary().get_lag_from_regressor_code(laglist)
    assert r1 == output


def test_model_information_empty_list():
    laglist = np.array([])
    output = 1
    r1 = RegressorDictionary().get_lag_from_regressor_code(laglist)
    assert r1 == output


def test_get_max_lag_from_model_code():
    model = np.array(
        [
            [1001, 0],  # y(k-1)
            [2001, 1001],  # x1(k-1)y(k-1)
            [2002, 0],  # x1(k-2)
        ]
    )
    assert RegressorDictionary().get_max_lag_from_model_code(model) == 2


def test_process_lag():
    x_train, _, _, _ = get_miso_data(
        n=10, colored_noise=False, sigma=0.001, train_percentage=90
    )
    assert_raises(ValueError, InformationMatrix(xlag=2)._process_xlag, X=x_train)


def test_process_lag_n1():
    x_train, _, _, _ = get_siso_data(
        n=10, colored_noise=False, sigma=0.001, train_percentage=90
    )
    n_inputs, xlag = InformationMatrix(xlag=2)._process_xlag(X=x_train)
    assert n_inputs == 1
    assert list(xlag) == [1, 2]


def test_create_lagged_x():
    X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
    r = InformationMatrix(xlag=[1, 2])._create_lagged_X(X=X, n_inputs=1)
    assert_equal(
        r,
        np.array(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0], [4.0, 3.0], [5.0, 4.0]]
        ),
    )


def test_create_lagged_x_miso():
    X = np.array(range(1, 13)).reshape(-1, 2)
    r = InformationMatrix(xlag=[[1, 2], [1, 2]])._create_lagged_X(X=X, n_inputs=2)
    assert_equal(
        r,
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 2.0, 0.0],
                [3.0, 1.0, 4.0, 2.0],
                [5.0, 3.0, 6.0, 4.0],
                [7.0, 5.0, 8.0, 6.0],
                [9.0, 7.0, 10.0, 8.0],
            ]
        ),
    )


def test_model_predict():
    model = FROLS(
        n_terms=5,
        err_tol=None,
        ylag=[1, 2],
        xlag=2,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=2),
    )
    model.fit(X=X_train, y=y_train)
    print(model.final_model, model.err.sum())
    yhat = model.predict(X=X_test, y=y_test)
    assert_almost_equal(yhat, y_test, decimal=10)


def test_model_nfir():
    model = FROLS(
        n_terms=5,
        # extended_least_squares=False,
        xlag=2,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=2),
        model_type="NFIR",
    )
    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test)
    assert_almost_equal(yhat.mean(), y_test.mean(), decimal=2)


def test_model_predict_steps_none():
    model = FROLS(
        n_terms=5,
        err_tol=None,
        ylag=[1, 2],
        xlag=2,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=2),
    )
    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test, steps_ahead=1)
    assert_almost_equal(yhat, y_test, decimal=10)


def test_model_predict_steps_3():
    model = FROLS(
        n_terms=5,
        err_tol=None,
        ylag=[1, 2],
        xlag=2,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=2),
    )
    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test, steps_ahead=3)
    assert_almost_equal(yhat, y_test, decimal=10)


def test_model_predict_fourier_steps_none():
    model = FROLS(
        order_selection=True,
        err_tol=None,
        ylag=[1, 2],
        xlag=2,
        estimator=RecursiveLeastSquares(),
        basis_function=Fourier(degree=2, n=1),
    )
    model.fit(X=X_train, y=y_train)
    yhat = model._basis_function_predict(X=X_test, y_initial=y_test)
    assert_almost_equal(yhat.mean(), y_test[model.max_lag : :].mean(), decimal=6)


def test_model_predict_fourier_steps_1():
    model = FROLS(
        order_selection=True,
        err_tol=None,
        ylag=[1, 2],
        xlag=2,
        estimator=RecursiveLeastSquares(),
        basis_function=Fourier(degree=2, n=1),
    )
    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test, steps_ahead=1)
    assert_almost_equal(yhat.mean(), y_test.mean(), decimal=6)


def test_model_predict_fourier_nar_inputs():
    model = FROLS(
        order_selection=True,
        # extended_least_squares=False,
        ylag=[1, 2],
        xlag=2,
        estimator=RecursiveLeastSquares(),
        basis_function=Fourier(degree=2, n=1),
        model_type="NAR",
    )
    model.fit(X=X_train, y=y_train)
    model.predict(X=X_test, y=y_test)
    assert_equal(model.n_inputs, 0)


def test_model_predict_fourier_raises():
    model = FROLS(
        order_selection=True,
        # extended_least_squares=False,
        ylag=[1, 2],
        xlag=2,
        estimator=RecursiveLeastSquares(),
        basis_function=Fourier(degree=2, n=1),
        model_type="NARMAX",
    )
    model.fit(X=X_train, y=y_train)
    assert_raises(
        Exception, model._basis_function_n_step_prediction, X=X_test, y=y_test[:1]
    )


def test_model_predict_fourier_value_error():
    model = FROLS(
        order_selection=True,
        # extended_least_squares=False,
        ylag=[1, 2],
        xlag=2,
        estimator=RecursiveLeastSquares(),
        basis_function=Fourier(degree=2, n=1),
        model_type="NARMAX",
    )
    model.fit(X=X_train, y=y_train)
    model.model_type = "NARRARMAX"
    assert_raises(
        ValueError,
        model._basis_function_n_step_prediction,
        X=X_test,
        y=y_test,
        steps_ahead=1,
        forecast_horizon=None,
    )


def test_model_predict_fourier_horizon_error():
    model = FROLS(
        order_selection=True,
        # extended_least_squares=False,
        ylag=[1, 2],
        xlag=2,
        estimator=RecursiveLeastSquares(),
        basis_function=Fourier(degree=2, n=1),
        model_type="NARMAX",
    )
    model.fit(X=X_train, y=y_train)
    model.model_type = "NARRARMAX"
    assert_raises(
        ValueError,
        model._basis_function_n_steps_horizon,
        X=X_test,
        y=y_test,
        steps_ahead=1,
        forecast_horizon=10,
    )
