from sysidentpy.polynomial_basis import PolynomialNarmax

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from numpy.testing import assert_raises


def create_test_data(n=1000):
    np.random.seed(42)
    x = np.random.uniform(-1, 1, n).T
    y = np.zeros((n, 1))
    theta = np.array([[0.6], [-0.5], [0.7], [-0.7], [0.2]])
    lag = 2
    for k in range(lag, len(x)):
        y[k] = theta[4]*y[k-1]**2 + theta[2]*y[k-1]*x[k-1] + theta[0]*x[k-2] \
               + theta[3]*y[k-2]*x[k-2] + theta[1]*y[k-2]

    y = np.reshape(y, (len(y), 1))
    x = np.reshape(x, (len(x), 1))

    return x, y, theta


def test_error_reduction_ration():
    piv = np.array([4, 2, 7, 11, 5, 0, 3, 6, 12, 8, 14, 1, 10, 13, 9])
    model_code = np.array([[2002,    0],
                           [1002,    0],
                           [2001, 1001],
                           [2002, 1002],
                           [1001, 1001],
                           [0,       0],
                           [2001,    0],
                           [1002, 1001],
                           [2001, 2001],
                           [2002, 1001],
                           [2002, 2002],
                           [1001,    0],
                           [2001, 1002],
                           [2002, 2001],
                           [1002, 1002]])
    x, y, theta = create_test_data()
    model = PolynomialNarmax(non_degree=2,
                             n_terms=15,
                             order_selection=True,
                             n_info_values=15,
                             info_criteria='aic',
                             extended_least_squares=False,
                             ylag=[1, 2], xlag=2,
                             estimator='least_squares',
                             )
    model.fit(x, y)
    assert_array_equal(model.pivv, piv)
    assert_array_equal(model.final_model, model_code)


def test_fit_with_information_criteria():
    x, y, theta = create_test_data()
    model = PolynomialNarmax(non_degree=2,
                             n_terms=15,
                             order_selection=True,
                             extended_least_squares=False,
                             )
    model.fit(x, y)
    assert 'info_values' in dir(model)


def test_fit_without_information_criteria():
    x, y, theta = create_test_data()
    model = PolynomialNarmax(non_degree=2,
                             n_terms=15,
                             extended_least_squares=False,
                             )
    model.fit(x, y)
    assert 'info_values' not in dir(model)


def test_default_values():
    default = {'non_degree': 2,
               'ylag': 2,
               'xlag': 2,
               'order_selection': False,
               'info_criteria': 'aic',
               'n_terms': None,
               'n_inputs': 1,
               'n_info_values': 10,
               'estimator': 'least_squares',
               'extended_least_squares': True,
               'aux_lag': 1,
               'lam': 0.98,
               'delta': 0.01,
               'offset_covariance': 0.2,
               'mu': 0.01,
               'eps': np.finfo(np.float).eps,
               'gama': 0.2,
               'weight': 0.02}
    model = PolynomialNarmax()
    model_values = [model.non_degree, model.ylag, model.xlag,
                    model._order_selection, model.info_criteria,
                    model.n_terms, model._n_inputs, model.n_info_values,
                    model.estimator, model._extended_least_squares,
                    model._aux_lag, model._lam, model._delta,
                    model._offset_covariance, model._mu, model._eps,
                    model._gama, model._weight]

    assert list(default.values()) == model_values


def test_validate_non_degree():
    assert_raises(ValueError, PolynomialNarmax, non_degree=-1)
    assert_raises(ValueError, PolynomialNarmax, non_degree=1.3)


def test_validate_ylag():
    assert_raises(ValueError, PolynomialNarmax, ylag=-1)
    assert_raises(ValueError, PolynomialNarmax, ylag=1.3)


def test_validate_xlag():
    assert_raises(ValueError, PolynomialNarmax, xlag=-1)
    assert_raises(ValueError, PolynomialNarmax, xlag=1.3)


def test_model_order_selection():
    assert_raises(TypeError, PolynomialNarmax, order_selection=1)
    assert_raises(TypeError, PolynomialNarmax, order_selection='True')
    assert_raises(TypeError, PolynomialNarmax, order_selection=None)


def test_n_terms():
    assert_raises(ValueError, PolynomialNarmax, n_terms=1.2)
    assert_raises(ValueError, PolynomialNarmax, n_terms=-1)


def test_n_inputs():
    assert_raises(ValueError, PolynomialNarmax, n_inputs=1.2)
    assert_raises(ValueError, PolynomialNarmax, n_inputs=-1)


def test_n_info_values():
    assert_raises(ValueError, PolynomialNarmax, n_info_values=1.2)
    assert_raises(ValueError, PolynomialNarmax, n_info_values=-1)


def test_extended_least_squares():
    assert_raises(TypeError, PolynomialNarmax, extended_least_squares=1)
    assert_raises(TypeError, PolynomialNarmax, extended_least_squares='True')
    assert_raises(TypeError, PolynomialNarmax, extended_least_squares=None)


def test_info_criteria():
    assert_raises(ValueError, PolynomialNarmax, info_criteria='AIC')


def test_unbiased_estimator():
    x, y, theta = create_test_data()
    mu, sigma = 0, 0.2
    nu = np.random.normal(mu, sigma, 1000).T
    e = np.zeros((1000, 1))
    lag = 2
    for k in range(lag, len(x)):
        e[k] = 0.2*nu[k-1] + nu[k]

    y = y + e
    model = PolynomialNarmax(non_degree=2,
                             n_terms=5,
                             extended_least_squares=True,
                             ylag=[1, 2], xlag=2,
                             estimator='least_squares',
                             )
    model.fit(x, y)
    assert_almost_equal(model.theta, theta, decimal=2)


def test_predict():
    x, y, theta = create_test_data()

    train_percentage = 90
    split_data = int(len(x)*(train_percentage/100))

    X_train = x[0:split_data, 0]
    X_test = x[split_data::, 0]

    y1 = y[0:split_data, 0]
    y_test = y[split_data::, 0]
    y_train = y1.copy()

    y_train = np.reshape(y_train, (len(y_train), 1))
    X_train = np.reshape(X_train, (len(X_train), 1))

    y_test = np.reshape(y_test, (len(y_test), 1))
    X_test = np.reshape(X_test, (len(X_test), 1))
    model = PolynomialNarmax(non_degree=2,
                             n_terms=5,
                             extended_least_squares=False,
                             ylag=[1, 2], xlag=2,
                             estimator='least_squares',
                             )
    model.fit(X_train, y_train)
    yhat = model.predict(X_test, y_test)
    assert_almost_equal(yhat, y_test, decimal=10)


def test_model_prediction():
    x, y, theta = create_test_data()

    train_percentage = 90
    split_data = int(len(x)*(train_percentage/100))

    X_train = x[0:split_data, 0]
    X_test = x[split_data::, 0]

    y1 = y[0:split_data, 0]
    y_test = y[split_data::, 0]
    y_train = y1.copy()

    y_train = np.reshape(y_train, (len(y_train), 1))
    X_train = np.reshape(X_train, (len(X_train), 1))

    y_test = np.reshape(y_test, (len(y_test), 1))
    X_test = np.reshape(X_test, (len(X_test), 1))
    model = PolynomialNarmax(non_degree=2,
                             n_terms=5,
                             extended_least_squares=False,
                             ylag=[1, 2], xlag=2,
                             estimator='least_squares',
                             )
    model.fit(X_train, y_train)
    assert_raises(Exception, model.predict, X_test, y_test[:1])


def test_information_criteria():
    x, y, theta = create_test_data()

    model = PolynomialNarmax(non_degree=2,
                             n_terms=5,
                             extended_least_squares=False,
                             order_selection=True,
                             info_criteria='aic',
                             n_info_values=5,
                             ylag=[1, 2], xlag=2,
                             estimator='least_squares',
                             )
    model.fit(x, y)
    info_values = np.array([-1769.7907939, -2329.9129067, -2991.1078461,
                            -4481.5306496, -71792.6413])
    assert_almost_equal(model.info_values, info_values)


def test_information_criteria_bic():
    x, y, theta = create_test_data()

    model = PolynomialNarmax(non_degree=2,
                             n_terms=5,
                             extended_least_squares=False,
                             order_selection=True,
                             info_criteria='bic',
                             n_info_values=5,
                             ylag=[1, 2], xlag=2,
                             estimator='least_squares',
                             )
    model.fit(x, y)
    info_values = np.array([-1764.8850406, -2320.1014001, -2976.3905863,
                            -4461.9076365, -71768.1125336])
    assert_almost_equal(model.info_values, info_values)


def test_information_criteria_fpe():
    x, y, theta = create_test_data()

    model = PolynomialNarmax(non_degree=2,
                             n_terms=5,
                             extended_least_squares=False,
                             order_selection=True,
                             info_criteria='fpe',
                             n_info_values=5,
                             ylag=[1, 2], xlag=2,
                             estimator='least_squares',
                             )
    model.fit(x, y)
    info_values = np.array([-1769.7907932, -2329.9129013, -2991.1078281,
                            -4481.5306067, -71792.6412163])
    assert_almost_equal(model.info_values, info_values)


def test_information_criteria_lilc():
    x, y, theta = create_test_data()

    model = PolynomialNarmax(non_degree=2,
                             n_terms=5,
                             extended_least_squares=False,
                             order_selection=True,
                             info_criteria='lilc',
                             n_info_values=5,
                             ylag=[1, 2], xlag=2,
                             estimator='least_squares',
                             )
    model.fit(x, y)
    info_values = np.array([-1767.9260841, -2326.1834872, -2985.5137169,
                            -4474.0718106, -71783.3177513])
    assert_almost_equal(model.info_values, info_values)


def test_results():
    x, y, theta = create_test_data()

    model = PolynomialNarmax(non_degree=2,
                             n_terms=5,
                             extended_least_squares=False,
                             order_selection=True,
                             info_criteria='lilc',
                             n_info_values=5,
                             ylag=[1, 2], xlag=2,
                             estimator='least_squares',
                             )
    model.fit(x, y)
    results = model.results(err_precision=8, dtype='dec')
    assert isinstance(results, list)
    assert_raises(ValueError, model.results, theta_precision=-1)
    assert_raises(ValueError, model.results, err_precision=-1)
    assert_raises(ValueError, model.results, dtype='DEC')
