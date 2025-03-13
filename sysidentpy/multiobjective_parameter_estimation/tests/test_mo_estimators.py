"""Tests for AILS methods."""

import numpy as np
from sysidentpy.multiobjective_parameter_estimation.estimators import AILS
from sysidentpy.multiobjective_parameter_estimation.estimators import (
    get_term_clustering,
    get_cost_function,
)
from sysidentpy.utils.narmax_tools import set_weights
from numpy.testing import assert_array_equal, assert_almost_equal, assert_allclose


df_train = np.genfromtxt(
    "https://raw.githubusercontent.com/wilsonrljr/sysidentpy-data/refs/heads/main/datasets/buck/buck_id.csv",
    delimiter=",",
    skip_header=True,
)
df_valid = np.genfromtxt(
    "https://raw.githubusercontent.com/wilsonrljr/sysidentpy-data/refs/heads/main/datasets/buck/buck_valid.csv",
    delimiter=",",
    skip_header=True,
)

Vd = 24
Uo = np.linspace(0, 4, 50)
Yo = (4 - Uo) * Vd / 3
Uo = Uo.reshape(-1, 1)
Yo = Yo.reshape(-1, 1)

gain = -8 * np.ones(len(Uo)).reshape(-1, 1)
x_train = df_train[:, 1].reshape(-1, 1)
y_train = df_train[:, 2].reshape(-1, 1)
x_valid = df_valid[:, 1].reshape(-1, 1)
y_valid = df_valid[:, 2].reshape(-1, 1)

w = set_weights(static_function=True, static_gain=True)

final_model = np.array(
    [
        [1001, 0],
        [1002, 0],
        [2001, 1001],
        [0, 0],
        [1001, 1001],
        [2001, 0],
        [1002, 1001],
        [1002, 1002],
    ]
)


def test_default_values():
    default = {
        "static_gain": True,
        "static_function": True,
        "final_model": np.zeros((1, 1)),
        "normalize": True,
    }
    model = AILS()
    model_values = [
        model.static_gain,
        model.static_function,
        model.final_model,
        model.normalize,
    ]
    assert list(default.values()) == model_values


def test_get_term_clustering():
    qit = np.array([[0, 0], [1, 0], [2, 0], [1, 1], [2, 1]])

    expected_result = np.array(np.array([[0, 0], [1, 0], [0, 1], [2, 0], [1, 1]]))
    result = get_term_clustering(qit)
    assert_array_equal(result, expected_result)


def test_build_linear_mapping():
    estimator = AILS(final_model=final_model)
    R, qit = estimator.build_linear_mapping()
    expected_result = np.array([[0, 0], [1, 0], [0, 1], [2, 0], [1, 1]])
    assert_array_equal(expected_result, qit)
    assert isinstance(R, np.ndarray)
    assert isinstance(qit, np.ndarray)
    assert R.shape == (5, 8)
    assert qit.shape == (5, 2)
    assert R.dtype == int
    assert qit.dtype == int


def test_build_static_function_information():
    mo_estimator = AILS(final_model=final_model)
    QR, static_covariance, static_response = (
        mo_estimator.build_static_function_information(Uo, Yo)
    )
    # those values were taken from matlab implementation to compare
    # with this scenario
    static_covariance_mat = np.array(
        [
            [
                1.72408163e04,
                1.72408163e04,
                1.67183673e04,
                8.00000000e02,
                4.17959184e05,
                1.04489796e03,
                4.17959184e05,
                4.17959184e05,
            ],
            [
                1.72408163e04,
                1.72408163e04,
                1.67183673e04,
                8.00000000e02,
                4.17959184e05,
                1.04489796e03,
                4.17959184e05,
                4.17959184e05,
            ],
            [
                1.67183673e04,
                1.67183673e04,
                2.67605287e04,
                1.04489796e03,
                3.20903526e05,
                2.08979592e03,
                3.20903526e05,
                3.20903526e05,
            ],
            [
                8.00000000e02,
                8.00000000e02,
                1.04489796e03,
                5.00000000e01,
                1.72408163e04,
                1.00000000e02,
                1.72408163e04,
                1.72408163e04,
            ],
            [
                4.17959184e05,
                4.17959184e05,
                3.20903526e05,
                1.72408163e04,
                1.08074657e07,
                1.67183673e04,
                1.08074657e07,
                1.08074657e07,
            ],
            [
                1.04489796e03,
                1.04489796e03,
                2.08979592e03,
                1.00000000e02,
                1.67183673e04,
                2.69387755e02,
                1.67183673e04,
                1.67183673e04,
            ],
            [
                4.17959184e05,
                4.17959184e05,
                3.20903526e05,
                1.72408163e04,
                1.08074657e07,
                1.67183673e04,
                1.08074657e07,
                1.08074657e07,
            ],
            [
                4.17959184e05,
                4.17959184e05,
                3.20903526e05,
                1.72408163e04,
                1.08074657e07,
                1.67183673e04,
                1.08074657e07,
                1.08074657e07,
            ],
        ]
    )
    static_response_mat = np.array(
        [
            [17240.81632653],
            [17240.81632653],
            [16718.36734694],
            [800.0],
            [417959.18367347],
            [1044.89795918],
            [417959.18367347],
            [417959.18367347],
        ]
    )
    assert_allclose(static_covariance_mat, static_covariance)
    assert_allclose(static_response_mat, static_response)
    assert_almost_equal(QR.mean(), 136.2933673469388)
    assert QR.shape == (50, 8)


def test_build_static_gain_information():
    mo_estimator = AILS(final_model=final_model)
    HR, gain_covariance, gain_response = mo_estimator.build_static_gain_information(
        Uo, Yo, gain
    )
    # those values were taken from matlab implementation to compare
    # with this scenario
    gain_covariance_mat = np.array(
        [
            [
                3.20000000e03,
                3.20000000e03,
                2.80000000e03,
                0.00000000e00,
                1.02400000e05,
                -4.00000000e02,
                1.02400000e05,
                1.02400000e05,
            ],
            [
                3.20000000e03,
                3.20000000e03,
                2.80000000e03,
                0.00000000e00,
                1.02400000e05,
                -4.00000000e02,
                1.02400000e05,
                1.02400000e05,
            ],
            [
                2.80000000e03,
                2.80000000e03,
                2.45000000e03,
                0.00000000e00,
                8.96000000e04,
                -3.50000000e02,
                8.96000000e04,
                8.96000000e04,
            ],
            [
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
            ],
            [
                1.02400000e05,
                1.02400000e05,
                8.96000000e04,
                0.00000000e00,
                4.41364898e06,
                -1.28000000e04,
                4.41364898e06,
                4.41364898e06,
            ],
            [
                -4.00000000e02,
                -4.00000000e02,
                -3.50000000e02,
                0.00000000e00,
                -1.28000000e04,
                5.00000000e01,
                -1.28000000e04,
                -1.28000000e04,
            ],
            [
                1.02400000e05,
                1.02400000e05,
                8.96000000e04,
                0.00000000e00,
                4.41364898e06,
                -1.28000000e04,
                4.41364898e06,
                4.41364898e06,
            ],
            [
                1.02400000e05,
                1.02400000e05,
                8.96000000e04,
                0.00000000e00,
                4.41364898e06,
                -1.28000000e04,
                4.41364898e06,
                4.41364898e06,
            ],
        ]
    )
    gain_response_mat = np.array(
        [
            [3200.0],
            [3200.0],
            [2800.0],
            [0.0],
            [102400.0],
            [-400.0],
            [102400.0],
            [102400.0],
        ]
    )
    assert_allclose(gain_covariance_mat, gain_covariance)
    assert_allclose(gain_response_mat, gain_response)
    assert_almost_equal(HR.mean(), -98.75)
    assert HR.shape == (50, 8)


def test_estimate():
    mo_estimator = AILS(final_model=final_model)
    J, E, theta, _, _, position = mo_estimator.estimate(
        y=y_train, gain=gain, y_static=Yo, X_static=Uo, X=x_train, weighing_matrix=w
    )
    expected_theta = np.array(
        [
            [1.54048245],
            [0.29686668],
            [0.64693309],
            [-0.41302211],
            [0.27670633],
            [-0.53473868],
            [0.00406244],
            [0.25831577],
        ]
    )
    assert_allclose(expected_theta, theta[position, :].reshape(-1, 1), rtol=1e-05)
    assert J.shape == (3, 2295)
    assert E.shape == (2295,)
    assert position == 1065
