import numpy as np
from sysidentpy.utils.simulation import (
    list_output_regressor_code,
    list_input_regressor_code,
    get_index_from_regressor_code,
)


def test_list_input_regressor():
    model = np.array(
        [
            [1001, 0],  # y(k-1)
            [2001, 1001],  # x1(k-1)y(k-1)
            [2002, 0],  # x1(k-2)
        ]
    )

    x_code = list_input_regressor_code(model)
    assert (x_code == np.array([2001, 2002])).all()


def test_list_output_regressor():
    model = np.array(
        [
            [1001, 0],  # y(k-1)
            [2001, 1001],  # x1(k-1)y(k-1)
            [2002, 0],  # x1(k-2)
        ]
    )

    y_code = list_output_regressor_code(model)
    assert (y_code == np.array([1001, 1001])).all()


def test_get_index_from_regressor_code_space1():
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
    index = get_index_from_regressor_code(
        regressor_code=regressor_space, model_code=model
    )

    assert (index == np.array([1, 3, 5])).all()


def test_get_index_from_regressor_code():
    regressor_code = np.array(
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
    model_code = np.array(
        [
            [1001, 0],  # y(k-1)
            [2001, 1001],  # x1(k-1)y(k-1)
            [2002, 0],  # x1(k-2)
        ]
    )

    expected_output = np.array(
        [1, 4, 7]
    )  # Expected indices of model_code in regressor_code
    result = get_index_from_regressor_code(regressor_code, model_code)
    np.testing.assert_array_equal(result, expected_output)


def test_list_output_regressor_code():
    model_code = np.array(
        [
            [1001, 0],  # y(k-1)
            [2001, 1001],  # x1(k-1)y(k-1)
            [2002, 0],  # x1(k-2)
        ]
    )

    expected = np.array([1001, 1001])
    result = list_output_regressor_code(model_code)
    np.testing.assert_array_equal(result, expected)


def test_list_input_regressor_code():
    model_code = np.array(
        [
            [1001, 0],  # y(k-1)
            [2001, 1001],  # x1(k-1)y(k-1)
            [2002, 0],  # x1(k-2)
        ]
    )

    expected = np.array([2001, 2002])
    result = list_input_regressor_code(model_code)
    np.testing.assert_array_equal(result, expected)
