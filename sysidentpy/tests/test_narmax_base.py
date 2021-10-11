import numpy as np
from sysidentpy.narmax_base import GenerateRegressors
from sysidentpy.narmax_base import HouseHolder
from sysidentpy.narmax_base import ModelInformation
from sysidentpy.narmax_base import InformationMatrix

from numpy.testing import assert_almost_equal, assert_array_equal, assert_equal

IM = InformationMatrix()
MI = ModelInformation()
HH = HouseHolder()
GR = GenerateRegressors()

def test_create_narmax_code():
    output1 = np.array([2001, 2002]), ([1001, 1002])
    r1 = GR.create_narmax_code(non_degree=1, xlag=2, ylag=2, n_inputs=1)
    assert_array_equal(output1, r1)
    
def test_regressor_space():
    output1 = np.array([[   0],
                        [1001],
                        [1002],
                        [2001],
                        [2002]])
    r1 = GR.regressor_space(non_degree=1, xlag=2, ylag=2, n_inputs=1)
    assert_array_equal(output1, r1)
    output2 = np.array([[   0,    0],
                        [1001,    0],
                        [1002,    0],
                        [2001,    0],
                        [2002,    0],
                        [1001, 1001],
                        [1002, 1001],
                        [2001, 1001],
                        [2002, 1001],
                        [1002, 1002],
                        [2001, 1002],
                        [2002, 1002],
                        [2001, 2001],
                        [2002, 2001],
                        [2002, 2002]])
    r2 = GR.regressor_space(non_degree=2, xlag=2, ylag=2, n_inputs=1)
    assert_array_equal(output2, r2)
    output3 = np.array([[   0,    0],
                        [1001,    0],
                        [1002,    0],
                        [2001,    0],
                        [2002,    0],
                        [3001,    0],
                        [3002,    0],
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
                        [3002, 3002]])
    r3 = GR.regressor_space(non_degree=2, xlag=[[1, 2], [1, 2]], ylag=2, n_inputs=2)
    assert_array_equal(output3, r3)

def test_house():
    a = np.array([0.42544384, 0.39365905, 0.22209413, 0.69760074, 0.88183369,
       0.24818225, 0.78482346, 0.26967285, 0.53987842, 0.17367185])
    
    output = np.array([1, 0.18970318, 0.10702653, 0.33617182, 0.42495315,
       0.11959832, 0.3782042 , 0.12995458, 0.26016588, 0.08369197])
    assert_almost_equal(HH._house(a), output)
    
def test_row_house():
    a = np.array([0.42544384, 0.39365905, 0.22209413, 0.69760074, 0.88183369,
       0.24818225, 0.78482346, 0.26967285, 0.53987842, 0.17367185]).reshape(-1, 1)
    
    b = np.array([0.90009285, 0.21392929, 0.58429212, 0.55761456, 0.65178413,
       0.4061564 , 0.4353402 , 0.02365408, 0.52291863, 0.185921]).reshape(-1, 1)
    
    output = np.array([[-1.1861246 ],
                       [ 0.01063002],
                       [-0.82404988],
                       [-0.30077851],
                       [-0.28515117],
                       [-0.47901921],
                       [ 0.00536996],
                       [ 0.22732148],
                       [-0.39637961],
                       [-0.15920982]])
    assert_almost_equal(HH._rowhouse(a, b), output)

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
    index = MI._get_index_from_regressor_code(
        regressor_code=regressor_space, model_code=model
    )

    assert (index == np.array([1, 3, 5])).all()


def test_list_output_regressor():
    model = np.array(
        [
            [1001, 0],  # y(k-1)
            [2001, 1001],  # x1(k-1)y(k-1)
            [2002, 0],  # x1(k-2)
        ]
    )

    y_code = MI._list_output_regressor_code(model)
    assert (y_code == np.array([1001, 1001])).all()


def test_list_input_regressor():
    model = np.array(
        [
            [1001, 0],  # y(k-1)
            [2001, 1001],  # x1(k-1)y(k-1)
            [2002, 0],  # x1(k-2)
        ]
    )

    x_code = MI._list_input_regressor_code(model)
    assert (x_code == np.array([2001, 2002])).all()


def test_get_lag_from_regressor_code():
    list_regressor1 = np.array([2001, 2002])
    list_regressor2 = np.array([1004, 1002])
    max_lag1 = MI._get_lag_from_regressor_code(list_regressor1)
    max_lag2 = MI._get_lag_from_regressor_code(list_regressor2)

    assert max_lag1 == 2
    assert max_lag2 == 4
    
def test_get_max_lag():
    output1 = 1
    r = MI._get_max_lag(ylag=1, xlag=1)
    output2 = 3
    r2 = MI._get_max_lag(ylag=3, xlag=1)
    assert_equal(output1, r)
    assert_equal(output2, r2)
    
def test_shift_column():
    a = np.array([0.42544384, 0.39365905, 0.22209413, 0.69760074, 0.88183369,
       0.24818225, 0.78482346, 0.26967285, 0.53987842, 0.17367185]).reshape(-1, 1)
    
    output = np.array([[0.        ],
                       [0.        ],
                       [0.42544384],
                       [0.39365905],
                       [0.22209413],
                       [0.69760074],
                       [0.88183369],
                       [0.24818225],
                       [0.78482346],
                       [0.26967285]])
    r = IM.shift_column(a, 2)
    assert_almost_equal(output, r)
    
def test_process_xlag():
    a = np.array([0.42544384, 0.39365905, 0.22209413, 0.69760074, 0.88183369,
       0.24818225, 0.78482346, 0.26967285, 0.53987842, 0.17367185]).reshape(-1, 1)
    
    n_inputs, xlag = IM._process_xlag(a.reshape(-1, 1), 2)
    output1 = 1
    output2 = range(1, 3)
    assert_equal(output1, n_inputs)
    assert_equal(output2, xlag)
    
def test_process_xlag():
    ylag = IM._process_ylag(2)
    output1 = range(1, 3)
    assert_equal(output1, ylag)
