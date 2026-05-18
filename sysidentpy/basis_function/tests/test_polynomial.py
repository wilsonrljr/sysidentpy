import numpy as np
from numpy.testing import assert_array_equal

from sysidentpy.basis_function import Polynomial


def _lagged_with_bias(n_rows, n_features):
    """Build a synthetic lagged matrix with the bias column at index 0."""
    rng = np.random.default_rng(0)
    data = rng.uniform(size=(n_rows, n_features))
    data[:, 0] = 1.0
    return data


def test_polynomial_init_defaults():
    p = Polynomial()
    assert p.degree == 2
    assert p.include_bias is True


def test_polynomial_init_explicit():
    p = Polynomial(degree=3, include_bias=False)
    assert p.degree == 3
    assert p.include_bias is False


def test_polynomial_default_matches_original_behavior():
    """include_bias=True must preserve pre-PR fit output exactly."""
    p = Polynomial(degree=2)
    data = np.array([[1, 1, 1], [2, 3, 4], [3, 3, 3]])
    expected = np.array([[4, 6, 8, 9, 12, 16], [9, 9, 9, 9, 9, 9]])
    assert_array_equal(p.fit(data=data, max_lag=1), expected)


def test_polynomial_include_bias_false_drops_constant():
    """include_bias=False drops the (0,0,...,0) pure-bias column."""
    p_with = Polynomial(degree=2, include_bias=True)
    p_without = Polynomial(degree=2, include_bias=False)
    data = np.array([[1, 1, 1], [2, 3, 4], [3, 3, 3]])
    psi_with = p_with.fit(data=data, max_lag=1)
    psi_without = p_without.fit(data=data, max_lag=1)

    assert psi_with.shape[1] == psi_without.shape[1] + 1
    assert_array_equal(psi_without, psi_with[:, 1:])


def test_polynomial_include_bias_false_no_constant_column():
    """The remaining columns under include_bias=False are not pure constants."""
    p = Polynomial(degree=2, include_bias=False)
    data = _lagged_with_bias(20, 4)
    psi = p.fit(data=data, max_lag=2)

    constant_columns = [
        i for i in range(psi.shape[1]) if np.allclose(psi[:, i], psi[0, i])
    ]
    assert constant_columns == []


def test_polynomial_transform_matches_fit():
    p = Polynomial(degree=2, include_bias=False)
    data = _lagged_with_bias(10, 3)
    assert_array_equal(p.fit(data=data, max_lag=1), p.transform(data=data, max_lag=1))


def test_polynomial_predefined_regressors_index_into_candidate_set():
    """predefined_regressors indexes the candidate set after include_bias is applied."""
    p = Polynomial(degree=2, include_bias=False)
    data = np.array([[1, 1, 1], [2, 3, 4], [3, 3, 3]])
    full = p.fit(data=data, max_lag=1)
    picked = p.fit(data=data, max_lag=1, predefined_regressors=np.array([0, 2]))
    assert_array_equal(picked, full[:, [0, 2]])
