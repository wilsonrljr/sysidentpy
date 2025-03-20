import pytest
import numpy as np
from sysidentpy.utils.display_results import results


def test_results_valid_input():
    """Test results function with valid input."""
    final_model = np.array(
        [[1001, 2002], [1002, 0], [3001, 1001]]
    )  # Example model regressors
    theta = np.array([[1.2345], [2.3456], [3.4567]])  # Parameters
    err = np.array([0.12345678, 0.23456789, 0.34567890])  # ERR values
    n_terms = 3

    output = results(
        final_model=final_model,
        theta=theta,
        err=err,
        n_terms=n_terms,
        theta_precision=3,
        err_precision=6,
        dtype="dec",
    )

    expected_output = [
        ["y(k-1)x1(k-2)", "1.234", "0.123457"],
        ["y(k-2)", "2.346", "0.234568"],
        ["x2(k-1)y(k-1)", "3.457", "0.345679"],
    ]

    assert output == expected_output


def test_results_invalid_theta_precision():
    """Test results function with an invalid theta_precision."""
    with pytest.raises(ValueError, match="theta_precision must be integer and > zero"):
        results(theta_precision=0)


def test_results_invalid_err_precision():
    """Test results function with an invalid err_precision."""
    with pytest.raises(ValueError, match="err_precision must be integer and > zero"):
        results(err_precision=0)


def test_results_invalid_dtype():
    """Test results function with an invalid dtype."""
    with pytest.raises(ValueError, match="dtype must be dec or sci"):
        results(dtype="invalid")
