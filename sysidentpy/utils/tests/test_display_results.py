import pytest
import numpy as np
from sysidentpy.utils.display_results import results


def test_results_valid_input():
    """Test results function with valid input."""
    final_model = np.array([[1001, 2002], [1002, 0], [3001, 1001]])
    theta = np.array([[1.2345], [2.3456], [3.4567]])
    err = np.array([0.12345678, 0.23456789, 0.34567890])
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


def test_results_scientific_notation():
    """Ensure scientific formatting is applied when requested."""
    final_model = np.array([[1001, 0]])
    theta = np.array([[1.23456789]])
    err = np.array([0.00001234])
    output = results(
        final_model=final_model,
        theta=theta,
        err=err,
        n_terms=1,
        dtype="sci",
        theta_precision=2,
        err_precision=2,
    )
    assert output[0][1].upper().endswith("E+00")
    assert output[0][2].upper().endswith("E-05")


def test_results_constant_regressor_representation():
    """Regressors with only constants should be rendered as '1'."""
    final_model = np.array([[0, 0]])
    theta = np.array([[2.0]])
    err = np.array([0.1])
    output = results(
        final_model=final_model,
        theta=theta,
        err=err,
        n_terms=1,
    )
    assert output[0][0] == "1"


def test_results_handles_zero_regressor_key_inside_term():
    """Keys lower than 1 inside a term should be ignored gracefully."""
    final_model = np.array([[0, 1001]])
    theta = np.array([[0.5]])
    err = np.array([0.2])
    output = results(
        final_model=final_model,
        theta=theta,
        err=err,
        n_terms=1,
    )
    # Constant element should be dropped while y term remains
    assert output[0][0] == "y(k-1)"


def test_results_renders_exponent_for_repeated_regressors():
    final_model = np.array([[1001, 1001]])
    theta = np.array([[1.0]])
    err = np.array([0.1])
    output = results(
        final_model=final_model,
        theta=theta,
        err=err,
        n_terms=1,
    )
    assert output[0][0].startswith("y(k-1)^2")
