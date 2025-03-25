from typing import Tuple

import pytest
import numpy as np

from sysidentpy.model_structure_selection.ofr_base import (
    OFRBase,
)
from sysidentpy.parameter_estimation import RecursiveLeastSquares
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import RidgeRegression


# Create a subclass to instantiate the abstract class
class TestOFRBase(OFRBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run_mss_algorithm(
        self, psi: np.ndarray, y: np.ndarray, process_term_number: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass


def test_ofrbase_initialization_default():
    """Test initialization of OFRBase with default parameters."""
    model = TestOFRBase(
        ylag=2,
        xlag=2,
        elag=2,
        order_selection=True,
        info_criteria="aic",
        n_terms=None,
        n_info_values=15,
        estimator=RecursiveLeastSquares(),
        basis_function=Polynomial(),
        model_type="NARMAX",
        eps=np.finfo(np.float64).eps,
        alpha=0,
        err_tol=None,
    )

    # Check if the attributes are correctly initialized
    assert model.ylag == 2
    assert model.xlag == 2
    assert model.elag == 2
    assert model.order_selection is True
    assert model.info_criteria == "aic"
    assert model.n_info_values == 15
    assert model.n_terms is None
    assert isinstance(model.estimator, RecursiveLeastSquares)
    assert isinstance(model.basis_function, Polynomial)
    assert model.model_type == "NARMAX"
    assert model.eps == np.finfo(np.float64).eps
    assert model.alpha == 0
    assert model.err_tol is None


def test_ofrbase_initialization_with_ridge():
    """Test initialization of OFRBase with RidgeRegression estimator."""
    ridge_estimator = RidgeRegression(alpha=0.5)
    model = TestOFRBase(
        ylag=2,
        xlag=2,
        elag=2,
        order_selection=True,
        info_criteria="aic",
        n_terms=None,
        n_info_values=15,
        estimator=ridge_estimator,
        basis_function=Polynomial(),
        model_type="NARMAX",
        eps=np.finfo(np.float64).eps,
        alpha=0.5,
        err_tol=None,
    )

    # Check if the alpha value from the estimator is correctly set
    assert model.alpha == 0.5


def test_ofrbase_initialization_invalid_info_criteria():
    """Test initialization of OFRBase with an invalid info_criteria."""
    with pytest.raises(
        ValueError,
        match="info_criteria must be aic, bic, fpe or lilc. Got invalid_criteria",
    ):
        TestOFRBase(
            ylag=2,
            xlag=2,
            elag=2,
            order_selection=True,
            info_criteria="invalid_criteria",  # Invalid criteria
            n_terms=None,
            n_info_values=15,
            estimator=RecursiveLeastSquares(),
            basis_function=Polynomial(),
            model_type="NARMAX",
            eps=np.finfo(np.float64).eps,
            alpha=0,
            err_tol=None,
        )


def test_ofrbase_initialization_with_n_terms():
    """Test initialization of OFRBase with a defined n_terms."""
    model = TestOFRBase(
        ylag=2,
        xlag=2,
        elag=2,
        order_selection=True,
        info_criteria="aic",
        n_terms=10,
        n_info_values=15,
        estimator=RecursiveLeastSquares(),
        basis_function=Polynomial(),
        model_type="NARMAX",
        eps=np.finfo(np.float64).eps,
        alpha=0,
        err_tol=None,
    )

    # Assert that the model's n_terms is correctly set
    assert model.n_terms == 10


def test_ofrbase_invalid_eps():
    """Test invalid eps value for OFRBase initialization."""
    with pytest.raises(ValueError, match="eps must be float and > zero. Got -1"):
        TestOFRBase(
            ylag=2,
            xlag=2,
            elag=2,
            order_selection=True,
            info_criteria="aic",
            n_terms=None,
            n_info_values=15,
            estimator=RecursiveLeastSquares(),
            basis_function=Polynomial(),
            model_type="NARMAX",
            eps=-1,  # Invalid eps
            alpha=0,
            err_tol=None,
        )


# Create a mock of the class to test the method
class MockOFRBase(OFRBase):
    def __init__(self):
        # Initialize necessary parameters for testing
        self.max_lag = 2
        self.alpha = 0.01
        self.eps = np.finfo(np.float64).eps
        self.err_tol = None  # Set to None or some value depending on the test case
        self.n_terms = 10

    def run_mss_algorithm(self, psi, y, process_term_number):
        return self.error_reduction_ratio(psi, y, process_term_number)


@pytest.fixture
def setup_data():
    # Create mock data for psi and y to simulate real-world input
    n_samples = 200
    n_features = 10
    max_lag = 2
    y = np.random.randn(n_samples + max_lag, 1)
    psi = np.random.randn(n_samples, n_features)
    process_term_number = 5

    return y, psi, process_term_number


def test_error_reduction_ratio_basic(setup_data):
    """Test basic functionality of the error_reduction_ratio method."""
    y, psi, process_term_number = setup_data
    model = MockOFRBase()

    # Call the method
    err, piv, psi_orthogonal = model.error_reduction_ratio(psi, y, process_term_number)

    # Ensure that the output is as expected
    assert isinstance(err, np.ndarray)
    assert isinstance(piv, np.ndarray)
    assert isinstance(psi_orthogonal, np.ndarray)

    # Check that the lengths of the returned arrays match the expected number
    # of regressors (n_features)
    assert err.shape[0] == psi.shape[1]  # Number of model elements (n_features)
    assert piv.shape[0] == process_term_number
    assert psi_orthogonal.shape[1] == process_term_number


def test_error_reduction_ratio_with_large_alpha(setup_data):
    """Test the ERR with a large alpha to observe the effect of regularization."""
    y, psi, process_term_number = setup_data
    model = MockOFRBase()

    # Set a very large alpha to test its effect on the error reduction ratio
    model.alpha = 1000

    # Call the method
    err, piv, psi_orthogonal = model.error_reduction_ratio(psi, y, process_term_number)

    # Ensure that the returned values are not empty and match the expected dimensions
    assert err.shape[0] == psi.shape[1]
    assert piv.shape[0] == process_term_number
    assert psi_orthogonal.shape[1] == process_term_number

    # Additional check: Assert that the error values are reasonably
    # affected by a large alpha
    assert np.all(
        err < 1
    )  # Assuming that large alpha reduces the magnitude of the errors


def test_error_reduction_ratio_with_small_eps(setup_data):
    """Test ERR with a very small epsilon to see its effect on stability."""
    y, psi, process_term_number = setup_data
    model = MockOFRBase()

    # Set a very small epsilon value to test numerical stability
    model.eps = 1e-12

    # Call the method
    err, piv, psi_orthogonal = model.error_reduction_ratio(psi, y, process_term_number)

    # Ensure the returned values are valid
    assert err.shape[0] == psi.shape[1]
    assert piv.shape[0] == process_term_number
    assert psi_orthogonal.shape[1] == process_term_number
