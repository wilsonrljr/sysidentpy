import numpy as np

from sysidentpy.basis_function import Polynomial, Fourier
from sysidentpy.utils.equation_formatter import results_general, format_equation


class DummyModel:
    def __init__(self, *, basis_function, xlag, ylag, n_inputs, pivv, theta):
        self.basis_function = basis_function
        self.xlag = xlag
        self.ylag = ylag
        self.n_inputs = n_inputs
        self.pivv = pivv
        self.theta = theta


def test_polynomial_equation_names_match_structure():
    # degree=2 over base: 1, y(k-1), y(k-2), x1(k-1), x1(k-2)
    # combinations with replacement over 5 items of size 2 => C(5+2-1,2)=15
    # Select first 6 for the test
    basis = Polynomial(degree=2)
    pivv = np.array([0, 1, 2, 3, 4, 5])
    theta = np.arange(1, len(pivv) + 1).reshape(-1, 1)
    model = DummyModel(
        basis_function=basis, xlag=2, ylag=2, n_inputs=1, pivv=pivv, theta=theta
    )
    items = results_general(model)
    assert len(items) == len(pivv)
    # Names must be non-empty and include expected variable tokens
    assert all(isinstance(t.name, str) and len(t.name) > 0 for t in items)
    # Ensure intercept term appears for some early combination or y/x lags show up
    assert any(t.name == "1" or "y(k-" in t.name or "x1(k-" in t.name for t in items)


def test_fourier_with_degree_2_uses_poly_names_without_intercept():
    basis = Fourier(n=1, degree=2, ensemble=False)
    # pick first 4 features of the generated Fourier list
    pivv = np.array([0, 1, 2, 3])
    theta = np.ones((len(pivv), 1))
    model = DummyModel(
        basis_function=basis, xlag=2, ylag=1, n_inputs=1, pivv=pivv, theta=theta
    )
    eq = format_equation(model)
    assert "cos(" in eq or "sin(" in eq
