import numpy as np
import warnings

from sysidentpy.basis_function import (
    Polynomial,
    Fourier,
    Bilinear,
    Legendre,
)
from sysidentpy.utils.equation_formatter import (
    results_general,
    format_equation,
    register_equation_renderer,
    RendererRegistry,
    _warned_unknown_bases,
)


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


def test_fourier_ensemble_includes_raw_then_harmonics():
    basis = Fourier(n=2, degree=1, ensemble=True)
    pivv = None  # use all
    theta = np.ones((10, 1))  # more than enough; truncation handled
    model = DummyModel(
        basis_function=basis, xlag=2, ylag=2, n_inputs=1, pivv=pivv, theta=theta
    )
    items = results_general(model)
    # Raw lags should appear before cos/sin terms
    names = [it.name for it in items]
    first_cos_index = next(i for i, n in enumerate(names) if n.startswith("cos("))
    assert any(n.startswith("y(k-") for n in names[:first_cos_index])


def test_legendre_bias_and_ensemble_order():
    basis = Legendre(degree=2, include_bias=True, ensemble=True)
    model = DummyModel(
        basis_function=basis, xlag=1, ylag=2, n_inputs=1, pivv=None, theta=[0.1] * 20
    )
    items = results_general(model)
    names = [it.name for it in items]
    # Ensemble means raw lags precede bias '1'
    assert "1" in names
    bias_pos = names.index("1")
    # One raw lag must appear before bias
    assert any(n.startswith("y(k-") for n in names[:bias_pos])
    # Expansions should follow after bias
    post_bias = names[bias_pos + 1 :]
    assert any(n.startswith("P1(") for n in post_bias)


def test_bilinear_generates_cross_terms():
    basis = Bilinear(degree=2)
    model = DummyModel(
        basis_function=basis, xlag=2, ylag=2, n_inputs=1, pivv=None, theta=[0.2] * 50
    )
    items = results_general(model)
    names = [it.name for it in items]
    # Should contain a mixed product y * x
    assert any("y(k-1)" in n and "x1(k-1)" in n for n in names)
    # Should not contain a pure squared y term when degree=2 (bilinear removes pure y)
    assert not any(n.startswith("y(k-1)") and "y(k-" in n[1:] for n in names)


def test_no_pivv_uses_sequential_mapping():
    basis = Polynomial(degree=2)
    theta = [0.5] * 10
    model = DummyModel(
        basis_function=basis, xlag=1, ylag=1, n_inputs=1, pivv=None, theta=theta
    )
    items = results_general(model)
    assert len(items) <= len(theta)


def test_invalid_pivv_index_generates_placeholder():
    basis = Polynomial(degree=2)
    # Use an out-of-range index intentionally
    pivv = np.array([999])
    theta = [1.23]
    model = DummyModel(
        basis_function=basis, xlag=1, ylag=1, n_inputs=1, pivv=pivv, theta=theta
    )
    items = results_general(model)
    assert items[0].name.startswith("f_")


def test_format_equation_ascii_and_first_sign():
    basis = Polynomial(degree=1)
    pivv = np.array([0, 1])
    theta = [1.0, -0.5]
    model = DummyModel(
        basis_function=basis, xlag=1, ylag=1, n_inputs=1, pivv=pivv, theta=theta
    )
    eq = format_equation(model, ascii_mode=True, first_sign=True)
    rhs = eq.split("=", 1)[1]
    assert "*" in rhs
    assert "+" in rhs


def test_fallback_warning_for_unknown_basis():
    class UnknownBasis:
        def __init__(self):
            self.degree = 2

    # Reset suppression set so this test is deterministic
    _warned_unknown_bases.clear()

    model = DummyModel(
        basis_function=UnknownBasis(),
        xlag=1,
        ylag=1,
        n_inputs=1,
        pivv=None,
        theta=[0.1] * 5,
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        items = results_general(model)
        assert items  # got some items
        assert any("fallback used" in str(warn.message) for warn in w)


def test_fallback_warning_emitted_only_once():
    class UnknownBasis:
        def __init__(self):
            self.degree = 1

    # Reset suppression set
    _warned_unknown_bases.clear()

    model = DummyModel(
        basis_function=UnknownBasis(),
        xlag=1,
        ylag=1,
        n_inputs=1,
        pivv=None,
        theta=[0.3, 0.4],
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = results_general(model)
        _ = results_general(model)  # second call should not warn again
        warns = [warn for warn in w if "fallback used" in str(warn.message)]
        assert len(warns) == 1, "Expected exactly one fallback warning"


def test_dynamic_renderer_registration_avoids_fallback_warning():
    class CustomBasis:
        def __init__(self):
            self.degree = 1

    basis = CustomBasis()

    # Ensure no prior renderer is registered (defensive cleanup)
    if "CustomBasis" in RendererRegistry:
        del RendererRegistry["CustomBasis"]

    def _custom_renderer(model, input_names, output_name):
        # Touch model to silence unused arg warnings
        _ = getattr(model, "basis_function", None)
        return ["__const__", f"{output_name}(k-1)", f"{input_names[0]}(k-1)"]

    register_equation_renderer("CustomBasis", _custom_renderer, overwrite=True)

    theta = np.array([0.5, 1.2, -0.3])
    model = DummyModel(
        basis_function=basis,
        xlag=1,
        ylag=1,
        n_inputs=1,
        pivv=None,
        theta=theta,
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        items = results_general(model, input_names=["u"], output_name="y")
        assert (
            not w
        ), "No fallback warning should be emitted when custom renderer is registered"
    names = [it.name for it in items]
    assert names == ["__const__", "y(k-1)", "u(k-1)"]
