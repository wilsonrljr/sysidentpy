import numpy as np
import pytest
import warnings

from sysidentpy.basis_function import (
    Polynomial,
    Fourier,
    Bilinear,
    Legendre,
    Hermite,
    HermiteNormalized,
    Laguerre,
    Bernstein,
)
from sysidentpy.utils.equation_formatter import (
    results_general,
    format_equation,
    register_equation_renderer,
    RendererRegistry,
    _warned_unknown_bases,
    _format_coefficient,
    _normalize_xlag,
    _ensure_input_names,
    _as_sequence,
    _is_polynomial_model,
)
from sysidentpy.utils.information_matrix import build_lagged_matrix


class DummyModel:
    def __init__(
        self,
        *,
        basis_function,
        xlag,
        ylag,
        n_inputs,
        pivv,
        theta,
        model_type="NARMAX",
    ):
        self.basis_function = basis_function
        self.xlag = xlag
        self.ylag = ylag
        self.n_inputs = n_inputs
        self.pivv = pivv
        self.theta = theta
        self.model_type = model_type


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


def test_polynomial_equation_without_bias_aligns_first_feature_name():
    basis = Polynomial(degree=2, include_bias=False)
    model = DummyModel(
        basis_function=basis,
        xlag=1,
        ylag=1,
        n_inputs=1,
        pivv=np.array([0, 1]),
        theta=np.ones((2, 1)),
    )

    names = [item.name for item in results_general(model)]

    assert names[0] == "y(k-1)"
    assert "1" not in names


@pytest.mark.parametrize(
    ("model_type", "expected_names"),
    [
        (
            "NAR",
            [
                "z(k-1)",
                "z(k-2)",
                "z(k-1)^2",
                "z(k-1)z(k-2)",
                "z(k-2)^2",
            ],
        ),
        (
            "NFIR",
            [
                "u(k-1)",
                "u(k-2)",
                "u(k-1)^2",
                "u(k-1)u(k-2)",
                "u(k-2)^2",
            ],
        ),
    ],
)
def test_polynomial_without_bias_respects_model_type_and_feature_order(
    model_type, expected_names
):
    model = DummyModel(
        basis_function=Polynomial(degree=2, include_bias=False),
        xlag=2,
        ylag=2,
        n_inputs=1,
        pivv=np.arange(5),
        theta=np.ones((5, 1)),
        model_type=model_type,
    )

    names = [
        item.name for item in results_general(model, input_names=["u"], output_name="z")
    ]

    assert names == expected_names


@pytest.mark.parametrize(
    ("basis_cls", "basis_kwargs"),
    [
        (Polynomial, {"degree": 2, "include_bias": False}),
        (Bilinear, {"degree": 2, "include_bias": False}),
        (Fourier, {"degree": 2, "n": 2, "ensemble": True}),
        (Legendre, {"degree": 2, "include_bias": False, "ensemble": True}),
        (Hermite, {"degree": 2, "include_bias": False, "ensemble": True}),
        (
            HermiteNormalized,
            {"degree": 2, "include_bias": False, "ensemble": True},
        ),
        (Laguerre, {"degree": 2, "include_bias": False, "ensemble": True}),
        (Bernstein, {"degree": 2, "include_bias": False, "ensemble": True}),
    ],
)
@pytest.mark.parametrize("model_type", ["NAR", "NFIR"])
def test_native_renderer_width_and_variables_match_model_type(
    basis_cls, basis_kwargs, model_type
):
    rng = np.random.default_rng(193)
    x = rng.normal(size=(12, 1))
    y = rng.normal(size=(12, 1))
    basis_function = basis_cls(**basis_kwargs)
    lagged_data = build_lagged_matrix(
        None if model_type == "NAR" else x,
        y,
        xlag=2,
        ylag=2,
        model_type=model_type,
    )
    regressor_matrix = basis_function.fit(
        lagged_data,
        max_lag=2,
        ylag=2,
        xlag=2,
        model_type=model_type,
        predefined_regressors=None,
    )
    model = DummyModel(
        basis_function=basis_function,
        xlag=2,
        ylag=2,
        n_inputs=1,
        pivv=None,
        theta=np.ones((regressor_matrix.shape[1], 1)),
        model_type=model_type,
    )

    renderer = RendererRegistry[basis_cls.__name__]
    names = renderer(model, ["u"], "z")
    rendered_text = " ".join(names)

    assert len(names) == regressor_matrix.shape[1]
    if model_type == "NAR":
        assert "z(k-" in rendered_text
        assert "u(k-" not in rendered_text
    else:
        assert "u(k-" in rendered_text
        assert "z(k-" not in rendered_text


def test_results_general_accepts_numpy_xlag_single_input():
    basis = Polynomial(degree=1)
    pivv = np.array([0, 1, 2, 3])
    theta = np.ones((len(pivv), 1))
    model = DummyModel(
        basis_function=basis,
        xlag=np.array([1, 2]),
        ylag=1,
        n_inputs=1,
        pivv=pivv,
        theta=theta,
    )
    items = results_general(model)
    names = [item.name for item in items]
    assert "x1(k-2)" in names


def test_results_general_accepts_numpy_object_xlag_multi_input():
    basis = Polynomial(degree=1)
    pivv = np.array([0, 1, 2, 3, 4])
    theta = np.ones((len(pivv), 1))
    xlag = np.array([[1, 2], [1]], dtype=object)
    model = DummyModel(
        basis_function=basis,
        xlag=xlag,
        ylag=1,
        n_inputs=2,
        pivv=pivv,
        theta=theta,
    )
    items = results_general(model, input_names=["u", "v"])
    names = [item.name for item in items]
    assert "v(k-1)" in names


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


def test_fourier_degree_above_two_names_match_historical_polynomial_width():
    basis = Fourier(n=1, degree=3, ensemble=True)
    model = DummyModel(
        basis_function=basis,
        xlag=1,
        ylag=1,
        n_inputs=1,
        pivv=None,
        theta=np.ones((100, 1)),
    )

    names = [item.name for item in results_general(model)]

    # Fourier.fit uses Polynomial() (degree=2) whenever Fourier.degree > 1:
    # five non-bias monomials, followed by cosine/sine for each monomial.
    assert len(names) == 15


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


def test_bernstein_uses_short_symbol_token():
    basis = Bernstein(degree=2)
    model = DummyModel(
        basis_function=basis,
        xlag=1,
        ylag=1,
        n_inputs=1,
        pivv=np.array([0, 1, 2]),
        theta=np.ones((3, 1)),
    )
    items = results_general(model)
    names = [it.name for it in items]
    assert any(name.startswith("B1(") for name in names)


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


def test_bilinear_equation_without_bias_uses_basis_ordering():
    basis = Bilinear(degree=2, include_bias=False)
    model = DummyModel(
        basis_function=basis,
        xlag=2,
        ylag=2,
        n_inputs=1,
        pivv=None,
        theta=np.ones(50),
    )

    names = [item.name for item in results_general(model)]
    expected_count = len(
        basis._get_combination_list(
            np.empty((1, 5)), ylag=2, xlag=2, predefined_regressors=None
        )
    )

    assert len(names) == expected_count
    assert "1" not in names


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


def test_format_equation_omits_constant_multiplier():
    basis = Polynomial(degree=1)
    pivv = np.array([0, 1])
    theta = [2.0, -0.5]
    model = DummyModel(
        basis_function=basis, xlag=1, ylag=1, n_inputs=1, pivv=pivv, theta=theta
    )
    eq = format_equation(model)
    assert "·1" not in eq
    rhs = eq.split("=", 1)[1]
    assert rhs.strip().startswith("2")


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
    RendererRegistry["CustomBasis"] = lambda *_args, **_kwargs: []
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


def test__format_coefficient_positive_first_without_first_sign():
    assert (
        _format_coefficient(1.23, coef_format=".2f", leading=False, first_sign=False)
        == "1.23"
    )


def test__format_coefficient_positive_first_with_first_sign():
    assert (
        _format_coefficient(1.0, coef_format=".1f", leading=False, first_sign=True)
        == "+1.0"
    )


def test__format_coefficient_positive_non_first():
    assert (
        _format_coefficient(2, coef_format=".0f", leading=True, first_sign=False)
        == "+2"
    )


def test__format_coefficient_negative_first():
    assert (
        _format_coefficient(-3.1415, coef_format=".3f", leading=False, first_sign=False)
        == "-3.142"
    )


def test__format_coefficient_negative_non_first_still_negative():
    # leading flag shouldn't affect negative sign handling
    assert (
        _format_coefficient(-5, coef_format=".0f", leading=True, first_sign=True)
        == "-5"
    )


# =====================================================================
# Additional tests for 100% coverage
# =====================================================================


class TestNormalizeXlag:
    """Tests for _normalize_xlag edge cases."""

    def test_n_inputs_zero_returns_empty(self):
        """Line 138: n_inputs <= 0 returns []."""
        assert _normalize_xlag([1, 2], n_inputs=0) == []
        assert _normalize_xlag(2, n_inputs=-1) == []

    def test_xlag_none_returns_empty_lists(self):
        """Line 144: xlag=None returns empty lists per input."""
        assert _normalize_xlag(None, n_inputs=2) == [[], []]
        assert _normalize_xlag(None, n_inputs=3) == [[], [], []]

    def test_xlag_single_integer_entry_in_list(self):
        """Lines 154-155: when entry in seq is Integral, wrap in list."""
        # Each integer in list becomes [int]
        result = _normalize_xlag([1, 2, 3], n_inputs=3)
        assert result == [[1], [2], [3]]

    def test_xlag_fewer_than_n_inputs_pads(self):
        """Line 160: pad with empty lists when normalized < n_inputs."""
        result = _normalize_xlag([[1, 2]], n_inputs=3)
        assert result == [[1, 2], [], []]

    def test_xlag_more_than_n_inputs_trims(self):
        """Line 162: trim when normalized > n_inputs."""
        result = _normalize_xlag([[1], [2], [3], [4]], n_inputs=2)
        assert result == [[1], [2]]


class TestEnsureInputNames:
    """Tests for _ensure_input_names edge cases."""

    def test_pad_input_names_when_fewer_than_n_inputs(self):
        """Line 120: pad with x{i+1} when len(names) < n_inputs."""
        result = _ensure_input_names(n_inputs=3, input_names=["a"])
        assert result == ["a", "x2", "x3"]

    def test_trim_input_names_when_more_than_n_inputs(self):
        """Line 120: trim when len(names) > n_inputs."""
        result = _ensure_input_names(n_inputs=2, input_names=["a", "b", "c", "d"])
        assert result == ["a", "b"]


class TestAsSequence:
    """Tests for _as_sequence edge cases."""

    def test_as_sequence_with_generator(self):
        """Line 129: handles generic iterables via list()."""

        def gen():
            yield 1
            yield 2

        result = _as_sequence(gen())
        assert result == [1, 2]

    def test_as_sequence_with_range(self):
        """_as_sequence handles range objects."""
        result = _as_sequence(range(3))
        assert result == [0, 1, 2]


class TestIsPolynomialModel:
    """Tests for _is_polynomial_model."""

    def test_is_polynomial_model_true(self):
        """Lines 531-532: returns True for Polynomial basis."""
        model = DummyModel(
            basis_function=Polynomial(degree=2),
            xlag=1,
            ylag=1,
            n_inputs=1,
            pivv=None,
            theta=[1.0],
        )
        assert _is_polynomial_model(model) is True

    def test_is_polynomial_model_false_for_other_basis(self):
        """_is_polynomial_model returns False for non-Polynomial."""
        model = DummyModel(
            basis_function=Legendre(degree=2),
            xlag=1,
            ylag=1,
            n_inputs=1,
            pivv=None,
            theta=[1.0],
        )
        assert _is_polynomial_model(model) is False

    def test_is_polynomial_model_false_no_basis(self):
        """_is_polynomial_model returns False when no basis_function."""
        model = DummyModel(
            basis_function=None,
            xlag=1,
            ylag=1,
            n_inputs=1,
            pivv=None,
            theta=[1.0],
        )
        assert _is_polynomial_model(model) is False


class TestRenderersWithoutNInputs:
    """Test renderers when model has no n_inputs attribute."""

    def test_polynomial_renderer_infers_n_inputs_from_xlag_int(self):
        """Lines 386-387: infer n_inputs=1 from int xlag."""

        class ModelNoNInputs:
            basis_function = Polynomial(degree=1)
            xlag = 2
            ylag = 1
            pivv = np.array([0, 1])
            theta = np.array([[1.0], [0.5]])

        model = ModelNoNInputs()
        items = results_general(model)
        assert len(items) == 2

    def test_polynomial_renderer_infers_n_inputs_from_xlag_list(self):
        """Lines 386-387: infer n_inputs from len(xlag) when list."""

        class ModelNoNInputs:
            basis_function = Polynomial(degree=1)
            xlag = [[1], [1, 2]]  # noqa: RUF012
            ylag = 1
            pivv = np.array([0, 3, 4])  # indices 3,4 are v(k-1), v(k-2)
            theta = np.array([[1.0], [0.5], [0.3]])

        model = ModelNoNInputs()
        items = results_general(model, input_names=["u", "v"])
        names = [it.name for it in items]
        assert "v(k-1)" in names or "v(k-2)" in names

    def test_fourier_renderer_infers_n_inputs(self):
        """Lines 403-404: Fourier renderer infers n_inputs."""

        class ModelNoNInputs:
            basis_function = Fourier(n=1, degree=1, ensemble=False)
            xlag = 1
            ylag = 1
            pivv = np.array([0, 1])
            theta = np.array([[1.0], [0.5]])

        model = ModelNoNInputs()
        items = results_general(model)
        assert len(items) == 2
        assert any("cos(" in it.name or "sin(" in it.name for it in items)

    def test_legendre_renderer_infers_n_inputs(self):
        """Lines 421-422: Legendre renderer infers n_inputs."""

        class ModelNoNInputs:
            basis_function = Legendre(degree=2, include_bias=True)
            xlag = 1
            ylag = 1
            pivv = np.array([0, 1])
            theta = np.array([[1.0], [0.5]])

        model = ModelNoNInputs()
        items = results_general(model)
        assert len(items) == 2

    def test_bilinear_renderer_infers_n_inputs(self):
        """Lines 440-441: Bilinear renderer infers n_inputs."""

        class ModelNoNInputs:
            basis_function = Bilinear(degree=2)
            xlag = 1
            ylag = 1
            pivv = np.array([0, 1])
            theta = np.array([[1.0], [0.5]])

        model = ModelNoNInputs()
        items = results_general(model)
        assert len(items) == 2

    def test_fallback_renderer_infers_n_inputs(self):
        """Lines 474-475: Fallback renderer infers n_inputs."""

        class UnknownBasis:
            degree = 1

        class ModelNoNInputs:
            basis_function = UnknownBasis()
            xlag = [[1], [2]]  # noqa: RUF012
            ylag = 1
            pivv = None
            theta = np.array([[1.0], [0.5], [0.3]])

        _warned_unknown_bases.clear()
        model = ModelNoNInputs()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            items = results_general(model)
        assert len(items) == 3


class TestRegisterEquationRendererValidation:
    """Tests for register_equation_renderer validation errors."""

    def test_register_empty_name_raises(self):
        """Line 519: empty name raises ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            register_equation_renderer("", lambda m, i, o: [])

    def test_register_none_name_raises(self):
        """Line 519: None name raises ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            register_equation_renderer(None, lambda m, i, o: [])

    def test_register_non_callable_raises(self):
        """Line 521: non-callable renderer raises ValueError."""
        with pytest.raises(ValueError, match="must be callable"):
            register_equation_renderer("TestBasis", "not_a_function")

    def test_register_duplicate_without_overwrite_raises(self):
        """Lines 523-525: duplicate without overwrite raises ValueError."""
        # Polynomial is already registered
        with pytest.raises(ValueError, match="already registered"):
            register_equation_renderer("Polynomial", lambda m, i, o: [])


class TestResultsGeneralEdgeCases:
    """Tests for results_general edge cases."""

    def test_theta_none_returns_empty(self):
        """Line 554: theta=None returns []."""
        model = DummyModel(
            basis_function=Polynomial(degree=1),
            xlag=1,
            ylag=1,
            n_inputs=1,
            pivv=None,
            theta=None,
        )
        assert results_general(model) == []

    def test_theta_empty_returns_empty(self):
        """Line 557: theta with size 0 returns []."""
        model = DummyModel(
            basis_function=Polynomial(degree=1),
            xlag=1,
            ylag=1,
            n_inputs=1,
            pivv=None,
            theta=np.array([]),
        )
        assert results_general(model) == []


class TestFormatEquationStyles:
    """Tests for format_equation style options."""

    def test_format_equation_empty_model_returns_empty_string(self):
        """Line 601: empty items returns ''."""
        model = DummyModel(
            basis_function=Polynomial(degree=1),
            xlag=1,
            ylag=1,
            n_inputs=1,
            pivv=None,
            theta=None,
        )
        assert format_equation(model) == ""

    def test_format_equation_latex_style(self):
        r"""Line 616: latex style uses \, separator."""
        basis = Polynomial(degree=1)
        pivv = np.array([1, 2])
        theta = np.array([[0.5], [-0.3]])
        model = DummyModel(
            basis_function=basis,
            xlag=1,
            ylag=1,
            n_inputs=1,
            pivv=pivv,
            theta=theta,
        )
        eq = format_equation(model, style="latex")
        # latex uses \, as thin space separator
        assert "\\," in eq

    def test_format_equation_latex_with_intercept(self):
        """Latex style with intercept term (name='1')."""
        basis = Polynomial(degree=1)
        pivv = np.array([0, 1])  # 0 is intercept
        theta = np.array([[2.5], [0.8]])
        model = DummyModel(
            basis_function=basis,
            xlag=1,
            ylag=1,
            n_inputs=1,
            pivv=pivv,
            theta=theta,
        )
        eq = format_equation(model, style="latex")
        assert "2.5" in eq
        # Intercept should appear without multiplication
        assert "\\,1" not in eq


class TestHermiteAndLaguerreRenderers:
    """Tests for Hermite and Laguerre basis functions."""

    def test_hermite_renderer(self):
        """Test Hermite basis function rendering."""
        model = DummyModel(
            basis_function=Hermite(degree=2, include_bias=True),
            xlag=1,
            ylag=1,
            n_inputs=1,
            pivv=None,
            theta=[0.1] * 10,
        )
        items = results_general(model)
        names = [it.name for it in items]
        assert any("H1(" in n or "H2(" in n for n in names)

    def test_laguerre_renderer(self):
        """Test Laguerre basis function rendering."""
        model = DummyModel(
            basis_function=Laguerre(degree=2, include_bias=True),
            xlag=1,
            ylag=1,
            n_inputs=1,
            pivv=None,
            theta=[0.1] * 10,
        )
        items = results_general(model)
        names = [it.name for it in items]
        assert any("L1(" in n or "L2(" in n for n in names)
