import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import t as student_t

from sysidentpy.basis_function import Fourier, Polynomial
from sysidentpy.model_structure_selection import (
    meta_model_structure_selection as meta_module,
)
from sysidentpy.model_structure_selection.meta_model_structure_selection import MetaMSS
from sysidentpy.parameter_estimation.estimators import TotalLeastSquares


def test_meta_mss_fit_requires_y():
    model = MetaMSS()
    with pytest.raises(ValueError, match="y cannot be None"):
        model.fit(y=None)


@pytest.mark.parametrize("y", [np.ones(10), np.ones((10, 2))])
def test_meta_mss_fit_requires_one_output_column(y):
    model = MetaMSS()

    with pytest.raises(ValueError, match="exactly one output column"):
        model.fit(y=y)


def test_meta_mss_fit_sets_default_input_count(monkeypatch):
    model = MetaMSS(maxiter=1, n_agents=1, random_state=0)
    captured = {}

    def stop_at_evaluation(*_args):
        captured["n_inputs"] = model.n_inputs
        raise RuntimeError("stop early")

    monkeypatch.setattr(model, "evaluate_objective_function", stop_at_evaluation)

    y = np.arange(8, dtype=float).reshape(-1, 1)
    with pytest.raises(RuntimeError, match="stop early"):
        model.fit(X=None, y=y)

    assert captured["n_inputs"] == 1


def test_meta_mss_fit_records_mean_of_only_finite_candidates(monkeypatch):
    model = MetaMSS(
        maxiter=1,
        n_agents=2,
        xlag=1,
        ylag=1,
        basis_function=Polynomial(degree=1),
        random_state=0,
    )

    monkeypatch.setattr(
        model,
        "evaluate_objective_function",
        lambda *_args: np.array([1.0, np.nan]),
    )

    def fake_simulate(**kwargs):
        model.theta = np.ones((len(kwargs["model_code"]), 1))
        return np.copy(kwargs["y_test"])

    monkeypatch.setattr(model, "simulate", fake_simulate)
    monkeypatch.setattr(model, "_get_max_lag", lambda: 1)

    x = np.arange(20, dtype=float).reshape(-1, 1)
    y = np.arange(20, dtype=float).reshape(-1, 1)
    model.fit(X=x, y=y)

    assert_allclose(model.mean_by_iter, [1.0])
    assert np.all(np.isfinite(model.best_by_iter))


def test_evaluate_objective_function_resamples_empty_agent(monkeypatch):
    model = MetaMSS(loss_func="capture_loss")
    model.dimension = 1
    model.regressor_code = np.array([[1001, 0]])
    model._search_space_max_lag = 1
    model.tested_models = []
    population = np.zeros((1, 1), dtype=int)

    def fake_simulate(self, **kwargs):
        self.max_lag = 1
        self.xlag = 1
        self.ylag = 1
        self.pivv = np.array([0])
        self.theta = np.ones((1, 1))
        return np.copy(kwargs["y_test"])

    monkeypatch.setattr(MetaMSS, "simulate", fake_simulate)
    monkeypatch.setattr(
        MetaMSS,
        "perform_t_test",
        lambda *_args: (np.array([], dtype=np.intp), None, None),
    )
    monkeypatch.setattr(MetaMSS, "_generate_nonempty_agent", lambda self: np.ones(1))
    model.basis_function = type(
        "DummyBasis", (), {"fit": lambda self, *_args, **_kwargs: np.ones((3, 1))}
    )()
    model.capture_loss = lambda *_args: 2.0

    fitness = model.evaluate_objective_function(
        np.zeros((4, 1)),
        np.zeros((4, 1)),
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        population,
    )

    assert fitness == [2.0]
    assert np.array_equal(population[:, 0], np.ones(1))


def test_evaluate_objective_function_runs_loss_without_print(monkeypatch, capsys):
    model = MetaMSS()
    model.theta = np.ones((2, 1))
    model.pivv = None
    model.dimension = 2
    model.regressor_code = np.array([[1001, 0], [2001, 0]])
    model.steps_ahead = None
    model.tested_models = []

    monkeypatch.setattr(MetaMSS, "_get_max_lag", lambda self: 1)

    def fake_simulate(self, **kwargs):
        self.max_lag = 1
        self.theta = np.ones((2, 1))
        self.pivv = np.array([0, 1])
        return np.copy(kwargs["y_test"])

    monkeypatch.setattr(MetaMSS, "simulate", fake_simulate)

    class DummyBasis:
        def fit(self, *args, **kwargs):
            return np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])

    model.basis_function = DummyBasis()

    def fake_t_test(self, psi, theta, residues):
        return np.array([], dtype=int), np.zeros((1, 1)), np.zeros((1, 1))

    monkeypatch.setattr(MetaMSS, "perform_t_test", fake_t_test)

    x_train = np.ones((4, 1))
    y_train = np.ones((4, 1))
    x_test = np.ones((2, 1))
    y_test = np.ones((2, 1))
    population = np.ones((2, 1), dtype=int)

    fitness = model.evaluate_objective_function(
        x_train, y_train, x_test, y_test, population
    )
    printed = capsys.readouterr().out.strip()

    assert fitness == [0]
    assert printed == ""
    assert len(model.tested_models) == 1


def test_perform_t_test_rejects_insufficient_degrees_of_freedom():
    model = MetaMSS()
    psi = np.ones((2, 2))
    theta = np.ones((2, 1))
    residues = np.zeros((2, 1))

    with pytest.raises(ValueError, match="more samples than regressors"):
        model.perform_t_test(psi, theta, residues)


def test_perform_t_test_matches_analytical_ols_calculation():
    model = MetaMSS(p_value=0.05)
    psi = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
    theta = np.array([[1.0], [0.1]])
    residues = np.array([[0.2], [-0.1], [0.05], [-0.2], [0.1]])

    pos, t_test, tail2p = model.perform_t_test(psi, theta, residues)

    dof = psi.shape[0] - psi.shape[1]
    variance = np.sum(residues**2) / dof
    standard_error = np.sqrt(variance * np.diag(np.linalg.inv(psi.T @ psi))).reshape(
        -1, 1
    )
    expected_t = theta / standard_error
    expected_p = 2 * student_t.sf(np.abs(expected_t), dof)

    assert_allclose(t_test, expected_t)
    assert_allclose(tail2p, expected_p)
    expected_positions = np.flatnonzero(expected_p.ravel() > model.p_value).reshape(
        1, -1
    )
    assert np.array_equal(pos, expected_positions)


def test_perform_t_test_handles_perfect_fit_without_nan():
    model = MetaMSS()
    psi = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    theta = np.array([[1.0], [0.5]])
    residues = np.zeros((3, 1))

    _, t_test, tail2p = model.perform_t_test(psi, theta, residues)

    assert np.all(np.isfinite(t_test))
    assert np.all(np.isfinite(tail2p))


def test_perform_t_test_is_invariant_to_response_scale():
    model = MetaMSS()
    psi = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    theta = np.array([[1.0], [0.5]])
    residues = np.array([[0.2], [-0.1], [0.1], [-0.2]])

    _, t_test, p_values = model.perform_t_test(psi, theta, residues)

    for scale in (1e-200, 1e-12, 1e12, 1e200):
        _, scaled_t, scaled_p = model.perform_t_test(
            psi, theta * scale, residues * scale
        )
        assert_allclose(scaled_t, t_test, rtol=1e-13)
        assert_allclose(scaled_p, p_values, rtol=1e-13)


def test_perform_t_test_is_stable_for_ill_conditioned_full_rank_design():
    rng = np.random.default_rng(42)
    n_samples = 200
    x = rng.normal(size=n_samples)
    almost_x = x + 1e-8 * rng.normal(size=n_samples)
    psi = np.column_stack([np.ones(n_samples), x, almost_x])
    y = (1 + 0.7 * x + rng.normal(0, 0.1, n_samples)).reshape(-1, 1)
    theta = np.linalg.lstsq(psi, y, rcond=None)[0]
    residues = y - psi @ theta

    model = MetaMSS(p_value=0.05)
    positions, t_test, p_values = model.perform_t_test(psi, theta, residues)

    psi_pinv = np.linalg.pinv(psi)
    covariance_diagonal = np.sum(psi_pinv**2, axis=1)
    degrees_of_freedom = n_samples - psi.shape[1]
    residual_variance = float(np.sum(residues**2)) / degrees_of_freedom
    expected_t = theta / np.sqrt(residual_variance * covariance_diagonal).reshape(-1, 1)
    expected_p = 2 * student_t.sf(np.abs(expected_t), degrees_of_freedom)

    assert_allclose(t_test, expected_t, rtol=1e-6)
    assert_allclose(p_values, expected_p, rtol=1e-6)
    assert_allclose(positions, [[1, 2]])


def test_perform_t_test_rejects_rank_deficient_regressors():
    model = MetaMSS()
    psi = np.ones((4, 2))
    theta = np.ones((2, 1))
    residues = np.zeros((4, 1))

    with pytest.raises(ValueError, match="full-column-rank"):
        model.perform_t_test(psi, theta, residues)


def test_perform_t_test_rejects_misaligned_residues():
    model = MetaMSS()
    psi = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    theta = np.ones((2, 1))
    residues = np.zeros((2, 1))

    with pytest.raises(ValueError, match="same number of samples"):
        model.perform_t_test(psi, theta, residues)


@pytest.mark.parametrize("p_value", [-0.01, 1.01, np.nan, "0.05", True])
def test_meta_mss_rejects_invalid_p_value(p_value):
    with pytest.raises(ValueError, match=r"p_value.*\[0, 1\]"):
        MetaMSS(p_value=p_value)


@pytest.mark.parametrize(
    ("theta", "residues", "message"),
    [
        (np.ones((2, 2)), np.ones((3, 1)), "theta must contain one column"),
        (np.ones((2, 1)), np.ones((3, 2)), "residues must contain one column"),
    ],
)
def test_perform_t_test_rejects_multioutput_arrays(theta, residues, message):
    model = MetaMSS()
    psi = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])

    with pytest.raises(ValueError, match=message):
        model.perform_t_test(psi, theta, residues)


def test_perform_t_test_rejects_nonfinite_design_before_rank_computation():
    model = MetaMSS()
    psi = np.array([[1.0, 0.0], [1.0, np.nan], [1.0, 2.0]])

    with pytest.raises(ValueError, match="only finite values"):
        model.perform_t_test(psi, np.ones((2, 1)), np.ones((3, 1)))


@pytest.mark.parametrize(
    "psi",
    [
        np.array([[1.0 + 0j, 0j], [1.0 + 0j, 1.0 + 0j], [1.0 + 0j, 2.0j]]),
        np.array([["1", "0"], ["1", "1"], ["1", "2"]]),
    ],
)
def test_perform_t_test_rejects_non_real_numeric_design(psi):
    model = MetaMSS()

    with pytest.raises(ValueError, match="real numeric arrays"):
        model.perform_t_test(psi, np.ones((2, 1)), np.ones((3, 1)))


def test_perform_t_test_accepts_single_output_vectors():
    model = MetaMSS()
    psi = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    theta = np.array([1.0, 0.5])
    residues = np.array([0.2, -0.1, 0.1, -0.2])

    vector_result = model.perform_t_test(psi, theta, residues)
    column_result = model.perform_t_test(
        psi, theta.reshape(-1, 1), residues.reshape(-1, 1)
    )

    for vector_value, column_value in zip(vector_result, column_result, strict=True):
        assert_allclose(vector_value, column_value)


def test_evaluate_residuals_match_actual_ols_design(monkeypatch):
    rng = np.random.default_rng(321)
    x = rng.normal(size=(180, 1))
    y = np.zeros((180, 1))
    noise = rng.normal(0, 0.05, size=180)
    for k in range(2, len(y)):
        y[k, 0] = (
            0.4 * y[k - 1, 0]
            + 0.5 * x[k - 2, 0]
            + 0.1 * y[k - 1, 0] * x[k - 1, 0]
            + noise[k]
        )

    model = MetaMSS(
        xlag=2,
        ylag=2,
        basis_function=Polynomial(degree=2),
        loss_func="capture_loss",
        random_state=0,
    )
    model.n_inputs = 1
    model.regressor_code = model.regressor_space(model.n_inputs)
    model.dimension = model.regressor_code.shape[0]
    model.tested_models = []
    captured = {}
    original_t_test = model.perform_t_test

    def capture_t_test(psi, theta, residues):
        captured["psi"] = psi.copy()
        captured["theta"] = theta.copy()
        captured["residues"] = residues.copy()
        _, t_values, p_values = original_t_test(psi, theta, residues)
        return np.array([], dtype=np.intp), t_values, p_values

    monkeypatch.setattr(model, "perform_t_test", capture_t_test)
    model.capture_loss = lambda *_args: 0.0
    population = np.ones((model.dimension, 1), dtype=int)
    x_train, x_test = x[:140], x[140:]
    y_train, y_test = y[:140], y[140:]

    fitness = model.evaluate_objective_function(
        x_train, y_train, x_test, y_test, population
    )

    psi = captured["psi"]
    target = y_train[2:]
    expected_theta = np.linalg.lstsq(psi, target, rcond=None)[0]
    expected_residues = target - psi @ expected_theta
    assert fitness == [0.0]
    assert psi.shape == (y_train.shape[0] - 2, model.dimension)
    assert_allclose(captured["theta"], expected_theta, rtol=1e-12, atol=1e-12)
    assert_allclose(captured["residues"], expected_residues, rtol=1e-12, atol=1e-12)
    assert_allclose(psi.T @ captured["residues"], 0, atol=1e-11)


def test_aic_computation():
    model = MetaMSS()
    y = np.array([[1.0], [2.0], [3.0]])
    yhat = np.array([[1.1], [1.9], [3.1]])

    result = model.aic(y, yhat, 2)
    expected = y.shape[0] * np.log(np.mean((y - yhat) ** 2)) + 2 * 2

    assert pytest.approx(result) == expected


@pytest.mark.parametrize("criterion", ["aic", "bic"])
def test_information_criteria_are_finite_for_perfect_fit(criterion):
    model = MetaMSS()
    y = np.arange(1, 5, dtype=float).reshape(-1, 1)

    result = getattr(model, criterion)(y, y.copy(), 2)

    assert np.isfinite(result)


def test_bic_computation():
    model = MetaMSS()
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    yhat = np.array([[0.9], [2.1], [3.0], [4.2]])

    result = model.bic(y, yhat, 3)
    mse = np.mean((y - yhat) ** 2)
    expected = y.shape[0] * np.log(mse) + 3 * np.log(y.shape[0])

    assert pytest.approx(result) == expected


def test_metamss_loss_returns_zero_for_perfect_constant_prediction():
    model = MetaMSS()
    model.dimension = 3
    y_test = np.ones((3, 1))
    yhat = np.ones((3, 1))

    fitness = model.metamss_loss(y_test, yhat, n_terms=1)

    assert fitness == 0


def test_metamss_loss_returns_fallback_for_nonfinite_error():
    model = MetaMSS()
    model.dimension = 3
    y_test = np.ones((3, 1))
    yhat = np.zeros((3, 1))

    fitness = model.metamss_loss(y_test, yhat, n_terms=1)

    assert fitness == 30


def test_evaluate_uses_identification_residues_and_common_scoring_window(
    monkeypatch,
):
    model = MetaMSS(loss_func="capture_loss")
    model.dimension = 2
    model.regressor_code = np.array([[1001, 0], [2001, 0]])
    model.steps_ahead = None
    model.tested_models = []
    model._search_space_max_lag = 2
    psi = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    captured = {}

    class DummyBasis:
        def fit(self, *args, **kwargs):
            return psi

    model.basis_function = DummyBasis()
    monkeypatch.setattr(meta_module, "build_lagged_matrix", lambda *_args: None)

    def fake_simulate(self, **kwargs):
        self.max_lag = 1
        self.xlag = 1
        self.ylag = 1
        self.pivv = np.array([0, 1])
        self.theta = np.array([[1.0], [1.0]])
        captured["validation_y"] = kwargs["y_test"].copy()
        return kwargs["y_test"].copy()

    def fake_t_test(self, psi_value, theta, residues):
        captured["residues"] = residues.copy()
        return np.array([], dtype=np.intp), np.zeros_like(theta), np.zeros_like(theta)

    def capture_loss(y_value, yhat_value, n_terms):
        captured["score"] = (y_value.copy(), yhat_value.copy(), n_terms)
        return 12.0

    monkeypatch.setattr(MetaMSS, "simulate", fake_simulate)
    monkeypatch.setattr(MetaMSS, "perform_t_test", fake_t_test)
    model.capture_loss = capture_loss
    x_train = np.ones((4, 1))
    y_train = np.array([[99.0], [1.5], [2.5], [3.5]])
    x_test = np.ones((4, 1))
    y_test = np.array([[10.0], [20.0], [3.0], [4.0]])

    fitness = model.evaluate_objective_function(
        x_train, y_train, x_test, y_test, np.ones((2, 1), dtype=int)
    )

    assert fitness == [12.0]
    assert_allclose(captured["residues"], np.full((3, 1), 0.5))
    assert_allclose(captured["validation_y"], np.vstack((y_train[-1:], y_test)))
    assert_allclose(captured["score"][0], y_test)
    assert_allclose(captured["score"][1], y_test)
    assert captured["score"][2] == 2


def test_validation_initial_conditions_always_come_from_identification_tail():
    x_train = np.arange(8, dtype=float).reshape(-1, 1)
    y_train = (10 + np.arange(8, dtype=float)).reshape(-1, 1)
    x_test = np.arange(3, dtype=float).reshape(-1, 1) + 100
    y_test = np.arange(3, dtype=float).reshape(-1, 1) + 200

    for candidate_lag in (1, 3):
        x_validation, y_validation = MetaMSS._validation_data_with_training_tail(
            x_train, y_train, x_test, y_test, candidate_lag
        )

        assert_allclose(x_validation[:candidate_lag], x_train[-candidate_lag:])
        assert_allclose(y_validation[:candidate_lag], y_train[-candidate_lag:])
        assert_allclose(x_validation[candidate_lag:], x_test)
        assert_allclose(y_validation[candidate_lag:], y_test)


def test_evaluate_adds_removed_terms_to_metamss_penalty(monkeypatch):
    model = MetaMSS(loss_func="metamss_loss")
    model.dimension = 2
    model.regressor_code = np.array([[1001, 0], [2001, 0]])
    model.steps_ahead = None
    model.tested_models = []
    model._search_space_max_lag = 1
    captured = {}

    class DummyBasis:
        def fit(self, *args, **kwargs):
            return np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])

    model.basis_function = DummyBasis()
    monkeypatch.setattr(meta_module, "build_lagged_matrix", lambda *_args: None)

    def fake_simulate(self, **kwargs):
        n_terms = len(kwargs["model_code"])
        self.max_lag = 1
        self.xlag = 1
        self.ylag = 1
        self.pivv = np.arange(n_terms)
        self.theta = np.ones((n_terms, 1))
        return np.copy(kwargs["y_test"])

    def fake_t_test(self, psi, theta, residues):
        return np.array([0]), np.zeros_like(theta), np.zeros_like(theta)

    def capture_loss(y_value, yhat_value, n_terms):
        captured["n_terms"] = n_terms
        return 0.0

    monkeypatch.setattr(MetaMSS, "simulate", fake_simulate)
    monkeypatch.setattr(MetaMSS, "perform_t_test", fake_t_test)
    model.metamss_loss = capture_loss
    population = np.ones((2, 1), dtype=int)

    model.evaluate_objective_function(
        np.ones((4, 1)),
        np.arange(4, dtype=float).reshape(-1, 1),
        np.ones((3, 1)),
        np.ones((3, 1)),
        population,
    )

    assert captured["n_terms"] == 2
    assert np.array_equal(population[:, 0], np.array([0, 1]))


def test_evaluate_resamples_when_all_terms_are_insignificant(
    monkeypatch,
):
    model = MetaMSS(loss_func="metamss_loss")
    model.dimension = 2
    model.regressor_code = np.array([[1001, 0], [2001, 0]])
    model.steps_ahead = None
    model.tested_models = []
    model._search_space_max_lag = 1

    class DummyBasis:
        def fit(self, *args, **kwargs):
            n_terms = len(kwargs["predefined_regressors"])
            if n_terms == 1:
                return np.ones((3, 1))
            return np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])

    model.basis_function = DummyBasis()
    monkeypatch.setattr(meta_module, "build_lagged_matrix", lambda *_args: None)

    def fake_simulate(self, **kwargs):
        n_terms = len(kwargs["model_code"])
        self.max_lag = 1
        self.xlag = 1
        self.ylag = 1
        self.pivv = np.arange(n_terms)
        self.theta = np.ones((n_terms, 1))
        return np.copy(kwargs["y_test"])

    calls = 0

    def fake_t_test(self, psi, theta, residues):
        nonlocal calls
        calls += 1
        if calls == 1:
            return np.arange(theta.shape[0]), np.zeros_like(theta), np.ones_like(theta)
        return np.array([], dtype=np.intp), np.zeros_like(theta), np.zeros_like(theta)

    monkeypatch.setattr(MetaMSS, "simulate", fake_simulate)
    monkeypatch.setattr(MetaMSS, "perform_t_test", fake_t_test)
    monkeypatch.setattr(
        MetaMSS, "_generate_nonempty_agent", lambda self: np.array([0, 1])
    )
    population = np.ones((2, 1), dtype=int)

    model.evaluate_objective_function(
        np.ones((4, 1)),
        np.arange(4, dtype=float).reshape(-1, 1),
        np.ones((3, 1)),
        np.ones((3, 1)),
        population,
    )

    assert np.array_equal(population[:, 0], np.array([0, 1]))
    assert calls == 2


def test_evaluate_skips_t_test_for_rank_deficient_candidate(monkeypatch):
    model = MetaMSS(loss_func="capture_loss")
    model.dimension = 2
    model.regressor_code = np.array([[1001, 0], [2001, 0]])
    model.tested_models = []
    model._search_space_max_lag = 1

    class DummyBasis:
        def fit(self, *args, **kwargs):
            return np.ones((3, 2))

    model.basis_function = DummyBasis()
    monkeypatch.setattr(meta_module, "build_lagged_matrix", lambda *_args: None)

    def fake_simulate(self, **kwargs):
        self.max_lag = 1
        self.xlag = 1
        self.ylag = 1
        self.pivv = np.array([0, 1])
        self.theta = np.ones((2, 1))
        return np.copy(kwargs["y_test"])

    def forbidden_t_test(*_args, **_kwargs):
        raise AssertionError("rank-deficient candidates must not be pruned")

    monkeypatch.setattr(MetaMSS, "simulate", fake_simulate)
    monkeypatch.setattr(MetaMSS, "perform_t_test", forbidden_t_test)
    model.capture_loss = lambda *_args: 1.0
    population = np.ones((2, 1), dtype=int)

    fitness = model.evaluate_objective_function(
        np.ones((4, 1)),
        np.arange(4, dtype=float).reshape(-1, 1),
        np.ones((3, 1)),
        np.ones((3, 1)),
        population,
    )

    assert fitness == [1.0]
    assert np.array_equal(population[:, 0], np.ones(2, dtype=int))


def test_evaluate_skips_ols_t_test_for_non_ols_estimator(monkeypatch):
    model = MetaMSS(loss_func="capture_loss", estimator=TotalLeastSquares())
    model.dimension = 1
    model.regressor_code = np.array([[1001, 0]])
    model.tested_models = []
    model._search_space_max_lag = 1

    class DummyBasis:
        def fit(self, *args, **kwargs):
            return np.ones((3, 1))

    model.basis_function = DummyBasis()
    monkeypatch.setattr(meta_module, "build_lagged_matrix", lambda *_args: None)

    def fake_simulate(self, **kwargs):
        self.max_lag = 1
        self.xlag = 1
        self.ylag = 1
        self.pivv = np.array([0])
        self.theta = np.ones((1, 1))
        return np.copy(kwargs["y_test"])

    def forbidden_t_test(*_args, **_kwargs):
        raise AssertionError("OLS inference must not prune ridge estimates")

    monkeypatch.setattr(MetaMSS, "simulate", fake_simulate)
    monkeypatch.setattr(MetaMSS, "perform_t_test", forbidden_t_test)
    model.capture_loss = lambda *_args: 1.0

    fitness = model.evaluate_objective_function(
        np.ones((4, 1)),
        np.arange(4, dtype=float).reshape(-1, 1),
        np.ones((3, 1)),
        np.ones((3, 1)),
        np.ones((1, 1), dtype=int),
    )

    assert fitness == [1.0]


def test_fit_accepts_validation_set_not_longer_than_search_lag(monkeypatch):
    model = MetaMSS(
        ylag=5,
        xlag=1,
        model_type="NAR",
        maxiter=1,
        n_agents=2,
        random_state=0,
        test_size=0.1,
    )
    y = np.random.default_rng(42).normal(size=(20, 1))

    monkeypatch.setattr(
        model,
        "evaluate_objective_function",
        lambda *_args: np.array([1.0, 2.0]),
    )

    def fake_simulate(**kwargs):
        model.theta = np.ones((len(kwargs["model_code"]), 1))
        return kwargs["y_test"].copy()

    monkeypatch.setattr(model, "simulate", fake_simulate)

    model.fit(y=y)

    assert model.final_model is not None


def test_fit_rejects_identification_set_not_longer_than_search_lag():
    model = MetaMSS(
        ylag=5,
        xlag=1,
        model_type="NAR",
        maxiter=1,
        n_agents=2,
        random_state=0,
        test_size=0.6,
    )

    with pytest.raises(ValueError, match="identification set must contain more"):
        model.fit(y=np.arange(12, dtype=float).reshape(-1, 1))


def test_invalid_fit_does_not_advance_external_generator():
    random_state = np.random.default_rng(42)
    state_before = random_state.bit_generator.state.copy()
    model = MetaMSS(
        ylag=5,
        xlag=1,
        model_type="NAR",
        maxiter=1,
        n_agents=2,
        random_state=random_state,
        test_size=0.25,
    )

    with pytest.raises(ValueError, match="identification set must contain more"):
        model.fit(y=np.arange(6, dtype=float).reshape(-1, 1))

    assert random_state.bit_generator.state == state_before


def test_invalid_fit_preserves_existing_model_state():
    model = MetaMSS(
        ylag=5,
        xlag=1,
        model_type="NAR",
        maxiter=1,
        n_agents=2,
        random_state=0,
        test_size=0.25,
    )
    existing_model = np.array([[1001, 0]])
    existing_space = np.array([[0, 0], [1001, 0]])
    model.xlag = 1
    model.ylag = 1
    model.max_lag = 1
    model.n_inputs = 1
    model.final_model = existing_model.copy()
    model.regressor_code = existing_space.copy()

    with pytest.raises(ValueError, match="identification set must contain more"):
        model.fit(y=np.arange(6, dtype=float).reshape(-1, 1))

    assert model.xlag == 1
    assert model.ylag == 1
    assert model.max_lag == 1
    assert model.n_inputs == 1
    assert np.array_equal(model.final_model, existing_model)
    assert np.array_equal(model.regressor_code, existing_space)


def test_fit_supports_state_loaded_from_legacy_pickle(monkeypatch):
    model = MetaMSS(maxiter=1, n_agents=1, random_state=0)
    del model._search_xlag
    del model._search_ylag

    def stop_at_evaluation(*_args):
        raise RuntimeError("state restored")

    monkeypatch.setattr(model, "evaluate_objective_function", stop_at_evaluation)

    with pytest.raises(RuntimeError, match="state restored"):
        model.fit(y=np.arange(8, dtype=float).reshape(-1, 1))

    assert model._search_xlag == 1
    assert model._search_ylag == 1


def test_predict_free_run_uses_model_prediction(monkeypatch):
    model = MetaMSS()
    model.max_lag = 1
    calls = {}

    def fake_model_prediction(self, X, y, forecast_horizon=None):
        calls["args"] = (X, y, forecast_horizon)
        return np.array([[10.0], [11.0], [12.0]])

    monkeypatch.setattr(MetaMSS, "_model_prediction", fake_model_prediction)

    X = np.arange(4, dtype=float).reshape(-1, 1)
    y = np.arange(4, dtype=float).reshape(-1, 1)
    result = model.predict(X=X, y=y, forecast_horizon=2)

    assert calls["args"][2] == 2
    assert np.array_equal(result[: model.max_lag], y[: model.max_lag])
    assert np.array_equal(
        result[model.max_lag :],
        np.array([[10.0], [11.0], [12.0]]),
    )


def test_predict_one_step_branch(monkeypatch):
    model = MetaMSS()
    model.max_lag = 1

    def fake_one_step(self, X, y):
        return np.array([[5.0], [6.0], [7.0]])

    monkeypatch.setattr(MetaMSS, "_one_step_ahead_prediction", fake_one_step)

    X = np.arange(4, dtype=float).reshape(-1, 1)
    y = np.arange(4, dtype=float).reshape(-1, 1)
    result = model.predict(X=X, y=y, steps_ahead=1)

    assert np.array_equal(result[: model.max_lag], y[: model.max_lag])
    assert np.array_equal(
        result[model.max_lag :],
        np.array([[5.0], [6.0], [7.0]]),
    )


def test_predict_n_step_branch(monkeypatch):
    model = MetaMSS()
    model.max_lag = 2

    def fake_n_step(self, X, y, steps_ahead=None):
        return np.array([[7.0], [8.0], [9.0], [10.0]])

    monkeypatch.setattr(MetaMSS, "_n_step_ahead_prediction", fake_n_step)

    X = np.arange(6, dtype=float).reshape(-1, 1)
    y = np.arange(6, dtype=float).reshape(-1, 1)
    result = model.predict(X=X, y=y, steps_ahead=2)

    assert np.array_equal(result[: model.max_lag], y[: model.max_lag])
    assert np.array_equal(
        result[model.max_lag :],
        np.array([[7.0], [8.0], [9.0], [10.0]]),
    )


@pytest.mark.parametrize("steps_ahead", [None, 1, 3])
def test_predict_rejects_non_polynomial_basis(steps_ahead):
    model = MetaMSS(basis_function=Fourier())

    with pytest.raises(NotImplementedError, match="other than polynomial"):
        model.predict(
            X=np.ones((2, 1)),
            y=np.ones((2, 1)),
            steps_ahead=steps_ahead,
        )


def test_predict_nar_promotes_integer_output_without_mutating_state():
    model = MetaMSS(
        model_type="NAR",
        basis_function=Polynomial(degree=1, include_bias=False),
    )
    model.max_lag = 1
    model.n_inputs = 1
    model.final_model = np.array([[1001]])
    model.theta = np.array([[0.5]])
    y = np.array([[3], [100], [100], [100], [100]])

    prediction = model.predict(X=None, y=y, steps_ahead=2)

    assert model.n_inputs == 1
    assert np.issubdtype(prediction.dtype, np.floating)
    np.testing.assert_allclose(
        prediction,
        np.array([[3.0], [1.5], [0.75], [50.0], [25.0]]),
        rtol=1e-12,
        atol=1e-12,
    )


def test_predict_nfir_uses_shared_memoryless_contract():
    model = MetaMSS(
        model_type="NFIR",
        basis_function=Polynomial(degree=1, include_bias=False),
    )
    model.max_lag = 1
    model.n_inputs = 1
    model.final_model = np.array([[2001]])
    model.pivv = np.array([0])
    model.theta = np.array([[0.5]])
    x = np.arange(1, 7, dtype=np.int64).reshape(-1, 1)
    y = np.full((6, 1), 9, dtype=np.int64)

    prediction = model.predict(X=x, y=y, steps_ahead=3)

    assert prediction.shape == y.shape
    assert np.issubdtype(prediction.dtype, np.floating)
    np.testing.assert_allclose(
        prediction,
        np.array([[9.0], [0.5], [1.0], [1.5], [2.0], [2.5]]),
        rtol=1e-12,
        atol=1e-12,
    )
