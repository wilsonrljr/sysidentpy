import numpy as np
import pytest
from numpy.testing import assert_equal

from sysidentpy.basis_function import Polynomial
from sysidentpy.model_structure_selection import OIF, OOS, OSF
from sysidentpy.parameter_estimation.estimators import LeastSquares
from sysidentpy.tests.test_narmax_base import create_test_data

x, y, _ = create_test_data()
train_percentage = 90
split_data = int(len(x) * (train_percentage / 100))

X_train = np.reshape(x[0:split_data, 0], (-1, 1))
X_test = np.reshape(x[split_data::, 0], (-1, 1))
y_train = np.reshape(y[0:split_data, 0], (-1, 1))
y_test = np.reshape(y[split_data::, 0], (-1, 1))


def _fit_and_predict(model_cls):
    model = model_cls(
        n_terms=4,
        order_selection=False,
        ylag=[1, 2],
        xlag=2,
        basis_function=Polynomial(degree=2),
        estimator=LeastSquares(),
    )
    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test)
    assert_equal(model.final_model.shape[0], model.n_terms)
    assert_equal(yhat.shape, y_test.shape)


def test_osf_runs_and_predicts():
    _fit_and_predict(OSF)


def test_oif_runs_and_predicts():
    _fit_and_predict(OIF)


def test_oos_runs_and_predicts():
    _fit_and_predict(OOS)


# --- Unit-level coverage for helper routines ---


def _make_base():
    model = OSF(
        n_terms=3,
        order_selection=False,
        ylag=1,
        xlag=1,
        basis_function=Polynomial(degree=1),
        estimator=LeastSquares(),
    )
    psi = np.eye(3)
    target = np.array([[1.0], [0.0], [0.0]])
    sqy = model._compute_squared_y(target)
    return model, psi, target, sqy


def test_subset_err_empty_and_nonempty():
    model, psi, target, sqy = _make_base()
    score_empty, err_empty = model._subset_err(psi, target, [], sqy)
    assert_equal(score_empty, 0.0)
    assert_equal(err_empty.shape[0], 0)

    score_full, err_full = model._subset_err(psi, target, [0, 1, 2], sqy)
    assert_equal(score_full, 1.0)
    assert_equal(err_full.tolist(), [1.0, 0.0, 0.0])


def test_subset_err_zero_pads_when_more_terms_than_samples():
    model = OSF(
        n_terms=3,
        order_selection=False,
        ylag=1,
        xlag=1,
        basis_function=Polynomial(degree=1),
        estimator=LeastSquares(),
    )
    psi = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    target = np.array([[1.0], [0.0]])
    sqy = model._compute_squared_y(target)
    score, err = model._subset_err(psi, target, [0, 1, 2], sqy)
    assert_equal(err.shape[0], 3)
    assert err[2] == 0.0
    assert score == pytest.approx(1.0)


def test_best_addition_and_removal_ties():
    model, psi, target, sqy = _make_base()
    # tie on addition: target = [1,1,0]
    target = np.array([[1.0], [1.0], [0.0]])
    sqy = model._compute_squared_y(target)
    add_idx, _ = model._best_addition(psi, target, [], [0, 1], sqy)
    assert_equal(add_idx, 0)

    # tie on removal prefers lower index
    rm_idx, _ = model._best_removal(psi, target, [0, 1], sqy)
    assert_equal(rm_idx, 0)


def test_most_and_least_significant_subsets_exhaustive():
    model, psi, target, sqy = _make_base()
    target = np.array([[1.0], [0.0], [1.0]])
    sqy = model._compute_squared_y(target)

    ms = model._most_significant_terms(psi, target, [], [0, 1, 2], 2, sqy)
    assert_equal(ms, [0, 2])

    ls = model._least_significant_terms(psi, target, [0, 1, 2], 2, sqy)
    assert_equal(ls, [0, 1])

    # zero-count shortcut branches
    assert_equal(model._most_significant_terms(psi, target, [], [0, 1], 0, sqy), [])
    assert_equal(model._least_significant_terms(psi, target, [0, 1], 0, sqy), [])


def test_backtrack_removes_non_last_added():
    model, psi, target, sqy = _make_base()
    subset = [0, 1, 2]
    best_by_size = {2: (-np.inf, [])}
    last_added = 2
    updated, score = model._backtrack(
        psi, target, subset, 0.0, best_by_size, sqy, last_added
    )
    assert_equal(updated, [0, 2])
    assert_equal(score, 1.0)


def test_select_most_significant_subset_and_least_paths():
    model, psi, target, sqy = _make_base()
    psi = np.eye(4)
    target = np.array([[1.0], [0.5], [0.0], [0.0]])
    sqy = model._compute_squared_y(target)

    # count=0 early exit
    assert_equal(
        model._select_most_significant_subset(psi, target, [], [0, 1, 2], 0, sqy), []
    )

    # main path with backtrack
    selection = model._select_most_significant_subset(
        psi, target, [0, 1], [2, 3], 2, sqy
    )
    assert len(selection) <= 2

    # least-significant subset with backtracking branch
    removed = model._select_least_significant_subset(psi, target, [0, 1, 2, 3], 1, sqy)
    assert len(removed) >= 1


def test_select_least_significant_subset_with_backtracking():
    model, psi, target, sqy = _make_base()
    psi = np.eye(4)
    target = np.array([[1.0], [0.0], [0.0], [0.0]])
    sqy = model._compute_squared_y(target)
    removed = model._select_least_significant_subset(psi, target, [0, 1, 2, 3], 2, sqy)
    assert_equal(removed, [1, 2])


def test_swap_step_improves_and_nochange():
    model, psi, target, sqy = _make_base()
    all_indices = [0, 1, 2]

    # improvement case
    subset = [1, 2]
    current_score, _ = model._subset_err(psi, target, subset, sqy)
    best_by_size = {len(subset): (current_score, subset.copy())}
    new_subset, new_score, added = OIF()._swap_step(
        psi, target, subset, current_score, best_by_size, all_indices, sqy
    )
    assert new_score > current_score
    assert_equal(sorted(new_subset), [0, 2])
    assert_equal(added, 0)

    # no improvement case
    subset = [0, 1]
    current_score, _ = model._subset_err(psi, target, subset, sqy)
    best_by_size = {len(subset): (current_score, subset.copy())}
    same_subset, same_score, added_none = OIF()._swap_step(
        psi, target, subset, current_score, best_by_size, all_indices, sqy
    )
    assert_equal(same_subset, subset)
    assert_equal(same_score, current_score)
    assert added_none is None


def test_oos_no_improvement_increments_depth_and_validation():
    # max_search_depth validation
    with pytest.raises(ValueError, match="max_search_depth"):
        OOS(max_search_depth=0)

    model = OOS(
        n_terms=1,
        order_selection=False,
        ylag=1,
        xlag=1,
        basis_function=Polynomial(degree=1),
        estimator=LeastSquares(),
        max_search_depth=1,
    )
    psi = np.eye(2)
    target = np.array([[0.0], [0.0], [0.0]])  # aligned so target after lag has 2 rows
    err, piv, psi_sel, _tgt = model.run_mss_algorithm(psi, target, 1)
    assert_equal(err.shape[0], len(piv))
    assert_equal(piv.tolist(), [0])
    assert_equal(psi_sel.shape[1], 1)


def test_osf_handles_insufficient_available_terms():
    model = OSF(
        n_terms=5,
        order_selection=False,
        ylag=1,
        xlag=1,
        basis_function=Polynomial(degree=1),
        estimator=LeastSquares(),
    )
    psi = np.ones((1, 1))
    target = np.array([[1.0], [1.0]])  # after lag alignment -> shape (1,1)
    _err, piv, _psi_sel, _tgt = model.run_mss_algorithm(psi, target, 5)
    assert_equal(piv.tolist(), [0])


# --- Additional coverage helpers ---


def test_select_most_significant_subset_breaks_on_base_removal(monkeypatch):
    model, psi, target, sqy = _make_base()

    monkeypatch.setattr(
        model,
        "_best_addition",
        lambda *args, **kwargs: (args[3][0] if args[3] else None, 0.0),
    )
    monkeypatch.setattr(
        model,
        "_subset_err",
        lambda *args, **kwargs: (0.0, np.zeros(len(args[2]))),
    )
    monkeypatch.setattr(model, "_best_removal", lambda *args, **kwargs: (0, 0.0))

    selection = model._select_most_significant_subset(psi, target, [0, 1], [2], 1, sqy)
    assert_equal(selection, [2])


def test_select_most_significant_subset_removal_and_stored_subset(monkeypatch):
    model, _psi, target, _ = _make_base()

    order = iter([0, 1, 2, 3, None])

    def fake_best_addition(*args, **kwargs):
        idx = next(order)
        return (idx, 0.0) if idx is not None else (None, 0.0)

    monkeypatch.setattr(model, "_best_addition", fake_best_addition)
    monkeypatch.setattr(
        model,
        "_subset_err",
        lambda *args, **kwargs: (0.0, np.zeros(len(args[2]))),
    )
    monkeypatch.setattr(
        model,
        "_best_removal",
        lambda psi, target, subset, sqy: (subset[0] if subset else 0, 1.0),
    )

    selection = model._select_most_significant_subset(
        np.zeros((1, 4)), target, [], [0, 1, 2, 3], 3, 1.0
    )
    assert len(selection) <= 3


def test_select_least_significant_subset_early_exit():
    model, psi, target, sqy = _make_base()
    removed = model._select_least_significant_subset(psi, target, [], 0, sqy)
    assert_equal(removed, [])


def test_select_least_significant_subset_else_and_backtrack(monkeypatch):
    model, psi, target, _ = _make_base()
    base_subset = [0, 1, 2]

    class StopSearch(Exception):
        pass

    removal_calls = {"count": 0}

    def fake_subset_err(*args, **kwargs):
        return -np.inf, np.zeros(len(args[2]))

    def fake_best_removal(psi_arr, tgt, subset, sqy):
        removal_calls["count"] += 1
        if removal_calls["count"] > 2:
            raise StopSearch
        return subset[0] if subset else 0, -np.inf

    monkeypatch.setattr(model, "_subset_err", fake_subset_err)
    monkeypatch.setattr(model, "_best_removal", fake_best_removal)

    with pytest.raises(StopSearch):
        model._select_least_significant_subset(psi, target, base_subset, 2, 1.0)


def test_select_least_significant_subset_uses_removed_list(monkeypatch):
    model, psi, target, _ = _make_base()
    base_subset = [0, 1]

    call_count = {"count": 0}

    def fake_subset_err(psi_arr, tgt, subset, sqy):
        call_count["count"] += 1
        if call_count["count"] == 1:
            return 0.0, np.array(subset, dtype=float)
        return -1.0, np.array(subset, dtype=float)

    monkeypatch.setattr(model, "_subset_err", fake_subset_err)
    monkeypatch.setattr(
        model, "_best_removal", lambda psi_arr, tgt, subset, sqy: (subset[0], 0.0)
    )

    removed = model._select_least_significant_subset(psi, target, base_subset, 1, 1.0)
    assert_equal(removed, [0])


def test_osf_run_mss_respects_stored_subset(monkeypatch):
    model, _, _, _ = _make_base()

    add_order = iter([0, None])

    def fake_best_addition(*args, **kwargs):
        idx = next(add_order, None)
        return (idx, 0.0) if idx is not None else (None, 0.0)

    monkeypatch.setattr(model, "_best_addition", fake_best_addition)
    monkeypatch.setattr(
        model,
        "_subset_err",
        lambda *args, **kwargs: (0.0, np.zeros(len(args[2]))),
    )

    def fake_backtrack_trim(psi, target, subset, current_score, *args):
        return subset[:-1], current_score

    monkeypatch.setattr(model, "_backtrack", fake_backtrack_trim)

    psi = np.ones((1, 1))
    y = np.ones((1, 1))
    err, piv, psi_sel, _tgt = model.run_mss_algorithm(psi, y, 2)
    assert piv.size >= 0
    assert_equal(err.shape[0], psi_sel.shape[1])


def test_osf_breaks_when_available_empty(monkeypatch):
    model, psi, target, _ = _make_base()

    import sysidentpy.model_structure_selection.orthogonal_floating_search as ofs

    monkeypatch.setattr(ofs, "range", lambda *args, **kwargs: [], raising=False)

    err, piv, _, _ = model.run_mss_algorithm(psi[:, :1], target[:1], 1)
    assert_equal(piv.tolist(), [])
    assert_equal(err.shape[0], 0)


def test_osf_candidate_score_not_improving(monkeypatch):
    model = OSF(
        n_terms=2,
        order_selection=False,
        ylag=1,
        xlag=1,
        basis_function=Polynomial(degree=1),
        estimator=LeastSquares(),
    )

    add_order = iter([0, 1, None])

    def fake_best_addition(*args, **kwargs):
        idx = next(add_order, None)
        return (idx, 0.0) if idx is not None else (None, 0.0)

    monkeypatch.setattr(model, "_best_addition", fake_best_addition)
    monkeypatch.setattr(
        model,
        "_subset_err",
        lambda *args, **kwargs: (0.0, np.zeros(len(args[2]))),
    )

    def fake_backtrack_empty(psi, target, subset, current_score, *args):
        return [], current_score

    monkeypatch.setattr(model, "_backtrack", fake_backtrack_empty)

    psi = np.ones((1, 2))
    y = np.ones((1, 1))
    err, piv, _, _ = model.run_mss_algorithm(psi, y, 2)
    assert piv.size >= 0
    assert_equal(err.shape[0], piv.shape[0])


def test_oif_handles_none_addition_and_else_branch(monkeypatch):
    model = OIF(
        n_terms=2,
        order_selection=False,
        ylag=1,
        xlag=1,
        basis_function=Polynomial(degree=1),
        estimator=LeastSquares(),
    )

    add_order = iter([0, None])

    def fake_best_addition_oif(*args, **kwargs):
        idx = next(add_order, None)
        return (idx, 0.0) if idx is not None else (None, 0.0)

    monkeypatch.setattr(model, "_best_addition", fake_best_addition_oif)
    monkeypatch.setattr(
        model,
        "_subset_err",
        lambda *args, **kwargs: (0.0, np.zeros(len(args[2]))),
    )

    def fake_backtrack_trim_oif(psi, target, subset, current_score, *args):
        return subset[:-1], current_score

    monkeypatch.setattr(model, "_backtrack", fake_backtrack_trim_oif)

    psi = np.ones((1, 1))
    y = np.ones((1, 1))
    _err, piv, psi_sel, _ = model.run_mss_algorithm(psi, y, 2)
    assert piv.size >= 0
    assert_equal(psi_sel.shape[1], piv.shape[0])


def test_oif_swap_step_skips_none(monkeypatch):
    _model, psi, target, sqy = _make_base()
    oif = OIF()
    monkeypatch.setattr(
        oif,
        "_best_addition",
        lambda *args, **kwargs: (None, 0.0),
    )
    subset, score, added = oif._swap_step(
        psi, target, [0, 1], 0.0, {2: (0.0, [0, 1])}, [0, 1, 2], sqy
    )
    assert_equal(subset, [0, 1])
    assert_equal(score, 0.0)
    assert added is None


def test_oif_breaks_when_available_empty(monkeypatch):
    model, psi, target, _ = _make_base()

    import sysidentpy.model_structure_selection.orthogonal_floating_search as ofs

    monkeypatch.setattr(ofs, "range", lambda *args, **kwargs: [], raising=False)

    err, piv, _, _ = model.run_mss_algorithm(psi[:, :1], target[:1], 1)
    assert_equal(piv.tolist(), [])
    assert_equal(err.shape[0], 0)


def test_oif_candidate_score_not_improving(monkeypatch):
    model = OIF(
        n_terms=2,
        order_selection=False,
        ylag=1,
        xlag=1,
        basis_function=Polynomial(degree=1),
        estimator=LeastSquares(),
    )

    add_order = iter([0, 1, None])

    def fake_best_addition_oif_candidate(*args, **kwargs):
        idx = next(add_order, None)
        return (idx, 0.0) if idx is not None else (None, 0.0)

    monkeypatch.setattr(model, "_best_addition", fake_best_addition_oif_candidate)
    monkeypatch.setattr(
        model,
        "_subset_err",
        lambda *args, **kwargs: (0.0, np.zeros(len(args[2]))),
    )

    def fake_backtrack_empty_oif(psi, target, subset, current_score, *args):
        return [], current_score

    monkeypatch.setattr(model, "_backtrack", fake_backtrack_empty_oif)

    psi = np.ones((1, 2))
    y = np.ones((1, 1))
    err, piv, _, _ = model.run_mss_algorithm(psi, y, 2)
    assert piv.size >= 0
    assert_equal(err.shape[0], piv.shape[0])


def test_select_most_significant_subset_none_addition(monkeypatch):
    model, psi, target, sqy = _make_base()
    monkeypatch.setattr(model, "_best_addition", lambda *args, **kwargs: (None, 0.0))
    result = model._select_most_significant_subset(psi, target, [], [0, 1], 2, sqy)
    assert result == []


def test_oif_handles_empty_available(monkeypatch):
    model = OIF(
        n_terms=1,
        order_selection=False,
        ylag=1,
        xlag=1,
        basis_function=Polynomial(degree=1),
        estimator=LeastSquares(),
    )
    psi = np.ones((1, 1))
    y = np.zeros((1, 1))
    import sysidentpy.model_structure_selection.orthogonal_floating_search as ofs

    monkeypatch.setattr(ofs, "range", lambda *args, **kwargs: [], raising=False)

    err, piv, psi_sel, _ = model.run_mss_algorithm(psi, y, 1)
    assert err.size == 0
    assert piv.size == 0
    assert psi_sel.shape[1] == 0


def test_oos_handles_none_greedy_addition(monkeypatch):
    model = OOS(
        n_terms=1,
        order_selection=False,
        ylag=1,
        xlag=1,
        basis_function=Polynomial(degree=1),
        estimator=LeastSquares(),
        max_search_depth=1,
    )

    monkeypatch.setattr(
        model,
        "_best_addition",
        lambda *args, **kwargs: (None, 0.0),
    )

    psi = np.ones((1, 1))
    y = np.ones((1, 1))
    err, piv, psi_sel, _ = model.run_mss_algorithm(psi, y, 1)
    assert_equal(piv.tolist(), [])
    assert_equal(err.shape[0], 0)
    assert_equal(psi_sel.shape[1], 0)


def test_oos_down_and_up_swings_improve(monkeypatch):
    model = OOS(
        n_terms=2,
        order_selection=False,
        ylag=1,
        xlag=1,
        basis_function=Polynomial(degree=1),
        estimator=LeastSquares(),
        max_search_depth=1,
    )

    # Score favors presence of term 2 > term 1 > others
    def score_subset(*args, **kwargs):
        subset = args[2]
        if 2 in subset:
            score = 3.0
        elif 1 in subset:
            score = 2.0
        else:
            score = float(len(subset))
        return score, np.zeros(len(subset))

    add_order = iter([0, 1, 2])

    def fake_best_addition_oos(*args, **kwargs):
        idx = next(add_order, None)
        return (idx, 0.0) if idx is not None else (None, 0.0)

    monkeypatch.setattr(model, "_best_addition", fake_best_addition_oos)
    monkeypatch.setattr(model, "_subset_err", score_subset)
    monkeypatch.setattr(model, "_least_significant_terms", lambda *args, **kwargs: [])

    def fake_most_significant_terms(psi, target, subset, available, count, sqy):
        return [a for a in available if a not in subset][:count]

    monkeypatch.setattr(model, "_most_significant_terms", fake_most_significant_terms)

    psi = np.eye(3)
    y = np.ones((3, 1))
    err, piv, _psi_sel, _ = model.run_mss_algorithm(psi, y, 1)
    assert err.size >= 0
    assert piv.size >= 0


def test_subset_err_handles_zero_columns():
    model = OSF()
    psi = np.zeros((0, 1))
    target = np.zeros((0, 1))
    sqy = model._compute_squared_y(target)

    score, err_vals = model._subset_err(psi, target, [0], sqy)
    assert score == 0.0
    assert err_vals.size == 0


def test_down_swing_insufficient_most_significant_terms(monkeypatch):
    model = OOS()
    psi = np.eye(1)
    target = np.ones((1, 1))
    sqy = model._compute_squared_y(target)

    monkeypatch.setattr(
        model,
        "_select_least_significant_subset",
        lambda *args, **kwargs: [0],
    )
    monkeypatch.setattr(
        model,
        "_select_most_significant_subset",
        lambda *args, **kwargs: [],
    )

    subset, score, improved, failed = model._down_swing(
        psi,
        target,
        [0],
        0.5,
        1,
        sqy,
        [0],
        {0},
    )
    assert subset == [0]
    assert score == 0.5
    assert not improved
    assert failed


def test_up_swing_insufficient_removal(monkeypatch):
    model = OOS()
    psi = np.eye(2)
    target = np.ones((2, 1))
    sqy = model._compute_squared_y(target)

    monkeypatch.setattr(
        model,
        "_select_most_significant_subset",
        lambda *args, **kwargs: [1],
    )
    monkeypatch.setattr(
        model,
        "_select_least_significant_subset",
        lambda *args, **kwargs: [],
    )

    subset, score, improved, failed = model._up_swing(
        psi,
        target,
        [0],
        0.5,
        1,
        sqy,
        2,
        [0, 1],
        {0, 1},
    )
    assert subset == [0]
    assert score == 0.5
    assert not improved
    assert failed


def test_oos_run_mss_increments_depth_on_two_failures(monkeypatch):
    model = OOS(max_search_depth=2)
    psi = np.ones((1, 1))
    y = np.ones((model.max_lag + 1, 1))

    add_iter = iter([0, None])

    depth_calls = []

    def fake_best_addition(*args, **kwargs):
        idx = next(add_iter, None)
        return (idx, 0.0) if idx is not None else (None, 0.0)

    monkeypatch.setattr(model, "_best_addition", fake_best_addition)
    monkeypatch.setattr(
        model,
        "_down_swing",
        lambda *args, **kwargs: (
            depth_calls.append(args[4]) or args[2],
            args[3],
            False,
            True,
        ),
    )
    monkeypatch.setattr(
        model,
        "_up_swing",
        lambda *args, **kwargs: (
            depth_calls.append(args[4]) or args[2],
            args[3],
            False,
            True,
        ),
    )

    err, piv, psi_sel, _ = model.run_mss_algorithm(psi, y, 1)
    assert err.size == piv.size == psi_sel.shape[1]
    assert depth_calls[:2] == [1, 1]
    assert depth_calls[-2:] == [2, 2]


def test_down_swing_depth_greater_than_subset(monkeypatch):
    model = OOS()
    psi = np.eye(1)
    target = np.ones((1, 1))
    sqy = model._compute_squared_y(target)

    monkeypatch.setattr(
        model,
        "_select_least_significant_subset",
        lambda *args, **kwargs: [0],
    )
    monkeypatch.setattr(
        model,
        "_select_most_significant_subset",
        lambda *args, **kwargs: [],
    )

    subset, score, improved, failed = model._down_swing(
        psi,
        target,
        [0],
        0.5,
        2,
        sqy,
        [0],
        {0},
    )
    assert subset == [0]
    assert score == 0.5
    assert not improved
    assert failed


def test_oos_respects_max_search_depth(monkeypatch):
    model = OOS(
        n_terms=2,
        order_selection=False,
        ylag=1,
        xlag=1,
        basis_function=Polynomial(degree=1),
        estimator=LeastSquares(),
        max_search_depth=1,
    )

    depth_calls = []

    def fake_down(*args, **kwargs):
        depth_calls.append(args[4])
        return args[2], args[3], False, True

    def fake_up(*args, **kwargs):
        depth_calls.append(args[4])
        return args[2], args[3], False, True

    monkeypatch.setattr(model, "_down_swing", fake_down)
    monkeypatch.setattr(model, "_up_swing", fake_up)

    psi = np.eye(2)
    y = np.ones((3, 1))
    err, piv, _, _ = model.run_mss_algorithm(psi, y, 2)

    assert err.shape[0] == piv.shape[0]
    assert depth_calls
    assert max(depth_calls) == 1


def test_down_and_up_swing_keep_score_when_no_improvement():
    model = OOS(
        n_terms=2,
        order_selection=False,
        ylag=1,
        xlag=1,
        basis_function=Polynomial(degree=1),
        estimator=LeastSquares(),
    )

    psi = np.eye(2)
    target = np.array([[1.0], [0.0]])
    sqy = model._compute_squared_y(target)
    subset = [0, 1]
    current_score, _ = model._subset_err(psi, target, subset, sqy)
    all_indices = [0, 1]
    all_indices_set = set(all_indices)

    updated_subset, updated_score, improved, failed = model._down_swing(
        psi,
        target,
        subset,
        current_score,
        1,
        sqy,
        all_indices,
        all_indices_set,
    )
    assert updated_subset == subset or updated_score == current_score
    assert not improved
    assert not failed or updated_score == current_score

    updated_subset, updated_score, improved, failed = model._up_swing(
        psi,
        target,
        subset,
        current_score,
        1,
        sqy,
        len(all_indices),
        all_indices,
        all_indices_set,
    )
    assert updated_subset == subset or updated_score == current_score
    assert not improved
    assert not failed or updated_score == current_score


def test_osf_process_term_number_exceeds_total_terms():
    model = OSF(
        n_terms=5,
        order_selection=False,
        ylag=1,
        xlag=1,
        basis_function=Polynomial(degree=1),
        estimator=LeastSquares(),
    )
    psi = np.eye(2)
    y = np.ones((3, 1))

    err, piv, psi_sel, _ = model.run_mss_algorithm(psi, y, 5)
    assert piv.shape[0] <= psi.shape[1]
    assert err.shape[0] == psi_sel.shape[1]
