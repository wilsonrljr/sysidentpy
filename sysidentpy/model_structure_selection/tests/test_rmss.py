import numpy as np
import pytest

from sysidentpy import config_context
from sysidentpy.basis_function import Polynomial
from sysidentpy.model_structure_selection import (
    robust_model_structure_selection as rmss_module,
)
from sysidentpy.model_structure_selection import RMSS
from sysidentpy.parameter_estimation.estimators import LeastSquares
from sysidentpy.tests.test_narmax_base import create_test_data

x_full, y_full, _ = create_test_data()
x_small = x_full[:60]
y_small = y_full[:60]

split_data = 40
X_train = np.reshape(x_small[:split_data, 0], (-1, 1))
X_test = np.reshape(x_small[split_data:, 0], (-1, 1))
y_train = np.reshape(y_small[:split_data, 0], (-1, 1))
y_test = np.reshape(y_small[split_data:, 0], (-1, 1))


def test_rmss_basic_fit():
    model = RMSS(
        n_terms=3,
        order_selection=False,
        ylag=[1, 2],
        xlag=2,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=2),
        average_theta=False,
    )

    model.fit(X=X_train, y=y_train)

    assert model.final_model.shape == (3, 2)
    assert model.theta.shape[0] == 3
    assert model.pivv.shape[0] == 3


def test_rmss_invalid_error_measure():
    with pytest.raises(ValueError, match="error_measure"):
        RMSS(error_measure="invalid", basis_function=Polynomial(degree=2))


def test_rmss_predict_shape():
    model = RMSS(
        n_terms=3,
        order_selection=False,
        ylag=[1, 2],
        xlag=2,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=2),
        average_theta=False,
    )

    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test)
    assert yhat.shape == y_test.shape


def test_rmss_rejects_array_api_dispatch_with_clear_error():
    xp = pytest.importorskip("array_api_strict")
    model = RMSS(
        n_terms=2,
        order_selection=False,
        ylag=1,
        xlag=1,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=1),
    )

    with config_context(array_api_dispatch=True):
        with pytest.raises(NotImplementedError, match=r"RMSS.*requires NumPy"):
            model.fit(X=xp.asarray(X_train[:10]), y=xp.asarray(y_train[:10]))


def test_rmss_average_theta_warns():
    model = RMSS(
        n_terms=2,
        order_selection=False,
        ylag=[1, 2],
        xlag=2,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=2),
        average_theta=False,
    )

    with pytest.warns(UserWarning, match="average_theta=False skips"):
        model.fit(X=X_train, y=y_train)


def test_rmss_bootstrap_fit():
    model = RMSS(
        n_terms=2,
        order_selection=False,
        ylag=[1, 2],
        xlag=2,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=2),
        resampling="bootstrap",
        n_subsets=5,
        subset_size=20,
        random_state=0,
    )

    model.fit(X=X_train, y=y_train)
    assert model.theta.shape[0] == 2
    assert model.pivv.shape[0] == 2


def test_rmss_multi_dataset_unbiased_warns():
    model = RMSS(
        n_terms=2,
        order_selection=False,
        ylag=[1, 2],
        xlag=2,
        estimator=LeastSquares(unbiased=True),
        basis_function=Polynomial(degree=2),
    )

    X_list = [X_train, X_train]
    y_list = [y_train, y_train]

    with pytest.warns(UserWarning, match="Unbiased correction is not applied"):
        model.fit(X=X_list, y=y_list)
    assert model.theta.shape[0] == 2
    assert model.pivv.shape[0] == 2


@pytest.mark.parametrize("error_measure", ["mae", "mse", "smape", "rmse_ratio"])
def test_rmss_error_measure_variants(error_measure):
    model = RMSS(
        n_terms=2,
        order_selection=False,
        ylag=[1, 2],
        xlag=2,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=2),
        error_measure=error_measure,
    )

    model.fit(X=X_train, y=y_train)
    assert model.theta.shape[0] == 2


def test_rmss_invalid_resampling():
    with pytest.raises(ValueError, match="Unsupported resampling strategy"):
        RMSS(resampling="bad")


def test_rmss_smape_warning_sets_phi3():
    with pytest.warns(DeprecationWarning, match="smape"):
        model = RMSS(error_measure="smape")
    assert model.error_measure == "phi3"


def test_rmss_average_theta_type_error():
    with pytest.raises(TypeError, match="average_theta must be a boolean"):
        RMSS(average_theta="yes")


def test_rmss_bootstrap_param_validation():
    with pytest.raises(ValueError, match="n_subsets must be a positive"):
        RMSS(resampling="bootstrap", n_subsets=0)

    with pytest.raises(ValueError, match="subset_size must be a positive"):
        RMSS(resampling="bootstrap", subset_size=0)

    with pytest.raises(TypeError, match="random_state must be an integer"):
        RMSS(resampling="bootstrap", random_state="seed")


def test_rmss_multi_resampling_type_error():
    with pytest.raises(TypeError, match="multi_resampling must be a boolean"):
        RMSS(multi_resampling="yes")


def test_create_sub_datasets_too_small():
    model = RMSS()
    reg = np.ones((1, 2))
    tgt = np.ones((1, 1))
    with pytest.raises(ValueError, match="Need at least two samples"):
        model._create_sub_datasets(reg, tgt)


@pytest.mark.parametrize("metric", ["mse", "phi3", "rmse_ratio"])
def test_overall_error_branches(metric):
    model = RMSS(error_measure=metric)
    psi = np.array(
        [
            [[1.0, 2.0], [2.0, 0.0]],
            [[0.5, 1.5], [1.0, 1.0]],
        ]
    )
    y_views = np.array([[1.0, 2.0], [0.5, 1.5]])
    out = model._overall_error(psi, y_views)
    assert out.shape == (2,)


def test_overall_error_multi_resampled_views():
    model = RMSS(error_measure="mae")
    psi_views = np.array([[[1.0], [0.0]], [[2.0], [1.0]]])
    y_views = np.array([[1.0, 0.0], [2.0, 1.0]])
    metric = model._overall_error_multi([psi_views], [y_views])
    assert metric.shape == (1,)


def test_orthogonalize_remaining_helpers():
    model = RMSS()
    psi_views = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.5], [0.5, -0.5]],
        ]
    )
    selected_q = psi_views[:, :, 0]
    ortho = model._orthogonalize_remaining_views(psi_views, selected_q)
    # new first column should be orthogonal to selected_q for each view
    for k in range(ortho.shape[0]):
        assert np.allclose(np.dot(ortho[k, :, 1], selected_q[k]), 0.0)

    psi = np.array([[1.0, 0.0], [0.0, 1.0]])
    q = psi[:, 0]
    ortho_multi = model._orthogonalize_remaining_multi([psi], [q])[0]
    assert np.allclose(np.dot(ortho_multi[:, 1], q), 0.0)


def test_prepare_datasets_length_mismatch():
    model = RMSS()
    with pytest.raises(ValueError, match="X and y lists must have the same length"):
        model._prepare_datasets([np.ones((5, 1))], [np.ones((5, 1)), np.ones((5, 1))])


def test_prepare_datasets_input_dim_mismatch():
    model = RMSS()
    X_list = [np.ones((5, 1)), np.ones((5, 2))]
    y_list = [np.ones((5, 1)), np.ones((5, 1))]
    with pytest.raises(ValueError, match="same number"):
        model._prepare_datasets(X_list, y_list)


def test_run_mss_algorithm_err_tol_break():
    model = RMSS(err_tol=0.0)
    psi = np.array([[1.0, 0.5], [0.5, 1.0]])
    y = np.array([[1.0], [0.0]])
    err, piv, _, _ = model.run_mss_algorithm(psi, y, process_term_number=2)
    assert len(piv) == 1  # stopped early due to err_tol
    assert err.shape[0] == 1


def test_run_mss_algorithm_multi_resampling_views():
    model = RMSS(multi_resampling=True, n_terms=2, order_selection=False)
    rm1 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    rm2 = np.array([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]])
    tgt1 = np.array([[1.0], [0.0], [1.0]])
    tgt2 = np.array([[0.5], [1.0], [0.0]])
    err, piv, _, _ = model.run_mss_algorithm([rm1, rm2], [tgt1, tgt2], 2)
    assert piv.size >= 1
    assert err.size >= 1


def test_estimate_theta_unbiased_single_dataset():
    model = RMSS(ylag=1, xlag=1, estimator=LeastSquares(unbiased=True))
    reg = np.array([[1.0], [2.0], [3.0]])
    tgt = np.array([[1.0], [2.0], [3.0]])
    theta = model._estimate_theta([reg], [tgt], piv=np.array([0]))
    assert theta.shape == (1, 1)


def test_information_criterion_multi_dataset_apress():
    model = RMSS(info_criteria="apress", n_info_values=2)
    rm1 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    rm2 = np.array([[1.0, 1.0], [0.5, 0.5], [0.0, 1.0]])
    tgt1 = np.array([[1.0], [0.0], [1.0]])
    tgt2 = np.array([[0.5], [1.0], [0.0]])
    info = model.information_criterion([rm1, rm2], [tgt1, tgt2])
    assert info.shape == (2,)


def test_fit_with_order_selection_true():
    model = RMSS(
        order_selection=True, info_criteria="apress", n_terms=None, n_info_values=2
    )
    model.fit(X=X_train, y=y_train)
    assert model.n_terms >= 1
    assert model.info_values.shape[0] == 2


def test_fit_order_selection_false_without_n_terms():
    model = RMSS(order_selection=False, n_terms=None)
    with pytest.raises(ValueError, match="define n_terms value"):
        model.fit(X=X_train, y=y_train)


def test_create_sub_datasets_invalid_strategy_after_init():
    model = RMSS()
    model.resampling = "invalid"
    reg = np.ones((2, 1))
    tgt = np.ones((2, 1))
    with pytest.raises(ValueError, match="Unsupported resampling strategy"):
        model._create_sub_datasets(reg, tgt)


def test_overall_error_multi_rmse_ratio():
    model = RMSS(error_measure="rmse_ratio")
    psi_list = [np.array([[1.0, 0.0], [0.0, 1.0]])]
    y_list = [np.array([[1.0], [0.0]])]
    metric = model._overall_error_multi(psi_list, y_list)
    assert metric.shape == (2,)


def test_prepare_datasets_replicates_single_x_for_list_y():
    model = RMSS()
    X = np.ones((5, 1))
    y_list = [np.ones((5, 1)), np.ones((5, 1))]
    reg_matrices, targets = model._prepare_datasets(X, y_list)
    assert len(reg_matrices) == 2
    assert len(targets) == 2


def test_prepare_datasets_regressor_space_mismatch(monkeypatch):
    model = RMSS()

    call_count = {"i": 0}

    def fake_build_lagged_matrix(X, y, *args, **kwargs):
        call_count["i"] += 1
        cols = 2 if call_count["i"] == 1 else 3
        return np.ones((3, cols))

    def fake_fit(data, *args, **kwargs):
        return data

    monkeypatch.setattr(rmss_module, "build_lagged_matrix", fake_build_lagged_matrix)
    monkeypatch.setattr(
        model, "basis_function", type("FB", (), {"fit": staticmethod(fake_fit)})()
    )

    y_list = [np.ones((3, 1)), np.ones((3, 1))]
    with pytest.raises(ValueError, match="regressor space"):
        model._prepare_datasets([None, None], y_list)


def test_prepare_datasets_num_features_mismatch(monkeypatch):
    model = RMSS()
    calls = {"count": 0}

    def fake_num_features(_):
        calls["count"] += 1
        return 1 if calls["count"] == 1 else 2

    monkeypatch.setattr(rmss_module, "num_features", fake_num_features)
    monkeypatch.setattr(
        rmss_module, "build_lagged_matrix", lambda X, y, *a, **k: np.ones((3, 2))
    )
    monkeypatch.setattr(
        model,
        "basis_function",
        type("FB", (), {"fit": staticmethod(lambda data, *a, **k: data)})(),
    )

    with pytest.raises(ValueError, match="input dimension"):
        model._prepare_datasets(
            [np.ones((3, 1)), np.ones((3, 1))], [np.ones((3, 1)), np.ones((3, 1))]
        )


def test_prepare_datasets_single_path_sets_n_inputs_none():
    model = RMSS()
    y = np.ones((5, 1))

    # Avoid num_features(None) errors
    rmss_module_build = rmss_module.build_lagged_matrix
    rmss_module_basis = model.basis_function
    rmss_module.build_lagged_matrix = lambda X, y, *a, **k: np.ones((3, 1))
    model.basis_function = type(
        "FB", (), {"fit": staticmethod(lambda data, *a, **k: data)}
    )()

    reg_matrices, targets = model._prepare_datasets(None, y)
    assert model.n_inputs == 1
    assert len(reg_matrices) == 1
    assert len(targets) == 1

    # restore
    rmss_module.build_lagged_matrix = rmss_module_build
    model.basis_function = rmss_module_basis


def test_run_mss_algorithm_single_feature_breaks():
    model = RMSS(n_terms=1, order_selection=False)
    psi = np.array([[1.0], [0.5], [0.2]])
    y = np.array([[1.0], [0.0], [0.5]])
    err, piv, _, _ = model.run_mss_algorithm(psi, y, process_term_number=2)
    assert err.size == 1
    assert piv.size == 1


def test_run_mss_algorithm_multi_err_tol():
    model = RMSS(err_tol=0.0, n_terms=2, order_selection=False)
    rm1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    rm2 = np.array([[0.5, 0.5], [0.5, -0.5]])
    tgt1 = np.array([[1.0], [0.0]])
    tgt2 = np.array([[0.5], [0.5]])
    err, piv, _, _ = model.run_mss_algorithm([rm1, rm2], [tgt1, tgt2], 2)
    assert err.size == 1
    assert piv.size == 1


def test_estimate_theta_empty_piv():
    model = RMSS()
    out = model._estimate_theta([np.ones((3, 1))], [np.ones((3, 1))], piv=np.array([]))
    assert out.shape == (0, 1)


def test_information_criterion_truncate_n_info_values():
    model = RMSS(n_info_values=5)
    rm = np.ones((4, 2))
    tgt = np.ones((4, 1))
    info = model.information_criterion(rm, tgt)
    assert info.shape == (2,)


def test_information_criterion_single_non_apress():
    model = RMSS(info_criteria="aic", n_info_values=1)
    rm = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    tgt = np.array([[1.0], [0.0], [1.0]])
    info = model.information_criterion(rm, tgt)
    assert info.shape == (1,)


def test_information_criterion_multi_non_apress():
    model = RMSS(info_criteria="aic", n_info_values=1)
    rm1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    rm2 = np.array([[0.5, 0.5], [0.5, -0.5]])
    tgt1 = np.array([[1.0], [0.0]])
    tgt2 = np.array([[0.5], [0.5]])
    info = model.information_criterion([rm1, rm2], [tgt1, tgt2])
    assert info.shape == (1,)


def test_fit_y_none_raises():
    model = RMSS()
    with pytest.raises(ValueError, match="y cannot be None"):
        model.fit(X=X_train, y=None)


def test_fit_model_length_non_apress():
    model = RMSS(
        order_selection=True, info_criteria="aic", n_terms=None, n_info_values=1
    )
    model.fit(X=X_train, y=y_train)
    assert model.n_terms >= 1


def test_fit_with_non_polynomial_basis():
    from sysidentpy.basis_function import Fourier

    model = RMSS(order_selection=False, n_terms=1, basis_function=Fourier(degree=1))
    model.fit(X=X_train, y=y_train)
    assert model.final_model.shape[0] == 1
