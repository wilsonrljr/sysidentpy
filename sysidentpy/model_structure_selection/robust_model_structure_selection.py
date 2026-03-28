"""Robust Model Structure Selection (RMSS).

This module implements the RMSS algorithm described in the paper attached in
``RMSS.md``. The method selects model terms using an overall mean absolute
error (OMAE) criterion computed over resampled sub-datasets (leave-one-out by
default). It follows the same interface conventions as other model structure
selection classes (e.g., :class:`~sysidentpy.model_structure_selection.FROLS`),
reusing estimators, basis functions and prediction utilities already provided
by SysIdentPy.

Key points
----------
- Supports all parameter estimators and basis functions available to OFR-based
  classes.
- Uses leave-one-out resampling to score candidate regressors with OMAE (or
  alternative error measures inspired by the paper).
- Keeps output attributes (``final_model``, ``theta``, ``pivv``) compatible
  with equation formatter utilities.

References
----------
- Gu, Y., & Wei, H.-L. "A Robust Model Structure Selection Method for Small
  Sample Size and Multiple Datasets Problems."
"""

from __future__ import annotations

import copy
import warnings
from itertools import repeat
from typing import List, Literal, Optional, Tuple, Union

import numpy as np

from .._lib._array_api import (
    get_namespace,
    _is_numpy_namespace,
    _copy,
    _require_numpy_namespace,
)
from ..basis_function import Fourier, Polynomial
from ..parameter_estimation.estimators import RecursiveLeastSquares
from ..utils.check_arrays import num_features
from ..utils.information_matrix import build_lagged_matrix
from .ofr_base import Estimators, OFRBase, apress, get_min_info_value

# Type aliases for RMSS-specific parameters
ResamplingStrategy = Literal["loo", "bootstrap"]
ErrorMeasure = Literal["mae", "mse", "phi3", "rmse_ratio"]
ModelType = Literal["NARMAX", "NAR", "NFIR"]

# Valid parameter sets for validation
_VALID_RESAMPLING_STRATEGIES: frozenset[str] = frozenset({"loo", "bootstrap"})
_VALID_ERROR_MEASURES: frozenset[str] = frozenset({"mae", "mse", "phi3", "rmse_ratio"})
_DEPRECATED_ERROR_MEASURES: dict[str, str] = {"smape": "phi3"}


class RMSS(OFRBase):
    r"""Robust Model Structure Selection.

    The RMSS algorithm ranks candidate regressors using an overall error metric
    computed over resampled sub-datasets (leave-one-out, as suggested in the
    paper for small-sample problems). At each step it selects the regressor with
    the smallest aggregated error, orthogonalizes the remaining candidates, and
    repeats until the desired number of terms is reached.

    Parameters
    ----------
    ylag : int or list, default=2
        Maximum output lag.
    xlag : int or list, default=2
        Maximum input lag.
    elag : int or list, default=2
        Maximum residue lag (used when estimator requires it).
    order_selection : bool, default=True
        Whether to use information criteria to choose model size.
    info_criteria : {'aic','aicc','bic','fpe','lilc','apress'}, default='apress'
        Information criterion when ``order_selection`` is enabled.
    n_terms : int, optional
        Number of terms to select. Required when ``order_selection`` is False.
    n_info_values : int, default=15
        Maximum number of terms evaluated by the information criterion.
    estimator : Estimators, optional
        Parameter estimator. Defaults to ``RecursiveLeastSquares()`` when not provided.
    basis_function : Polynomial or Fourier, default=Polynomial()
        Basis function generator.
    model_type : {'NARMAX','NAR','NFIR'}, default='NARMAX'
        Model type.
    eps : float, default=np.finfo(np.float64).eps
        Numerical stability constant.
    alpha : float, default=0
        Regularization parameter (used when estimator is RidgeRegression).
    err_tol : float, optional
        Cumulative ERR/OMAE threshold to stop early.
    resampling : {'loo','bootstrap'}, default='loo'
        Resampling strategy. ``'loo'`` performs leave-one-out as proposed in the
        paper. ``'bootstrap'`` draws ``n_subsets`` bootstrap samples (with
        replacement) of size ``subset_size``.
    error_measure : {'mae','mse','phi3','rmse_ratio'}, default='mae'
        Aggregated error used to rank candidates. ``'mae'`` matches the OMAE in
        the paper. ``'phi3'`` matches the normalized MAE ratio of eq. (19).
        ``'smape'`` is kept as a backward-compatible alias for ``'phi3'``.
    average_theta : bool, default=True
        If True, estimate parameters on every sub-dataset and average the
        resulting coefficients. If False, uses the estimator once on the full
        data (aligned with OFRBase behaviour).
    apress_lambda : float, default=1.0
        Lambda factor used in APRESS (eq. 9). Only used when
        ``info_criteria='apress'``.
    n_subsets : int, optional
        Number of subsets to draw when ``resampling='bootstrap'``. Defaults to
        ``n_samples`` (one subset per leave-one-out equivalent) when not set.
    subset_size : int, optional
        Subset size when ``resampling='bootstrap'``. Defaults to ``n_samples - 1``
        to mimic the sensitivity study in the paper.
    random_state : int, optional
        Seed for bootstrap resampling.
    multi_resampling : bool, default=False
        When multiple datasets are provided, apply the chosen resampling
        strategy to each dataset before scoring candidates (keeps parity with
        the small-sample discussion in the RMSS paper).

    Notes
    -----
    - The implementation follows the same prediction and formatting interfaces
        as other SysIdentPy MSS classes to remain drop-in compatible with
        utilities such as ``equation_formatter``.
    - Setting ``average_theta=False`` skips the per-sub-dataset averaging in
        eq. (28) of the paper; keep it ``True`` for the canonical RMSS behaviour.
    """

    def __init__(
        self,
        *,
        ylag: Union[int, List[int]] = 2,
        xlag: Union[int, List[int]] = 2,
        elag: Union[int, List[int]] = 2,
        order_selection: bool = True,
        info_criteria: str = "apress",
        n_terms: Optional[int] = None,
        n_info_values: int = 15,
        estimator: Optional[Estimators] = None,
        basis_function: Union[Polynomial, Fourier] = Polynomial(),
        model_type: ModelType = "NARMAX",
        eps: float = np.finfo(np.float64).eps,
        alpha: float = 0.0,
        err_tol: Optional[float] = None,
        resampling: ResamplingStrategy = "loo",
        error_measure: ErrorMeasure = "mae",
        average_theta: bool = True,
        apress_lambda: float = 1.0,
        n_subsets: Optional[int] = None,
        subset_size: Optional[int] = None,
        random_state: Optional[int] = None,
        multi_resampling: bool = False,
    ):
        self.resampling = resampling
        self.error_measure = error_measure
        self.average_theta = average_theta
        self.n_subsets = n_subsets
        self.subset_size = subset_size
        self.random_state = random_state
        self.multi_resampling = multi_resampling
        self.omae_history: List[np.ndarray] = []
        self._reg_matrices: List[np.ndarray] = []
        self._targets: List[np.ndarray] = []

        estimator = RecursiveLeastSquares() if estimator is None else estimator

        super().__init__(
            ylag=ylag,
            xlag=xlag,
            elag=elag,
            order_selection=order_selection,
            info_criteria=info_criteria,
            n_terms=n_terms,
            n_info_values=n_info_values,
            estimator=estimator,
            basis_function=basis_function,
            model_type=model_type,
            eps=eps,
            alpha=alpha,
            err_tol=err_tol,
            apress_lambda=apress_lambda,
        )

        self._validate_rmss_params()

    def _validate_rmss_params(self) -> None:
        """Validate RMSS-specific parameters.

        Raises
        ------
        ValueError
            If resampling strategy or error measure is invalid, or if
            bootstrap parameters are out of range.
        TypeError
            If boolean or integer parameters have wrong types.
        """
        self._validate_resampling_strategy()
        self._validate_error_measure()
        self._validate_boolean_params()
        self._validate_bootstrap_params()

    def _validate_resampling_strategy(self) -> None:
        """Validate resampling strategy parameter."""
        if self.resampling not in _VALID_RESAMPLING_STRATEGIES:
            valid_options = ", ".join(sorted(_VALID_RESAMPLING_STRATEGIES))
            raise ValueError(
                f"Unsupported resampling strategy: '{self.resampling}'. "
                f"Valid options are: {valid_options}."
            )

    def _validate_error_measure(self) -> None:
        """Validate error measure parameter, handling deprecated aliases."""
        if self.error_measure in _DEPRECATED_ERROR_MEASURES:
            new_measure = _DEPRECATED_ERROR_MEASURES[self.error_measure]
            warnings.warn(
                f"error_measure='{self.error_measure}' is deprecated; "
                f"use '{new_measure}' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            self.error_measure = new_measure

        if self.error_measure not in _VALID_ERROR_MEASURES:
            valid_options = ", ".join(sorted(_VALID_ERROR_MEASURES))
            raise ValueError(
                f"error_measure must be one of: {valid_options}. "
                f"Got '{self.error_measure}'."
            )

    def _validate_boolean_params(self) -> None:
        """Validate boolean parameters."""
        bool_params = {
            "average_theta": self.average_theta,
            "multi_resampling": self.multi_resampling,
        }
        for name, value in bool_params.items():
            if not isinstance(value, bool):
                raise TypeError(
                    f"{name} must be a boolean value. Got {type(value).__name__}."
                )

    def _validate_bootstrap_params(self) -> None:
        """Validate bootstrap-specific parameters."""
        if self.resampling != "bootstrap":
            return

        if self.n_subsets is not None and self.n_subsets < 1:
            raise ValueError(
                "n_subsets must be a positive integer when provided. "
                f"Got {self.n_subsets}."
            )
        if self.subset_size is not None and self.subset_size < 1:
            raise ValueError(
                "subset_size must be a positive integer when provided. "
                f"Got {self.subset_size}."
            )
        if self.random_state is not None and not isinstance(self.random_state, int):
            raise TypeError(
                "random_state must be an integer when provided. "
                f"Got {type(self.random_state).__name__}."
            )

    def _create_sub_datasets(
        self, reg_matrix: np.ndarray, target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate leave-one-out or bootstrap views for a single dataset."""
        if reg_matrix.shape[0] < 2:
            raise ValueError("Need at least two samples to perform RMSS resampling.")

        if self.resampling == "loo":
            n_samples, n_features = reg_matrix.shape
            psi_views = np.empty(
                (n_samples, n_samples - 1, n_features), dtype=np.float64
            )
            y_views = np.empty((n_samples, n_samples - 1), dtype=np.float64)

            for idx in range(n_samples):
                mask = np.ones(n_samples, dtype=bool)
                mask[idx] = False
                psi_views[idx] = reg_matrix[mask]
                y_views[idx] = target[mask, 0]

            return psi_views, y_views

        if self.resampling == "bootstrap":
            rng = np.random.default_rng(self.random_state)
            n_samples, n_features = reg_matrix.shape
            k_subsets = self.n_subsets or n_samples
            subset_size = self.subset_size or max(1, n_samples - 1)

            psi_views = np.empty((k_subsets, subset_size, n_features), dtype=np.float64)
            y_views = np.empty((k_subsets, subset_size), dtype=np.float64)

            for k in range(k_subsets):
                idx = rng.choice(n_samples, size=subset_size, replace=True)
                psi_views[k] = reg_matrix[idx]
                y_views[k] = target[idx, 0]

            return psi_views, y_views

        raise ValueError(f"Unsupported resampling strategy: {self.resampling}")

    def _compute_error_metric(
        self,
        errors: np.ndarray,
        y_ref: np.ndarray,
        preds: np.ndarray,
        axis: int,
    ) -> np.ndarray:
        """Compute the selected error metric along the specified axis.

        Parameters
        ----------
        errors : np.ndarray
            Prediction errors (y - y_hat).
        y_ref : np.ndarray
            Reference output values (ground truth).
        preds : np.ndarray
            Model predictions.
        axis : int
            Axis along which to compute the metric.

        Returns
        -------
        np.ndarray
            Computed error metric values.
        """
        xp = get_namespace(errors)
        _eps_val = xp.asarray(self.eps, dtype=errors.dtype)

        if self.error_measure == "mae":
            return xp.mean(xp.abs(errors), axis=axis)

        if self.error_measure == "mse":
            return xp.mean(errors ** 2, axis=axis)

        if self.error_measure == "phi3":
            numerator = xp.sum(xp.abs(errors), axis=axis)
            denom = xp.sum(xp.abs(y_ref), axis=axis) + xp.sum(
                xp.abs(preds), axis=axis
            )
            denom = xp.where(xp.abs(denom) < self.eps, _eps_val, denom)
            return numerator / denom

        # rmse_ratio
        rmse = xp.sqrt(xp.mean(errors ** 2, axis=axis))
        y_rmse = xp.sqrt(xp.mean(y_ref ** 2, axis=axis))
        pred_rmse = xp.sqrt(xp.mean(preds ** 2, axis=axis))
        denom = y_rmse + pred_rmse
        denom = xp.where(xp.abs(denom) < self.eps, _eps_val, denom)
        return rmse / denom

    def _overall_error(self, psi_views: np.ndarray, y_views: np.ndarray) -> np.ndarray:
        """Compute aggregated error for each candidate across sub-datasets.

        Parameters
        ----------
        psi_views : np.ndarray
            Resampled regressor views with shape (K, N', M) where K is the
            number of sub-datasets, N' is the subset size, and M is the
            number of candidate regressors.
        y_views : np.ndarray
            Resampled target views with shape (K, N').

        Returns
        -------
        np.ndarray
            Overall error for each candidate regressor (shape: M,).
        """
        xp = get_namespace(psi_views, y_views)
        _eps_val = xp.asarray(self.eps, dtype=psi_views.dtype)
        # einsum("knm,kn->km") -> sum(psi * y[..., None], axis=-2)
        y_expanded_for_mul = xp.reshape(y_views, y_views.shape + (1,))
        numerators = xp.sum(psi_views * y_expanded_for_mul, axis=-2)
        # einsum("knm,knm->km") -> sum(psi * psi, axis=-2)
        denominators = xp.sum(psi_views * psi_views, axis=-2)
        denominators = xp.where(
            xp.abs(denominators) < self.eps, _eps_val, denominators
        )
        alphas = numerators / denominators

        preds = psi_views * alphas[:, None, :]
        errors = y_views[:, :, None] - preds

        # Expand y_views for metric computation: (K, N') -> (K, N', 1)
        y_expanded = y_views[:, :, None]
        metric = self._compute_error_metric(errors, y_expanded, preds, axis=1)

        return xp.mean(metric, axis=0)

    def _overall_error_multi(
        self, psi_list: List[np.ndarray], y_list: List[np.ndarray]
    ) -> np.ndarray:
        """Compute aggregated error across multiple datasets.

        Parameters
        ----------
        psi_list : List[np.ndarray]
            List of regressor matrices or resampled views per dataset.
        y_list : List[np.ndarray]
            List of target arrays or resampled target views per dataset.

        Returns
        -------
        np.ndarray
            Mean error across all datasets for each candidate.
        """
        per_dataset = []
        for psi_k, y_k in zip(psi_list, y_list, strict=True):
            if psi_k.ndim == 3:
                # Resampled views (K_k, N', M) - delegate to _overall_error
                metric = self._overall_error(psi_k, y_k)
            else:
                metric = self._compute_2d_error(psi_k, y_k)
            per_dataset.append(metric)

        return np.mean(np.stack(per_dataset, axis=0), axis=0)

    def _compute_2d_error(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute error metric for a 2D regressor matrix.

        Parameters
        ----------
        psi : np.ndarray
            Regressor matrix with shape (N, M).
        y : np.ndarray
            Target array with shape (N,) or (N, 1).

        Returns
        -------
        np.ndarray
            Error metric for each candidate (shape: M,).
        """
        xp = get_namespace(psi, y)
        _eps_val = xp.asarray(self.eps, dtype=psi.dtype)
        y_vec = xp.reshape(y, (-1,))
        numerators = psi.T @ y_vec
        # einsum("ij,ij->j") -> sum(psi * psi, axis=0)
        denominators = xp.sum(psi * psi, axis=0)
        denominators = xp.where(
            xp.abs(denominators) < self.eps, _eps_val, denominators
        )
        alphas = numerators / denominators

        preds = psi * alphas[None, :]
        errors = y_vec[:, None] - preds

        return self._compute_error_metric(errors, y_vec[:, None], preds, axis=0)

    def _orthogonalize_remaining_views(
        self, psi_views: np.ndarray, selected_q: np.ndarray
    ) -> np.ndarray:
        """Orthogonalize remaining candidates against the selected vector.

        Applies Gram-Schmidt orthogonalization to remove the component of
        each candidate regressor that lies along the selected vector.

        Parameters
        ----------
        psi_views : np.ndarray
            Resampled regressor views with shape (K, N', M).
        selected_q : np.ndarray
            Selected regressor vector with shape (K, N').

        Returns
        -------
        np.ndarray
            Orthogonalized regressor views with same shape as input.
        """
        xp = get_namespace(psi_views, selected_q)
        _eps_val = xp.asarray(self.eps, dtype=psi_views.dtype)
        # einsum("kn,kn->k") -> sum(q * q, axis=-1)
        denom = xp.sum(selected_q * selected_q, axis=-1)
        denom = xp.where(xp.abs(denom) < self.eps, _eps_val, denom)

        # einsum("kn,knm->km") -> sum(q[..., None] * psi, axis=-2)
        projection = xp.sum(
            xp.reshape(selected_q, selected_q.shape + (1,)) * psi_views, axis=-2
        )
        coeff = projection / denom[:, None]
        return psi_views - selected_q[:, :, None] * coeff[:, None, :]

    # Alias for backward compatibility
    _orthogonalize_remaining = _orthogonalize_remaining_views

    def _orthogonalize_matrix(
        self, psi_matrix: np.ndarray, selected_q: np.ndarray
    ) -> np.ndarray:
        """Orthogonalize a 2D regressor matrix against the selected vector.

        Parameters
        ----------
        psi_matrix : np.ndarray
            Regressor matrix with shape (N, M).
        selected_q : np.ndarray
            Selected regressor vector with shape (N,).

        Returns
        -------
        np.ndarray
            Orthogonalized regressor matrix with same shape as input.
        """
        xp = get_namespace(psi_matrix, selected_q)
        denom = float(xp.sum(selected_q * selected_q))
        denom = self.eps if abs(denom) < self.eps else denom
        projection = psi_matrix.T @ selected_q
        coeff = projection / denom
        # outer product: q[:, None] @ coeff[None, :]
        return psi_matrix - xp.reshape(selected_q, (-1, 1)) @ xp.reshape(coeff, (1, -1))

    def _orthogonalize_remaining_multi(
        self, psi_list: List[np.ndarray], selected_q_list: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Orthogonalize remaining candidates for multiple 2D datasets.

        Parameters
        ----------
        psi_list : List[np.ndarray]
            List of regressor matrices, each with shape (N_k, M).
        selected_q_list : List[np.ndarray]
            List of selected regressor vectors, each with shape (N_k,).

        Returns
        -------
        List[np.ndarray]
            List of orthogonalized regressor matrices.
        """
        return [
            self._orthogonalize_matrix(psi_k, q_k)
            for psi_k, q_k in zip(psi_list, selected_q_list, strict=True)
        ]

    def _prepare_datasets(
        self,
        X: Optional[Union[np.ndarray, List[Optional[np.ndarray]]]],
        y: Union[np.ndarray, List[np.ndarray]],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Build regressor matrices and targets for single or multiple datasets.

        Parameters
        ----------
        X : np.ndarray, List[np.ndarray], or None
            Input data. Can be a single array, a list of arrays for multiple
            datasets, or None for NAR models.
        y : np.ndarray or List[np.ndarray]
            Output data. Can be a single array or a list for multiple datasets.

        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray]]
            A tuple containing:
            - reg_matrices: List of regressor matrices.
            - targets: List of target arrays.

        Raises
        ------
        ValueError
            If X and y lists have different lengths, or if datasets have
            inconsistent input dimensions or regressor spaces.
        """
        if isinstance(y, (list, tuple)):
            y_list = list(y)
            if X is None or isinstance(X, np.ndarray):
                X_list = list(repeat(X, len(y_list)))
            else:
                X_list = list(X)
            if len(X_list) != len(y_list):
                raise ValueError("X and y lists must have the same length.")

            reg_matrices: List[np.ndarray] = []
            targets: List[np.ndarray] = []
            self.n_inputs = None

            for Xi, yi in zip(X_list, y_list, strict=True):
                lagged_data = build_lagged_matrix(
                    Xi, yi, self.xlag, self.ylag, self.model_type
                )
                reg_matrix = self.basis_function.fit(
                    lagged_data,
                    self.max_lag,
                    self.ylag,
                    self.xlag,
                    self.model_type,
                    predefined_regressors=None,
                )

                target = self._default_estimation_target(yi)
                reg_matrices.append(reg_matrix)
                targets.append(target)

                if self.n_inputs is None:
                    self.n_inputs = num_features(Xi) if Xi is not None else 1
                elif self.n_inputs != (num_features(Xi) if Xi is not None else 1):
                    raise ValueError(
                        "All datasets must share the same input dimension."
                    )

            n_features = {rm.shape[1] for rm in reg_matrices}
            if len(n_features) != 1:
                raise ValueError("All datasets must produce the same regressor space.")

            return reg_matrices, targets

        # Single dataset path
        lagged_data = build_lagged_matrix(X, y, self.xlag, self.ylag, self.model_type)
        reg_matrix = self.basis_function.fit(
            lagged_data,
            self.max_lag,
            self.ylag,
            self.xlag,
            self.model_type,
            predefined_regressors=None,
        )

        if X is not None:
            self.n_inputs = num_features(X)
        else:
            self.n_inputs = 1

        target = self._default_estimation_target(y)
        return [reg_matrix], [target]

    def run_mss_algorithm(
        self,
        psi: Union[np.ndarray, List[np.ndarray]],
        y: Union[np.ndarray, List[np.ndarray]],
        process_term_number: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Perform RMSS selection over single or multiple datasets.

        This method implements the core RMSS algorithm, selecting regressors
        one at a time based on their aggregated error across resampled
        sub-datasets.

        Parameters
        ----------
        psi : np.ndarray or List[np.ndarray]
            Regressor matrix or list of matrices for multiple datasets.
        y : np.ndarray or List[np.ndarray]
            Target array or list of arrays for multiple datasets.
        process_term_number : int
            Maximum number of terms to select.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            - err: Array of error values for each selected term.
            - piv: Array of selected regressor indices.
            - psi_selected: Regressor matrix with only selected columns.
            - target: Target array used for estimation.
        """
        self.omae_history = []

        reg_matrices, targets = self._normalize_inputs(psi, y)

        if len(reg_matrices) == 1:
            return self._run_single_dataset(
                reg_matrices[0], targets[0], process_term_number
            )

        return self._run_multi_dataset(reg_matrices, targets, process_term_number)

    def _normalize_inputs(
        self,
        psi: Union[np.ndarray, List[np.ndarray]],
        y: Union[np.ndarray, List[np.ndarray]],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Normalize inputs to lists for consistent processing."""
        if not isinstance(psi, list):
            return [psi], [y]
        targets = y if isinstance(y, list) else [y]
        return psi, targets

    def _run_single_dataset(
        self,
        psi: np.ndarray,
        target: np.ndarray,
        process_term_number: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run RMSS selection for a single dataset."""
        psi_views, y_views = self._create_sub_datasets(psi, target)

        available_indices = np.arange(psi.shape[1])
        selected_indices: List[int] = []
        err_trace: List[float] = []

        current_views = psi_views
        max_terms = min(process_term_number, psi.shape[1])

        for _ in range(max_terms):
            omae = self._overall_error(current_views, y_views)
            self.omae_history.append(omae)

            best_local_idx = int(np.argmin(omae))
            selected_indices.append(int(available_indices[best_local_idx]))
            err_trace.append(float(omae[best_local_idx]))

            if self._should_stop_selection(err_trace):
                break

            selected_q = current_views[:, :, best_local_idx]
            available_indices = np.delete(available_indices, best_local_idx)
            current_views = np.delete(current_views, best_local_idx, axis=2)

            if current_views.shape[2] == 0:
                break

            current_views = self._orthogonalize_remaining(current_views, selected_q)

        return self._build_result(selected_indices, err_trace, psi, target)

    def _run_multi_dataset(
        self,
        reg_matrices: List[np.ndarray],
        targets: List[np.ndarray],
        process_term_number: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run RMSS selection for multiple datasets."""
        psi_list, target_list = self._prepare_multi_dataset_views(reg_matrices, targets)

        available_indices = np.arange(psi_list[0].shape[-1])
        selected_indices: List[int] = []
        err_trace: List[float] = []

        current_views = psi_list
        max_terms = min(process_term_number, psi_list[0].shape[-1])

        for _ in range(max_terms):
            omae = self._overall_error_multi(current_views, target_list)
            self.omae_history.append(omae)

            best_local_idx = int(np.argmin(omae))
            selected_indices.append(int(available_indices[best_local_idx]))
            err_trace.append(float(omae[best_local_idx]))

            if self._should_stop_selection(err_trace):
                break

            updated_views, selected_q_list = self._extract_and_remove_selected(
                current_views, best_local_idx
            )
            available_indices = np.delete(available_indices, best_local_idx)

            if updated_views[0].shape[-1] == 0:
                break

            current_views = self._orthogonalize_multi_views(
                updated_views, selected_q_list
            )

        return self._build_result(
            selected_indices, err_trace, reg_matrices[0], targets[0]
        )

    def _should_stop_selection(self, err_trace: List[float]) -> bool:
        """Check if selection should stop based on error tolerance."""
        return self.err_tol is not None and np.sum(err_trace) >= self.err_tol

    def _prepare_multi_dataset_views(
        self,
        reg_matrices: List[np.ndarray],
        targets: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Prepare views for multi-dataset processing."""
        psi_list: List[np.ndarray] = []
        target_list: List[np.ndarray] = []

        for rm, tgt in zip(reg_matrices, targets, strict=True):
            if self.multi_resampling:
                views, yv = self._create_sub_datasets(rm, tgt)
                psi_list.append(views)
                target_list.append(yv)
            else:
                psi_list.append(rm.copy())
                target_list.append(tgt)

        return psi_list, target_list

    def _extract_and_remove_selected(
        self,
        views: List[np.ndarray],
        best_idx: int,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Extract selected regressors and remove them from views."""
        selected_q_list: List[np.ndarray] = []
        updated_views: List[np.ndarray] = []

        for view in views:
            if view.ndim == 3:
                selected_q_list.append(view[:, :, best_idx])
                updated_views.append(np.delete(view, best_idx, axis=2))
            else:
                selected_q_list.append(view[:, best_idx])
                updated_views.append(np.delete(view, best_idx, axis=1))

        return updated_views, selected_q_list

    def _orthogonalize_multi_views(
        self,
        views: List[np.ndarray],
        selected_q_list: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Orthogonalize views against selected regressors."""
        new_views: List[np.ndarray] = []

        for view, q in zip(views, selected_q_list, strict=True):
            if view.ndim == 3:
                new_views.append(self._orthogonalize_remaining_views(view, q))
            else:
                new_views.append(self._orthogonalize_matrix(view, q))

        return new_views

    def _build_result(
        self,
        selected_indices: List[int],
        err_trace: List[float],
        psi: np.ndarray,
        target: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build the result tuple from selection data."""
        piv = np.array(selected_indices, dtype=int)
        err = np.array(err_trace, dtype=float)
        psi_selected = psi[:, piv] if piv.size else psi[:, :0]
        return err, piv, psi_selected, target

    def _estimate_theta(
        self,
        reg_matrices: List[np.ndarray],
        targets: List[np.ndarray],
        piv: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Estimate model parameters using the selected regressors.

        For a single dataset with ``average_theta=True``, parameters are
        estimated on each resampled sub-dataset and averaged (eq. 28 in the
        RMSS paper). For multiple datasets, parameters are averaged across
        all datasets.

        Parameters
        ----------
        reg_matrices : List[np.ndarray]
            List of regressor matrices.
        targets : List[np.ndarray]
            List of target arrays.
        piv : np.ndarray, optional
            Indices of selected regressors. If None, uses ``self.pivv``.

        Returns
        -------
        np.ndarray
            Estimated parameter vector with shape (n_terms, 1).
        """
        piv = self.pivv if piv is None else piv
        if piv is None or len(piv) == 0:
            return np.empty((0, 1))

        def _select_columns(mat: np.ndarray) -> np.ndarray:
            return mat[:, piv]

        if len(reg_matrices) == 1:
            psi = _select_columns(reg_matrices[0])
            target = targets[0]
            if not self.average_theta:
                warnings.warn(
                    "average_theta=False skips the per-subset averaging in eq.(28) "
                    "of the RMSS paper; use True to match the reference method.",
                    UserWarning,
                    stacklevel=2,
                )
                theta = self.estimator.optimize(psi, target)
            else:
                psi_views, y_views = self._create_sub_datasets(psi, target)
                thetas = []
                for k in range(psi_views.shape[0]):
                    est_copy = copy.deepcopy(self.estimator)
                    theta_k = est_copy.optimize(psi_views[k], y_views[k].reshape(-1, 1))
                    thetas.append(theta_k.reshape(-1, 1))
                theta = np.mean(np.stack(thetas, axis=2), axis=2)

            if getattr(self.estimator, "unbiased", False) is True:
                theta = self.estimator.unbiased_estimator(
                    psi,
                    target,
                    theta,
                    self.elag,
                    self.max_lag,
                    self.estimator,
                    self.basis_function,
                    self.estimator.uiter,
                )
            return theta

        # Multiple datasets: average parameters across datasets (eq. 28)
        if getattr(self.estimator, "unbiased", False) is True:
            warnings.warn(
                "Unbiased correction is not applied when fitting multiple datasets "
                "with RMSS; results may differ from single-dataset unbiased fits.",
                UserWarning,
                stacklevel=2,
            )
        thetas = []
        for reg_matrix, target in zip(reg_matrices, targets, strict=True):
            psi_sel = _select_columns(reg_matrix)
            est_copy = copy.deepcopy(self.estimator)
            theta_k = est_copy.optimize(psi_sel, target)
            thetas.append(theta_k.reshape(-1, 1))

        theta = np.mean(np.stack(thetas, axis=2), axis=2)
        return theta

    def information_criterion(
        self,
        x: Union[np.ndarray, List[np.ndarray]],
        y: Union[np.ndarray, List[np.ndarray]],
    ) -> np.ndarray:
        """Compute information criteria using robust parameter estimation.

        Evaluates models of increasing complexity (1 to ``n_info_values``
        terms) and returns the information criterion value for each.

        Parameters
        ----------
        x : np.ndarray or List[np.ndarray]
            Regressor matrix or list of matrices.
        y : np.ndarray or List[np.ndarray]
            Target array or list of arrays.

        Returns
        -------
        np.ndarray
            Information criterion values for each model size.
        """
        reg_matrices = x if isinstance(x, list) else [x]
        targets = y if isinstance(y, list) else [y]

        n_info_values = self.n_info_values or reg_matrices[0].shape[1]
        n_info_values = min(n_info_values, reg_matrices[0].shape[1])
        self.n_info_values = n_info_values

        output_vector = np.zeros(n_info_values)
        output_vector[:] = np.nan

        for i in range(n_info_values):
            n_theta = i + 1
            _, piv, _, _ = self.run_mss_algorithm(reg_matrices, targets, n_theta)

            tmp_theta = self._estimate_theta(reg_matrices, targets, piv)

            if len(reg_matrices) == 1:
                psi_sel = reg_matrices[0][:, piv]
                target_sel = targets[0]
                tmp_yhat = np.dot(psi_sel, tmp_theta)
                tmp_residual = target_sel - tmp_yhat

                if self.info_criteria == "apress":
                    mse = np.mean(np.square(tmp_residual))
                    output_vector[i] = apress(
                        n_theta, target_sel.shape[0], mse, self.apress_lambda
                    )
                else:
                    e_var = np.var(tmp_residual, ddof=1)
                    output_vector[i] = self.info_criteria_function(
                        n_theta, target_sel.shape[0], e_var
                    )
            else:
                per_dataset_vals = []
                for rm, tgt in zip(reg_matrices, targets, strict=True):
                    psi_sel = rm[:, piv]
                    yhat = np.dot(psi_sel, tmp_theta)
                    residual = tgt - yhat

                    if self.info_criteria == "apress":
                        mse = np.mean(np.square(residual))
                        val = apress(n_theta, tgt.shape[0], mse, self.apress_lambda)
                    else:
                        e_var = np.var(residual, ddof=1)
                        val = self.info_criteria_function(n_theta, tgt.shape[0], e_var)
                    per_dataset_vals.append(val)

                output_vector[i] = float(np.mean(per_dataset_vals))

            if i == n_info_values - 1:
                self.pivv = piv

        return output_vector

    def fit(
        self, *, X: Optional[np.ndarray] = None, y: Union[np.ndarray, List[np.ndarray]]
    ) -> "RMSS":
        """Fit the RMSS model to the data.

        Parameters
        ----------
        X : np.ndarray, optional
            Input data with shape (n_samples, n_inputs). Can be None for
            NAR models.
        y : np.ndarray or List[np.ndarray]
            Output data with shape (n_samples, 1). Can be a list for
            fitting with multiple datasets.

        Returns
        -------
        self : RMSS
            The fitted model instance.

        Raises
        ------
        ValueError
            If y is None or if order_selection is False without n_terms.
        """
        if y is None:
            raise ValueError("y cannot be None")

        xp = get_namespace(y) if X is None else get_namespace(X, y)
        _require_numpy_namespace(xp, feature="RMSS", dependency="SciPy")

        self.max_lag = self._get_max_lag()

        reg_matrices, targets = self._prepare_datasets(X, y)
        self._reg_matrices = reg_matrices
        self._targets = targets

        self.regressor_code = self.regressor_space(self.n_inputs)

        if self.order_selection is True:
            self.info_values = self.information_criterion(reg_matrices, targets)

        if self.n_terms is None and self.order_selection is True:
            if self.info_criteria == "apress":
                model_length = int(np.nanargmin(self.info_values)) + 1
            else:
                model_length = get_min_info_value(self.info_values)
            self.n_terms = model_length
        elif self.n_terms is None and self.order_selection is not True:
            raise ValueError(
                "If order_selection is False, you must define n_terms value."
            )
        else:
            model_length = self.n_terms

        mss_result = self.run_mss_algorithm(reg_matrices, targets, model_length)
        self.err, self.pivv, _psi, _estimation_target = self._unpack_mss_output(
            mss_result, targets[0]
        )

        model_length = min(model_length, len(self.pivv))
        self.n_terms = model_length

        tmp_piv = self.pivv[0:model_length]
        repetition = len(reg_matrices[0])
        if isinstance(self.basis_function, Polynomial):
            self.final_model = self.regressor_code[tmp_piv, :].copy()
        else:
            self.regressor_code = np.sort(
                np.tile(self.regressor_code[1:, :], (repetition, 1)),
                axis=0,
            )
            self.final_model = self.regressor_code[tmp_piv, :].copy()

        self.theta = self._estimate_theta(self._reg_matrices, self._targets)
        return self

    def predict(
        self,
        *,
        X: Optional[np.ndarray] = None,
        y: np.ndarray,
        steps_ahead: Optional[int] = None,
        forecast_horizon: Optional[int] = None,
    ) -> np.ndarray:
        """Predict using the fitted RMSS model.

        Parameters
        ----------
        X : np.ndarray, optional
            Input data for prediction. Can be None for NAR models.
        y : np.ndarray
            Output data with initial conditions for prediction.
        steps_ahead : int, optional
            Number of steps ahead for multi-step prediction. If None,
            performs free-run simulation.
        forecast_horizon : int, optional
            Number of samples to forecast beyond the input data.

        Returns
        -------
        np.ndarray
            Predicted output values.
        """
        return super().predict(
            X=X, y=y, steps_ahead=steps_ahead, forecast_horizon=forecast_horizon
        )
