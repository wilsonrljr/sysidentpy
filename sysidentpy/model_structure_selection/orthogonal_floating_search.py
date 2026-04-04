"""Orthogonal floating search algorithms (OSF, OIF, OOS/O2S).

The algorithms implemented here follow the Orthogonal Floating Search
framework described in the attached OFSA manuscript. They adapt
well-known floating feature selection strategies to the NARX model
structure selection problem by combining: (i) orthogonal projections of
candidate regressors and (ii) the classical Error Reduction Ratio (ERR)
criterion. All classes are drop-in compatible with the existing SysIdentPy
API (estimators, basis functions, equation formatter, etc.).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
from itertools import combinations

import numpy as np

from .._lib._array_api import get_namespace, _is_numpy_namespace
from ..basis_function import Fourier, Polynomial
from ..parameter_estimation.estimators import RecursiveLeastSquares
from .ofr_base import Estimators, OFRBase


class _OrthogonalFloatingBase(OFRBase):
    """Shared helpers for the Orthogonal Floating Search family."""

    def __init__(
        self,
        *,
        ylag: Union[int, List[int]] = 2,
        xlag: Union[int, List[int]] = 2,
        elag: Union[int, List[int]] = 2,
        order_selection: bool = True,
        info_criteria: str = "aic",
        n_terms: Optional[int] = None,
        n_info_values: int = 15,
        estimator: Estimators = RecursiveLeastSquares(),
        basis_function: Union[Polynomial, Fourier] = Polynomial(),
        model_type: str = "NARMAX",
        eps: float = np.finfo(np.float64).eps,
        alpha: float = 0.0,
        err_tol: Optional[float] = None,
        apress_lambda: float = 1.0,
    ):
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

    def _subset_err(
        self,
        psi: np.ndarray,
        target: np.ndarray,
        subset: List[int],
        squared_y: float,
    ) -> Tuple[float, np.ndarray]:
        """Return ERR score and per-term ERRs for the given subset."""
        xp = get_namespace(psi, target)
        if not subset:
            return 0.0, xp.zeros(0, dtype=psi.dtype)

        psi_sel = psi[:, subset]
        if psi_sel.size == 0:
            return 0.0, xp.zeros(0, dtype=psi.dtype)

        q, _ = xp.linalg.qr(psi_sel)
        # For "reduced" mode, q has shape (n, min(n, k)); Array API qr
        # returns full by default, so slice to the reduced shape.
        k = psi_sel.shape[1]
        q = q[:, :k]
        g = q.T @ target

        err_vals = xp.zeros(len(subset), dtype=psi.dtype)
        mapped = g.shape[0]
        if _is_numpy_namespace(xp):
            err_vals[:mapped] = xp.reshape(g, (-1,)) ** 2 / squared_y
        else:
            vals = xp.reshape(g, (-1,)) ** 2 / squared_y
            err_vals = xp.concat([vals, err_vals[mapped:]])
        score = float(xp.sum(err_vals))
        return score, err_vals

    def _compute_squared_y(self, target: np.ndarray) -> float:
        """Compute ||y||^2 with numerical floor to avoid division by zero."""
        xp = get_namespace(target)
        squared_y = float(xp.sum(target * target))
        return squared_y if squared_y > self.eps else float(self.eps)

    def _best_addition(
        self,
        psi: np.ndarray,
        target: np.ndarray,
        subset: List[int],
        available: List[int],
        squared_y: float,
    ) -> Tuple[Optional[int], float]:
        """Pick the most significant term (Definition 1)."""
        best_idx: Optional[int] = None
        best_score = float("-inf")
        for idx in available:
            score, _ = self._subset_err(psi, target, subset + [idx], squared_y)
            if score > best_score or (
                abs(score - best_score) < 1e-9 * max(abs(score), 1.0)
                and (best_idx is None or idx < best_idx)
            ):
                best_score = score
                best_idx = idx
        return best_idx, best_score

    def _best_removal(
        self,
        psi: np.ndarray,
        target: np.ndarray,
        subset: List[int],
        squared_y: float,
    ) -> Tuple[int, float]:
        """Pick the least significant term (Definition 2)."""
        best_idx = subset[0]
        best_score = float("-inf")
        for idx in subset:
            remaining = [t for t in subset if t != idx]
            score, _ = self._subset_err(psi, target, remaining, squared_y)
            if score > best_score or (
                abs(score - best_score) < 1e-9 * max(abs(score), 1.0) and idx < best_idx
            ):
                best_score = score
                best_idx = idx
        return best_idx, best_score

    def _most_significant_terms(
        self,
        psi: np.ndarray,
        target: np.ndarray,
        subset: List[int],
        available: List[int],
        count: int,
        squared_y: float,
    ) -> List[int]:
        """Select the most significant ``count``-subset (Definition 3)."""
        k = min(count, len(available))
        if k <= 0:
            return []

        best_score = float("-inf")
        best_combo: Optional[Tuple[int, ...]] = None

        for combo in combinations(available, k):
            score, _ = self._subset_err(psi, target, subset + list(combo), squared_y)
            if score > best_score or (
                abs(score - best_score) < 1e-9 * max(abs(score), 1.0)
                and (best_combo is None or combo < best_combo)
            ):
                best_score = score
                best_combo = combo

        return list(best_combo) if best_combo is not None else []

    def _least_significant_terms(
        self,
        psi: np.ndarray,
        target: np.ndarray,
        subset: List[int],
        count: int,
        squared_y: float,
    ) -> List[int]:
        """Remove the least significant ``count``-subset (Definition 4)."""
        k = min(count, len(subset))
        if k <= 0:
            return []

        best_score = float("-inf")
        best_combo: Optional[Tuple[int, ...]] = None

        for combo in combinations(subset, k):
            candidate_subset = [t for t in subset if t not in combo]
            score, _ = self._subset_err(psi, target, candidate_subset, squared_y)
            if score > best_score or (
                abs(score - best_score) < 1e-9 * max(abs(score), 1.0)
                and (best_combo is None or combo < best_combo)
            ):
                best_score = score
                best_combo = combo

        return list(best_combo) if best_combo is not None else []

    def _select_most_significant_subset(
        self,
        psi: np.ndarray,
        target: np.ndarray,
        base_subset: List[int],
        available: List[int],
        count: int,
        squared_y: float,
    ) -> List[int]:
        """Floating forward search to pick ``count`` terms (OSF-style).

        Constrains additions to build upon ``base_subset`` and backtracks only
        terms added in this routine, never those in ``base_subset``.
        """
        if count <= 0 or not available:
            return []

        xp = get_namespace(psi, target)
        _, base_score_arr = self._subset_err(psi, target, base_subset, squared_y)
        base_score = float(xp.sum(base_score_arr))

        best_by_size: Dict[int, Tuple[float, Tuple[int, ...]]] = {0: (base_score, ())}
        selected: List[int] = []
        current_score = base_score
        last_added: Optional[int] = None

        def backtrack(
            selected_terms: List[int],
            score: float,
            last_added_term: Optional[int],
        ) -> Tuple[List[int], float]:
            flag_first_removal = 1
            while (
                len(base_subset) + len(selected_terms) > 2 and len(selected_terms) > 0
            ):
                full_subset = base_subset + selected_terms
                ls_idx, ls_score = self._best_removal(
                    psi, target, full_subset, squared_y
                )
                # Do not remove any term that belongs to the fixed base subset.
                if ls_idx in base_subset:
                    break

                prev_best_score = best_by_size.get(
                    len(selected_terms) - 1, (float("-inf"), ())
                )[0]
                if (flag_first_removal == 1 and ls_idx == last_added_term) or (
                    ls_score <= prev_best_score
                ):
                    break

                selected_terms = [t for t in selected_terms if t != ls_idx]
                score = ls_score
                if ls_score > prev_best_score:
                    best_by_size[len(selected_terms)] = (
                        ls_score,
                        tuple(selected_terms),
                    )

                flag_first_removal = 0

            return selected_terms, score

        while len(selected) < count and len(available) > 0:
            ms_idx, _ = self._best_addition(
                psi, target, base_subset + selected, available, squared_y
            )
            if ms_idx is None:
                break

            candidate_selected = selected + [ms_idx]
            candidate_score, _ = self._subset_err(
                psi, target, base_subset + candidate_selected, squared_y
            )

            stored_score, stored_subset = best_by_size.get(
                len(candidate_selected), (float("-inf"), ())
            )

            if candidate_score > stored_score:
                selected = candidate_selected
                current_score = candidate_score
                best_by_size[len(selected)] = (current_score, tuple(selected))
                last_added = ms_idx
            else:
                selected = list(stored_subset)
                current_score = stored_score
                last_added = ms_idx

            available = [a for a in available if a not in selected]
            selected, current_score = backtrack(selected, current_score, last_added)

        return selected

    def _select_least_significant_subset(
        self,
        psi: np.ndarray,
        target: np.ndarray,
        base_subset: List[int],
        count: int,
        squared_y: float,
    ) -> List[int]:
        """Sequential backward floating-style removal of ``count`` terms.

        Evaluates removals that maximize the criterion. Backtracking avoids
        undoing the first removal when it is the most recent change.
        """
        if count <= 0 or len(base_subset) == 0:
            return []

        base_score, _ = self._subset_err(psi, target, base_subset, squared_y)
        best_by_size: Dict[int, Tuple[float, Tuple[int, ...]]] = {
            len(base_subset): (base_score, tuple(base_subset))
        }
        removed: List[int] = []
        current_score = base_score
        last_removed: Optional[int] = None

        while len(removed) < count and (len(base_subset) - len(removed)) > 0:
            working_subset = [t for t in base_subset if t not in removed]
            ls_idx, _ = self._best_removal(psi, target, working_subset, squared_y)
            candidate_removed = removed + [ls_idx]
            candidate_subset = [t for t in base_subset if t not in candidate_removed]
            candidate_score, _ = self._subset_err(
                psi, target, candidate_subset, squared_y
            )

            stored_score, stored_removed = best_by_size.get(
                len(candidate_subset), (float("-inf"), ())
            )

            if candidate_score > stored_score:
                removed = candidate_removed
                current_score = candidate_score
                best_by_size[len(candidate_subset)] = (
                    current_score,
                    tuple(removed),
                )
                last_removed = ls_idx
            else:
                removed = list(stored_removed)
                current_score = stored_score
                last_removed = ls_idx

            # Backtrack: avoid immediately re-removing the last removed term.
            flag_first_removal = 1
            while len(base_subset) - len(removed) > 2:
                working_subset = [t for t in base_subset if t not in removed]
                next_idx, next_score = self._best_removal(
                    psi, target, working_subset, squared_y
                )
                prev_best_score = best_by_size.get(
                    len(working_subset) - 1, (float("-inf"), ())
                )[0]
                if (flag_first_removal == 1 and next_idx == last_removed) or (
                    next_score <= prev_best_score
                ):
                    break

                removed.append(next_idx)
                current_score = next_score
                best_by_size[len(working_subset) - 1] = (
                    current_score,
                    tuple(removed),
                )
                flag_first_removal = 0

            if len(removed) >= count:
                break

        return removed

    def _backtrack(
        self,
        psi: np.ndarray,
        target: np.ndarray,
        subset: List[int],
        current_score: float,
        best_by_size: Dict[int, Tuple[float, Tuple[int, ...]]],
        squared_y: float,
        last_added: Optional[int],
    ) -> Tuple[List[int], float]:
        """Adaptive backward step used by OSF/OIF."""
        flag_first_removal = 1
        while len(subset) > 2:
            ls_idx, ls_score = self._best_removal(psi, target, subset, squared_y)
            prev_best_score = best_by_size.get(len(subset) - 1, (float("-inf"), ()))[0]
            if (flag_first_removal == 1 and ls_idx == last_added) or (
                ls_score <= prev_best_score
            ):
                break

            subset = [t for t in subset if t != ls_idx]
            current_score = ls_score
            if ls_score > prev_best_score:
                best_by_size[len(subset)] = (ls_score, tuple(subset))

            flag_first_removal = 0

        return subset, current_score

    def _floating_forward_search(
        self,
        psi: np.ndarray,
        target: np.ndarray,
        process_term_number: int,
        squared_y: float,
        swap_callback=None,
    ) -> List[int]:
        """Shared floating forward search used by OSF and OIF."""
        total_terms = psi.shape[1]
        all_indices = list(range(total_terms))

        best_by_size: Dict[int, Tuple[float, Tuple[int, ...]]] = {0: (0.0, ())}
        subset: List[int] = []
        current_score = 0.0
        last_added: Optional[int] = None

        while len(subset) < process_term_number and len(subset) < total_terms:
            available = [idx for idx in all_indices if idx not in subset]
            if not available:
                break

            ms_idx, _ = self._best_addition(psi, target, subset, available, squared_y)
            if ms_idx is None:
                break

            candidate_subset = subset + [ms_idx]
            candidate_score, _ = self._subset_err(
                psi, target, candidate_subset, squared_y
            )

            stored_score, stored_subset = best_by_size.get(
                len(candidate_subset), (float("-inf"), ())
            )

            if candidate_score > stored_score:
                subset = candidate_subset
                current_score = candidate_score
                best_by_size[len(subset)] = (current_score, tuple(subset))
                last_added = ms_idx
            else:
                subset = list(stored_subset)
                current_score = stored_score
                last_added = ms_idx

            subset, current_score = self._backtrack(
                psi,
                target,
                subset,
                current_score,
                best_by_size,
                squared_y,
                last_added,
            )

            if swap_callback is not None:
                subset, current_score, swap_added = swap_callback(
                    psi,
                    target,
                    subset,
                    current_score,
                    best_by_size,
                    all_indices,
                    squared_y,
                )
                if swap_added is not None:
                    subset, current_score = self._backtrack(
                        psi,
                        target,
                        subset,
                        current_score,
                        best_by_size,
                        squared_y,
                        swap_added,
                    )

        return subset


class OSF(_OrthogonalFloatingBase):
    """Orthogonal Sequential Floating search (Section 3.2 of the OFSA paper).

    Iteratively adds regressors using ERR and backtracks to drop weak terms.
    Keeps the same constructor surface as other SysIdentPy MSS classes.
    """

    def run_mss_algorithm(
        self, psi: np.ndarray, y: np.ndarray, process_term_number: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run OSF and return (err_vals, piv, psi_selected, target).

        Parameters
        ----------
        psi : np.ndarray
            Candidate regressor matrix with shape (n_samples, n_terms).
        y : np.ndarray
            Output signal aligned with ``psi``.
        process_term_number : int
            Maximum number of terms to select.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            ``(err_vals, piv, psi_selected, target)`` matching the MSS API.
        """
        target = self._default_estimation_target(y)
        squared_y = self._compute_squared_y(target)
        subset = self._floating_forward_search(
            psi, target, process_term_number, squared_y
        )

        _, err_vals = self._subset_err(psi, target, subset, squared_y)
        piv = np.array(subset, dtype=int)
        psi_selected = psi[:, piv] if len(piv) else psi[:, :0]
        return err_vals, piv, psi_selected, target


class OIF(OSF):
    """Orthogonal Improved Floating search (Section 3.3 of the OFSA paper).

    Extends OSF with a swap step that replaces weak terms with better ones
    when the ERR score increases.
    """

    def run_mss_algorithm(
        self, psi: np.ndarray, y: np.ndarray, process_term_number: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run OIF and return (err_vals, piv, psi_selected, target).

        Parameters
        ----------
        psi : np.ndarray
            Candidate regressor matrix with shape (n_samples, n_terms).
        y : np.ndarray
            Output signal aligned with ``psi``.
        process_term_number : int
            Maximum number of terms to select.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            ``(err_vals, piv, psi_selected, target)`` matching the MSS API.
        """
        target = self._default_estimation_target(y)
        squared_y = self._compute_squared_y(target)
        subset = self._floating_forward_search(
            psi,
            target,
            process_term_number,
            squared_y,
            swap_callback=self._swap_step,
        )

        _, err_vals = self._subset_err(psi, target, subset, squared_y)
        piv = np.array(subset, dtype=int)
        psi_selected = psi[:, piv] if len(piv) else psi[:, :0]
        return err_vals, piv, psi_selected, target

    def _swap_step(
        self,
        psi: np.ndarray,
        target: np.ndarray,
        subset: List[int],
        current_score: float,
        best_by_size: Dict[int, Tuple[float, Tuple[int, ...]]],
        all_indices: List[int],
        squared_y: float,
    ) -> Tuple[List[int], float, Optional[int]]:
        best_subset = subset.copy()
        best_score = current_score
        best_added: Optional[int] = None

        for idx in subset:
            reduced = [t for t in subset if t != idx]
            available = [i for i in all_indices if i not in reduced]
            ms_idx, _ = self._best_addition(psi, target, reduced, available, squared_y)
            if ms_idx is None or ms_idx in reduced:
                continue

            candidate_subset = reduced + [ms_idx]
            candidate_score, _ = self._subset_err(
                psi, target, candidate_subset, squared_y
            )

            if candidate_score > best_score:
                best_score = candidate_score
                best_subset = candidate_subset
                best_added = ms_idx

        if best_score > current_score:
            best_by_size[len(best_subset)] = (best_score, tuple(best_subset))
            return best_subset, best_score, best_added

        return subset, current_score, None


class OOS(_OrthogonalFloatingBase):
    """Orthogonal Oscillating Search (paper Section 3.4).

    This class is named ``OOS`` (Orthogonal Oscillating Search) to avoid
    the caret notation ``O^2S`` in code. It corresponds to the method
    described as O2S in the paper.
    """

    def __init__(
        self,
        *,
        ylag: Union[int, List[int]] = 2,
        xlag: Union[int, List[int]] = 2,
        elag: Union[int, List[int]] = 2,
        order_selection: bool = True,
        info_criteria: str = "aic",
        n_terms: Optional[int] = None,
        n_info_values: int = 15,
        estimator: Estimators = RecursiveLeastSquares(),
        basis_function: Union[Polynomial, Fourier] = Polynomial(),
        model_type: str = "NARMAX",
        eps: float = np.finfo(np.float64).eps,
        alpha: float = 0.0,
        err_tol: Optional[float] = None,
        apress_lambda: float = 1.0,
        max_search_depth: Optional[int] = None,
    ):
        self.max_search_depth = max_search_depth
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
        self._validate_oos_params()

    def _validate_oos_params(self) -> None:
        if self.max_search_depth is None:
            return
        if not isinstance(self.max_search_depth, int) or self.max_search_depth < 1:
            msg = (
                "max_search_depth must be a positive int or None; "
                f"got {self.max_search_depth!r} "
                f"(type={type(self.max_search_depth).__name__})"
            )
            raise ValueError(msg)

    def _resolve_search_depth(self, process_term_number: int, total_terms: int) -> int:
        """Choose search depth per OFSA guideline (25% of the smaller side)."""
        if self.max_search_depth is not None:
            return self.max_search_depth

        smaller_side = max(
            1, min(process_term_number, max(0, total_terms - process_term_number))
        )
        depth = int(0.25 * smaller_side)
        return max(1, depth)

    def _down_swing(
        self,
        psi: np.ndarray,
        target: np.ndarray,
        subset: List[int],
        current_score: float,
        depth: int,
        squared_y: float,
        all_indices: List[int],
        all_indices_set: set,
    ) -> Tuple[List[int], float, bool, bool]:
        """Perform a down swing (remove then add) returning updated state."""
        if len(subset) < depth:
            return subset, current_score, False, True

        working_subset = subset.copy()
        to_remove = self._select_least_significant_subset(
            psi, target, working_subset, depth, squared_y
        )

        if len(to_remove) != depth:
            return subset, current_score, False, True

        working_subset = [t for t in working_subset if t not in to_remove]
        available_set = all_indices_set.difference(working_subset)
        if len(available_set) < depth:
            return subset, current_score, False, True

        available_down = [idx for idx in all_indices if idx in available_set]
        ms_terms = self._select_most_significant_subset(
            psi,
            target,
            working_subset,
            available_down,
            depth,
            squared_y,
        )

        if len(ms_terms) != depth:
            return subset, current_score, False, True

        working_subset = working_subset + ms_terms
        down_score, _ = self._subset_err(psi, target, working_subset, squared_y)

        if down_score > current_score:
            return working_subset, down_score, True, False

        return subset, current_score, False, True

    def _up_swing(
        self,
        psi: np.ndarray,
        target: np.ndarray,
        subset: List[int],
        current_score: float,
        depth: int,
        squared_y: float,
        total_terms: int,
        all_indices: List[int],
        all_indices_set: set,
    ) -> Tuple[List[int], float, bool, bool]:
        """Perform an up swing (add then remove) returning updated state."""
        if len(subset) + depth > total_terms:
            return subset, current_score, False, True

        available_set = all_indices_set.difference(subset)
        if len(available_set) < depth:
            return subset, current_score, False, True

        available_up = [idx for idx in all_indices if idx in available_set]
        ms_terms_up = self._select_most_significant_subset(
            psi,
            target,
            subset,
            available_up,
            depth,
            squared_y,
        )

        if len(ms_terms_up) != depth:
            return subset, current_score, False, True

        working_subset = subset + ms_terms_up
        to_remove_up = self._select_least_significant_subset(
            psi, target, working_subset, depth, squared_y
        )

        if len(to_remove_up) != depth:
            return subset, current_score, False, True

        working_subset = [t for t in working_subset if t not in to_remove_up]
        up_score, _ = self._subset_err(psi, target, working_subset, squared_y)

        if up_score > current_score:
            return working_subset, up_score, True, False

        return subset, current_score, False, True

    def run_mss_algorithm(
        self, psi: np.ndarray, y: np.ndarray, process_term_number: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run oscillating search and return (err_vals, piv, psi_selected, target).

        Parameters
        ----------
        psi : np.ndarray
            Candidate regressor matrix with shape (n_samples, n_terms).
        y : np.ndarray
            Output signal aligned with ``psi``.
        process_term_number : int
            Maximum number of terms to select.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            ``(err_vals, piv, psi_selected, target)`` matching the MSS API.
        """
        target = self._default_estimation_target(y)
        squared_y = self._compute_squared_y(target)
        total_terms = psi.shape[1]
        all_indices = list(range(total_terms))
        all_indices_set = set(all_indices)

        # Resolve depth once we know xi and n, as suggested by the paper.
        max_depth = self._resolve_search_depth(process_term_number, total_terms)

        # Initial model: greedy inclusion of ``process_term_number`` terms.
        subset: List[int] = []
        available_set = all_indices_set.copy()
        for _ in range(min(process_term_number, total_terms)):
            available_sorted = [idx for idx in all_indices if idx in available_set]
            ms_idx, _ = self._best_addition(
                psi, target, subset, available_sorted, squared_y
            )
            if ms_idx is None:
                break
            subset.append(ms_idx)
            available_set.discard(ms_idx)

        current_score, _ = self._subset_err(psi, target, subset, squared_y)

        depth = 1
        failed_down = False
        failed_up = False

        while depth <= max_depth and len(subset) > 0:
            improvement = False

            subset, current_score, down_improved, failed_down = self._down_swing(
                psi,
                target,
                subset,
                current_score,
                depth,
                squared_y,
                all_indices,
                all_indices_set,
            )
            improvement = improvement or down_improved

            if failed_down and failed_up:
                depth += 1
                failed_down = False
                failed_up = False
                if depth > max_depth:
                    break

            subset, current_score, up_improved, failed_up = self._up_swing(
                psi,
                target,
                subset,
                current_score,
                depth,
                squared_y,
                total_terms,
                all_indices,
                all_indices_set,
            )
            improvement = improvement or up_improved

            if improvement:
                depth = 1
                failed_down = False
                failed_up = False
            elif failed_down and failed_up:
                depth += 1
                failed_down = False
                failed_up = False

        _, err_vals = self._subset_err(psi, target, subset, squared_y)
        piv = np.array(subset, dtype=int)
        psi_selected = psi[:, piv] if len(piv) else psi[:, :0]
        return err_vals, piv, psi_selected, target


# Alias matching the notation O²S used in the paper.
O2S = OOS

__all__ = ["O2S", "OIF", "OOS", "OSF"]
