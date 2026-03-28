"""Helpers shared by error reduction ratio implementations."""

import numpy as np

from ._array_api import get_namespace


def _compute_err_slice(
    tmp_psi: np.ndarray,
    tmp_y: np.ndarray,
    start_idx: int,
    squared_y: float,
    alpha: float,
    eps: float,
) -> np.ndarray:
    """Compute ERR values for remaining regressors using vectorized math."""
    xp = get_namespace(tmp_psi, tmp_y)
    psi_block = tmp_psi[start_idx:, start_idx:]
    if psi_block.size == 0:
        return xp.zeros(0, dtype=tmp_psi.dtype)

    y_block = tmp_y[start_idx:, :]
    numerators = psi_block.T @ y_block
    denominators = xp.sum(psi_block * psi_block, axis=0)
    denominators = denominators + alpha
    eps_value = xp.asarray(np.finfo(np.float64).eps, dtype=denominators.dtype)
    denominators = xp.where(denominators == 0, eps_value, denominators)

    eps_value = xp.asarray(float(eps), dtype=denominators.dtype)
    return (xp.reshape(numerators, (-1,)) ** 2 / (denominators * squared_y)) + eps_value
