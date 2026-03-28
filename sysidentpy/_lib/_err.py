"""Helpers shared by error reduction ratio implementations."""

import numpy as np

from ._array_api import _asarray, _zeros, device as _device, get_namespace


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
        return _zeros(xp, 0, dtype=tmp_psi.dtype, target_device=_device(tmp_psi))

    y_block = tmp_y[start_idx:, :]
    numerators = psi_block.T @ y_block
    denominators = xp.sum(psi_block * psi_block, axis=0)
    denominators = denominators + alpha
    eps_value = _asarray(
        np.finfo(np.float64).eps,
        xp=xp,
        dtype=denominators.dtype,
        target_device=_device(denominators),
    )
    denominators = xp.where(denominators == 0, eps_value, denominators)

    eps_value = _asarray(
        float(eps),
        xp=xp,
        dtype=denominators.dtype,
        target_device=_device(denominators),
    )
    return (xp.reshape(numerators, (-1,)) ** 2 / (denominators * squared_y)) + eps_value
