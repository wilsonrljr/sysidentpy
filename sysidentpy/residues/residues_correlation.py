# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
#           Luan Pascoal da Costa Andrade <luan_pascoal13@hotmail.com>
#           Samuel Carlos Pessoa Oliveira <samuelcpoliveira@gmail.com>
#           Samir Angelo Milani Martins <martins@ufsj.edu.br>
# License: BSD 3 clause

import numpy as np

from sysidentpy._lib._array_api import (
    get_namespace,
    _is_numpy_namespace,
    _copy,
    _vstack,
    _to_numpy,
)


def compute_residues_autocorrelation(y, yhat):
    """Compute the autocorrelation of the residues.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        True values.
    yhat : array-like of shape (n_samples,)
        Predicted values.

    Returns
    -------
    e_acf : ndarray of shape (n_samples,)
        Autocorrelation of the residues.
    upper_bound : float
        Upper bound for the confidence interval.
    lower_bound : float
        Lower bound for the confidence interval.
    """
    xp = get_namespace(y, yhat)
    e = calculate_residues(y, yhat)
    unnormalized_e_acf = get_unnormalized_e_acf(e)
    half_of_symmetry_autocorr = int(unnormalized_e_acf.shape[0] / 2)

    e_acf = (
        unnormalized_e_acf[half_of_symmetry_autocorr:]
        / unnormalized_e_acf[half_of_symmetry_autocorr]
    )

    upper_bound = 1.96 / float(xp.sqrt(xp.asarray(float(unnormalized_e_acf.shape[0]))))
    lower_bound = upper_bound * (-1)
    return e_acf, upper_bound, lower_bound


def calculate_residues(y, yhat):
    """Calculate the residues (errors) between true and predicted values.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        True values.
    yhat : array-like of shape (n_samples,)
        Predicted values.

    Returns
    -------
    residues : ndarray of shape (n_samples,)
        Residues (errors) between true and predicted values.
    """
    xp = get_namespace(y, yhat)
    return xp.reshape(y - yhat, (-1,))


def get_unnormalized_e_acf(e):
    """Compute the unnormalized autocorrelation function of the residues.

    Parameters
    ----------
    e : array-like of shape (n_samples,)
        Residues (errors).

    Returns
    -------
    unnormalized_e_acf : ndarray of shape (2*n_samples-1,)
        Unnormalized autocorrelation function of the residues.
    """
    xp = get_namespace(e)
    if _is_numpy_namespace(xp):
        return np.correlate(e, e, mode="full")

    # np.correlate is not in the Array API standard.
    # Convert to numpy, compute, convert back.
    e_np = _to_numpy(e)
    result_np = np.correlate(e_np, e_np, mode="full")
    return xp.asarray(result_np)


def compute_cross_correlation(y, yhat, arr):
    """Compute the cross-correlation between the residues and another array.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        True values.
    yhat : array-like of shape (n_samples,)
        Predicted values.
    arr : array-like of shape (n_samples,)
        Another array to compute the cross-correlation with.

    Returns
    -------
    ccf : ndarray of shape (n_samples,)
        Cross-correlation function.
    upper_bound : float
        Upper bound for the confidence interval.
    lower_bound : float
        Lower bound for the confidence interval.
    """
    e = calculate_residues(y, yhat)
    n = len(e) * 2 - 1
    ccf, upper_bound, lower_bound = _input_ccf(e, arr, n)
    return ccf, upper_bound, lower_bound


def _input_ccf(e, a, n):
    """Compute the cross-correlation function and confidence bounds.

    Parameters
    ----------
    e : array-like of shape (n_samples,)
        Residues (errors).
    a : array-like of shape (n_samples,)
        Another array to compute the cross-correlation with.
    n : int
        Length of the cross-correlation function.

    Returns
    -------
    ccf : ndarray of shape (n_samples,)
        Cross-correlation function.
    upper_bound : float
        Upper bound for the confidence interval.
    lower_bound : float
        Lower bound for the confidence interval.
    """
    xp = get_namespace(e, a)
    ccf = _normalized_correlation(a, e)
    upper_bound = 1.96 / float(xp.sqrt(xp.asarray(float(n))))
    lower_bound = upper_bound * (-1)
    return ccf, upper_bound, lower_bound


def _normalized_correlation(a, b):
    r"""Compute the normalized correlation between two signals.

    Parameters
    ----------
    a : array-like of shape (n_samples,)
        First signal.
    b : array-like of shape (n_samples,)
        Second signal.

    Returns
    -------
    ruy : ndarray of shape (n_samples,)
        The normalized cross-correlation between the two signals.

    Notes
    -----
    The normalized cross-correlation is computed as:

    $$
    r_{uy}[i] = \frac{\sum_{j=0}^{N-i-1} (a[j] - \bar{a})(b[j+i] - \bar{b})}
        {\sqrt{\sum_{j=0}^{N-i-1} (a[j] - \bar{a})^2} \sqrt{\sum_{j=0}^{N-i-1}
        (b[j+i] - \bar{b})^2}}
    $$

    where $\bar{a}$ and $\bar{b}$ are the means of $a$ and $b$, respectively.

    """
    xp = get_namespace(a, b)
    y = xp.reshape(a - xp.mean(a), (-1,))
    u = xp.reshape(b - xp.mean(b), (-1,))
    t = int(a.shape[0] / 2)
    ruy = xp.zeros((t,), dtype=xp.float64)
    ruy[0] = float(xp.sum(y * u)) / (
        float(xp.sqrt(xp.sum(y**2))) * float(xp.sqrt(xp.sum(u**2)))
    )

    for i in range(1, t):
        y = xp.reshape(a - xp.mean(a[:i]), (-1,))
        u = xp.reshape(b - xp.mean(b[i:]), (-1,))
        ruy[i] = float(xp.sum(y[:-i] * u[i:])) / (
            float(xp.sqrt(xp.sum(y[:-i] ** 2))) * float(xp.sqrt(xp.sum(u[i:] ** 2)))
        )

    return ruy
