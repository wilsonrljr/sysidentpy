# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
#           Luan Pascoal da Costa Andrade <luan_pascoal13@hotmail.com>
#           Samuel Carlos Pessoa Oliveira <samuelcpoliveira@gmail.com>
#           Samir Angelo Milani Martins <martins@ufsj.edu.br>
# License: BSD 3 clause

import numpy as np


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
    e = calculate_residues(y, yhat)
    unnormalized_e_acf = get_unnormalized_e_acf(e)
    half_of_symmetry_autocorr = int(np.floor(unnormalized_e_acf.size / 2))

    e_acf = (
        unnormalized_e_acf[half_of_symmetry_autocorr:]
        / unnormalized_e_acf[half_of_symmetry_autocorr]
    )

    upper_bound = 1.96 / np.sqrt(len(unnormalized_e_acf))
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
    return (y - yhat).flatten()


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
    return np.correlate(e, e, mode="full")


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
    ccf = _normalized_correlation(a, e)
    upper_bound = 1.96 / np.sqrt(n)
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
    y = (a - np.mean(a)).flatten()
    u = (b - np.mean(b)).flatten()
    t = int(np.floor(len(a) / 2))
    ruy = np.array(np.zeros(t))
    ruy[0] = np.sum(y * u) / (np.sqrt(np.sum(y**2)) * np.sqrt(np.sum(u**2)))

    for i in range(1, t):
        y = (a - np.mean(a[:i])).flatten()
        u = (b - np.mean(b[i:])).flatten()
        ruy[i] = np.sum(y[:-i] * u[i:]) / (
            np.sqrt(np.sum(y[:-i] ** 2)) * np.sqrt(np.sum(u[i:] ** 2))
        )

    return ruy
