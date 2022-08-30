# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
#           Luan Pascoal da Costa Andrade <luan_pascoal13@hotmail.com>
#           Samuel Carlos Pessoa Oliveira <samuelcpoliveira@gmail.com>
#           Samir Angelo Milani Martins <martins@ufsj.edu.br>
# License: BSD 3 clause


import numpy as np


def compute_residues_autocorrelation(y, yhat):
    e = calculate_residues(y, yhat)
    unnormalized_e_acf = get_unnormalized_e_acf(e)
    half_of_symmetry_autocorr = int(np.floor(unnormalized_e_acf.size / 2))

    e_acf = (
        unnormalized_e_acf[half_of_symmetry_autocorr:]
        / unnormalized_e_acf[half_of_symmetry_autocorr]
    )

    half_of_symmetry_autocorr = int(np.floor(unnormalized_e_acf.size / 2))
    upper_bound = 1.96 / np.sqrt(len(unnormalized_e_acf))
    lower_bound = upper_bound * (-1)
    return e_acf, upper_bound, lower_bound


def calculate_residues(y, yhat):
    return (y - yhat).flatten()


def get_unnormalized_e_acf(e):
    return np.correlate(e, e, mode="full")


def compute_cross_correlation(y, yhat, arr):
    e = calculate_residues(y, yhat)
    n = len(e) * 2 - 1
    ccf, upper_bound, lower_bound = _input_ccf(e, arr, n)
    return ccf, upper_bound, lower_bound


def _input_ccf(e, a, n):
    ccf = _normalized_correlation(a, e)
    upper_bound = 1.96 / np.sqrt(n)
    lower_bound = upper_bound * (-1)
    return ccf, upper_bound, lower_bound


def _normalized_correlation(a, b):
    """Compute the normalized correlation between two signals.

    Parameters
    ----------
    a : array-like of shape = n_samples.
    b : array-like of shape = n_samples.

    Returns
    -------
    ruy : ndarray of floats:
        The normalized cross correlation between the two signals.

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
