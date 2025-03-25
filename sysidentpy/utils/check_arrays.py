"""Utilities fo data validation."""

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause

from warnings import warn

import numpy as np
import numbers


# copy-pasted/adapted from scikit-learn utils/validation.py
def check_random_state(seed):
    """Turn `seed` into a `np.random.RandomState` instance.

    Parameters
    ----------
    seed : {None, int, `numpy.random.Generator`,
            `numpy.random.RandomState`}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    Returns
    -------
    seed : {`numpy.random.Generator`, `numpy.random.RandomState`}
        Random number generator.

    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.default_rng(seed)
    if isinstance(seed, (np.random.RandomState, np.random.Generator)):
        return seed

    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )


def num_features(x):
    return x.shape[1]


def check_positive_int(value, name):
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be integer and > zero. Got {value}")
    else:
        pass


def check_infinity(x, y):
    """Check that x and y have no NaN or Inf samples.

    If there is any NaN or Inf samples a ValueError is raised.

    Parameters
    ----------
    x : ndarray of floats
        The input data.
    y : ndarray of floats
        The output data.

    """
    if np.isinf(x).any():
        msg_error = (
            "Input contains invalid values (e.g. NaN, Inf) on "
            f"index {np.argwhere(np.isinf(x))}"
        )
        raise ValueError(msg_error)

    if np.isinf(y).any():
        msg_error = (
            "Output contains invalid values (e.g Inf) on "
            f"index {np.argwhere(np.isinf(y))}"
        )
        raise ValueError(msg_error)


def check_nan(x, y):
    """Check that x and y have no NaN or Inf samples.

    If there is any NaN or Inf samples a ValueError is raised.

    Parameters
    ----------
    x : ndarray of floats
        The input data.
    y : ndarray of floats
        The output data.

    """
    if np.isnan(x).any():
        msg_error = (
            "Input contains invalid values (e.g. NaN, Inf) on "
            f"index {np.argwhere(np.isnan(x))}"
        )
        raise ValueError(msg_error)

    if not ~np.isnan(y).any():
        msg_error = (
            "Output contains invalid values (e.g. NaN, Inf) on "
            f"index {np.argwhere(np.isnan(y))}"
        )
        raise ValueError(msg_error)


def check_length(x, y):
    """Check that x and y have the same number of samples.

    If the length of x and y are different a ValueError is raised.

    Parameters
    ----------
    x : ndarray of floats
        The input data.
    y : ndarray of floats
        The output data.

    """
    if x.shape[0] != y.shape[0]:
        msg_error = (
            "Input and output data must have the same number of "
            f"samples. x has dimension {x.shape} and "
            f"y has dimension {y.shape}"
        )
        raise ValueError(msg_error)


def check_dimension(x, y):
    """Check if x and y have only real values.

    If there is any string or object samples a ValueError is raised.

    Parameters
    ----------
    x : ndarray of floats
        The input data.
    y : ndarray of floats
        The output data.

    """
    if x.ndim == 0:
        raise ValueError(
            "Input must be a 2d array, got scalar instead. Reshape your data using"
            " array.reshape(-1, 1)"
        )

    if x.ndim == 1:
        raise ValueError(
            "Input must be a 2d array, got 1d array instead. "
            "Reshape your data using array.reshape(-1, 1)"
        )

    if y.ndim == 0:
        raise ValueError(
            "Output must be a 2d array, got scalar instead. "
            "Reshape your data using array.reshape(-1, 1)"
        )

    if y.ndim == 1:
        raise ValueError(
            "Output must be a 2d array, got 1d array instead. "
            "Reshape your data using array.reshape(-1, 1)"
        )


def check_x_y(x, y):
    """Validate input and output data using some crucial tests.

    Parameters
    ----------
    x : ndarray of floats
        The input data.
    y : ndarray of floats
        The output data.

    """
    check_length(x, y)
    check_dimension(x, y)
    check_infinity(x, y)
    check_nan(x, y)


def check_linear_dependence_rows(psi):
    """Check for linear dependence in the rows of the Psi matrix.

    Parameters
    ----------
    psi : ndarray of floats
        The information matrix of the model.

    Warns
    -----
    UserWarning
        If the Psi matrix has linearly dependent rows.
    """
    if np.linalg.matrix_rank(psi) != psi.shape[1]:
        warn(
            "Psi matrix might have linearly dependent rows."
            "Be careful and check your data",
            stacklevel=2,
        )
