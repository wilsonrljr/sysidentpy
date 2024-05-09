"""Utilities fo data validation."""

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause


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


def _num_features(X):
    return X.shape[1]


def _check_positive_int(value, name):
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be integer and > zero. Got {value}")
    else:
        pass


def check_infinity(X, y):
    """Check that X and y have no NaN or Inf samples.

    If there is any NaN or Inf samples a ValueError is raised.

    Parameters
    ----------
    X : ndarray of floats
        The input data.
    y : ndarray of floats
        The output data.

    """
    if np.isinf(X).any():
        msg_error = (
            "Input contains invalid values (e.g. NaN, Inf) on "
            f"index {np.argwhere(np.isinf(X))}"
        )
        raise ValueError(msg_error)

    if np.isinf(y).any():
        msg_error = (
            "Output contains invalid values (e.g Inf) on "
            f"index {np.argwhere(np.isinf(y))}"
        )
        raise ValueError(msg_error)


def check_nan(X, y):
    """Check that X and y have no NaN or Inf samples.

    If there is any NaN or Inf samples a ValueError is raised.

    Parameters
    ----------
    X : ndarray of floats
        The input data.
    y : ndarray of floats
        The output data.

    """
    if np.isnan(X).any():
        msg_error = (
            "Input contains invalid values (e.g. NaN, Inf) on "
            f"index {np.argwhere(np.isnan(X))}"
        )
        raise ValueError(msg_error)

    if not ~np.isnan(y).any():
        msg_error = (
            "Output contains invalid values (e.g. NaN, Inf) on "
            f"index {np.argwhere(np.isnan(y))}"
        )
        raise ValueError(msg_error)


def check_length(X, y):
    """Check that X and y have the same number of samples.

    If the length of X and y are different a ValueError is raised.

    Parameters
    ----------
    X : ndarray of floats
        The input data.
    y : ndarray of floats
        The output data.

    """
    if X.shape[0] != y.shape[0]:
        msg_error = (
            "Input and output data must have the same number of "
            f"samples. X has dimension {X.shape} and "
            f"y has dimension {y.shape}"
        )
        raise ValueError(msg_error)


def check_dimension(X, y):
    """Check if X and y have only real values.

    If there is any string or object samples a ValueError is raised.

    Parameters
    ----------
    X : ndarray of floats
        The input data.
    y : ndarray of floats
        The output data.

    """
    if X.ndim == 0:
        raise ValueError(
            "Input must be a 2d array, got scalar instead. Reshape your data using"
            " array.reshape(-1, 1)"
        )

    if X.ndim == 1:
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


def check_X_y(X, y):
    """Validate input and output data using some crucial tests.

    Parameters
    ----------
    X : ndarray of floats
        The input data.
    y : ndarray of floats
        The output data.

    """
    check_length(X, y)
    check_dimension(X, y)
    check_infinity(X, y)
    check_nan(X, y)
