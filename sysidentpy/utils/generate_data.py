"""Utilities for data generation."""

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause

import numpy as np


def get_siso_data(n=5000, colored_noise=False, sigma=0.05, train_percentage=90):
    """Perform the Error Reduction Ration algorithm.

    Parameters
    ----------
    n : int
        The number of samples.
    colored_noise : bool
        Select white noise or colored noise (autoregressive noise).
    sigma : float
        The standard deviation of the random distribution to generate
        the noise.
    train_percentage : int
        The percentage of the data to be used as train data.

    Returns
    -------
    x_train, x_valid : array-like
        The input data to be used in identification and validation,
        respectively.
    y_train, y_valid : array-like
        The output data to be used in identification and validation,
        respectively.

    """
    mu = 0  # mean of the distribution
    nu = np.random.normal(mu, sigma, n).T
    e = np.zeros((n, 1))

    lag = 2
    if colored_noise is True:
        for k in range(lag, len(e)):
            e[k] = 0.8 * nu[k - 1] + nu[k]
    else:
        e = nu

    x = np.random.uniform(-1, 1, n).T
    y = np.zeros((n, 1))
    theta = np.array([[0.2], [0.1], [0.9]])
    lag = 2
    for k in range(lag, len(x)):
        y[k] = (
            theta[0] * y[k - 1]
            + theta[1] * y[k - 1] * x[k - 1]
            + theta[2] * x[k - 2]
            + e[k]
        )

    split_data = int(len(x) * (train_percentage / 100))

    x_train = x[0:split_data].reshape(-1, 1)
    x_valid = x[split_data::].reshape(-1, 1)

    y_train = y[0:split_data].reshape(-1, 1)
    y_valid = y[split_data::].reshape(-1, 1)

    return x_train, x_valid, y_train, y_valid


def get_miso_data(n=5000, colored_noise=False, sigma=0.05, train_percentage=90):
    """Perform the Error Reduction Ration algorithm.

    Parameters
    ----------
    n : int
        The number of samples.
    colored_noise : bool
        Select white noise or colored noise (autoregressive noise).
    sigma : float
        The standard deviation of the random distribution to generate
        the noise.
    train_percentage : int
        The percentage of the data to be used as train data.

    Returns
    -------
    x_train, x_valid : array-like
        The input data to be used in identification and validation,
        respectively.
    y_train, y_valid : array-like
        The output data to be used in identification and validation,
        respectively.

    """
    mu = 0  # mean of the distribution
    nu = np.random.normal(mu, sigma, n).T
    e = np.zeros((n, 1))

    lag = 2
    if colored_noise is True:
        for k in range(lag, len(e)):
            e[k] = 0.8 * nu[k - 1] + nu[k]
    else:
        e = nu

    x1 = np.random.uniform(-1, 1, n).T
    x2 = np.random.uniform(-1, 1, n).T
    y = np.zeros((n, 1))
    theta = np.array([[0.4], [0.1], [0.6], [-0.3]])

    lag = 2
    for k in range(lag, len(e)):
        y[k] = (
            theta[0] * y[k - 1] ** 2
            + theta[1] * y[k - 1] * x1[k - 1]
            + theta[2] * x2[k - 1]
            + theta[3] * x1[k - 1] * x2[k - 2]
            + e[k]
        )

    split_data = int(len(x1) * (train_percentage / 100))
    x1_train = x1[0:split_data].reshape(-1, 1)
    x2_train = x2[0:split_data].reshape(-1, 1)
    x1_valid = x1[split_data::].reshape(-1, 1)
    x2_valid = x2[split_data::].reshape(-1, 1)

    x_train = np.hstack([x1_train, x2_train])
    x_valid = np.hstack([x1_valid, x2_valid])

    y_train = y[0:split_data].reshape(-1, 1)
    y_valid = y[split_data::].reshape(-1, 1)

    return x_train, x_valid, y_train, y_valid
