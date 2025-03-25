"""Utilities for data generation."""

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause

import numpy as np


def get_siso_data(n=5000, colored_noise=False, sigma=0.05, train_percentage=90):
    r"""Generate synthetic data for Single-Input Single-Output system identification.

    This function simulates input-output data for a SISO system based on a predefined
    nonlinear difference equation. The system output is affected by either white noise
    or colored noise (autoregressive noise) depending on the `colored_noise` flag.

    Parameters
    ----------
    n : int, optional (default=5000)
        Number of samples to generate.
    colored_noise : bool, optional (default=False)
        If True, adds colored (autoregressive) noise to the system; otherwise, white
        noise is used.
    sigma : float, optional (default=0.05)
        Standard deviation of the noise distribution.
    train_percentage : int, optional (default=90)
        Percentage of the dataset allocated for training. The rest is used for
        validation.

    Returns
    -------
    x_train : ndarray
        Input data for system identification (training).
    x_valid : ndarray
        Input data for system validation (testing).
    y_train : ndarray
        Output data corresponding to `x_train`.
    y_valid : ndarray
        Output data corresponding to `x_valid`.

    Notes
    -----
    - The system follows the nonlinear difference equation:

      y[k] = 0.2 * y[k-1] + 0.1 * y[k-1] * x[k-1] + 0.9 * x[k-2] + e[k]

      where `e[k]` is either white or colored noise.

    - The input `x` is uniformly sampled from the range [-1, 1].
    - The dataset is split based on `train_percentage`, ensuring a clear separation
      between training and validation data.

    """
    if train_percentage < 0 or train_percentage > 100:
        raise ValueError("train_percentage must be smaller than 100")

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
    """Generate synthetic data for Multiple-Input Single-Output system identification.

    This function simulates input-output data for a nonlinear MISO system using two
    input signals. The system output is influenced by both inputs and can be affected
    by either white or colored (autoregressive) noise based on the `colored_noise` flag.

    Parameters
    ----------
    n : int, optional (default=5000)
        Number of samples to generate.
    colored_noise : bool, optional (default=False)
        If True, adds colored (autoregressive) noise to the system; otherwise, white
        noise is used.
    sigma : float, optional (default=0.05)
        Standard deviation of the noise distribution.
    train_percentage : int, optional (default=90)
        Percentage of the dataset allocated for training. The remainder is used
        for validation.

    Returns
    -------
    x_train : ndarray
        Input data matrix (features) for system identification (training).
    x_valid : ndarray
        Input data matrix (features) for system validation (testing).
    y_train : ndarray
        Output data corresponding to `x_train`.
    y_valid : ndarray
        Output data corresponding to `x_valid`.

    Notes
    -----
    - The system follows the nonlinear difference equation:

      y[k] = 0.4 * y[k-1]² + 0.1 * y[k-1] * x1[k-1] + 0.6 * x2[k-1]
             - 0.3 * x1[k-1] * x2[k-2] + e[k]

      where `e[k]` is either white or colored noise.

    - The inputs `x1` and `x2` are independently sampled from a uniform distribution in
      the range [-1, 1].
    - The dataset is split into training and validation sets based on `train_percentage`
      , ensuring a clear separation between them.
    - The function returns `x_train` and `x_valid` as stacked arrays, where each row
      represents a sample and each column corresponds to an input variable
      (`x1` or `x2`).

    """
    if train_percentage < 0 or train_percentage > 100:
        raise ValueError("train_percentage must be smaller than 100")

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
