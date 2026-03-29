"""Common metrics to assess performance on NARX models."""

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
#           Luan Pascoal da Costa Andrade <luan_pascoal13@hotmail.com>
#           Samuel Carlos Pessoa Oliveira <samuelcpoliveira@gmail.com>
#           Samir Angelo Milani Martins <martins@ufsj.edu.br>
# License: BSD 3 clause

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from sysidentpy._lib._array_api import (
    get_namespace,
    _is_numpy_namespace,
    _median,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


__ALL__ = [
    "forecast_error",
    "mean_forecast_error",
    "mean_squared_error",
    "root_mean_squared_error",
    "normalized_root_mean_squared_error",
    "root_relative_squared_error",
    "mean_absolute_error",
    "mean_squared_log_error",
    "median_absolute_error",
    "explained_variance_score",
    "r2_score",
    "symmetric_mean_absolute_percentage_error",
]


def forecast_error(y: NDArray, yhat: NDArray) -> NDArray:
    """Calculate the forecast error in a regression model.

    Parameters
    ----------
    y : array-like of shape = number_of_outputs
        Represent the target values.
    yhat : array-like of shape = number_of_outputs
        Target values predicted by the model.

    Returns
    -------
    loss : ndarray of floats
        The difference between the true target values and the predicted
        or forecast value in regression or any other phenomenon.

    References
    ----------
    - Wikipedia entry on the Forecast error
       https://en.wikipedia.org/wiki/Forecast_error

    Examples
    --------
    >>> y = [3, -0.5, 2, 7]
    >>> yhat = [2.5, 0.0, 2, 8]
    >>> forecast_error(y, yhat)
    [0.5, -0.5, 0, -1]

    """
    return y - yhat


def mean_forecast_error(y: NDArray, yhat: NDArray) -> NDArray:
    """Calculate the mean of forecast error of a regression model.

    Parameters
    ----------
    y : array-like of shape = number_of_outputs
        Represent the target values.
    yhat : array-like of shape = number_of_outputs
        Target values predicted by the model.

    Returns
    -------
    loss : float
        The mean  value of the difference between the true target
        values and the predicted or forecast value in regression
        or any other phenomenon.

    References
    ----------
    - Wikipedia entry on the Forecast error
       https://en.wikipedia.org/wiki/Forecast_error

    Examples
    --------
    >>> y = [3, -0.5, 2, 7]
    >>> yhat = [2.5, 0.0, 2, 8]
    >>> mean_forecast_error(y, yhat)
    -0.25

    """
    xp = get_namespace(y, yhat)
    return float(xp.mean(y - yhat))


def mean_squared_error(y: NDArray, yhat: NDArray) -> NDArray:
    """Calculate the Mean Squared Error.

    Parameters
    ----------
    y : array-like of shape = number_of_outputs
        Represent the target values.
    yhat : array-like of shape = number_of_outputs
        Target values predicted by the model.

    Returns
    -------
    loss : float
        MSE output is non-negative values. Becoming 0.0 means your
        model outputs are exactly matched by true target values.

    References
    ----------
    - Wikipedia entry on the Mean Squared Error
       https://en.wikipedia.org/wiki/Mean_squared_error

    Examples
    --------
    >>> y = [3, -0.5, 2, 7]
    >>> yhat = [2.5, 0.0, 2, 8]
    >>> mean_squared_error(y, yhat)
    0.375

    """
    xp = get_namespace(y, yhat)
    return float(xp.mean((y - yhat) ** 2))


def root_mean_squared_error(y: NDArray, yhat: NDArray) -> NDArray:
    """Calculate the Root Mean Squared Error.

    Parameters
    ----------
    y : array-like of shape = number_of_outputs
        Represent the target values.
    yhat : array-like of shape = number_of_outputs
        Target values predicted by the model.

    Returns
    -------
    loss : float
        RMSE output is non-negative values. Becoming 0.0 means your
        model outputs are exactly matched by true target values.

    References
    ----------
    - Wikipedia entry on the Root Mean Squared Error
       https://en.wikipedia.org/wiki/Root-mean-square_deviation

    Examples
    --------
    >>> y = [3, -0.5, 2, 7]
    >>> yhat = [2.5, 0.0, 2, 8]
    >>> root_mean_squared_error(y, yhat)
    0.612

    """
    return float(mean_squared_error(y, yhat) ** 0.5)


def normalized_root_mean_squared_error(y: NDArray, yhat: NDArray) -> NDArray:
    """Calculate the normalized Root Mean Squared Error.

    Parameters
    ----------
    y : array-like of shape = number_of_outputs
        Represent the target values.
    yhat : array-like of shape = number_of_outputs
        Target values predicted by the model.

    Returns
    -------
    loss : float
        nRMSE output is non-negative values. Becoming 0.0 means your
        model outputs are exactly matched by true target values.

    References
    ----------
    - Wikipedia entry on the normalized Root Mean Squared Error
       https://en.wikipedia.org/wiki/Root-mean-square_deviation

    Examples
    --------
    >>> y = [3, -0.5, 2, 7]
    >>> yhat = [2.5, 0.0, 2, 8]
    >>> normalized_root_mean_squared_error(y, yhat)
    0.081

    """
    xp = get_namespace(y, yhat)
    return float(root_mean_squared_error(y, yhat) / (xp.max(y) - xp.min(y)))


def root_relative_squared_error(y: NDArray, yhat: NDArray) -> NDArray:
    """Calculate the Root Relative Mean Squared Error.

    Parameters
    ----------
    y : array-like of shape = number_of_outputs
        Represent the target values.
    yhat : array-like of shape = number_of_outputs
        Target values predicted by the model.

    Returns
    -------
    loss : float
        RRSE output is non-negative values. Becoming 0.0 means your
        model outputs are exactly matched by true target values.

    Examples
    --------
    >>> y = [3, -0.5, 2, 7]
    >>> yhat = [2.5, 0.0, 2, 8]
    >>> root_relative_mean_squared_error(y, yhat)
    0.206

    """
    xp = get_namespace(y, yhat)
    numerator = xp.sum((yhat - y) ** 2)
    denominator = xp.sum((y - xp.mean(y, axis=0)) ** 2)
    return float(xp.sqrt(numerator / denominator))


def mean_absolute_error(y: NDArray, yhat: NDArray) -> NDArray:
    """Calculate the Mean absolute error.

    Parameters
    ----------
    y : array-like of shape = number_of_outputs
        Represent the target values.
    yhat : array-like of shape = number_of_outputs
        Target values predicted by the model.

    Returns
    -------
    loss : float or ndarray of floats
        MAE output is non-negative values. Becoming 0.0 means your
        model outputs are exactly matched by true target values.

    References
    ----------
    - Wikipedia entry on the Mean absolute error
       https://en.wikipedia.org/wiki/Mean_absolute_error

    Examples
    --------
    >>> y = [3, -0.5, 2, 7]
    >>> yhat = [2.5, 0.0, 2, 8]
    >>> mean_absolute_error(y, yhat)
    0.5

    """
    xp = get_namespace(y, yhat)
    return float(xp.mean(xp.abs(y - yhat)))


def mean_squared_log_error(y: NDArray, yhat: NDArray) -> NDArray:
    """Calculate the Mean Squared Logarithmic Error.

    Parameters
    ----------
    y : array-like of shape = number_of_outputs
        Represent the target values.
    yhat : array-like of shape = number_of_outputs
        Target values predicted by the model.

    Returns
    -------
    loss : float
        MSLE output is non-negative values. Becoming 0.0 means your
        model outputs are exactly matched by true target values.

    Examples
    --------
    >>> y = [3, 5, 2.5, 7]
    >>> yhat = [2.5, 5, 4, 8]
    >>> mean_squared_log_error(y, yhat)
    0.039

    """
    xp = get_namespace(y, yhat)
    return mean_squared_error(xp.log(y + 1), xp.log(yhat + 1))


def median_absolute_error(y: NDArray, yhat: NDArray) -> NDArray:
    """Calculate the Median Absolute Error.

    Parameters
    ----------
    y : array-like of shape = number_of_outputs
        Represent the target values.
    yhat : array-like of shape = number_of_outputs
        Target values predicted by the model.

    Returns
    -------
    loss : float
        MdAE output is non-negative values. Becoming 0.0 means your
        model outputs are exactly matched by true target values.

    References
    ----------
    - Wikipedia entry on the Median absolute deviation
       https://en.wikipedia.org/wiki/Median_absolute_deviation

    Examples
    --------
    >>> y = [3, -0.5, 2, 7]
    >>> yhat = [2.5, 0.0, 2, 8]
    >>> median_absolute_error(y, yhat)
    0.5

    """
    xp = get_namespace(y, yhat)
    return float(_median(xp, xp.abs(y - yhat)))


def explained_variance_score(y: NDArray, yhat: NDArray) -> NDArray:
    """Calculate the Explained Variance Score.

    Parameters
    ----------
    y : array-like of shape = number_of_outputs
        Represent the target values.
    yhat : array-like of shape = number_of_outputs
        Target values predicted by the model.

    Returns
    -------
    loss : float
        EVS output is non-negative values. Becoming 1.0 means your
        model outputs are exactly matched by true target values.
        Lower values means worse results.

    References
    ----------
    - Wikipedia entry on the Explained Variance
       https://en.wikipedia.org/wiki/Explained_variation

    Examples
    --------
    >>> y = [3, -0.5, 2, 7]
    >>> yhat = [2.5, 0.0, 2, 8]
    >>> explained_variance_score(y, yhat)
    0.957

    """
    xp = get_namespace(y, yhat)
    y_diff_avg = xp.mean(y - yhat)
    numerator = xp.mean((y - yhat - y_diff_avg) ** 2)
    y_avg = xp.mean(y)
    denominator = xp.mean((y - y_avg) ** 2)
    nonzero_numerator = numerator != 0
    nonzero_denominator = denominator != 0
    valid_score = nonzero_numerator & nonzero_denominator
    output_scores = xp.ones(y.shape[0], dtype=y.dtype)
    output_scores[valid_score] = 1 - (numerator[valid_score] / denominator[valid_score])
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.0
    return float(xp.mean(output_scores))


def r2_score(y: NDArray, yhat: NDArray) -> NDArray:
    """Calculate the R2 score. Based on sklearn solution.

    Parameters
    ----------
    y : array-like of shape = number_of_outputs
        Represent the target values.
    yhat : array-like of shape = number_of_outputs
        Target values predicted by the model.

    Returns
    -------
    loss : float
        R2 output can be non-negative values or negative value.
        Becoming 1.0 means your model outputs are exactly
        matched by true target values. Lower values means worse results.

    Notes
    -----
    This is not a symmetric function.

    References
    ----------
    - Wikipedia entry on the Coefficient of determination
       https://en.wikipedia.org/wiki/Coefficient_of_determination

    Examples
    --------
    >>> y = [3, -0.5, 2, 7]
    >>> yhat = [2.5, 0.0, 2, 8]
    >>> explained_variance_score(y, yhat)
    0.948

    """
    xp = get_namespace(y, yhat)
    if _is_numpy_namespace(xp):
        numerator = ((y - yhat) ** 2).sum(axis=0, dtype=np.float64)
        denominator = ((y - np.average(y, axis=0)) ** 2).sum(axis=0, dtype=np.float64)
    else:
        numerator = xp.sum((y - yhat) ** 2, axis=0)
        denominator = xp.sum((y - xp.mean(y, axis=0)) ** 2, axis=0)
    nonzero_denominator = denominator != 0
    nonzero_numerator = numerator != 0
    valid_score = nonzero_denominator & nonzero_numerator
    output_scores = xp.ones(y.shape[1], dtype=y.dtype)
    output_scores[valid_score] = 1 - (numerator[valid_score] / denominator[valid_score])
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.0
    return float(xp.mean(output_scores))


def symmetric_mean_absolute_percentage_error(y: NDArray, yhat: NDArray) -> NDArray:
    """Calculate the SMAPE score.

    Parameters
    ----------
    y : array-like of shape = number_of_outputs
        Represent the target values.
    yhat : array-like of shape = number_of_outputs
        Target values predicted by the model.

    Returns
    -------
    loss : float
        SMAPE output is a non-negative value.
        The results are percentages values.

    Notes
    -----
    One supposed problem with SMAPE is that it is not symmetric since
    over-forecasts and under-forecasts are not treated equally.

    References
    ----------
    - Wikipedia entry on the Symmetric mean absolute percentage error
       https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error

    Examples
    --------
    >>> y = [3, -0.5, 2, 7]
    >>> yhat = [2.5, 0.0, 2, 8]
    >>> symmetric_mean_absolute_percentage_error(y, yhat)
    57.87

    """
    xp = get_namespace(y, yhat)
    n = y.shape[0]
    return float(100 / n * xp.sum(2 * xp.abs(yhat - y) / (xp.abs(y) + xp.abs(yhat))))
