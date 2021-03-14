"""Common metrics to assess performance on NARX models."""

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
#           Luan Pascoal da Costa Andrade <luan_pascoal13@hotmail.com>
#           Samuel Carlos Pessoa Oliveira <samuelcpoliveira@gmail.com>
#           Samir Angelo Milani Martins <martins@ufsj.edu.br>
# License: BSD 3 clause

import numpy as np


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


def forecast_error(y, y_predicted):
    """Calculate the forecast error in a regression model.

    Parameters
    ----------
    y : array-like of shape = number_of_outputs
        Represent the target values.
    y_predicted : array-like of shape = number_of_outputs
        Target values predicted by the model.

    Returns
    -------
    loss : ndarray of floats
        The difference between the true target values and the predicted
        or forecast value in regression or any other phenomenon.

    References
    ----------
    [1] Wikipedia entry on the Forecast error
        https://en.wikipedia.org/wiki/Forecast_error

    Examples
    --------
    >>> y = [3, -0.5, 2, 7]
    >>> y_predicted = [2.5, 0.0, 2, 8]
    >>> forecast_error(y, y_predicted)
    [0.5, -0.5, 0, -1]

    """
    return y - y_predicted


def mean_forecast_error(y, y_predicted):
    """Calculate the mean of forecast error of a regression model.

    Parameters
    ----------
    y : array-like of shape = number_of_outputs
        Represent the target values.
    y_predicted : array-like of shape = number_of_outputs
        Target values predicted by the model.

    Returns
    -------
    loss : float
        The mean  value of the difference between the true target
        values and the predicted or forecast value in regression
        or any other phenomenon.

    References
    ----------
    [1] Wikipedia entry on the Forecast error
        https://en.wikipedia.org/wiki/Forecast_error

    Examples
    --------
    >>> y = [3, -0.5, 2, 7]
    >>> y_predicted = [2.5, 0.0, 2, 8]
    >>> mean_forecast_error(y, y_predicted)
    -0.25

    """
    return np.average(y - y_predicted)


def mean_squared_error(y, y_predicted):
    """Calculate the Mean Squared Error.

    Parameters
    ----------
    y : array-like of shape = number_of_outputs
        Represent the target values.
    y_predicted : array-like of shape = number_of_outputs
        Target values predicted by the model.

    Returns
    -------
    loss : float
        MSE output is non-negative values. Becoming 0.0 means your
        model outputs are exactly matched by true target values.

    References
    ----------
    [1] Wikipedia entry on the Mean Squared Error
        https://en.wikipedia.org/wiki/Mean_squared_error

    Examples
    --------
    >>> y = [3, -0.5, 2, 7]
    >>> y_predicted = [2.5, 0.0, 2, 8]
    >>> mean_squared_error(y, y_predicted)
    0.375

    """
    output_error = np.average((y - y_predicted) ** 2)
    return np.average(output_error)


def root_mean_squared_error(y, y_predicted):
    """Calculate the Root Mean Squared Error.

    Parameters
    ----------
    y : array-like of shape = number_of_outputs
        Represent the target values.
    y_predicted : array-like of shape = number_of_outputs
        Target values predicted by the model.

    Returns
    -------
    loss : float
        RMSE output is non-negative values. Becoming 0.0 means your
        model outputs are exactly matched by true target values.

    References
    ----------
    [1] Wikipedia entry on the Root Mean Squared Error
        https://en.wikipedia.org/wiki/Root-mean-square_deviation

    Examples
    --------
    >>> y = [3, -0.5, 2, 7]
    >>> y_predicted = [2.5, 0.0, 2, 8]
    >>> root_mean_squared_error(y, y_predicted)
    0.612

    """
    return np.sqrt(mean_squared_error(y, y_predicted))


def normalized_root_mean_squared_error(y, y_predicted):
    """Calculate the normalized Root Mean Squared Error.

    Parameters
    ----------
    y : array-like of shape = number_of_outputs
        Represent the target values.
    y_predicted : array-like of shape = number_of_outputs
        Target values predicted by the model.

    Returns
    -------
    loss : float
        nRMSE output is non-negative values. Becoming 0.0 means your
        model outputs are exactly matched by true target values.

    References
    ----------
    [1] Wikipedia entry on the normalized Root Mean Squared Error
        https://en.wikipedia.org/wiki/Root-mean-square_deviation

    Examples
    --------
    >>> y = [3, -0.5, 2, 7]
    >>> y_predicted = [2.5, 0.0, 2, 8]
    >>> normalized_root_mean_squared_error(y, y_predicted)
    0.081

    """
    return root_mean_squared_error(y, y_predicted) / (y.max() - y.min())


def root_relative_squared_error(y, y_predicted):
    """Calculate the Root Relative Mean Squared Error.

    Parameters
    ----------
    y : array-like of shape = number_of_outputs
        Represent the target values.
    y_predicted : array-like of shape = number_of_outputs
        Target values predicted by the model.

    Returns
    -------
    loss : float
        RRSE output is non-negative values. Becoming 0.0 means your
        model outputs are exactly matched by true target values.

    Examples
    --------
    >>> y = [3, -0.5, 2, 7]
    >>> y_predicted = [2.5, 0.0, 2, 8]
    >>> root_relative_mean_squared_error(y, y_predicted)
    0.206

    """
    numerator = np.sum(np.square((y_predicted - y)))
    denominator = np.sum(np.square((y_predicted - np.mean(y, axis=0))))
    return np.sqrt(np.divide(numerator, denominator))


def mean_absolute_error(y, y_predicted):
    """Calculate the Mean absolute error.

    Parameters
    ----------
    y : array-like of shape = number_of_outputs
        Represent the target values.
    y_predicted : array-like of shape = number_of_outputs
        Target values predicted by the model.

    Returns
    -------
    loss : float or ndarray of floats
        MAE output is non-negative values. Becoming 0.0 means your
        model outputs are exactly matched by true target values.

    References
    ----------
    [1] Wikipedia entry on the Mean absolute error
        https://en.wikipedia.org/wiki/Mean_absolute_error

    Examples
    --------
    >>> y = [3, -0.5, 2, 7]
    >>> y_predicted = [2.5, 0.0, 2, 8]
    >>> mean_absolute_error(y, y_predicted)
    0.5

    """
    output_errors = np.average(np.abs(y - y_predicted))
    return np.average(output_errors)


def mean_squared_log_error(y, y_predicted):
    """Calculate the Mean Squared Logarithmic Error.

    Parameters
    ----------
    y : array-like of shape = number_of_outputs
        Represent the target values.
    y_predicted : array-like of shape = number_of_outputs
        Target values predicted by the model.

    Returns
    -------
    loss : float
        MSLE output is non-negative values. Becoming 0.0 means your
        model outputs are exactly matched by true target values.

    Examples
    --------
    >>> y = [3, 5, 2.5, 7]
    >>> y_predicted = [2.5, 5, 4, 8]
    >>> mean_squared_log_error(y, y_predicted)
    0.039

    """
    return mean_squared_error(np.log1p(y), np.log1p(y_predicted))


def median_absolute_error(y, y_predicted):
    """Calculate the Median Absolute Error.

    Parameters
    ----------
    y : array-like of shape = number_of_outputs
        Represent the target values.
    y_predicted : array-like of shape = number_of_outputs
        Target values predicted by the model.

    Returns
    -------
    loss : float
        MdAE output is non-negative values. Becoming 0.0 means your
        model outputs are exactly matched by true target values.

    References
    ----------
    [1] Wikipedia entry on the Median absolute deviation
        https://en.wikipedia.org/wiki/Median_absolute_deviation

    Examples
    --------
    >>> y = [3, -0.5, 2, 7]
    >>> y_predicted = [2.5, 0.0, 2, 8]
    >>> median_absolute_error(y, y_predicted)
    0.5

    """
    return np.median(np.abs(y - y_predicted))


def explained_variance_score(y, y_predicted):
    """Calculate the Explained Variance Score.

    Parameters
    ----------
    y : array-like of shape = number_of_outputs
        Represent the target values.
    y_predicted : array-like of shape = number_of_outputs
        Target values predicted by the model.

    Returns
    -------
    loss : float
        EVS output is non-negative values. Becoming 1.0 means your
        model outputs are exactly matched by true target values.
        Lower values means worse results.

    References
    ----------
    [1] Wikipedia entry on the Explained Variance
        https://en.wikipedia.org/wiki/Explained_variation

    Examples
    --------
    >>> y = [3, -0.5, 2, 7]
    >>> y_predicted = [2.5, 0.0, 2, 8]
    >>> explained_variance_score(y, y_predicted)
    0.957

    """
    y_diff_avg = np.average(y - y_predicted)
    numerator = np.average((y - y_predicted - y_diff_avg) ** 2)
    y_avg = np.average(y)
    denominator = np.average((y - y_avg) ** 2)
    nonzero_numerator = numerator != 0
    nonzero_denominator = denominator != 0
    valid_score = nonzero_numerator & nonzero_denominator
    output_scores = np.ones(y.shape[0])
    output_scores[valid_score] = 1 - (numerator[valid_score] / denominator[valid_score])
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.0
    return np.average(output_scores)


def r2_score(y, y_predicted):
    """Calculate the R2 score.

    Parameters
    ----------
    y : array-like of shape = number_of_outputs
        Represent the target values.
    y_predicted : array-like of shape = number_of_outputs
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
    [1] Wikipedia entry on the Coefficient of determination
        https://en.wikipedia.org/wiki/Coefficient_of_determination

    Examples
    --------
    >>> y = [3, -0.5, 2, 7]
    >>> y_predicted = [2.5, 0.0, 2, 8]
    >>> explained_variance_score(y, y_predicted)
    0.948

    """
    numerator = ((y - y_predicted) ** 2).sum(axis=0, dtype=np.float64)
    denominator = ((y - np.average(y, axis=0)) ** 2).sum(axis=0, dtype=np.float64)
    nonzero_denominator = denominator != 0
    nonzero_numerator = numerator != 0
    valid_score = nonzero_denominator & nonzero_numerator
    output_scores = np.ones([y.shape[0]])
    output_scores[valid_score] = 1 - (numerator[valid_score] / denominator[valid_score])
    # arbitrary set to zero to avoid -inf scores, having a constant
    # y_true is not interesting for scoring a regression anyway
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.0
    return np.average(output_scores)


def symmetric_mean_absolute_percentage_error(y, y_predicted):
    """Calculate the SMAPE score.

    Parameters
    ----------
    y : array-like of shape = number_of_outputs
        Represent the target values.
    y_predicted : array-like of shape = number_of_outputs
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
    [1] Wikipedia entry on the Symmetric mean absolute percentage error
        https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error

    Examples
    --------
    >>> y = [3, -0.5, 2, 7]
    >>> y_predicted = [2.5, 0.0, 2, 8]
    >>> symmetric_mean_absolute_percentage_error(y, y_predicted)
    57.87

    """
    return (
        100
        / len(y)
        * np.sum(2 * np.abs(y_predicted - y) / (np.abs(y) + np.abs(y_predicted)))
    )
