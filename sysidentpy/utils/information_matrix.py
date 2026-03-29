import numpy as np
from numpy.lib.stride_tricks import as_strided

from sysidentpy._lib._array_api import (
    _asarray,
    _concat,
    _copy,
    _is_numpy_namespace,
    _ones,
    _zeros,
    device as _device,
    get_namespace,
)
from sysidentpy.utils.lags import _process_xlag, _process_ylag


def _ensure_2d_array(data, xp):
    """Convert input data to a 2D array without changing its namespace."""
    if _is_numpy_namespace(xp):
        array = np.asarray(data, dtype=float)
        if array.ndim == 1:
            return array.reshape(-1, 1)
        return array

    array = _asarray(data, xp=xp, target_device=_device(data))
    if array.ndim == 1:
        return xp.reshape(array, (-1, 1))
    return array


def _normalize_lag_input(lag_value):
    """Return lag values as a numpy array while preserving user order."""
    if isinstance(lag_value, (list, tuple, np.ndarray)):
        return np.asarray(lag_value, dtype=int)

    return np.asarray([lag_value], dtype=int)


def _build_sliding_windows(data: np.ndarray, max_lag: int) -> np.ndarray:
    """Create a sliding window view along time for efficient lag reuse."""
    if max_lag < 1:
        raise ValueError("max_lag must be >= 1 to build lagged windows")

    xp = get_namespace(data)

    if _is_numpy_namespace(xp):
        padded = np.vstack([np.zeros((max_lag, data.shape[1])), data])
        n_samples = data.shape[0]
        window = max_lag + 1
        shape = (n_samples, window, data.shape[1])
        strides = (padded.strides[0], padded.strides[0], padded.strides[1])
        return as_strided(padded, shape=shape, strides=strides, writeable=False)

    padded = _concat(
        xp,
        [
            _zeros(
                xp,
                (max_lag, data.shape[1]),
                dtype=data.dtype,
                target_device=_device(data),
            ),
            data,
        ],
        axis=0,
    )
    n_samples = data.shape[0]
    window = max_lag + 1
    windows = []
    for offset in range(window):
        current_window = padded[offset : offset + n_samples, :]
        windows.append(xp.reshape(current_window, (n_samples, 1, data.shape[1])))

    return _concat(xp, windows, axis=1)


def shift_column(col_to_shift: np.ndarray, lag: int) -> np.ndarray:
    """Shift an array by a specified lag, introducing zeros for missing values.

    Parameters
    ----------
    col_to_shift : array-like of shape (n_samples,)
        The input or output time-series data to be lagged.
    lag : int
        The number of time steps to shift the data.

    Returns
    -------
    tmp_column : ndarray of shape (n_samples, 1)
        The shifted array, where the first `lag` values are replaced with zeros.

    Examples
    --------
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> shift_column(y, 1)
    array([[0],
           [1],
           [2],
           [3],
           [4]])

    """
    if lag < 0:
        raise ValueError("lag must be non-negative")

    xp = get_namespace(col_to_shift)
    n_samples = col_to_shift.shape[0]
    tmp_column = _zeros(
        xp,
        (n_samples, 1),
        dtype=col_to_shift.dtype,
        target_device=_device(col_to_shift),
    )
    if lag == 0:
        return _copy(xp, col_to_shift)

    if lag < n_samples:
        tmp_column[lag:, 0] = col_to_shift[: n_samples - lag, 0]
    return tmp_column


def _create_lagged_x(x: np.ndarray, n_inputs: int, xlag) -> np.ndarray:
    """Create a lagged matrix of input variables without interaction terms.

    Parameters
    ----------
    x : array-like
        Input data used during the training phase.
    n_inputs : int
        The number of input variables.

    Returns
    -------
    x_lagged : ndarray
        A matrix where each column represents a lagged version of an input variable.

    """
    xp = get_namespace(x)
    x = _ensure_2d_array(x, xp)

    if n_inputs == 1:
        lag_list = _normalize_lag_input(xlag)
    else:
        lag_list = [_normalize_lag_input(lag_value) for lag_value in xlag]

    max_lag = (
        int(max(lag.max() for lag in lag_list))
        if isinstance(lag_list, list)
        else int(np.max(lag_list))
    )
    windows = _build_sliding_windows(x, max_lag)
    lagged_columns = []

    if n_inputs == 1:
        for lag in lag_list:
            index = max_lag - int(lag)
            lagged_columns.append(xp.reshape(windows[:, index, 0], (-1, 1)))
    else:
        for col in range(n_inputs):
            lags = lag_list[col]
            for lag in lags:
                index = max_lag - int(lag)
                lagged_columns.append(xp.reshape(windows[:, index, col], (-1, 1)))

    if len(lagged_columns) == 1:
        return lagged_columns[0]

    return _concat(xp, lagged_columns, axis=1)


def _create_lagged_y(y: np.ndarray, ylag) -> np.ndarray:
    """Create a lagged matrix of the output variable.

    Parameters
    ----------
    y : array-like
        Output data used on training phase.

    Returns
    -------
    y_lagged : ndarray
        A matrix where each column represents a lagged version of the output
        variable.

    """
    xp = get_namespace(y)
    y = _ensure_2d_array(y, xp)

    lag_array = _normalize_lag_input(ylag)
    max_lag = int(np.max(lag_array))
    windows = _build_sliding_windows(y, max_lag)
    lagged_columns = []
    for lag in lag_array:
        index = max_lag - int(lag)
        lagged_columns.append(xp.reshape(windows[:, index, 0], (-1, 1)))

    if len(lagged_columns) == 1:
        return lagged_columns[0]

    return _concat(xp, lagged_columns, axis=1)


def initial_lagged_matrix(x: np.ndarray, y: np.ndarray, xlag, ylag) -> np.ndarray:
    """Construct a matrix with lagged versions of input and output variables.

    Parameters
    ----------
    x : array-like
        Input data used during the training phase.
    y : array-like
        Output data used during the training phase.
    xlag : int, list of int, or nested list of int
        Input that can be a single integer, a list, or a nested list.
    ylag : int or list of int
        The range of lags according to user definition.

    Returns
    -------
    lagged_data : ndarray
        The combined matrix containing lagged input and output values.

    Examples
    --------
    If `xlag=2` and `ylag=2`, the resulting matrix will contain columns:
    Y[k-1], Y[k-2], x[k-1], x[k-2].

    """
    xp = get_namespace(x, y)
    n_inputs, xlag = _process_xlag(x, xlag)
    ylag = _process_ylag(ylag)
    x_lagged = _create_lagged_x(x, n_inputs, xlag)
    y_lagged = _create_lagged_y(y, ylag)
    lagged_data = _concat(xp, [y_lagged, x_lagged], axis=1)
    return lagged_data


def build_output_matrix(y, ylag: np.ndarray) -> np.ndarray:
    """Build the information matrix of output values.

    Each column of the information matrix represents a candidate
    regressor. The set of candidate regressors are based on xlag,
    ylag, and degree entered by the user.

    Parameters
    ----------
    y : array-like
        Output data used during the training phase.
    ylag : int or list of int
        The range of lags according to user definition.

    Returns
    -------
    data = ndarray of floats
        The constructed output regressor matrix.

    """
    # Generate a lagged data which each column is an input or output
    # related to its respective lags. With this approach we can create
    # the information matrix by using all possible combination of
    # the columns as a product in the iterations
    xp = get_namespace(y)
    ylag = _process_ylag(ylag)
    y_lagged = _create_lagged_y(y, ylag)
    constant = _ones(
        xp,
        (y_lagged.shape[0], 1),
        dtype=y_lagged.dtype,
        target_device=_device(y_lagged),
    )
    data = _concat(xp, [constant, y_lagged], axis=1)
    return data


def build_input_matrix(x, xlag: np.ndarray) -> np.ndarray:
    """Build the information matrix of input values.

    Each column of the information matrix represents a candidate
    regressor. The set of candidate regressors are based on xlag,
    ylag, and degree entered by the user.

    Parameters
    ----------
    x : array-like
        Input data used during the training phase.
    xlag : int, list of int, or nested list of int
        Input that can be a single integer, a list, or a nested list.

    Returns
    -------
    data = ndarray of floats
        The lagged matrix built in respect with each lag and column.

    """
    # Generate a lagged data which each column is a input or output
    # related to its respective lags. With this approach we can create
    # the information matrix by using all possible combination of
    # the columns as a product in the iterations

    xp = get_namespace(x)
    n_inputs, xlag = _process_xlag(x, xlag)
    x_lagged = _create_lagged_x(x, n_inputs, xlag)
    constant = _ones(
        xp,
        (x_lagged.shape[0], 1),
        dtype=x_lagged.dtype,
        target_device=_device(x_lagged),
    )
    data = _concat(xp, [constant, x_lagged], axis=1)
    return data


def build_input_output_matrix(x: np.ndarray, y: np.ndarray, xlag, ylag) -> np.ndarray:
    """Build the information matrix.

    Each column of the information matrix represents a candidate
    regressor. The set of candidate regressors are based on xlag,
    ylag, and degree entered by the user.

    Parameters
    ----------
    x : array-like
        Input data used on training phase.
    y : array-like
        Target data used on training phase.
    xlag : int, list of int, or nested list of int
        Input that can be a single integer, a list, or a nested list.
    ylag : int or list of int
        The range of lags according to user definition.

    Returns
    -------
    data = ndarray of floats
        The constructed information matrix.

    """
    # Generate a lagged data which each column is a input or output
    # related to its respective lags. With this approach we can create
    # the information matrix by using all possible combination of
    # the columns as a product in the iterations
    xp = get_namespace(x, y)
    lagged_data = initial_lagged_matrix(x, y, xlag, ylag)
    constant = _ones(
        xp,
        (lagged_data.shape[0], 1),
        dtype=lagged_data.dtype,
        target_device=_device(lagged_data),
    )
    data = _concat(xp, [constant, lagged_data], axis=1)
    return data


def get_build_io_method(model_type):
    """Get info criteria method.

    Parameters
    ----------
    model_type = str
        The type of the model (NARMAX, NAR or NFIR)

    Returns
    -------
    build_method = Self
        Method to build the input-output matrix
    """
    build_matrix_options = {
        "NARMAX": build_input_output_matrix,
        "NFIR": build_input_matrix,
        "NAR": build_output_matrix,
    }
    return build_matrix_options.get(model_type, None)


def build_lagged_matrix(x, y, xlag, ylag, model_type):
    build_matrix = get_build_io_method(model_type)
    if model_type == "NARMAX":
        return build_matrix(x, y, xlag, ylag)
    if model_type == "NFIR":
        return build_matrix(x, xlag)

    return build_matrix(y, ylag)


def count_model_regressors(
    *,
    x: np.ndarray,
    y: np.ndarray,
    xlag: int,
    ylag: int,
    model_type: str,
    basis_function,
    is_neural_narx: bool = False,
) -> int:
    """
    Compute the number of model regressors after applying the basis function.

    Parameters
    ----------
    x : np.ndarray
        Input data.
    y : np.ndarray
        Output data.
    xlag : int
        Number of lags for input variables.
    ylag : int
        Number of lags for output variables.
    model_type : str
        The type of model ('NARMAX', 'NAR', 'NFIR', etc.).
    basis_function : object
        The basis function used for feature transformation.
    is_neural_narx : bool, optional
        Whether to adjust for a neural NARX model, by default False.

    Returns
    -------
    int
        The number of regressors/features after transformation.
    """
    data = build_lagged_matrix(x, y, xlag, ylag, model_type)
    n_features = basis_function.fit(data[:3, :]).shape[1]
    if is_neural_narx:
        return n_features - 1

    return n_features
