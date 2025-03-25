import numpy as np

from sysidentpy.utils.lags import _process_xlag, _process_ylag


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
    n_samples = col_to_shift.shape[0]
    tmp_column = np.zeros((n_samples, 1))
    aux = col_to_shift[0 : n_samples - lag].reshape(-1, 1)
    tmp_column[lag:, 0] = aux[:, 0]
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
    if n_inputs == 1:
        x_lagged = np.column_stack([shift_column(x[:, 0], lag) for lag in xlag])
    else:
        x_lagged = np.zeros([len(x), 1])  # just to stack other columns
        # if user input a nested list like [[1, 2], 4], the following
        # line convert it to [[1, 2], [4]].
        # Remember, for multiple inputs all lags must be entered explicitly
        xlag = [[i] if isinstance(i, int) else i for i in xlag]
        for col in range(n_inputs):
            x_lagged_col = np.column_stack(
                [shift_column(x[:, col], lag) for lag in xlag[col]]
            )
            x_lagged = np.column_stack([x_lagged, x_lagged_col])

        x_lagged = x_lagged[:, 1:]  # remove the column of 0 created above

    return x_lagged


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
    y_lagged = np.column_stack([shift_column(y[:, 0], lag) for lag in ylag])
    return y_lagged


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
    n_inputs, xlag = _process_xlag(x, xlag)
    ylag = _process_ylag(ylag)
    x_lagged = _create_lagged_x(x, n_inputs, xlag)
    y_lagged = _create_lagged_y(y, ylag)
    lagged_data = np.concatenate([y_lagged, x_lagged], axis=1)
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
    ylag = _process_ylag(ylag)
    y_lagged = _create_lagged_y(y, ylag)
    constant = np.ones([y_lagged.shape[0], 1])
    data = np.concatenate([constant, y_lagged], axis=1)
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

    n_inputs, xlag = _process_xlag(x, xlag)
    x_lagged = _create_lagged_x(x, n_inputs, xlag)
    constant = np.ones([x_lagged.shape[0], 1])
    data = np.concatenate([constant, x_lagged], axis=1)
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
    lagged_data = initial_lagged_matrix(x, y, xlag, ylag)
    constant = np.ones([lagged_data.shape[0], 1])
    data = np.concatenate([constant, lagged_data], axis=1)
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
