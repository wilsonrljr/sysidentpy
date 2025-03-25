from .check_arrays import (
    check_length,
    check_dimension,
    check_infinity,
    check_nan,
    check_x_y,
)

from .generate_data import get_miso_data, get_siso_data
from .display_results import results
from sysidentpy.utils.lags import (
    get_lag_from_regressor_code,
    get_max_lag_from_model_code,
    get_max_xlag,
    get_max_ylag,
)

# TODO: narmax_tools need refactor to avoid circular import
# from sysidentpy.utils.narmax_tools import regressor_code, set_weights,
# train_test_split
from sysidentpy.utils.information_matrix import (
    count_model_regressors,
    build_lagged_matrix,
)
from sysidentpy.utils.plotting import plot_results, plot_residues_correlation
from sysidentpy.utils.save_load import save_model, load_model
from sysidentpy.utils.simulation import (
    get_index_from_regressor_code,
    list_input_regressor_code,
    list_output_regressor_code,
)

__ALL__ = [
    "check_length",
    "check_dimension",
    "check_infinity",
    "check_nan",
    "check_X_y",
    "get_miso_data",
    "get_siso_data",
    "results",
    "get_lag_from_regressor_code",
    "get_max_lag_from_model_code",
    "get_max_xlag",
    "get_max_ylag",
    "plot_results",
    "plot_residues_correlation",
    "save_model",
    "load_model",
    "get_index_from_regressor_code",
    "list_input_regressor_code",
    "list_output_regressor_code",
    "count_model_regressors",
    "build_lagged_matrix",
]
