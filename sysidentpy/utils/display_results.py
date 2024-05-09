"""Display results formatted for the user."""

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause

from collections import Counter

import numpy as np


def results(
    final_model=None,
    theta=None,
    err=None,
    n_terms=None,
    theta_precision=4,
    err_precision=8,
    dtype="dec",
):
    """Write the model regressors, parameters and ERR values.

    This function returns the model regressors, its respective parameter
    and ERR value on a string matrix.

    Parameters
    ----------
    theta_precision : int (default: 4)
        Precision of shown parameters values.
    err_precision : int (default: 8)
        Precision of shown ERR values.
    dtype : string (default: 'dec')
        Type of representation:
        sci - Scientific notation;
        dec - Decimal notation.

    Returns
    -------
    output_matrix : string
        Where:
            First column represents each regressor element;
            Second column represents associated parameter;
            Third column represents the error reduction ratio associated
            to each regressor.

    """
    if not isinstance(theta_precision, int) or theta_precision < 1:
        raise ValueError(
            f"theta_precision must be integer and > zero. Got {theta_precision}."
        )

    if not isinstance(err_precision, int) or err_precision < 1:
        raise ValueError(
            f"err_precision must be integer and > zero. Got {err_precision}."
        )

    if dtype not in ("dec", "sci"):
        raise ValueError(f"dtype must be dec or sci. Got {dtype}.")

    output_matrix = []
    theta_output_format = "{:." + str(theta_precision)
    err_output_format = "{:." + str(err_precision)

    if dtype == "dec":
        theta_output_format = theta_output_format + "f}"
        err_output_format = err_output_format + "f}"
    else:
        theta_output_format = theta_output_format + "E}"
        err_output_format = err_output_format + "E}"

    for i in range(n_terms):
        if np.max(final_model[i]) < 1:
            tmp_regressor = str(1)
        else:
            regressor_dic = Counter(final_model[i])
            regressor_string = []
            for j in range(len(list(regressor_dic.keys()))):
                regressor_key = list(regressor_dic.keys())[j]
                if regressor_key < 1:
                    translated_key = ""
                    translated_exponent = ""
                else:
                    delay_string = str(
                        int(regressor_key - np.floor(regressor_key / 1000) * 1000)
                    )
                    if int(regressor_key / 1000) < 2:
                        translated_key = "y(k-" + delay_string + ")"
                    else:
                        translated_key = (
                            "x"
                            + str(int(regressor_key / 1000) - 1)
                            + "(k-"
                            + delay_string
                            + ")"
                        )
                    if regressor_dic[regressor_key] < 2:
                        translated_exponent = ""
                    else:
                        translated_exponent = "^" + str(regressor_dic[regressor_key])
                regressor_string.append(translated_key + translated_exponent)
            tmp_regressor = "".join(regressor_string)

        current_parameter = theta_output_format.format(theta[i, 0])
        current_err = err_output_format.format(err[i])
        current_output = [tmp_regressor, current_parameter, current_err]
        output_matrix.append(current_output)

    return output_matrix
