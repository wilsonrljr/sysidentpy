""" Build Polynomial NARMAX Models using FROLS algorithm """

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
#           Luan Pascoal da Costa Andrade <luan_pascoal13@hotmail.com>
#           Samuel Carlos Pessoa Oliveira <samuelcpoliveira@gmail.com>
#           Samir Angelo Milani Martins <martins@ufsj.edu.br>
# License: BSD 3 clause


import warnings
import numpy as np
from collections import Counter
from ..base import GenerateRegressors
from ..base import HouseHolder
from ..base import InformationMatrix
from ..parameter_estimation.estimators import Estimators
from ..residues.residues_correlation import ResiduesAnalysis
from ..utils._check_arrays import check_X_y, _check_positive_int
from ..base import _get_max_lag
import warnings


class FROLS(
    Estimators, GenerateRegressors, HouseHolder, InformationMatrix, ResiduesAnalysis
):
    """Forward Regression Orthogonal Least Squares algorithm.

    Parameters
    ----------
    ylag : int, default=2
        The maximum lag of the output.
    xlag : int, default=2
        The maximum lag of the input.
    order_selection: bool, default=False
        Whether to use information criteria for order selection.
    info_criteria : str, default="aic"
        The information criteria method to be used.
    n_terms : int, default=None
        The number of the model terms to be selected.
        Note that n_terms overwrite the information criteria
        values.
    n_inputs : int, default=1
        The number of inputs of the system.
    n_info_values : int, default=10
        The number of iterations of the information
        criteria method.
    estimator : str, default="least_squares"
        The parameter estimation method.
    extended_least_squares : bool, default=False
        Whether to use extended least squares method
        for parameter estimation.
        Note that we define a specific set of noise regressors.
    aux_lag : int, default=1
        Temporary lag value used only for parameter estimation.
        This value is overwritten by the max_lag value and will
        be removed in v0.1.4.
    lam : float, default=0.98
        Forgetting factor of the Recursive Least Squares method.
    delta : float, default=0.01
        Normalization factor of the P matrix.
    offset_covariance : float, default=0.2
        The offset covariance factor of the affine least mean squares
        filter.
    mu : float, default=0.01
        The convergence coefficient (learning rate) of the filter.
    eps : float
        Normalization factor of the normalized filters.
    gama : float, default=0.2
        The leakage factor of the Leaky LMS method.
    weight : float, default=0.02
        Weight factor to control the proportions of the error norms
        and offers an extra degree of freedom within the adaptation
        of the LMS mixed norm method.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from sysidentpy.polynomial_basis import PolynomialNarmax
    >>> from sysidentpy.metrics import root_relative_squared_error
    >>> from sysidentpy.utils.generate_data import get_miso_data, get_siso_data
    >>> x_train, x_valid, y_train, y_valid = get_siso_data(n=1000,
    ...                                                    colored_noise=True,
    ...                                                    sigma=0.2,
    ...                                                    train_percentage=90)
    >>> model = PolynomialNarmax(non_degree=2,
    ...                          order_selection=True,
    ...                          n_info_values=10,
    ...                          extended_least_squares=False,
    ...                          ylag=2, xlag=2,
    ...                          info_criteria='aic',
    ...                          estimator='least_squares',
    ...                          )
    >>> model.fit(x_train, y_train)
    >>> yhat = model.predict(x_valid, y_valid)
    >>> rrse = root_relative_squared_error(y_valid, yhat)
    >>> print(rrse)
    0.001993603325328823
    >>> results = pd.DataFrame(model.results(err_precision=8,
    ...                                      dtype='dec'),
    ...                        columns=['Regressors', 'Parameters', 'ERR'])
    >>> print(results)
        Regressors Parameters         ERR
    0        x1(k-2)     0.9000  0.95556574
    1         y(k-1)     0.1999  0.04107943
    2  x1(k-1)y(k-1)     0.1000  0.00335113

    References
    ----------
    [1] Manuscript: Orthogonal least squares methods and their application
        to non-linear system identification
        https://eprints.soton.ac.uk/251147/1/778742007_content.pdf
    [2] Manuscript (portuguese): Identificação de Sistemas não Lineares
        Utilizando Modelos NARMAX Polinomiais – Uma Revisão
        e Novos Resultados
    """

    def __init__(
        self,
        ylag=2,
        xlag=2,
        order_selection=False,
        info_criteria="aic",
        n_terms=None,
        n_inputs=1,
        n_info_values=10,
        estimator="recursive_least_squares",
        extended_least_squares=True,
        lam=0.98,
        delta=0.01,
        offset_covariance=0.2,
        mu=0.01,
        eps=np.finfo(np.float64).eps,
        gama=0.2,
        weight=0.02,
        basis_function=None,
        model_type="NARMAX"
    ):

        self.non_degree = basis_function.non_degree
        self._order_selection = order_selection
        self._n_inputs = n_inputs
        self.ylag = ylag
        self.xlag = xlag
        self.regressor_code = self.regressor_space(
            self.non_degree, xlag, ylag, n_inputs, model_type
        )
        self.max_lag = _get_max_lag(ylag, xlag)
        self.info_criteria = info_criteria
        self.n_info_values = n_info_values
        self.n_terms = n_terms
        self.estimator = estimator
        self._extended_least_squares = extended_least_squares
        self._eps = eps
        self._mu = mu
        self._offset_covariance = offset_covariance
        self._lam = lam
        self._delta = delta
        self._gama = gama
        self._weight = weight  # 0 < weight <1
        self.model_type = model_type
        self._validate_params()
        self.basis_function = basis_function

    def _validate_params(self):
        """Validate input params."""
        if not isinstance(self.n_info_values, int) or self.n_info_values < 1:
            raise ValueError(
                "n_info_values must be integer and > zero. Got %f" % self.n_info_values
            )

        if not isinstance(self._n_inputs, int) or self._n_inputs < 1:
            raise ValueError(
                "n_inputs must be integer and > zero. Got %f" % self._n_inputs
            )

        if not isinstance(self._order_selection, bool):
            raise TypeError(
                "order_selection must be False or True. Got %f" % self._order_selection
            )

        if not isinstance(self._extended_least_squares, bool):
            raise TypeError(
                "extended_least_squares must be False or True. Got %f"
                % self._extended_least_squares
            )

        if self.info_criteria not in ["aic", "bic", "fpe", "lilc"]:
            raise ValueError(
                "info_criteria must be aic, bic, fpe or lilc. Got %s"
                % self.info_criteria
            )
            
        if self.model_type not in ["NARMAX", "NAR", "NFIR"]:
            raise ValueError(
                "model_type must be NARMAX, NAR or NFIR. Got %s"
                % self.model_type
            )

        if (
            not isinstance(self.n_terms, int) or self.n_terms < 1
        ) and self.n_terms is not None:
            raise ValueError(
                "n_terms must be integer and > zero. Got %f" % self.n_terms
            )

        if self.n_terms is not None and self.n_terms > self.regressor_code.shape[0]:
            self.n_terms = self.regressor_code.shape[0]
            warnings.warn(
                (
                    "n_terms is greater than the maximum number of "
                    "all regressors space considering the chosen y_lag,"
                    "u_lag, and non_degree. We set as "
                    "%d "
                )
                % self.regressor_code.shape[0],
                stacklevel=2,
            )