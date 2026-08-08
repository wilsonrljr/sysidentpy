"""Base class for Basis Function."""

from abc import ABCMeta, abstractmethod
from itertools import combinations_with_replacement
from typing import Optional

import numpy as np


class BaseBasisFunction(metaclass=ABCMeta):
    """Base class for Model Structure Selection."""

    @abstractmethod
    def __init__(self, degree: int = 1):
        self.degree = degree

    def _validate_include_bias(self, include_bias: bool) -> None:
        """Validate the flag controlling the explicit constant regressor."""
        if not isinstance(include_bias, bool):
            raise TypeError(f"include_bias must be False or True. Got {include_bias}")

    def _get_feature_codes(
        self,
        base_codes: np.ndarray,
        *,
        xlag=1,
        ylag=1,
        model_type: str = "NARMAX",
    ) -> np.ndarray:
        """Return polynomial feature codes built from the lagged-variable codes.

        This concrete default builds the legacy polynomial candidate layout for
        custom basis functions inheriting from :class:`BaseBasisFunction`. The
        regressor dictionary adjusts that candidate to the fitted matrix width for
        custom layouts that do not provide canonical codes. Built-in basis functions
        whose feature layout differs from Polynomial override this method.

        Parameters
        ----------
        base_codes : ndarray of int of shape (n_lagged_variables + 1,)
            Codes for the intercept followed by the output and input lags.
        xlag : int or list, default=1
            Input-variable lags. Used by layouts whose ordering depends on lags.
        ylag : int or list, default=1
            Output-variable lags. Used by layouts whose ordering depends on lags.
        model_type : str, default="NARMAX"
            Model type associated with the lagged-variable codes.

        Returns
        -------
        ndarray of int of shape (n_features, degree)
            Codes in the same order as the generated feature columns.
        """
        combinations = np.asarray(
            list(combinations_with_replacement(base_codes, self.degree)),
            dtype=base_codes.dtype,
        )
        return combinations[:, ::-1]

    def _get_univariate_feature_codes(
        self,
        base_codes: np.ndarray,
        *,
        include_bias: bool,
        ensemble: bool,
    ) -> np.ndarray:
        """Return codes for univariate degree expansions of every lagged variable."""
        lag_codes = base_codes[base_codes != 0]
        code_width = self.degree
        linear_codes = np.zeros(
            (lag_codes.shape[0], code_width), dtype=base_codes.dtype
        )
        linear_codes[:, 0] = lag_codes

        expansion_codes = []
        for code in lag_codes:
            for degree in range(1, self.degree + 1):
                row = np.zeros(code_width, dtype=base_codes.dtype)
                row[:degree] = code
                expansion_codes.append(row)

        sections = []
        if ensemble:
            sections.append(linear_codes)
        if include_bias:
            sections.append(np.zeros((1, code_width), dtype=base_codes.dtype))
        if expansion_codes:
            sections.append(np.asarray(expansion_codes, dtype=base_codes.dtype))

        if not sections:
            return np.empty((0, code_width), dtype=base_codes.dtype)

        return np.vstack(sections)

    @abstractmethod
    def fit(
        self,
        data: np.ndarray,
        max_lag: int = 1,
        ylag: int = 1,
        xlag: int = 1,
        model_type: str = "NARMAX",
        predefined_regressors: Optional[np.ndarray] = None,
    ):
        """Abstract method."""

    @abstractmethod
    def transform(
        self,
        data: np.ndarray,
        max_lag: int = 1,
        ylag: int = 1,
        xlag: int = 1,
        model_type: str = "NARMAX",
        predefined_regressors: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Abstract methods."""
