"""Base class for Basis Function."""

from itertools import chain
from abc import ABCMeta, abstractmethod
from typing import Optional

import numpy as np


class BaseBasisFunction(metaclass=ABCMeta):
    """Base class for Model Structure Selection."""

    @abstractmethod
    def __init__(self, degree: int = 1):
        self.degree = degree

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

    def get_max_ylag(self, ylag: int = 1):
        """Get maximum ylag.

        Parameters
        ----------
        ylag : ndarray of int
            The range of lags according to user definition.

        Returns
        -------
        ny : list
            Maximum value of ylag.

        """
        ny = np.max(list(chain.from_iterable([[ylag]])))
        return ny

    def get_max_xlag(self, xlag: int = 1):
        """Get maximum xlag.

        Parameters
        ----------
        xlag : ndarray of int
            The range of lags according to user definition.

        Returns
        -------
        nx : list
            Maximum value of xlag.

        """
        nx = np.max(list(chain.from_iterable([[np.array(xlag, dtype=object)]])))
        return nx

    def get_iterable_list(
        self, ylag: int = 1, xlag: int = 1, model_type: str = "NARMAX"
    ):
        """Get iterable list.

        Parameters
        ----------
        ylag : ndarray of int
            The range of lags according to user definition.
        xlag : ndarray of int
            The range of lags according to user definition.
        model_type : str
            The type of the model (NARMAX, NAR or NFIR).

        Returns
        -------
        iterable_list : list
            List of tuples of the regressor combinations.

        """
        if model_type == "NARMAX":
            ny = self.get_max_ylag(ylag)
            nx = self.get_max_xlag(xlag)
            iterable_list = list(range(ny + nx + 1))
        elif model_type == "NAR":
            ny = self.get_max_ylag(ylag)
            iterable_list = list(range(ny + 1))
        else:
            nx = self.get_max_xlag(xlag)
            iterable_list = list(range(nx + 1))
        return iterable_list
