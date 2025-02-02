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
        """Get maximum value from various xlag structures.

        Parameters
        ----------
        xlag : int, list of int, or nested list of int
            Input that can be a single integer, a list, or a nested list.

        Returns
        -------
        int
            Maximum value found.
        """
        if isinstance(xlag, int):  # Case 1: Single integer
            return xlag

        if isinstance(xlag, list):
            # Case 2: Flat list of integers
            if all(isinstance(i, int) for i in xlag):
                return max(xlag)
            # Case 3: Nested list
            return max(chain.from_iterable(xlag))

        raise ValueError("Unsupported data type for xlag")

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
        # TODO: Need to check this method for more than 3 inputs. Its not working
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
