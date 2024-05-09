"""Base class for Basis Function."""

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
        predefined_regressors: Optional[np.ndarray] = None,
    ):
        """Abstract method."""

    @abstractmethod
    def transform(
        self,
        data: np.ndarray,
        max_lag: int = 1,
        predefined_regressors: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Abstract methods."""
