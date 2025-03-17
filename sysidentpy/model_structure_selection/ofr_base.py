"""Base methods for Orthogonal Forward Regression algorithm."""

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause

import warnings
from abc import ABCMeta, abstractmethod
from typing import Union, Tuple, Optional

import numpy as np

from sysidentpy.narmax_base import house, rowhouse
from sysidentpy.utils.check_arrays import check_positive_int, num_features

from ..basis_function import Fourier, Polynomial
from ..narmax_base import BaseMSS


class OFRBase(metaclass=ABCMeta):
    """Base class for Model Structure Selection."""

    @abstractmethod
    def __init__(self):
        super().__init__(self)
        self.max_lag = None
        self.n_inputs = None
        self.theta = None
        self.final_model = None
        self.pivv = None

    @abstractmethod
    def fit(self, *, X, y):
        """Abstract method."""

    @abstractmethod
    def predict(
        self,
        *,
        X: Optional[np.ndarray] = None,
        y: np.ndarray,
        steps_ahead: Optional[int] = None,
        forecast_horizon: int = 1,
    ) -> np.ndarray:
        """Abstract method."""
