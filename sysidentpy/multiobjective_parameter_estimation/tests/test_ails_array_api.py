import numpy as np
import pytest

from sysidentpy import config_context
from sysidentpy.multiobjective_parameter_estimation.estimators import AILS


def test_ails_rejects_array_api_dispatch_with_clear_error():
    xp = pytest.importorskip("array_api_strict")
    final_model = np.array([[1001, 0], [2001, 0]])
    estimator = AILS(
        final_model=final_model,
        static_gain=False,
        static_function=False,
    )
    x = np.arange(12, dtype=float).reshape(-1, 1)
    y = (0.5 * x).reshape(-1, 1)

    with config_context(array_api_dispatch=True):
        with pytest.raises(NotImplementedError, match=r"AILS.*requires NumPy"):
            estimator.estimate(
                y=xp.asarray(y),
                X=xp.asarray(x),
                weighing_matrix=np.ones((1, 1)),
            )
