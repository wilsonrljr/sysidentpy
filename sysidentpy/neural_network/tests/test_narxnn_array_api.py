import numpy as np
import pytest

from sysidentpy import config_context
from sysidentpy.basis_function import Polynomial


def test_narxnn_rejects_array_api_dispatch_with_clear_error():
    torch = pytest.importorskip("torch")
    xp = pytest.importorskip("array_api_strict")
    from sysidentpy.neural_network import NARXNN

    model = NARXNN(
        ylag=1,
        xlag=1,
        net=torch.nn.Linear(1, 1),
        epochs=1,
        basis_function=Polynomial(degree=1),
    )
    x = np.arange(12, dtype=float).reshape(-1, 1)
    y = (0.5 * x).reshape(-1, 1)

    with config_context(array_api_dispatch=True):
        with pytest.raises(NotImplementedError, match=r"NARXNN.*requires NumPy"):
            model.fit(X=xp.asarray(x), y=xp.asarray(y))


@pytest.mark.parametrize("steps_ahead", [None, 1, 3])
def test_narxnn_predict_rejects_array_api_dispatch_with_clear_error(steps_ahead):
    torch = pytest.importorskip("torch")
    xp = pytest.importorskip("array_api_strict")
    from sysidentpy.neural_network import NARXNN

    y = np.linspace(0.0, 1.0, 12).reshape(-1, 1)
    model = NARXNN(
        ylag=1,
        xlag=1,
        model_type="NAR",
        net=torch.nn.Linear(1, 1),
        epochs=1,
        verbose=False,
        basis_function=Polynomial(degree=1, include_bias=False),
    )
    model.fit(X=None, y=y)

    with config_context(array_api_dispatch=True):
        with pytest.raises(NotImplementedError, match=r"NARXNN.*requires NumPy"):
            model.predict(
                X=None,
                y=xp.asarray(y),
                steps_ahead=steps_ahead,
                forecast_horizon=3,
            )
