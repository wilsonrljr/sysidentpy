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
