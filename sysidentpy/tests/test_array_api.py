import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from sysidentpy import config_context
from sysidentpy._lib import _array_api as array_api_utils
from sysidentpy._lib._array_api import (
    _column_stack,
    _concat,
    _copy,
    _diag,
    _einsum_ij_ij_j,
    _get_namespace_and_device,
    _hstack,
    _lstsq,
    _median,
    _nanargmin,
    _pow,
    _require_numpy_namespace,
    _set_element,
    _to_numpy,
    _vector_norm,
    _vstack,
    device as array_device,
    get_namespace,
)


class _DeviceNamespace:
    class Device:
        def __init__(self, name):
            self.name = name

    @staticmethod
    def asarray(array, *, target_device=None):
        del array, target_device
        raise TypeError("device copy unsupported")


class _DeviceProtocolArray:
    def __init__(self, values):
        self._values = np.asarray(values)
        self.device = "device1"

    def __array_namespace__(self, api_version=None):
        del api_version
        return _DeviceNamespace

    def __array__(self, dtype=None):
        return np.asarray(self._values, dtype=dtype)


class _DlpackNamespace:
    pass


class _DlpackFallbackArray:
    def __init__(self, values):
        self._values = np.asarray(values)

    def __array_namespace__(self, api_version=None):
        del api_version
        return _DlpackNamespace

    def __dlpack__(self, *args, **kwargs):
        raise BufferError("dlpack unavailable")

    def __array__(self, dtype=None):
        return np.asarray(self._values, dtype=dtype)


class _SequenceNamespace:
    pass


class _SequenceFallbackArray:
    def __init__(self, values):
        self._values = list(values)

    def __array_namespace__(self, api_version=None):
        del api_version
        return _SequenceNamespace

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __getitem__(self, item):
        return self._values[item]


class _NoVectorNormNamespace:
    sqrt = staticmethod(np.sqrt)
    sum = staticmethod(np.sum)

    class linalg:
        pass


class _PowerOnlyNamespace:
    @staticmethod
    def power(x1, x2):
        return np.power(x1, x2)


class _NoCopyKeywordNamespace:
    @staticmethod
    def asarray(array):
        return np.asarray(array)

    @staticmethod
    def ones_like(array):
        return np.ones_like(array)


class _AtSetter:
    def __init__(self, values, idx):
        self._values = values
        self._idx = idx

    def set(self, value):
        updated = self._values.copy()
        updated[self._idx] = value
        return updated


class _AtIndexer:
    def __init__(self, values):
        self._values = values

    def __getitem__(self, idx):
        return _AtSetter(self._values, idx)


class _ImmutableArrayWithAt:
    def __init__(self, values):
        self._values = np.asarray(values)
        self.at = _AtIndexer(self._values)

    def __setitem__(self, idx, value):
        raise TypeError("immutable array")


class _ImmutableArrayWithoutAt:
    def __init__(self, values):
        self._values = np.asarray(values)

    @property
    def values(self):
        return self._values

    def __setitem__(self, idx, value):
        raise TypeError("immutable array")


def _array_api_strict_namespace():
    return pytest.importorskip("array_api_strict")


def test_get_namespace_returns_numpy_for_non_array_api_inputs_under_dispatch():
    with config_context(array_api_dispatch=True):
        xp = get_namespace(None, [1.0, 2.0], 3.0)

    assert xp is np


def test_get_namespace_and_device_returns_numpy_for_non_array_api_inputs():
    with config_context(array_api_dispatch=True):
        xp, target_device = _get_namespace_and_device(None, [1.0, 2.0], 3.0)

    assert xp is np
    assert target_device is None


def test_device_returns_none_when_no_array_api_objects_are_present():
    assert array_device(None, [1.0, 2.0], 3.0) is None


def test_device_rejects_mixed_array_api_devices():
    xp = _array_api_strict_namespace()
    first = xp.asarray([1.0], device=xp.Device("CPU_DEVICE"))
    second = xp.asarray([2.0], device=xp.Device("device1"))

    with pytest.raises(ValueError, match="different devices"):
        array_device(first, second)


def test_require_numpy_namespace_is_a_noop_for_numpy():
    _require_numpy_namespace(np, feature="LeastSquares")


def test_require_numpy_namespace_mentions_feature_and_dependency():
    xp = _array_api_strict_namespace()

    with pytest.raises(NotImplementedError, match="MetaMSS") as exc_info:
        _require_numpy_namespace(xp, feature="MetaMSS", dependency="SciPy")

    assert "SciPy" in str(exc_info.value)


def test_to_numpy_returns_numpy_array_without_copy():
    array = np.array([[1.0], [2.0], [3.0]])

    assert _to_numpy(array) is array


def test_to_numpy_falls_back_to_array_protocol_after_failed_device_copy():
    array = _DeviceProtocolArray([1.0, 2.0, 3.0])

    result = _to_numpy(array)

    assert_array_equal(result, np.array([1.0, 2.0, 3.0]))


def test_to_numpy_falls_back_from_dlpack_to_array_protocol():
    array = _DlpackFallbackArray([4.0, 5.0, 6.0])

    result = _to_numpy(array)

    assert_array_equal(result, np.array([4.0, 5.0, 6.0]))


def test_to_numpy_uses_numpy_as_final_sequence_fallback():
    array = _SequenceFallbackArray([7.0, 8.0, 9.0])

    result = _to_numpy(array)

    assert_array_equal(result, np.array([7.0, 8.0, 9.0]))


def test_vector_norm_fallback_matches_euclidean_norm():
    values = np.array([3.0, 4.0])

    result = _vector_norm(_NoVectorNormNamespace, values)

    assert result == pytest.approx(5.0)


def test_pow_fallback_uses_power_when_pow_is_missing():
    values = np.array([2.0, 3.0, 4.0])

    result = _pow(_PowerOnlyNamespace, values, 2)

    assert_array_equal(result, np.array([4.0, 9.0, 16.0]))


def test_lstsq_array_api_fallback_matches_numpy_for_rank_deficient_system():
    xp = _array_api_strict_namespace()
    a_np = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    b_np = np.array([[1.0], [2.0], [3.0]])
    a = xp.asarray(a_np, dtype=xp.float64)
    b = xp.asarray(b_np, dtype=xp.float64)

    result = _lstsq(xp, a, b)
    expected = np.linalg.lstsq(a_np, b_np, rcond=None)[0]

    assert_allclose(np.asarray(result), expected, rtol=1e-10, atol=1e-10)


def test_concat_moves_inputs_to_reference_device():
    xp = _array_api_strict_namespace()
    first = xp.asarray([1.0, 2.0], device=xp.Device("device1"))
    second = xp.asarray([3.0, 4.0], device=xp.Device("CPU_DEVICE"))

    result = _concat(xp, [first, second], axis=0)

    assert_array_equal(_to_numpy(result), np.array([1.0, 2.0, 3.0, 4.0]))
    assert str(result.device) == str(xp.Device("device1"))


@pytest.mark.parametrize(
    ("helper", "arrays_np", "expected"),
    [
        (
            _column_stack,
            [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
            np.array([[1.0, 3.0], [2.0, 4.0]]),
        ),
        (
            _vstack,
            [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
            np.array([[1.0, 2.0], [3.0, 4.0]]),
        ),
        (
            _hstack,
            [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
            np.array([1.0, 2.0, 3.0, 4.0]),
        ),
        (
            _hstack,
            [
                np.array([[1.0], [2.0]]),
                np.array([[3.0], [4.0]]),
            ],
            np.array([[1.0, 3.0], [2.0, 4.0]]),
        ),
    ],
)
def test_stack_helpers_match_numpy_semantics(helper, arrays_np, expected):
    xp = _array_api_strict_namespace()
    arrays = [xp.asarray(array, dtype=xp.float64) for array in arrays_np]

    result = helper(xp, arrays)

    assert_array_equal(np.asarray(result), expected)


def test_nanargmin_ignores_nan_values_for_array_api_inputs():
    xp = _array_api_strict_namespace()
    values_np = np.array([[np.nan, 4.0], [2.0, 1.0], [3.0, np.nan]])
    values = xp.asarray(values_np, dtype=xp.float64)

    result = _nanargmin(xp, values, axis=0)

    assert_array_equal(np.asarray(result), np.nanargmin(values_np, axis=0))


@pytest.mark.parametrize(
    ("values_np", "axis"),
    [
        (np.array([[1.0, 5.0, 3.0], [2.0, 4.0, 6.0], [7.0, 8.0, 9.0]]), 0),
        (np.array([[1.0, 5.0, 3.0], [2.0, 4.0, 6.0]]), 0),
        (np.array([1.0, 7.0, 3.0, 5.0]), None),
    ],
)
def test_median_matches_numpy_for_array_api_inputs(values_np, axis):
    xp = _array_api_strict_namespace()
    values = xp.asarray(values_np, dtype=xp.float64)

    result = _median(xp, values, axis=axis)

    assert_allclose(np.asarray(result), np.median(values_np, axis=axis))


def test_einsum_helper_matches_numpy_columnwise_inner_product():
    xp = _array_api_strict_namespace()
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]])
    b_np = np.array([[5.0, 6.0], [7.0, 8.0]])
    a = xp.asarray(a_np, dtype=xp.float64)
    b = xp.asarray(b_np, dtype=xp.float64)

    result = _einsum_ij_ij_j(xp, a, b)

    assert_array_equal(np.asarray(result), np.einsum("ij,ij->j", a_np, b_np))


def test_diag_creates_diagonal_matrix_for_array_api_inputs():
    xp = _array_api_strict_namespace()
    values = xp.asarray([1.0, 2.0, 3.0], dtype=xp.float64)

    result = _diag(xp, values)

    assert_array_equal(np.asarray(result), np.diag([1.0, 2.0, 3.0]))


def test_diag_extracts_diagonal_for_array_api_inputs():
    xp = _array_api_strict_namespace()
    values_np = np.array([[1.0, 2.0], [3.0, 4.0]])
    values = xp.asarray(values_np, dtype=xp.float64)

    result = _diag(xp, values)

    assert_array_equal(np.asarray(result), np.diag(values_np))


def test_diag_rejects_three_dimensional_inputs():
    xp = _array_api_strict_namespace()
    values = xp.asarray(np.zeros((2, 2, 2)), dtype=xp.float64)

    with pytest.raises(ValueError, match="1-D or 2-D"):
        _diag(xp, values)


def test_set_element_uses_at_protocol_when_assignment_fails():
    values = _ImmutableArrayWithAt([1.0, 2.0, 3.0])

    result = _set_element(np, values, 1, 9.0)

    assert_array_equal(result, np.array([1.0, 9.0, 3.0]))


def test_set_element_uses_vendor_fallback_when_at_protocol_is_missing(
    monkeypatch,
):
    values = _ImmutableArrayWithoutAt([1.0, 2.0, 3.0])
    calls = {}

    class _FallbackSetter:
        def __init__(self, array, idx):
            self._array = array
            self._idx = idx

        def set(self, value, xp=None):
            calls["xp"] = xp
            updated = self._array.values.copy()
            updated[self._idx] = value
            return updated

    def fake_at(array, idx):
        calls["array"] = array
        calls["idx"] = idx
        return _FallbackSetter(array, idx)

    monkeypatch.setattr(array_api_utils, "_at", fake_at)

    result = _set_element(np, values, 1, 11.0)

    assert calls == {"array": values, "idx": 1, "xp": np}
    assert_array_equal(result, np.array([1.0, 11.0, 3.0]))


def test_copy_falls_back_when_copy_keyword_is_unsupported():
    values = np.array([1.0, 2.0, 3.0])

    result = _copy(_NoCopyKeywordNamespace, values)

    assert_array_equal(result, values)
    assert result is not values
