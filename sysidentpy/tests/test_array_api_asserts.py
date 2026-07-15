import numpy as np
import pytest

from sysidentpy.tests._array_api_asserts import assert_allclose, assert_array_equal


def test_assert_array_equal_accepts_numpy_array_and_literal_expected():
    actual = np.array([1.0, 2.0, 3.0])

    assert_array_equal(actual, [1.0, 2.0, 3.0])


@pytest.mark.parametrize("desired", [[1.0, 2.0, 3.0], (1.0, 2.0, 3.0)])
def test_assert_array_equal_accepts_array_api_strict_with_literal_expected(desired):
    xp = pytest.importorskip("array_api_strict")
    actual = xp.asarray([1.0, 2.0, 3.0], dtype=xp.float64)

    assert_array_equal(actual, desired)


def test_assert_allclose_accepts_array_api_strict_with_numpy_expected():
    xp = pytest.importorskip("array_api_strict")
    actual = xp.asarray([1.0, 2.0, 3.0], dtype=xp.float64)
    desired = np.array([1.0, 2.0, 3.0])

    assert_allclose(actual, desired, rtol=1e-12, atol=1e-12)


def test_assert_allclose_accepts_torch_cpu_tensor_with_numpy_expected():
    torch = pytest.importorskip("torch")
    actual = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    desired = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    assert_allclose(actual, desired, rtol=1e-12, atol=1e-12)


def test_assert_array_equal_checks_dtype_by_default():
    actual = np.array([1.0, 2.0, 3.0])
    desired = np.array([1, 2, 3])

    with pytest.raises(AssertionError, match="dtypes do not match"):
        assert_array_equal(actual, desired)


def test_assert_array_equal_checks_numpy_expected_dtype_for_other_backends():
    xp = pytest.importorskip("array_api_strict")
    actual = xp.asarray([1.0, 2.0, 3.0], dtype=xp.float64)
    desired = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    with pytest.raises(AssertionError, match="dtypes do not match"):
        assert_array_equal(actual, desired)


def test_assert_array_equal_allows_dtype_check_opt_out():
    actual = np.array([1.0, 2.0, 3.0])
    desired = np.array([1, 2, 3])

    assert_array_equal(actual, desired, check_dtype=False)


def test_assert_array_equal_checks_shape_by_default():
    actual = np.array([1.0, 2.0, 3.0])
    desired = [[1.0, 2.0, 3.0]]

    with pytest.raises(AssertionError, match="shapes do not match"):
        assert_array_equal(actual, desired)


def test_assert_array_equal_rejects_cross_namespace_array_expected():
    xp = pytest.importorskip("array_api_strict")
    actual = np.array([1.0, 2.0, 3.0])
    desired = xp.asarray([1.0, 2.0, 3.0], dtype=xp.float64)

    with pytest.raises(AssertionError, match=r"Namespaces .* do not match"):
        assert_array_equal(actual, desired)


def test_assert_allclose_accepts_zero_dimensional_tolerances():
    xp = pytest.importorskip("array_api_strict")
    actual = xp.asarray([1.0, 2.0, 3.0], dtype=xp.float64)
    desired = xp.asarray([1.0, 2.0, 3.0], dtype=xp.float64)

    assert_allclose(
        actual,
        desired,
        rtol=xp.asarray(1e-12, dtype=xp.float64),
        atol=xp.asarray(1e-12, dtype=xp.float64),
    )


@pytest.mark.parametrize("desired", [True, 1, 1.0, 1.0 + 2.0j])
def test_assert_array_equal_accepts_python_numeric_scalar_expected(desired):
    actual = np.asarray(desired)

    assert_array_equal(actual, desired)


def test_assert_array_equal_accepts_numpy_scalar_expected():
    actual = np.asarray(1.0)

    assert_array_equal(actual, np.float64(1.0))


def test_assert_array_equal_accepts_strict_zero_dimensional_array():
    xp = pytest.importorskip("array_api_strict")
    actual = xp.asarray(1.0, dtype=xp.float64)

    assert_array_equal(actual, 1.0)
    assert_array_equal(actual, np.float64(1.0))


@pytest.mark.parametrize("actual", [1.0, np.float64(1.0), [1.0]])
def test_array_api_assertions_reject_non_array_actual_values(actual):
    with pytest.raises(AssertionError, match="'actual' to be an array"):
        assert_array_equal(actual, 1.0)


def test_array_api_assertions_reject_masked_array_actual():
    actual = np.ma.array([1.0, 2.0], mask=[False, True])

    with pytest.raises(AssertionError, match="ndarray subclasses"):
        assert_array_equal(actual, np.array([1.0, 2.0]))


def test_array_api_assertions_reject_masked_array_expected():
    actual = np.array([1.0, 2.0])
    desired = np.ma.array([1.0, 2.0], mask=[False, True])

    with pytest.raises(AssertionError, match="ndarray subclasses"):
        assert_array_equal(actual, desired)


def test_array_api_assertions_reject_unsupported_expected_value():
    actual = np.array([1.0, 2.0])

    with pytest.raises(AssertionError, match="Expected values must be"):
        assert_array_equal(actual, object())


def _is_array_api_strict_numpy_1_copy_error(exc):
    if int(np.__version__.split(".", maxsplit=1)[0]) >= 2:
        return False
    if str(exc) != "asarray() got an unexpected keyword argument 'copy'":
        return False

    traceback = exc.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        if (
            frame.f_globals.get("__name__") == "array_api_strict._array_object"
            and frame.f_code.co_name == "to_device"
        ):
            return True
        traceback = traceback.tb_next

    return False


def test_assert_array_equal_documents_array_api_strict_non_default_device_edge_case():
    xp = pytest.importorskip("array_api_strict")
    actual = xp.asarray([1.0, 2.0, 3.0], device=xp.Device("device1"))

    try:
        assert_array_equal(actual, [1.0, 2.0, 3.0])
    except TypeError as exc:
        if _is_array_api_strict_numpy_1_copy_error(exc):
            pytest.xfail(
                "array-api-strict calls "
                "np.asarray(copy=True) while moving to CPU; NumPy 1.x does not "
                "support that keyword (confirmed with strict 2.1.3 through 2.6)"
            )
        raise
