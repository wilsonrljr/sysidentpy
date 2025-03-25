import pytest
import warnings
from sysidentpy.utils.deprecation import deprecated


# Define a sample function and class using the deprecated decorator
@deprecated(version="1.0", future_version="2.0", alternative="new_function")
def old_function():
    return "This is a deprecated function."


@deprecated(version="1.0", future_version="2.0", alternative="NewClass")
class OldClass:
    def __init__(self):
        self.value = "This is a deprecated class."


def test_deprecated_function():
    """Test that calling a deprecated function raises a FutureWarning."""
    with pytest.warns(
        FutureWarning, match="Function old_function has been deprecated since 1.0"
    ):
        result = old_function()
    assert result == "This is a deprecated function."


def test_deprecated_class():
    """Test that instantiating a deprecated class raises a FutureWarning."""
    with pytest.warns(
        FutureWarning, match="Class OldClass has been deprecated since 1.0"
    ):
        obj = OldClass()
    assert obj.value == "This is a deprecated class."
