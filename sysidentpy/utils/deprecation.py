# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause

import functools
import warnings


def deprecated(version, future_version=None, message=None, alternative=None, **kwargs):
    """Decorate deprecated methods.

    This decorator is adapted from astroML decorator:
    https://github.com/astroML/astroML/blob/f66558232f6d33cb34ecd1bed8a80b9db7ae1c30/astroML/utils/decorators.py#L120

    """

    def deprecate_function(
        func,
        version=version,
        future_version=future_version,
        message=message,
        alternative=alternative,
    ):
        if message is None:
            message = f"Function {func.__name__} has been deprecated since {version}."
            if alternative is not None:
                message += (
                    f"\n You'll have to use {alternative} instead."
                    "This module was deprecated in favor of "
                    f"{alternative} module into which all the refactored "
                    "classes and functions are moved."
                )
            if future_version is not None:
                message += (
                    f"\n This change will be applied in version {future_version}."
                )

        @functools.wraps(func)
        def deprecated_func(*args, **kwargs):
            warnings.warn(message, FutureWarning, stacklevel=1)
            return func(*args, **kwargs)

        return deprecated_func

    def deprecate_class(
        cls,
        version=version,
        future_version=future_version,
        message=message,
        alternative=alternative,
    ):
        if message is None:
            message = f"Class {cls.__name__} has been deprecated since {version}."
            if alternative is not None:
                message += alternative
            if future_version is not None:
                message += (
                    f"\n This change will be applied in version {future_version}."
                )

        cls.__init__ = deprecate_function(cls.__init__, message=message)

        return cls

    def deprecate_warning(obj):
        if isinstance(obj, type):
            return deprecate_class(obj)

        return deprecate_function(obj)

    return deprecate_warning
