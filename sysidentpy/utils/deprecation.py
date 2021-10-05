# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause


from functools import wraps
import warnings
import inspect
import functools


def deprecated(version, future_version=None, message=None, alternative=None, **kwargs):
    """ This docorater is adapted from astroML decorator:
    https://github.com/astroML/astroML/blob/f66558232f6d33cb34ecd1bed8a80b9db7ae1c30/astroML/utils/decorators.py#L120
    """
    def deprecate_function(func, version=version, future_version=future_version, message=message,
                            alternative=alternative):
        if message is None:
            message = (f'Function {func.__name__} has been deprecated since {version}.')
            if alternative is not None:
                message += (f'\n Use {alternative} instead.'
                            f"This module was deprecated in favor of "
                            f"{alternative} module into which all the refactored "
                            f"classes and functions are moved.")
            if future_version is not None:
                message += f'\n This feature will be removed in version {future_version}.'
            
        @functools.wraps(func)
        def deprecated_func(*args, **kwargs):
            warnings.warn(message, FutureWarning)
            return func(*args, **kwargs)
        return deprecated_func

    def deprecate_class(cls, version=version, future_version=future_version, message=message,
                        alternative=alternative):
        if message is None:
            message = (f'Class {cls.__name__} has been deprecated since {version}.')
            if alternative is not None:
                message += (f'\n Use {alternative} instead.'
                            f"This module was deprecated in favor of "
                            f"{alternative} module into which all the refactored "
                            f"classes and functions are moved.")
            if future_version is not None:
                message += f'\n This feature will be removed in version {future_version}.'

        cls.__init__ = deprecate_function(cls.__init__, message=message)

        return cls

    def deprecate_warning(obj):
        if isinstance(obj, type):
            return deprecate_class(obj)
        else:
            return deprecate_function(obj)

    return deprecate_warning

def deprecated_method(since, message='', alternative=None, **kwargs):
        def deprecate_function(func, message=message, since=since,
                               alternative=alternative):
            if message == '':
                message = ('Function {} has been deprecated since {}.'
                           .format(func.__name__, since))
                if alternative is not None:
                    message += '\n Use {} instead.'.format(alternative)

            @functools.wraps(func)
            def deprecated_func(*args, **kwargs):
                warnings.warn(message, FutureWarning)
                return func(*args, **kwargs)
            return deprecated_func

        def deprecate_class(cls, message=message, since=since,
                            alternative=alternative):
            if message == '':
                message = ('Class {} has been deprecated since {}.'
                           .format(cls.__name__, since))
                if alternative is not None:
                    message += '\n Use {} instead.'.format(alternative)

            cls.__init__ = deprecate_function(cls.__init__, message=message)

            return cls

        def deprecate(obj):
            if isinstance(obj, type):
                return deprecate_class(obj)
            else:
                return deprecate_function(obj)

        return deprecate


def _deprecate_positional_args(func):
    """Decorator to mark a function or class as deprecated.
    """
    # @wraps(func)
    def wrapper(*args, **kwargs):
        # func(*args, **kwargs)
        warnings.warn(
                f"Pass arguments as keyword args. From version "
                f"v0.2.0 passing these as positional arguments "
                "will result in an error",
                FutureWarning,
            )
        func()

    return wrapper

def _deprecate_polynomial_narmax(func):
    """Decorator to mark a function or class as deprecated.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
        warnings.warn(
                "This module was deprecated in version 0.2 in favor of "
                "the model_structure_selection.FROLS module into which all the refactored "
                "classes and functions are moved.",
                FutureWarning,
            )

    return wrapper
