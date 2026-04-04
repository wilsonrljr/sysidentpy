"""Global configuration for SysIdentPy."""

import threading
from contextlib import contextmanager

_global_config = {
    "array_api_dispatch": False,
}

_thread_local = threading.local()


def _get_threadlocal_config():
    """Get the thread-local config, falling back to global."""
    if not hasattr(_thread_local, "global_config"):
        _thread_local.global_config = _global_config.copy()
    return _thread_local.global_config


def get_config():
    """Retrieve current values for configuration set by :func:`set_config`.

    Returns
    -------
    config : dict
        Keys are parameter names that can be passed to :func:`set_config`.

    Examples
    --------
    >>> from sysidentpy import get_config
    >>> get_config()
    {'array_api_dispatch': False}
    """
    return _get_threadlocal_config().copy()


def set_config(*, array_api_dispatch=None):
    """Set global SysIdentPy configuration.

    Parameters
    ----------
    array_api_dispatch : bool, optional
        If True, SysIdentPy uses the Array API standard to dispatch
        array operations to the namespace inferred from the input arrays.
        This allows using CuPy, PyTorch, JAX, and other Array API
        compatible backends. Default is False (use NumPy).

    Examples
    --------
    >>> from sysidentpy import set_config
    >>> set_config(array_api_dispatch=True)
    """
    local_config = _get_threadlocal_config()
    if array_api_dispatch is not None:
        local_config["array_api_dispatch"] = array_api_dispatch


@contextmanager
def config_context(*, array_api_dispatch=None):
    """Context manager for temporary SysIdentPy configuration.

    Parameters
    ----------
    array_api_dispatch : bool, optional
        If True, enables Array API dispatch within the context.

    Yields
    ------
    None

    Examples
    --------
    >>> from sysidentpy import config_context
    >>> with config_context(array_api_dispatch=True):
    ...     pass  # Array API dispatch is enabled here
    """
    old_config = get_config()
    set_config(array_api_dispatch=array_api_dispatch)
    try:
        yield
    finally:
        set_config(**old_config)
