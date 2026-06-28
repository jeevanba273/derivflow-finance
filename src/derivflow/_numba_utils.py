"""
DERIVFLOW-FINANCE: Optional numba acceleration
==============================================

numba is an optional accelerator. This module exposes ``HAS_NUMBA`` plus
``njit``/``prange`` symbols that fall back to no-op decorators and the builtin
``range`` when numba is not installed, so the rest of the package can use a
single code path whether or not numba is available.
"""

try:
    from numba import njit, prange  # type: ignore
    HAS_NUMBA = True
except Exception:  # pragma: no cover - exercised only without numba installed
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        """No-op stand-in for numba.njit when numba is unavailable."""
        # Support both @njit and @njit(...) usage.
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(func):
            return func
        return decorator

    prange = range
