"""Barrier and Asian options."""

# Barrier Options
# Asian Options
from .asian_options import (
    AsianOptionResult,
    AsianOptions,
    AsianType,
    AverageType,
    compare_asian_types,
    price_asian_option,
)
from .barrier_options import BarrierOptionResult, BarrierOptions, BarrierType

# Convenience exports
__all__ = [
    # Barrier Options
    "BarrierOptions",
    "BarrierType",
    "BarrierOptionResult",
    # Asian Options
    "AsianOptions",
    "AsianType",
    "AverageType",
    "AsianOptionResult",
    "price_asian_option",
    "compare_asian_types",
    "price_barrier_option",
]


def price_barrier_option(
    S: float,
    K: float,
    H: float,
    T: float,
    r: float,
    sigma: float,
    barrier_type: str,
    option_type: str = "call",
) -> float:
    """Closed-form price of a continuously monitored single-barrier option with
    no rebate (see :meth:`BarrierOptions.analytical_price`)."""
    return BarrierOptions().price(S, K, H, T, r, sigma, barrier_type, option_type).price
