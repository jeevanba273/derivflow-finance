"""Input checks shared by the pricers. All raise ValueError naming the parameter."""

from __future__ import annotations

import math


def _positive(name: str, value: float, what: str) -> None:
    try:
        ok = math.isfinite(value) and value > 0
    except TypeError:
        ok = False
    if not ok:
        raise ValueError(f"{name} must be positive and finite ({what}), got {value!r}")


def check_option_type(option_type: str) -> str:
    """Return 'call' or 'put' (case-insensitive) or raise."""
    ot = str(option_type).lower()
    if ot not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    return ot


def check_market_inputs(S: float, K: float, T: float, sigma: float | None = None) -> None:
    """Spot and strike positive; T finite (T <= 0 means expired, handled by the
    caller); sigma positive whenever the option has not expired."""
    _positive("S", S, "spot price")
    _positive("K", K, "strike")
    try:
        t_ok = math.isfinite(T)
    except TypeError:
        t_ok = False
    if not t_ok:
        raise ValueError(f"T must be a finite time to expiry in years, got {T!r}")
    if sigma is not None and T > 0:
        _positive("sigma", sigma, "annualised volatility")


def check_live_inputs(S: float, K: float, T: float, sigma: float | None = None) -> None:
    """As :func:`check_market_inputs` but T must also be positive."""
    check_market_inputs(S, K, T, sigma)
    if T <= 0:
        raise ValueError(f"T must be positive (years to expiry), got {T!r}")
