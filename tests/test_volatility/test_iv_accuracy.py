"""Implied volatility: round trips across moneyness and short expiries.

Prices come from an independent Black-Scholes-Merton oracle (the
``bs_ref`` fixture). Where the option's vega is tiny the volatility is poorly
determined by the price, so the volatility round trip is only asserted when
vega > 1e-6 * S; the price round trip is asserted for every solved case.
"""

import itertools

import numpy as np
import pytest
from scipy.stats import norm

from derivflow_finance.volatility.implied_vol import (
    implied_volatility,
    implied_volatility_vectorized,
)

S, R, Q = 100.0, 0.03, 0.01
GRID = list(
    itertools.product(
        [1 / 365, 7 / 365, 30 / 365, 0.25, 1.0, 5.0],  # T
        [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 2.0],  # K / S
        [0.05, 0.2, 0.5, 1.5],  # sigma
        ["call", "put"],
    )
)


def _vega(K, T, sigma):
    F = S * np.exp((R - Q) * T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    return S * np.exp(-Q * T) * norm.pdf(d1) * np.sqrt(T)


def _time_value(bs_ref, K, T, sigma):
    F = S * np.exp((R - Q) * T)
    otm = "call" if K >= F else "put"
    return bs_ref(S, K, T, R, sigma, otm, Q)


@pytest.mark.parametrize(("T", "m", "sigma", "ot"), GRID)
def test_round_trip(bs_ref, T, m, sigma, ot):
    K = S * m
    price = bs_ref(S, K, T, R, sigma, ot, Q)
    # Time value below ~1e-12 of the option price cannot be resolved in double
    # precision: the volatility is not identifiable and a clear error is raised.
    if _time_value(bs_ref, K, T, sigma) <= 1e-12 * max(price, 1e-300):
        with pytest.raises(ValueError, match=r"identif|intrinsic|time value"):
            implied_volatility(price, S, K, T, R, ot, Q)
        return
    iv = implied_volatility(price, S, K, T, R, ot, Q)
    assert bs_ref(S, K, T, R, iv, ot, Q) == pytest.approx(price, abs=1e-11 * S)
    if _vega(K, T, sigma) > 1e-6 * S:
        assert iv == pytest.approx(sigma, abs=1e-9)


def test_low_vega_case_is_accurate():
    """Deep in the money, one day to expiry, high volatility: vega is small, so
    a price-tolerance stopping rule loses accuracy in volatility."""
    from derivflow_finance.core.pricing_engine import black_scholes_price

    price = black_scholes_price(100.0, 70.0, 1 / 365, 0.03, 1.5, "call", 0.01)
    iv = implied_volatility(price, 100.0, 70.0, 1 / 365, 0.03, "call", 0.01)
    assert iv == pytest.approx(1.5, abs=1e-8)


def test_vectorised_matches_scalar(bs_ref):
    """The array solver agrees with the scalar one.

    The two evaluate the price with NumPy and the math module respectively,
    whose exp and log may round differently in the last bit (SIMD builds of
    NumPy do). A price difference of a few ULP moves the volatility by about
    that amount divided by vega, so the allowed difference scales with 1/vega,
    and both solutions must reprice the option.
    """
    T, m, sig, ot = (np.array(x) for x in zip(*GRID, strict=True))
    K = S * m
    prices = np.array(
        [bs_ref(S, k, t, R, s, o, Q) for k, t, s, o in zip(K, T, sig, ot, strict=True)]
    )
    vec = implied_volatility_vectorized(prices, S, K, T, R, ot, Q)
    eps = np.finfo(float).eps
    for n in range(len(prices)):
        try:
            ref = implied_volatility(prices[n], S, K[n], T[n], R, ot[n], Q)
        except ValueError:
            assert np.isnan(vec[n])
            continue
        allowed = 1e-12 + 64 * eps * S / _vega(K[n], T[n], sig[n])
        assert vec[n] == pytest.approx(ref, abs=allowed)
        assert bs_ref(S, K[n], T[n], R, vec[n], ot[n], Q) == pytest.approx(prices[n], abs=1e-11 * S)


@pytest.mark.parametrize(
    ("price", "K", "T", "ot", "match"),
    [
        (100.0, 100.0, 1.0, "call", "upper bound"),  # call worth the whole (discounted) spot
        (1.0, 80.0, 1.0, "call", "intrinsic"),  # below the forward intrinsic value
        (5.0, 100.0, 0.0, "call", "T"),
        (5.0, -1.0, 1.0, "put", "K"),
    ],
)
def test_no_solution_raises_a_clear_error(price, K, T, ot, match):
    with pytest.raises(ValueError, match=match):
        implied_volatility(price, S, K, T, R, ot, Q)
    assert np.isnan(implied_volatility(price, S, K, T, R, ot, Q, on_fail="nan"))
