"""Cross-check against py_vollib / vollib, which implement Jaeckel's
"Let's Be Rational" (2015) implied volatility algorithm (skipped when not
installed)."""

import itertools
import warnings

import numpy as np
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)  # py_vollib points to vollib
    try:
        from vollib.black_scholes_merton import black_scholes_merton as lbr_price
        from vollib.black_scholes_merton.implied_volatility import (
            implied_volatility as lbr_iv,
        )
    except ImportError:
        pytest.importorskip("py_vollib.black_scholes_merton")
        from py_vollib.black_scholes_merton import black_scholes_merton as lbr_price
        from py_vollib.black_scholes_merton.implied_volatility import (
            implied_volatility as lbr_iv,
        )

from derivflow import black_scholes_price, implied_volatility

S, R, Q = 100.0, 0.03, 0.01
CASES = list(
    itertools.product(
        [7 / 365, 30 / 365, 0.25, 1.0, 3.0],
        [70.0, 90.0, 100.0, 110.0, 140.0],
        [0.1, 0.25, 0.6],
        ["call", "put"],
    )
)


@pytest.mark.parametrize(("T", "K", "sigma", "ot"), CASES)
def test_prices_agree(T, K, sigma, ot):
    ours = black_scholes_price(S, K, T, R, sigma, ot, Q)
    ref = lbr_price(ot[0], S, K, T, R, sigma, Q)
    assert ours == pytest.approx(ref, abs=1e-10)


@pytest.mark.parametrize(("T", "K", "sigma", "ot"), CASES)
def test_implied_volatilities_agree(T, K, sigma, ot):
    price = black_scholes_price(S, K, T, R, sigma, ot, Q)
    F = S * np.exp((R - Q) * T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    vega = S * np.exp(-Q * T) * np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi) * np.sqrt(T)
    if vega < 1e-6 * S:
        pytest.skip("volatility poorly determined by the price (tiny vega)")
    ours = implied_volatility(price, S, K, T, R, ot, Q)
    ref = lbr_iv(price, S, K, T, R, Q, ot[0])
    assert ours == pytest.approx(ref, abs=1e-9)
    assert ours == pytest.approx(sigma, abs=1e-9)
