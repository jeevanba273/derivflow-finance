"""Implied volatility solver: round-trip accuracy, bounds, and chain inversion."""

import numpy as np
import pytest

from derivflow_finance.core.pricing_engine import price_european_option
from derivflow_finance.volatility.implied_vol import (
    implied_vol_from_chain,
    implied_volatility,
    implied_volatility_vectorized,
)


@pytest.mark.parametrize("sigma", [0.10, 0.20, 0.35, 0.60])
@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("K", [90, 100, 110])
def test_round_trip(sigma, option_type, K):
    price = price_european_option(100, K, 0.5, 0.05, sigma, option_type)
    iv = implied_volatility(price, 100, K, 0.5, 0.05, option_type)
    assert iv == pytest.approx(sigma, abs=1e-6)


def test_round_trip_with_dividend():
    price = price_european_option(100, 100, 1.0, 0.05, 0.3, "call", q=0.04)
    iv = implied_volatility(price, 100, 100, 1.0, 0.05, "call", q=0.04)
    assert iv == pytest.approx(0.3, abs=1e-6)


def test_price_above_upper_bound_raises_by_default():
    # A call cannot be worth more than the (dividend-adjusted) spot.
    with pytest.raises(ValueError, match="upper bound"):
        implied_volatility(1000, 100, 100, 1.0, 0.05, "call")
    assert np.isnan(implied_volatility(1000, 100, 100, 1.0, 0.05, "call", on_fail="nan"))


def test_out_of_bounds_raises_when_requested():
    with pytest.raises(ValueError):
        implied_volatility(-1.0, 100, 100, 1.0, 0.05, "call", on_fail="raise")


def test_expiry_zero_raises_by_default():
    with pytest.raises(ValueError, match="T must be positive"):
        implied_volatility(5.0, 100, 100, 0.0, 0.05, "call")
    assert np.isnan(implied_volatility(5.0, 100, 100, 0.0, 0.05, "call", on_fail="nan"))


def test_vectorized_solver():
    prices = np.array(
        [price_european_option(100, k, 0.5, 0.05, 0.25, "call") for k in (90, 100, 110)]
    )
    ivs = implied_volatility_vectorized(prices, 100, np.array([90, 100, 110]), 0.5, 0.05, "call")
    assert np.allclose(ivs, 0.25, atol=1e-6)


def test_from_chain_list_of_dicts():
    rows = [
        {
            "strike": 100,
            "expiry": 0.5,
            "option_type": "call",
            "market_price": price_european_option(100, 100, 0.5, 0.05, 0.22, "call"),
        }
    ]
    out = implied_vol_from_chain(rows, S=100, r=0.05)
    assert out[0]["volatility"] == pytest.approx(0.22, abs=1e-4)


def test_from_chain_honours_on_fail():
    rows = [{"strike": 60, "expiry": 0.5, "option_type": "call", "market_price": 1.0}]
    assert np.isnan(implied_vol_from_chain(rows, S=100, r=0.05)[0]["volatility"])
    with pytest.raises(ValueError, match="intrinsic"):
        implied_vol_from_chain(rows, S=100, r=0.05, on_fail="raise")


@pytest.mark.parametrize("field", ["bid", "ask", "last", "mid"])
def test_from_chain_uses_price_field_for_dict_rows(field):
    prices = {
        "bid": price_european_option(100, 100, 0.5, 0.05, 0.20, "call"),
        "ask": price_european_option(100, 100, 0.5, 0.05, 0.30, "call"),
        "last_price": price_european_option(100, 100, 0.5, 0.05, 0.40, "call"),
    }
    row = {"strike": 100, "expiry": 0.5, "option_type": "call", **prices}
    out = implied_vol_from_chain([row], S=100, r=0.05, price_field=field)[0]
    expected = {"bid": 0.20, "ask": 0.30, "last": 0.40}.get(field)
    if field == "mid":
        assert out["market_price"] == pytest.approx(0.5 * (prices["bid"] + prices["ask"]))
    else:
        assert out["volatility"] == pytest.approx(expected, abs=1e-9)
