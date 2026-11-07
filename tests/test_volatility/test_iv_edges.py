"""Implied volatility: failure paths, solver options and option-chain inputs."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from derivflow_finance import black_scholes_price
from derivflow_finance.volatility.implied_vol import (
    implied_vol_from_chain,
    implied_volatility,
    implied_volatility_vectorized,
)

PRICE = black_scholes_price(100, 100, 1.0, 0.05, 0.2, "call")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"option_type": "straddle"}, "option_type"),
        ({"S": 0.0}, "S must"),
        ({"S": math.nan}, "S must"),
        ({"K": -1.0}, "K must"),
        ({"price": None}, "price must"),
        ({"price": math.nan}, "price must"),
        ({"price": math.inf}, "price must"),
    ],
)
def test_invalid_inputs_raise_or_return_nan(kwargs, match):
    args = {"price": PRICE, "S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05} | kwargs
    with pytest.raises(ValueError, match=match):
        implied_volatility(**args)
    assert math.isnan(implied_volatility(**args, on_fail="nan"))


def test_scalar_rejects_an_unknown_on_fail():
    with pytest.raises(ValueError, match="on_fail"):
        implied_volatility(PRICE, 100, 100, 1.0, 0.05, on_fail="warn")


def test_volatility_above_the_cap_has_no_solution():
    # One hour to expiry, a call worth 99.9% of spot: sigma would exceed 10,000%.
    with pytest.raises(ValueError, match="above"):
        implied_volatility(99.9, 100, 100, 1e-4, 0.0, "call")


@pytest.mark.parametrize("guess", [0.5, 1e6, 1e-9])
def test_initial_guess_inside_or_outside_the_bracket_converges(guess):
    assert implied_volatility(PRICE, 100, 100, 1.0, 0.05, initial_guess=guess) == pytest.approx(
        0.2, abs=1e-12
    )


def test_iteration_limit():
    with pytest.raises(ValueError, match="no convergence"):
        implied_volatility(PRICE, 100, 100, 1.0, 0.05, max_iter=1)
    assert math.isnan(implied_volatility(PRICE, 100, 100, 1.0, 0.05, max_iter=1, on_fail="nan"))


def test_vectorized_broadcasting_and_errors():
    K = np.array([[90.0], [110.0]])
    T = np.array([0.25, 0.5, 1.0])
    prices = black_scholes_price(100.0, K, T, 0.03, 0.3, "put", 0.01)
    ivs = implied_volatility_vectorized(prices, 100.0, K, T, 0.03, "put", 0.01)
    assert ivs.shape == (2, 3)
    assert np.allclose(ivs, 0.3, atol=1e-10)
    with pytest.raises(ValueError, match="option_type"):
        implied_volatility_vectorized([PRICE], 100, 100, 1.0, 0.05, "straddle")
    with pytest.raises(ValueError, match="upper bound"):
        implied_volatility_vectorized([PRICE, 500.0], 100, 100, 1.0, 0.05, on_fail="raise")
    assert np.all(np.isnan(implied_volatility_vectorized([-1.0, 0.0], 100, 100, 1.0, 0.05)))


def _option(**kw):
    base = {"strike": 100.0, "option_type": "call", "last_price": None, "bid": None, "ask": None}
    return SimpleNamespace(**(base | kw))


def test_chain_from_a_market_data_result():
    """MarketDataResult-like input: date-string expiries are converted with
    ACT/365 to 16:00 New York; rows without a usable price or expiry are
    skipped."""
    from derivflow_finance._dates import new_york, year_fraction_to_expiry

    try:
        new_york()
    except Exception:  # pragma: no cover - no IANA time-zone data
        pytest.skip("needs the America/New_York time zone (tzdata)")
    expiry = "2099-12-18"
    T = year_fraction_to_expiry(expiry)
    mid = black_scholes_price(100, 100, T, 0.03, 0.25, "call")
    chain = SimpleNamespace(
        options_data=[
            _option(expiry=expiry, bid=mid * 0.99, ask=mid * 1.01, last_price=0.5),
            _option(expiry=0.5, last_price=black_scholes_price(100, 100, 0.5, 0.03, 0.4, "call")),
            _option(expiry=expiry, last_price=0.0),  # no price: skipped
            _option(expiry="not a date", last_price=5.0),  # unreadable expiry: skipped
            _option(expiry=-0.1, last_price=5.0),  # expired: skipped
        ]
    )
    out = implied_vol_from_chain(chain, S=100, r=0.03)
    assert len(out) == 2
    assert out[0]["expiry"] == pytest.approx(T)
    assert out[0]["market_price"] == pytest.approx(mid)
    assert out[0]["volatility"] == pytest.approx(0.25, abs=1e-9)
    assert out[1]["volatility"] == pytest.approx(0.4, abs=1e-9)


def test_chain_mid_falls_back_to_last_price_without_a_two_sided_quote():
    last = black_scholes_price(100, 100, 0.5, 0.03, 0.3, "call")
    rows = [
        {
            "strike": 100,
            "expiry": 0.5,
            "option_type": "call",
            "bid": None,
            "ask": 9.0,
            "last_price": last,
        },
        {
            "strike": 100,
            "expiry": 0.5,
            "option_type": "call",
            "bid": "n/a",
            "ask": 9.0,
            "last_price": last,
        },
        {
            "strike": 100,
            "expiry": 0.5,
            "option_type": "call",
            "bid": 0.0,
            "ask": 0.0,
            "last_price": None,
        },
    ]
    out = implied_vol_from_chain(rows, S=100, r=0.03)
    assert len(out) == 2  # the row without any price is skipped
    assert all(row["volatility"] == pytest.approx(0.3, abs=1e-9) for row in out)


def test_chain_from_a_dataframe():
    pd = pytest.importorskip("pandas")
    T = np.array([0.25, 1.0])
    df = pd.DataFrame(
        {
            "strike": [95.0, 105.0],
            "expiry": T,
            "option_type": ["put", "call"],
            "market_price": [
                black_scholes_price(100, 95, 0.25, 0.03, 0.2, "put"),
                black_scholes_price(100, 105, 1.0, 0.03, 0.35, "call"),
            ],
        }
    )
    out = implied_vol_from_chain(df, S=100, r=0.03)
    assert [round(row["volatility"], 9) for row in out] == [0.2, 0.35]


def test_chain_rejects_an_unknown_price_field():
    with pytest.raises(ValueError, match="price_field"):
        implied_vol_from_chain([], S=100, r=0.03, price_field="close")
