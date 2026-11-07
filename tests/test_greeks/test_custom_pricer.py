"""Numerical Greeks with a user-supplied pricing function."""

import pytest

from derivflow_finance import GreeksCalculator, black_scholes_price


def test_dividend_yield_reaches_a_custom_pricing_function():
    received = []

    def pricer(S, K, T, r, sigma, option_type="call", q=0.0):
        received.append(q)
        return black_scholes_price(S, K, T, r, sigma, option_type, q)

    num = GreeksCalculator("numerical").calculate_greeks(
        100, 100, 1.0, 0.05, 0.2, "call", pricing_func=pricer, q=0.04
    )
    ana = GreeksCalculator().calculate_greeks(100, 100, 1.0, 0.05, 0.2, "call", q=0.04)
    assert set(received) == {0.04}
    assert num.delta == pytest.approx(ana.delta, abs=1e-4)
    assert num.vega == pytest.approx(ana.vega, abs=1e-4)


def test_a_dividend_yield_for_a_pricer_without_q_is_an_error_not_ignored():
    def pricer(S, K, T, r, sigma, option_type="call"):
        return black_scholes_price(S, K, T, r, sigma, option_type)

    with pytest.raises(TypeError, match="q"):
        GreeksCalculator("numerical").calculate_greeks(
            100, 100, 1.0, 0.05, 0.2, "call", pricing_func=pricer, q=0.04
        )
    # without a dividend yield such a pricer works as before
    res = GreeksCalculator("numerical").calculate_greeks(
        100, 100, 1.0, 0.05, 0.2, "call", pricing_func=pricer
    )
    assert res.delta == pytest.approx(
        GreeksCalculator().calculate_greeks(100, 100, 1.0, 0.05, 0.2).delta, abs=1e-4
    )


@pytest.mark.parametrize(
    ("S", "option_type"), [(110, "call"), (90, "call"), (90, "put"), (110, "put")]
)
def test_numerical_greeks_at_expiry_match_the_analytical_ones(S, option_type):
    """At T = 0 the numerical Greeks are those of the intrinsic value, like
    the analytical ones, instead of dividing by a zero time step."""
    num = GreeksCalculator("numerical").calculate_greeks(S, 100, 0.0, 0.05, 0.2, option_type)
    ana = GreeksCalculator().calculate_greeks(S, 100, 0.0, 0.05, 0.2, option_type)
    for greek in ("delta", "gamma", "theta", "vega", "rho"):
        assert getattr(num, greek) == pytest.approx(getattr(ana, greek), abs=1e-12), greek
