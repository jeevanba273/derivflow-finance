"""GreeksCalculator helpers: portfolio aggregation, ladders and the report.

A long call and a short put with the same strike and expiry are a forward,
S e^{-qT} - K e^{-rT}, whose Greeks are known in closed form: delta e^{-qT},
no gamma or vega, theta (q S e^{-qT} - r K e^{-rT}) / 365 per calendar day and
rho T K e^{-rT} / 100 per rate point.
"""

import math

import numpy as np
import pytest

from derivflow_finance import GreeksCalculator
from derivflow_finance.greeks.calculator import format_greeks_report

S, K, T, R, Q, SIGMA = 100.0, 95.0, 0.75, 0.04, 0.02, 0.25


def test_call_minus_put_has_the_greeks_of_a_forward():
    calc = GreeksCalculator()
    call = calc.calculate_greeks(S, K, T, R, SIGMA, "call", q=Q)
    put = calc.calculate_greeks(S, K, T, R, SIGMA, "put", q=Q)
    book = calc.portfolio_greeks([{"quantity": 1, "greeks": call}, {"quantity": -1, "greeks": put}])
    assert book["delta"] == pytest.approx(math.exp(-Q * T), abs=1e-12)
    assert book["gamma"] == pytest.approx(0.0, abs=1e-12)
    assert book["vega"] == pytest.approx(0.0, abs=1e-12)
    theta = (Q * S * math.exp(-Q * T) - R * K * math.exp(-R * T)) / 365
    assert book["theta"] == pytest.approx(theta, abs=1e-12)
    assert book["rho"] == pytest.approx(T * K * math.exp(-R * T) / 100, abs=1e-12)
    assert book["total_notional"] == 2.0


def test_portfolio_greeks_weight_by_quantity_and_notional():
    calc = GreeksCalculator()
    g = calc.calculate_greeks(S, K, T, R, SIGMA, "call", q=Q)
    book = calc.portfolio_greeks(
        [{"quantity": 3, "greeks": g, "notional": 100.0}, {"quantity": -2, "greeks": g}]
    )
    for greek in ("delta", "gamma", "theta", "vega", "rho"):
        assert book[greek] == pytest.approx(298 * getattr(g, greek), rel=1e-12), greek
    assert book["total_notional"] == 302.0
    assert calc.portfolio_greeks([])["delta"] == 0.0


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_greeks_ladder_matches_the_greeks_at_each_spot(option_type):
    calc = GreeksCalculator()
    ladder = calc.greeks_ladder(S, K, T, R, SIGMA, (0.7, 1.3), 13, option_type, q=Q)
    spots = ladder["spot_prices"]
    assert spots.shape == (13,)
    assert spots[0] == pytest.approx(70.0) and spots[-1] == pytest.approx(130.0)
    for i, spot in enumerate(spots):
        g = calc.analytical_greeks(spot, K, T, R, SIGMA, option_type, Q)
        for greek in ("delta", "gamma", "theta", "vega", "rho"):
            assert ladder[greek][i] == getattr(g, greek), (greek, spot)
    assert np.all(np.diff(ladder["delta"]) > 0)  # delta rises with spot for calls and puts
    assert np.all(ladder["gamma"] > 0)


@pytest.mark.parametrize(
    ("spot", "option_type", "label"),
    [
        (100.0, "call", "ATM"),
        (101.5, "put", "ATM"),
        (110.0, "call", "ITM"),
        (90.0, "put", "ITM"),
        (90.0, "call", "OTM"),
        (110.0, "put", "OTM"),
    ],
)
def test_report_moneyness_and_units(spot, option_type, label):
    g = GreeksCalculator().calculate_greeks(spot, 100.0, T, R, SIGMA, option_type)
    report = format_greeks_report(g, spot, 100.0, option_type)
    first = report.splitlines()[0]
    assert first.startswith(option_type.upper()) and label in first
    assert "per calendar day" in report and "per vol point" in report
    assert "volga" in report  # analytical Greeks include the higher orders


def test_report_omits_missing_higher_order_greeks():
    g = GreeksCalculator("numerical").calculate_greeks(S, K, T, R, SIGMA)
    report = format_greeks_report(g, S, K)
    assert len(report.splitlines()) == 6 and "volga" not in report


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="method"):
        GreeksCalculator("monte_carlo")
