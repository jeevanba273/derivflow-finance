"""Numerical (finite-difference) Greeks vs analytical.

Both ``GreeksCalculator`` methods report in the same units (delta/gamma raw,
vega/rho per 1% move, theta per day), so delta/gamma/vega/rho compare directly.
Theta uses a per-year forward difference numerically vs a per-day analytical
value, so only the sign is asserted (a documented convention difference).
"""

import pytest

from derivflow.greeks.calculator import GreeksCalculator

analytical = GreeksCalculator('analytical')
numerical = GreeksCalculator('numerical')


def test_first_order_greeks_match(atm_params, tol):
    ga = analytical.calculate_greeks(option_type='call', **atm_params)
    gn = numerical.calculate_greeks(option_type='call', **atm_params)
    assert gn.delta == pytest.approx(ga.delta, abs=tol['greek_fd'])
    assert gn.gamma == pytest.approx(ga.gamma, abs=tol['greek_fd'])
    assert gn.vega == pytest.approx(ga.vega, rel=1e-2)
    assert gn.rho == pytest.approx(ga.rho, rel=1e-2)


def test_put_greeks_match(atm_params, tol):
    ga = analytical.calculate_greeks(option_type='put', **atm_params)
    gn = numerical.calculate_greeks(option_type='put', **atm_params)
    assert gn.delta == pytest.approx(ga.delta, abs=tol['greek_fd'])
    assert gn.vega == pytest.approx(ga.vega, rel=1e-2)


def test_theta_sign_agrees(atm_params):
    ga = analytical.calculate_greeks(option_type='call', **atm_params)
    gn = numerical.calculate_greeks(option_type='call', **atm_params)
    assert (ga.theta < 0) == (gn.theta < 0)
