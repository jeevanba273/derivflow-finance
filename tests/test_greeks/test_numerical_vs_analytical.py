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


def test_positional_pricing_func_backward_compatible():
    """Regression: pricing_func must remain the 8th positional arg (q is now
    keyword-only), so legacy positional callers don't bind a function to q."""
    gc = GreeksCalculator('numerical')
    # Passing the pricer positionally (legacy pattern) must not raise.
    g = gc.calculate_greeks(100, 105, 0.25, 0.05, 0.2, 'call', gc._black_scholes_price)
    assert g.delta == pytest.approx(0.3773, abs=1e-3)


def test_q_is_keyword_only():
    import pytest as _pytest
    gc = GreeksCalculator('analytical')
    # q must be supplied by keyword; positional past pricing_func is rejected.
    with _pytest.raises(TypeError):
        gc.calculate_greeks(100, 100, 1.0, 0.05, 0.2, 'call', None, 0.03)
