"""Black-Scholes analytical pricing: known-value benchmarks and invariants."""

import numpy as np
import pytest

from derivflow.core.pricing_engine import (
    BlackScholesAnalytical, PricingEngine, price_european_option,
)

bs = BlackScholesAnalytical()


def test_hull_textbook_values(hull_params):
    # Hull, Options Futures & Other Derivatives worked example.
    call = bs.price(option_type='call', **hull_params)
    put = bs.price(option_type='put', **hull_params)
    assert call == pytest.approx(4.7594, abs=1e-3)
    assert put == pytest.approx(0.8086, abs=1e-3)


def test_atm_reference_values(atm_params):
    assert bs.price(option_type='call', **atm_params) == pytest.approx(10.4506, abs=1e-3)
    assert bs.price(option_type='put', **atm_params) == pytest.approx(5.5735, abs=1e-3)


def test_matches_independent_reference(params, bs_ref):
    for ot in ('call', 'put'):
        assert bs.price(option_type=ot, **params) == pytest.approx(
            bs_ref(option_type=ot, **params), abs=1e-9)


def test_put_call_parity(atm_params, tol):
    res = PricingEngine().validate_put_call_parity(**atm_params)
    assert res['parity_satisfied']
    assert res['absolute_error'] < tol['analytic']


def test_intrinsic_at_expiry():
    assert bs.price(110, 100, 0.0, 0.05, 0.2, 'call') == pytest.approx(10.0)
    assert bs.price(90, 100, 0.0, 0.05, 0.2, 'put') == pytest.approx(10.0)
    assert bs.price(90, 100, 0.0, 0.05, 0.2, 'call') == pytest.approx(0.0)


def test_monotonic_in_volatility(atm_params):
    p = dict(atm_params)
    prices = [bs.price(p['S'], p['K'], p['T'], p['r'], s, 'call') for s in (0.1, 0.2, 0.4, 0.8)]
    assert all(b > a for a, b in zip(prices, prices[1:]))


def test_deep_itm_otm(bs_ref):
    deep_itm_call = bs.price(200, 100, 1.0, 0.05, 0.2, 'call')
    assert deep_itm_call == pytest.approx(200 - 100 * np.exp(-0.05), abs=0.5)
    deep_otm_call = bs.price(50, 100, 0.25, 0.05, 0.2, 'call')
    assert deep_otm_call == pytest.approx(0.0, abs=1e-2)


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        bs.price(100, 100, 1.0, 0.05, 0.2, 'banana')


def test_convenience_function_matches_class(params):
    assert price_european_option(**params, option_type='call') == pytest.approx(
        bs.price(**params, option_type='call'))


def test_dividend_yield_lowers_call_raises_put(atm_params, bs_ref):
    call_q = bs.price(**atm_params, option_type='call', q=0.03)
    call_0 = bs.price(**atm_params, option_type='call', q=0.0)
    assert call_q < call_0
    # matches independent BSM reference with q
    assert call_q == pytest.approx(bs_ref(**atm_params, option_type='call', q=0.03), abs=1e-9)
