"""Binomial tree: convergence to Black-Scholes and American-exercise invariants."""

import pytest

from derivflow.core.pricing_engine import BinomialTree


def test_converges_to_black_scholes(atm_params, bs_ref):
    tree = BinomialTree(steps=500)
    for ot in ('call', 'put'):
        price = tree.price(option_type=ot, **atm_params)
        assert price == pytest.approx(bs_ref(option_type=ot, **atm_params), abs=2e-2)


def test_american_call_equals_european_no_dividend(atm_params):
    # Without dividends, an American call should not be exercised early.
    tree = BinomialTree(steps=300)
    eu = tree.price(option_type='call', american=False, **atm_params)
    am = tree.price(option_type='call', american=True, **atm_params)
    assert am == pytest.approx(eu, abs=1e-2)


def test_american_put_premium_nonnegative(atm_params):
    tree = BinomialTree(steps=300)
    eu = tree.price(option_type='put', american=False, **atm_params)
    am = tree.price(option_type='put', american=True, **atm_params)
    assert am >= eu - 1e-6
