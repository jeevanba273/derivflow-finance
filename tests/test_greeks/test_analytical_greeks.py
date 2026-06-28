"""Analytical Greeks: known values, put-call relations, and scaling conventions."""

import numpy as np
import pytest
import scipy.stats as stats

from derivflow.greeks.calculator import GreeksCalculator

calc = GreeksCalculator('analytical')


def test_atm_known_values(atm_params):
    g = calc.calculate_greeks(option_type='call', **atm_params)
    assert g.delta == pytest.approx(0.6368, abs=1e-3)
    assert g.gamma == pytest.approx(0.0188, abs=1e-3)
    assert g.vega == pytest.approx(0.3752, abs=1e-3)


def test_put_call_delta_relation(atm_params):
    gc = calc.calculate_greeks(option_type='call', **atm_params)
    gp = calc.calculate_greeks(option_type='put', **atm_params)
    # With no dividends, delta_call - delta_put == 1.
    assert gc.delta - gp.delta == pytest.approx(1.0, abs=1e-9)


def test_gamma_vega_identical_call_put(atm_params):
    gc = calc.calculate_greeks(option_type='call', **atm_params)
    gp = calc.calculate_greeks(option_type='put', **atm_params)
    assert gc.gamma == pytest.approx(gp.gamma, abs=1e-12)
    assert gc.vega == pytest.approx(gp.vega, abs=1e-12)


def test_vega_scaling_convention(atm_params):
    """Vega is reported per 1% vol change: S*phi(d1)*sqrt(T)/100."""
    S, K, T, r, sigma = (atm_params[k] for k in ('S', 'K', 'T', 'r', 'sigma'))
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    expected = S * stats.norm.pdf(d1) * np.sqrt(T) / 100
    g = calc.calculate_greeks(option_type='call', **atm_params)
    assert g.vega == pytest.approx(expected, abs=1e-9)


def test_theta_negative_for_long_option(atm_params):
    g = calc.calculate_greeks(option_type='call', **atm_params)
    assert g.theta < 0  # time decay (per day)


def test_dividend_yield_reduces_call_delta(atm_params):
    g0 = calc.calculate_greeks(option_type='call', q=0.0, **atm_params)
    gq = calc.calculate_greeks(option_type='call', q=0.05, **atm_params)
    assert gq.delta < g0.delta  # discounted by exp(-qT)
