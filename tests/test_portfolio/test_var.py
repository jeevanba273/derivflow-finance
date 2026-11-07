"""Portfolio VaR against closed-form delta-normal values.

Convention under test: volatilities are annualised and a horizon of h trading
days has standard deviation sigma * sqrt(h / 252). VaR and expected shortfall
are reported as positive losses.
"""

import numpy as np
import pytest
from scipy.stats import norm

from derivflow_finance.greeks.calculator import GreeksCalculator
from derivflow_finance.portfolio.portfolio_risk import PortfolioRiskAnalyzer

Z99 = norm.ppf(0.99)
ES99 = norm.pdf(Z99) / 0.01


def _stock_book(*positions, r=0.0):
    pa = PortfolioRiskAnalyzer(risk_free_rate=r)
    for symbol, qty, price, vol in positions:
        pa.add_stock_position(symbol, qty, price, vol)
    return pa


def test_single_stock_one_day_var_uses_daily_volatility():
    pa = _stock_book(("X", 100, 100.0, 0.20))
    var, es = pa.calculate_var_parametric(0.99, time_horizon=1)
    sigma_1d = 10_000 * 0.20 / np.sqrt(252)
    assert var == pytest.approx(Z99 * sigma_1d, rel=1e-12)  # 293.09
    assert es == pytest.approx(ES99 * sigma_1d, rel=1e-12)


def test_var_scales_with_square_root_of_horizon():
    pa = _stock_book(("X", 100, 100.0, 0.20))
    v1, _ = pa.calculate_var_parametric(0.99, time_horizon=1)
    v10, _ = pa.calculate_var_parametric(0.99, time_horizon=10)
    assert v10 == pytest.approx(v1 * np.sqrt(10), rel=1e-12)


def test_short_position_has_the_same_var_as_long():
    long_var, _ = _stock_book(("X", 100, 100.0, 0.2)).calculate_var_parametric(0.99)
    short_var, _ = _stock_book(("X", -100, 100.0, 0.2)).calculate_var_parametric(0.99)
    assert short_var == pytest.approx(long_var, rel=1e-12)


def test_offsetting_positions_in_one_underlying_have_zero_var():
    pa = _stock_book(("X", 100, 100.0, 0.2), ("X", -100, 100.0, 0.2))
    var, es = pa.calculate_var_parametric(0.99)
    assert var == pytest.approx(0.0, abs=1e-9)
    assert es == pytest.approx(0.0, abs=1e-9)


def test_option_exposure_is_delta_times_multiplier_times_spot():
    pa = PortfolioRiskAnalyzer(risk_free_rate=0.03)
    pa.add_option_position("X", 2, 100.0, 105.0, 0.5, "call", 0.25)
    delta = GreeksCalculator().calculate_greeks(100.0, 105.0, 0.5, 0.03, 0.25, "call").delta
    exposure = 2 * 100 * delta * 100.0
    var, _ = pa.calculate_var_parametric(0.99)
    assert var == pytest.approx(Z99 * abs(exposure) * 0.25 / np.sqrt(252), rel=1e-12)


@pytest.mark.parametrize("rho", [-0.5, 0.0, 0.8])
def test_two_assets_with_correlation(rho):
    pa = _stock_book(("A", 100, 50.0, 0.30), ("B", -40, 200.0, 0.20))
    pa.set_correlation(["A", "B"], [[1.0, rho], [rho, 1.0]])
    e = np.array([100 * 50.0, -40 * 200.0])
    s = np.array([0.30, 0.20]) / np.sqrt(252)
    sigma_p = np.sqrt((e[0] * s[0]) ** 2 + (e[1] * s[1]) ** 2 + 2 * rho * e[0] * e[1] * s[0] * s[1])
    var, es = pa.calculate_var_parametric(0.99)
    assert var == pytest.approx(Z99 * sigma_p, rel=1e-12)
    assert es == pytest.approx(ES99 * sigma_p, rel=1e-12)


def test_invalid_correlation_is_rejected():
    pa = _stock_book(("A", 1, 50.0, 0.3), ("B", 1, 50.0, 0.3))
    with pytest.raises(ValueError):
        pa.set_correlation(["A", "B"], [[1.0, 1.5], [1.5, 1.0]])
    with pytest.raises(ValueError):
        pa.set_correlation(["A", "B"], [[1.0, 0.2], [0.3, 1.0]])


# --- Monte Carlo VaR -----------------------------------------------------------
# Simulated log-returns over h trading days are N(-s^2/2, s^2), s = sigma sqrt(h/252)
# (zero expected price change), with one shared draw per underlying.


def test_monte_carlo_flat_book_has_zero_var():
    pa = _stock_book(("X", 100, 100.0, 0.2), ("X", -100, 100.0, 0.2))
    var, es = pa.calculate_var_monte_carlo(0.99, 1, num_sims=5_000, random_seed=1)
    assert var == pytest.approx(0.0, abs=1e-9)
    assert es == pytest.approx(0.0, abs=1e-9)


def test_monte_carlo_single_stock_matches_lognormal_quantile():
    pa = _stock_book(("X", 100, 100.0, 0.20))
    var, _ = pa.calculate_var_monte_carlo(0.99, 10, num_sims=400_000, random_seed=2)
    s = 0.20 * np.sqrt(10 / 252)
    exact = -10_000 * (np.exp(-0.5 * s * s + s * norm.ppf(0.01)) - 1.0)
    assert var == pytest.approx(exact, rel=1e-2)


def test_monte_carlo_long_call_var_uses_full_revaluation():
    from derivflow_finance.core.pricing_engine import black_scholes_price

    pa = PortfolioRiskAnalyzer(risk_free_rate=0.03)
    pa.add_option_position("X", 1, 100.0, 100.0, 0.25, "call", 0.30)
    h = 5
    var, _ = pa.calculate_var_monte_carlo(0.99, h, num_sims=400_000, random_seed=3)
    s = 0.30 * np.sqrt(h / 252)
    s_low = 100.0 * np.exp(-0.5 * s * s + s * norm.ppf(0.01))  # call value is increasing in S
    v0 = black_scholes_price(100.0, 100.0, 0.25, 0.03, 0.30)
    v_low = black_scholes_price(s_low, 100.0, 0.25 - h / 252, 0.03, 0.30)
    assert var == pytest.approx(100 * (v0 - v_low), rel=1e-2)


def test_monte_carlo_matches_parametric_for_a_linear_book():
    pa = _stock_book(("A", 100, 50.0, 0.30), ("B", 40, 200.0, 0.20))
    pa.set_correlation(["A", "B"], [[1.0, 0.6], [0.6, 1.0]])
    param, _ = pa.calculate_var_parametric(0.99, 1)
    mc, _ = pa.calculate_var_monte_carlo(0.99, 1, num_sims=400_000, random_seed=4)
    assert mc == pytest.approx(param, rel=2e-2)


def test_monte_carlo_var_is_reproducible_with_seed():
    pa = _stock_book(("X", 100, 100.0, 0.2))
    a = pa.calculate_var_monte_carlo(0.95, 1, num_sims=2_000, random_seed=7)
    b = pa.calculate_var_monte_carlo(0.95, 1, num_sims=2_000, random_seed=7)
    assert a == b
