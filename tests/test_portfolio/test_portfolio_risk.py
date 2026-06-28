"""Portfolio risk: aggregation, sample portfolio, and parametric VaR ordering."""

import pytest

from derivflow.portfolio.portfolio_risk import PortfolioRiskAnalyzer, create_sample_portfolio
from derivflow.greeks.calculator import GreeksCalculator


def test_sample_portfolio_builds():
    analyzer = create_sample_portfolio()
    assert len(analyzer.positions) == 5
    assert analyzer.calculate_portfolio_value() > 0


def test_single_call_delta_aggregation():
    analyzer = PortfolioRiskAnalyzer(risk_free_rate=0.05)
    analyzer.add_option_position('X', 1, 100, 100, 1.0, 'call', 0.2)
    option_delta = GreeksCalculator('analytical').calculate_greeks(
        100, 100, 1.0, 0.05, 0.2, 'call').delta
    greeks = analyzer.calculate_portfolio_greeks()
    # One contract carries a x100 multiplier.
    assert greeks['delta'] == pytest.approx(100 * option_delta, rel=1e-6)


def test_parametric_var_ordering():
    analyzer = create_sample_portfolio()
    var95, es95 = analyzer.calculate_var_parametric(0.95)
    var99, es99 = analyzer.calculate_var_parametric(0.99)
    assert var95 > 0 and var99 > 0
    assert var95 < var99      # higher confidence -> larger VaR
    assert es95 >= var95      # expected shortfall is at least VaR
