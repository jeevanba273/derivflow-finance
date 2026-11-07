"""PortfolioRiskAnalyzer: input errors and empty or flat books."""

import numpy as np
import pytest

from derivflow_finance.portfolio.portfolio_risk import PortfolioRiskAnalyzer
from derivflow_finance.portfolio.risk_measures import value_at_risk


def _book():
    book = PortfolioRiskAnalyzer()
    book.add_stock_position("AAA", 100, 50.0, 0.3)
    book.add_option_position("AAA", 2, 50.0, 55.0, 0.5, "call", 0.3)
    return book


@pytest.mark.parametrize(("confidence", "horizon"), [(0.0, 1), (1.0, 1), (1.5, 1), (0.95, 0)])
def test_var_inputs_are_checked(confidence, horizon):
    book = _book()
    with pytest.raises(ValueError, match=r"confidence_level|time_horizon"):
        book.calculate_var_parametric(confidence, horizon)
    with pytest.raises(ValueError, match=r"confidence_level|time_horizon"):
        book.calculate_var_monte_carlo(confidence, horizon, num_sims=100)


def test_an_empty_book_has_no_risk():
    empty = PortfolioRiskAnalyzer()
    assert empty.calculate_var_parametric() == (0.0, 0.0)
    assert np.array_equal(empty.simulate_pnl(num_sims=5), np.zeros(5))


def test_a_flat_book_has_zero_contributions():
    flat = PortfolioRiskAnalyzer()
    flat.add_stock_position("AAA", 100, 50.0, 0.3)
    flat.add_stock_position("AAA", -100, 50.0, 0.3)
    assert flat.var_contributions() == {"AAA": 0.0}


def test_simulation_needs_two_paths():
    with pytest.raises(ValueError, match="num_sims"):
        _book().simulate_pnl(num_sims=1)


@pytest.mark.parametrize(
    ("matrix", "match"),
    [
        ([[1.0, 0.5, 0.0], [0.5, 1.0, 0.0]], "2x2"),
        ([[1.0, 0.5], [0.4, 1.0]], "symmetric"),
        ([[0.9, 0.5], [0.5, 1.0]], "unit diagonal"),
        ([[1.0, 1.2], [1.2, 1.0]], "unit diagonal"),
    ],
)
def test_correlation_matrix_is_validated(matrix, match):
    with pytest.raises(ValueError, match=match):
        _book().set_correlation(["AAA", "BBB"], matrix)


def test_correlation_matrix_must_be_positive_semi_definite():
    with pytest.raises(ValueError, match="positive semi-definite"):
        _book().set_correlation(
            ["A", "B", "C"], [[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]]
        )


def test_an_underlying_without_a_volatility_is_an_error():
    book = PortfolioRiskAnalyzer()
    book.add_stock_position("AAA", 100, 50.0, 0.0)
    with pytest.raises(ValueError, match="No positive volatility"):
        book.calculate_var_parametric()


def test_scenario_method_and_hedge_symbol_are_checked():
    book = _book()
    with pytest.raises(ValueError, match="method"):
        book.scenario_analysis({"down": {"AAA": -0.1}}, method="historical")
    with pytest.raises(ValueError, match="No position found"):
        book.calculate_hedge_ratio("ZZZ")


def test_value_at_risk_of_an_empty_sample_is_an_error():
    with pytest.raises(ValueError, match="empty"):
        value_at_risk([], 0.99)
