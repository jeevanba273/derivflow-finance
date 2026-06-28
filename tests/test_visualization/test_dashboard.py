"""Visualization dashboard: figures are produced for the headline plots.

(Replaces the original print-style smoke script with real assertions.)
"""

import plotly.graph_objects as go

from derivflow.visualization.dashboard import (
    DerivativesDashboard, quick_payoff_plot, quick_greeks_plot,
)


def test_dashboard_instantiates():
    dash = DerivativesDashboard()
    assert dash.theme == 'plotly_dark'
    # The constructor previously failed at import (MarketDataProvider).
    from derivflow.utils.market_data import AdvancedMarketData
    assert isinstance(dash.market_data, AdvancedMarketData)


def test_option_payoff_figure():
    dash = DerivativesDashboard()
    fig = dash.plot_option_payoff(100, 5.0, 'call')
    assert isinstance(fig, go.Figure)


def test_greeks_dashboard_figure():
    dash = DerivativesDashboard()
    fig = dash.plot_greeks_dashboard(100, 100, 0.25, 0.05, 0.2, 'call')
    assert isinstance(fig, go.Figure)


def test_quick_helpers_return_figures():
    assert isinstance(quick_payoff_plot(100, 5.0, 'call'), go.Figure)
    assert isinstance(quick_greeks_plot(100, 100, 0.25, 0.05, 0.2), go.Figure)
