"""Plotly dashboard: each plot method returns a figure (needs the [plot] extra)."""

import pytest

go = pytest.importorskip("plotly.graph_objects")
pytest.importorskip("pandas")

from derivflow_finance.visualization.dashboard import (  # noqa: E402
    DerivativesDashboard,
    quick_greeks_plot,
    quick_payoff_plot,
)


def test_dashboard_instantiates():
    dash = DerivativesDashboard()
    assert dash.theme == "plotly_dark"
    assert dash._market_data is None  # market data client is created only when used


def test_option_payoff_figure():
    dash = DerivativesDashboard()
    fig = dash.plot_option_payoff(100, 5.0, "call")
    assert isinstance(fig, go.Figure)


def test_greeks_dashboard_figure():
    dash = DerivativesDashboard()
    fig = dash.plot_greeks_dashboard(100, 100, 0.25, 0.05, 0.2, "call")
    assert isinstance(fig, go.Figure)


def test_quick_helpers_return_figures():
    assert isinstance(quick_payoff_plot(100, 5.0, "call"), go.Figure)
    assert isinstance(quick_greeks_plot(100, 100, 0.25, 0.05, 0.2), go.Figure)
