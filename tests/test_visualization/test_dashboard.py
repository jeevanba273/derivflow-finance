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


def test_pnl_analysis_option_type_is_case_insensitive():
    dash = DerivativesDashboard()
    lower = dash.plot_pnl_analysis(100, 100, 0.25, 0.05, 0.2, "call", 1, 2.0)
    title = dash.plot_pnl_analysis(100, 100, 0.25, 0.05, 0.2, "Call", 1, 2.0)
    for a, b in zip(lower.data, title.data, strict=True):
        assert a.name == b.name and list(a.y) == list(b.y)


@pytest.mark.parametrize(
    ("T", "names"),
    [
        (3 / 365, ["At Expiry", "Current"]),
        (0.05, ["At Expiry", "1 Week Left", "Current"]),
        (1.0, ["At Expiry", "1 Week Left", "1 Month Left", "Current"]),
    ],
)
def test_pnl_analysis_only_shows_times_within_the_option_life(T, names):
    fig = DerivativesDashboard().plot_pnl_analysis(100, 100, T, 0.05, 0.2, "put", 1, 2.0)
    assert [trace.name for trace in fig.data] == names


def test_price_sensitivity_option_type_is_case_insensitive():
    dash = DerivativesDashboard()
    lower = dash.plot_price_sensitivity(100, 100, 0.5, 0.05, 0.2, "call")
    upper = dash.plot_price_sensitivity(100, 100, 0.5, 0.05, 0.2, "CALL")
    assert lower.to_plotly_json() == upper.to_plotly_json()
