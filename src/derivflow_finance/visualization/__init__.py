"""Plotly figures for payoffs, Greeks and surfaces (requires the ``[plot]`` extra)."""

from .dashboard import DerivativesDashboard, quick_greeks_plot, quick_payoff_plot

__all__ = ["DerivativesDashboard", "quick_greeks_plot", "quick_payoff_plot"]
