"""
DERIVFLOW-FINANCE: Visualization Dashboard
=========================================

Interactive visualization dashboard for derivatives analytics.
"""

from .dashboard import (
    DerivativesDashboard,
    quick_payoff_plot,
    quick_greeks_plot
)

__all__ = [
    'DerivativesDashboard',
    'quick_payoff_plot',
    'quick_greeks_plot'
]