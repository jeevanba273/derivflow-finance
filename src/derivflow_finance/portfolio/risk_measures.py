"""
Value-at-Risk and expected shortfall.

Sign convention: inputs are P&L (gains positive), outputs are losses reported
as positive numbers. For a confidence level c (e.g. 0.99),

- VaR_c is the loss that is exceeded with probability 1 - c:
  VaR_c = -Q_{1-c}(P&L);
- ES_c (expected shortfall, also CVaR) is the expected loss given that the
  loss is at least VaR_c: ES_c = -E[P&L | P&L <= -VaR_c].

The functions are unit-agnostic: pass currency P&L or returns over whatever
horizon you need. Horizon scaling and annualisation are the caller's choice;
the rest of the package uses 252 trading days per year.

Reference: McNeil, Frey and Embrechts, Quantitative Risk Management (2nd ed.,
2015), section 2.3.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

__all__ = [
    "expected_shortfall",
    "normal_es",
    "normal_var",
    "student_t_es",
    "student_t_var",
    "value_at_risk",
]


def _check_confidence(c: float) -> None:
    if not 0.0 < c < 1.0:
        raise ValueError("confidence must be in (0, 1)")


def value_at_risk(pnl: ArrayLike, confidence: float = 0.99) -> float:
    """Empirical VaR of a P&L sample: minus the (1 - confidence) sample quantile.

    The quantile uses NumPy's default linear interpolation between order
    statistics (Hyndman and Fan definition 7).
    """
    _check_confidence(confidence)
    x = np.asarray(pnl, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("pnl is empty")
    return float(-np.quantile(x, 1.0 - confidence))


def expected_shortfall(pnl: ArrayLike, confidence: float = 0.99) -> float:
    """Empirical expected shortfall: minus the mean P&L at or below -VaR."""
    var = value_at_risk(pnl, confidence)
    x = np.asarray(pnl, dtype=float).ravel()
    return float(-x[x <= -var].mean())


def normal_var(sigma: float, confidence: float = 0.99, mu: float = 0.0) -> float:
    """VaR of a normal P&L with mean mu and standard deviation sigma:
    sigma * z_c - mu, with z_c = N^{-1}(c)."""
    _check_confidence(confidence)
    return float(sigma * stats.norm.ppf(confidence) - mu)


def normal_es(sigma: float, confidence: float = 0.99, mu: float = 0.0) -> float:
    """Expected shortfall of a normal P&L: sigma * phi(z_c) / (1 - c) - mu."""
    _check_confidence(confidence)
    z = stats.norm.ppf(confidence)
    return float(sigma * stats.norm.pdf(z) / (1.0 - confidence) - mu)


def student_t_var(scale: float, df: float, confidence: float = 0.99, mu: float = 0.0) -> float:
    """VaR of P&L = mu + scale * T with T ~ Student t(df): scale * t_c - mu.

    ``scale`` is the t scale parameter, not the standard deviation (the
    standard deviation is scale * sqrt(df / (df - 2)) for df > 2).
    """
    _check_confidence(confidence)
    return float(scale * stats.t.ppf(confidence, df) - mu)


def student_t_es(scale: float, df: float, confidence: float = 0.99, mu: float = 0.0) -> float:
    """Expected shortfall of P&L = mu + scale * T, T ~ t(df), df > 1:

    scale * f(t_c) / (1 - c) * (df + t_c^2) / (df - 1) - mu,

    where f is the t density and t_c its c-quantile.
    """
    _check_confidence(confidence)
    if df <= 1:
        raise ValueError("expected shortfall needs df > 1 (finite mean)")
    q = stats.t.ppf(confidence, df)
    tail = stats.t.pdf(q, df) / (1.0 - confidence) * (df + q * q) / (df - 1.0)
    return float(scale * tail - mu)
