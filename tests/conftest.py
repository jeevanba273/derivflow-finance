"""Shared fixtures and tolerances for the DERIVFLOW-FINANCE test suite."""

import numpy as np
import pytest
import scipy.stats as stats

# Shared numerical tolerances
ANALYTIC_TOL = 1e-6     # exact closed-form agreement
GREEK_FD_TOL = 1e-3     # analytical vs central finite-difference Greeks
MC_SIGMA = 4            # Monte Carlo within N standard errors of truth
MC_SEED = 42


def _bs_reference(S, K, T, r, sigma, option_type='call', q=0.0):
    """Independent closed-form Black-Scholes-Merton oracle (with dividend yield q)."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * np.exp(-q * T) * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * np.exp(-q * T) * stats.norm.cdf(-d1)


@pytest.fixture
def bs_ref():
    """Return the independent Black-Scholes reference pricer."""
    return _bs_reference


@pytest.fixture
def tol():
    """Shared tolerance bundle."""
    return {
        'analytic': ANALYTIC_TOL,
        'greek_fd': GREEK_FD_TOL,
        'mc_sigma': MC_SIGMA,
        'mc_seed': MC_SEED,
    }


@pytest.fixture
def params():
    """Repo demo case (out-of-the-money 3M call)."""
    return {'S': 100, 'K': 105, 'T': 0.25, 'r': 0.05, 'sigma': 0.20}


@pytest.fixture
def atm_params():
    """At-the-money 1Y case."""
    return {'S': 100, 'K': 100, 'T': 1.0, 'r': 0.05, 'sigma': 0.20}


@pytest.fixture
def hull_params():
    """Hull textbook worked example (call 4.7594, put 0.8086)."""
    return {'S': 42, 'K': 40, 'T': 0.5, 'r': 0.10, 'sigma': 0.20}
