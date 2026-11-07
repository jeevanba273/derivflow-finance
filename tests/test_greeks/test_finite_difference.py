"""Every analytical Greek against a central finite difference of the price.

The finite differences are taken on an independent Black-Scholes-Merton
oracle (the ``bs_ref`` fixture in conftest.py), not on code from the package. Units
follow the convention documented in ``GreeksCalculator.analytical_greeks``:

- theta and color are per calendar day (derivative w.r.t. calendar time / 365),
- vega, rho, vanna and zomma are per 1 percentage point (0.01) of vol or rate,
- volga is per (1 vol point) squared.
"""

import itertools

import pytest

from derivflow_finance.greeks.calculator import GreeksCalculator

calc = GreeksCalculator("analytical")

CASES = list(
    itertools.product(
        [80.0, 100.0, 125.0],  # spot
        [0.1, 0.75],  # T
        [0.15, 0.45],  # sigma
        [0.0, 0.03],  # q
        ["call", "put"],
    )
)
K, R = 100.0, 0.04


def _fd(bs_ref, S, T, sigma, q, ot):
    """Central differences of the oracle price.

    First- and second-order references use small steps; third-order references
    are nested differences, so they use larger steps to keep rounding noise
    (which grows like eps / h**3) below the truncation error.
    """

    def v(S_=S, T_=T, s_=sigma, r_=R):
        return bs_ref(S_, K, T_, r_, s_, ot, q)

    def gamma(S_=S, T_=T, s_=sigma, h=1e-4 * S):
        return (v(S_ + h, T_, s_) - 2 * v(S_, T_, s_) + v(S_ - h, T_, s_)) / h**2

    def delta(S_=S, s_=sigma, h=1e-4 * S):
        return (v(S_ + h, T, s_) - v(S_ - h, T, s_)) / (2 * h)

    def vega_raw(s_=sigma, h=1e-4):
        return (v(S, T, s_ + h) - v(S, T, s_ - h)) / (2 * h)

    hv, hT, hr = 1e-4, 1e-5, 1e-5
    # Step sizes scale with S*sigma*sqrt(T), the natural scale of d1.
    hS3, hv3, hT3 = 1e-2 * S * sigma * T**0.5, 1e-4, 1e-4 * T
    return {
        "delta": delta(),
        "gamma": gamma(),
        "theta": -(v(S, T + hT) - v(S, T - hT)) / (2 * hT) / 365.0,
        "vega": vega_raw() / 100.0,
        "rho": (v(S, T, sigma, R + hr) - v(S, T, sigma, R - hr)) / (2 * hr) / 100.0,
        "volga": (vega_raw(sigma + hv) - vega_raw(sigma - hv)) / (2 * hv) / 1e4,
        "vanna": (delta(S, sigma + hv) - delta(S, sigma - hv)) / (2 * hv) / 100.0,
        "speed": (gamma(S + hS3, h=hS3) - gamma(S - hS3, h=hS3)) / (2 * hS3),
        "zomma": (gamma(S, T, sigma + hv3, h=hS3) - gamma(S, T, sigma - hv3, h=hS3))
        / (2 * hv3)
        / 100.0,
        "color": -(gamma(S, T + hT3, h=hS3) - gamma(S, T - hT3, h=hS3)) / (2 * hT3) / 365.0,
    }


GREEKS = ["delta", "gamma", "theta", "vega", "rho", "volga", "vanna", "speed", "zomma", "color"]


@pytest.mark.parametrize("greek", GREEKS)
@pytest.mark.parametrize(("S", "T", "sigma", "q", "ot"), CASES)
def test_greek_matches_central_difference(bs_ref, S, T, sigma, q, ot, greek):
    g = calc.analytical_greeks(S, K, T, R, sigma, ot, q)
    ref = _fd(bs_ref, S, T, sigma, q, ot)[greek]
    # Third-order nested differences carry more truncation/rounding error.
    rel = 2e-3 if greek in ("speed", "zomma", "color") else 1e-5
    assert getattr(g, greek) == pytest.approx(ref, rel=rel, abs=1e-8)
