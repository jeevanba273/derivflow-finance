"""Properties that must hold for every input, checked with Hypothesis.

Each test states a no-arbitrage relation or an identity of the model and lets
Hypothesis search the input space for a counterexample. Inputs span spot 1 to
1,000, strike 0.25x to 4x spot, 1 day to 10 years, rates -2% to 15%,
dividend yields 0 to 10% and volatilities 1% to 300%.

CI runs the fixed 'ci' profile (derandomised, so results do not change from
run to run). For a wider search run
``DERIVFLOW_HYPOTHESIS_PROFILE=hunt pytest tests/test_properties.py``.
"""

import math
import os
import warnings

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, assume, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from derivflow_finance import (  # noqa: E402
    AsianOptions,
    BarrierOptions,
    BinomialTree,
    BlackScholesAnalytical,
    GreeksCalculator,
    black_scholes_price,
    implied_volatility,
    implied_volatility_vectorized,
)
from derivflow_finance.portfolio.risk_measures import expected_shortfall, value_at_risk  # noqa: E402

settings.register_profile(
    "ci",
    max_examples=200,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "hunt", max_examples=5000, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
settings.load_profile(os.environ.get("DERIVFLOW_HYPOTHESIS_PROFILE", "ci"))

spot = st.floats(1.0, 1000.0)
moneyness = st.floats(0.25, 4.0)
expiry = st.floats(1 / 365, 10.0)
rate = st.floats(-0.02, 0.15)
dividend = st.floats(0.0, 0.10)
vol = st.floats(0.01, 3.0)
option_type = st.sampled_from(["call", "put"])
EPS = np.finfo(float).eps


def _tol(S, K):
    return 1e-10 * max(S, K)


@given(spot, moneyness, expiry, rate, dividend, vol)
def test_prices_lie_within_the_no_arbitrage_bounds(S, m, T, r, q, sigma):
    K = S * m
    fwd_s, fwd_k = S * math.exp(-q * T), K * math.exp(-r * T)
    c = black_scholes_price(S, K, T, r, sigma, "call", q)
    p = black_scholes_price(S, K, T, r, sigma, "put", q)
    tol = _tol(S, K)
    assert max(fwd_s - fwd_k, 0.0) - tol <= c <= fwd_s + tol
    assert max(fwd_k - fwd_s, 0.0) - tol <= p <= fwd_k + tol
    assert abs((c - p) - (fwd_s - fwd_k)) <= tol


@given(spot, moneyness, expiry, rate, dividend, vol, st.floats(1e-3, 0.5))
def test_prices_move_the_right_way_with_strike(S, m, T, r, q, sigma, step):
    K1 = S * m
    K2 = K1 * (1 + step)
    Km = 0.5 * (K1 + K2)
    tol = _tol(S, K2)
    c1, cm, c2 = (black_scholes_price(S, k, T, r, sigma, "call", q) for k in (K1, Km, K2))
    p1, p2 = (black_scholes_price(S, k, T, r, sigma, "put", q) for k in (K1, K2))
    assert c2 <= c1 + tol and p2 >= p1 - tol
    # vertical spread: a call spread is worth at most the discounted strike gap
    assert c1 - c2 <= math.exp(-r * T) * (K2 - K1) + tol
    # butterfly: call prices are convex in strike
    assert c1 + c2 - 2 * cm >= -tol


@given(spot, moneyness, expiry, rate, dividend, vol, st.floats(1e-3, 1.0), option_type)
def test_prices_rise_with_volatility(S, m, T, r, q, sigma, step, ot):
    K = S * m
    low = black_scholes_price(S, K, T, r, sigma, ot, q)
    high = black_scholes_price(S, K, T, r, sigma * (1 + step), ot, q)
    assert high >= low - _tol(S, K)


@given(spot, moneyness, expiry, rate, dividend, vol, option_type)
def test_scalar_and_vectorised_pricers_agree(S, m, T, r, q, sigma, ot):
    K = S * m
    scalar = BlackScholesAnalytical().price(S, K, T, r, sigma, ot, q)
    vector = black_scholes_price(S, K, T, r, sigma, ot, q)
    assert scalar == pytest.approx(vector, rel=1e-12, abs=_tol(S, K))


@given(spot, moneyness, expiry, rate, dividend, vol, option_type)
def test_implied_volatility_recovers_sigma_and_reprices(S, m, T, r, q, sigma, ot):
    K = S * m
    price = black_scholes_price(S, K, T, r, sigma, ot, q)
    forward = S * math.exp((r - q) * T)
    otm = "call" if K >= forward else "put"
    time_value = black_scholes_price(S, K, T, r, sigma, otm, q)
    assume(time_value > 1e-8 * S)  # identifiable in double precision
    iv = implied_volatility(price, S, K, T, r, ot, q)
    assert black_scholes_price(S, K, T, r, iv, ot, q) == pytest.approx(price, abs=1e-10 * S)
    # The solver stops within 1e-12 * max(1, sigma); on top of that a price
    # rounding error of a few ULP moves the volatility by about ULP / vega.
    vega = GreeksCalculator().analytical_greeks(S, K, T, r, sigma, ot, q).vega * 100
    rounding = 64 * EPS * max(S, K) / vega if vega > 0 else math.inf
    allowed = 2e-12 * max(1.0, sigma) + rounding
    assert iv == pytest.approx(sigma, abs=allowed)
    vec = float(implied_volatility_vectorized(price, S, K, T, r, ot, q))
    assert vec == pytest.approx(iv, abs=2 * allowed)


@given(spot, moneyness, expiry, rate, dividend, vol)
def test_greek_signs_and_call_put_identities(S, m, T, r, q, sigma):
    K = S * m
    calc = GreeksCalculator()
    c = calc.analytical_greeks(S, K, T, r, sigma, "call", q)
    p = calc.analytical_greeks(S, K, T, r, sigma, "put", q)
    dq, dr = math.exp(-q * T), math.exp(-r * T)
    tiny = 1e-12
    assert -tiny <= c.delta <= dq + tiny and -dq - tiny <= p.delta <= tiny
    assert c.gamma >= 0 and c.vega >= 0
    assert c.delta - p.delta == pytest.approx(dq, abs=1e-12)
    assert c.gamma == pytest.approx(p.gamma, rel=1e-9, abs=1e-15)
    assert c.vega == pytest.approx(p.vega, rel=1e-9, abs=1e-15)
    assert c.rho - p.rho == pytest.approx(T * K * dr / 100, rel=1e-9, abs=1e-12)
    theta = (q * S * dq - r * K * dr) / 365
    assert c.theta - p.theta == pytest.approx(theta, rel=1e-9, abs=1e-12)


barrier_level = st.floats(0.3, 0.99)


@given(
    spot,
    moneyness,
    st.floats(1 / 365, 5.0),
    st.floats(0.0, 0.15),
    st.floats(0.05, 1.0),
    barrier_level,
    st.sampled_from(["down", "up"]),
    option_type,
)
def test_barrier_in_plus_out_is_the_vanilla(S, m, T, r, sigma, level, side, ot):
    K = S * m
    H = S * level if side == "down" else S / level
    barrier = BarrierOptions()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # strike on the far side of the barrier
        out = barrier.analytical_price(S, K, H, T, r, sigma, f"{side}_and_out", ot).price
        inn = barrier.analytical_price(S, K, H, T, r, sigma, f"{side}_and_in", ot).price
    vanilla = black_scholes_price(S, K, T, r, sigma, ot)
    tol = 1e-8 * max(S, K)
    assert out >= -tol and inn >= -tol
    assert out <= vanilla + tol and inn <= vanilla + tol
    assert out + inn == pytest.approx(vanilla, abs=tol)


@given(spot, moneyness, expiry, rate, vol, st.integers(1, 365))
def test_geometric_asian_put_call_parity(S, m, T, r, sigma, n):
    K = S * m
    asian = AsianOptions()
    c = asian.price_geometric_asian_analytical(S, K, T, r, sigma, "call", "discrete", n).price
    p = asian.price_geometric_asian_analytical(S, K, T, r, sigma, "put", "discrete", n).price
    t = T * np.arange(1, n + 1) / n
    forward = math.exp(
        math.log(S) + (r - sigma**2 / 2) * t.mean() + sigma**2 * np.minimum.outer(t, t).mean() / 2
    )
    assert c >= 0 and p >= 0
    assert c - p == pytest.approx(math.exp(-r * T) * (forward - K), abs=1e-9 * max(S, K, forward))


@given(
    spot, moneyness, st.floats(1 / 52, 5.0), st.floats(0.0, 0.15), dividend, st.floats(0.05, 1.0)
)
def test_american_put_dominates_the_european_put_and_its_intrinsic_value(S, m, T, r, q, sigma):
    K = S * m
    tree = BinomialTree(steps=200)
    american = tree.price(S, K, T, r, sigma, "put", american=True, q=q)
    european = tree.price(S, K, T, r, sigma, "put", american=False, q=q)
    tol = 1e-10 * max(S, K)
    assert american >= european - tol
    assert american >= K - S - tol


@given(spot, moneyness, st.floats(1 / 52, 5.0), st.floats(0.0, 0.15), st.floats(0.05, 1.0))
def test_american_call_without_dividends_is_the_european_call(S, m, T, r, sigma):
    K = S * m
    tree = BinomialTree(steps=200)
    american = tree.price(S, K, T, r, sigma, "call", american=True)
    european = tree.price(S, K, T, r, sigma, "call", american=False)
    assert american == pytest.approx(european, abs=1e-10 * max(S, K))


pnl_samples = st.lists(st.floats(-1e6, 1e6), min_size=20, max_size=300)


@given(pnl_samples, st.floats(0.5, 0.98), st.floats(0.001, 0.019))
def test_var_rises_with_confidence_and_shortfall_exceeds_var(sample, c1, gap):
    c2 = c1 + gap
    v1, v2 = value_at_risk(sample, c1), value_at_risk(sample, c2)
    assert v2 >= v1 - 1e-9 * (1 + abs(v1))
    assert expected_shortfall(sample, c2) >= v2 - 1e-9 * (1 + abs(v2))
