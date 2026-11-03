"""Asian options: geometric analytical vs MC, AM-GM ordering, control-variate gain."""

import pytest

from derivflow.exotic.asian_options import AsianOptions


def test_geometric_analytical_vs_mc(tol):
    asian = AsianOptions(num_sims=100000, random_seed=tol["mc_seed"])
    analytic = asian.price_geometric_asian_analytical(100, 100, 1.0, 0.05, 0.2, "call")
    mc = asian.price_geometric_asian_mc(100, 100, 1.0, 0.05, 0.2, "call", num_steps=100)
    assert abs(mc.price - analytic.price) < tol["mc_sigma"] * mc.std_error


@pytest.mark.slow
def test_arithmetic_ge_geometric_call():
    asian = AsianOptions(num_sims=100000, random_seed=42)
    arith = asian.price(100, 100, 1.0, 0.05, 0.2, "call", asian_type="arithmetic", num_steps=50)
    geo = asian.price(100, 100, 1.0, 0.05, 0.2, "call", asian_type="geometric", num_steps=50)
    # By AM-GM, the arithmetic average dominates the geometric average.
    assert arith.price >= geo.price - 3 * (arith.std_error or 0.0)


@pytest.mark.slow
def test_control_variate_reduces_variance():
    asian = AsianOptions(num_sims=50000, random_seed=42)
    with_cv = asian.price_arithmetic_asian_mc(
        100, 100, 1.0, 0.05, 0.2, "call", num_steps=50, use_control_variate=True
    )
    without_cv = asian.price_arithmetic_asian_mc(
        100, 100, 1.0, 0.05, 0.2, "call", num_steps=50, use_control_variate=False
    )
    assert with_cv.std_error < without_cv.std_error
    ratio = with_cv.convergence_info.get("variance_reduction_ratio")
    assert ratio is not None and ratio > 10  # conservative floor; usually far higher


def test_arithmetic_asian_cheaper_than_european():
    asian = AsianOptions(num_sims=50000, random_seed=42)
    res = asian.price(100, 100, 1.0, 0.05, 0.2, "call", asian_type="arithmetic", num_steps=50)
    from derivflow.core.pricing_engine import price_european_option

    european = price_european_option(100, 100, 1.0, 0.05, 0.2, "call")
    # Averaging reduces effective volatility -> cheaper than the vanilla.
    assert res.price < european


def _discrete_geometric_reference(S, K, T, r, sigma, n, option_type):
    """Geometric average of n equally spaced fixings t_i = i T / n (i = 1..n).

    ln G is normal with mean ln S + (r - sigma^2/2) T (n+1) / (2n) and variance
    sigma^2 T (n+1)(2n+1) / (6 n^2); the price is the discounted lognormal
    expectation (Kemna and Vorst 1990 for the continuous limit).
    """
    import numpy as np
    from scipy.stats import norm

    m = np.log(S) + (r - 0.5 * sigma**2) * T * (n + 1) / (2 * n)
    s = sigma * np.sqrt(T * (n + 1) * (2 * n + 1) / (6 * n**2))
    d2 = (m - np.log(K)) / s
    d1 = d2 + s
    fwd = np.exp(m + 0.5 * s**2)
    if option_type == "call":
        return np.exp(-r * T) * (fwd * norm.cdf(d1) - K * norm.cdf(d2))
    return np.exp(-r * T) * (K * norm.cdf(-d2) - fwd * norm.cdf(-d1))


@pytest.mark.parametrize("n", [1, 4, 12, 52, 252])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_discrete_geometric_closed_form(n, option_type):
    asian = AsianOptions()
    got = asian.price_geometric_asian_analytical(
        100, 95, 1.0, 0.05, 0.25, option_type, average_type="discrete", num_fixings=n
    ).price
    ref = _discrete_geometric_reference(100, 95, 1.0, 0.05, 0.25, n, option_type)
    assert got == pytest.approx(ref, rel=1e-12)


def test_single_fixing_geometric_equals_european(bs_ref):
    asian = AsianOptions()
    got = asian.price_geometric_asian_analytical(
        100, 95, 1.0, 0.05, 0.25, "call", average_type="discrete", num_fixings=1
    ).price
    assert got == pytest.approx(bs_ref(100, 95, 1.0, 0.05, 0.25, "call"), rel=1e-12)
