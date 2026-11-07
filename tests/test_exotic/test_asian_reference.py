"""Asian options against references computed independently in this file.

- Geometric average: put-call parity C - P = e^{-rT} (E[G] - K), with E[G]
  for discrete fixings from the lognormal moments of ln G built directly from
  the fixing times (Var ln G = sigma^2 * mean of min(t_i, t_j)).
- Arithmetic average: put-call parity C - P = e^{-rT} (E[A] - K) with
  E[A] = S mean(e^{r t_i}), which holds for any model with the right forward.
- Greeks of the geometric closed form: analytic derivatives of
  e^{-rT} [F N(d1) - K N(d2)].
- Arithmetic delta: the pathwise estimator e^{-rT} E[1{A > K} A / S] from a
  separate simulation.

Fixings are t_i = i T / n, i = 1..n, as documented in the module.
"""

import numpy as np
import pytest
from scipy.stats import norm

from derivflow_finance.exotic.asian_options import (
    AsianOptionResult,
    AsianOptions,
    compare_asian_types,
    price_asian_option,
)

S, R, SIGMA = 100.0, 0.05, 0.2


def _fixings(T, n):
    return T * np.arange(1, n + 1) / n


def _geometric_forward(S, T, r, sigma, n=None):
    """E[G] for n discrete fixings (or the continuous average if n is None)."""
    if n is None:
        mean_ln, var_ln = np.log(S) + (r - sigma**2 / 2) * T / 2, sigma**2 * T / 3
    else:
        t = _fixings(T, n)
        mean_ln = np.log(S) + (r - sigma**2 / 2) * t.mean()
        var_ln = sigma**2 * np.minimum.outer(t, t).mean()
    return np.exp(mean_ln + var_ln / 2)


@pytest.mark.parametrize("n", [None, 1, 12, 252])
@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
@pytest.mark.parametrize("T", [0.25, 1.0, 3.0])
def test_geometric_put_call_parity(n, K, T):
    eng = AsianOptions()
    avg = "continuous" if n is None else "discrete"
    kw = {} if n is None else {"num_fixings": n}
    c = eng.price_geometric_asian_analytical(S, K, T, R, SIGMA, "call", avg, **kw).price
    p = eng.price_geometric_asian_analytical(S, K, T, R, SIGMA, "put", avg, **kw).price
    expected = np.exp(-R * T) * (_geometric_forward(S, T, R, SIGMA, n) - K)
    assert c - p == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("use_cv", [True, False])
@pytest.mark.parametrize("K", [90.0, 100.0, 110.0])
def test_arithmetic_put_call_parity(K, use_cv):
    T, n = 1.0, 12
    expected_average = S * np.exp(R * _fixings(T, n)).mean()
    c = AsianOptions(num_sims=100_000, random_seed=21).price_arithmetic_asian_mc(
        S, K, T, R, SIGMA, "call", n, use_cv
    )
    p = AsianOptions(num_sims=100_000, random_seed=21).price_arithmetic_asian_mc(
        S, K, T, R, SIGMA, "put", n, use_cv
    )
    gap = c.price - p.price - np.exp(-R * T) * (expected_average - K)
    assert abs(gap) < 4 * (c.std_error + p.std_error)


@pytest.mark.parametrize("K", [90.0, 110.0])
def test_geometric_monte_carlo_put_matches_closed_form(K):
    eng = AsianOptions(num_sims=100_000, random_seed=31)
    mc = eng.price_geometric_asian_mc(S, K, 1.0, R, SIGMA, "put", num_steps=12)
    cf = eng.price_geometric_asian_analytical(S, K, 1.0, R, SIGMA, "put", "discrete", 12)
    assert abs(mc.price - cf.price) < 4 * mc.std_error


def test_control_variate_falls_back_when_no_path_pays():
    res = AsianOptions(num_sims=5_000, random_seed=1).price_arithmetic_asian_mc(
        S, 1000.0, 1.0, R, SIGMA, "call"
    )
    assert res.price == 0.0 and res.std_error == 0.0
    assert res.convergence_info["control_variate_used"] is False
    assert "variance" in res.convergence_info["reason"]


def _geometric_greeks(S, K, T, r, sigma, n):
    """Analytic Greeks of the discrete geometric call: vega per unit of
    volatility, rho per unit of rate, theta = -dV/dT per year."""
    a, b = (n + 1) / (2 * n), (n + 1) * (2 * n + 1) / (6 * n**2)
    tau, c = a * T, b * T
    m = np.log(S) + (r - sigma**2 / 2) * tau
    s = sigma * np.sqrt(c)
    F = np.exp(m + s**2 / 2)
    d2 = (m - np.log(K)) / s
    d1 = d2 + s
    D = np.exp(-r * T)
    V = D * (F * norm.cdf(d1) - K * norm.cdf(d2))
    dF_dT = F * ((r - sigma**2 / 2) * a + sigma**2 * b / 2)
    ds_dT = sigma * np.sqrt(b) / (2 * np.sqrt(T))
    return {
        "delta": D * F / S * norm.cdf(d1),
        "gamma": D * F / S**2 * norm.pdf(d1) / s,
        "vega": D * F * (norm.cdf(d1) * sigma * (c - tau) + norm.pdf(d1) * np.sqrt(c)),
        "rho": -T * V + D * F * tau * norm.cdf(d1),
        "theta": -(-r * V + D * (norm.cdf(d1) * dF_dT + F * norm.pdf(d1) * ds_dT)),
    }


@pytest.mark.parametrize(
    ("K", "T", "r", "sigma"),
    [(100.0, 1.0, 0.05, 0.2), (90.0, 0.5, 0.03, 0.35), (120.0, 2.0, 0.01, 0.15)],
)
def test_geometric_sensitivities_match_analytic_greeks(K, T, r, sigma):
    """sensitivity_analysis prices with 252 fixings; its central differences
    agree with the analytic derivatives to within the finite-difference error."""
    ref = _geometric_greeks(S, K, T, r, sigma, 252)
    eng = AsianOptions(random_seed=1)
    got = {}
    for param in ("spot", "volatility", "rate", "time"):
        got.update(eng.sensitivity_analysis(S, K, T, r, sigma, "call", "geometric", param=param))
    for greek, value in ref.items():
        assert got[greek] == pytest.approx(value, rel=5e-3), greek


def test_arithmetic_delta_matches_the_pathwise_estimator():
    K, T, n = 100.0, 1.0, 252
    rng = np.random.default_rng(12345)
    dt = T / n
    log_s = np.log(S) + np.cumsum(
        (R - SIGMA**2 / 2) * dt + SIGMA * np.sqrt(dt) * rng.standard_normal((50_000, n)), axis=1
    )
    average = np.exp(log_s).mean(axis=1)
    pathwise = np.exp(-R * T) * (average > K) * average / S
    reference, se = pathwise.mean(), pathwise.std(ddof=1) / np.sqrt(pathwise.size)
    delta = AsianOptions(num_sims=50_000, random_seed=5).sensitivity_analysis(
        S, K, T, R, SIGMA, "call", "arithmetic", param="spot"
    )["delta"]
    assert abs(delta - reference) < 4 * se + 0.005


def test_sensitivity_analysis_rejects_unknown_parameter_and_restores_the_generator():
    eng = AsianOptions(num_sims=100, random_seed=2)
    rng = eng.rng
    with pytest.raises(ValueError, match="Unknown parameter"):
        eng.sensitivity_analysis(S, 100, 1.0, R, SIGMA, param="dividend")
    assert eng.rng is rng


def test_price_routing():
    eng = AsianOptions(num_sims=2_000, random_seed=3)
    assert eng.price(S, 100, 1.0, R, SIGMA, asian_type="geometric").pricing_method == "analytical"
    ari = eng.price(S, 100, 1.0, R, SIGMA, asian_type="arithmetic")
    assert ari.pricing_method == "monte_carlo" and ari.convergence_info["control_variate_used"]
    plain = eng.price(S, 100, 1.0, R, SIGMA, use_control_variate=False)
    assert plain.convergence_info == {"control_variate_used": False}
    geo_mc = eng.price(S, 100, 1.0, R, SIGMA, asian_type="geometric", method="monte_carlo")
    assert geo_mc.pricing_method == "monte_carlo" and geo_mc.average_type == "discrete"
    with pytest.raises(ValueError, match="Unsupported"):
        eng.price(S, 100, 1.0, R, SIGMA, asian_type="arithmetic", method="analytical")
    with pytest.raises(ValueError, match="Unsupported"):
        eng.price(S, 100, 1.0, R, SIGMA, asian_type="geometric", method="lattice")
    with pytest.raises(ValueError, match="average_type"):
        eng.price(S, 100, 1.0, R, SIGMA, average_type="weekly")


def test_constructor_and_closed_form_input_checks():
    with pytest.raises(ValueError, match="num_sims"):
        AsianOptions(num_sims=1)
    eng = AsianOptions()
    with pytest.raises(ValueError, match="num_fixings"):
        eng.price_geometric_asian_analytical(S, 100, 1.0, R, SIGMA, "call", "discrete", 0)
    with pytest.raises(ValueError, match="average_type"):
        eng.price_geometric_asian_analytical(S, 100, 1.0, R, SIGMA, "call", "weekly")


def test_convenience_functions():
    geo = price_asian_option(S, 100, 1.0, R, SIGMA, "call", "geometric")
    expected = AsianOptions().price_geometric_asian_analytical(
        S, 100, 1.0, R, SIGMA, "call", "discrete"
    )
    assert geo == pytest.approx(expected.price, abs=1e-14)

    reference = AsianOptions(num_sims=100_000, random_seed=7).price(S, 100, 1.0, R, SIGMA)
    assert isinstance(reference, AsianOptionResult)
    ari = price_asian_option(S, 100, 1.0, R, SIGMA)  # unseeded: compare with a margin
    assert abs(ari - reference.price) < 10 * reference.std_error * 2**0.5

    both = compare_asian_types(S, 100, 1.0, R, SIGMA)
    assert both["geometric_asian"] == pytest.approx(expected.price, abs=1e-14)
    assert both["arithmetic_asian"] > both["geometric_asian"]
    assert both["price_difference"] == pytest.approx(
        both["arithmetic_asian"] - both["geometric_asian"]
    )
    assert both["relative_difference"] == pytest.approx(
        both["price_difference"] / both["geometric_asian"]
    )
