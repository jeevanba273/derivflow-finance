"""Heston model: parameter validation, Feller warning, and MC sanity."""

import pytest

from derivflow.core.pricing_engine import price_european_option
from derivflow.models.heston import HestonModel


def test_feller_condition_warns():
    model = HestonModel()
    # 2*kappa*theta = 0.08 < sigma^2 = 0.25 violates the Feller condition.
    with pytest.warns(UserWarning):
        model.set_parameters(v0=0.04, kappa=1.0, theta=0.04, sigma=0.5, rho=-0.5)


def test_feller_satisfied_does_not_warn(recwarn):
    model = HestonModel()
    # 2*kappa*theta = 0.16 > sigma^2 = 0.01 satisfies Feller.
    model.set_parameters(v0=0.04, kappa=2.0, theta=0.04, sigma=0.1, rho=-0.5)
    assert not any(issubclass(w.category, UserWarning) for w in recwarn.list)


def test_invalid_parameters_raise():
    model = HestonModel()
    with pytest.raises(ValueError):
        model.set_parameters(v0=0.04, kappa=-1.0, theta=0.04, sigma=0.1, rho=0.0)
    with pytest.raises(ValueError):
        model.set_parameters(v0=0.04, kappa=2.0, theta=0.04, sigma=0.1, rho=1.5)


@pytest.mark.slow
def test_mc_approaches_black_scholes_low_vol_of_vol():
    model = HestonModel()
    # Low vol-of-vol, zero correlation, v0=theta -> close to BS at sqrt(v0).
    model.set_parameters(v0=0.04, kappa=2.0, theta=0.04, sigma=0.05, rho=0.0)
    res = model.monte_carlo_price(100, 100, 1.0, 0.05, "call", num_sims=100000, num_steps=252)
    bs = price_european_option(100, 100, 1.0, 0.05, 0.2, "call")
    assert res.price == pytest.approx(bs, abs=0.3)


# Reference prices for the Heston model computed by Alan Lewis and used as
# regression targets in QuantLib's Heston test suite. Parameters: S=100, T=1,
# r=1%, q=2%, v0=0.04, kappa=4, theta=0.25, sigma=1, rho=-0.5 (European calls).
LEWIS_CALLS = {
    80: 26.774758743998854,
    90: 20.933349000596710,
    100: 16.070154917028834,
    110: 12.132211516709845,
    120: 9.024913483457836,
}


@pytest.mark.parametrize("K", sorted(LEWIS_CALLS))
def test_analytic_matches_lewis_reference(K):
    model = HestonModel()
    model.set_parameters(v0=0.04, kappa=4.0, theta=0.25, sigma=1.0, rho=-0.5)
    price = model.price_option(100.0, K, 1.0, 0.01, "call", method="analytic", q=0.02).price
    assert price == pytest.approx(LEWIS_CALLS[K], abs=1e-8)


@pytest.mark.parametrize("K", [70.0, 100.0, 140.0])
@pytest.mark.parametrize("T", [0.05, 1.0, 5.0])
def test_analytic_put_call_parity(K, T):
    import numpy as np

    model = HestonModel()
    model.set_parameters(v0=0.05, kappa=1.5, theta=0.06, sigma=0.6, rho=-0.7)
    c = model.price_option(100.0, K, T, 0.03, "call", method="analytic", q=0.01).price
    p = model.price_option(100.0, K, T, 0.03, "put", method="analytic", q=0.01).price
    assert c - p == pytest.approx(100 * np.exp(-0.01 * T) - K * np.exp(-0.03 * T), abs=1e-9)


def test_analytic_tends_to_black_scholes_as_vol_of_vol_vanishes(bs_ref):
    model = HestonModel()
    model.set_parameters(v0=0.04, kappa=2.0, theta=0.04, sigma=1e-4, rho=0.0)
    for K in (80.0, 100.0, 120.0):
        price = model.price_option(100.0, K, 1.0, 0.05, "call", method="analytic").price
        assert price == pytest.approx(bs_ref(100.0, K, 1.0, 0.05, 0.2, "call"), abs=1e-5)


@pytest.mark.parametrize(
    "params",
    [
        (0.04, 2.0, 0.04, 0.3, -0.7),  # Feller condition holds
        (0.04, 1.0, 0.04, 0.9, -0.7),  # Feller condition violated
    ],
)
@pytest.mark.parametrize("K", [100.0, 120.0])
def test_monte_carlo_confidence_interval_contains_analytic(params, K):
    import warnings

    model = HestonModel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model.set_parameters(*params)
    analytic = model.analytic_price(100.0, K, 1.0, 0.05, "call")
    mc = model.monte_carlo_price(
        100.0, K, 1.0, 0.05, "call", num_sims=100_000, num_steps=50, random_seed=3
    )
    assert mc.std_error > 0
    lo, hi = mc.confidence_interval
    assert lo < analytic < hi or abs(mc.price - analytic) < 3 * mc.std_error


def test_monte_carlo_full_truncation_euler_scheme_runs():
    model = HestonModel()
    model.set_parameters(0.04, 2.0, 0.04, 0.3, -0.7)
    analytic = model.analytic_price(100.0, 100.0, 1.0, 0.05, "call")
    mc = model.monte_carlo_price(
        100.0, 100.0, 1.0, 0.05, "call", num_sims=100_000, num_steps=100,
        scheme="euler", random_seed=4,
    )
    assert abs(mc.price - analytic) < 4 * mc.std_error


def test_monte_carlo_is_reproducible_with_seed():
    model = HestonModel()
    model.set_parameters(0.04, 2.0, 0.04, 0.3, -0.7)
    a = model.monte_carlo_price(100, 100, 1.0, 0.05, num_sims=5_000, num_steps=10, random_seed=9)
    b = model.monte_carlo_price(100, 100, 1.0, 0.05, num_sims=5_000, num_steps=10, random_seed=9)
    assert a.price == b.price
