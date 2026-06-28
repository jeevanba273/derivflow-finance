"""Heston model: parameter validation, Feller warning, and MC sanity."""

import pytest

from derivflow.models.heston import HestonModel
from derivflow.core.pricing_engine import price_european_option


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
    res = model.monte_carlo_price(100, 100, 1.0, 0.05, 'call', num_sims=100000, num_steps=252)
    bs = price_european_option(100, 100, 1.0, 0.05, 0.2, 'call')
    assert res.price == pytest.approx(bs, abs=0.3)
