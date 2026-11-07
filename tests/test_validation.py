"""Invalid inputs raise ValueError with a message naming the parameter."""

import numpy as np
import pytest

from derivflow_finance import (
    AsianOptions,
    BarrierOptions,
    BlackScholesAnalytical,
    GreeksCalculator,
    HestonModel,
    price_european_option,
)

GOOD = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2}
BAD = [
    ({"sigma": -0.2}, "sigma"),
    ({"sigma": 0.0}, "sigma"),
    ({"S": 0.0}, "S"),
    ({"S": -5.0}, "S"),
    ({"K": -1.0}, "K"),
    ({"S": np.nan}, "S"),
    ({"T": np.inf}, "T"),
]


def _bs(**kw):
    return BlackScholesAnalytical().price(**kw)


def _greeks(**kw):
    return GreeksCalculator().calculate_greeks(**kw)


def _barrier(**kw):
    return BarrierOptions().price(H=80.0, barrier_type="down_and_out", **kw)


def _asian(**kw):
    return AsianOptions(num_sims=100).price_geometric_asian_analytical(**kw)


PRICERS = [price_european_option, _bs, _greeks, _barrier, _asian]


@pytest.mark.parametrize("pricer", PRICERS, ids=lambda f: f.__name__)
@pytest.mark.parametrize(("override", "name"), BAD)
def test_bad_inputs_raise(pricer, override, name):
    with pytest.raises(ValueError, match=name):
        pricer(**(GOOD | override))


def test_expired_option_returns_intrinsic_not_an_error():
    assert price_european_option(110.0, 100.0, 0.0, 0.05, 0.2, "call") == 10.0
    assert GreeksCalculator().calculate_greeks(110.0, 100.0, 0.0, 0.05, 0.2).delta == 1.0


def test_barrier_level_must_be_positive():
    with pytest.raises(ValueError, match="H"):
        BarrierOptions().price(100, 100, -1.0, 1.0, 0.05, 0.2, "down_and_out")


def test_unknown_barrier_type_raises():
    with pytest.raises(ValueError, match="barrier_type"):
        BarrierOptions().price(100, 100, 90, 1.0, 0.05, 0.2, "sideways_and_out")


def test_heston_rejects_negative_initial_variance():
    with pytest.raises(ValueError, match="v0"):
        HestonModel().set_parameters(v0=-0.01, kappa=1.0, theta=0.04, sigma=0.3, rho=-0.5)


def test_heston_rejects_bad_market_inputs():
    model = HestonModel()
    model.set_parameters(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)
    with pytest.raises(ValueError, match="K"):
        model.analytic_price(100.0, -5.0, 1.0, 0.05)
