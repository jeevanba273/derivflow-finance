"""Heston: Monte Carlo puts, method routing and input errors."""

import warnings

import pytest

from derivflow_finance.models.heston import HestonModel, HestonParameters, heston_call_price

FELLER_OK = (0.04, 2.0, 0.04, 0.3, -0.7)
FELLER_VIOLATED = (0.04, 1.0, 0.04, 0.9, -0.7)


def _model(params):
    model = HestonModel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model.set_parameters(*params)
    return model


@pytest.mark.parametrize("params", [FELLER_OK, FELLER_VIOLATED])
@pytest.mark.parametrize("K", [80.0, 100.0])
def test_monte_carlo_put_matches_the_analytic_put(params, K):
    model = _model(params)
    analytic = model.analytic_price(100.0, K, 1.0, 0.03, "put", q=0.01)
    mc = model.monte_carlo_price(
        100.0, K, 1.0, 0.03, "put", q=0.01, num_sims=100_000, num_steps=50, random_seed=4
    )
    assert abs(mc.price - analytic) < 4 * mc.std_error


def test_price_option_routes_to_each_method():
    model = _model(FELLER_OK)
    analytic = model.price_option(100.0, 100.0, 1.0, 0.03, "call", method="analytic")
    fourier = model.price_option(100.0, 100.0, 1.0, 0.03, "call", method="Fourier")
    assert analytic.price == fourier.price
    mc = model.price_option(
        100.0, 100.0, 1.0, 0.03, "call", method="monte_carlo", num_sims=20_000, random_seed=1
    )
    assert abs(mc.price - analytic.price) < 4 * mc.std_error
    with pytest.raises(ValueError, match="Unknown method"):
        model.price_option(100.0, 100.0, 1.0, 0.03, method="pde")


def test_methods_need_parameters():
    model = HestonModel()
    with pytest.raises(ValueError, match="not set"):
        model.analytic_price(100.0, 100.0, 1.0, 0.03)
    with pytest.raises(ValueError, match="not set"):
        model.monte_carlo_price(100.0, 100.0, 1.0, 0.03)
    with pytest.raises(ValueError, match="not set"):
        model.price_option(100.0, 100.0, 1.0, 0.03)


def test_monte_carlo_input_errors():
    model = _model(FELLER_OK)
    with pytest.raises(ValueError, match="num_sims"):
        model.monte_carlo_price(100.0, 100.0, 1.0, 0.03, num_sims=1)
    with pytest.raises(ValueError, match="num_steps"):
        model.monte_carlo_price(100.0, 100.0, 1.0, 0.03, num_steps=0)
    with pytest.raises(ValueError, match="scheme"):
        model.monte_carlo_price(100.0, 100.0, 1.0, 0.03, num_sims=10, scheme="milstein")


def test_call_at_expiry_is_intrinsic_and_calibration_needs_quotes():
    params = HestonParameters(0.04, 2.0, 0.04, 0.3, -0.7)
    assert heston_call_price(110.0, 100.0, 0.0, 0.03, 0.0, params) == 10.0
    assert heston_call_price(90.0, 100.0, 0.0, 0.03, 0.0, params) == 0.0
    with pytest.raises(ValueError, match="empty"):
        HestonModel().calibrate_to_market([], 100.0, 0.03)
