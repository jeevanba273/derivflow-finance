"""Asian options: geometric analytical vs MC, AM-GM ordering, control-variate gain."""

import pytest

from derivflow.exotic.asian_options import AsianOptions, compare_asian_types


def test_geometric_analytical_vs_mc(tol):
    asian = AsianOptions(num_sims=100000, random_seed=tol['mc_seed'])
    analytic = asian.price_geometric_asian_analytical(100, 100, 1.0, 0.05, 0.2, 'call')
    mc = asian.price_geometric_asian_mc(100, 100, 1.0, 0.05, 0.2, 'call', num_steps=100)
    assert abs(mc.price - analytic.price) < tol['mc_sigma'] * mc.std_error


@pytest.mark.slow
def test_arithmetic_ge_geometric_call():
    asian = AsianOptions(num_sims=100000, random_seed=42)
    arith = asian.price(100, 100, 1.0, 0.05, 0.2, 'call', asian_type='arithmetic', num_steps=50)
    geo = asian.price(100, 100, 1.0, 0.05, 0.2, 'call', asian_type='geometric', num_steps=50)
    # By AM-GM, the arithmetic average dominates the geometric average.
    assert arith.price >= geo.price - 3 * (arith.std_error or 0.0)


@pytest.mark.slow
def test_control_variate_reduces_variance():
    asian = AsianOptions(num_sims=50000, random_seed=42)
    with_cv = asian.price_arithmetic_asian_mc(100, 100, 1.0, 0.05, 0.2, 'call',
                                              num_steps=50, use_control_variate=True)
    without_cv = asian.price_arithmetic_asian_mc(100, 100, 1.0, 0.05, 0.2, 'call',
                                                 num_steps=50, use_control_variate=False)
    assert with_cv.std_error < without_cv.std_error
    ratio = with_cv.convergence_info.get('variance_reduction_ratio')
    assert ratio is not None and ratio > 10  # conservative floor; usually far higher


def test_arithmetic_asian_cheaper_than_european():
    asian = AsianOptions(num_sims=50000, random_seed=42)
    res = asian.price(100, 100, 1.0, 0.05, 0.2, 'call', asian_type='arithmetic', num_steps=50)
    from derivflow.core.pricing_engine import price_european_option
    european = price_european_option(100, 100, 1.0, 0.05, 0.2, 'call')
    # Averaging reduces effective volatility -> cheaper than the vanilla.
    assert res.price < european
