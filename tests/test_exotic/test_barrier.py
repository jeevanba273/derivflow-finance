"""Barrier options: in/out parity, analytical-vs-MC, and basic bounds."""

import pytest

from derivflow.exotic.barrier_options import BarrierOptions

barrier = BarrierOptions()


def _price(btype, H, option_type='call', method='analytical'):
    return barrier.price(100, 100, H, 1.0, 0.05, 0.2, btype, option_type, method).price


def test_down_in_out_parity(bs_ref, tol):
    vanilla = bs_ref(100, 100, 1.0, 0.05, 0.2, 'call')
    di = _price('down_and_in', 90)
    do = _price('down_and_out', 90)
    assert di + do == pytest.approx(vanilla, abs=tol['analytic'])


def test_up_in_out_parity(bs_ref, tol):
    vanilla = bs_ref(100, 100, 1.0, 0.05, 0.2, 'call')
    ui = _price('up_and_in', 110)
    uo = _price('up_and_out', 110)
    assert ui + uo == pytest.approx(vanilla, abs=tol['analytic'])


def test_knockout_cheaper_than_vanilla(bs_ref):
    vanilla = bs_ref(100, 100, 1.0, 0.05, 0.2, 'call')
    assert _price('down_and_out', 90) <= vanilla + 1e-9
    assert _price('up_and_out', 110) <= vanilla + 1e-9


def test_survival_probability_in_range():
    res = barrier.price(100, 100, 90, 1.0, 0.05, 0.2, 'down_and_out', 'call')
    assert 0.0 <= res.probability_survival <= 1.0


@pytest.mark.slow
def test_analytical_vs_monte_carlo():
    # Discrete-monitoring MC carries a known bias vs continuous analytical, so
    # we use a loose relative tolerance rather than a std-error-tight bound.
    analytic = _price('down_and_out', 90, method='analytical')
    mc = barrier.price(100, 100, 90, 1.0, 0.05, 0.2, 'down_and_out', 'call',
                       'monte_carlo').price
    assert mc == pytest.approx(analytic, rel=0.1)
